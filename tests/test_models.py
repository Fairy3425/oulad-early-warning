"""Fast offline CPU contract tests for PyTorch TabularMLP and TabularPreprocessor (P1-B4)."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from oulad.config import SEED, TEST_PRESENTATION
from oulad.features import split_feature_target
from oulad.split import presentation_forward_split
from oulad.torch_models import OULADDataset, TabularMLP, TabularPreprocessor
from oulad.train_torch import (
    load_torch_checkpoint,
    predict_torch_proba,
    save_torch_checkpoint,
    train_tabular_mlp,
)


def test_preprocessor_maps_unseen_and_missing_strictly_to_zero(
    synthetic_feature_table: pd.DataFrame,
) -> None:
    """AUD-17: Train categories must be 1..K; unseen and NaN categories must map strictly to 0."""
    split = presentation_forward_split(synthetic_feature_table, seed=SEED)
    X_tr, _ = split_feature_target(split.train)
    X_va, _ = split_feature_target(split.val_select)

    prep = TabularPreprocessor()
    X_tr_num, X_tr_cat = prep.fit_transform(X_tr)
    assert prep.state_ is not None

    # Every non-null Train category must be in [1, K], never colliding with reserved index 0
    assert (X_tr_cat >= 1).all()
    assert np.isfinite(X_tr_num).all()
    assert X_tr_num.flags.writeable and X_tr_cat.flags.writeable

    # Inject brand-new unseen category and explicit NaN into Validation
    X_va_mut = X_va.copy()
    region_idx = prep.state_.cat_cols.index("region")
    imd_idx = prep.state_.cat_cols.index("imd_band")
    X_va_mut.loc[0, "region"] = "Unseen Mars Region"
    X_va_mut.loc[1, "imd_band"] = np.nan

    X_va_num, X_va_cat = prep.transform(X_va_mut)
    assert X_va_cat[0, region_idx] == 0
    assert X_va_cat[1, imd_idx] == 0
    assert np.isfinite(X_va_num).all()


def test_tabular_mlp_raw_logits_single_sample_and_eval_determinism() -> None:
    """TabularMLP outputs 1D raw logits, supports batch_size=1, and is deterministic in eval()."""
    torch.manual_seed(SEED)
    model = TabularMLP(n_num_features=5, cat_cardinalities=[4, 6], hidden_dims=(32, 16))
    model.eval()

    x_num = torch.randn(1, 5)
    x_cat = torch.tensor([[0, 3]], dtype=torch.long)

    with torch.no_grad():
        out1 = model(x_num, x_cat)
        out2 = model(x_num, x_cat)

    assert out1.shape == (1,)
    assert torch.isfinite(out1).all()
    assert torch.equal(out1, out2)

    # Confirm final head outputs raw logits (unconstrained linear, not wrapped in Sigmoid)
    assert not isinstance(model.head, nn.Sigmoid)


def test_one_gradient_step_updates_trainable_weights() -> None:
    """One backward pass with unweighted BCEWithLogitsLoss + AdamW must update parameters."""
    torch.manual_seed(SEED)
    x_num = np.random.default_rng(SEED).normal(size=(16, 4)).astype(np.float32)
    x_cat = np.array([[i % 3, i % 4] for i in range(16)], dtype=np.int64)
    y = np.array([i % 2 for i in range(16)], dtype=np.float32)

    ds = OULADDataset(x_num, x_cat, y)
    model = TabularMLP(n_num_features=4, cat_cardinalities=[3, 4], hidden_dims=(16, 8))
    before = [p.detach().clone() for p in model.parameters()]

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

    model.train()
    optimizer.zero_grad()
    logits = model(ds.X_num, ds.X_cat)
    assert ds.y is not None
    loss = criterion(logits, ds.y)
    loss.backward()
    optimizer.step()

    after = [p.detach() for p in model.parameters()]
    changed = [not torch.equal(b, a) for b, a in zip(before, after, strict=True)]
    assert all(changed), "Every trainable parameter tensor must update after one gradient step"


def test_train_checkpoint_roundtrip_and_2014j_isolation(
    synthetic_feature_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """End-to-end CPU train + checkpoint save/load matches bit-for-bit and ignores 2014J."""
    df_orig = synthetic_feature_table.copy()
    df_mut = df_orig.copy()
    mask_test = df_mut["code_presentation"] == TEST_PRESENTATION
    df_mut.loc[mask_test, "clicks_total"] = 999999
    df_mut.loc[mask_test, "target"] = 1 - df_mut.loc[mask_test, "target"]

    s1 = presentation_forward_split(df_orig, seed=SEED)
    s2 = presentation_forward_split(df_mut, seed=SEED)

    X_tr1, y_tr1 = split_feature_target(s1.train)
    X_va1, y_va1 = split_feature_target(s1.val_select)
    X_tr2, y_tr2 = split_feature_target(s2.train)
    X_va2, y_va2 = split_feature_target(s2.val_select)

    res1 = train_tabular_mlp(
        X_tr1,
        y_tr1,
        X_va1,
        y_va1,
        hidden_dims=(32, 16),
        max_epochs=6,
        patience=3,
        seed=SEED,
        prefer_cuda=False,
    )
    res2 = train_tabular_mlp(
        X_tr2,
        y_tr2,
        X_va2,
        y_va2,
        hidden_dims=(32, 16),
        max_epochs=6,
        patience=3,
        seed=SEED,
        prefer_cuda=False,
    )

    p1 = predict_torch_proba(res1.model, res1.preprocessor, X_va1)
    p2 = predict_torch_proba(res2.model, res2.preprocessor, X_va2)
    assert np.all((p1 >= 0.0) & (p1 <= 1.0))
    np.testing.assert_allclose(p1, p2, rtol=0.0, atol=0.0)

    # Checkpoint round-trip must produce identical predictions
    ckpt = tmp_path / "mlp_test.pt"
    save_torch_checkpoint(res1, ckpt)
    loaded_model, loaded_prep = load_torch_checkpoint(ckpt, device=torch.device("cpu"))
    p_loaded = predict_torch_proba(loaded_model, loaded_prep, X_va1)
    np.testing.assert_allclose(p1, p_loaded, rtol=0.0, atol=0.0)
