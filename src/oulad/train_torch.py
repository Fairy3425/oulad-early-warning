"""Training, inference, learning-curve plotting, and Val evaluation for TabularMLP (P1-B4).

Enforces AUD-17:
- Auto device selection: cuda if available, clean CPU fallback for CI / CPU-only machines.
- AdamW(lr=1e-3, weight_decay=1e-4) + UNWEIGHTED nn.BCEWithLogitsLoss(): avoids
  intentional prior-odds distortion from class weighting (pos_weight); does NOT by
  itself guarantee calibrated probabilities or preserve deployment base rate under
  temporal shift — post-hoc probability calibration & thresholding happen in P1-B5.
- Strict model.eval() + torch.no_grad() for validation/inference, copy.deepcopy() for
  best state_dict, and Early Stopping (patience=5) strictly on val_select (2014J locked).
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from oulad.baselines import compute_binary_metrics, evaluate_returning_vs_new
from oulad.config import (
    FIGURES_DIR,
    PRIMARY_CUTOFF,
    PROCESSED_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SEED,
    TEST_PRESENTATION,
)
from oulad.features import split_feature_target
from oulad.split import SplitData, load_feature_parquet, presentation_forward_split
from oulad.torch_models import OULADDataset, TabularMLP, TabularPreprocessor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Select CUDA GPU automatically when available, otherwise fall back cleanly to CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_torch_seed(seed: int = SEED) -> None:
    """Seed NumPy and PyTorch RNGs for reproducible TabularMLP training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TorchTrainResult:
    """Trained TabularMLP bundle with frozen Train-only preprocessor and history."""

    model: TabularMLP
    preprocessor: TabularPreprocessor
    device_used: str
    best_epoch: int
    best_val_loss: float
    history: list[dict[str, float | int]]


@torch.no_grad()
def predict_torch_proba(
    model: TabularMLP,
    preprocessor: TabularPreprocessor,
    X: pd.DataFrame,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    """Predict uncalibrated positive-class probabilities in eval mode without gradient tracking."""
    target_device = device if device is not None else next(model.parameters()).device
    X_num, X_cat = preprocessor.transform(X)
    dataset = OULADDataset(X_num, X_cat, y=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    was_training = model.training
    model.eval()
    probs: list[np.ndarray] = []
    for batch_num, batch_cat in loader:
        logits = model(batch_num.to(target_device), batch_cat.to(target_device))
        batch_probs = torch.sigmoid(logits).cpu().numpy()
        probs.append(batch_probs)
    if was_training:
        model.train()

    return np.concatenate(probs, axis=0).astype(np.float64) if probs else np.array([], dtype=float)


def train_tabular_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 40,
    patience: int = 5,
    seed: int = SEED,
    prefer_cuda: bool = True,
) -> TorchTrainResult:
    """Train TabularMLP on (X_train, y_train) with early stopping strictly on val_select."""
    set_torch_seed(seed)
    device = get_device(prefer_cuda=prefer_cuda)

    preprocessor = TabularPreprocessor()
    X_tr_num, X_tr_cat = preprocessor.fit_transform(X_train)
    X_va_num, X_va_cat = preprocessor.transform(X_val)
    assert preprocessor.state_ is not None

    train_ds = OULADDataset(X_tr_num, X_tr_cat, y_train.to_numpy(dtype=np.float32))
    val_ds = OULADDataset(X_va_num, X_va_cat, y_val.to_numpy(dtype=np.float32))

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = TabularMLP(
        n_num_features=X_tr_num.shape[1],
        cat_cardinalities=preprocessor.state_.cat_cardinalities,
        hidden_dims=hidden_dims,
        dropout=dropout,
        norm_type="layernorm",
    ).to(device)

    # Unweighted BCEWithLogitsLoss avoids class-reweighting prior-odds shift (AUD-17);
    # post-hoc probability calibration & cost-aware thresholding are separate steps in P1-B5.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    history: list[dict[str, float | int]] = []

    y_val_np = y_val.to_numpy(dtype=np.int64)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_seen = 0
        for batch_num, batch_cat, batch_y in train_loader:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_num, batch_cat)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            bs = int(batch_y.shape[0])
            train_loss_sum += float(loss.item()) * bs
            n_train_seen += bs

        train_loss = train_loss_sum / max(n_train_seen, 1)

        # Validation pass strictly in eval() + no_grad()
        model.eval()
        val_loss_sum = 0.0
        n_val_seen = 0
        val_probs_list: list[np.ndarray] = []
        with torch.no_grad():
            for batch_num, batch_cat, batch_y in val_loader:
                batch_num = batch_num.to(device)
                batch_cat = batch_cat.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_num, batch_cat)
                loss = criterion(logits, batch_y)
                bs = int(batch_y.shape[0])
                val_loss_sum += float(loss.item()) * bs
                n_val_seen += bs
                val_probs_list.append(torch.sigmoid(logits).cpu().numpy())

        val_loss = val_loss_sum / max(n_val_seen, 1)
        val_probs = np.concatenate(val_probs_list, axis=0)
        val_metrics = compute_binary_metrics(y_val_np, val_probs)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_pr_auc": float(val_metrics["pr_auc"]) if val_metrics["pr_auc"] else 0.0,
                "val_roc_auc": float(val_metrics["roc_auc"]) if val_metrics["roc_auc"] else 0.0,
            }
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    return TorchTrainResult(
        model=model,
        preprocessor=preprocessor,
        device_used=str(device),
        best_epoch=best_epoch,
        best_val_loss=round(best_val_loss, 5),
        history=history,
    )


def save_torch_checkpoint(result: TorchTrainResult, out_path: Path) -> None:
    """Save best TabularMLP weights (CPU-portable) and frozen Train-only preprocessor state."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in result.model.state_dict().items()}
    payload = {
        "model_state_dict": cpu_state,
        "model_config": {
            "n_num_features": result.model.n_num_features,
            "cat_cardinalities": result.model.cat_cardinalities,
            "hidden_dims": result.model.hidden_dims,
            "dropout": result.model.dropout,
            "norm_type": result.model.norm_type,
        },
        "preprocessor_state": result.preprocessor.to_dict(),
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "device_used": result.device_used,
    }
    torch.save(payload, out_path)


def load_torch_checkpoint(
    ckpt_path: Path, device: torch.device | None = None
) -> tuple[TabularMLP, TabularPreprocessor]:
    """Load a saved TabularMLP checkpoint onto target device in eval() mode."""
    target_device = device if device is not None else get_device(prefer_cuda=False)
    payload = torch.load(ckpt_path, map_location=target_device, weights_only=False)
    cfg = payload["model_config"]
    model = TabularMLP(
        n_num_features=cfg["n_num_features"],
        cat_cardinalities=cfg["cat_cardinalities"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
        norm_type=cfg["norm_type"],
    ).to(target_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    preprocessor = TabularPreprocessor.from_dict(payload["preprocessor_state"])
    return model, preprocessor


def plot_learning_curve(
    history: list[dict[str, float | int]],
    best_epoch: int,
    out_path: Path = FIGURES_DIR / "learning_curve.png",
) -> None:
    """Save 2-panel learning curve (Train vs Val Loss + Val PR-AUC/ROC-AUC)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(r["epoch"]) for r in history]
    tr_loss = [float(r["train_loss"]) for r in history]
    va_loss = [float(r["val_loss"]) for r in history]
    va_pr = [float(r["val_pr_auc"]) for r in history]
    va_roc = [float(r["val_roc_auc"]) for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].plot(epochs, tr_loss, marker="o", markersize=3, label="Train Loss (BCE)")
    axes[0].plot(epochs, va_loss, marker="s", markersize=3, label="Val Select Loss (BCE)")
    axes[0].axvline(
        best_epoch, color="#dc2626", linestyle="--", label=f"Best Epoch = {best_epoch}"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Unweighted BCEWithLogitsLoss")
    axes[0].set_title("TabularMLP Learning Curve — Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, va_pr, marker="o", markersize=3, label="Val Select PR-AUC")
    axes[1].plot(epochs, va_roc, marker="s", markersize=3, label="Val Select ROC-AUC")
    axes[1].axvline(
        best_epoch, color="#dc2626", linestyle="--", label=f"Best Epoch = {best_epoch}"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("TabularMLP Validation Ranking Metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "PyTorch TabularMLP (T=28 Operational) — val_select (2014J Locked)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_p1_b4_torch_suite(
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    seed: int = SEED,
    prefer_cuda: bool = True,
) -> dict[str, Any]:
    """Train TabularMLP on T=28 Operational, save checkpoint/plot, and update metrics_val.json."""
    df = load_feature_parquet(PRIMARY_CUTOFF, mode="operational", processed_dir=processed_dir)
    split: SplitData = presentation_forward_split(df, seed=seed)
    split.assert_invariants()
    assert TEST_PRESENTATION not in set(split.train["code_presentation"])
    assert TEST_PRESENTATION not in set(split.val_select["code_presentation"])

    X_train, y_train = split_feature_target(split.train)
    X_val, y_val = split_feature_target(split.val_select)

    result = train_tabular_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=seed,
        prefer_cuda=prefer_cuda,
    )

    val_probs = predict_torch_proba(result.model, result.preprocessor, X_val)
    mlp_metrics = compute_binary_metrics(y_val, val_probs)
    mlp_metrics["best_epoch"] = result.best_epoch
    mlp_metrics["best_val_loss"] = result.best_val_loss
    mlp_metrics["device_used"] = result.device_used

    mlp_ret_new = evaluate_returning_vs_new(split.val_select, split.train, val_probs)

    ckpt_path = artifacts_dir / "tabular_mlp_best.pt"
    save_torch_checkpoint(result, ckpt_path)
    plot_learning_curve(result.history, result.best_epoch, figures_dir / "learning_curve.png")

    metrics_path = reports_dir / "metrics_val.json"
    if metrics_path.exists():
        metrics_val: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics_val = {}

    metrics_val.setdefault("models_val_select_t28_operational", {})["tabular_mlp"] = mlp_metrics
    metrics_val.setdefault("returning_vs_new_val_select_t28", {})["tabular_mlp"] = mlp_ret_new
    metrics_val["tabular_mlp_training_info"] = {
        "device_used": result.device_used,
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "best_val_loss": result.best_val_loss,
        "hidden_dims": list(result.model.hidden_dims),
        "dropout": result.model.dropout,
        "norm_type": result.model.norm_type,
        "loss_function": (
            "BCEWithLogitsLoss (unweighted — avoids class-reweighting prior-odds shift; "
            "post-hoc calibration & thresholding handled separately in P1-B5)"
        ),
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "early_stopping_patience": 5,
        "checkpoint_path": "artifacts/tabular_mlp_best.pt (gitignored)",
        "learning_curve_path": "learning_curve.png",
        "test_2014j_untouched": True,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_val, indent=2), encoding="utf-8")
    return metrics_val


if __name__ == "__main__":
    updated = run_p1_b4_torch_suite()
    print("=== 3-Model Comparison on val_select (T=28 Operational, 2014J Locked) ===")
    print(json.dumps(updated["models_val_select_t28_operational"], indent=2))
    print("=== TabularMLP Training Info ===")
    print(json.dumps(updated["tabular_mlp_training_info"], indent=2))
