"""PyTorch TabularPreprocessor, OULADDataset, and TabularMLP architecture (P1-B4).

Enforces AUD-17:
- TabularPreprocessor fits strictly on Train ONLY:
  - Numeric features -> median imputer + StandardScaler (train-only stats).
  - Categorical features -> Unknown / Missing = 0, Train categories = 1..K.
- OULADDataset wraps np.array(..., copy=True) to guarantee writable tensors on Pandas 3+.
- TabularMLP outputs RAW LOGITS (no trailing Sigmoid) for numerical stability with
  unweighted nn.BCEWithLogitsLoss().
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from oulad.baselines import infer_feature_types


@dataclass
class TabularPreprocessorState:
    """Serializable, frozen Train-only preprocessing parameters for TabularMLP."""

    num_cols: list[str]
    cat_cols: list[str]
    num_medians: list[float]
    num_means: list[float]
    num_stds: list[float]
    cat_to_idx: dict[str, dict[str, int]]
    cat_cardinalities: list[int]


class TabularPreprocessor:
    """Leakage-free Train-only preprocessor for mixed numeric + categorical tabular data."""

    def __init__(self) -> None:
        self.state_: TabularPreprocessorState | None = None

    def fit(self, X_train: pd.DataFrame) -> "TabularPreprocessor":
        """Learn numeric medians/means/stds and categorical 1..K vocabularies on Train ONLY."""
        num_cols, cat_cols = infer_feature_types(X_train)

        num_medians: list[float] = []
        num_means: list[float] = []
        num_stds: list[float] = []

        for col in num_cols:
            vals = pd.to_numeric(X_train[col], errors="coerce").to_numpy(dtype=np.float64)
            finite_mask = np.isfinite(vals)
            median = float(np.median(vals[finite_mask])) if finite_mask.any() else 0.0
            filled = np.where(finite_mask, vals, median)
            mean = float(np.mean(filled))
            std = float(np.std(filled))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            num_medians.append(median)
            num_means.append(mean)
            num_stds.append(std)

        # Categorical contract (AUD-17): Unknown / Missing = 0, Train known categories = 1..K
        cat_to_idx: dict[str, dict[str, int]] = {}
        cat_cardinalities: list[int] = []
        for col in cat_cols:
            non_null = X_train[col].dropna().astype(str).unique().tolist()
            known_sorted = sorted(non_null)
            mapping = {val: idx + 1 for idx, val in enumerate(known_sorted)}
            cat_to_idx[col] = mapping
            cat_cardinalities.append(len(known_sorted) + 1)  # +1 for reserved index 0

        self.state_ = TabularPreprocessorState(
            num_cols=num_cols,
            cat_cols=cat_cols,
            num_medians=num_medians,
            num_means=num_means,
            num_stds=num_stds,
            cat_to_idx=cat_to_idx,
            cat_cardinalities=cat_cardinalities,
        )
        return self

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform DataFrame into writable (X_num_float32, X_cat_int64) arrays."""
        if self.state_ is None:
            raise RuntimeError("TabularPreprocessor must be fitted before calling transform().")

        st = self.state_
        n_rows = len(X)

        if st.num_cols:
            X_num = np.empty((n_rows, len(st.num_cols)), dtype=np.float32)
            for j, col in enumerate(st.num_cols):
                vals = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=np.float64)
                filled = np.where(np.isfinite(vals), vals, st.num_medians[j])
                scaled = (filled - st.num_means[j]) / st.num_stds[j]
                X_num[:, j] = scaled.astype(np.float32)
        else:
            X_num = np.zeros((n_rows, 0), dtype=np.float32)

        if st.cat_cols:
            X_cat = np.zeros((n_rows, len(st.cat_cols)), dtype=np.int64)
            for j, col in enumerate(st.cat_cols):
                mapping = st.cat_to_idx[col]
                # Missing (NaN/None) and any unseen category strictly map to 0 (Unknown)
                codes = [
                    0 if pd.isna(v) else mapping.get(str(v), 0)
                    for v in X[col].tolist()
                ]
                X_cat[:, j] = np.asarray(codes, dtype=np.int64)
        else:
            X_cat = np.zeros((n_rows, 0), dtype=np.int64)

        return np.array(X_num, dtype=np.float32, copy=True), np.array(
            X_cat, dtype=np.int64, copy=True
        )

    def fit_transform(self, X_train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit strictly on X_train and transform it."""
        return self.fit(X_train).transform(X_train)

    def to_dict(self) -> dict[str, Any]:
        """Export frozen preprocessor state as a plain Python dict for checkpointing."""
        if self.state_ is None:
            raise RuntimeError("Cannot serialize unfitted TabularPreprocessor.")
        st = self.state_
        return {
            "num_cols": st.num_cols,
            "cat_cols": st.cat_cols,
            "num_medians": st.num_medians,
            "num_means": st.num_means,
            "num_stds": st.num_stds,
            "cat_to_idx": st.cat_to_idx,
            "cat_cardinalities": st.cat_cardinalities,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TabularPreprocessor":
        """Restore a fitted TabularPreprocessor from a serialized checkpoint dict."""
        prep = cls()
        prep.state_ = TabularPreprocessorState(
            num_cols=list(payload["num_cols"]),
            cat_cols=list(payload["cat_cols"]),
            num_medians=[float(x) for x in payload["num_medians"]],
            num_means=[float(x) for x in payload["num_means"]],
            num_stds=[float(x) for x in payload["num_stds"]],
            cat_to_idx={
                str(k): {str(ck): int(cv) for ck, cv in v.items()}
                for k, v in payload["cat_to_idx"].items()
            },
            cat_cardinalities=[int(c) for c in payload["cat_cardinalities"]],
        )
        return prep


class OULADDataset(Dataset[tuple[torch.Tensor, ...]]):
    """PyTorch Dataset wrapping explicitly copied writable NumPy arrays."""

    def __init__(
        self,
        X_num: np.ndarray,
        X_cat: np.ndarray,
        y: np.ndarray | pd.Series | None = None,
    ) -> None:
        self.X_num = torch.from_numpy(np.array(X_num, dtype=np.float32, copy=True))
        self.X_cat = torch.from_numpy(np.array(X_cat, dtype=np.int64, copy=True))
        self.y = (
            torch.from_numpy(np.array(y, dtype=np.float32, copy=True))
            if y is not None
            else None
        )

    def __len__(self) -> int:
        return int(self.X_num.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if self.y is None:
            return self.X_num[idx], self.X_cat[idx]
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


def compute_embedding_dim(num_embeddings: int) -> int:
    """Compute a compact embedding dimension given table size K + 1."""
    return min(16, max(2, (num_embeddings + 1) // 2))


class TabularMLP(nn.Module):
    """Tabular MLP with Entity Embeddings (Unknown=0) outputting RAW LOGITS.

    Architecture:
      - Per-categorical nn.Embedding(K + 1, emb_dim) concatenated with numeric features.
      - Two hidden blocks: Linear -> LayerNorm/BatchNorm1d -> ReLU -> Dropout(dropout).
      - Final Linear(..., 1) producing raw logits (NO Sigmoid) for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: Sequence[int],
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
        norm_type: Literal["layernorm", "batchnorm"] = "layernorm",
    ) -> None:
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]
        self.hidden_dims = hidden_dims
        self.dropout = float(dropout)
        self.norm_type = norm_type

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=card, embedding_dim=compute_embedding_dim(card))
                for card in self.cat_cardinalities
            ]
        )
        total_emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        in_dim = self.n_num_features + total_emb_dim

        blocks: list[nn.Module] = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            norm_layer: nn.Module = (
                nn.LayerNorm(h_dim) if norm_type == "layernorm" else nn.BatchNorm1d(h_dim)
            )
            blocks.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    norm_layer,
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout),
                ]
            )
            prev_dim = h_dim

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Return 1D tensor of raw logits of shape (batch_size,)."""
        parts: list[torch.Tensor] = []
        if self.n_num_features > 0:
            parts.append(x_num)
        for j, emb in enumerate(self.embeddings):
            parts.append(emb(x_cat[:, j]))
        features = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        hidden = self.backbone(features)
        return self.head(hidden).squeeze(-1)
