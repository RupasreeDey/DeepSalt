"""Tensor datasets and dataloaders for student training.

The zero-inflation handling and the categorical-covariate bookkeeping both
live here so that training scripts stay short.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils import torch_generator


@dataclass
class FeatureLayout:
    """Which columns of the design matrix are what.

    The categorical block is stripped from the model input (the paper's claim
    is that salinity is predicted from spectra plus climate/texture, not from
    a Köppen label) but retained separately so evaluation can stratify by it.
    """

    names: list[str]
    n_spectral: int
    n_ancillary: int
    categorical_slice: slice

    @property
    def model_input_dim(self) -> int:
        return self.n_spectral + self.n_ancillary

    @property
    def categorical_names(self) -> list[str]:
        return self.names[self.categorical_slice]


def build_layout(feature_names: list[str], n_spectral: int, n_ancillary: int) -> FeatureLayout:
    start = n_spectral + n_ancillary
    return FeatureLayout(
        names=list(feature_names),
        n_spectral=n_spectral,
        n_ancillary=n_ancillary,
        categorical_slice=slice(start, len(feature_names)),
    )


def split_categorical(X: np.ndarray, layout: FeatureLayout) -> tuple[np.ndarray, np.ndarray]:
    """Return (model inputs, categorical block)."""
    return X[:, : layout.model_input_dim], X[:, layout.categorical_slice]


def undersample_zeros(
    X: torch.Tensor,
    y: torch.Tensor,
    keep_fraction: float = 0.10,
    seed: int = 42,
    tolerance: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample exact-zero targets, reproducibly.

    Zero-EC records dominate SSURGO and swamp the regression. The original did
    this with a bare ``torch.randperm``, so the retained subset changed between
    runs even with the global seed set; a dedicated generator fixes that.

    Applied to the TRAINING SPLIT ONLY. Undersampling validation or test would
    change the target distribution the metrics are computed over.
    """
    zero_mask = y.abs() <= tolerance
    zero_idx = torch.nonzero(zero_mask, as_tuple=True)[0]
    nonzero_idx = torch.nonzero(~zero_mask, as_tuple=True)[0]

    if len(zero_idx) == 0:
        return X, y

    n_keep = max(1, int(keep_fraction * len(zero_idx)))
    perm = torch.randperm(len(zero_idx), generator=torch_generator(seed))
    kept = zero_idx[perm[:n_keep]]

    keep_idx = torch.cat([kept, nonzero_idx])
    keep_idx = keep_idx[torch.argsort(keep_idx)]  # preserve original ordering
    print(
        f"  zero undersampling: kept {len(kept)}/{len(zero_idx)} zeros, "
        f"{len(nonzero_idx)} non-zeros -> {len(keep_idx)} samples"
    )
    return X[keep_idx], y[keep_idx]


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    drop_last: bool = False,
) -> DataLoader | None:
    if X is None or len(X) == 0:
        return None
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=torch_generator(seed) if shuffle else None,
    )
