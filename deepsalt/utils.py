"""Shared utilities: reproducibility, device selection, checkpoint I/O, metrics."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def set_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG this project touches.

    ``deterministic=True`` also disables cuDNN autotuning, which costs
    throughput but makes runs bitwise comparable across machines.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def torch_generator(seed: int) -> torch.Generator:
    """A seeded generator, for sampling that must not depend on global state."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def resolve(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Read ``cfg['a']['b']`` as ``resolve(cfg, 'a.b')``."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
# Every checkpoint carries a sidecar JSON describing how it was produced. This
# exists because the original scripts stored bare state_dicts, and a bare
# state_dict does not record the output parameterization of the prediction head.
# The same FTIR weights were loaded into heads scaling to [0.05, 1.1] and to
# [0.05, 77.0]; nothing in the file flagged the mismatch. The sidecar does.


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    meta = dict(metadata or {})
    meta.setdefault("class_name", type(model).__name__)
    with open(path.with_suffix(path.suffix + ".json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device,
    expect: dict[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load weights and verify the sidecar against ``expect``.

    Raises on mismatch rather than warning. A silent unit mismatch between a
    teacher and the head it is loaded into is exactly the failure this guards.
    """
    path = Path(path)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=strict)

    sidecar = path.with_suffix(path.suffix + ".json")
    meta: dict[str, Any] = {}
    if sidecar.exists():
        with open(sidecar) as fh:
            meta = json.load(fh)
    elif expect:
        raise FileNotFoundError(
            f"{path} has no sidecar metadata ({sidecar.name}). Checkpoints "
            "produced before this refactor are untracked -- retrain, or write "
            "the sidecar by hand after confirming how the weights were made."
        )

    for key, want in (expect or {}).items():
        got = meta.get(key)
        if got != want:
            raise ValueError(
                f"Checkpoint {path.name} metadata mismatch on '{key}': "
                f"file says {got!r}, caller expects {want!r}."
            )
    return meta


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    mape: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mape": self.mape,
            "n": self.n,
        }

    def __str__(self) -> str:
        return (
            f"MAE={self.mae:.4f}  RMSE={self.rmse:.4f}  "
            f"R2={self.r2:.4f}  MAPE={self.mape:.2f}%  (n={self.n})"
        )


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10
) -> float:
    """MAPE over samples with |y_true| > epsilon.

    Undefined at zero salinity, and this dataset is zero-inflated, so the
    denominator count differs from the sample count. Report both.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > epsilon
    if not mask.any():
        return float("nan")
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100.0
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        mape=mean_absolute_percentage_error(y_true, y_pred),
        n=int(y_true.size),
    )
