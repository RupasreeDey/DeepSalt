"""Regression tests for the specific bugs documented in PROVENANCE.md.

Each test names the section it guards. If one fails, that bug has returned.
"""

import numpy as np
import pytest
import torch

from deepsalt.data.pairing import pair_by_location
from deepsalt.losses import DistillationObjective, LayerWiseDistillationLoss
from deepsalt.models import (
    OutputRange,
    SpectralAdaptationUnit,
    StudentTransformer,
    TeacherTransformer,
)
from deepsalt.utils import compute_metrics, set_seeds


def test_haversine_uses_radians():
    """PROVENANCE 2.4 -- 0.013 deg lon at lat 40.6 is ~1.10 km."""
    ftir = np.array([[40.5853, -105.0844]])
    enmap = np.array([[40.5853, -105.0714]])
    pairs = pair_by_location(ftir, enmap, max_distance_km=2.0)
    assert len(pairs) == 1
    assert pairs.distances_km[0] == pytest.approx(1.10, abs=0.05)


def test_pairing_radius_excludes_distant():
    ftir = np.array([[40.0, -105.0]])
    enmap = np.array([[41.0, -105.0]])  # ~111 km
    assert len(pair_by_location(ftir, enmap, max_distance_km=10.0)) == 0


def test_pairing_is_order_independent():
    """PROVENANCE 2.4 -- mutual matching must not depend on row order."""
    rng = np.random.default_rng(0)
    ftir = rng.uniform([39, -106], [41, -104], (40, 2))
    enmap = rng.uniform([39, -106], [41, -104], (60, 2))
    a = pair_by_location(ftir, enmap, 50.0, "mutual")
    perm = rng.permutation(len(ftir))
    b = pair_by_location(ftir[perm], enmap, 50.0, "mutual")
    assert set(zip(perm[b.ftir_indices], b.enmap_indices, strict=True)) == set(
        zip(a.ftir_indices, a.enmap_indices, strict=True)
    )


def test_layer_count_mismatch_raises():
    """PROVENANCE 2.5 -- zip() must not silently drop the deepest layer."""
    feats = [torch.randn(4, 1, 64) for _ in range(4)]
    with pytest.raises(ValueError, match="distilled layers"):
        LayerWiseDistillationLoss()(feats, feats[:3])


def test_layer_weight_count_validated():
    feats = [torch.randn(4, 1, 64) for _ in range(4)]
    with pytest.raises(ValueError, match="layer weights"):
        LayerWiseDistillationLoss([1.0, 1.0, 1.0])(feats, feats)


def test_no_dead_kl_term():
    """PROVENANCE 2.3 -- response distillation must actually contribute."""
    obj = DistillationObjective(task_weight=1.0, feature_weight=0.0, response_weight=1.0)
    s_out = torch.full((8, 1), 5.0, requires_grad=True)
    t_out = torch.full((8, 1), 9.0)
    feats = [torch.randn(8, 1, 64)]
    _, parts = obj(s_out, t_out, feats, feats, torch.full((8,), 5.0))
    assert parts["response"] > 0.0


@pytest.mark.parametrize("mode", ["truncate", "linear", "residual"])
def test_sau_reaches_teacher_dim(mode):
    sau = SpectralAdaptationUnit(d_student=72, d_teacher=64, num_layers=4, mode=mode)
    out = sau([torch.randn(6, 1, 72) for _ in range(4)])
    assert all(o.shape == (6, 1, 64) for o in out)


def test_output_range_respected():
    """PROVENANCE 2.6 -- predictions must stay inside the declared support."""
    r = OutputRange(0.05, 77.0)
    model = StudentTransformer(input_dim=72, output_range=r)
    out, _, _ = model(torch.randn(32, 72) * 50)
    assert out.min() >= r.low and out.max() <= r.high


def test_teacher_rejects_projection_input_dim():
    with pytest.raises(ValueError, match="no input projection"):
        TeacherTransformer(input_dim=72, d_model=64)


def test_metrics_are_per_sample():
    """PROVENANCE 2.2 -- a perfect fit on a subset must score zero error."""
    y = np.array([1.0, 5.0, 9.0, 2.0])
    assert compute_metrics(y, y.copy()).mae == pytest.approx(0.0)
    assert compute_metrics(y, y + 2.0).mae == pytest.approx(2.0)


def test_undersampling_is_reproducible():
    """PROVENANCE 2.10 -- same seed, same retained subset."""
    from deepsalt.data.datasets import undersample_zeros

    X = torch.arange(200).float().unsqueeze(1)
    y = torch.cat([torch.zeros(150), torch.rand(50) * 10])
    set_seeds(0)
    _, a = undersample_zeros(X, y, 0.1, seed=7)
    set_seeds(999)
    _, b = undersample_zeros(X, y, 0.1, seed=7)
    assert torch.equal(a, b)


def test_undersampling_leaves_nonzeros_intact():
    from deepsalt.data.datasets import undersample_zeros

    X = torch.arange(200).float().unsqueeze(1)
    y = torch.cat([torch.zeros(150), torch.rand(50) * 10 + 1])
    _, kept = undersample_zeros(X, y, 0.1, seed=7)
    assert (kept > 0).sum().item() == 50
    assert (kept == 0).sum().item() == 15
