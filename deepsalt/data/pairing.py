"""Geographic pairing of FTIR sampling sites with EnMAP pixels.

Two fixes relative to ``align_enmap_ftir.py``:

1. UNITS. ``BallTree(metric="haversine")`` requires radians and returns
   distances in radians of great-circle arc. The original passed degrees and
   compared against ``max_distance=0.01``, so both the neighbour ranking and
   the radius cut were operating on a meaningless scale. Coordinates are now
   converted with ``np.radians`` and the threshold is specified in kilometres,
   converted internally by dividing by the Earth radius.

2. ORDER DEPENDENCE. The original claimed each EnMAP pixel for the first FTIR
   site that reached it, iterating in row order, and silently dropped later
   sites. Matching is now mutual-nearest-neighbour by default: a pair is kept
   only if each is the other's closest partner, which is symmetric and
   independent of row order. ``greedy`` reproduces the original behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088

MatchStrategy = Literal["mutual", "greedy"]


@dataclass
class PairingResult:
    ftir_indices: np.ndarray
    enmap_indices: np.ndarray
    distances_km: np.ndarray

    def __len__(self) -> int:
        return len(self.ftir_indices)

    def summary(self) -> str:
        if not len(self):
            return "0 pairs formed"
        d = self.distances_km
        return (
            f"{len(self)} pairs | distance km: "
            f"min={d.min():.3f} median={np.median(d):.3f} "
            f"mean={d.mean():.3f} max={d.max():.3f}"
        )


def pair_by_location(
    ftir_locations: np.ndarray,
    enmap_locations: np.ndarray,
    max_distance_km: float = 1.0,
    strategy: MatchStrategy = "mutual",
) -> PairingResult:
    """Match FTIR sites to EnMAP pixels by great-circle distance.

    Parameters
    ----------
    ftir_locations, enmap_locations
        ``(n, 2)`` arrays of ``[latitude, longitude]`` in DEGREES.
    max_distance_km
        Pairs farther apart than this are discarded. Choose it with the EnMAP
        ground sample distance in mind (30 m nominal); a 1 km threshold admits
        pairs more than 30 pixels apart, so report whatever you use.
    """
    ftir_locations = np.asarray(ftir_locations, dtype=float)
    enmap_locations = np.asarray(enmap_locations, dtype=float)
    if ftir_locations.ndim != 2 or ftir_locations.shape[1] != 2:
        raise ValueError("ftir_locations must be (n, 2) [lat, lon] in degrees")
    if enmap_locations.ndim != 2 or enmap_locations.shape[1] != 2:
        raise ValueError("enmap_locations must be (n, 2) [lat, lon] in degrees")
    if len(ftir_locations) == 0 or len(enmap_locations) == 0:
        empty = np.empty(0, dtype=int)
        return PairingResult(empty, empty, np.empty(0))

    ftir_rad = np.radians(ftir_locations)
    enmap_rad = np.radians(enmap_locations)
    radius_rad = max_distance_km / EARTH_RADIUS_KM

    enmap_tree = BallTree(enmap_rad, metric="haversine")
    dist_f2e, idx_f2e = enmap_tree.query(ftir_rad, k=1)
    dist_f2e = dist_f2e.ravel()
    idx_f2e = idx_f2e.ravel()

    within = dist_f2e <= radius_rad

    if strategy == "mutual":
        ftir_tree = BallTree(ftir_rad, metric="haversine")
        _, idx_e2f = ftir_tree.query(enmap_rad, k=1)
        idx_e2f = idx_e2f.ravel()
        ftir_ids = np.arange(len(ftir_rad))
        keep = within & (idx_e2f[idx_f2e] == ftir_ids)
        ftir_sel = ftir_ids[keep]
        enmap_sel = idx_f2e[keep]
        dist_sel = dist_f2e[keep]
    elif strategy == "greedy":
        ftir_sel, enmap_sel, dist_sel, claimed = [], [], [], set()
        for f_idx in np.flatnonzero(within):
            e_idx = int(idx_f2e[f_idx])
            if e_idx in claimed:
                continue
            claimed.add(e_idx)
            ftir_sel.append(int(f_idx))
            enmap_sel.append(e_idx)
            dist_sel.append(float(dist_f2e[f_idx]))
        ftir_sel = np.asarray(ftir_sel, dtype=int)
        enmap_sel = np.asarray(enmap_sel, dtype=int)
        dist_sel = np.asarray(dist_sel, dtype=float)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    return PairingResult(ftir_sel, enmap_sel, dist_sel * EARTH_RADIUS_KM)
