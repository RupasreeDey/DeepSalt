"""Spectral Adaptation Unit (SAU).

READ THIS BEFORE CITING THIS FILE IN THE PAPER
==============================================
The original scripts contained no module by this name. This file is a
reconstruction of what the abstract calls the Spectral Adaptation Unit,
assembled from two mechanisms that were present but scattered:

  (A) Latent-space alignment. ``align_enmap_ftir.py`` trained the EnMAP and
      FTIR autoencoders jointly, adding a cosine objective over geographically
      paired samples so that a satellite spectrum and a laboratory spectrum
      from the same place land near each other in the 64-d code. This is what
      makes it legitimate to query an FTIR-trained teacher with EnMAP latents.

  (B) Projection into teacher feature space. The student runs at d_model=72
      (64 spectral + 8 ancillary) while the teacher runs at 64. The original
      layer-wise distillation loss bridged this by slicing ``[:, :, :64]`` --
      truncating the student's last 8 channels. That is a projection with a
      fixed, non-learned, rank-deficient matrix.

Both are implemented here, and (B) is selectable so previously reported
numbers stay reproducible:

  ``mode="truncate"``  reproduces the original slice. Use for legacy runs.
  ``mode="linear"``    learned Linear(d_student -> d_teacher) + LayerNorm.
  ``mode="residual"``  learned projection with a residual path on the
                       spectral channels, so the ancillary features modulate
                       rather than replace the spectral code.

VERIFY BEFORE PUBLISHING
------------------------
If your paper's SAU is a different construction, replace this module and keep
the interface. Do not let this file define the contribution by default.

CRITICAL PROVENANCE NOTE
------------------------
In the code as originally submitted, mechanism (A) was trained but never
consumed: ``align_enmap_ftir.py`` wrote ``enmap_encoder_ssae_cosine.pth`` and
every downstream script loaded ``enmap_ssae.pth``, which came from
``enmap_encoder.py`` (reconstruction + sparsity only, no FTIR term). If the
reported results were produced that way, they were produced WITHOUT alignment.
This package wires (A) into the pipeline by default and records which encoder
was used in each checkpoint's sidecar metadata. Re-run before reporting SAU
ablations.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

ProjectionMode = Literal["truncate", "linear", "residual"]


class FeatureProjection(nn.Module):
    """Maps one student hidden state into the teacher's feature dimension."""

    def __init__(
        self,
        d_student: int,
        d_teacher: int,
        mode: ProjectionMode = "linear",
    ):
        super().__init__()
        if d_teacher > d_student:
            raise ValueError(
                f"teacher dim {d_teacher} exceeds student dim {d_student}; "
                "the student must be at least as wide as the teacher."
            )
        self.d_student = d_student
        self.d_teacher = d_teacher
        self.mode = mode

        if mode == "truncate":
            self.proj = None
        elif mode == "linear":
            self.proj = nn.Sequential(
                nn.Linear(d_student, d_teacher), nn.LayerNorm(d_teacher)
            )
        elif mode == "residual":
            self.proj = nn.Sequential(
                nn.Linear(d_student, d_teacher), nn.LayerNorm(d_teacher)
            )
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"unknown projection mode {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "truncate":
            return x[..., : self.d_teacher]
        projected = self.proj(x)
        if self.mode == "residual":
            # gate starts at 0 -> initially an identity on the spectral block,
            # so training begins from the legacy truncation behaviour and
            # learns away from it.
            return x[..., : self.d_teacher] + torch.tanh(self.gate) * projected
        return projected


class SpectralAdaptationUnit(nn.Module):
    """Per-layer projections from student space into teacher space.

    One projection per distilled layer: the layers sit at different depths and
    a shared map would force them into the same subspace.
    """

    def __init__(
        self,
        d_student: int,
        d_teacher: int,
        num_layers: int,
        mode: ProjectionMode = "linear",
    ):
        super().__init__()
        self.mode = mode
        self.projections = nn.ModuleList(
            FeatureProjection(d_student, d_teacher, mode) for _ in range(num_layers)
        )

    def forward(self, student_features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(student_features) < len(self.projections):
            raise ValueError(
                f"SAU built for {len(self.projections)} layers but received "
                f"{len(student_features)}."
            )
        return [
            proj(feat)
            for proj, feat in zip(self.projections, student_features, strict=False)
        ]

    def extra_repr(self) -> str:
        return f"mode={self.mode}, num_layers={len(self.projections)}"


# --------------------------------------------------------------------------- #
# Mechanism (A): the alignment objective used to pretrain the two encoders.
# --------------------------------------------------------------------------- #


def latent_alignment_loss(
    enmap_latent: torch.Tensor,
    ftir_latent: torch.Tensor,
    negatives: bool = True,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Pull paired satellite/lab codes together in the 64-d space.

    ``negatives=False`` reproduces the original objective: mean cosine
    distance over positive pairs only, with nothing pushing unrelated samples
    apart. It was labelled "contrastive" in the original code but has no
    negatives, so it admits the degenerate solution where both encoders map
    everything to one direction.

    ``negatives=True`` (default) uses InfoNCE over the in-batch pairs, which is
    contrastive in the usual sense. Call it what you use in the paper.
    """
    a = F.normalize(enmap_latent, p=2, dim=1)
    b = F.normalize(ftir_latent, p=2, dim=1)

    if not negatives:
        return 1.0 - (a * b).sum(dim=1).mean()

    logits = (a @ b.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    # symmetric: satellite->lab and lab->satellite
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
