"""Losses for knowledge distillation.

Two changes from the original ``knowledge_distillation_loss``:

1. The KL term is gone. It computed
   ``F.kl_div(log_softmax(student_out, -1), softmax(teacher_out, -1))`` on
   tensors of shape ``(B, 1)``. A softmax over a length-1 axis is identically
   1, its log is identically 0, so the term returned exactly 0.0 for every
   batch and contributed no gradient. Its ``0.1 *`` weight was decorative. For
   scalar regression there is no distribution over classes to match; response
   distillation here is a direct regression of the student onto the teacher's
   prediction, which is what ``response_weight`` now does.

2. Layer weights are validated against the number of distilled layers instead
   of being silently clamped. The original used
   ``layer_weights[min(idx, len-1)]`` with 3 weights and 4 student layers, so
   the deepest student layer reused the last weight while
   ``num_layers = min(4, 3) = 3`` meant it received no supervision at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerWiseDistillationLoss(nn.Module):
    """Feature-matching loss between projected student features and teacher features.

    The student features passed in are expected to have ALREADY been mapped
    into teacher dimension by the Spectral Adaptation Unit. This module does no
    slicing of its own -- the original ``[:, :, :64]`` truncation now lives in
    ``SpectralAdaptationUnit(mode="truncate")``, where it is visible and
    switchable.
    """

    def __init__(
        self,
        layer_weights: list[float] | None = None,
        criterion: nn.Module | None = None,
        normalize_weights: bool = True,
    ):
        super().__init__()
        self.criterion = criterion or nn.SmoothL1Loss()
        self.layer_weights = layer_weights
        self.normalize_weights = normalize_weights

    def forward(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
    ) -> torch.Tensor:
        n_student, n_teacher = len(student_features), len(teacher_features)
        if n_student != n_teacher:
            raise ValueError(
                f"student has {n_student} distilled layers, teacher has "
                f"{n_teacher}. Pair them explicitly (e.g. map student layers "
                "[0,1,2,3] onto teacher layers [0,1,2,2]) rather than relying "
                "on zip() to drop the tail."
            )

        weights = self.layer_weights or [1.0] * n_student
        if len(weights) != n_student:
            raise ValueError(
                f"{len(weights)} layer weights supplied for {n_student} "
                "distilled layers."
            )

        total = student_features[0].new_zeros(())
        for w, s_feat, t_feat in zip(weights, student_features, teacher_features, strict=True):
            if s_feat.shape != t_feat.shape:
                raise ValueError(
                    f"shape mismatch after SAU projection: {tuple(s_feat.shape)} "
                    f"vs teacher {tuple(t_feat.shape)}"
                )
            total = total + w * self.criterion(s_feat, t_feat)

        denom = sum(weights) if self.normalize_weights else float(n_student)
        return total / denom


class DistillationObjective(nn.Module):
    """task + feature + (optional) response distillation.

    Weights are not constrained to sum to 1; report them as given.
    """

    def __init__(
        self,
        task_weight: float,
        feature_weight: float,
        response_weight: float = 0.0,
        task_criterion: nn.Module | None = None,
        feature_criterion: nn.Module | None = None,
        response_criterion: nn.Module | None = None,
    ):
        super().__init__()
        self.task_weight = task_weight
        self.feature_weight = feature_weight
        self.response_weight = response_weight
        self.task_criterion = task_criterion or nn.HuberLoss()
        self.feature_criterion = feature_criterion or LayerWiseDistillationLoss()
        self.response_criterion = response_criterion or nn.SmoothL1Loss()

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        targets = targets.reshape_as(student_output)

        task = self.task_criterion(student_output, targets)
        feature = self.feature_criterion(student_features, teacher_features)

        total = self.task_weight * task + self.feature_weight * feature
        parts = {"task": task.item(), "feature": feature.item(), "response": 0.0}

        if self.response_weight > 0.0:
            response = self.response_criterion(student_output, teacher_output.detach())
            total = total + self.response_weight * response
            parts["response"] = response.item()

        parts["total"] = total.item()
        return total, parts
