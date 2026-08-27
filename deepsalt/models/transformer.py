"""Transformer regressors for salinity.

The teacher consumes 64-d FTIR latents. The student consumes 64-d EnMAP
latents concatenated with ancillary covariates.

OUTPUT RANGE
------------
The original code carried three different output parameterizations across
scripts that shared checkpoint files -- ReLU on min-max-scaled targets,
``0.05 + 1.05 * sigmoid`` for [0.05, 1.1], and ``0.05 + 76.95 * sigmoid`` for
[0.05, 77.0] dS/m. A ``state_dict`` does not record which, so the same weights
could be loaded into a head that rescaled them by ~70x without error.

Here the range is a constructor argument, it is written into the checkpoint
sidecar, and ``load_checkpoint`` refuses to load a teacher whose recorded range
disagrees with the head being built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class OutputRange:
    """Target support, in the units the model predicts (dS/m unless stated)."""

    low: float
    high: float
    units: str = "dS/m"

    @property
    def span(self) -> float:
        return self.high - self.low

    def as_dict(self) -> dict:
        return {"low": self.low, "high": self.high, "units": self.units}


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding over the sequence axis.

    Retained for fidelity to the original models. Note that these models use a
    sequence length of 1 (each sample is one token), so the encoding adds a
    constant vector and carries no positional information. It is kept because
    removing it would change the learned bias and invalidate existing
    checkpoints; if you retrain from scratch, consider dropping it and say so.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        return x + self.pe[: x.size(1), :].unsqueeze(0)


class _TransformerRegressor(nn.Module):
    """Shared trunk. Subclasses differ only in input handling."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_range: OutputRange,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError(f"d_model={d_model} not divisible by nhead={nhead}")
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.output_range = output_range

        self.pos_encoder = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            layer, num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def _trunk(self, x: torch.Tensor):
        """x: (batch, d_model) -> predictions, pooled, per-layer features."""
        x = x.unsqueeze(1)  # (batch, seq=1, d_model)
        x = self.pos_encoder(x)

        intermediate = []
        for layer in self.transformer_encoder.layers:
            x = layer(x)
            intermediate.append(x)

        pooled = x.mean(dim=1)
        r = self.output_range
        predictions = r.low + r.span * self.sigmoid(self.fc(pooled))
        return predictions, pooled, intermediate

    def config(self) -> dict:
        return {
            "class_name": type(self).__name__,
            "d_model": self.d_model,
            "num_encoder_layers": self.num_encoder_layers,
            "output_range": self.output_range.as_dict(),
        }


class TeacherTransformer(_TransformerRegressor):
    """Laboratory-spectroscopy regressor. Input is a 64-d FTIR latent."""

    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_range: OutputRange | None = None,
    ):
        if input_dim != d_model:
            raise ValueError(
                "the teacher has no input projection; input_dim must equal "
                f"d_model (got {input_dim} vs {d_model})."
            )
        super().__init__(
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            output_range or OutputRange(0.05, 77.0),
        )

    def forward(self, x: torch.Tensor):
        return self._trunk(x)


class StudentTransformer(_TransformerRegressor):
    """Satellite regressor. Input is an EnMAP latent plus ancillary features."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 72,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_range: OutputRange | None = None,
    ):
        super().__init__(
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            output_range or OutputRange(0.05, 77.0),
        )
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor):
        return self._trunk(self.input_projection(x))

    def config(self) -> dict:
        cfg = super().config()
        cfg["input_dim"] = self.input_dim
        return cfg
