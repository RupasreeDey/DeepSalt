"""Sparse stacked autoencoder used to compress both spectral domains to 64-d.

Architecture is unchanged from the original scripts (``enmap_encoder.py`` and
``ftir_encoder.py``): 1024-512-256-latent with ReLU throughout, mirrored
decoder, Sigmoid output. The Sigmoid means inputs MUST be min-max scaled to
[0, 1] -- see ``deepsalt/data/preprocess.py``, which persists the fitted
scalers so every consumer applies the same transform.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseStackedAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def embed_numpy(self, array, device: torch.device, batch_size: int = 4096):
        """Encode a numpy array in batches. Input must already be scaled."""
        import numpy as np

        self.eval()
        out = []
        for start in range(0, len(array), batch_size):
            chunk = torch.as_tensor(
                array[start : start + batch_size], dtype=torch.float32, device=device
            )
            out.append(self.encode(chunk).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.empty((0, self.latent_dim))


class KLSparsityLoss(nn.Module):
    """KL divergence between a target activation rate rho and the batch mean.

    Applied to the latent code. Because the latent activation is ReLU (range
    [0, inf)) rather than a saturating nonlinearity, rho_hat is clamped into
    (0, 1) before the log; activations above 1 are therefore treated as fully
    active. This matches the original implementation.
    """

    def __init__(self, rho: float = 0.05, beta: float = 0.1, eps: float = 1e-10):
        super().__init__()
        self.rho = rho
        self.beta = beta
        self.eps = eps

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        rho_hat = torch.clamp(activations.mean(dim=0), self.eps, 1.0 - self.eps)
        kl = self.rho * torch.log(self.rho / rho_hat) + (1 - self.rho) * torch.log(
            (1 - self.rho) / (1 - rho_hat)
        )
        return self.beta * torch.sum(kl)


def reconstruction_sparsity_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    latent: torch.Tensor,
    sparsity: KLSparsityLoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = nn.functional.mse_loss(recon, target)
    sparse_loss = sparsity(latent)
    return recon_loss + sparse_loss, recon_loss, sparse_loss
