"""Stage 1 -- train a sparse stacked autoencoder on one spectral domain.

    python -m deepsalt.train.train_ssae --domain ftir  --config configs/default.yaml
    python -m deepsalt.train.train_ssae --domain enmap --config configs/default.yaml

FIXES APPLIED
-------------
* The fitted input scaler is SAVED alongside the weights. In the original,
  ``ftir_encoder.py`` fit a MinMaxScaler and trained on scaled spectra, then
  ``ftir_train.py`` fed the encoder RAW spectra straight from the OPUS files.
  The teacher therefore learned on latents the encoder was never trained to
  produce. Every consumer here loads the same persisted scaler.
* The autoencoder is fit on the TRAIN SPLIT only, not the full table, so its
  representation does not absorb validation and test spectra.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from ..data.preprocess import load_dataset
from ..models.autoencoder import (
    KLSparsityLoss,
    SparseStackedAutoencoder,
    reconstruction_sparsity_loss,
)
from ..utils import get_device, load_config, save_checkpoint, set_seeds, torch_generator


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)


def load_ftir_spectra(cfg: dict) -> np.ndarray:
    """Load raw FTIR spectra from the wide CSV.

    The original had two FTIR sources -- a wide CSV (``ftir_encoder.py``) and
    per-sample OPUS files indexed through ``input.csv`` (``ftir_train.py``).
    Both were 1765-d but nothing verified they were the same samples on the
    same wavenumber grid in the same order. This project uses the wide CSV as
    the single source of truth; ``train_teacher.py`` reads its salinity column
    from the same file, so the two cannot drift apart.
    """
    import pandas as pd

    section = cfg["ftir"]
    df = pd.read_csv(section["csv_path"], low_memory=False)
    first = int(section.get("first_spectral_column", 5))
    spectra = df.iloc[:, first:].to_numpy(dtype=float)
    print(f"FTIR spectra: {spectra.shape[0]} samples x {spectra.shape[1]} channels")
    if spectra.shape[1] != int(section.get("expected_channels", spectra.shape[1])):
        raise ValueError(
            f"expected {section['expected_channels']} FTIR channels, got "
            f"{spectra.shape[1]}. Check ftir.first_spectral_column."
        )
    return spectra


def load_enmap_reflectance(cfg: dict) -> np.ndarray:
    """Load the TRAIN split reflectance written by preprocess.py."""
    data = load_dataset(cfg["data"]["output_dir"])
    n_bands = data["n_bands"]
    X = data["X_train_reflectance"]
    print(f"EnMAP train reflectance: {X.shape[0]} samples x {n_bands} bands")
    return X[:, :n_bands]


def train_ssae(
    spectra: np.ndarray,
    *,
    domain: str,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    rho: float,
    beta: float,
    grad_clip: float,
    seed: int,
    output_path: Path,
    scaler_path: Path,
    device: torch.device,
    already_scaled: bool = False,
) -> SparseStackedAutoencoder:
    set_seeds(seed)

    if already_scaled:
        scaled = spectra
        scaler = None
        if scaled.min() < -1e-6 or scaled.max() > 1 + 1e-6:
            raise ValueError(
                "already_scaled=True but values fall outside [0, 1]; the "
                "decoder's Sigmoid cannot represent them."
            )
    else:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(spectra)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"  saved input scaler -> {scaler_path}")

    tensor = torch.as_tensor(scaled, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch_generator(seed),
    )

    model = SparseStackedAutoencoder(tensor.shape[1], latent_dim).to(device)
    model.apply(_init_weights)
    sparsity = KLSparsityLoss(rho=rho, beta=beta)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(3)
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, latent = model(batch)
            loss, recon_loss, sparse_loss = reconstruction_sparsity_loss(
                recon, batch, latent, sparsity
            )
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            totals += [loss.item(), recon_loss.item(), sparse_loss.item()]
        totals /= max(len(loader), 1)
        print(
            f"  epoch {epoch:3d}/{epochs}  total={totals[0]:.6f}  "
            f"recon={totals[1]:.6f}  sparsity={totals[2]:.6f}",
            flush=True,
        )

    save_checkpoint(
        model,
        output_path,
        {
            "stage": "ssae",
            "domain": domain,
            "input_dim": int(tensor.shape[1]),
            "latent_dim": latent_dim,
            "scaler_path": str(scaler_path) if scaler is not None else None,
            "aligned": False,
            "sparsity": {"rho": rho, "beta": beta},
            "seed": seed,
            "epochs": epochs,
        },
    )
    print(f"  saved {domain} SSAE -> {output_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sparse stacked autoencoder.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--domain", choices=["ftir", "enmap"], required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ssae_cfg = cfg["ssae"]
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    device = get_device()
    print(f"Training {args.domain} SSAE on {device}")

    if args.domain == "ftir":
        spectra = load_ftir_spectra(cfg)
        already_scaled = False
    else:
        spectra = load_enmap_reflectance(cfg)
        already_scaled = True  # preprocess.py already applied the band scaler

    train_ssae(
        spectra,
        domain=args.domain,
        latent_dim=ssae_cfg["latent_dim"],
        epochs=ssae_cfg["epochs"],
        batch_size=ssae_cfg["batch_size"],
        lr=ssae_cfg["lr"],
        weight_decay=ssae_cfg["weight_decay"],
        rho=ssae_cfg["rho"],
        beta=ssae_cfg["beta"],
        grad_clip=ssae_cfg.get("grad_clip", 1.0),
        seed=cfg["seed"],
        output_path=ckpt_dir / f"{args.domain}_ssae.pth",
        scaler_path=ckpt_dir / f"{args.domain}_input_scaler.pkl",
        device=device,
        already_scaled=already_scaled,
    )


if __name__ == "__main__":
    main()
