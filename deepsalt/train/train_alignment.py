"""Stage 2 -- align the EnMAP and FTIR latent spaces (SAU mechanism A).

    python -m deepsalt.train.train_alignment --config configs/default.yaml

Writes ``enmap_ssae_aligned.pth``, which ``train_student.py`` loads by default.

THE BUG THIS STAGE EXISTS TO FIX
--------------------------------
In the original code this stage ran and was then discarded.
``align_enmap_ftir.py`` saved ``enmap_encoder_ssae_cosine.pth``; every
downstream script loaded ``enmap_ssae.pth``, produced by ``enmap_encoder.py``
with no FTIR term at all. Across all 21 original scripts the aligned encoder
had zero consumers. Whatever numbers were reported for the Spectral Adaptation
Unit were, on that evidence, produced without it.

Here the output filename is what the student loads, the alignment flag is
written into the checkpoint sidecar, and ``train_student.py`` prints which
encoder it used. The ``--no-alignment`` ablation is a config switch, not a
different file.

OTHER FIXES
-----------
* Both reconstructions enter the loss. The original computed ``ftir_recon``
  and discarded it, leaving the FTIR encoder anchored only by the cosine term
  and free to drift.
* Scalers are loaded from stage 1 rather than refit, and are fit on train
  data only, so alignment does not leak test spectra.
* The objective defaults to InfoNCE with in-batch negatives. The original had
  no negatives despite the variable being named ``contrastive_loss``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..data.pairing import pair_by_location
from ..models.autoencoder import KLSparsityLoss, SparseStackedAutoencoder
from ..models.sau import latent_alignment_loss
from ..utils import (
    get_device,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seeds,
    torch_generator,
)


def _load_encoder(
    path: Path, input_dim: int, latent_dim: int, device: torch.device
) -> SparseStackedAutoencoder:
    model = SparseStackedAutoencoder(input_dim, latent_dim).to(device)
    load_checkpoint(model, path, device, expect={"stage": "ssae"})
    return model


def build_pairs(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (paired scaled EnMAP reflectance, paired scaled FTIR spectra)."""
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    align_cfg = cfg["alignment"]

    enmap_df = pd.read_csv(cfg["data"]["csv_path"], low_memory=False)
    ftir_df = pd.read_csv(cfg["ftir"]["csv_path"], low_memory=False)

    from ..data.preprocess import PreprocessConfig, band_columns

    bands = band_columns(
        PreprocessConfig(
            csv_path=cfg["data"]["csv_path"],
            output_dir=cfg["data"]["output_dir"],
            band_prefix=cfg["data"]["band_prefix"],
            n_bands=cfg["data"]["n_bands"],
            excluded_bands=tuple(cfg["data"]["excluded_bands"]),
        )
    )

    lat, lon = cfg["data"]["latitude_column"], cfg["data"]["longitude_column"]
    enmap_ok = enmap_df[[lat, lon, *bands]].notna().all(axis=1)
    enmap_df = enmap_df[enmap_ok]

    first = int(cfg["ftir"].get("first_spectral_column", 5))
    ftir_ok = ftir_df[[lat, lon]].notna().all(axis=1)
    ftir_df = ftir_df[ftir_ok]

    pairs = pair_by_location(
        ftir_df[[lat, lon]].to_numpy(float),
        enmap_df[[lat, lon]].to_numpy(float),
        max_distance_km=align_cfg["max_distance_km"],
        strategy=align_cfg.get("match_strategy", "mutual"),
    )
    print(f"Pairing: {pairs.summary()}")
    if len(pairs) < align_cfg.get("min_pairs", 32):
        raise ValueError(
            f"only {len(pairs)} pairs within {align_cfg['max_distance_km']} km. "
            "Alignment on this few pairs will not generalize -- widen the "
            "radius (and report it) or check that both tables use the same "
            "coordinate reference."
        )

    band_scaler = joblib.load(Path(cfg["data"]["output_dir"]) / "scalers" / "band_scaler.pkl")
    ftir_scaler = joblib.load(ckpt_dir / "ftir_input_scaler.pkl")

    enmap_paired = band_scaler.transform(
        enmap_df.iloc[pairs.enmap_indices][bands].to_numpy(float)
    )
    ftir_paired = ftir_scaler.transform(
        ftir_df.iloc[pairs.ftir_indices].iloc[:, first:].to_numpy(float)
    )
    return np.clip(enmap_paired, 0, 1), np.clip(ftir_paired, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align EnMAP and FTIR latent spaces.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    align_cfg = cfg["alignment"]
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    device = get_device()
    set_seeds(cfg["seed"])

    enmap_X, ftir_X = build_pairs(cfg)
    latent_dim = cfg["ssae"]["latent_dim"]

    enmap_encoder = _load_encoder(
        ckpt_dir / "enmap_ssae.pth", enmap_X.shape[1], latent_dim, device
    )
    ftir_encoder = _load_encoder(
        ckpt_dir / "ftir_ssae.pth", ftir_X.shape[1], latent_dim, device
    )

    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(enmap_X, dtype=torch.float32),
            torch.as_tensor(ftir_X, dtype=torch.float32),
        ),
        batch_size=align_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        generator=torch_generator(cfg["seed"]),
    )

    sparsity = KLSparsityLoss(rho=cfg["ssae"]["rho"], beta=cfg["ssae"]["beta"])
    optimizer = optim.Adam(
        list(enmap_encoder.parameters()) + list(ftir_encoder.parameters()),
        lr=align_cfg["lr"],
        weight_decay=align_cfg["weight_decay"],
    )

    w_recon = align_cfg["reconstruction_weight"]
    w_align = align_cfg["alignment_weight"]
    use_negatives = align_cfg.get("use_negatives", True)

    for epoch in range(1, align_cfg["epochs"] + 1):
        enmap_encoder.train()
        ftir_encoder.train()
        totals = np.zeros(3)
        for enmap_batch, ftir_batch in loader:
            enmap_batch = enmap_batch.to(device)
            ftir_batch = ftir_batch.to(device)
            optimizer.zero_grad()

            enmap_recon, enmap_z = enmap_encoder(enmap_batch)
            ftir_recon, ftir_z = ftir_encoder(ftir_batch)

            # BOTH reconstructions -- the original dropped the FTIR one.
            recon = nn.functional.mse_loss(
                enmap_recon, enmap_batch
            ) + nn.functional.mse_loss(ftir_recon, ftir_batch)
            align = latent_alignment_loss(
                enmap_z,
                ftir_z,
                negatives=use_negatives,
                temperature=align_cfg.get("temperature", 0.07),
            )
            sparse = sparsity(enmap_z) + sparsity(ftir_z)

            loss = w_recon * recon + w_align * align + sparse
            loss.backward()
            optimizer.step()
            totals += [loss.item(), recon.item(), align.item()]

        totals /= max(len(loader), 1)
        print(
            f"  epoch {epoch:3d}/{align_cfg['epochs']}  total={totals[0]:.6f}  "
            f"recon={totals[1]:.6f}  align={totals[2]:.6f}",
            flush=True,
        )

    shared = {
        "stage": "ssae",
        "latent_dim": latent_dim,
        "aligned": True,
        "alignment": {
            "objective": "infonce" if use_negatives else "cosine_positives_only",
            "n_pairs": int(len(enmap_X)),
            "max_distance_km": align_cfg["max_distance_km"],
            "match_strategy": align_cfg.get("match_strategy", "mutual"),
            "alignment_weight": w_align,
        },
        "seed": cfg["seed"],
    }
    save_checkpoint(
        enmap_encoder,
        ckpt_dir / "enmap_ssae_aligned.pth",
        shared | {"domain": "enmap", "input_dim": int(enmap_X.shape[1])},
    )
    save_checkpoint(
        ftir_encoder,
        ckpt_dir / "ftir_ssae_aligned.pth",
        shared | {"domain": "ftir", "input_dim": int(ftir_X.shape[1])},
    )
    print(f"Saved aligned encoders -> {ckpt_dir}")
    print("train_student.py loads enmap_ssae_aligned.pth by default; set")
    print("student.use_aligned_encoder=false for the no-adaptation ablation.")


if __name__ == "__main__":
    main()
