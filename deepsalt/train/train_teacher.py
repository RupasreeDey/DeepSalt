"""Stage 3 -- train the laboratory-spectroscopy teacher.

    python -m deepsalt.train.train_teacher --config configs/default.yaml

Produces ``teacher.pth``. This is the checkpoint the original repository never
generated: ``ftir_salinity_model_ssae_clipped_final.pth`` (and ``_v3``,
``_resnet``) were loaded by eight scripts and written by none.

FIXES APPLIED
-------------
* The teacher predicts in PHYSICAL UNITS (dS/m), on the same scale as the
  student's targets. The original trained on MinMax-scaled targets and emitted
  [0, 1] or [0.05, 1.1], while the student worked in dS/m and the shared
  ``TransformerModelTeacher`` class rescaled the same weights to [0.05, 77] --
  a ~70x discrepancy with nothing to catch it. The output range is recorded in
  the sidecar and checked on load.
* Spectra pass through the SAME persisted scaler the encoder was fit with,
  instead of going in raw.
* One FTIR source, one split, fit before any scaling.
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ..models.autoencoder import SparseStackedAutoencoder
from ..models.transformer import OutputRange, TeacherTransformer
from ..utils import (
    compute_metrics,
    get_device,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seeds,
    torch_generator,
)


def load_ftir_supervised(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    section = cfg["ftir"]
    df = pd.read_csv(section["csv_path"], low_memory=False)

    target_col = section["target_column"]
    if target_col not in df.columns:
        raise KeyError(
            f"FTIR salinity column {target_col!r} not in {section['csv_path']}. "
            f"Available non-spectral columns: {list(df.columns[:5])}"
        )

    first = int(section.get("first_spectral_column", 5))
    spectra = df.iloc[:, first:].to_numpy(dtype=float)
    targets = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    targets = targets * float(section.get("target_scale", 1.0))

    ok = np.isfinite(spectra).all(axis=1) & np.isfinite(targets)
    print(f"FTIR supervised set: {ok.sum()}/{len(df)} usable rows")
    return spectra[ok], targets[ok]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FTIR teacher.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--aligned",
        action="store_true",
        help="encode with the alignment-trained FTIR encoder rather than the "
        "stage-1 encoder. Use this when the student will consume the aligned "
        "EnMAP encoder, so both sit in the same latent space.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["teacher"]
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    device = get_device()
    set_seeds(cfg["seed"])

    spectra, targets = load_ftir_supervised(cfg)

    # Split BEFORE any fitting.
    X_tr, X_te, y_tr, y_te = train_test_split(
        spectra, targets, test_size=tcfg["test_fraction"], random_state=cfg["seed"]
    )

    scaler = joblib.load(ckpt_dir / "ftir_input_scaler.pkl")
    X_tr = np.clip(scaler.transform(X_tr), 0, 1)
    X_te = np.clip(scaler.transform(X_te), 0, 1)

    encoder_name = "ftir_ssae_aligned.pth" if args.aligned else "ftir_ssae.pth"
    encoder = SparseStackedAutoencoder(spectra.shape[1], cfg["ssae"]["latent_dim"]).to(device)
    meta = load_checkpoint(encoder, ckpt_dir / encoder_name, device, expect={"stage": "ssae"})
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Encoder: {encoder_name} (aligned={meta.get('aligned')})")

    z_tr = encoder.embed_numpy(X_tr, device)
    z_te = encoder.embed_numpy(X_te, device)

    output_range = OutputRange(**cfg["model"]["output_range"])
    observed = (float(y_tr.min()), float(y_tr.max()))
    if observed[1] > output_range.high or observed[0] < output_range.low:
        raise ValueError(
            f"FTIR targets span {observed} but output_range is "
            f"[{output_range.low}, {output_range.high}] {output_range.units}. "
            "A sigmoid head cannot reach targets outside its range."
        )
    print(f"Targets span {observed} {output_range.units}")

    model = TeacherTransformer(
        input_dim=cfg["ssae"]["latent_dim"],
        d_model=tcfg["d_model"],
        nhead=tcfg["nhead"],
        num_encoder_layers=tcfg["num_encoder_layers"],
        dim_feedforward=tcfg["dim_feedforward"],
        dropout=tcfg["dropout"],
        output_range=output_range,
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(z_tr, dtype=torch.float32),
            torch.as_tensor(y_tr, dtype=torch.float32),
        ),
        batch_size=tcfg["batch_size"],
        shuffle=True,
        generator=torch_generator(cfg["seed"]),
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(z_te, dtype=torch.float32),
            torch.as_tensor(y_te, dtype=torch.float32),
        ),
        batch_size=tcfg["batch_size"],
    )

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0)
    )

    best = float("inf")
    patience = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch, target in train_loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad()
            predictions, _, _ = model(batch)
            loss = criterion(predictions, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch, target in test_loader:
                p, _, _ = model(batch.to(device))
                preds.append(p.cpu().numpy().ravel())
                trues.append(target.numpy().ravel())
        metrics = compute_metrics(np.concatenate(trues), np.concatenate(preds))

        if epoch % tcfg.get("log_every", 10) == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d}/{tcfg['epochs']}  train={train_loss:.4f}  {metrics}",
                flush=True,
            )

        if metrics.mae < best - 1e-6:
            best = metrics.mae
            patience = 0
            save_checkpoint(
                model,
                ckpt_dir / "teacher.pth",
                model.config()
                | {
                    "stage": "teacher",
                    "encoder": encoder_name,
                    "encoder_aligned": bool(meta.get("aligned")),
                    "test_metrics": metrics.as_dict(),
                    "seed": cfg["seed"],
                    "epoch": epoch,
                },
            )
        else:
            patience += 1
            if patience >= tcfg.get("early_stopping_patience", 25):
                print(f"  early stop at epoch {epoch}")
                break

    print(f"Best teacher test MAE: {best:.4f} {output_range.units}")
    print(f"Saved -> {ckpt_dir / 'teacher.pth'}")


if __name__ == "__main__":
    main()
