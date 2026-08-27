"""Stage 4 -- distil the teacher into the satellite student.

    python -m deepsalt.train.train_student --config configs/default.yaml

ABLATIONS (config switches, not separate scripts)
-------------------------------------------------
    student.use_aligned_encoder: false   # no SAU alignment
    sau.projection_mode: truncate        # legacy [:, :, :64] slice
    loss.feature_weight: 0.0             # no feature distillation
    loss.response_weight: 0.0            # no response distillation

Setting both distillation weights to 0 gives the EnMAP-only baseline, so every
row of the ablation table comes from this one code path.

FIXES APPLIED
-------------
* The dead KL term is gone (see deepsalt/losses.py).
* Student features reach the teacher's dimension through the SAU rather than a
  hardcoded slice.
* ``teacher_input`` is the explicit spectral block from the feature layout,
  not ``data[:, :64]``, which silently assumed no categorical column had been
  deleted from the first 64 positions.
* The teacher's recorded output range is verified against the student's on
  load; a mismatch raises instead of scaling predictions by ~70x.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..data.datasets import build_layout, make_loader, split_categorical, undersample_zeros
from ..data.preprocess import load_dataset
from ..losses import DistillationObjective, LayerWiseDistillationLoss
from ..models.autoencoder import SparseStackedAutoencoder
from ..models.sau import SpectralAdaptationUnit
from ..models.transformer import OutputRange, StudentTransformer, TeacherTransformer
from ..utils import (
    compute_metrics,
    get_device,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seeds,
)


def embed_split(encoder, X, n_bands, device):
    """Replace the reflectance block with its 64-d code, keep the rest."""
    if X is None or len(X) == 0:
        return None
    latent = encoder.embed_numpy(X[:, :n_bands], device)
    return np.hstack([latent, X[:, n_bands:]])


def run_epoch(
    loader,
    student,
    sau,
    teacher,
    objective,
    layout,
    device,
    optimizer=None,
    teacher_layer_map=None,
):
    training = optimizer is not None
    student.train(training)
    sau.train(training)
    teacher.eval()

    sums: dict[str, float] = {}
    n_batches = 0

    with torch.set_grad_enabled(training):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if training:
                optimizer.zero_grad()

            student_out, _, student_feats = student(data)
            projected = sau(student_feats)

            with torch.no_grad():
                teacher_input = data[:, : layout.n_spectral]
                teacher_out, _, teacher_feats = teacher(teacher_input)

            paired_teacher = [teacher_feats[i] for i in teacher_layer_map]

            loss, parts = objective(
                student_out, teacher_out, projected, paired_teacher, target
            )

            if training:
                loss.backward()
                optimizer.step()

            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(loader, student, device, categorical, layout):
    """Overall and per-category metrics.

    The original per-category loop read ``all_predictions[-1]`` inside a loop
    over the batch, so every sample in a batch was scored with the LAST
    sample's prediction. Indices are tracked explicitly here.
    """
    student.eval()
    preds, trues = [], []
    for data, target in loader:
        out, _, _ = student(data.to(device))
        preds.append(out.cpu().numpy().ravel())
        trues.append(target.numpy().ravel())

    preds = np.concatenate(preds) if preds else np.empty(0)
    trues = np.concatenate(trues) if trues else np.empty(0)
    result = {"overall": compute_metrics(trues, preds).as_dict()}

    if categorical is None or categorical.size == 0:
        return result, preds, trues

    if len(categorical) != len(preds):
        raise ValueError(
            f"categorical block has {len(categorical)} rows but {len(preds)} "
            "predictions -- the evaluation loader must not shuffle or drop."
        )

    names = layout.categorical_names
    for prefix in ("koppen", "soil"):
        columns = [i for i, n in enumerate(names) if n.startswith(f"{prefix}_")]
        if not columns:
            continue
        assigned = np.asarray(columns)[np.argmax(categorical[:, columns], axis=1)]
        bucket = {}
        for col in np.unique(assigned):
            mask = assigned == col
            label = names[col][len(prefix) + 1 :]
            bucket[label] = compute_metrics(trues[mask], preds[mask]).as_dict()
        result[prefix] = bucket

    return result, preds, trues


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the student by distillation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default="student", help="checkpoint / results name")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scfg, sau_cfg, lcfg = cfg["student"], cfg["sau"], cfg["loss"]
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    set_seeds(cfg["seed"])

    data = load_dataset(cfg["data"]["output_dir"])
    n_bands = data["n_bands"]
    latent_dim = cfg["ssae"]["latent_dim"]
    n_ancillary = len(data["ancillary_columns"])
    layout = build_layout(data["feature_names_after_embedding"], latent_dim, n_ancillary)

    # ---- encoder --------------------------------------------------------- #
    use_aligned = scfg.get("use_aligned_encoder", True)
    encoder_name = "enmap_ssae_aligned.pth" if use_aligned else "enmap_ssae.pth"
    encoder = SparseStackedAutoencoder(n_bands, latent_dim).to(device)
    enc_meta = load_checkpoint(encoder, ckpt_dir / encoder_name, device, expect={"stage": "ssae"})
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Encoder: {encoder_name}  aligned={enc_meta.get('aligned')}")

    splits = {}
    for name in ("train", "val", "test"):
        X = embed_split(encoder, data.get(f"X_{name}_reflectance"), n_bands, device)
        splits[name] = (X, data.get(f"y_{name}"))
    unseen_X = embed_split(encoder, data.get("unseen_X_test_reflectance"), n_bands, device)
    unseen_y = data.get("unseen_y_test")

    # ---- inputs vs categorical ------------------------------------------- #
    prepared = {}
    for name, (X, y) in splits.items():
        if X is None:
            prepared[name] = (None, None, None)
            continue
        model_X, cat = split_categorical(X, layout)
        prepared[name] = (model_X, y, cat)

    train_X = torch.as_tensor(prepared["train"][0], dtype=torch.float32)
    train_y = torch.as_tensor(prepared["train"][1], dtype=torch.float32)
    if scfg.get("undersample_zeros", True):
        train_X, train_y = undersample_zeros(
            train_X, train_y, scfg.get("zero_keep_fraction", 0.10), cfg["seed"]
        )

    train_loader = make_loader(
        train_X.numpy(), train_y.numpy(), scfg["batch_size"], True, cfg["seed"], drop_last=True
    )
    val_loader = make_loader(prepared["val"][0], prepared["val"][1], scfg["batch_size"], False)
    test_loader = make_loader(prepared["test"][0], prepared["test"][1], scfg["batch_size"], False)

    unseen_loader = unseen_cat = None
    if unseen_X is not None and len(unseen_X):
        unseen_model_X, unseen_cat = split_categorical(unseen_X, layout)
        unseen_loader = make_loader(unseen_model_X, unseen_y, scfg["batch_size"], False)
        print(f"Unseen holdout: {len(unseen_y)} samples")
    else:
        print("Unseen holdout: EMPTY -- configure data.holdout_regions to report "
              "geographic generalization.")

    # ---- models ---------------------------------------------------------- #
    output_range = OutputRange(**cfg["model"]["output_range"])
    student = StudentTransformer(
        input_dim=layout.model_input_dim,
        d_model=scfg["d_model"],
        nhead=scfg["nhead"],
        num_encoder_layers=scfg["num_encoder_layers"],
        dim_feedforward=scfg["dim_feedforward"],
        dropout=scfg["dropout"],
        output_range=output_range,
    ).to(device)

    teacher = TeacherTransformer(
        input_dim=latent_dim,
        d_model=cfg["teacher"]["d_model"],
        nhead=cfg["teacher"]["nhead"],
        num_encoder_layers=cfg["teacher"]["num_encoder_layers"],
        dim_feedforward=cfg["teacher"]["dim_feedforward"],
        dropout=cfg["teacher"]["dropout"],
        output_range=output_range,
    ).to(device)
    load_checkpoint(
        teacher,
        ckpt_dir / "teacher.pth",
        device,
        expect={"stage": "teacher", "output_range": output_range.as_dict()},
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student has more layers than the teacher; state the pairing explicitly
    # instead of letting zip() drop the deepest student layer unsupervised.
    teacher_layer_map = scfg.get("teacher_layer_map")
    if teacher_layer_map is None:
        last = cfg["teacher"]["num_encoder_layers"] - 1
        teacher_layer_map = [min(i, last) for i in range(scfg["num_encoder_layers"])]
    print(f"Layer pairing student->teacher: {teacher_layer_map}")

    sau = SpectralAdaptationUnit(
        d_student=scfg["d_model"],
        d_teacher=cfg["teacher"]["d_model"],
        num_layers=scfg["num_encoder_layers"],
        mode=sau_cfg["projection_mode"],
    ).to(device)
    print(f"SAU: {sau}")

    layer_weights = lcfg.get("layer_weights") or [1.0] * scfg["num_encoder_layers"]
    objective = DistillationObjective(
        task_weight=lcfg["task_weight"],
        feature_weight=lcfg["feature_weight"],
        response_weight=lcfg.get("response_weight", 0.0),
        feature_criterion=LayerWiseDistillationLoss(layer_weights=layer_weights),
    )

    optimizer = optim.Adam(
        list(student.parameters()) + list(sau.parameters()),
        lr=scfg["lr"],
        weight_decay=scfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    # ---- training loop --------------------------------------------------- #
    history, best_val, patience = [], float("inf"), 0
    for epoch in range(1, scfg["epochs"] + 1):
        train_parts = run_epoch(
            train_loader, student, sau, teacher, objective, layout, device,
            optimizer, teacher_layer_map,
        )
        val_parts = run_epoch(
            val_loader, student, sau, teacher, objective, layout, device,
            None, teacher_layer_map,
        )
        history.append({"epoch": epoch, "train": train_parts, "val": val_parts})
        scheduler.step(val_parts["total"])

        print(
            f"  epoch {epoch:3d}/{scfg['epochs']}  "
            f"train {train_parts['total']:.4f} "
            f"(task {train_parts['task']:.4f} feat {train_parts['feature']:.4f}) | "
            f"val {val_parts['total']:.4f} "
            f"(task {val_parts['task']:.4f} feat {val_parts['feature']:.4f})",
            flush=True,
        )

        if val_parts["total"] < best_val - 1e-6:
            best_val, patience = val_parts["total"], 0
            save_checkpoint(
                student,
                ckpt_dir / f"{args.tag}.pth",
                student.config()
                | {
                    "stage": "student",
                    "encoder": encoder_name,
                    "encoder_aligned": bool(enc_meta.get("aligned")),
                    "sau_mode": sau_cfg["projection_mode"],
                    "loss_weights": dict(lcfg),
                    "val_loss": best_val,
                    "seed": cfg["seed"],
                    "epoch": epoch,
                },
            )
            save_checkpoint(sau, ckpt_dir / f"{args.tag}_sau.pth", {"stage": "sau"} | dict(sau_cfg))
        else:
            patience += 1
            if patience >= scfg.get("early_stopping_patience", 10):
                print(f"  early stop at epoch {epoch}")
                break

    load_checkpoint(student, ckpt_dir / f"{args.tag}.pth", device, expect={"stage": "student"})

    # ---- evaluation ------------------------------------------------------ #
    report = {
        "config": {
            "encoder": encoder_name,
            "encoder_aligned": bool(enc_meta.get("aligned")),
            "sau_mode": sau_cfg["projection_mode"],
            "loss_weights": dict(lcfg),
            "seed": cfg["seed"],
        },
        "history": history,
    }

    test_results, _, _ = evaluate(test_loader, student, device, prepared["test"][2], layout)
    report["test"] = test_results
    overall = test_results["overall"]
    print(
        f"\nTest   MAE={overall['mae']:.4f}  RMSE={overall['rmse']:.4f}  "
        f"R2={overall['r2']:.4f}  MAPE={overall['mape']:.2f}%  (n={overall['n']})"
    )
    for group in ("koppen", "soil"):
        for label, m in sorted(
            test_results.get(group, {}).items(), key=lambda kv: -kv[1]["mae"]
        ):
            print(
                f"  {group:6s} {label:12s} MAE={m['mae']:.4f}  "
                f"MAPE={m['mape']:.2f}%  (n={m['n']})"
            )

    if unseen_loader is not None:
        unseen_results, _, _ = evaluate(unseen_loader, student, device, unseen_cat, layout)
        report["unseen"] = unseen_results
        print(f"Unseen: {unseen_results['overall']}")

    path = results_dir / f"{args.tag}_results.json"
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
