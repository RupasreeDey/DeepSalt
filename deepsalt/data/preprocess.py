"""Build the on-disk training tensors consumed by every training script.

WHAT THIS FILE REPLACES
-----------------------
The original repository had no script that wrote
``preprocessed_data_improved_split_CA_CO_koppen_soiltype/``. Every training
script read that directory; nothing created it. This module is that missing
step, reconstructed from the loading code plus the inline preprocessing in
``enmap_anc_spatial_split_train.py``.

WHAT YOU MUST VERIFY
--------------------
Column names, the ancillary feature list, the EC unit conversion, the holdout
regions, and the Köppen/soil source columns are all CONFIGURATION, not
hardcoded guesses -- see ``configs/default.yaml``. The defaults come from what
was observable in the original scripts. The Köppen and soil-type columns were
NOT observable anywhere (the strings "koppen" and "soil_" appear in no source
file, only in the names of columns the loader expected), so those defaults are
placeholders. Set them to your real column names before running, or leave them
null to build the dataset without those covariates.

FIXES APPLIED
-------------
* Scalers are fit on TRAIN ONLY and persisted, so the autoencoder, the
  student, and evaluation all apply the identical transform. The original fit
  a global MinMaxScaler in ``enmap_encoder.py``, a train-only per-band min/max
  in the spatial-split script, and a single scalar min/max over the whole
  matrix in ``evaluate_teacher.py`` -- three different transforms feeding one
  encoder.
* ``.loc`` label indexing throughout. The original mixed
  ``train_test_split(df.index)`` (labels) with ``gdf.iloc[...]`` (positions),
  which scrambles the split whenever any row has been dropped upstream.
* The unseen-region split writes keys ``unseen_X_test`` / ``unseen_y_test``,
  matching what the loader actually reads. The original wrote one naming and
  read another, so the generalization evaluation silently never ran.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PreprocessConfig:
    csv_path: str
    output_dir: str

    band_prefix: str = "Mean_Reflectance_Band_"
    n_bands: int = 224
    excluded_bands: tuple[int, ...] = tuple(range(130, 136))

    ancillary_columns: tuple[str, ...] = (
        "chorizon.sandtotal_r",
        "chorizon.claytotal_r",
        "tm_min",
        "tm_max",
        "tm_avg",
        "pr_min",
        "pr_max",
        "pr_avg",
    )

    target_column: str = "chorizon.ec_r"
    target_scale: float = 0.55  # saturation-paste conversion applied in the original
    latitude_column: str = "latitude"
    longitude_column: str = "longitude"

    # Categorical covariates. One-hot encoded and appended, then removed again
    # at training time -- they exist so evaluation can stratify by them.
    # SET THESE. Defaults are placeholders; see module docstring.
    koppen_column: str | None = None
    soil_type_column: str | None = None

    # Geographic holdout. Rows whose `region_column` value is in
    # `holdout_regions` become the unseen test set and never enter train/val.
    region_column: str | None = None
    holdout_regions: tuple[str, ...] = ()

    n_spatial_clusters: int = 3
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    random_seed: int = 42

    drop_outliers_iqr: bool = False
    iqr_multiplier: float = 1.5

    feature_names: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #


def band_columns(cfg: PreprocessConfig) -> list[str]:
    excluded = set(cfg.excluded_bands)
    return [
        f"{cfg.band_prefix}{i}"
        for i in range(1, cfg.n_bands + 1)
        if i not in excluded
    ]


def _require_columns(df: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        head = ", ".join(missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise KeyError(
            f"{what}: {len(missing)} column(s) absent from the CSV: {head}{more}. "
            "Check the column naming in configs/default.yaml -- the original "
            "scripts used both 'Mean_Band_{i}' and 'Mean_Reflectance_Band_{i}' "
            "against different CSVs."
        )


def _remove_outliers_iqr(
    df: pd.DataFrame, column: str, multiplier: float
) -> pd.DataFrame:
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
    kept = df[(df[column] >= lo) & (df[column] <= hi)]
    print(f"  IQR filter on {column}: {len(df)} -> {len(kept)} rows [{lo:.3f}, {hi:.3f}]")
    return kept


def _one_hot(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    values = df[column].astype("string").fillna("unknown")
    dummies = pd.get_dummies(values, prefix=prefix, dtype=float)
    dummies.index = df.index
    return dummies


def spatial_cluster_split(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Stratify a train/val/test split across KMeans clusters of coordinates.

    Clustering on coordinates then splitting WITHIN each cluster keeps all
    three sets geographically comparable. It does not produce spatially
    disjoint sets -- neighbouring points can land in different sets, so this
    split does not measure spatial generalization. That is what the
    `holdout_regions` unseen set is for; describe the two separately.
    """
    coords = df[[cfg.latitude_column, cfg.longitude_column]].to_numpy(dtype=float)
    n_clusters = min(cfg.n_spatial_clusters, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg.random_seed, n_init=10)
    labels = pd.Series(kmeans.fit_predict(coords), index=df.index, name="cluster")

    train_idx, val_idx, test_idx = [], [], []
    holdout = cfg.val_fraction + cfg.test_fraction

    for cluster_id in range(n_clusters):
        members = labels.index[labels == cluster_id]
        if len(members) < 3:
            train_idx.extend(members)
            continue
        train, rest = train_test_split(
            members, test_size=holdout, random_state=cfg.random_seed
        )
        val, test = train_test_split(
            rest,
            test_size=cfg.test_fraction / holdout,
            random_state=cfg.random_seed,
        )
        train_idx.extend(train)
        val_idx.extend(val)
        test_idx.extend(test)

    return pd.Index(train_idx), pd.Index(val_idx), pd.Index(test_idx)


def build_dataset(cfg: PreprocessConfig) -> dict:
    """Read the CSV, split it, scale it, and write everything to disk."""
    out = Path(cfg.output_dir)
    (out / "train_data").mkdir(parents=True, exist_ok=True)
    (out / "scalers").mkdir(parents=True, exist_ok=True)
    (out / "unseen_test_data").mkdir(parents=True, exist_ok=True)

    print(f"Reading {cfg.csv_path}")
    df = pd.read_csv(cfg.csv_path, low_memory=False)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    bands = band_columns(cfg)
    _require_columns(df, bands, "reflectance bands")
    _require_columns(df, list(cfg.ancillary_columns), "ancillary features")
    _require_columns(
        df,
        [cfg.target_column, cfg.latitude_column, cfg.longitude_column],
        "target/coordinates",
    )

    for col in (*cfg.ancillary_columns, cfg.target_column):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[cfg.target_column] = df[cfg.target_column] * cfg.target_scale

    essential = [*bands, *cfg.ancillary_columns, cfg.target_column,
                 cfg.latitude_column, cfg.longitude_column]
    before = len(df)
    df = df.dropna(subset=essential)
    print(f"  dropped {before - len(df)} rows with missing essential values")

    if cfg.drop_outliers_iqr:
        df = _remove_outliers_iqr(df, cfg.target_column, cfg.iqr_multiplier)

    # ---- categorical covariates ------------------------------------------ #
    categorical_frames = []
    if cfg.koppen_column:
        _require_columns(df, [cfg.koppen_column], "koppen column")
        categorical_frames.append(_one_hot(df, cfg.koppen_column, "koppen"))
    if cfg.soil_type_column:
        _require_columns(df, [cfg.soil_type_column], "soil type column")
        categorical_frames.append(_one_hot(df, cfg.soil_type_column, "soil"))

    # ---- geographic holdout ----------------------------------------------- #
    if cfg.region_column and cfg.holdout_regions:
        _require_columns(df, [cfg.region_column], "region column")
        is_holdout = df[cfg.region_column].astype("string").isin(cfg.holdout_regions)
        unseen_df, modelling_df = df[is_holdout], df[~is_holdout]
        print(
            f"  unseen holdout on {cfg.region_column} in {cfg.holdout_regions}: "
            f"{len(unseen_df)} rows held out, {len(modelling_df)} for modelling"
        )
        if unseen_df.empty:
            raise ValueError(
                f"holdout_regions={cfg.holdout_regions} matched no rows. "
                f"Observed values: {sorted(df[cfg.region_column].dropna().unique())[:20]}"
            )
    else:
        unseen_df = df.iloc[0:0]
        modelling_df = df
        print("  no geographic holdout configured -- unseen test set will be empty")

    train_idx, val_idx, test_idx = spatial_cluster_split(modelling_df, cfg)
    print(
        f"  split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"unseen={len(unseen_df)}"
    )

    # ---- scaling (fit on train only) -------------------------------------- #
    band_scaler = MinMaxScaler().fit(modelling_df.loc[train_idx, bands].to_numpy(float))
    anc_scaler = MinMaxScaler().fit(
        modelling_df.loc[train_idx, list(cfg.ancillary_columns)].to_numpy(float)
    )
    joblib.dump(band_scaler, out / "scalers" / "band_scaler.pkl")
    joblib.dump(anc_scaler, out / "scalers" / "ancillary_scaler.pkl")

    cat_all = pd.concat(categorical_frames, axis=1) if categorical_frames else None
    feature_names = [f"spectral_{i}" for i in range(64)]
    feature_names += list(cfg.ancillary_columns)
    if cat_all is not None:
        feature_names += list(cat_all.columns)

    def assemble(frame: pd.DataFrame, index: pd.Index | None = None):
        sub = frame if index is None else frame.loc[index]
        if sub.empty:
            n_cat = cat_all.shape[1] if cat_all is not None else 0
            width = len(bands) + len(cfg.ancillary_columns) + n_cat
            return np.empty((0, width)), np.empty(0)
        reflectance = band_scaler.transform(sub[bands].to_numpy(float))
        ancillary = anc_scaler.transform(
            sub[list(cfg.ancillary_columns)].to_numpy(float)
        )
        blocks = [reflectance, ancillary]
        if cat_all is not None:
            blocks.append(cat_all.loc[sub.index].to_numpy(float))
        return np.hstack(blocks), sub[cfg.target_column].to_numpy(float)

    splits = {
        "train": assemble(modelling_df, train_idx),
        "val": assemble(modelling_df, val_idx),
        "test": assemble(modelling_df, test_idx),
    }
    unseen_X, unseen_y = assemble(unseen_df)

    # Reflectance stays UNEMBEDDED here. The autoencoder is trained on the
    # train split of this file (train_ssae.py), then train_student.py replaces
    # the reflectance block with its 64-d code. Embedding here would leak the
    # encoder's training data into val/test.
    for name, (X, y) in splits.items():
        np.save(out / "train_data" / f"X_{name}_reflectance.npy", X)
        np.save(out / "train_data" / f"y_{name}.npy", y)
    np.save(out / "unseen_test_data" / "X_test_reflectance.npy", unseen_X)
    np.save(out / "unseen_test_data" / "y_test.npy", unseen_y)

    metadata = {
        "n_bands": len(bands),
        "band_columns": bands,
        "ancillary_columns": list(cfg.ancillary_columns),
        "categorical_columns": list(cat_all.columns) if cat_all is not None else [],
        "feature_names_after_embedding": feature_names,
        "target_column": cfg.target_column,
        "target_scale": cfg.target_scale,
        "target_range_observed": [
            float(splits["train"][1].min()) if splits["train"][1].size else None,
            float(splits["train"][1].max()) if splits["train"][1].size else None,
        ],
        "counts": {k: int(len(v[1])) for k, v in splits.items()}
        | {"unseen": int(len(unseen_y))},
        "holdout_regions": list(cfg.holdout_regions),
        "random_seed": cfg.random_seed,
    }
    joblib.dump(metadata, out / "train_data" / "metadata.pkl")
    with open(out / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Wrote dataset to {out}")
    print(
        "  observed train target range: "
        f"{metadata['target_range_observed']} {cfg.target_column} units -- "
        "set model.output_range in the config to bracket this."
    )
    return metadata


def load_dataset(output_dir: str | Path) -> dict:
    """Read back what ``build_dataset`` wrote."""
    out = Path(output_dir)
    if not out.exists():
        raise FileNotFoundError(
            f"{out} does not exist. Run `python -m deepsalt.data.preprocess` first."
        )

    data: dict = {}
    for pkl in (out / "scalers").glob("*.pkl"):
        data[pkl.stem] = joblib.load(pkl)

    for npy in (out / "train_data").glob("*.npy"):
        data[npy.stem] = np.load(npy, allow_pickle=True)

    meta_path = out / "train_data" / "metadata.pkl"
    if meta_path.exists():
        data.update(joblib.load(meta_path))

    unseen = out / "unseen_test_data"
    if unseen.exists():
        for npy in unseen.glob("*.npy"):
            # key naming matches what the loaders read -- see module docstring
            data[f"unseen_{npy.stem}"] = np.load(npy, allow_pickle=True)

    return data


def main() -> None:
    import argparse

    from ..utils import load_config

    parser = argparse.ArgumentParser(description="Build the DEEPSALT dataset.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    section = dict(cfg_dict["data"])
    section["excluded_bands"] = tuple(section.get("excluded_bands", range(130, 136)))
    section["ancillary_columns"] = tuple(section["ancillary_columns"])
    section["holdout_regions"] = tuple(section.get("holdout_regions") or ())
    build_dataset(PreprocessConfig(**section))


if __name__ == "__main__":
    main()
