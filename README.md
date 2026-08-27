# DEEPSALT

Deep spectral transfer from laboratory spectroscopy to hyperspectral satellite
imagery for soil salinity estimation.

Code accompanying **"DEEPSALT: Bridging Laboratory and Satellite Spectra
through Domain Adaptation and Knowledge Distillation for Large-Scale Soil
Salinity Estimation"**, IEEE International Conference on Big Data (BigData)
2025, pp. 1913–1923.
[[IEEE]](https://ieeexplore.ieee.org/document/11401634)
[[arXiv]](https://arxiv.org/abs/2510.23124)

Rupasree Dey, Abdul Matin, Everett Lewark, Tanjim Bin Faruk, Andrei Bachinin,
Sam Leuthold, M. Francesca Cotrufo, Shrideep Pallickara, Sangmi Lee Pallickara
— Colorado State University.

Soil salinization limits plants' ability to absorb water and reduces crop
productivity. It also alters a soil's spectral response, which makes it
observable remotely. Laboratory FTIR spectroscopy measures that response
precisely but depends on in-situ sampling, which does not scale to regional or
global monitoring. Hyperspectral satellite imagery covers wide areas but lacks
the spectral detail of laboratory instruments.

DEEPSALT bridges the two. A transformer regressor is trained on FTIR spectra
and distilled into a satellite-side model through a Spectral Adaptation Unit
that aligns the laboratory and satellite spectral domains, enabling large-scale
salinity estimation without extensive ground sampling.

---

## Project status

This repository is under active development. The core pipeline runs end to end,
and the following are still being finalized:

- **Dataset construction.** The scripts that assemble
  `multiplied_enmap_ssurgo_map.csv` and `ftir_reflectance_with_salinity.csv`
  from raw EnMAP scenes and SSURGO records are not yet included. See
  [Data](#data).
- **Configuration defaults.** Several fields in `configs/default.yaml` are
  marked `REVIEW` and need to be set for your data layout before running. See
  [Configuration](#configuration).
- **Baseline models.** The ResNet student and several alternative distillation
  objectives are kept in `legacy/` and have not yet been ported into the
  package.
- **Released artifacts.** Pretrained checkpoints and a reproduction of the
  published results tables are not yet published here.

Interfaces may change before the first tagged release.

---

## Method

```
FTIR spectra ──► [1] FTIR SSAE ──────┐
                                     ├──► [3] Spectral Adaptation Unit ──┐
EnMAP reflectance ──► [2] EnMAP SSAE ┘                                   │
                                                                         ▼
                        [4] Teacher (FTIR latents → dS/m) ──────► [5] Student
                                                                (distillation)
```

Sparse stacked autoencoders compress both domains to a shared 64-dimensional
code. The Spectral Adaptation Unit aligns those codes using geographically
paired samples, so a satellite spectrum and a laboratory spectrum from the same
location occupy comparable positions in latent space. A transformer teacher is
trained on FTIR latents, then distilled layer-wise into a student that consumes
EnMAP latents plus ancillary climate and soil-texture covariates.

---

## Installation

```bash
git clone https://github.com/RupasreeDey/DeepSalt.git
cd DeepSalt
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python ≥3.10 and PyTorch ≥2.1.

---

## Usage

Run the full pipeline:

```bash
./scripts/run_pipeline.sh configs/default.yaml
```

Or stage by stage:

| Stage | Command | Produces |
|---|---|---|
| 0 | `python -m deepsalt.data.preprocess` | `data/processed/` |
| 1 | `python -m deepsalt.train.train_ssae --domain ftir` | `ftir_ssae.pth` + scaler |
| 2 | `python -m deepsalt.train.train_ssae --domain enmap` | `enmap_ssae.pth` |
| 3 | `python -m deepsalt.train.train_alignment` | `*_ssae_aligned.pth` |
| 4 | `python -m deepsalt.train.train_teacher --aligned` | `teacher.pth` |
| 5 | `python -m deepsalt.train.train_student` | `student.pth`, `results/*.json` |

Every stage accepts `--config`.

### Ablations

Each ablation is a configuration override on the same code path:

```bash
./scripts/run_ablations.sh configs/default.yaml
```

| Variant | Override |
|---|---|
| Full DEEPSALT | — |
| Without domain adaptation | `student.use_aligned_encoder: false` |
| Fixed projection | `sau.projection_mode: truncate` |
| Without feature distillation | `loss.feature_weight: 0.0` |
| EnMAP only | both of the above |

---

## Data

Two input tables are required and are not distributed with this repository:

- **`multiplied_enmap_ssurgo_map.csv`** — EnMAP reflectance averaged per SSURGO
  map unit (224 bands, with 130–135 excluded for water-vapor absorption),
  joined to SSURGO `chorizon` properties, PRISM climate summaries, and
  coordinates.
- **`ftir_reflectance_with_salinity.csv`** — laboratory MIR spectra (1765
  channels) with measured electrical conductivity and coordinates.

Scripts to build these from source products are being prepared for release.
Contact the authors in the meantime.

Redistribution of derived products is subject to the terms of the underlying
EnMAP, SSURGO, and spectral library data.

---

## Configuration

All settings live in `configs/default.yaml`. Fields marked `REVIEW` depend on
your column naming and must be set before the pipeline will run:

| Field | What to set |
|---|---|
| `data.csv_path`, `ftir.csv_path` | Paths to the two input tables |
| `data.band_prefix` | Reflectance column naming in your CSV |
| `data.koppen_column`, `data.soil_type_column` | Source columns for the climate-zone and soil-type covariates |
| `data.region_column`, `data.holdout_regions` | Which regions form the unseen-region test set |
| `ftir.target_column` | The electrical-conductivity column in the FTIR table |
| `alignment.max_distance_km` | Maximum FTIR–EnMAP pairing distance |

The Köppen, soil-type, and region fields may be left `null`; the pipeline then
builds a dataset without those covariates, losing the stratified result tables
and the unseen-region evaluation.

---

## Repository layout

```
deepsalt/
├── models/
│   ├── autoencoder.py      Sparse stacked autoencoder
│   ├── sau.py              Spectral Adaptation Unit
│   └── transformer.py      Teacher and student regressors
├── data/
│   ├── preprocess.py       CSV → tensors, splits, scalers
│   ├── pairing.py          Geographic FTIR–EnMAP matching
│   └── datasets.py         Loaders and class balancing
├── train/
│   ├── train_ssae.py       Stages 1–2
│   ├── train_alignment.py  Stage 3
│   ├── train_teacher.py    Stage 4
│   └── train_student.py    Stage 5
├── losses.py               Distillation objectives
└── utils.py                Seeding, checkpoint I/O, metrics

configs/    Experiment configuration
scripts/    Pipeline and ablation runners
tests/      Regression tests
legacy/     Earlier research scripts, retained for reference
```

---

## Implementation notes

**Checkpoint metadata.** Each `.pth` is written with a `.pth.json` sidecar
recording the output range, source encoder, and seed. These are verified on
load, so checkpoints are always paired with a matching model head.

**Scaling.** One scaler per domain, fit on the training split and persisted, so
every stage applies the same transform.

**Evaluation splits.** The KMeans cluster split stratifies train/validation/test
across the study area but does not make them spatially disjoint; it measures
in-distribution performance. Geographic generalization is measured separately
on the held-out regions configured in `data.holdout_regions`.

**Reproducibility.** `set_seeds()` covers Python, NumPy, and PyTorch and sets
`cudnn.deterministic`. Stochastic sampling uses explicit generators. Exact GPU
reproducibility still depends on driver and cuDNN versions; record
`torch.__version__` and `torch.version.cuda` alongside results.

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{dey2025deepsalt,
  title     = {{DEEPSALT}: Bridging Laboratory and Satellite Spectra through
               Domain Adaptation and Knowledge Distillation for Large-Scale
               Soil Salinity Estimation},
  author    = {Dey, Rupasree and Matin, Abdul and Lewark, Everett and
               Bin Faruk, Tanjim and Bachinin, Andrei and Leuthold, Sam and
               Cotrufo, M. Francesca and Pallickara, Shrideep and
               Pallickara, Sangmi Lee},
  booktitle = {2025 IEEE International Conference on Big Data (BigData)},
  pages     = {1913--1923},
  year      = {2025},
  publisher = {IEEE},
  address   = {Macau, China},
  doi       = {TODO},
  url       = {https://ieeexplore.ieee.org/document/11401634}
}
```

A preprint is available on arXiv:

```bibtex
@misc{dey2025deepsalt-arxiv,
  title         = {{DeepSalt}: Bridging Laboratory and Satellite Spectra through
                   Domain Adaptation and Knowledge Distillation for Large-Scale
                   Soil Salinity Estimation},
  author        = {Dey, Rupasree and Matin, Abdul and Lewark, Everett and
                   Bin Faruk, Tanjim and Bachinin, Andrei and Leuthold, Sam and
                   Cotrufo, M. Francesca and Pallickara, Shrideep and
                   Pallickara, Sangmi Lee},
  year          = {2025},
  eprint        = {2510.23124},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2510.23124}
}
```

---

## License

Released under the MIT License. See [LICENSE](LICENSE).

The license covers the source code in this repository. Use of the underlying
EnMAP, SSURGO, and spectral library data is governed by the terms of those
providers.
