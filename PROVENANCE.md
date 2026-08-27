# Provenance and change log

This repository is a refactor of 21 original research scripts. This document
records what changed, what was reconstructed, and which changes affect numbers
that may already be in the paper.

**Bottom line:** items in §1 and §2 can change reported results. Re-run before
submitting.

---

## 1. The alignment step was never connected

`align_enmap_ftir.py` trained the cross-domain alignment and saved
`enmap_encoder_ssae_cosine.pth` and `ftir_encoder_ssae_cosine.pth`.

Across all 21 original scripts, **neither file had a single consumer.** Every
downstream script loaded `enmap_ssae.pth`, produced by `enmap_encoder.py` from
reconstruction and sparsity alone, with no FTIR term.

Consequence: on the evidence of the code, results attributed to the Spectral
Adaptation Unit were produced *without* alignment. The KD scripts fed EnMAP
latents into an FTIR-trained teacher across two latent spaces with nothing
relating them.

**Fixed:** `train_alignment.py` writes `enmap_ssae_aligned.pth`, which
`train_student.py` loads by default. Each checkpoint records `aligned:
true|false`, and the student prints which encoder it used. The
no-adaptation ablation is `student.use_aligned_encoder: false`.

**Action:** re-run the SAU ablation. If the reported gain came from an
unaligned encoder, it was measuring something else.

---

## 2. Bugs that affect reported metrics

### 2.1 The unseen-region evaluation never ran

`load_preprocessed_data` wrote keys `unseen_X_test` / `unseen_y_test`;
`create_dataloaders_from_saved` read `X_unseen_test` / `y_unseen_test`. Those
return `None`, so `unseen_test_loader` was `None` and `main()` printed *"No
unseen test data available."*

Any claim that the model "generalized to unseen geographic regions" needs to be
traced to a run that actually produced numbers.

**Fixed:** one naming, defined in `preprocess.py` and read in
`train_student.py`. The integration test confirms the unseen loader is
populated and evaluated.

### 2.2 Per-category tables scored every sample with the last sample's prediction

In `test_epoch_by_categories`:

```python
for i in range(len(output)):
    pred = all_predictions[-1]   # ← same value for every i
    true = all_targets[-1]
```

`all_predictions[-1]` is the last element of the accumulated list, not sample
`i`. Every per-Köppen and per-soil MAE and MAPE was computed from one
prediction per batch, replicated `batch_size` times. Overall metrics were
unaffected — they used the full arrays.

**Fixed:** `evaluate()` in `train_student.py` indexes explicitly and asserts
that the categorical block and prediction array have equal length.

### 2.3 A distillation term was identically zero

```python
kl_loss = F.kl_div(F.log_softmax(student_output, dim=-1),
                   F.softmax(teacher_output, dim=-1), reduction='batchmean')
```

`student_output` has shape `(B, 1)`. A softmax over a length-1 axis is
identically 1, its log identically 0, so this returned exactly 0.0 every batch.
The `0.1 *` weight was decorative.

**Fixed:** removed. For scalar regression there is no class distribution to
match. `loss.response_weight` now performs response distillation as a direct
regression of student onto teacher prediction; it defaults to 0.0, matching the
original's effective behaviour.

### 2.4 Geographic pairing used the wrong units

`BallTree(metric="haversine")` requires **radians**. The original passed
degrees and compared distances against `max_distance=0.01`. Both the neighbour
ranking and the radius cut operated on a meaningless scale, so the FTIR–EnMAP
pair set — the training data for alignment — was not what it appeared to be.

**Fixed:** `np.radians` conversion, threshold in kilometres. Verified: a 0.013°
longitude offset at latitude 40.6 now resolves to 1.098 km (analytically
1.10 km).

Matching is also mutual-nearest-neighbour by default. The original claimed each
EnMAP pixel for the first FTIR site that reached it in row order, silently
dropping later sites, making the pair set dependent on CSV ordering.
`match_strategy: greedy` restores the old behaviour.

### 2.5 The deepest student layer was unsupervised

`layer_weights = [1.35, 1.05, 0.65]` with 4 student layers, and
`num_layers = min(len(student), len(teacher)) = min(4, 3) = 3`. The fourth
student layer received no distillation signal, and `zip()` silently dropped it.

**Fixed:** `LayerWiseDistillationLoss` raises if the layer counts disagree.
`student.teacher_layer_map` states the pairing explicitly (default `[0,1,2,2]`).

### 2.6 Three output parameterizations shared checkpoint files

| Script | Head | Range |
|---|---|---|
| `ftir_train.py` | ReLU on MinMax-scaled targets | ~[0, 1] scaled |
| `ftir_train_clip_preds.py` | `0.05 + 1.05·σ` | [0.05, 1.1] |
| `TransformerModelTeacher` | `0.05 + 76.95·σ` | [0.05, 77] dS/m |

A `state_dict` does not record which. Weights trained to emit [0.05, 1.1] emit
values ~70× too large when loaded into the [0.05, 77] head, with no error.

**Fixed:** the range is a constructor argument, written to the sidecar, and
verified on load. Verified: loading the teacher into a mismatched head raises.

### 2.7 Label-versus-position indexing in the spatial split

`train_test_split(cluster_df.index, ...)` returns index *labels*; the result
was passed to `gdf.iloc[...]`, which treats them as *positions*. Harmless only
if no row was ever dropped upstream — and rows were dropped for NaN
coordinates.

**Fixed:** `.loc` throughout `preprocess.py`.

### 2.8 The FTIR teacher bypassed its encoder's scaler

`ftir_encoder.py` fit a MinMaxScaler and trained the SSAE on scaled spectra
with a Sigmoid decoder. `ftir_train.py` then fed the encoder **raw**
`spectral_values` from the OPUS files. The teacher learned on latents the
encoder was never trained to produce.

There were also two FTIR sources — a wide CSV and per-sample OPUS files indexed
through `input.csv` — both 1765-d, with nothing verifying they were the same
samples on the same wavenumber grid in the same order.

**Fixed:** scalers are persisted at stage 1 and loaded by every consumer. One
FTIR source, so spectra and targets cannot drift apart.

### 2.9 Three different EnMAP normalizations

`enmap_encoder.py` fit a global MinMaxScaler on
`valid_enmap_reflectance_with_ec_mask.csv` (`Mean_Band_{i}` columns);
`enmap_anc_spatial_split_train.py` used train-split per-band min/max on
`multiplied_enmap_ssurgo_map.csv` (`Mean_Reflectance_Band_{i}`);
`evaluate_teacher.py` used a single scalar min/max over the entire matrix. One
encoder, three transforms, two CSVs, two column namings.

**Fixed:** one scaler, fit on train, persisted, applied everywhere. Both
namings are configurable and a missing-column error names the alternative.

### 2.10 Non-reproducible zero undersampling

Bare `torch.randperm` without a generator meant the retained zero subset varied
between runs despite the global seed.

**Fixed:** explicit seeded generator in `undersample_zeros`.

---

## 3. Components that were missing entirely

| Artefact | Original status | Now |
|---|---|---|
| `preprocessed_data_*/` | Read by every training script, written by none | `deepsalt/data/preprocess.py` |
| `ftir_salinity_model_ssae_clipped_final.pth` (`_v3`, `_resnet`) | Loaded by 8 scripts, saved by none | `deepsalt/train/train_teacher.py` |
| Spectral Adaptation Unit | No module by that name | `deepsalt/models/sau.py` (reconstruction) |
| Köppen / soil one-hots | No source file contained these strings | Config-driven in `preprocess.py` |
| CA/CO regional holdout | Implied by a directory name only | `region_column` + `holdout_regions` |

---

## 4. Still outside this repository

**Source data construction.** Nothing builds
`multiplied_enmap_ssurgo_map.csv` or `ftir_reflectance_with_salinity.csv`.
EnMAP acquisition, cloud masking, per-polygon band averaging, and the SSURGO
`chorizon` join are undocumented. Describe this in the paper even if the code
stays private — a reader cannot reconstruct the dataset otherwise.

**Baseline architectures.** `enmap_anc_kd_resnet_load_data.py` (ResNet
student), the attention-transfer and Euclidean feature losses, and the
gradient-reversal domain discriminator are preserved in `legacy/` but not
ported. Port them if the paper reports them.

**A note on the domain discriminator.** If you port it, it needs rework, not
translation. `domain_loss = BCE(domain_preds, ones_like(domain_preds))` labels
every sample as the same domain, so there is no second class to discriminate
against; and `optimizer_domain.step()` runs on gradient-reversed gradients, so
the discriminator ascends its own loss. As written it is not adversarial domain
adaptation. The most recent original script dropped it — if that is the
reported configuration, the paper should not credit adversarial DA.

---

## 5. Pre-submission checklist

- [ ] All `REVIEW` fields in `configs/default.yaml` are set.
- [ ] Column naming confirmed (`Mean_Band_` vs `Mean_Reflectance_Band_`).
- [ ] `ftir.target_column` and `target_scale` put FTIR and EnMAP targets in the
      same units. The EnMAP side applies `× 0.55`; confirm whether FTIR needs it.
- [ ] `model.output_range` brackets the observed target range (`preprocess.py`
      prints it).
- [ ] `alignment.max_distance_km` chosen from EnMAP GSD and stated in the paper.
- [ ] Pair count and distance distribution reported (`train_alignment.py`
      prints them).
- [ ] SAU ablation re-run with a genuinely aligned encoder (§1).
- [ ] Per-category tables regenerated (§2.2 — the old ones are invalid).
- [ ] Unseen-region numbers come from a run where the loader was populated (§2.1).
- [ ] Teacher checkpoint's recorded output range matches the student's (§2.6).
- [ ] Paper's SAU description matches `models/sau.py`, or the module is replaced.
- [ ] Cluster split described as stratified, not as spatial generalization.
- [ ] `LICENSE` added; data redistribution terms checked.
- [ ] `torch.__version__` and CUDA version recorded with results.
