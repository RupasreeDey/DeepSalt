# Original scripts

The 21 research scripts as originally written, unmodified, kept for
provenance and for porting the baselines that have not yet been moved into
the package.

They are NOT importable, NOT part of the pipeline, and several contain the
bugs documented in `../PROVENANCE.md`. Do not run them expecting the
package's behaviour.

Mapping to the refactored package:

| Original | Refactored to |
|---|---|
| `enmap_encoder.py`, `ftir_encoder.py` | `deepsalt/train/train_ssae.py` |
| `align_enmap_ftir.py` | `deepsalt/train/train_alignment.py`, `deepsalt/data/pairing.py` |
| `ftir_train.py`, `ftir_train_clip_preds.py` | `deepsalt/train/train_teacher.py` |
| `enmap_anc_kd_transformer_load_data.py`, `enmap_anc_spatial_split_train.py` | `deepsalt/train/train_student.py` |
| `enmap_anc_spatial_split_train.py` (inline preprocessing) | `deepsalt/data/preprocess.py` |
| `enmap_train_only_enmap*.py` | `loss.feature_weight: 0.0` ablation |
| `enmap_train_feature_KD_{v3,attention,HG,...}.py` | not ported — loss variants, see PROVENANCE §4 |
| `enmap_anc_kd_resnet_load_data.py` | not ported — ResNet baseline |
| `evaluate_teacher.py` | superseded by teacher metrics in the checkpoint sidecar |
