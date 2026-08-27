#!/usr/bin/env bash
# Every ablation row from one code path, varying only config overrides.
# Assumes scripts/run_pipeline.sh has already produced the shared artefacts.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
mkdir -p configs/generated

variant () {
  local tag="$1"; shift
  local out="configs/generated/${tag}.yaml"
  cp "${CONFIG}" "${out}"
  python - "$out" "$@" <<'PY'
import sys, yaml
path, *pairs = sys.argv[1:]
cfg = yaml.safe_load(open(path))
for pair in pairs:
    dotted, raw = pair.split("=", 1)
    node = cfg
    *parents, leaf = dotted.split(".")
    for p in parents:
        node = node[p]
    node[leaf] = yaml.safe_load(raw)
yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
PY
  echo "--- ${tag} ---"
  python -m deepsalt.train.train_student --config "${out}" --tag "${tag}"
}

variant full
variant no_alignment      student.use_aligned_encoder=false
variant legacy_truncation sau.projection_mode=truncate
variant no_feature_kd     loss.feature_weight=0.0
variant enmap_only        loss.feature_weight=0.0 student.use_aligned_encoder=false

echo "Ablation results in results/"
