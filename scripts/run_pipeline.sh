#!/usr/bin/env bash
# Full DEEPSALT pipeline. Each stage depends on the artefacts of the previous
# one; running out of order fails loudly rather than silently loading a
# mismatched checkpoint.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
echo "=== DEEPSALT pipeline | config: ${CONFIG} ==="

echo; echo "--- [1/5] Build dataset ---"
python -m deepsalt.data.preprocess --config "${CONFIG}"

echo; echo "--- [2/5] Train FTIR autoencoder ---"
python -m deepsalt.train.train_ssae --config "${CONFIG}" --domain ftir

echo; echo "--- [3/5] Train EnMAP autoencoder ---"
python -m deepsalt.train.train_ssae --config "${CONFIG}" --domain enmap

echo; echo "--- [4/5] Align latent spaces (SAU) ---"
python -m deepsalt.train.train_alignment --config "${CONFIG}"

echo; echo "--- [5/5] Train teacher, then student ---"
python -m deepsalt.train.train_teacher --config "${CONFIG}" --aligned
python -m deepsalt.train.train_student --config "${CONFIG}" --tag deepsalt

echo; echo "=== Done. Results in results/ ==="
