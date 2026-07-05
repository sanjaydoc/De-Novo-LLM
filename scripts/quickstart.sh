#!/usr/bin/env bash
# Quickstart: install the package and run the CPU smoke test end-to-end.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Installing (editable)"
pip install -e . >/dev/null

echo "==> Running unit tests (no ML deps needed)"
python -m pytest -q || echo "(install pytest with: pip install pytest)"

echo "==> Smoke test: prepare -> train -> generate -> evaluate on CPU"
echo "    (downloads a tiny random model; takes ~1 minute)"
denovo pipeline -c configs/smoke.yaml

echo "==> Done. Next: pick a real config from configs/ and see docs/MODELS.md"
