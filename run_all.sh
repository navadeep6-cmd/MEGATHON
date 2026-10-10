#!/usr/bin/env bash
# End-to-end pipeline. Usage:
#   ./run_all.sh            full run (about 25k images; roughly 1-2 h on a laptop, mostly crawling/downloading)
#   ./run_all.sh --quick    smoke test (10 seeds x 150 files)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
FETCH_ARGS=()
if [[ "${1:-}" == "--quick" ]]; then
  FETCH_ARGS=(--max-seeds 10 --max-files-per-seed 150)
fi

if [[ -z "${COMMONS_CONTACT:-}" ]]; then
  echo "Please export COMMONS_CONTACT=<your e-mail or project URL> (Wikimedia User-Agent policy)." >&2
  exit 1
fi

$PY -m unittest discover -s tests
$PY scripts/s01_fetch_commons.py "${FETCH_ARGS[@]+"${FETCH_ARGS[@]}"}"
$PY scripts/s02_download_images.py
$PY scripts/s03_build_dataset.py
$PY scripts/s04_embed.py
$PY scripts/s05_train_head.py
$PY scripts/s06_build_index.py
$PY scripts/s07_fit_fusion.py
$PY scripts/s13_apply_feedback.py   # re-applies any logged editor feedback to the rebuilt index
$PY scripts/s08_evaluate.py
$PY scripts/s09_robustness.py
$PY scripts/s10_benchmark.py
$PY scripts/s11_report.py
echo "Done. See reports/RESULTS.md. Demo: $PY app/demo.py"
