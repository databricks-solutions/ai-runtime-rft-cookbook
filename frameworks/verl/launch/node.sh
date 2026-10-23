#!/usr/bin/env bash
# Entry point of every verl job: install the cookbook into the image's interpreter, then run
# one stage of the run the job's env_variables name (RFT_RECIPE, RFT_RUN_ID, RFT_VOLUME_ROOT).
#
#   bash frameworks/verl/launch/node.sh <stage>
#
# The packages are installed with --no-deps --no-build-isolation: everything they import
# (verl, torch, vLLM, PyYAML, ...) comes from the image, and nothing is fetched at job start.
# Ray workers and agent-loop processes use the same interpreter, so they import the same code.
set -euo pipefail
STAGE="${1:?stage}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# A job's command can start with a PATH that lacks the image's venv (where python, ray and verl live).
command -v ray >/dev/null 2>&1 || export PATH="/opt/venv/bin:${PATH}"
PY="${RFT_PYTHON:-python3}"
export RFT_RECIPES="${ROOT}/recipes"   # the snapshot's recipes; the packages are installed elsewhere

"${PY}" -m pip install --quiet --no-deps --no-build-isolation "${ROOT}" "${ROOT}/frameworks/verl"
cd /tmp   # import the installed packages, not the snapshot's source tree
"${PY}" -c "import rft_cookbook, rft_verl, verl; print('rft_cookbook', rft_cookbook.__file__); print('verl', verl.__version__)"
exec "${PY}" -m rft_verl.launch "${STAGE}"
