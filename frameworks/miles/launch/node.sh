#!/usr/bin/env bash
# Entry point of every miles job: fetch the pinned upstream miles source, install it and the
# cookbook into the image's interpreter, then run one stage of the run the job's env_variables
# name (RFT_RECIPE, RFT_RUN_ID, RFT_VOLUME_ROOT).
#
#   bash frameworks/miles/launch/node.sh <stage>
#
# The image carries miles' dependencies (SGLang, Megatron-LM, kernels) but not miles itself:
# upstream miles comes from the commit in rft_miles/miles.lock.json, checked against its sha256, and is
# installed editable because its launchers find train.py and tools/ next to the package. Only the
# stages that run miles install it: the others run the same way for every framework, and the
# evaluation stages run on the verl image (vLLM) with no miles at all.
set -euo pipefail
STAGE="${1:?stage}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
command -v ray >/dev/null 2>&1 || export PATH="/opt/venv/bin:${PATH}"
PY="${RFT_PYTHON:-python3}"
export RFT_RECIPES="${ROOT}/recipes"   # the snapshot's recipes; the packages are installed elsewhere

case "${STAGE}" in
  stage_model|train)
    LOCK="${ROOT}/frameworks/miles/rft_miles/miles.lock.json"
    read -r URL SHA COMMIT < <("${PY}" -c "import json,sys; l=json.load(open(sys.argv[1])); print(l['tarball'], l['tarball_sha256'], l['commit'])" "${LOCK}")
    export RFT_MILES_SRC="/tmp/miles-${COMMIT:0:12}"
    if [ ! -f "${RFT_MILES_SRC}/train.py" ]; then
      curl -fsSL --retry 5 --retry-delay 5 "${URL}" -o /tmp/miles.tgz
      echo "${SHA}  /tmp/miles.tgz" | sha256sum -c -
      mkdir -p "${RFT_MILES_SRC}"
      tar -xzf /tmp/miles.tgz -C "${RFT_MILES_SRC}" --strip-components 1
    fi
    "${PY}" -m pip install --quiet --no-deps --no-build-isolation -e "${RFT_MILES_SRC}"
    export PYTHONPATH="${RFT_MILES_SRC}${PYTHONPATH:+:${PYTHONPATH}}"   # miles_plugins and scripts/ beside miles
    CHECK="import miles, sglang; print('miles', miles.__file__, '${COMMIT}')" ;;
  *) CHECK="pass" ;;
esac
"${PY}" -m pip install --quiet --no-deps --no-build-isolation "${ROOT}" "${ROOT}/frameworks/miles"
cd /tmp   # import the installed packages, not the snapshot's source tree
"${PY}" -c "import rft_cookbook, rft_miles; print('rft_cookbook', rft_cookbook.__file__); ${CHECK}"
exec "${PY}" -m rft_miles.launch "${STAGE}"
