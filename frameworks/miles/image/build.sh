#!/usr/bin/env bash
# Build the miles image this adapter is pinned to (linux/amd64).
#
#   bash frameworks/miles/image/build.sh <registry>/<repository>:<tag>
#
# The miles commit comes from rft_miles/miles.lock.json; the build fails unless the reference image
# was built from it, or if the image is over MAX_IMAGE_GB (decimal GB, unpacked; default 19.5, under
# AI Runtime's 20 GB limit). PIP_INDEX_URL, if set, reaches the build as a BuildKit secret (never a
# build arg, so it is not recorded in the image history). Extra docker flags go in BUILD_ARGS.
set -euo pipefail
TAG="${1:?usage: build.sh <registry>/<repository>:<tag>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['commit'])" "${HERE}/../rft_miles/miles.lock.json")"
SECRET=()
[ -n "${PIP_INDEX_URL:-}" ] && SECRET=(--secret id=pip_index,env=PIP_INDEX_URL)
# shellcheck disable=SC2086
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 ${SECRET[@]+"${SECRET[@]}"} --build-arg MILES_COMMIT="${COMMIT}" ${BUILD_ARGS:-} \
  -f "${HERE}/Dockerfile" -t "${TAG}" "${HERE}/.."
SIZE="$(docker image inspect "${TAG}" --format '{{.Size}}')"
python3 -c "import sys; size, limit = int(sys.argv[1]), float(sys.argv[2]) * 1e9
print(f'{sys.argv[3]}: {size / 1e9:.2f} GB unpacked (limit {limit / 1e9:g} GB)')
sys.exit(size > limit)" "${SIZE}" "${MAX_IMAGE_GB:-19.5}" "${TAG}" || { echo "the image is over the size limit" >&2; exit 1; }
echo "built ${TAG}; push it, pin its digest in site.env (RFT_MILES_IMAGE), and register it with AI Runtime"
