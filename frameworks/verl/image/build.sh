#!/usr/bin/env bash
# Build the verl image this adapter is pinned to (linux/amd64; about 17 GB).
#
#   bash frameworks/verl/image/build.sh <registry>/<repository>:<tag>
#
# PIP_INDEX_URL, if set, reaches the build as a BuildKit secret (never a build arg, so it is not
# recorded in the image history). Extra docker flags go in BUILD_ARGS, e.g. a proxy:
#   BUILD_ARGS="--build-arg HTTPS_PROXY=http://proxy:3128" bash build.sh <tag>
# Behind a TLS-intercepting proxy, put the CA bundle in certs/ first. Pre-fetched wheels and
# source trees in vendor/ (see README) are used instead of downloading them.
set -euo pipefail
TAG="${1:?usage: build.sh <registry>/<repository>:<tag>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/certs" "${HERE}/vendor"
SECRET=()
[ -n "${PIP_INDEX_URL:-}" ] && SECRET=(--secret id=pip_index,env=PIP_INDEX_URL)
# shellcheck disable=SC2086
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 ${SECRET[@]+"${SECRET[@]}"} \
  --build-arg IMAGE_TAG="${TAG##*:}" ${BUILD_ARGS:-} -f "${HERE}/Dockerfile" -t "${TAG}" "${HERE}"
echo "built ${TAG}; push it, pin its digest in site.env (RFT_VERL_IMAGE), and register it with AI Runtime"
