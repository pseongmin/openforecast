#!/bin/bash
# Push the Agenthon Track-2 agent image to a public registry and print the digest
# that goes into submission.json's image.digest.
# Requires: a GitHub token with `write:packages` in $GH_TOKEN (or $PSEONGMIN_GITHUB_TOKEN).
# Docker credentials are written to a throwaway config dir and deleted on exit.
set -euo pipefail
USER_NAME="${GH_USER:-pseongmin}"
IMAGE="${IMAGE:-ghcr.io/$USER_NAME/openforecast-t2}"
TAG="${TAG:-v0.1}"
LOCAL="${LOCAL:-openforecast-t2:dev}"
TOKEN="${GH_TOKEN:-${PSEONGMIN_GITHUB_TOKEN:-}}"
[ -n "$TOKEN" ] || { echo "no token: set GH_TOKEN or PSEONGMIN_GITHUB_TOKEN (needs write:packages)"; exit 2; }

CFG=$(mktemp -d); trap 'rm -rf "$CFG"' EXIT
echo "$TOKEN" | DOCKER_CONFIG="$CFG" docker login ghcr.io -u "$USER_NAME" --password-stdin
DOCKER_CONFIG="$CFG" docker tag "$LOCAL" "$IMAGE:$TAG"
DOCKER_CONFIG="$CFG" docker push "$IMAGE:$TAG"
DIGEST=$(DOCKER_CONFIG="$CFG" docker inspect --format='{{index .RepoDigests 0}}' "$IMAGE:$TAG" | cut -d@ -f2)
echo
echo "image.registry   = ghcr.io"
echo "image.repository = $USER_NAME/openforecast-t2"
echo "image.digest     = $DIGEST"
echo
echo "Make the package public once: https://github.com/users/$USER_NAME/packages/container/openforecast-t2/settings"
echo "Then verify an anonymous pull from a logged-out client:"
echo "  DOCKER_CONFIG=\$(mktemp -d) docker pull $IMAGE@$DIGEST"
