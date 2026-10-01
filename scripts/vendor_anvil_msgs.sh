#!/usr/bin/env bash
# Extracts the prebuilt anvil_msgs install tree from the loader's ros2 image into
# vendor/anvil_msgs, for docker/classical/Dockerfile to overlay. Run on the robot
# PC. vendor/ is gitignored: the messages are proprietary.
#
# LOADER_IMAGE defaults to the image of the running loader ros2 container.
set -euo pipefail

cd "$(dirname "$0")/.."

LOADER_IMAGE="${LOADER_IMAGE:-$(docker inspect --format '{{.Config.Image}}' anvil-loader-ros2-1 2>/dev/null || true)}"
if [ -z "$LOADER_IMAGE" ]; then
    echo "loader container not running — set LOADER_IMAGE to the loader's ros2 image" >&2
    exit 1
fi

rm -rf vendor/anvil_msgs
mkdir -p vendor
CID=$(docker create --entrypoint bash "$LOADER_IMAGE")
trap 'docker rm "$CID" >/dev/null' EXIT
docker cp "$CID":/workspace/install/anvil_msgs vendor/anvil_msgs
echo "vendored anvil_msgs from $LOADER_IMAGE"
