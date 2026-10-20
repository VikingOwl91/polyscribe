#!/usr/bin/env bash
# Install the system packages needed to build and test PolyScribe.
#
# whisper.cpp is compiled from source and its bindings are generated with
# bindgen, so the build needs a C/C++ toolchain, cmake and libclang. ffmpeg
# decodes the audio that the tests and the e2e job transcribe. GitHub runners
# ship these packages, other runners (for example Gitea act containers) do
# not, so install them explicitly instead of relying on the runner image.
#
# Usage: scripts/ci/install-system-deps.sh [--dry-run]

set -euo pipefail

packages=(
    build-essential
    cmake
    clang
    libclang-dev
    pkg-config
    ffmpeg
)

if [ "$(id -u)" -eq 0 ]; then
    apt=(apt-get)
else
    apt=(sudo apt-get)
fi

if [ "${1:-}" = "--dry-run" ]; then
    printf '%q ' "${apt[@]}" update
    printf '\n'
    printf '%q ' "${apt[@]}" install -y "${packages[@]}"
    printf '\n'
    exit 0
fi

"${apt[@]}" update
"${apt[@]}" install -y "${packages[@]}"
