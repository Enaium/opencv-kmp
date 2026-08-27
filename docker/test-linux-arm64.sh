#!/usr/bin/env bash
#
# Runs the linuxArm64 native test suite (plus the JVM suite against the
# linux-aarch64 JNI artifact) inside a native arm64 Linux container.
#
# Prereqs: Docker Desktop on an Apple Silicon Mac (or any arm64 host),
#          the repo checked out with submodules (git submodule update
#          --init --recursive).
#
# Usage:  docker/test-linux-arm64.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Building arm64 test image"
docker build -t opencv-kmp-linux-arm64 docker/

# The OpenCV submodule patches are applied automatically by
# native/CMakeLists.txt on configure, so no explicit git apply here.

echo "==> Running JVM tests against the native arm64 JNI artifact in container"
# Kotlin/Native has no linux-aarch64 host distribution (this Kotlin
# version), so the linuxArm64 klib tests cannot run here either; the
# container verifies the arm64 JNI library end to end.
docker run --rm \
    -v "$(pwd):/workspace" \
    -w /workspace \
    opencv-kmp-linux-arm64 \
    ./gradlew --no-daemon \
        :opencv-kmp:jvmTest
