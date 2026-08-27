# Native arm64 Linux test environment for opencv-kmp.
#
# Runs on an arm64 host (Apple M-series Docker Desktop, or an arm64 Linux
# runner) so the linuxArm64 klib and the linux-aarch64 JNI artifact are
# compiled natively — no qemu/x86 emulation involved.
#
# Build:  docker build -t opencv-kmp-linux-arm64 docker/
# Run:    see test-linux-arm64.sh
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-21-jdk-headless \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        curl \
        pkg-config \
        # linuxArm64 klib build uses the aarch64-linux-gnu toolchain
        # (native on arm64 hosts) and GTK3 for the highgui window backend.
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu \
        libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV GRADLE_USER_HOME=/workspace/.gradle-container

WORKDIR /workspace
