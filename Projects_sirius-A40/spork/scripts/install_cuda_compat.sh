#!/usr/bin/env bash
# Extract NVIDIA's official user-mode compatibility libraries locally.
# Does not install packages or change the system driver / conda environment.
set -euo pipefail
SPORK_SETUP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SPORK_COMPAT_ROOT="$SPORK_SETUP_ROOT/.runtime/cuda-compat-13-0"
SPORK_COMPAT_DEB="$SPORK_SETUP_ROOT/.runtime/cuda-compat-13-0_580.196.07-1ubuntu1_amd64.deb"
SPORK_COMPAT_SHA=b0ee4fb1f439e919ca5c5fd7c5cff3e8b8afdbdb0b5ee38f7c09c047ea41a035
mkdir -p "$SPORK_SETUP_ROOT/.runtime"
if [[ ! -f "$SPORK_COMPAT_DEB" ]]; then
  curl --fail --location --retry 2 --connect-timeout 20 --max-time 300 \
    --continue-at - --output "$SPORK_COMPAT_DEB.incomplete" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-13-0_580.196.07-1ubuntu1_amd64.deb
  printf '%s  %s\n' "$SPORK_COMPAT_SHA" "$SPORK_COMPAT_DEB.incomplete" | sha256sum --check -
  mv -- "$SPORK_COMPAT_DEB.incomplete" "$SPORK_COMPAT_DEB"
fi
printf '%s  %s\n' "$SPORK_COMPAT_SHA" "$SPORK_COMPAT_DEB" | sha256sum --check -
mkdir -p "$SPORK_COMPAT_ROOT"
dpkg-deb --extract "$SPORK_COMPAT_DEB" "$SPORK_COMPAT_ROOT"
printf 'Local CUDA compatibility directory: %s\n' "$SPORK_COMPAT_ROOT/usr/local/cuda-13.0/compat"
