#!/usr/bin/env bash

set -euo pipefail

mode="${1:-quick}"
if [[ "$mode" == "-h" || "$mode" == "--help" ]]; then
    mode=help
fi
if [[ $# -gt 0 ]]; then
    shift
fi

backend=cpu
filter=
build_root=
jobs=
reconfigure=OFF
use_sccache=ON

usage() {
    cat <<'EOF'
Usage:
  tools/verify.sh quick --filter 'Dense.*:DenseNoBiasTest.*' [--backend cpu|cuda]
  tools/verify.sh cpu|cuda|full

Options:
  --backend cpu|cuda  Backend for quick mode (default: cpu)
  --filter PATTERN    GoogleTest filter; required for quick mode
  --build-root PATH   Override the external verification cache root
  --jobs N            Limit parallel build jobs
  --reconfigure       Force CMake configuration to run again
  --no-sccache        Do not use sccache even when it is installed
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            backend="${2:?--backend requires cpu or cuda}"
            shift 2
            ;;
        --filter)
            filter="${2:?--filter requires a GoogleTest pattern}"
            shift 2
            ;;
        --build-root)
            build_root="${2:?--build-root requires a path}"
            shift 2
            ;;
        --jobs)
            jobs="${2:?--jobs requires a positive integer}"
            shift 2
            ;;
        --reconfigure)
            reconfigure=ON
            shift
            ;;
        --no-sccache)
            use_sccache=OFF
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$mode" in
    quick|cpu|cuda|full) ;;
    help)
        usage
        exit 0
        ;;
    *)
        echo "Mode must be quick, cpu, cuda, or full." >&2
        exit 2
        ;;
esac

case "$backend" in
    cpu|cuda) ;;
    *)
        echo "Backend must be cpu or cuda." >&2
        exit 2
        ;;
esac

if [[ "$mode" == quick && -z "$filter" ]]; then
    echo "Quick verification requires --filter." >&2
    usage >&2
    exit 2
fi

if [[ -n "$jobs" && ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs must be a positive integer." >&2
    exit 2
fi

command -v cmake >/dev/null 2>&1 || {
    echo "cmake was not found on PATH." >&2
    exit 127
}
command -v ninja >/dev/null 2>&1 || {
    echo "ninja was not found on PATH." >&2
    exit 127
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmake_args=(
    "-DOPENNN_VERIFY_MODE=$mode"
    "-DOPENNN_VERIFY_BACKEND=$backend"
    "-DOPENNN_VERIFY_RECONFIGURE=$reconfigure"
    "-DOPENNN_USE_SCCACHE=$use_sccache"
)
if [[ -n "$filter" ]]; then
    cmake_args+=("-DOPENNN_TEST_FILTER=$filter")
fi
if [[ -n "$build_root" ]]; then
    cmake_args+=("-DOPENNN_BUILD_ROOT:PATH=$build_root")
fi
if [[ -n "$jobs" ]]; then
    cmake_args+=("-DOPENNN_VERIFY_JOBS=$jobs")
fi

cmake "${cmake_args[@]}" -P "$script_dir/verify.cmake"
