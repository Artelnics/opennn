#!/usr/bin/env bash
# Use an existing CUDA verification build. The ordinary CUDA gate runs first.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: tools/check_cuda_memory.sh BUILD_DIR [GTEST_FILTER]" >&2
    exit 2
fi
build_dir="$(cd -- "$1" && pwd)"
filter="${2:--*DeathTest*}"
export OPENNN_TEST_REQUIRE_CUDA=1
export OPENNN_THREADS="${OPENNN_THREADS:-4}"
# GoogleTest's fresh-process death test hangs under child-process injection on
# the tested WSL stack. It is covered by the ordinary full CUDA gate instead.
# Keep leak detection enabled and fail even when all GoogleTest assertions pass.
cd "$build_dir"
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
    --target-processes all ./bin/opennn_tests \
    --gtest_filter="$filter" --gtest_fail_if_no_test_selected --gtest_color=no
