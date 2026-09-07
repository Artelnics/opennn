#!/usr/bin/env bash
# Usage: check_headers.sh <c++-compiler> <repo-root> [eigen-include-dirs(;-separated)]
# Compiles every opennn/**/*.h in isolation (-fsyntax-only, CPU mode) so that
# self-sufficiency regressions fail fast instead of hiding behind the PCH.
set -u
cxx=$1
root=$2
export CHK_CXX=$cxx
export CHK_ROOT=$root
export CHK_EIGEN_DIRS="${3:-}"

find "$root/opennn" -path '*/flash_attention_shim' -prune -o -name '*.h' -print0 |
xargs -0 -P "${OPENNN_HEADER_JOBS:-$(nproc)}" -I{} bash -c '
    # Keep each include path as one argument, including checkouts with spaces.
    flags=(-std=c++20 -fsyntax-only -fopenmp -Wno-interference-size -I"$CHK_ROOT")
    IFS=";" read -ra eigen_dirs <<< "$CHK_EIGEN_DIRS"
    for directory in "${eigen_dirs[@]}"; do
        [ -n "$directory" ] && flags+=(-I"$directory")
    done
    rel=${0#"$CHK_ROOT"/}
    out=$(echo "#include \"$rel\"" | "$CHK_CXX" "${flags[@]}" -x c++ - 2>&1) ||
        { printf "FAIL: %s\n%s\n" "$rel" "$(head -c 2000 <<< "$out")"; exit 1; }
' {}
status=$?
[ $status -eq 0 ] && echo "OK: all opennn headers compile in isolation"
exit $status
