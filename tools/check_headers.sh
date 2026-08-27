#!/usr/bin/env bash
# Usage: check_headers.sh <c++-compiler> <repo-root> [eigen-include-dirs(;-separated)]
# Compiles every opennn/**/*.h in isolation (-fsyntax-only, CPU mode) so that
# self-sufficiency regressions fail fast instead of hiding behind the PCH.
set -u
cxx=$1
root=$2
IFS=';' read -ra eigen_dirs <<< "${3:-}"

# Only the repo root goes on the include path: headers name their folder, so a
# stale bare include has to fail here rather than resolve against its neighbours.
inc=(-I"$root")
for d in "${eigen_dirs[@]}"; do
    [ -n "$d" ] && inc+=(-I"$d")
done

export CHK_CXX=$cxx
export CHK_ROOT=$root
export CHK_FLAGS="-std=c++20 -fsyntax-only -fopenmp -Wno-interference-size ${inc[*]}"

find "$root/opennn" -path '*/flash_attention_shim' -prune -o -name '*.h' -print0 |
xargs -0 -P "$(nproc)" -I{} bash -c '
    rel=${0#"$CHK_ROOT"/}
    out=$(echo "#include \"$rel\"" | $CHK_CXX $CHK_FLAGS -x c++ - 2>&1) ||
        { printf "FAIL: %s\n%s\n" "$rel" "$(head -c 2000 <<< "$out")"; exit 1; }
' {}
status=$?
[ $status -eq 0 ] && echo "OK: all opennn headers compile in isolation"
exit $status
