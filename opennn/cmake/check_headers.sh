#!/usr/bin/env bash
# Usage: check_headers.sh <c++-compiler> <repo-root> [eigen-include-dirs(;-separated)]
# Compiles every opennn/*.h in isolation (-fsyntax-only, CPU mode) so that
# self-sufficiency regressions fail fast instead of hiding behind the PCH.
set -u
cxx=$1
root=$2
IFS=';' read -ra eigen_dirs <<< "${3:-}"

inc=(-I"$root" -I"$root/opennn")
for d in "${eigen_dirs[@]}"; do
    [ -n "$d" ] && inc+=(-I"$d")
done

export CHK_CXX=$cxx
export CHK_FLAGS="-std=c++20 -fsyntax-only -fopenmp ${inc[*]}"

printf '%s\n' "$root"/opennn/*.h |
xargs -P "$(nproc)" -I{} bash -c '
    name=$(basename "{}")
    out=$(echo "#include \"opennn/$name\"" | $CHK_CXX $CHK_FLAGS -x c++ - 2>&1) ||
        { printf "FAIL: %s\n%s\n" "$name" "$(head -c 2000 <<< "$out")"; exit 1; }
'
status=$?
[ $status -eq 0 ] && echo "OK: all opennn headers compile in isolation"
exit $status
