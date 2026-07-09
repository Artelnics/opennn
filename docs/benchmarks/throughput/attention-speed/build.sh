#!/usr/bin/env bash
# Build the attention-speed Transformer binaries via the benchmarks CMake targets,
# then symlink them here so the run_*.py harnesses find ./opennn_transformer_*.
#
#   build.sh                 # build all four
#   build.sh NAME [NAME...]  # build only the named targets
#
# Targets: opennn_transformer_resident, opennn_transformer_infer,
#          opennn_transformer_train, opennn_attention_validate
#
# Portable: no machine-specific paths. Point BENCH_BUILD_DIR at your configured
# benchmark build tree (default <repo>/build-benchmarks).
set -e
cd "$(dirname "$0")"

REPO_ROOT="$(cd ../../../.. && pwd)"
BUILD_DIR="${BENCH_BUILD_DIR:-$REPO_ROOT/build-benchmarks}"

ALL=(opennn_transformer_resident opennn_transformer_infer \
     opennn_transformer_train opennn_attention_validate)
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=("${ALL[@]}")

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  echo "Configure the benchmark build first, e.g.:"
  echo "  cmake -S \"$REPO_ROOT\" -B \"$BUILD_DIR\" -DOpenNN_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release"
  echo "(add your usual CUDA flags). Override the build dir with BENCH_BUILD_DIR."
  exit 1
fi

cmake --build "$BUILD_DIR" --target "${targets[@]}" -j

for t in "${targets[@]}"; do
  bin="$(find "$BUILD_DIR" -type f -name "$t" -perm -u+x 2>/dev/null | head -1)"
  if [ -n "$bin" ]; then ln -sf "$bin" "./$t"; echo "linked ./$t -> $bin"; fi
done
