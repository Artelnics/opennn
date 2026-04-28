#!/bin/bash
# Numerical parity comparison: master vs dev-refactor (CPU only, FP32)
# Usage: bash tests/numerical_parity/run_comparison.sh
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_MASTER="$ROOT/build_parity_master"
BUILD_REFACTOR="$ROOT/build_parity_refactor"
PARITY_DIR="$ROOT/tests/numerical_parity"

echo "=== OpenNN Numerical Parity Test ==="
echo "Root: $ROOT"
echo ""

# -----------------------------------------------------------
# Step 1: Build and run on MASTER
# -----------------------------------------------------------
echo ">>> Step 1: Building master branch..."
git -C "$ROOT" stash --quiet 2>/dev/null || true
git -C "$ROOT" checkout master --quiet

mkdir -p "$BUILD_MASTER"
cd "$BUILD_MASTER"

# Create a minimal CMakeLists for the test
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(parity_master LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

# Add opennn as subdirectory
add_subdirectory(${OPENNN_ROOT}/opennn ${CMAKE_BINARY_DIR}/opennn_lib)

add_executable(parity_test ${PARITY_DIR}/master_test.cpp)
target_link_libraries(parity_test opennn)
if(OpenMP_FOUND)
    target_link_libraries(parity_test OpenMP::OpenMP_CXX)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(parity_test stdc++fs)
endif()
EOF

cmake . -DCMAKE_BUILD_TYPE=Release \
        -DOPENNN_ROOT="$ROOT" \
        -DPARITY_DIR="$PARITY_DIR" \
        -DOpenNN_DISABLE_CUDA=ON \
        2>&1 | tail -3

cmake --build . -j$(nproc) 2>&1 | tail -3
echo ">>> Running master test..."
./parity_test
cp master_outputs.txt "$PARITY_DIR/"

# -----------------------------------------------------------
# Step 2: Build and run on DEV-REFACTOR
# -----------------------------------------------------------
echo ""
echo ">>> Step 2: Building dev-refactor branch..."
git -C "$ROOT" checkout dev-refactor --quiet
git -C "$ROOT" stash pop --quiet 2>/dev/null || true

mkdir -p "$BUILD_REFACTOR"
cd "$BUILD_REFACTOR"

cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(parity_refactor LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
if(OpenMP_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

add_subdirectory(${OPENNN_ROOT}/opennn ${CMAKE_BINARY_DIR}/opennn_lib)

add_executable(parity_test ${PARITY_DIR}/refactor_test.cpp)
target_link_libraries(parity_test opennn)
if(OpenMP_FOUND)
    target_link_libraries(parity_test OpenMP::OpenMP_CXX)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(parity_test stdc++fs)
endif()
EOF

cmake . -DCMAKE_BUILD_TYPE=Release \
        -DOPENNN_ROOT="$ROOT" \
        -DPARITY_DIR="$PARITY_DIR" \
        -DOpenNN_DISABLE_CUDA=ON \
        2>&1 | tail -3

cmake --build . -j$(nproc) 2>&1 | tail -3
echo ">>> Running refactor test..."
./parity_test
cp refactor_outputs.txt "$PARITY_DIR/"

# -----------------------------------------------------------
# Step 3: Compare
# -----------------------------------------------------------
echo ""
echo "=== COMPARISON ==="
echo ""

cd "$PARITY_DIR"

python3 - << 'PYEOF'
import sys

def parse_file(path):
    results = {}
    current_test = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                current_test = line[2:]
                results[current_test] = {}
            elif line and current_test:
                parts = line.split()
                key = parts[0]
                if key == 'output':
                    i, j, val = int(parts[1]), int(parts[2]), float(parts[3])
                    results[current_test][f'output_{i}_{j}'] = val
                elif key == 'error':
                    results[current_test]['error'] = float(parts[1])
                elif key == 'params_number':
                    results[current_test]['params_number'] = int(parts[1])
                else:
                    results[current_test][key] = parts[1]
    return results

master = parse_file('master_outputs.txt')
refactor = parse_file('refactor_outputs.txt')

all_tests = set(master.keys()) | set(refactor.keys())
max_diff = 0
total_checks = 0
failed_checks = 0
tolerance = 1e-4

for test in sorted(all_tests):
    print(f"--- {test} ---")
    if test not in master:
        print(f"  MISSING in master")
        continue
    if test not in refactor:
        print(f"  MISSING in refactor")
        continue

    m = master[test]
    r = refactor[test]

    # Check params_number
    if 'params_number' in m and 'params_number' in r:
        mp, rp = m['params_number'], r['params_number']
        status = "OK" if mp == rp else f"MISMATCH (master={mp}, refactor={rp})"
        print(f"  params_number: master={mp}, refactor={rp} -> {status}")

    # Compare numerical values
    all_keys = set(k for k in m if k.startswith('output_') or k == 'error')
    all_keys |= set(k for k in r if k.startswith('output_') or k == 'error')

    for key in sorted(all_keys):
        total_checks += 1
        mv = m.get(key)
        rv = r.get(key)
        if mv is None or rv is None:
            print(f"  {key}: MISSING (master={mv}, refactor={rv})")
            failed_checks += 1
            continue
        diff = abs(mv - rv)
        max_diff = max(max_diff, diff)
        if diff > tolerance:
            print(f"  {key}: DRIFT master={mv:.10f} refactor={rv:.10f} diff={diff:.2e} > {tolerance}")
            failed_checks += 1
        else:
            print(f"  {key}: OK (diff={diff:.2e})")

print(f"\n=== SUMMARY ===")
print(f"Total numerical checks: {total_checks}")
print(f"Passed (diff <= {tolerance}): {total_checks - failed_checks}")
print(f"Failed (diff > {tolerance}): {failed_checks}")
print(f"Max absolute difference: {max_diff:.2e}")
if failed_checks > 0:
    print(f"\n*** NUMERICAL DRIFT DETECTED ***")
    sys.exit(1)
else:
    print(f"\n*** ALL OUTPUTS MATCH WITHIN TOLERANCE ***")
PYEOF
