#!/bin/bash
#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   End-to-end TinyML parity test for the exported iris model (WSL/Linux).
#
#   1. Runs the iris_plant example (trains and exports iris_model.c + iris_reference.csv)
#   2. Compiles the exported model natively (PC) and for ATmega328P (Arduino Uno)
#   3. Runs the AVR firmware in the simavr emulator
#   4. Checks output parity: OpenNN reference vs PC vs AVR emulator
#
#   Prerequisites (user-space, no root):
#     - OpenNN built at $BUILD (cmake --build ... --target iris_plant)
#     - arduino-cli AVR core in ~/.arduino15 (provides avr-gcc + avr-libc)
#     - simavr extracted at ~/simavr-local/root (dpkg -x of simavr + libsimavr2 + libelf1t64)
#
#   Usage: run_tinyml_test.sh [--skip-training]

set -e

# TINYML_DIR can be set from the environment (needed when piping this script
# into bash, where BASH_SOURCE is not available).
TINYML_DIR="${TINYML_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
BUILD="${BUILD:-$HOME/opennn-build}"
RUN_DIR="$BUILD/bin"
WORK="${WORK:-$HOME/tinyml-iris}"

N_INPUTS=4
N_OUTPUTS=3

AVR_GCC=$(find "$HOME/.arduino15/packages/arduino/tools/avr-gcc" -name avr-gcc -type f 2>/dev/null | head -1)
AVR_BIN_DIR=$(dirname "$AVR_GCC")
SIMAVR="$HOME/simavr-local/root/usr/bin/simavr"
export LD_LIBRARY_PATH="$HOME/simavr-local/root/usr/lib/x86_64-linux-gnu:$HOME/simavr-local/root/usr/lib:$LD_LIBRARY_PATH"

[ -x "$AVR_GCC" ] || { echo "avr-gcc not found (install the arduino:avr core)"; exit 1; }
[ -x "$SIMAVR" ] || { echo "simavr not found at $SIMAVR"; exit 1; }

mkdir -p "$WORK"

echo "=== 1. Train and export (iris_plant example) ==="
if [ "$1" != "--skip-training" ] || [ ! -f "$WORK/iris_model.c" ]; then
    (cd "$RUN_DIR" && ./iris_plant)
    cp "$RUN_DIR/iris_model.c" "$RUN_DIR/iris_model_tables.c" "$RUN_DIR/iris_reference.csv" "$WORK/"
fi

echo "=== 2. Generate test vectors header ==="
python3 "$TINYML_DIR/make_test_vectors.py" "$WORK/iris_reference.csv" "$WORK/test_vectors.h" $N_INPUTS $N_OUTPUTS

# Test both C backends: 'expression' (unrolled formulas) and 'tables' (CEmbedded)
FAILED=0
for VARIANT in expression tables; do
    if [ "$VARIANT" = "expression" ]; then MODEL_FILE="iris_model.c"; else MODEL_FILE="iris_model_tables.c"; fi

    echo ""
    echo "########## Variant: $VARIANT ($MODEL_FILE) ##########"

    echo "=== 3. Exported model on PC ==="
    gcc -O2 -std=c99 -DNN_MODEL_FILE="\"$MODEL_FILE\"" -I"$WORK" \
        -o "$WORK/pc_harness_$VARIANT" "$TINYML_DIR/pc_harness.c" -lm
    "$WORK/pc_harness_$VARIANT" > "$WORK/pc_output_$VARIANT.txt"

    echo "=== 4. Exported model for ATmega328P (Arduino Uno) ==="
    "$AVR_GCC" -mmcu=atmega328p -DF_CPU=16000000UL -Os -std=gnu99 \
        -DNN_MODEL_FILE="\"$MODEL_FILE\"" -I"$WORK" \
        -o "$WORK/iris_avr_$VARIANT.elf" "$TINYML_DIR/avr_harness.c" -lm
    echo "--- Memory footprint (ATmega328P: 32 KB flash, 2 KB RAM) ---"
    "$AVR_BIN_DIR/avr-size" "$WORK/iris_avr_$VARIANT.elf"

    echo "=== 5. Run in simavr emulator ==="
    # simavr echoes the firmware's UART on stderr (colorized), so capture both streams.
    timeout 120 "$SIMAVR" -m atmega328p -f 16000000 "$WORK/iris_avr_$VARIANT.elf" \
        > "$WORK/avr_output_$VARIANT.txt" 2>&1 || true

    echo "=== 6. Parity check ==="
    python3 "$TINYML_DIR/compare_outputs.py" "$WORK/iris_reference.csv" \
        "$WORK/pc_output_$VARIANT.txt" "$WORK/avr_output_$VARIANT.txt" || FAILED=1
done

echo ""
if [ "$FAILED" -ne 0 ]; then echo "RESULT: FAILED"; exit 1; fi
echo "RESULT: ALL VARIANTS PASSED"
