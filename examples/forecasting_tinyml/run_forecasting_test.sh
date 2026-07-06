#!/bin/bash
#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   TinyML parity test for exported forecasting models (LSTM and simple
#   recurrent), on the ATmega328P (Arduino Uno) simavr emulator.
#
#   Reuses the model-agnostic harnesses from examples/iris_plant/tinyml.
#   The expression backend unrolls the time loop (code grows with
#   time_steps * hidden^2), so its AVR build is best-effort: if it does not
#   fit the 32 KB flash the variant is checked on PC only, with a warning.
#
#   Usage: run_forecasting_test.sh [--skip-generation]

set -e

FORECAST_DIR="${FORECAST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
HARNESS_DIR="${HARNESS_DIR:-$FORECAST_DIR/../iris_plant/tinyml}"
BUILD="${BUILD:-$HOME/opennn-build}"
WORK="${WORK:-$HOME/tinyml-forecasting}"

N_INPUTS=12   # 6 time steps x 2 features, flattened row-major
N_OUTPUTS=1

# Toolchain discovery: environment override, then PATH (CI runners install via
# apt), then the user-space locations documented in the README (WSL, no sudo).
AVR_GCC="${AVR_GCC:-$(command -v avr-gcc || find "$HOME/.arduino15/packages/arduino/tools/avr-gcc" -name avr-gcc -type f 2>/dev/null | head -1)}"
AVR_BIN_DIR=$(dirname "$AVR_GCC")
SIMAVR="${SIMAVR:-$(command -v simavr || echo "$HOME/simavr-local/root/usr/bin/simavr")}"
export LD_LIBRARY_PATH="$HOME/simavr-local/root/usr/lib/x86_64-linux-gnu:$HOME/simavr-local/root/usr/lib:$LD_LIBRARY_PATH"

# Optional ARM Cortex-M stage
ARM_GCC="${ARM_GCC:-$(command -v arm-none-eabi-gcc || find "$HOME/arm-tools" -name arm-none-eabi-gcc -type f 2>/dev/null | head -1)}"
QEMU_ARM="${QEMU_ARM:-$(command -v qemu-system-arm || find "$HOME/arm-tools" -path '*qemu*' -name qemu-system-arm -type f 2>/dev/null | head -1)}"
ARM_HARNESS_DIR="$HARNESS_DIR/arm"

[ -x "$AVR_GCC" ] || { echo "avr-gcc not found (install the arduino:avr core)"; exit 1; }
[ -x "$SIMAVR" ] || { echo "simavr not found at $SIMAVR"; exit 1; }

mkdir -p "$WORK"

echo "=== 1. Generate and export forecasting models ==="
if [ "$1" != "--skip-generation" ] || [ ! -f "$WORK/lstm_model.c" ]; then
    (cd "$BUILD/bin" && ./forecasting_tinyml)
    for stem in lstm rnn; do
        cp "$BUILD/bin/${stem}_model.c" "$BUILD/bin/${stem}_model_tables.c" \
           "$BUILD/bin/${stem}_reference.csv" "$WORK/"
    done
fi

FAILED=0

for MODEL in lstm rnn; do
    echo ""
    echo "=== 2. Test vectors for $MODEL ==="
    python3 "$HARNESS_DIR/make_test_vectors.py" "$WORK/${MODEL}_reference.csv" \
        "$WORK/test_vectors.h" $N_INPUTS $N_OUTPUTS

    for VARIANT in expression tables; do
        if [ "$VARIANT" = "expression" ]; then MODEL_FILE="${MODEL}_model.c"; else MODEL_FILE="${MODEL}_model_tables.c"; fi

        echo ""
        echo "########## Variant: $MODEL/$VARIANT ($MODEL_FILE) ##########"

        echo "=== 3. Exported model on PC ==="
        gcc -O2 -std=c99 -DNN_MODEL_FILE="\"$MODEL_FILE\"" -I"$WORK" \
            -o "$WORK/pc_${MODEL}_${VARIANT}" "$HARNESS_DIR/pc_harness.c" -lm
        "$WORK/pc_${MODEL}_${VARIANT}" > "$WORK/pc_${MODEL}_${VARIANT}.txt"

        echo "=== 4. ATmega328P build (32 KB flash, 2 KB RAM) ==="
        AVR_OK=1
        if ! "$AVR_GCC" -mmcu=atmega328p -DF_CPU=16000000UL -Os -std=gnu99 \
                -DNN_MODEL_FILE="\"$MODEL_FILE\"" -I"$WORK" \
                -o "$WORK/avr_${MODEL}_${VARIANT}.elf" "$HARNESS_DIR/avr_harness.c" -lm 2> "$WORK/avr_${MODEL}_${VARIANT}_build.log"; then
            AVR_OK=0
            echo "WARNING: AVR build failed for $MODEL/$VARIANT; checking this variant on PC only."
            tail -3 "$WORK/avr_${MODEL}_${VARIANT}_build.log"
        else
            "$AVR_BIN_DIR/avr-size" "$WORK/avr_${MODEL}_${VARIANT}.elf"

            FLASH_BYTES=$("$AVR_BIN_DIR/avr-size" "$WORK/avr_${MODEL}_${VARIANT}.elf" | awk 'NR==2 {print $1 + $2}')
            RAM_BYTES=$("$AVR_BIN_DIR/avr-size" "$WORK/avr_${MODEL}_${VARIANT}.elf" | awk 'NR==2 {print $2 + $3}')

            if [ "$FLASH_BYTES" -gt 32768 ]; then
                AVR_OK=0
                echo "WARNING: firmware needs $FLASH_BYTES B of flash but the ATmega328P has 32768 B;"
                echo "checking this variant on PC only. (Expected for the unrolled expression backend"
                echo "on LSTM models: use the CEmbedded/tables backend for deployment.)"
            elif [ "$RAM_BYTES" -gt 1900 ]; then
                AVR_OK=0
                echo "WARNING: firmware static RAM ($RAM_BYTES B) leaves no stack headroom on the"
                echo "ATmega328P (2048 B); checking this variant on PC only."
            fi
        fi

        if [ "$AVR_OK" = "1" ]; then
            echo "=== 5. simavr run ==="
            timeout 300 "$SIMAVR" -m atmega328p -f 16000000 "$WORK/avr_${MODEL}_${VARIANT}.elf" \
                < /dev/null > "$WORK/avr_${MODEL}_${VARIANT}.txt" 2>&1 || true

            echo "=== 6. Parity check (reference vs PC vs AVR) ==="
            python3 "$HARNESS_DIR/compare_outputs.py" "$WORK/${MODEL}_reference.csv" \
                "$WORK/pc_${MODEL}_${VARIANT}.txt" "$WORK/avr_${MODEL}_${VARIANT}.txt" || FAILED=1
        else
            echo "=== 6. Parity check (reference vs PC) ==="
            python3 "$HARNESS_DIR/compare_outputs.py" "$WORK/${MODEL}_reference.csv" \
                "$WORK/pc_${MODEL}_${VARIANT}.txt" || FAILED=1
        fi

        if [ -n "$ARM_GCC" ] && [ -n "$QEMU_ARM" ]; then
            echo "=== 7. ARM Cortex-M3 build (QEMU mps2-an385, semihosting) ==="
            "$ARM_GCC" -mcpu=cortex-m3 -mthumb -O2 -std=gnu99 --specs=rdimon.specs \
                -Wl,-T,"$ARM_HARNESS_DIR/mps2.ld" \
                -DNN_MODEL_FILE="\"$MODEL_FILE\"" -I"$WORK" \
                "$ARM_HARNESS_DIR/vectors.c" "$ARM_HARNESS_DIR/arm_harness.c" \
                -o "$WORK/arm_${MODEL}_${VARIANT}.elf" -lm
            "${ARM_GCC%gcc}size" "$WORK/arm_${MODEL}_${VARIANT}.elf"

            echo "=== 8. QEMU run ==="
            timeout 120 "$QEMU_ARM" -M mps2-an385 -cpu cortex-m3 \
                -kernel "$WORK/arm_${MODEL}_${VARIANT}.elf" \
                -nographic -semihosting -no-reboot \
                < /dev/null > "$WORK/arm_${MODEL}_${VARIANT}.txt" 2>&1 || true

            echo "=== 9. Parity check (reference vs PC vs ARM) ==="
            python3 "$HARNESS_DIR/compare_outputs.py" "$WORK/${MODEL}_reference.csv" \
                "$WORK/pc_${MODEL}_${VARIANT}.txt" "$WORK/arm_${MODEL}_${VARIANT}.txt" || FAILED=1
        else
            echo "(ARM toolchain/QEMU not found under ~/arm-tools; skipping the Cortex-M stage)"
        fi
    done
done

echo ""
if [ "$FAILED" -ne 0 ]; then echo "RESULT: FAILED"; exit 1; fi
echo "RESULT: ALL VARIANTS PASSED"
