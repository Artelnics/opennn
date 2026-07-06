#!/bin/bash
#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   Shared toolchain discovery for the TinyML parity pipelines
#   (sourced by run_tinyml_test.sh and run_forecasting_test.sh).
#
#   Order: environment override, then PATH (CI runners install via apt), then
#   the user-space locations documented in the README (WSL without sudo).

discover_tinyml_toolchains() {
    AVR_GCC="${AVR_GCC:-$(command -v avr-gcc || find "$HOME/.arduino15/packages/arduino/tools/avr-gcc" -name avr-gcc -type f 2>/dev/null | head -1)}"
    AVR_BIN_DIR=$(dirname "$AVR_GCC")
    SIMAVR="${SIMAVR:-$(command -v simavr || echo "$HOME/simavr-local/root/usr/bin/simavr")}"
    export LD_LIBRARY_PATH="$HOME/simavr-local/root/usr/lib/x86_64-linux-gnu:$HOME/simavr-local/root/usr/lib:$LD_LIBRARY_PATH"

    # Optional ARM Cortex-M stage (skipped by the callers when not found)
    ARM_GCC="${ARM_GCC:-$(command -v arm-none-eabi-gcc || find "$HOME/arm-tools" -name arm-none-eabi-gcc -type f 2>/dev/null | head -1)}"
    QEMU_ARM="${QEMU_ARM:-$(command -v qemu-system-arm || find "$HOME/arm-tools" -path '*qemu*' -name qemu-system-arm -type f 2>/dev/null | head -1)}"

    [ -x "$AVR_GCC" ] || { echo "avr-gcc not found (install the arduino:avr core or apt gcc-avr)"; exit 1; }
    [ -x "$SIMAVR" ] || { echo "simavr not found at $SIMAVR"; exit 1; }
}
