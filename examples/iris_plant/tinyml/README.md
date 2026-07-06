# TinyML parity test for the exported iris model

Proves that a model trained with OpenNN and exported to plain C produces the
same outputs when compiled for an Arduino-class microcontroller (ATmega328P /
Arduino Uno) and run inside the [simavr](https://github.com/buserror/simavr)
emulator.

Both C export backends of `ModelExpression` are tested:

- `ProgrammingLanguage::C` — readable unrolled formulas, one line per neuron.
  Weights are embedded in the code, so flash cost is ~50 B per weight.
- `ProgrammingLanguage::CEmbedded` — weight tables + generic loops, float-only,
  no heap. Flash cost ~4 B per weight; this is the backend that scales to
  larger models. Caveat on AVR: `static const` tables live in `.data` (copied
  to RAM at startup, ~4 B RAM per weight) because AVR is a Harvard
  architecture; using `PROGMEM` to keep them flash-only is a possible future
  optimization. On ARM Cortex-M targets const tables stay in flash.

## Files

| File | Purpose |
| --- | --- |
| `run_tinyml_test.sh` | End-to-end orchestration (train → export → compile → emulate → compare) |
| `avr_harness.c` | ATmega328P firmware: runs the model over the test vectors, prints outputs as hex float bits over UART0, then sleeps so simavr exits |
| `pc_harness.c` | Same harness compiled natively, same output format |
| `make_test_vectors.py` | Converts `iris_reference.csv` (written by the iris example) into `test_vectors.h` |
| `compare_outputs.py` | Parity check: OpenNN reference vs PC vs AVR (tolerates simavr's ANSI-colored stderr echo) |
| `check_python_export.py` | Parity check for the Python export (runs `iris_model.py` with a numpy shim, no dependencies) |

## Prerequisites (user-space, no root)

- OpenNN built with the iris example, e.g. in WSL:

      cmake -S /mnt/c/Artelnics/opennn -B ~/opennn-build \
          -DCMAKE_BUILD_TYPE=Release -DOpenNN_DISABLE_CUDA=ON
      cmake --build ~/opennn-build -j --target iris_plant

- AVR toolchain from the Arduino ecosystem (installs to `~/.arduino15`):

      arduino-cli core install arduino:avr

- simavr extracted from the Ubuntu packages (no `sudo` needed):

      mkdir -p ~/simavr-local && cd ~/simavr-local
      apt-get download simavr libsimavr2 libelf1t64
      for f in *.deb; do dpkg -x "$f" root; done

## Run

    ./run_tinyml_test.sh                  # trains, exports and checks parity
    ./run_tinyml_test.sh --skip-training  # reuses the previous export

When piping the script into bash, set `TINYML_DIR` to this directory first.

## What "pass" means

- The exported model compiles unmodified with `avr-gcc -mmcu=atmega328p`
  (`OPENNN_EXPORT_NO_MAIN` removes the demo `main()`; inference uses no heap).
- The firmware fits the ATmega328P (32 KB flash, 2 KB RAM).
- Max abs difference vs the OpenNN reference outputs is below the tolerance
  (default 1e-4; measured ~9e-8, i.e. float32 round-off only) and the
  predicted class matches on every test vector.

Measured on 2026-07-06 (avr-gcc 7.3.0, simavr 1.6, WSL2 Ubuntu 24.04),
4-16-3 classification MLP (131 parameters):

    Variant      Flash (text+data)   RAM (data+bss)   Max abs diff vs OpenNN
    expression   6776 B              186 B            8.948e-08
    tables       3438 B              858 B            8.948e-08
    python       (PC only)           -                8.791e-08
    Predicted class agreement: 9/9 on all — RESULT: ALL VARIANTS PASSED
