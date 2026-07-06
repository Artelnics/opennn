# TinyML parity test for exported forecasting models (RNN / LSTM)

Proves that OpenNN forecasting networks exported to plain C reproduce the
library's outputs on microcontroller targets, using two emulators:

- **ATmega328P / Arduino Uno** (8-bit AVR) via [simavr](https://github.com/buserror/simavr)
- **ARM Cortex-M3** via QEMU (`mps2-an385` machine, semihosting console)

Two models (`ForecastingLstmNetwork` and `ForecastingNetwork`, 6 time steps x
2 features, hidden size 6) x two C backends (`C` expression and `CEmbedded`
weight tables) are checked against reference vectors computed by OpenNN.
Training quality is irrelevant here — the networks keep their Glorot
initialization; what is verified is that the exported inference code is a
faithful copy of the library's forward pass.

## Files

| File | Purpose |
| --- | --- |
| `main.cpp` | Builds the networks, exports both C backends and writes the reference CSVs |
| `run_forecasting_test.sh` | End-to-end orchestration over models x backends x targets |

The model-agnostic harnesses (AVR UART, ARM semihosting, PC, comparison
scripts) are shared from [`../iris_plant/tinyml/`](../iris_plant/tinyml/).

## Prerequisites (user-space, no root)

Same as the iris pipeline (OpenNN build, arduino:avr core, simavr), plus the
optional ARM stage:

    # xPack arm-none-eabi-gcc and xPack QEMU Arm, extracted under ~/arm-tools
    # (relocatable tarballs from github.com/xpack-dev-tools, no sudo needed)

If `~/arm-tools` is missing the Cortex-M stage is skipped with a note.

## Run

    ./run_forecasting_test.sh                    # export + all parity checks
    ./run_forecasting_test.sh --skip-generation  # reuse previous exports

## Results (2026-07-06, avr-gcc 7.3.0, simavr 1.6, arm-none-eabi-gcc 15.2.1, QEMU 9.2.4)

    Model/backend     AVR flash (text+data)  Cortex-M3    Max abs diff vs OpenNN
    lstm/expression   42538 B -> DOES NOT FIT  passes       ~2e-07 (PC/ARM)
    lstm/tables        4368 B  passes          passes       ~2e-07 AVR, bit-exact PC=ARM
    rnn/expression    12206 B  passes          passes       ~5e-07
    rnn/tables         3434 B  passes          passes       ~5e-07 AVR, bit-exact PC=ARM

    RESULT: ALL VARIANTS PASSED

Three takeaways:

- The unrolled expression backend grows O(time_steps x hidden^2): the LSTM
  does not fit the Uno's 32 KB flash (detected and reported by the script,
  which then checks that variant on PC/ARM only). The CEmbedded tables
  backend encodes the same model in ~4 KB.
- On AVR the CEmbedded weight tables live in flash (`PROGMEM`), so RAM only
  holds the working buffers (~0.6 KB here, of the Uno's 2 KB).
- On Cortex-M the CEmbedded output is bit-identical to the PC build of the
  same C file (pure float32, same operation order); differences vs OpenNN are
  float round-off only (~1e-07 with tanh/sigmoid chains).

The pipelines also run in CI (`.github/workflows/tinyml-parity.yml`).
