//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   TinyML harness for exported models on ARM Cortex-M (QEMU mps2-an385).
//
//   Same protocol as avr_harness.c: one line per test vector with the
//   IEEE-754 bits of each output in hex, wrapped in BEGIN/END. Output goes
//   through semihosting (printf), so run QEMU with -semihosting.
//
//   Build (see run_forecasting_test.sh):
//     arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -O2 -std=gnu99 \
//         --specs=rdimon.specs -Wl,-T,mps2.ld -DNN_MODEL_FILE='"model.c"' \
//         -I<work dir> vectors.c arm_harness.c -o arm_fw.elf -lm
//
//   Run:
//     qemu-system-arm -M mps2-an385 -cpu cortex-m3 -kernel arm_fw.elf \
//         -nographic -semihosting -no-reboot

#define OPENNN_EXPORT_NO_MAIN

#ifndef NN_MODEL_FILE
#define NN_MODEL_FILE "iris_model.c"
#endif
#include NN_MODEL_FILE

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "test_vectors.h"

extern void initialise_monitor_handles(void);

int main(void)
{
    initialise_monitor_handles();

    printf("BEGIN\n");

    for (int v = 0; v < NUM_VECTORS; ++v)
    {
        const float* outputs = calculate_outputs(test_inputs[v]);

        for (int o = 0; o < NUM_OUTPUTS; ++o)
        {
            uint32_t bits;
            const float value = outputs[o];
            memcpy(&bits, &value, sizeof bits);
            printf("%08lx%c", (unsigned long)bits, o + 1 < NUM_OUTPUTS ? ' ' : '\n');
        }
    }

    printf("END\n");

    exit(0);
}
