//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   PC-side harness for the exported iris model.
//
//   Runs the exported model over the generated test vectors and prints the
//   same format as avr_harness.c (hex IEEE-754 bits of each output, one line
//   per vector, wrapped in BEGIN/END), so both can be compared bit by bit.
//
//   Build: gcc -O2 -std=c99 -I<dir with iris_model.c and test_vectors.h> \
//              -o pc_harness pc_harness.c -lm

#define OPENNN_EXPORT_NO_MAIN

// Model source selectable at compile time:
//   -DNN_MODEL_FILE='"iris_model_tables.c"' tests the CEmbedded backend.
#ifndef NN_MODEL_FILE
#define NN_MODEL_FILE "iris_model.c"
#endif
#include NN_MODEL_FILE

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "test_vectors.h"

int main(void)
{
    printf("BEGIN\n");

    for (int v = 0; v < NUM_VECTORS; ++v)
    {
        const float* outputs = calculate_outputs(test_inputs[v]);

        for (int o = 0; o < NUM_OUTPUTS; ++o)
        {
            uint32_t bits;
            const float value = outputs[o];
            memcpy(&bits, &value, sizeof bits);
            printf("%08x%c", bits, o + 1 < NUM_OUTPUTS ? ' ' : '\n');
        }
    }

    printf("END\n");

    return 0;
}
