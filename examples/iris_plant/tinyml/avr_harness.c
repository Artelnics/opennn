//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   TinyML harness for the exported iris model (ATmega328P / Arduino Uno).
//
//   Runs the exported model over the generated test vectors and prints one
//   line per vector on UART0: the IEEE-754 bits of each output in hex,
//   separated by spaces. Ends with "END" and puts the CPU to sleep with
//   interrupts disabled, which makes simavr exit gracefully.
//
//   Build (avr-gcc):
//     avr-gcc -mmcu=atmega328p -DF_CPU=16000000UL -Os -std=gnu99 \
//         -I<dir with iris_model.c and test_vectors.h> -o iris_avr.elf avr_harness.c -lm
//
//   Run (simavr):
//     simavr -m atmega328p -f 16000000 iris_avr.elf

#define OPENNN_EXPORT_NO_MAIN

// Model source selectable at compile time:
//   -DNN_MODEL_FILE='"iris_model_tables.c"' tests the CEmbedded backend.
#ifndef NN_MODEL_FILE
#define NN_MODEL_FILE "iris_model.c"
#endif
#include NN_MODEL_FILE

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/sleep.h>
#include <stdint.h>

#include "test_vectors.h"

#ifndef F_CPU
#define F_CPU 16000000UL
#endif
#define BAUD 38400UL

static void uart_init(void)
{
    const uint16_t ubrr = (uint16_t)(F_CPU / (16UL * BAUD)) - 1;
    UBRR0H = (uint8_t)(ubrr >> 8);
    UBRR0L = (uint8_t)(ubrr & 0xFF);
    UCSR0B = _BV(TXEN0);
    UCSR0C = _BV(UCSZ01) | _BV(UCSZ00);
}

static void uart_putc(char c)
{
    while (!(UCSR0A & _BV(UDRE0)))
        ;
    UDR0 = (uint8_t)c;
}

static void uart_puts(const char* s)
{
    while (*s)
        uart_putc(*s++);
}

static void uart_put_hex_u32(uint32_t value)
{
    static const char hex[] = "0123456789abcdef";
    for (int8_t shift = 28; shift >= 0; shift -= 4)
        uart_putc(hex[(value >> shift) & 0xF]);
}

int main(void)
{
    uart_init();
    uart_puts("BEGIN\n");

    for (uint8_t v = 0; v < NUM_VECTORS; ++v)
    {
        const float* outputs = calculate_outputs(test_inputs[v]);

        for (uint8_t o = 0; o < NUM_OUTPUTS; ++o)
        {
            union { float f; uint32_t u; } bits;
            bits.f = outputs[o];
            uart_put_hex_u32(bits.u);
            uart_putc(o + 1 < NUM_OUTPUTS ? ' ' : '\n');
        }
    }

    uart_puts("END\n");

    // Sleep with interrupts off: simavr detects this and quits gracefully.
    cli();
    sleep_enable();
    sleep_cpu();

    return 0;
}
