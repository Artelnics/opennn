//   Minimal Cortex-M vector table for QEMU mps2-an385.
//
//   QEMU's M-profile reset reads the initial stack pointer from offset 0 and
//   the reset handler from offset 4 of the image base. The newlib/rdimon crt0
//   (_start) initializes .bss and the C runtime, then calls main().

#include <stdint.h>

extern void _start(void);

#define STACK_TOP 0x20400000u

__attribute__((section(".vectors"), used))
static const uint32_t vector_table[2] = {
    STACK_TOP,
    (uint32_t)&_start
};
