/* Minimal linker script for QEMU mps2-an385 (Cortex-M3) with semihosting.
 *
 * Everything (vectors, code, data, bss, heap) lives in the 4 MB ZBT SSRAM at
 * 0x00000000; the stack is placed at the top of the SRAM block at 0x20000000
 * (see vectors.c). QEMU loads the ELF segments directly, so no flash-to-RAM
 * copy is needed.
 */

MEMORY
{
    CODE_RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 4M
    DATA_RAM (rw)  : ORIGIN = 0x20000000, LENGTH = 4M
}

ENTRY(_start)

SECTIONS
{
    .text :
    {
        KEEP(*(.vectors))
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > CODE_RAM

    .init : { KEEP(*(.init)) } > CODE_RAM
    .fini : { KEEP(*(.fini)) } > CODE_RAM

    .ARM.exidx : { *(.ARM.exidx*) } > CODE_RAM

    .data :
    {
        . = ALIGN(4);
        *(.data*)
        . = ALIGN(4);
    } > CODE_RAM

    .bss :
    {
        . = ALIGN(4);
        __bss_start__ = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end__ = .;
    } > CODE_RAM

    . = ALIGN(8);
    end = .;
    __end__ = .;

    /* Semihosting crt0 reads the stack top from __stack. */
    __stack = ORIGIN(DATA_RAM) + LENGTH(DATA_RAM);
}
