# Transformer benchmark review

**Status: needs a new measured comparison before website publication.**

The historical model has six encoder and six decoder layers, width 512, eight attention heads and a 20,000-token vocabulary on WMT14 text. The old drivers differ in causal and padding masks, dropout, and the loss denominator. Align these before repeating training or inference. Translation quality must use generated text on held-out data; equal tensor shapes are not sufficient.

## Recorded observations

These are historical medians from paired sessions on the Intel Core i7-14700F / RTX 5070 Ti reference computer. They are not newly certified results. CPU uses FP32; GPU uses the recorded BF16 label, subject to the issues above.

| Configuration | Batch | Unit | OpenNN | PyTorch | OpenNN / PyTorch |
| --- | --- | --- | --- | --- | --- |
| GPU Transformer inference | 32 | sequences/s | 5,413 | 4,694 | 1.153x |
| GPU Transformer training | 32 | sequences/s | 1,352 | 1,145 | 1.181x |

OpenNN throughput / PyTorch throughput. Higher is better.

CPU training and inference are missing from this selected result set.

## Before publication

Every selected record lacks content hashes for its prepared inputs. Record those hashes and the exact build and runtime before the repeat. Check actual inputs and outputs, not only tensor sizes or an old passing flag. Retain each launch and review the 3% variation rule.

The [full review](publication-review.md) contains memory, energy, variation and all pending checks. The [selection manifest](../publication/selection.json) records each source path and SHA-256 hash.

The [archived analysis](archive/2026-09-11/transformer.md) retains the earlier investigation. Its detailed explanations refer to those historical configurations and must not be carried forward without evidence from the new runs.
