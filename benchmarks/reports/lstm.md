# LSTM benchmark review

**Status: needs a new measured comparison before website publication.**

Both engines use 128 LSTM units and 24-step windows of 15 weather features. The old GPU records are labelled BF16, but the archived analysis identifies FP16 recurrent kernels on the PyTorch side. Verify the actual precision and compare the same prediction task. The old PyTorch inference driver kept a full dataset on the GPU while replaying only one batch. Training also differs in scaling layers and data residency. Remove unused data from the inference comparison and decide whether each timing includes input transfers. CPU LSTM inference exceeds the throughput variation limit on the PyTorch side. Repeat it after stabilizing the machine.

## Recorded observations

These are historical medians from paired sessions on the Intel Core i7-14700F / RTX 5070 Ti reference computer. They are not newly certified results. CPU uses FP32; GPU uses the recorded BF16 label, subject to the issues above.

| Configuration | Batch | Unit | OpenNN | PyTorch | OpenNN / PyTorch |
| --- | --- | --- | --- | --- | --- |
| CPU LSTM inference | 256 | windows/s | 80,285 | 69,304 | 1.158x |
| CPU LSTM training | 256 | windows/s | 23,251 | 13,103 | 1.774x |
| GPU LSTM inference | 256 | windows/s | 2,716,548 | 513,813 | 5.287x |
| GPU LSTM training | 256 | windows/s | 823,255 | 95,842 | 8.590x |

OpenNN throughput / PyTorch throughput. Higher is better.

## Before publication

Every selected record lacks content hashes for its prepared inputs. Record those hashes and the exact build and runtime before the repeat. Check actual inputs and outputs, not only tensor sizes or an old passing flag. Retain each launch and review the 3% variation rule.

The [full review](publication-review.md) contains memory, energy, variation and all pending checks. The [selection manifest](../publication/selection.json) records each source path and SHA-256 hash.

The [archived analysis](archive/2026-09-11/lstm.md) retains the earlier investigation. Its detailed explanations refer to those historical configurations and must not be carried forward without evidence from the new runs.
