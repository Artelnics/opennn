# CNN benchmark review

**Status: needs a new measured comparison before website publication.**

The historical model is ResNet-50 v1.5 with 25,557,032 parameters, using 224 ? 224 images from a 50,000-image ImageNet subset. It is a different workload from the older CIFAR-sized website article. The old training runs use different image transforms, decoding paths, transfer sizes and warm-up counts. Feed both engines the same prepared pixels and match the timing boundary and warm-up. Separate resident-model inference from application input handling.

## Recorded observations

These are historical medians from paired sessions on the Intel Core i7-14700F / RTX 5070 Ti reference computer. They are not newly certified results. CPU uses FP32; GPU uses the recorded BF16 label, subject to the issues above.

| Configuration | Batch | Unit | OpenNN | PyTorch | OpenNN / PyTorch |
| --- | --- | --- | --- | --- | --- |
| GPU CNN inference | 128 | images/s | 7,060 | 5,634 | 1.253x |
| GPU CNN training | 64 | images/s | 1,667 | 1,401 | 1.190x |

OpenNN throughput / PyTorch throughput. Higher is better.

CPU training and inference are missing from this selected result set.

## Before publication

Every selected record lacks content hashes for its prepared inputs. Record those hashes and the exact build and runtime before the repeat. Check actual inputs and outputs, not only tensor sizes or an old passing flag. Retain each launch and review the 3% variation rule.

The [full review](publication-review.md) contains memory, energy, variation and all pending checks. The [selection manifest](../publication/selection.json) records each source path and SHA-256 hash.

The [archived analysis](archive/2026-09-11/cnn.md) retains the earlier investigation. Its detailed explanations refer to those historical configurations and must not be carried forward without evidence from the new runs.
