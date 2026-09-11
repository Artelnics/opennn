# Dense benchmark review

**Status: needs a new measured comparison before website publication.**

Both engines use a 28?1024?1024?1 classifier on HIGGS. The historical training paths use different sample-order policies; align and record those policies for the new comparison. GPU dense training exceeds the throughput variation limit on the PyTorch side. Its energy and throughput windows also differ in duration, so both boundaries need checking. Record library versions and the selected kernel configuration before the new session.

## Recorded observations

These are historical medians from paired sessions on the Intel Core i7-14700F / RTX 5070 Ti reference computer. They are not newly certified results. CPU uses FP32; GPU uses the recorded BF16 label, subject to the issues above.

| Configuration | Batch | Unit | OpenNN | PyTorch | OpenNN / PyTorch |
| --- | --- | --- | --- | --- | --- |
| CPU Dense inference | 4096 | samples/s | 220,512 | 170,329 | 1.295x |
| CPU Dense training | 4096 | samples/s | 70,120 | 54,520 | 1.286x |
| GPU Dense inference | 8192 | samples/s | 39,387,890 | 38,689,107 | 1.018x |
| GPU Dense training | 8192 | samples/s | 11,406,741 | 10,048,603 | 1.135x |

OpenNN throughput / PyTorch throughput. Higher is better.

## Before publication

Every selected record lacks content hashes for its prepared inputs. Record those hashes and the exact build and runtime before the repeat. Check actual inputs and outputs, not only tensor sizes or an old passing flag. Retain each launch and review the 3% variation rule.

The [full review](publication-review.md) contains memory, energy, variation and all pending checks. The [selection manifest](../publication/selection.json) records each source path and SHA-256 hash.

The [archived analysis](archive/2026-09-11/dense.md) retains the earlier investigation. Its detailed explanations refer to those historical configurations and must not be carried forward without evidence from the new runs.
