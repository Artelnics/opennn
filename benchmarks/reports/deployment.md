# Deployment and source size

**Status: file totals verified; target-bundle checks and source recount pending.**

The September 10 deployment study counts each small OpenNN application and its
native runtime files. The PyTorch column includes the application and a standard
Python installation. This is not a comparison of two minimal, pruned bundles.
It uses the laptop's Linux/WSL packages and does not describe a Windows bundle.

| Backend | Application | OpenNN (MB) | PyTorch (MB) | OpenNN / PyTorch |
| --- | --- | --- | --- | --- |
| cpu-eigen | Dense | 4.9 | 809.0 | 0.6% |
| cpu-eigen | LSTM | 4.9 | 809.0 | 0.6% |
| cpu-eigen | CNN | 4.9 | 809.0 | 0.6% |
| cpu-eigen | Transformer | 4.9 | 809.0 | 0.6% |
| cpu-mkl | Dense | 285.2 | 809.0 | 35.2% |
| cpu-mkl | LSTM | 285.2 | 809.0 | 35.2% |
| cpu-mkl | CNN | 285.2 | 809.0 | 35.2% |
| cpu-mkl | Transformer | 285.2 | 809.0 | 35.2% |
| cuda | Dense | 1,086.5 | 4,806.9 | 22.6% |
| cuda | LSTM | 1,359.2 | 4,806.9 | 28.3% |
| cuda | CNN | 1,694.7 | 4,806.9 | 35.3% |
| cuda | Transformer | 1,086.5 | 4,806.9 | 22.6% |

OpenNN deployment size / PyTorch deployment size x 100. Lower is better.
MB means 1,000,000 bytes. The GPU FP32 and BF16 file inventories match and are
shown once. We exclude trained weights, datasets, OS/driver files and caches.
A clean-target run must still verify that each collected bundle is complete.

## Lines of code

The source audit at commit `5b4dac2cc` counted 64,700 OpenNN code lines, excluding
blanks and comments. The paired breast-cancer application counted 28 OpenNN
lines and 32 PyTorch lines under the saved line-count rule. These are dated
example counts, not the shortest possible applications.

The separate 13/29 counts describe selected statement lines under different
language rules. Do not relabel them as the complete application line counts.
The inherited PyTorch library count of 834,319 lines has no matching raw count
in this evidence. Recount both source trees at pinned revisions with one rule
before publishing a library-size ratio.

## Dependencies

OpenNN uses no Python packages but still needs its native runtime libraries.
Count those separately. Use the installed package inventory for the exact CPU
or GPU PyTorch environment, rather than an unrelated development environment
or one package count for both devices.

See the [full review](publication-review.md), [application procedure](../APPLICATIONS.md)
and [archived source analysis](archive/2026-09-11/deployment.md).
