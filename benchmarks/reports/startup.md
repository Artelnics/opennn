# Application startup

**Status: repeat before publication.**

We measure the time from starting a fresh process to receiving its first
completed prediction in host memory. The September 10 Python comparison contains
660 timed launches, with no failed processes. However, 43 of the 44 engine
configurations exceed the 3% variation limit. Every paired comparison remains
diagnostic.

The computer was an Intel Core i7-12700H with an RTX 3060 Laptop GPU under WSL2.
Each configuration has 15 timed launches. CPU uses FP32. GPU tests cover FP32
and BF16, each with saved and empty application tuning caches. Filesystem
caches remain warm, and no trained model is loaded from disk.

| Backend | Model | Precision | Cache | OpenNN (ms) | PyTorch (ms) |
| --- | --- | --- | --- | --- | --- |
| cpu-eigen | Dense | fp32 | reused | 1.739 | 661.609 |
| cpu-eigen | LSTM | fp32 | reused | 4.804 | 652.580 |
| cpu-eigen | CNN | fp32 | reused | 3.427 | 670.453 |
| cpu-eigen | Transformer | fp32 | reused | 3.479 | 648.294 |
| cpu-mkl | Dense | fp32 | reused | 9.863 | 661.609 |
| cpu-mkl | LSTM | fp32 | reused | 16.025 | 652.580 |
| cpu-mkl | CNN | fp32 | reused | 11.198 | 670.453 |
| cpu-mkl | Transformer | fp32 | reused | 11.325 | 648.294 |
| cuda | Dense | fp32 | reused | 787.974 | 1,538.936 |
| cuda | Dense | fp32 | empty | 872.797 | 1,531.146 |
| cuda | Dense | bf16 | reused | 914.584 | 1,589.930 |
| cuda | Dense | bf16 | empty | 910.337 | 1,633.804 |
| cuda | LSTM | fp32 | reused | 848.630 | 1,533.287 |
| cuda | LSTM | fp32 | empty | 875.581 | 1,553.056 |
| cuda | LSTM | bf16 | reused | 928.217 | 1,613.484 |
| cuda | LSTM | bf16 | empty | 868.236 | 1,620.082 |
| cuda | CNN | fp32 | reused | 928.344 | 1,591.334 |
| cuda | CNN | fp32 | empty | 947.681 | 1,625.366 |
| cuda | CNN | bf16 | reused | 1,061.431 | 1,732.044 |
| cuda | CNN | bf16 | empty | 1,047.034 | 1,732.080 |
| cuda | Transformer | fp32 | reused | 871.942 | 1,649.388 |
| cuda | Transformer | fp32 | empty | 855.075 | 1,645.411 |
| cuda | Transformer | bf16 | reused | 922.554 | 1,760.148 |
| cuda | Transformer | bf16 | empty | 944.072 | 1,757.436 |

Lower is better. These are local diagnostic medians, not reference-computer
performance claims. Repeat with controlled background activity and clocks,
and retain all launches, ranges and variation. Do not replace this boundary
with the older process-lifetime measurement.

The [publication review](publication-review.md) includes the ratios and the
[application guide](../APPLICATIONS.md) gives the current reproduction commands.
