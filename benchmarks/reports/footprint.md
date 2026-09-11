# Earlier footprint tests

**Status: historical observations with different scopes.**

The earlier footprint family measured baseline memory, process lifetime and
model export. Its original report and values remain in the
[archive](archive/2026-09-11/footprint.md).

These observations do not provide a current startup comparison. The old
OpenNN startup process used the GPU while the PyTorch process used the CPU,
and the reported wall time included process teardown. The new
[startup benchmark](startup.md) uses matching devices and stops at the first
completed prediction.

The old baseline-memory readings were single-process observations. They are
not peak training memory or a measure of maximum model capacity. Keep current
RSS, anonymous memory and GPU memory separate when repeating them.

The old export test wrote files but did not execute both exported models.
A new export claim needs a load-and-predict check, an output comparison and an
explicit list of supported architectures. Source export for some dense or
recurrent models does not establish source export for CNNs and Transformers.

Use the [publication review](publication-review.md) for the next release.
Do not average these older observations with the new runtime results.
