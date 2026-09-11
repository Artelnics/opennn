# Website consolidation notes

Reviewed on 11 September 2026 against the indexed copy of the
[benchmark index](https://www.opennn.net/benchmarks/). A direct request returned
a browser-verification page, so this review does not claim a fresh authenticated
snapshot of the live WordPress content. No website content was changed.

The index currently points to several generations of tests. Its RTX 4080
articles and the repository's RTX 5070 Ti reports describe different result
sets. The new laptop measurements form a third set. Dates, hardware, model
geometry and framework versions must stay attached to each result.

| Existing topic | Treatment in the next release |
|---|---|
| Framework memory before useful work | Keep as a dated baseline study; do not mix it with training or inference peaks. |
| Installation requirements | Combine with deployment, using actual package and native-library inventories. |
| Library source size | Recount both pinned source trees with one rule. |
| Iris application length | Keep the same Iris task or rename the replacement task; do not silently substitute a different example. |
| Dataset capacity | Keep separate from peak memory; retain the memory cap and exact loading method. The old pandas workflow does not establish a universal PyTorch limit. |
| GPU dense learning speed | Replace with a reviewed paired result for the stated HIGGS model. |
| CPU dense learning speed | Replace with the current CPU result after the repeat and provenance checks. |
| GPU translation-model learning speed | Resolve masks, dropout and loss handling before replacing its result. |
| GPU image-model learning speed | Keep the old 32-pixel CIFAR task distinct from the 224-pixel ImageNet task. |
| Windows execution | Treat as a separate operating-system test; recheck capabilities and versions before updating claims. |
| Precision comparison | Use paired FP32/BF16 tests on the same model and verify the precision actually used. |
| Regression quality | Retain the Rosenbrock study as a dated separate task; it cannot fill the four new quality rows. |
| GPU application disk size | Consolidate with the other deployment configurations and verify the target bundle. |
| CPU application disk size | Show Eigen and MKL + oneDNN configurations separately. |
| First-use delay | Replace with a stable launch-to-first-prediction result. The old process-lifetime metric is different. |
| Standalone export | Keep as a capability and correctness test; execute the exported artifact and state the supported model families. |
| GPU translation-model prediction speed | Resolve the model differences and state whether the timing includes input handling. |
| GPU dense prediction speed | Retain batch size and the selected stable runtime configuration. |
| CPU dense prediction speed | Keep the thread count, affinity and memory definition visible. |
| GPU image-model prediction speed | State whether the test replays a resident batch or reads and processes images. |

Add the LSTM comparisons, CPU CNN/Transformer results, separate energy tables,
and the four-task quality table when their evidence is complete. Do not imply
that the new PyTorch-only release also remeasured TensorFlow or the Qwen runtime
comparisons.

The index should link to a concise result page and an evidence download for each
topic. Use the [website draft](website-draft.md) for the page structure and the
[publication review](../reports/publication-review.md) for the unresolved work.
