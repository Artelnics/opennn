# Master release preparation

## Version and compatibility

Prepare this development line as **9.0.0**, rather than an 8.0.x patch. It moves
public headers into module directories and changes the model/dataset API from
the 8.x line. For example, the former `opennn/neural_network.h` is now
`opennn/neural_network/neural_network.h`. Downstream applications need a source
migration review. Existing serialized models need loading, inference, saving,
and reloading checks using representative real 8.x artifacts.

The current CMake version remains 8.0.1 until the release branch is reconciled;
no 9.0.0 tag or release has been published. `CHANGELOG.md` records the unreleased
hardening work. Do not interpret its entries as completed compatibility tests.

## Merge review

On 2026-09-07, `origin/master` was `efd566b38`. It contains 64 commits absent
from the development branch by ancestry. A read-only trial merge against
`dev` produced 99 conflict messages. Many involve files moved into module
directories or superseded build files, but those conflicts must be reviewed
before discarding historical changes.

The review must account for the master-only fixes in these areas:

- Genetic-algorithm ranking, elitism, and per-individual scaling restoration.
- Growing-neuron selection using best-epoch validation error.
- Image scaling and honoring dataset display settings.
- Text classification labels and maximum input sequence length.
- Exported JavaScript escaping and variable names.
- Response optimization with empty inputs and affine constraints.
- Forecasting timestamps, training, and binary model parameters.
- CUDA fallback when no usable device is present.

Use an isolated release checkout to reconcile these against their current
implementations. Preserve dataset files until the provenance/replacement review
required by `AGENTS.md` is complete. Re-run all release gates on the resolved
merge, not only on its `dev` parent.

## Publication gates

- All hosted CPU, CUDA compilation, and sanitizer jobs pass on the candidate.
- The Linux GPU runner completes both unit and integration suites.
- Targeted NVIDIA memory checks pass; exclusions and toolchain limitations are
  stated in the verification report.
- Neural Designer builds against the candidate and representative saved models
  retain their outputs through a load/save round trip.
- Public documentation, example-data provenance, migration notes, the CMake
  version, and the final changelog agree with the release contents.
- The resolved merge is reviewed before merging into `master` and tagging 9.0.0.

The current validation evidence is in `RELEASE_VERIFICATION.md`. This checklist
prepares the release review; it does not certify or publish a release.
