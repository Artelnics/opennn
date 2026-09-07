# Changelog

## 9.0.0 (unreleased candidate)

This development line is being prepared for a major release. It is not a
published release, and the verification report records the tested scope and
remaining limitations.

### Reconciliation and release preparation

- Reconcile master through `efd566b38` with the current module layout.
- Restore all-missing histogram bins and bounded random genetic initialization.
- Fix JavaScript feature identifiers, categorical controls and HTML labels;
  execute generated JavaScript in the export regression suite.
- Add migration notes, a dataset inventory and explicit provenance gaps.
- Preserve unported master examples in `examples/legacy_8`.
- Set CMake/package compatibility to major version 9. Neural Designer
  validation is deferred at the owner's request.

### Reliability and portability

- Order CUDA buffer copies and clears with pending computation.
- Release late static GPU buffers without accessing destroyed backend state.
- Release lazily initialized CUDA library handles before library shutdown.
- Preserve thread settings between optimizer tests and retained matrix rows
  in response-optimization test helpers.
- Build the vision stage table and output-window storage with Clang 17.
- Avoid undefined floating-point-to-integer conversion when histogram bin
  widths are zero or subnormal.
- Restore JSON's rounded maximum-integer sentinel without an out-of-range
  cast; reject nonfinite and larger values during integer conversion.
- Provide configurable library logging, including concurrent callback changes.

### Packaging and verification

- Support consuming the installed CMake package after moving its prefix,
  including the JPEG dependency.
- Propagate Clang static-library LTO requirements to downstream linkers,
  including Debug consumers of a Release installation.
- Restore hosted Windows/Linux CPU checks and Linux CUDA compilation.
- Add a Linux GPU runner and CPU address/undefined-behavior sanitizer CI.
- Run response-optimization integration scenarios in full verification.
- Require an actual CUDA build and device in the CUDA verification gate.
- Pin matching cuDNN runtime, headers, and development packages for hosted CI;
  install both dependencies used by Python export tests.

### Migration review

The development tree reorganizes public headers and changes model/dataset
interfaces compared with the published 8.x line. Existing applications,
need an explicit source-compatibility review and
saved-model round trips before release. A green development test suite alone
does not establish compatibility with historical model files. See MIGRATION.md;
Neural Designer validation is outside the current scope.
