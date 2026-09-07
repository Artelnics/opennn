# OpenNN — instructions for coding agents

See [README.md](README.md) for the project overview, requirements, build
instructions and public CMake options. Keep this file focused on repository-wide
engineering rules that are not tied to one workstation.

## Compatibility and scope

- Preserve the public API and serialized model compatibility unless the task
  explicitly authorizes a breaking change.
- Neural Designer links against OpenNN and uses symbols that may appear unused
  inside this repository. Do not remove public or exported code based only on
  repository-local call sites.
- Preserve unrelated working-tree changes. Build products, downloaded models,
  generated data and raw benchmark results must remain outside Git.

## Code organization

- Follow neighboring files for naming, include order and class layout.
- Keep reusable tensor and device primitives in `opennn/core/`; datasets,
  network code, training, model selection and testing analysis must retain their
  existing dependency direction.
- Validate structural changes on both CPU and CUDA when they touch shared code.
  Some qualifications, includes and data-member ordering are intentionally
  significant even when a local edit suggests otherwise.

## Verification

Use the repository wrappers for routine verification. They create persistent
build trees outside the checkout and support focused GoogleTest filters:

```powershell
.\tools\verify.ps1 quick -Filter 'Dense.*:DenseNoBiasTest.*'
.\tools\verify.ps1 quick -Backend cuda -Filter '*Gpu*:*CUDA*'
.\tools\verify.ps1 full
```

```bash
./tools/verify.sh quick --filter 'Dense.*:DenseNoBiasTest.*'
./tools/verify.sh quick --backend cuda --filter '*Gpu*:*CUDA*'
./tools/verify.sh full
```

Use focused checks while editing and `full` as the final gate for a completed
batch. A library change is not complete until the relevant CPU and CUDA suites
pass, or an unavailable backend is reported clearly.

For non-standard CUDA installations, configure the wrappers through
`OPENNN_CUDA_ARCHITECTURES`, `OPENNN_CUDNN_INCLUDE_DIR` and
`OPENNN_CUDNN_LIBRARY`. Do not add workstation-specific paths to repository
files.

## Examples and benchmarks

- To run every example across the supported device/precision matrix, follow
  [tools/run-opennn-examples/SKILL.md](tools/run-opennn-examples/SKILL.md).
- Benchmark usage and the measurement contract live in
  [benchmarks/README.md](benchmarks/README.md) and
  [benchmarks/PROTOCOL.md](benchmarks/PROTOCOL.md).
- Raw benchmark output belongs in `benchmarks/results/`, which is ignored.
  Only reviewed reports belong in `benchmarks/reports/`.

## Pending repository hygiene

The bundled datasets under `examples/` are intentionally retained for now.
Their provenance, licensing and possible replacement with verified download
manifests still need a dedicated review. Do not remove them until each affected
example has a reproducible replacement.
