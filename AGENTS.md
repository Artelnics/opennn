# OpenNN — instructions for coding agents

See [README.md](README.md) for the project overview and first build, and
[DEVELOPMENT.md](DEVELOPMENT.md) for CMake options. Keep this file focused on repository-wide
engineering rules that are not tied to one workstation.

## Working branches

- Use the existing OpenNN checkout on `dev` for normal development. Do not create
  additional clones or worktrees unless the user explicitly requests them.
- Keep development and release preparation on `dev`. Merge into `master` only
  when the user explicitly decides the work is ready for release.
- Preserve uncommitted work and saved stashes when switching or consolidating
  branches. A folder cleanup does not authorize publishing a release.

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
  network code, training, model selection and evaluation must retain their
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

Before every commit, run the source checks that gate CI:

```bash
python tools/check_code_quality.py
python tools/check_architecture.py
python tools/check_dataset_manifest.py
```

`check_code_quality.py` is a ratchet: any metric above `CODE_QUALITY.json` fails
CI. When the commit legitimately grows the code (size metrics such as `nloc`,
`physical_lines`, `functions` or `files`), run
`python tools/check_code_quality.py --update` and include `CODE_QUALITY.json` in
the same commit. Do not update it to absorb worse quality metrics (long or complex
functions, duplication); simplify the code instead.

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
- Raw benchmark output belongs outside the checkout, in
  `../opennn-benchmark-results/` by default; `OPENNN_BENCH_RESULTS` overrides it.
  Only reviewed reports belong in `benchmarks/reports/`.

## Pending repository hygiene

The bundled datasets under `examples/` are intentionally retained for now.
Their provenance and unresolved licensing records are documented in
`DATASETS.md`, with indexed content inventories in `datasets.manifest.json`.
Do not remove them until each affected example has a reproducible replacement.
Stage reviewed asset changes before running `python tools/check_dataset_manifest.py`.
Bundled ZIPs preserve logical asset paths and bytes; the checker expands their
contents in memory. Update archives and loose files together, retaining source
notices. CMake unpacks active image datasets into the external build directory.
The `--release` check additionally requires every bundle's redistribution
clearance; do not mark an unknown source as cleared merely to pass that gate.

## Documentation

- Update the existing topic guides listed in `README.md`; consolidate overlapping
  explanations instead of adding a Markdown file for each task or audit.
- Keep release and verification procedures in `DEVELOPMENT.md`, migration in
  `CHANGELOG.md`, and current benchmark findings in `benchmarks/reports/README.md`.
- Link to an immutable Git revision for superseded reports. Keep raw evidence
  outside Git, and preserve dataset attribution notices and skill entry points.

- Keep the README focused on building, running the first example and locating
  the main guides. Put source maps and tool inventories in `DEVELOPMENT.md`.
- When changing example targets or dependencies, update the catalog in
  `examples/README.md`. Run data-dependent examples from the executable directory.
