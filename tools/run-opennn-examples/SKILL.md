---
name: run-opennn-examples
description: Build and run every CMake-registered OpenNN example across the explicit CPU and CUDA precision matrix, preserving the working tree and reporting every pass, failure, unsupported cell and missing prerequisite. Use for full end-to-end example validation, not for unit tests or benchmarks.
---

# Run the OpenNN example matrix

Exercise every target registered through `opennn_example(...)` in
`examples/CMakeLists.txt`. Treat that file as the source of truth; do not keep a
separate hard-coded target list in this skill.

## Matrix

Use the current two-argument configuration API from
`opennn/core/configuration.h`:

| Cell | Configuration |
| --- | --- |
| CPU FP32 | `Configuration::instance().set(Device::CPU, Type::FP32);` |
| CUDA FP32 | `Configuration::instance().set(Device::CUDA, Type::FP32);` |
| CUDA BF16 | `Configuration::instance().set(Device::CUDA, Type::BF16);` |
| CUDA INT8 | `Configuration::instance().set(Device::CUDA, Type::INT8);` |

Do not pass a third inference-precision argument; it no longer exists. Do not
add `Auto` rows to the comparison: `Device::Auto` and `Type::Auto` resolve to
one of the explicit cells according to available hardware. CPU with BF16 or
INT8 is invalid by design. CUDA BF16 and INT8 require compute capability 8.0 or
newer.

## Prepare

1. Read `AGENTS.md`, `README.md`, `examples/CMakeLists.txt` and the current
   `Configuration` declaration and implementation. The code overrides this
   document if the API changes again.
2. Record `git status --short`. Never overwrite or restore changes that were
   already present.
3. Discover the example targets from CMake and inspect each entry point for
   required arguments, model downloads, interactive input and an existing
   `Configuration::instance().set(...)` call.
4. Use separate Release build directories outside the checkout for CPU and
   CUDA. Configure with examples enabled and benchmarks/tests disabled unless
   the user requested otherwise.
5. Confirm required datasets and model assets before starting a long cell.
   Missing licensed or externally downloaded data is `BLOCKED`, not a code
   failure and not permission to download or publish it.

## Apply a matrix cell

Set the configuration before constructing a dataset, network, model or session.
For an example that already calls `Configuration::set`, temporarily replace
that complete call. For an example without a call, temporarily add the direct
`opennn/core/configuration.h` include and one call at the start of its `try`
block. The `blank` target does not use OpenNN; build and run it once and report
the four matrix cells as not applicable.

Before editing, save the exact original contents outside the repository. Restore
them in a `finally`-style cleanup path after every example, including build
failure, timeout or interruption. Verify restoration by comparing the file with
the saved original; do not use `git checkout`, `git restore` or another command
that could discard the user's changes.

Build only the target being exercised. Run it from the build's runtime-output
directory so paths such as `../data/<example>` resolve to the data copied by
CMake. Supply deterministic, minimal input to interactive programs; for Qwen,
pipe a prompt followed by `exit` and use a user-provided/prepared model
directory. Apply a finite timeout suited to the workload.

## Classify results

Every target must have a result for every matrix column:

- `PASS`: exit code zero and the expected final output or metric is present.
- `FAIL`: it built and ran with its prerequisites, but compilation, execution,
  numeric validity or output validation failed.
- `UNSUPPORTED`: the code explicitly rejects that device/type combination.
- `BLOCKED`: required hardware, data, model assets or external tooling is
  unavailable.
- `N/A`: the target does not instantiate OpenNN, as with `blank`.

Do not silently convert failures to unsupported results. Record the command,
exit code, elapsed time and a short diagnostic for every non-pass cell. Capture
the example's meaningful final metric when it has one, and reject NaN/Inf or an
unexpectedly empty result.

## Finish

Restore every temporarily edited source and verify that `git status --short`
differs from the initial snapshot only by artifacts explicitly requested by the
user. Report a table with one row per CMake example and these columns:

```text
| Example | CPU FP32 | CUDA FP32 | CUDA BF16 | CUDA INT8 |
```

Follow it with prerequisites, failures, unsupported paths, per-example elapsed
time and the exact build configuration. A partial matrix is useful, but never
describe it as complete.
