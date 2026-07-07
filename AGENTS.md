# OpenNN — instructions for Codex

See [README.md](README.md) for what this project is and generic build instructions.

## Build environment on this machine (Windows)

`cl.exe` and `ninja.exe` are not on PATH in a plain shell. Ninja ships bundled with
Visual Studio at:
`C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe`

To get a working MSVC environment (`cl`, `link`, `INCLUDE`/`LIB`), source one of:
- `C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat`
- `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`

### Existing build directories

| Dir | Generator | Config | Tests | Notes |
|---|---|---|---|---|
| `build` | Visual Studio 18 2026 | multi-config | OFF | needs `--config Release` on every `cmake --build` |
| `build-ninja` | Ninja | Release | ON | has a working `bin/run_tests.exe` already built; single-config, no `--config` flag needed |
| `build-fresh` | Ninja | Release | OFF | |
| `build-codex-tests` | Visual Studio | — | — | has a `RUN_TESTS` project but no `CMakeCache.txt` found; treat as possibly stale before relying on it |
| `build-cpu-audit`, `build-cuda-audit` | — | — | — | not inspected; audit/benchmark builds, check before assuming purpose |

Prefer `build-ninja` when you need both tests and a single-config build. The
`run-examples` skill assumes this directory.

## Project-local skills

`.agents/skills/` in this repo holds project-specific skills:
- `run-examples` — run the example matrix across CPU/GPU FP32/GPU BF16.
