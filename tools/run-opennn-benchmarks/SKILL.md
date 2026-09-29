---
name: run-opennn-benchmarks
description: Measure OpenNN against PyTorch with the benchmark families in benchmarks/, following benchmarks/PROTOCOL.md, and compare a code change before and after with compare.py. Use for throughput, memory and energy measurements, not for unit tests or the example matrix.
---

# Run the OpenNN benchmarks

[`benchmarks/PROTOCOL.md`](../../benchmarks/PROTOCOL.md) is the only source of
the measurement rules. This skill is the procedure; it does not restate
thresholds. If this skill and the protocol disagree, follow the protocol. If the
code (`benchmarks/run.py`, `benchmarks/tools/common.py`) disagrees with the
protocol, stop and tell the user instead of choosing one.

## Prepare

1. Read `benchmarks/PROTOCOL.md`, `benchmarks/README.md` and the output of
   `python benchmarks/run.py --help`. Note the sections you apply.
2. Agree the cell with the user: family, mode, device, precision and batch. The
   standard families are `dense`, `cnn`, `transformer` and `lstm`. For
   `startup`, `deployment` and `quality`, follow their protocol sections (§12
   and §13) instead of this procedure.
3. Record `git status --short`. A tree with uncommitted changes produces
   diagnostic results only (§2, §10). Never stash, reset or commit to make the
   tree clean without the user's permission.
4. Build in Release outside the checkout with `-DOpenNN_BUILD_BENCHMARKS=ON`
   (§3). Use the target `<family>_opennn`, or `benchmarks` for all of them.
5. Set `OPENNN_BENCH_DATA` and `OPENNN_BIN` (or `OPENNN_<FAMILY>_OPENNN_BIN`)
   to the driver of the family. Check the path exists; `run.py` does not
   substitute another binary.
6. Run `python benchmarks/prepare.py <family>` if the data is missing. Never
   copy data into the repository.

## Control the machine (§7, §8)

- Close other GPU and CPU work and ask the user to keep the machine idle.
- For CUDA, lock the clocks before measuring and restore them afterwards, even
  after a failure: `benchmarks/tools/gpu_clocks.sh` on Linux; on Windows
  `nvidia-smi` plus `OPENNN_BENCH_CLOCKS_LOCKED=1`. Clock changes need the
  user's consent and usually administrator rights.
- Set `OPENNN_BENCH_SESSION` to one value for every run you intend to compare.

## Measure a cell (§9)

```sh
python benchmarks/run.py --family dense --mode train --device cuda --precision bf16 --batch 8192 --rounds 3
```

Keep the default rounds unless the user asks otherwise. A comma list of batch
sizes gives a curve; `N:OOM` looks for the capacity limit. Exit status 3 means
the capacity limit is unconfirmed, not a pass.

## Compare before and after a change

1. Build the baseline binary before the change is applied, in its own build
   directory. If the change already exists, ask the user how to obtain the
   baseline; do not create worktrees or checkouts without permission.
2. Build the candidate in a second build directory.
3. Measure the same cell with both, same session, same machine state and
   alternating order when you repeat.
4. Run `python benchmarks/compare.py baseline.json candidate.json`. It fails on
   a throughput or memory change beyond its tolerance, or a smaller maximum
   batch.
5. Results from an uncommitted tree are diagnostic. Say so; a regression
   verdict that must count needs committed code and a clean rerun.

## Report

For every cell give: command, commit and tree state, device and precision,
throughput, peak memory and energy per engine, the `shape_gate` and
`quality_gate` results, the artifact path, and whether it is valid or
diagnostic. A file in `scratch/` is never a valid result (§10). Report
failures and missing energy readings as they are; never fill them in.

Before calling a session complete, walk through the checklist in §11 and state
which items could not be satisfied.
