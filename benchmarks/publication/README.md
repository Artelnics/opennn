# Preparing benchmark results for the website

The next release compares OpenNN C++ with **PyTorch's Python API**. The current
evidence is consolidated, but the comparison is not yet ready to publish.
Read the [publication review](../reports/publication-review.md) for the values
and the [website draft](website-draft.md) for the proposed page.

## One source for every number

`selection.json` identifies the observations already cited in the previous
reports and the complete Python startup/deployment comparison. Each source has
a SHA-256 hash. Selecting a record means it is being reviewed; it does not mean
it passed the publication requirements. We do not select the fastest run from
a collection of attempts.

Generate a new review directory with:

```powershell
python benchmarks/tools/consolidate_results.py --out benchmarks/results/scratch/publication-review-new
```

The tool checks the selected source hashes, recalculates performance statistics
from launches, checks startup medians against all timed processes, and checks
deployment totals against the file inventories. It writes a readable report,
an HTML preview, data tables, pending checks and a catalog of all result files.
It never changes the inputs or publishes anything. Its output is a review of
this historical selection, not an automatic certification tool.

The catalog gives identical files one content entry and retains all their
paths. It also identifies artifact types and copies the original status and
session labels; those labels are not new approvals. `provenance.json` collects
the selected source settings, software versions and recorded commands without
filling missing information by inference. The September 11 relocation manifest maps each old path to its archive
location. Copies and failed runs remain available; no observation was deleted
to improve a result.

## Complete the measurements

| Topic | Required scope |
|---|---|
| Throughput | Dense, LSTM, CNN and Transformer; training and inference; CPU and GPU |
| Memory | The same configurations, with CPU anonymous memory and GPU memory labelled separately |
| Energy | The same configurations where the instrument is available; CPU package and GPU board energy separately |
| Startup | All four small applications; CPU Eigen, CPU MKL + oneDNN, and GPU FP32/BF16 with saved/empty tuning caches |
| Deployment | All four small applications and each backend; include native runtime files and the Python runtime |
| Prediction quality | Four real-data tasks, held-out evaluation and five independent seeds |
| Source size | Both repositories at named revisions, counted with the same file and comment rules |
| Application size | Two applications that complete the same task, with imports and setup treated consistently |
| Dependencies | Installed Python packages and native runtime libraries counted separately |

Choose batch sizes, precision, training budgets, quality tolerances and stable
runtime options before collecting the final session. State whether inference
replays a resident batch or reads a dataset. State whether training includes
input preparation or transfer. Both engines must do equivalent work inside the
named measurement boundary.

Record the release commit, input hashes, compiler, build settings, hardware,
operating system, driver and library versions. Different machines and software
stacks form separate result sets. Source counts and file sizes do not require
locked clocks; timing and energy measurements require the applicable machine
controls in [PROTOCOL.md](../PROTOCOL.md).

Use three independent performance rounds and retain each launch. The primary
throughput and energy values are medians; memory uses the highest observed
peak. Include minimum, maximum, sample count and variation in the downloadable
data. Report missing counters as "Not measured", never zero. Do not discard a
slow run without a recorded failure that justifies excluding it.

Set an acceptable quality difference before the final quality runs. Compare
quality within each task, using mean and sample standard deviation across
seeds. Similar averages alone do not establish equivalence. A smoke test does
not establish trained-model quality, and a throughput test does not establish
time or energy to a target quality.

Check each deployment bundle on a clean target. A trace shows which files a
particular run used; it does not by itself prove that every required file was
collected. Show the normal Python installation as such. Do not describe it as
PyTorch's smallest possible deployment.

## Write the results in plain language

Use a short opening sentence, a table or chart, and a short explanation of what
the result means. Put the full settings and downloadable evidence below them.
Prefer a subject, a verb and an object. Explain a term the first time it appears.

| Metric | Caption |
|---|---|
| Throughput | OpenNN throughput ÷ PyTorch throughput. Higher is better. |
| Memory | OpenNN memory ÷ PyTorch memory × 100. Lower is better. |
| Energy | OpenNN energy ÷ PyTorch energy × 100. Lower is better. |
| Startup | OpenNN startup time ÷ PyTorch startup time × 100. Lower is better. |
| Deployment | OpenNN deployment size ÷ PyTorch deployment size × 100. Lower is better. |
| Source lines | OpenNN lines of code ÷ PyTorch lines of code × 100. Lower is better. |

Sixty percent of PyTorch's memory means forty percent less memory. Twice the
throughput does not mean two hundred percent faster. Use the full values to
calculate a ratio, then round it for display.

Keep CPU and GPU summaries separate. If a complete, predefined group merits a
summary, use the geometric mean of its paired ratios and name the group and
number of comparisons. Do not pool quality metrics, hardware generations,
unmatched sessions, startup timings and throughput into one number.

Describe the tested result and its scope. Say that shorter training iterations
can let a company test more configurations, rather than claim that a model will
reach a target accuracy sooner without measuring that. Component energy can
support an energy-efficiency statement; it does not directly measure a bill.
Source size alone does not establish usability or feature parity.

## Release order

1. Resolve the issues in the publication review and save the new raw results.
2. Check every value, ratio, unit, input record and measurement requirement.
3. Update the reviewed reports and replace the pinned selection deliberately.
4. Prepare the public raw-data download and test the reproduction commands from a clean environment.
5. Replace the relevant website articles and their index cards together.

A partial release may publish a clearly named subset that is ready. It must
show which comparisons remain unmeasured and omit claims about the full matrix.
The present consolidation does not change the live website.
