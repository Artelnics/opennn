# Making the code conceptual

A standing task prompt. Hand this to an agent as-is, or paste the "Your task"
section into a fresh session. It asks for a **plan**, not an implementation.

---

## Your task

Find places in OpenNN where the naming and the reality have come apart, and
**produce a plan** for closing the gap. That runs in two directions:

- **A concept with no name.** The code already does something coherent and makes
  every reader reconstruct it. Name it.
- **A name with no concept.** A struct, class, or enum whose name claims an idea
  it does not deliver — a bag of unrelated fields, one name covering several
  ideas, or several names covering one. Refactor it into what it actually is.

The second direction matters more, and is the one usually skipped. A missing
concept costs the reader work. A false one costs them trust: they take the name
at face value, reason from it, and are wrong. For an agent this is worse still,
because a name is most of what it has to go on.

The goal is that a human or an agent arriving cold can read a file and act on it
correctly. Do not lose performance. Do not add comments.

**Deliver a plan, not an implementation.** Stop when the plan is written and wait
for approval. Do not edit source files, do not start with "the small obvious one
first", and do not implement a finding merely because it looks safe. The audit is
the deliverable; the edit is a separate, later decision that is not yours to make.

This is not caution for its own sake. Several findings in this repo look
obviously correct and are wrong — the rejected list below is longer than the
accepted one — and the cost of catching that in a plan is a paragraph, where the
cost of catching it after a tree-wide mechanical rewrite is a day. The plan is
also where scope gets decided: a change touching 175 call sites is a different
conversation from one touching six, and only the reader of the plan can say which
is wanted right now.

---

## Why this instead of comments

This library carries no explanatory comments, on purpose. It is too large and
changes too fast for comments to stay true, and every past attempt left stale
text that misled the next reader. The replacement is not better comments — it is
code whose types and names already say what a comment would have said.

So the unit of work here is a **concept**: something the code was already doing
without a name, given one.

A comment is only ever appropriate for a fact the code genuinely cannot carry —
a defect worked around, an invariant enforced elsewhere, a non-obvious *why*.
One line, never a paragraph.

---

## What counts as a concept

The test is not "are these things used together." It is:

> **Do these members share a lifetime and change together, and does naming them
> let a caller stop knowing something it currently has to know?**

Both halves matter. Real examples from this repo:

**Accepted — `FeatureSelection { inputs, decoder, targets }`**
`Batch::fill` took four unnamed `const vector<Index>&` in a row: rows first, then
three columns. Nineteen signatures repeated the group. The three columns are one
question — which columns feed each role — and they are constant for a whole run.
Naming them collapsed nineteen signatures and let callers fetch once instead of
threading three vectors.

**Rejected — the same struct with `samples` added.**
That was the first proposal and it was wrong. Samples change on *every call*
where the three feature vectors are fixed for the run. Different lifetime,
therefore not the same concept. This is the most common way to get this wrong.

**Accepted — `ForwardPropagationMode` replacing a bare `bool`.**
The enum already existed and labelled the arena; the pass that ran on that arena
took an unnamed `bool`. 28 overrides and 175 call sites. The concept was already
in the codebase, just not used where it belonged.

**Rejected — merging `FillMode` into it.**
The member lists overlap and it looked like a duplicate. It is not: `FillMode`
branches on two independent things (scale, augment) and needs a `Validation`
state the arena has no meaning for. Verify what an enum *branches on* before
merging it with one that merely shares member names.

**Rejected — a 14-field struct bundling everything a matmul call touches.**
Grouping by "these are all passed to the same function" produces a bag, not a
concept. Structs must represent one idea.

**Rejected — a struct wrapping two values that always travel together but have
no name anyone would recognise.** Pairs are not automatically concepts. If you
cannot say what the thing *is* in three words, it is not one.

---

## What counts as a false concept

The same test, run backwards. A type is not one concept when its members do
**not** share a lifetime, or when disjoint parts of the code use disjoint subsets
of it. Ask what each field is for and who reads it; if the answers form separate
clusters, the type is several concepts wearing one name.

**Field count is not the test.** `ForwardPropagation` has 45 fields and is a
genuine concept — one arena for one pass — and a proposal to split it was
examined and rejected on those grounds. A large type that is really one thing is
fine. A small type that is really two is not.

The signals worth scanning for:

- **A name that describes nothing.** `...Data`, `...Info`, `...Context`,
  `...Params`, `...State`, `...Manager`, `...Helper`. These are not banned words,
  but each one is a place where somebody could not say what the thing was. Check
  whether they could have.
- **Disjoint readers.** Different call sites, or different subclasses, touching
  non-overlapping subsets of the fields.
- **Several names for one idea**, or one name spanning several.
- **A name that was true once.** The type still carries the name of what it did
  before the code moved on.

Two real ones from this repo, both now fixed:

**Fixed — optimizer workspace versus algorithm state —
[optimizer.h:32](../opennn/training_strategy/optimizer.h#L32).**
The former `OptimizerData` mixed a shared aligned tensor workspace with fields
used by only one algorithm. It now contains only `data`, `views`, and `set()` and
lives with `Optimizer`. Adam owns its update step, SGD owns its current learning
rate, quasi-Newton owns a `LineSearchState`, and Levenberg-Marquardt owns its
damping parameter and candidate parameters. Each state resets at the same
training-run boundary as before.

**Fixed — augmentation policy versus sampled transform —
[image_dataset.h:19](../opennn/dataset/image_dataset.h#L19),
[yolo_dataset.h:132](../opennn/dataset/yolo_dataset.h#L132),
[yolo_dataset.cpp:399](../opennn/dataset/yolo_dataset.cpp#L399).**
The former `AugmentationSettings` and `AugmentationConfig` were configured,
dataset-specific policies; the former `AugmentationParams` was the random
transform realised for one image. They are now `ImageAugmentationPolicy`,
`YoloDataset::AugmentationPolicy`, and the file-local `AugmentationTransform`.
The two lifetimes remain separate and their names now expose the distinction.

### Fixing one

In rough order of preference:

1. **Rename**, if the members really are one concept and only the name is wrong.
   Cheapest, and often enough.
2. **Split by cohesion** into the clusters the readers already imply. Each part
   must survive the concept test on its own, or you have only made more bags.
3. **Dissolve** it back into parameters, if it exists only because somebody
   wanted a shorter signature.

Splitting changes far more code than naming something new does, and it moves
call sites that were not broken. Say so in the plan, with the count. Prefer the
smallest change that makes the name true.

---

## Patterns worth scanning for

Prefer general patterns over one-off issues. A finding that touches one call site
is a bug report; a finding that touches fifty is worth this process.

- **Flag arguments → named enums.** Literal `true`/`false` at a call site is
  unreadable. Grep for `, true)` and `, false)`.
- **Repeated parameter groups → named struct.** Look for the same sequence of
  types recurring across many signatures, especially unnamed ones.
- **The same concept expressed twice.** Two enums with identical members; an enum
  and a parallel `bool`; hand-written translation between two types that mean the
  same thing.
- **Sentinel values → `optional` or a named constant.** `SIZE_MAX` and `-1`
  standing in for "none".
- **Index spaces that differ by a constant.** If callers write `size_t(Name) - 1`
  by hand, the conversion belongs at one boundary, not at every call site.
- **Out-parameters that could be return values.**

---

## Performance is a hard constraint

A concept that costs throughput is not an improvement. The rules:

- Aggregates in any per-batch, per-layer, or per-element path are passed by
  `const&`. Never by value. Check that you have not introduced a `vector` copy
  per call.
- Build the aggregate where its members are already built — once, outside the
  loop — and thread the reference. Never construct it inside a hot loop.
- Enums replacing bools, `inline` predicate functions, and type aliases are free.
  Prefer them.
- Never introduce virtual dispatch, `std::function`, or a heap allocation into a
  path that did not have one.
- Do not turn a compile-time branch into a runtime one.
- If the change touches a hot path, verify with a benchmark, not only with tests.
  Tests prove correctness, not throughput.

---

## Method

Steps 1 to 3 are your task. Steps 4 onward happen only after the plan is approved.

1. **Scan, don't guess.** Write a throwaway script that counts instances across
   the whole tree. Rank candidates by how many sites they touch. A hand-picked
   example is usually the weakest case. Reading files to confirm what a scan
   found is expected; editing them is not.
2. **Verify the premise before designing.** Read what the code actually branches
   on and what the lifetimes are. Several obvious-looking consolidations in this
   repo are wrong, and this is the step that catches them. A plan that skipped
   this is worth nothing, because the pattern that survives a count is not
   necessarily the pattern that survives a reading.
3. **Write the plan. Then stop.**

Then, and only if asked:

4. **Implement mechanically.** Prefer a script over hand edits for a change with
   many sites, and assert that each replacement matched exactly once.
5. **Verify against the baseline.** Both builds, both suites. See below.
6. **Commit with the reasoning**, including what you chose *not* to merge.

### What the plan must contain

- **The finding**, as a count: how many declarations, how many call sites, how
  many files. Numbers decide whether it is a pattern or an anecdote.
- **File and line references** for a representative sample, enough that the
  reader can check your reading without repeating your scan.
- **The concept you propose**, written out as the actual type or signature it
  would become — not a description of one.
- **A before and after** at one real call site, copied from the tree.
- **The blast radius**: which files change, whether public signatures move,
  whether Neural Designer is affected, whether tests need updating.
- **The performance argument**: which of the rules above the change touches, and
  why it costs nothing. If it touches a hot path, say which benchmark would show
  it.
- **What you rejected and why.** This is not padding. A plan that only lists what
  to do hides the judgement that matters most, and the reasons for rejection are
  what stop the next agent re-proposing the same wrong thing.
- **A recommendation.** Rank the findings and say which one you would do. Do not
  present a menu and leave the choice unmade.

---

## Before you finish

- **Check Neural Designer.** `C:\Users\Roberto\OneDrive - artelnics.com\neuraldesigner`
  links against this library. Before deleting anything, and before changing any
  public signature, grep ND for it. Symbols that look dead from inside this repo
  are routinely live there. If you change a signature ND uses, update ND's call
  sites too, and say so — do not commit in that repo without being asked.
- **Both builds must pass, both suites at baseline.** `build-cpu-verification`
  and `build-resnet-capacity`. The CUDA build compiles `.cu` paths the CPU build
  never sees, so a green CPU build proves little on its own.
- **Current baseline: CPU 1017 tests (989 passed / 28 skipped), GPU 1122 tests
  (1116 passed / 6 skipped), zero failures, one disabled test.** If Python is
  unavailable, the two expression-execution checks skip instead: CPU 987 / 30
  and GPU 1114 / 8. Update this line if the count legitimately changes, and say
  why in the commit.

Build recipe, environment traps (the cuDNN configure flags and the `PATH`
ordering hazard that kills the GPU suite mid-run), and how to create the two
build directories are in [../AGENTS.md](../AGENTS.md). Read it first.

---

## Already done — do not redo

| Concept | Where |
| --- | --- |
| `FeatureSelection` | `dataset/batch.h` |
| `ForwardPropagationMode` on the pass, replacing `bool is_training` | `neural_network/forward_propagation.h` |
| `DetectionClassActivation` absorbing `YoloNetwork::ClassActivation` | `neural_network/detection_head.h` |
| `DeviceResidency`, `ParameterStorage` | `neural_network/neural_network.h` |
| Named Boolean declaration intent (`has_header`, `is_training`, `has_validation`, and related parameters) | dataset, export, optimizer, and operator interfaces |
| `AffineMap` | `core/scaling.h` |
| `ConfusionCell` | `testing_analysis/testing_analysis.h` |
| Slot virtuals speaking slot ids, not spec indices | `neural_network/layers/*.h` |
| Augmentation policies separated from sampled transforms | `dataset/image_dataset.h`, `dataset/yolo_dataset.h`, `dataset/yolo_dataset.cpp` |
| `Transpose { No, Yes }` replacing the two adjacent `multiply` flags | `core/tensor_operations.h` and 16 call sites |
| Dataset role-count overloads remain visible on derived datasets | `dataset/tabular_dataset.h`, `dataset/yolo_dataset.h` |
| Optimizer workspace separated from algorithm-specific state | `training_strategy/optimizer.h`, four optimizer implementations |
| `Shuffle { No, Yes }` for batch ordering and `Trainability { Frozen, Trainable }` for layer construction | `dataset/dataset.h`, `neural_network/layers/layer.h` |
| `BatchNormalization { No, Yes }` replacing constructor and vision-builder flags | `neural_network/layers/layer.h`, dense and convolutional layers |
| `CausalMask { No, Yes }` replacing attention setup flags | `neural_network/operators/attention_operator.h`, multi-head attention |
| `ColumnContiguity { Unknown, NonContiguous, Contiguous }` replacing the dataset fill-path `-1`/`0`/`1` sentinel and `optional<bool>` bridge | `core/tensor_types.h`, `dataset/batch.h`, dataset fill interfaces |
| Optional operator-owned slot assignments replacing five `SIZE_MAX` configuration sentinels, with active-path validation for required mask and scratch slots | activation, dropout, combination, and C2PSA operators |

Deliberately **not** merged, with reasons in the commit messages: `SampleRole`
and `FillMode` into `ForwardPropagationMode`; the `Rung` family in
`core/device_backend.h` (different members, and the mechanism is already shared
through `template<typename Rung> rung()`); remaining `SIZE_MAX` and negative
sentinels (they represent graph recomputation and consumer-edge absence,
graph-relative indices, cache invalidation, file descriptors, and local search
state).

## Open candidates

These are leads, not a work queue. Each still needs the audit in step 2 before it
becomes a plan, and the counts below are from an earlier scan — re-derive them.

- Remaining bare `bool` occurrences are type-only callback signatures,
  Boolean container element types, or generated CUDA-stub parameter lists;
  they do not expose nameable function parameters.
