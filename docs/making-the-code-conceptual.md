# Making the code conceptual

A standing task prompt. Hand this to an agent as-is, or paste the "Your task"
section into a fresh session.

---

## Your task

Find places in OpenNN where the code makes the reader reconstruct a concept that
the code should have named, give that concept a name, and apply it everywhere it
occurs. Report what you found **before** you implement it. Do not lose
performance. Do not add comments.

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

1. **Scan, don't guess.** Write a throwaway script that counts instances across
   the whole tree. Rank candidates by how many sites they touch. A hand-picked
   example is usually the weakest case.
2. **Verify the premise before designing.** Read what the code actually branches
   on and what the lifetimes are. Several obvious-looking consolidations in this
   repo are wrong, and the audit is what catches them.
3. **Report findings and wait.** Give counts, file references, the recommendation,
   and what you deliberately excluded and why.
4. **Implement mechanically.** Prefer a script over hand edits for a change with
   many sites, and assert that each replacement matched exactly once.
5. **Verify against the baseline.** Both builds, both suites. See below.
6. **Commit with the reasoning**, including what you chose *not* to merge.

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
- **Current baseline: CPU 976 passed, GPU 1094 passed, zero failures.** Update
  this line if the count legitimately changes, and say why in the commit.

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
| `AffineMap` | `core/scaling.h` |
| `ConfusionCell` | `testing_analysis/testing_analysis.h` |
| Slot virtuals speaking slot ids, not spec indices | `neural_network/layers/*.h` |

Deliberately **not** merged, with reasons in the commit messages: `SampleRole`
and `FillMode` into `ForwardPropagationMode`; the `Rung` family in
`core/device_backend.h` (different members, and the mechanism is already shared
through `template<typename Rung> rung()`).

## Open candidates

- `multiply(a, bool, b, bool, out, alpha, beta)` in `core/tensor_operations.h` —
  two adjacent unnamed transpose flags, 16 unreadable call sites. Wants
  `Transpose { No, Yes }`, the way every BLAS names it.
- 29 further declarations taking a bare unnamed `bool`. Most only need the
  parameter named; `get_batches`'s shuffle flag and `Layer(LayerType, bool
  trainable)` are the two passed as literals often enough to earn an enum.
- Derived datasets hide the role-taking overloads of `get_samples_number` and
  friends, which is why two tests carry a `Dataset& base_dataset` alias to work
  around it. A `using` declaration per derived class fixes it. Note that the
  string overloads themselves must stay: Neural Designer calls them in 236 places.
