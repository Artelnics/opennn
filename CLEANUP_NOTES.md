# Cleanup Notes — Conservative Macro-Cleaning Passes

Findings that were **NOT changed** in either pass, with rationale.

## Pass 1 — macro cleaning (dead-code pass)

## Probable bugs (behavior-changing fixes — out of scope)

### `opennn/layer.cpp:77-82`
```cpp
for (const auto& [shape, dtype] : specs)
{
    if (shape.empty()) { views.emplace_back(); continue; }
    if (!is_aligned(pointer))
        throw runtime_error(string("Layer::") + tag + ": unaligned memory in layer \"" + self.get_name() + "\"");
    views.emplace_back(pointer, shape, Type::FP32);
    pointer += get_aligned_size(shape.size());
}
```
The `dtype` element of the structured binding is extracted but never used; the `TensorView` is constructed with hardcoded `Type::FP32` regardless of what the spec actually requests. Looks like an oversight — likely should be `views.emplace_back(pointer, shape, dtype);`. **Skipped: replacing the hardcoded type with `dtype` is a behavior change (any non-FP32 spec would now route differently), which is forbidden in this pass.** Worth a separate, deliberate fix.

### `tests/learning_rate_algorithm_test.cpp:183`
```cpp
Tensor1 gradient = loss.calculate_numerical_gradient();
```
Calls a method that does not exist on `Loss` (no such method declared anywhere — verified via grep). The test file does not compile. **Skipped: this lives under `tests/` (outside the conservative scope of the library), and "fixing" it would require deciding whether to delete the test or implement the method.**

## Public/protected APIs flagged but left intact

### `opennn/loss.h:127`
```cpp
void print() const {}
```
Empty public method on `Loss`. No callers anywhere in the library or tests. **Skipped per rule: "Do not remove public or protected methods just because they appear unused."**

## Commented-out code

### `opennn/testing_analysis.cpp:1261-1289`
Several blocks of `// for (...)` and `// if (...)` style commented-out code. **Skipped per standing project convention: commented-out code is treated as WIP during refactoring; not deleted by automated cleanup.**

### `opennn/growing_neurons.cpp:92`
`//neural_network->print();` — single commented-out line. Same rationale.

## Items examined and confirmed live

The two false-positive candidates the first scan agent flagged were verified to be in active use:
- `configuration.cpp:63` (`compute_capability`) — read on line 64 (`bf16_capable = gpu && (compute_capability >= 80)`).
- `quasi_newton_method.cpp:383` (`armijo_constant`) — read on line 399 (`new_loss <= current_loss + armijo_constant * alpha * training_slope`).

All `static` and anonymous-namespace functions in cpp files were verified to have at least one in-file caller.

## Build verification

Per the standing project preference (`No PowerShell builds` — user builds locally), the cleanup branch was not built or test-run by this pass. The user should compile and run tests against `cleanup/conservative-macro-cleaning` before merging.

## Pass 2 — clarity / trivial-clutter cleaning

### Items examined and intentionally not changed

- `opennn/optimizer.cpp:599` — `const bool profile_this = false;` next to comment "Profiler disabled by default; flip to true to capture per-section timings." Intentional dev toggle; left as-is.

- `opennn/scaling.cpp:76-77`, `:88-89` — Single-use locals like `const float minimum = column_descriptives.minimum;`. A scan agent suggested inlining them, but doing so replaces a short name (`minimum`) with a longer expression (`column_descriptives.minimum`) in a math formula, which arguably hurts readability. Per rule "Do not inline variables if it makes debugging or readability worse", left as-is.

- `opennn/model_selection.cpp:77` — `const Index validation_samples_number = dataset->get_samples_number("Validation");` used once in the very next condition. The name documents intent (it's a sample count for the Validation role) and the savings from inlining are marginal; left as-is.

- `opennn/json.cpp:88` and `opennn/operators.cpp:2709` — `return bool_value ? 1 : 0;` patterns. These are explicit `bool → int` conversions; the ternary form makes the conversion unmistakable. Not "redundant" in the sense of dead code, just a stylistic choice. Left as-is.

- "// Set dataset stuff" / "// Initialize OldParameters" style comments in `genetic_algorithm.cpp`, `growing_inputs.cpp`, `quasi_newton_method.cpp` — these label multi-line code sections rather than restating individual statements. Per the "Keep comments that explain intent" rule, left as-is.

- Comment-only lines surveyed across 60+ files: no cases found of a comment that literally restated the next single statement (e.g., `i++; // increment i`). Codebase is genuinely free of that anti-pattern.

### False-positive candidates rejected during pass 2

- `opennn/layer.cpp:77` — A scan agent again flagged `dtype` from a structured binding as "unused"; this is the same probable-bug case already in Pass 1 notes. Behavior change to fix; not touched here.

### Categories where pass 2 found nothing actionable

- Dead `else` after `return`/`throw`/`break`/`continue` — none found.
- Same null-check repeated in one function — none found.
- Same condition in consecutive `if` statements — none found.
- Same assignment in both branches of an `if`/`else` — none found.
- Empty catch blocks — none found.
- Redundant `static_cast<T>(x)` where `x` is already type `T` — none found via signature-only inspection.
- Repeated identical assignments inside a single function — none found beyond the one Pass 2 fixed (`Loss::set_normalization_coefficient`).

The codebase is in good shape for these categories; conservative cleanup yields diminishing returns after the first pass.
