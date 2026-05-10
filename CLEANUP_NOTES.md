# Cleanup Notes — Conservative Macro-Cleaning Pass

Findings that were **NOT changed** in this pass, with rationale.

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
