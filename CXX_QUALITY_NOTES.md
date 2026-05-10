# C++ Quality Pass — Notes

## Baseline-build constraint

The standing project preference (codified in the agent's persistent memory across sessions) is **"No PowerShell builds — user builds locally."** This pass therefore did not run a baseline `mingw32-make` / `cmake --build` to verify each batch. To compensate:

- Each safe change was committed in isolation so it can be cherry-picked or reverted on its own.
- Only changes whose behavior preservation is *mechanically obvious from reading the surrounding code* were made.
- All semantically risky candidates were skipped and recorded here.

The user should compile and run tests against `cleanup/cpp-quality-const-pass` before merging.

## Findings examined and intentionally skipped

### `random_utilities.cpp:152` — distribution local

```cpp
uniform_int_distribution<size_t> distribution(0, values.size() - 1);
```

A scan agent suggested `const uniform_int_distribution<size_t> distribution(...)`. **Skipped: `std::uniform_int_distribution::operator()` is not const-qualified** (it may update internal state across calls). Adding `const` would prevent the very-next-line call `distribution(get_generator())` from compiling. Same reasoning applies to all other `*_distribution` locals in this file (`uniform_real_distribution`, `normal_distribution`, `bernoulli_distribution`, etc.).

### `json.cpp:385` — Json local on a return path

```cpp
Json v = p.parse_value();
p.skip_ws();
if (p.i != text.size())
    throw std::runtime_error("JSON parse: trailing data");
return v;
```

A scan agent suggested `const Json v`. **Skipped:** while modern compilers will still apply NRVO to a `const` named return value, declaring it `const` removes the implicit-move fallback that the standard provides when copy elision isn't performed. The end result is observable only as a performance regression, not a behavior change, but the rule is "do not change behavior" and `const` here trades a mechanical guarantee for compiler-version-dependent behavior. Not worth it.

### Member-function `const` candidates

The pass did not attempt member-function `const` additions. The risk profile (must update both declaration and definition, must propagate through any caller that takes the receiver by `const&`, and must not create accidental virtual-dispatch surprises) is too high without a baseline build to verify each batch.

### `override` keyword

Audit of `*.h` files showed the project already applies `override` consistently. No missing `override` candidates were found in non-excluded files.

### `NULL` → `nullptr`

Only occurrences of `NULL` in the codebase are inside generated-code string literals (e.g., `model_expression.cpp` emits C source code as text). No real `NULL` usage in C++ source.

### `0`/`NULL` pointer initializations

A grep for `Type* name = 0;` patterns returned no matches.

## Categories where this pass found nothing actionable

- Add `const` to local variables (only one safe candidate found — applied).
- Add missing `override` (codebase already consistent).
- Replace `NULL` with `nullptr` (no real `NULL` usages).
- Range-for `const auto&` (loops already use `const auto&` where appropriate).
- Internal-linkage hardening for file-local helpers (already `static` or in anonymous namespaces — verified in pass 1).
- `[[nodiscard]]`, `explicit` on single-arg ctors, default member initializers, `= default`/`= delete` — all deferred per the explicit risk-rule of this pass.

## Recommendation for follow-up

A meaningful next pass on C++ quality requires running the build between batches. Once the user can host the build locally and pipe results back, the agent could safely tackle:

1. Adding `const` to member functions that don't modify state (would need build verification per batch).
2. Adding `[[nodiscard]]` to non-virtual return-value-only helpers.
3. Tightening compiler-warning compliance (signed/unsigned comparisons, shadowing).
4. Making one-arg ctors `explicit` where conversion is clearly unintended.

These were intentionally NOT attempted in this pass because each requires a build to safely verify.
