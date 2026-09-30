# Concrete Compressive Strength data

Yeh, I. (1998), *Concrete Compressive Strength*.
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength),
DOI [10.24432/C5PK67](https://doi.org/10.24432/C5PK67).
Yeh, I. (1998), *Modeling of strength of high-performance concrete using
artificial neural networks*, Cement and Concrete Research, 28(12), 1797-1808,
DOI [10.1016/S0008-8846(98)00165-3](https://doi.org/10.1016/S0008-8846(98)00165-3).

Data licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Preserve the attribution, source and licence link when redistributing.

OpenNN adaptation: `Concrete_Data.xls` converted to comma-separated values
with short column names added. All 1,030 rows match the upstream values in
order, verified 2026-09-30.

Columns, in the UCI order:

```text
cement, slag, fly_ash, water, sp, coarse_agg, fine_agg, age, strength
```

Ingredient masses are in kg/m^3, `age` is in days, and `strength` (the
compressive strength, the target) is in MPa.
