# Concrete Response Optimization Example

This example uses the UCI Concrete Compressive Strength dataset and a small pretrained OpenNN surrogate from the IDC paper companion material.

Decision variables:

```text
cement, slag, fly_ash, water, sp, coarse_agg, fine_agg, age
```

Response:

```text
strength
```

Objectives:

```text
maximize strength
minimize cement
```

Simplex / mass-balance expression:

```text
cement + slag + fly_ash + water + sp + coarse_agg + fine_agg = 2325.012558
```

With nonnegative ingredient bounds, that equality places the seven ingredient masses on a fixed-density simplex. The example writes it directly as a `ResponseOptimization::add_constraint(...)` expression.

Other affine constraints shown in `main.cpp`:

```text
cement + slag + fly_ash >= 200
water - 0.30 * cement >= 0
water - 0.70 * cement <= 0
slag - 0.70 * (cement + slag + fly_ash) <= 0
fly_ash - 0.40 * (cement + slag + fly_ash) <= 0
age = 28
```