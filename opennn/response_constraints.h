//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   C O N S T R A I N T S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

enum class FormulaShape { Affine, Nonlinear };
enum class FormulaScope { InputsOnly, OutputsOnly, Mixed };

struct RpnOp
{
    enum class Kind : uint8_t
    {
        PushConst, PushInput, PushOutput,
        Add, Sub, Mul, Div, Pow, Neg,
        Sqrt, Exp, Log, Abs, Sin, Cos, Tan, Min, Max
    };

    Kind kind = Kind::PushConst;
    Index index = 0;
    float constant = 0.0f;
};

struct NamedColumn
{
    string name;
    Index column_index = 0;
};

struct CompiledFormula
{
    vector<RpnOp> bytecode;

    FormulaShape shape = FormulaShape::Nonlinear;
    FormulaScope scope = FormulaScope::InputsOnly;

    vector<Index> input_indices;
    vector<Index> output_indices;

    vector<pair<Index, float>> affine_input_terms;
    vector<pair<Index, float>> affine_output_terms;
    float affine_constant = 0.0f;

    // First-order gradient of the expression, compiled to bytecode (one RPN
    // program per referenced input / output column). Populated for any smooth
    // nonlinear formula. input_gradient drives the input-only Gauss-Newton
    // repair; output_gradient supplies dg/dy for the chain rule dg/dy * df/dx
    // through the network in the output-constraint repair. Affine formulas keep
    // their constant affine_*_terms instead; non-smooth formulas (min/max)
    // leave these empty and gradient_available false.
    vector<pair<Index, vector<RpnOp>>> input_gradient;
    vector<pair<Index, vector<RpnOp>>> output_gradient;
    bool gradient_available = false;

    float evaluate(const VectorR& inputs_row, const VectorR& outputs_row) const;
};

// Evaluate a stand-alone RPN program (e.g. a compiled partial derivative) at a
// single point. Shared by CompiledFormula::evaluate and the nonlinear repair.
float evaluate_rpn(const vector<RpnOp>& bytecode,
                   const VectorR& inputs_row,
                   const VectorR& outputs_row);

CompiledFormula compile_formula(const string& expression,
                                              const vector<NamedColumn>& inputs,
                                              const vector<NamedColumn>& outputs);


enum class ComparisonOperator : uint8_t
{
    None, EqualTo, Between, GreaterEqualTo, LessEqualTo, GreaterThan, LessThan, AllowedSet
};


struct MultivariateConstraint
{
    string expression;
    function<float(const VectorR&, const VectorR&)> callback;
    bool uses_callback = false;

    ComparisonOperator comparison_operator = ComparisonOperator::None;
    float low_bound = 0.0f;
    float up_bound = 0.0f;

    // Membership target for ComparisonOperator::AllowedSet (expr in {values}).
    // ResponseOptimization expands it into one EqualTo branch per value before any
    // repair/filter runs, so the solver layers below never see AllowedSet directly.
    vector<float> allowed_values;

    CompiledFormula compiled;
};


struct LinearConstraintSet
{
    MatrixR A;       // (m_constraints, n_inputs + n_outputs); first n_inputs cols are input coefficients
    VectorR lower;   // (m_constraints), -infinity for one-sided upper-bound constraints
    VectorR upper;   // (m_constraints), +infinity for one-sided lower-bound constraints
};


inline float bound_tolerance(float bound) { return max(EPSILON, abs(bound) * 1e-4f); }

// Round one column to its integer lattice and clamp to [minimum, maximum].
void snap_to_lattice(MatrixR& inputs, Index column, float minimum, float maximum);

// True for a real, input-only constraint the repair pipeline can act on: a set
// comparison operator, no opaque callback, and no output dependence.
[[nodiscard]] inline bool is_input_only_repairable(const MultivariateConstraint& constraint)
{
    return !constraint.uses_callback
        && constraint.comparison_operator != ComparisonOperator::None
        && constraint.compiled.scope == FormulaScope::InputsOnly;
}

// True when one constraint is satisfied at (input_row, output_row), within bound_tolerance on
// every bound (strict GreaterThan/LessThan are treated like their non-strict forms). Evaluates
// the callback or the compiled expression as appropriate. Shared by the feasibility filter and
// the mixed-integer pump.
[[nodiscard]] bool constraint_is_satisfied(const MultivariateConstraint& constraint,
                                           const VectorR& input_row,
                                           const VectorR& output_row);

[[nodiscard]] bool all_formula_constraints_are_linear(const vector<MultivariateConstraint>& formula_constraints);

[[nodiscard]] LinearConstraintSet build_linear_constraint_set(const vector<MultivariateConstraint>& formula_constraints,
                                                              const Index n_in,
                                                              const Index n_out);

// Batched Dykstra projection of every random draw onto the intersection of the
// affine, input-only constraints and the box. Exact (to tolerance) for the
// feasible draws; handles equalities and inequalities (the latter via slacks).
void repair_affine_inputs(MatrixR& random_inputs,
                          const VectorR& inferior_frontier,
                          const VectorR& superior_frontier,
                          const vector<MultivariateConstraint>& formula_constraints,
                          Index max_correction_passes = 64);

// Per-point Gauss-Newton-on-manifold repair for smooth nonlinear input-only constraints:
// alternates an active-set GN projection (x <- x - J^T (J J^T)^-1 h) with a box clamp, with a
// per-point residual early-exit. Affine input-only constraints are folded into the same active
// set. No-op unless a genuinely nonlinear smooth input constraint exists (the all-affine case
// goes to repair_affine_inputs). Points still infeasible after the cap fall to the filter.
void repair_nonlinear_inputs(MatrixR& random_inputs,
                             const VectorR& inferior_frontier,
                             const VectorR& superior_frontier,
                             const vector<MultivariateConstraint>& formula_constraints,
                             Index max_correction_passes = 64);

// Single-pass random-sweep clamp-and-carry for one affine input-only
// constraint. Algebraically exact when the target is reachable within the box,
// and maximally diversity-preserving (satisfied points are left untouched).
void repair_single_affine_input(MatrixR& random_inputs,
                                const VectorR& inferior_frontier,
                                const VectorR& superior_frontier,
                                const MultivariateConstraint& constraint);

// Lattice analogue of repair_single_affine_input for one affine constraint over all-integer/
// binary variables (caller-guaranteed): snap each row to the lattice [ceil(inf), floor(sup)],
// then clamp-and-carry the residual in whole integer steps truncated toward zero (so an
// inequality is never overshot into its opposite bound). Exact in one pass for a unit-coefficient
// target reachable in the box (pure-integer knapsack); any irrepresentable residue falls to the
// pump. Single-constraint fast path; coupled/multi-constraint cases go to the mixed-integer pump.
void repair_single_affine_integer(MatrixR& random_inputs,
                                  const VectorR& inferior_frontier,
                                  const VectorR& superior_frontier,
                                  const MultivariateConstraint& constraint);

// Router over the input-only, non-callback constraints: one affine -> single
// random sweep; several affine -> Gram + Dykstra; any smooth nonlinear ->
// Gauss-Newton (which folds in the affine ones). Call this from the sampler.
void repair_inputs(MatrixR& random_inputs,
                   const VectorR& inferior_frontier,
                   const VectorR& superior_frontier,
                   const vector<MultivariateConstraint>& formula_constraints);

// Masked per-row affine projection: project each point's FREE input coordinates onto the
// input-only affine/smooth constraints while holding every `fixed_columns` coordinate (size =
// n_inputs, nonzero = fixed) at its per-row value via a zeroed Jacobian. The box clamp is the
// final op of every pass, so an affine-vs-box conflict surfaces as leftover residual (the
// caller's feasibility signal), never a box violation. The projection half of mixed-integer
// repair (round discrete columns, fix them, re-project the continuous slice). A constraint with
// all variables fixed is dropped. Empty `fixed_columns` projects every coordinate.
void repair_affine_inputs_with_fixed(MatrixR& random_inputs,
                                     const VectorR& inferior_frontier,
                                     const VectorR& superior_frontier,
                                     const vector<MultivariateConstraint>& formula_constraints,
                                     const vector<char>& fixed_columns,
                                     Index max_correction_passes = 64);

// Cyclic feasibility pump for coupled mixed-integer / cardinality input constraints: snap the
// discrete columns to the lattice, then repeatedly project the continuous slice with the discrete
// columns fixed (repair_affine_inputs_with_fixed), test each row, and perturb still-infeasible
// rows (escalating cardinality swap + free-integer unlock). Invariant: discrete columns move only
// via snap/perturbation, never the projection, so they stay on-grid and in-box with no re-round.
// The caller supplies the precomputed lattice / cardinality column groups. Rows still infeasible
// after outer_cap passes fall to the downstream feasibility filter.
void repair_mixed_integer_inputs(MatrixR& inputs,
                                 const VectorR& inferior_frontier,
                                 const VectorR& superior_frontier,
                                 const vector<MultivariateConstraint>& formula_constraints,
                                 const vector<char>& fixed_mask,
                                 const vector<Index>& lattice_columns,
                                 const vector<float>& lattice_min,
                                 const vector<float>& lattice_max,
                                 const vector<vector<Index>>& cardinality_columns,
                                 const vector<Index>& free_lattice_columns,
                                 const vector<float>& free_lattice_min,
                                 const vector<float>& free_lattice_max,
                                 Index outer_cap,
                                 float exploration_ratio);

// Surrogate callbacks for output-constraint repair (Regime 2 / reverse-mode
// VJP). The module stays network-agnostic: the caller supplies the forward map
// and the vector-Jacobian product, so the exact network Jacobian (the weight /
// activation chain) is reused rather than finite-differenced.
//   forward(x)        -> y = f(x)                 (n_out)
//   vjp(x, cotangent) -> (df/dx)^T * cotangent    (n_in), cotangent is n_out
using SurrogateForward = function<VectorR(const VectorR&)>;
using SurrogateVjp     = function<VectorR(const VectorR&, const VectorR&)>;

// Per-point Gauss-Newton repair for smooth output-referencing constraints (scope Mixed or
// OutputsOnly). Projects g(x, f(x)) = c with the chain-rule Jacobian dg/dx + dg/dy * df/dx,
// df/dx supplied exactly by vjp(); alternates a GN step with a box clamp and a residual
// early-exit. Local (the surrogate manifold is nonconvex); leftovers fall to the rejection
// filter. Input-only constraints are repaired pre-evaluation by repair_inputs.
void repair_output_constraints(MatrixR& inputs,
                               const VectorR& inferior_frontier,
                               const VectorR& superior_frontier,
                               const vector<MultivariateConstraint>& formula_constraints,
                               const SurrogateForward& forward,
                               const SurrogateVjp& vjp,
                               Index max_correction_passes = 64,
                               const vector<char>& fixed_columns = {});

// Same, but without an explicit VJP: the Jacobian is a box-scaled central difference over
// `forward` (no network internals touched). Still exact to tolerance — the exact forward plus
// GN drive the residual to zero; the finite-difference Jacobian only sets the step direction.
void repair_output_constraints(MatrixR& inputs,
                               const VectorR& inferior_frontier,
                               const VectorR& superior_frontier,
                               const vector<MultivariateConstraint>& formula_constraints,
                               const SurrogateForward& forward,
                               Index max_correction_passes = 64,
                               const vector<char>& fixed_columns = {});


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
