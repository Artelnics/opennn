//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N S T R A I N T   F O R M U L A S   H E A D E R
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
    None, EqualTo, Between, GreaterEqualTo, LessEqualTo, GreaterThan, LessThan
};


struct FormulaConstraint
{
    string expression;
    function<float(const VectorR&, const VectorR&)> callback;
    bool uses_callback = false;

    ComparisonOperator comparison_operator = ComparisonOperator::None;
    float low_bound = 0.0f;
    float up_bound = 0.0f;

    CompiledFormula compiled;
};


struct LinearConstraintSet
{
    MatrixR A;       // (m_constraints, n_inputs + n_outputs); first n_inputs cols are input coefficients
    VectorR lower;   // (m_constraints), -infinity for one-sided upper-bound constraints
    VectorR upper;   // (m_constraints), +infinity for one-sided lower-bound constraints
};


inline float bound_tolerance(float bound) { return max(EPSILON, abs(bound) * 1e-4f); }

[[nodiscard]] bool all_formula_constraints_are_linear(const vector<FormulaConstraint>& formula_constraints);

[[nodiscard]] LinearConstraintSet build_linear_constraint_set(const vector<FormulaConstraint>& formula_constraints,
                                                              const Index n_in,
                                                              const Index n_out);

// Batched Dykstra projection of every random draw onto the intersection of the
// affine, input-only constraints and the box. Exact (to tolerance) for the
// feasible draws; handles equalities and inequalities (the latter via slacks).
void repair_affine_inputs(MatrixR& random_inputs,
                          const VectorR& inferior_frontier,
                          const VectorR& superior_frontier,
                          const vector<FormulaConstraint>& formula_constraints,
                          Index max_correction_passes = 64);

// Per-point Gauss-Newton-on-manifold repair for smooth nonlinear input-only
// constraints. Each point alternates an active-set Gauss-Newton projection
// (x <- x - J^T (J J^T)^-1 h, with J the Jacobian of the currently-violated
// constraints and h their residual) with a box clamp, for a few passes. It is a
// no-op unless at least one genuinely nonlinear smooth input-only constraint is
// present; the all-affine case is handled (batched) by repair_affine_inputs. Any
// affine input-only constraints are folded into the same active set so they stay
// satisfied jointly. Points still infeasible after the cap are left for the
// downstream feasibility filter to drop. The pass cap is generous because the
// loop early-exits per point once its residual is negligible (so well-conditioned
// manifolds still cost only a handful of passes); stiff/anisotropic manifolds are
// the ones that actually use the budget (ellipse-class ~32, well-conditioned ~8).
void repair_nonlinear_inputs(MatrixR& random_inputs,
                             const VectorR& inferior_frontier,
                             const VectorR& superior_frontier,
                             const vector<FormulaConstraint>& formula_constraints,
                             Index max_correction_passes = 64);

// Single-pass random-sweep clamp-and-carry for one affine input-only
// constraint. Algebraically exact when the target is reachable within the box,
// and maximally diversity-preserving (satisfied points are left untouched).
void repair_single_affine_input(MatrixR& random_inputs,
                                const VectorR& inferior_frontier,
                                const VectorR& superior_frontier,
                                const FormulaConstraint& constraint);

// Router over the input-only, non-callback constraints: one affine -> single
// random sweep; several affine -> Gram + Dykstra; any smooth nonlinear ->
// Gauss-Newton (which folds in the affine ones). Call this from the sampler.
void repair_inputs(MatrixR& random_inputs,
                   const VectorR& inferior_frontier,
                   const VectorR& superior_frontier,
                   const vector<FormulaConstraint>& formula_constraints);

// Surrogate callbacks for output-constraint repair (Regime 2 / reverse-mode
// VJP). The module stays network-agnostic: the caller supplies the forward map
// and the vector-Jacobian product, so the exact network Jacobian (the weight /
// activation chain) is reused rather than finite-differenced.
//   forward(x)        -> y = f(x)                 (n_out)
//   vjp(x, cotangent) -> (df/dx)^T * cotangent    (n_in), cotangent is n_out
using SurrogateForward = function<VectorR(const VectorR&)>;
using SurrogateVjp     = function<VectorR(const VectorR&, const VectorR&)>;

// Per-point Gauss-Newton repair for smooth constraints that reference network
// outputs (scope Mixed or OutputsOnly). The constraint manifold g(x, f(x)) = c
// is projected with the chain-rule Jacobian d/dx g = dg/dx + dg/dy * df/dx,
// where df/dx is supplied exactly by vjp(). Each point alternates a Gauss-Newton
// step with a box clamp, with a residual early-exit. The surrogate manifold is
// nonconvex so this is local; points still infeasible after the cap are left for
// the downstream rejection filter. Input-only constraints are ignored here (they
// are repaired pre-evaluation by repair_inputs).
void repair_output_constraints(MatrixR& inputs,
                               const VectorR& inferior_frontier,
                               const VectorR& superior_frontier,
                               const vector<FormulaConstraint>& formula_constraints,
                               const SurrogateForward& forward,
                               const SurrogateVjp& vjp,
                               Index max_correction_passes = 64);

// Same, but without an explicit VJP: the network Jacobian is obtained by a
// box-scaled central difference over `forward` (the only network access is the
// public forward map, so no neural-network internals are touched). The repaired
// points are still exact to tolerance — the exact forward plus Gauss-Newton
// drive the residual to zero; the finite-difference Jacobian only sets the step
// direction, not the final feasibility.
void repair_output_constraints(MatrixR& inputs,
                               const VectorR& inferior_frontier,
                               const VectorR& superior_frontier,
                               const vector<FormulaConstraint>& formula_constraints,
                               const SurrogateForward& forward,
                               Index max_correction_passes = 64);


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
