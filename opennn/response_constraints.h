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

    vector<pair<Index, vector<RpnOp>>> input_gradient;
    vector<pair<Index, vector<RpnOp>>> output_gradient;
    bool gradient_available = false;

    float evaluate(const VectorR& inputs_row, const VectorR& outputs_row) const;
};

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


enum class ConstraintKind { Unrepairable, Callback, AffineInput, NonlinearInput, OutputDependent };


struct MultivariateConstraint
{
    string expression;
    function<float(const VectorR&, const VectorR&)> callback;
    bool uses_callback = false;

    ComparisonOperator comparison_operator = ComparisonOperator::None;
    float low_bound = 0.0f;
    float up_bound = 0.0f;

    vector<float> allowed_values;

    CompiledFormula compiled;
    ConstraintKind kind = ConstraintKind::Unrepairable;
};


struct LinearConstraintSet
{
    MatrixR A;
    VectorR lower;
    VectorR upper;
};


inline float bound_tolerance(float bound) { return max(EPSILON, abs(bound) * 1e-4f); }

void snap_to_lattice(MatrixR& inputs, Index column, float minimum, float maximum);

// Integer/binary columns with their per-column lattice bounds [min, max].
struct Lattice
{
    vector<Index> columns;
    vector<float> min;
    vector<float> max;
};

[[nodiscard]] ConstraintKind classify(const MultivariateConstraint& constraint);

[[nodiscard]] bool constraint_is_satisfied(const MultivariateConstraint& constraint,
                                           const VectorR& input_row,
                                           const VectorR& output_row);

[[nodiscard]] bool all_formula_constraints_are_linear(const vector<MultivariateConstraint>& formula_constraints);

[[nodiscard]] LinearConstraintSet build_linear_constraint_set(const vector<MultivariateConstraint>& formula_constraints,
                                                              const Index n_in,
                                                              const Index n_out);

void repair_affine_inputs(MatrixR& random_inputs,
                          const VectorR& inferior_frontier,
                          const VectorR& superior_frontier,
                          const vector<MultivariateConstraint>& formula_constraints,
                          Index max_correction_passes = 64);

void repair_nonlinear_inputs(MatrixR& random_inputs,
                             const VectorR& inferior_frontier,
                             const VectorR& superior_frontier,
                             const vector<MultivariateConstraint>& formula_constraints,
                             Index max_correction_passes = 64);

void repair_single_affine_input(MatrixR& random_inputs,
                                const VectorR& inferior_frontier,
                                const VectorR& superior_frontier,
                                const MultivariateConstraint& constraint);

void repair_single_affine_integer(MatrixR& random_inputs,
                                  const VectorR& inferior_frontier,
                                  const VectorR& superior_frontier,
                                  const MultivariateConstraint& constraint);

void repair_inputs(MatrixR& random_inputs,
                   const VectorR& inferior_frontier,
                   const VectorR& superior_frontier,
                   const vector<MultivariateConstraint>& formula_constraints);

void repair_affine_inputs_with_fixed(MatrixR& random_inputs,
                                     const VectorR& inferior_frontier,
                                     const VectorR& superior_frontier,
                                     const vector<MultivariateConstraint>& formula_constraints,
                                     const vector<char>& fixed_columns,
                                     Index max_correction_passes = 64);

void repair_mixed_integer_inputs(MatrixR& inputs,
                                 const VectorR& inferior_frontier,
                                 const VectorR& superior_frontier,
                                 const vector<MultivariateConstraint>& formula_constraints,
                                 const vector<char>& fixed_mask,
                                 const Lattice& lattice,
                                 const vector<vector<Index>>& cardinality_columns,
                                 const Lattice& free_lattice,
                                 Index outer_cap,
                                 float exploration_ratio);

using SurrogateForward = function<VectorR(const VectorR&)>;
using SurrogateVjp     = function<VectorR(const VectorR&, const VectorR&)>;

void repair_output_constraints(MatrixR& inputs,
                               const VectorR& inferior_frontier,
                               const VectorR& superior_frontier,
                               const vector<MultivariateConstraint>& formula_constraints,
                               const SurrogateForward& forward,
                               const SurrogateVjp& vjp,
                               Index max_correction_passes = 64,
                               const vector<char>& fixed_columns = {});

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
