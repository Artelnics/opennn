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

struct RpnOp // @todo change name
{
    enum class Kind 
    {
        PushConst, PushInput, PushOutput,
        Add, Sub, Mul, Div, Pow, Neg,
        Sqrt, Exp, Log, Abs, Sin, Cos, Tan, Min, Max
    };

    Kind kind = Kind::PushConst;
    Index index = 0;
    float constant = 0.0f;
};

struct CompiledFormula // @todo change name formula no good
{
    vector<RpnOp> bytecode; // @todo change name to XXX?

    FormulaShape shape = FormulaShape::Nonlinear;
    FormulaScope scope = FormulaScope::InputsOnly;

    vector<Index> input_indices;
    vector<Index> output_indices;

    vector<pair<Index, float>> affine_input_terms;
    vector<pair<Index, float>> affine_output_terms;
    
    float affine_constant = 0.0f;

    vector<pair<Index, vector<RpnOp>>> input_gradient;
    vector<pair<Index, vector<RpnOp>>> output_gradient;

    float evaluate(const VectorR&, const VectorR&) const;
};

float evaluate_rpn(const vector<RpnOp>&,
                   const VectorR&,
                   const VectorR&);

CompiledFormula compile_formula(const string&,
                                const vector<pair<string, Index>>&,
                            const vector<pair<string, Index>>&);


enum class ComparisonOperator
{
    None, EqualTo, Between, GreaterEqualTo, LessEqualTo, GreaterThan, LessThan, AllowedSet
};


enum class ConstraintKind { Unrepairable, Callback, AffineInput, NonlinearInput, OutputDependent }; // @todo change name another enum kind


struct MultivariateConstraint
{
    string expression; // @todo change !!!
    
    function<float(const VectorR&, const VectorR&)> callback;

    bool uses_callback = false;

    ComparisonOperator comparison_operator = ComparisonOperator::None;
    float low_bound = 0.0f;
    float up_bound = 0.0f;

    vector<float> allowed_values;

    CompiledFormula compiled;
    ConstraintKind kind = ConstraintKind::Unrepairable;

    bool is_satisfied(const VectorR&, const VectorR&);

};


struct UnivariateConstraint
{
    ComparisonOperator comparison; // @todo change name
    float low_bound;
    float up_bound;

    vector<float> allowed_values;

    UnivariateConstraint(ComparisonOperator new_comparison = ComparisonOperator::None, 
        float new_low_bound = 0.0f, 
        float new_up_bound = 0.0f)
        : comparison(new_comparison), low_bound(new_low_bound), up_bound(new_up_bound) {}
};


struct Cardinality
{
    vector<string> variable_names;
    Index k = 0; 

    bool force_nonzero = true;
};


struct LinearConstraintSet
{
    MatrixR A;
    VectorR lower;
    VectorR upper;
};


inline float bound_tolerance(float bound) { return max(EPSILON, abs(bound) * 1e-4f); }

void snap_to_lattice(MatrixR&, Index, float, float);

struct Lattice
{
    vector<Index> columns;
    vector<float> min;
    vector<float> max;
};

ConstraintKind classify(const MultivariateConstraint&);

vector<vector<MultivariateConstraint>> expand_constraint(const string&,
                                                         ComparisonOperator,
                                                         float, float,
                                                         const vector<pair<string, Index>>&,
                                                         const vector<pair<string, Index>>&);


bool all_formula_constraints_are_linear(const vector<MultivariateConstraint>&);

LinearConstraintSet build_linear_constraint_set(const vector<MultivariateConstraint>&,
                                                const Index,
                                                const Index);

void repair_affine_inputs(MatrixR&,
                          const VectorR&,
                          const VectorR&,
                          const vector<MultivariateConstraint>&,
                          Index max_correction_passes = 64);

void repair_nonlinear_inputs(MatrixR&,
                             const VectorR&,
                             const VectorR&,
                             const vector<MultivariateConstraint>&,
                             Index max_correction_passes = 64);

void repair_single_affine_input(MatrixR&,
                                const VectorR&,
                                const VectorR&,
                                const MultivariateConstraint&);

void repair_single_affine_integer(MatrixR&,
                                  const VectorR&,
                                  const VectorR&,
                                  const MultivariateConstraint&);

void repair_inputs(MatrixR&,
                   const VectorR&,
                   const VectorR&,
                   const vector<MultivariateConstraint>&);

void repair_affine_inputs_with_fixed(MatrixR&,
                                     const VectorR&,
                                     const VectorR&,
                                     const vector<MultivariateConstraint>&,
                                     const vector<char>&,
                                     Index max_correction_passes = 64);

void repair_mixed_integer_inputs(MatrixR&,
                                 const VectorR&,
                                 const VectorR&,
                                 const vector<MultivariateConstraint>&,
                                 const vector<char>&,
                                 const Lattice&,
                                 const vector<vector<Index>>&,
                                 const Lattice&,
                                 Index,
                                 float);

using SurrogateForward      = function<VectorR(const VectorR&)>;
using SurrogateVjp          = function<VectorR(const VectorR&, const VectorR&)>;
using SurrogateBatchForward = function<MatrixR(const MatrixR&)>;

void repair_output_constraints(MatrixR&,
                               const VectorR&,
                               const VectorR&,
                               const vector<MultivariateConstraint>&,
                               const SurrogateForward&,
                               const SurrogateVjp&,
                               Index max_correction_passes = 64,
                               const vector<char>& fixed_columns = {});

void repair_output_constraints(MatrixR&,
                               const VectorR&,
                               const VectorR&,
                               const vector<MultivariateConstraint>&,
                               const SurrogateBatchForward&,
                               Index max_correction_passes = 64,
                               const vector<char>& fixed_columns = {});


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
