//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   C O N S T R A I N T   M A N A G E R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "expression_evaluator.h"

namespace opennn
{

enum class Condition
{
    None, EqualTo, Between, GreaterEqualTo, LessEqualTo, GreaterThan, LessThan, AllowedSet, Integer, Cardinality
};


enum class RepairRegime { None, InputAffine, InputNonlinear, OutputCoupled };


struct MultivariateConstraint
{
    string expression;

    Condition condition = Condition::None;
    float low_bound = 0.0f;
    float up_bound = 0.0f;

    vector<float> allowed_values;

    CompiledExpression compiled;
    RepairRegime kind = RepairRegime::None;
};


struct UnivariateConstraint
{
    Condition condition;
    float low_bound;
    float up_bound;

    vector<float> allowed_values;

    UnivariateConstraint(Condition new_condition = Condition::None,
        float new_low_bound = 0.0f,
        float new_up_bound = 0.0f)
        : condition(new_condition), low_bound(new_low_bound), up_bound(new_up_bound) {}
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

RepairRegime classify(const MultivariateConstraint&);

vector<vector<MultivariateConstraint>> expand_constraint(const string&,
                                                         Condition,
                                                         float, float,
                                                         const vector<pair<string, Index>>&,
                                                         const vector<pair<string, Index>>&);


bool all_formula_constraints_are_linear(const vector<MultivariateConstraint>&);

bool constraint_is_satisfied(const MultivariateConstraint&,
                             const VectorR&,
                             const VectorR&);

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
