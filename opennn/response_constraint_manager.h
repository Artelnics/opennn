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
#include "variable.h"
#include "statistics.h"

namespace opennn
{

class NeuralNetwork;
class ResponseOptimization;

enum class Sense { Minimize, Maximize, Fixed };


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


struct ConstraintGeometry
{
    NeuralNetwork* neural_network = nullptr;
    function<bool(const string&)> is_history;
    const vector<Variable>* input_variables = nullptr;      // filtered "Input" variables
    const vector<Variable>* target_variables = nullptr;     // filtered "Target" variables
    const vector<Descriptives>* target_descriptives = nullptr;
};


struct ConstraintSet
{
    map<string, UnivariateConstraint> univariate;
    vector<MultivariateConstraint> multivariate;
    vector<vector<vector<MultivariateConstraint>>> disjunctive;
    vector<Cardinality> cardinality;
};


using SurrogateForward      = function<VectorR(const VectorR&)>;
using SurrogateVjp          = function<VectorR(const VectorR&, const VectorR&)>;
using SurrogateBatchForward = function<MatrixR(const MatrixR&)>;


// Network evaluation the algorithm injects for output-coupled repair: either the analytic
// forward+VJP pair (has_differential) or a batch forward fallback.
struct SurrogateOracle
{
    bool has_differential = false;
    SurrogateForward forward;
    SurrogateVjp vjp;
    SurrogateBatchForward batch_forward;
};


class ConstraintManager
{
public:

    ConstraintSet constraint_set;
    set<string> absorbed_objectives;

    vector<pair<string, Index>> input_columns;
    vector<pair<string, Index>> output_columns;
    set<string> input_names;

    float relative_tolerance = 1e-6f;

    void build(const ResponseOptimization&, const ConstraintGeometry&, bool lower);

    bool is_objective(const string&) const;
    Sense get_sense(const string&) const;
    float get_fixed_value(const string&) const;

    UnivariateConstraint get_constraint(const string&) const;

    Index get_objectives_number() const;
    Index get_optimizing_objectives_number() const;

    // Feasibility + output-coupled repair verbs (input/combinatorial enforcement stays with the
    // sampler, which is algorithm-specific). frontiers are the output-domain [inferior, superior].
    bool is_feasible(const VectorR& input_row, const VectorR& output_row) const;

    pair<MatrixR, MatrixR> filter_feasible(const MatrixR& inputs,
                                           const MatrixR& outputs,
                                           const VectorR& inferior_frontier,
                                           const VectorR& superior_frontier) const;

    void repair_outputs(MatrixR& points,
                        const VectorR& inferior_frontier,
                        const VectorR& superior_frontier,
                        const SurrogateOracle& oracle,
                        const vector<char>& fixed_columns) const;

private:

    const ResponseOptimization* problem = nullptr;
    ConstraintGeometry geometry;

    void build_columns();
    void add_constraint(const string& expression, Condition condition, const vector<float>& values);
    void add_cardinality(const string& expression, const vector<float>& values);
    void add_formula(const string&, Condition, float low, float up);
    void expand_fixed_objectives();
    void promote_single_variable_constraints();

    bool is_input_name(const string&) const;
    Index input_column_of(const string&) const;
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
