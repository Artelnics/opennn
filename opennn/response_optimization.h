//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "statistics.h"
#include "variable.h"
#include "response_constraints.h"
#include "network_differential.h"

namespace opennn
{

class NeuralNetwork;
class Dataset;

class ResponseOptimization
{
public:

    enum class Sense { Minimize, Maximize };

    enum class TimeType { PresentContinuous, PresentBatch, PastContinuous, PastBatch };

    enum class BranchMode { Budgeted, Exhaustive };

    struct UnivariateConstraint
    {
        ComparisonOperator comparison;
        float low_bound;
        float up_bound;

        vector<float> allowed_values;

        UnivariateConstraint(ComparisonOperator new_comparison = ComparisonOperator::None, float new_low_bound = 0.0f, float new_up_bound = 0.0f)
            : comparison(new_comparison), low_bound(new_low_bound), up_bound(new_up_bound) {}
    };

    struct CardinalityConstraint
    {
        vector<string> variable_names;
        Index k = 0;
    };

    struct Domain
    {
        Domain() = default;
        virtual ~Domain() = default;

        Domain(const vector<Variable>& variables,
               const vector<Descriptives>& descriptives,
               const float deformation_domain_factor = 1.0f)
        {
            set(variables, descriptives, deformation_domain_factor);
        }

        void set(const vector<Variable>& variables,
                 const vector<Descriptives>& descriptives,
                 const float deformation_domain_factor = 1.0f);

        void bound(const vector<Variable>& variables, const vector<UnivariateConstraint>& constraints);

        void reshape(const float zoom_factor,
                     const VectorR& center,
                     const MatrixR& points_inputs,
                     const vector<Variable>& vars);

        VectorR inferior_frontier;
        VectorR superior_frontier;
    };

    struct Objectives
    {
        Objectives(const ResponseOptimization& response_optimization);

        MatrixR source_and_column;

        MatrixR utopian_and_sense;

        MatrixR scale_and_offset;

        MatrixR extract(const MatrixR& inputs, const MatrixR& output) const;

        void normalize(MatrixR& objective_matrix) const;

        bool update_utopian_from_points(const MatrixR& unnormalized_objective_values);
    };

    ResponseOptimization(NeuralNetwork* = nullptr);

    ~ResponseOptimization();

    void set(NeuralNetwork* = nullptr);

    void clear_constraints();
    void clear_constraints(const string& name);

    void clear_objectives();
    void clear_objectives(const string& name);

    void clear_time_roles();
    void clear_time_roles(const string& name);

    void set_constraint(const string& name, const ComparisonOperator comparison = ComparisonOperator::None, float low = 0.0f, float up = 0.0f);

    void set_constraint(const string& name, const vector<float>& allowed_values);

    void set_cardinality_constraint(const vector<string>& variable_names, Index k);
    void clear_cardinality_constraints();

    void set_objective(const string& name, const Sense sense);

    void set_time_role(const string& name, const TimeType role);

    void set_formula_constraint(const string& expression,
                                ComparisonOperator comparison,
                                float low = 0.0f, float up = 0.0f);

    void set_formula_constraint(function<float(const VectorR&, const VectorR&)> callback,
                                ComparisonOperator comparison,
                                float low = 0.0f, float up = 0.0f);

    void set_formula_constraint(const string& expression, const vector<float>& allowed_values);

    void clear_formula_constraints();

    void set_min_feasible_ratio(float new_ratio);
    void set_max_oversample_factor(Index new_factor);
    void set_exploration_ratio(float new_ratio);

    void set_fixed_history(const Tensor3& history);
    void clear_fixed_history();

    void set_iterations(const int iterations);
    void set_zoom_factor(float new_zoom_factor);
    void set_evaluations_number(const int new_evaluations_number);
    void set_relative_tolerance(float new_relative_tolerance);
    void set_max_pareto_number(const Index new_max_pareto_number);
    void set_max_total_evaluations(const Index new_max_total_evaluations);
    void set_initial_sampling_factor(const Index new_initial_sampling_factor);

    void set_branch_mode(const BranchMode new_branch_mode);

    void set_deformation_domain_factor(float new_deformation_domain_factor);
    float get_deformation_domain_factor();

    vector<Descriptives> get_descriptives(const string& role) const;

    pair<vector<Variable>, vector<Descriptives>> get_variables_and_descriptives(const string& role) const;

    vector<float> get_utopian_point() const;

    const map<string, vector<Index>>& get_category_frequencies() const { return category_frequencies; }

    const vector<CardinalityConstraint>& get_cardinality_constraints() const { return cardinality_constraints; }

    pair<Index, VectorR> get_advised_point(const MatrixR& pareto_front,
                                                         const VectorR& importance_scale = VectorR()) const;

    Domain get_original_domain(const string role) const;

    MatrixR calculate_random_inputs(const Domain& input_domain, Index evaluations_count = -1) const;

    void build_input_lattice(const vector<Variable>& variables,
                             const vector<Index>& feature_dimensions,
                             const Domain& input_domain,
                             map<string, Index>& scalar_column_of,
                             vector<Index>& lattice_columns,
                             vector<float>& lattice_min,
                             vector<float>& lattice_max) const;

    vector<vector<Index>> resolve_cardinality_columns(const Domain& input_domain,
                                                      const map<string, Index>& scalar_column_of,
                                                      const vector<char>& fixed_mask,
                                                      float discrete_explore,
                                                      MatrixR& random_inputs) const;

    Tensor3 combine_input(const MatrixR& present_random_values) const;

    MatrixR calculate_outputs(const MatrixR& input) const;

    pair<MatrixR, MatrixR> filter_feasible_points(const MatrixR& inputs,
                                                                const MatrixR& outputs,
                                                                const Domain& output_domain) const;

    pair<MatrixR, MatrixR> sample_feasible_points(const Domain& input_domain,
                                                                const Domain& output_domain,
                                                                const Index evaluations_multiplier = 1) const;

    pair<MatrixR, MatrixR> calculate_optimal_points(const MatrixR& feasible_inputs,
                                                                  const MatrixR& feasible_outputs,
                                                                  const Objectives& objectives) const;

    pair<MatrixR, MatrixR> calculate_pareto(const MatrixR& inputs, const MatrixR& outputs, const MatrixR& objective_matrix) const;

    pair<float, float> calculate_quality_metrics(const MatrixR& inputs,
                                                               const MatrixR& outputs,
                                                               const Objectives& objectives) const;

    MatrixR perform_single_objective_optimization() const;

    MatrixR perform_multiobjective_optimization() const;

    MatrixR solve_once() const;

    MatrixR perform_response_optimization();

    Index get_objectives_number() const;

    Index get_evaluations_used() const;

    vector<NamedColumn> build_input_columns(const vector<Variable>& variables) const;
    vector<NamedColumn> build_output_columns(const vector<Variable>& variables) const;

    UnivariateConstraint get_constraint(const string& name) const;
    bool is_objective(const string& name) const;
    Sense get_sense(const string& name) const;
    bool is_history(const string& name) const;
    static bool is_past(const TimeType role);

    bool is_forecasting() const { return fixed_history.size() > 0; }

    bool row_satisfies_formula_constraints(const VectorR& input_row,
                                                         const VectorR& output_row) const;

    pair<MatrixR, MatrixR> generate_feasible_points(const Domain& input_domain,
                                                                  const Domain& output_domain,
                                                                  Index evaluations_count) const;

    void initialize_network_differential() const;

    void restore_cardinality_columns(Domain& domain, const Domain& original) const;

    void promote_single_variable_constraints();

    vector<char> discrete_column_mask(const vector<Variable>& variables) const;

private:

    NeuralNetwork* neural_network = nullptr;

    mutable unique_ptr<NetworkDifferential> network_differential;

    map<string, UnivariateConstraint> constraints;

    map<string, Sense> objectives;

    map<string, TimeType> time_roles;

    vector<MultivariateConstraint> formula_constraints;

    vector<CardinalityConstraint> cardinality_constraints;

    float min_feasible_ratio = 0.01f;
    Index max_oversample_factor = 8;
    float exploration_ratio = 0.1f;

    mutable float last_feasibility_rate = 1.0f;

    Index evaluations_number = 2000;

    Index max_iterations = 20;

    Index min_iterations = 4;

    float zoom_factor = 0.85f;

    float relative_tolerance = 1e-6f;

    Index max_pareto_number = 10000;

    Index max_total_evaluations = 0;
    mutable Index evaluations_used = 0;

    mutable map<string, vector<Index>> category_frequencies;

    mutable vector<char> cardinality_preferred;

    mutable map<string, Index> cardinality_indicator_columns;

    Index initial_sampling_factor = 1;

    BranchMode branch_mode = BranchMode::Budgeted;

    float deformation_domain_factor = 1.0f;

    Tensor3 fixed_history;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
