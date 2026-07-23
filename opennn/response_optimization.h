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

class ResponseOptimization
{
public:

    enum class Sense { Minimize, Maximize, Fixed };

    enum class TimeType { PresentContinuous, PresentBatch, PastContinuous, PastBatch };

    enum class BranchMode { Budgeted, Exhaustive };

    struct ConstraintSet
    {
        map<string, UnivariateConstraint> univariate;
        vector<MultivariateConstraint> multivariate;
        vector<vector<vector<MultivariateConstraint>>> disjunctive;
        vector<Cardinality> cardinality;
    };

    struct SamplingMemory
    {
        map<string, vector<Index>> category_frequencies;
        vector<char> cardinality_preferred;
        map<string, Index> cardinality_indicator_columns;
        float last_feasibility_rate = 1.0f;
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

        void set(const vector<Variable>&,
                 const vector<Descriptives>&,
                 const float deformation_domain_factor = 1.0f);

        void bound(const vector<Variable>&, const vector<UnivariateConstraint>&);

        void reshape(const float,
                     const VectorR&,
                     const MatrixR&,
                     const vector<Variable>&);

        VectorR inferior_frontier;
        VectorR superior_frontier;
    };

    struct Objectives
    {
        Objectives(const ResponseOptimization&);

        MatrixR source_and_column;

        MatrixR utopian_and_sense;

        MatrixR scale_and_offset;

        vector<char> closeness_mask;
        VectorR closeness_target;
        VectorR closeness_scale;

        MatrixR extract(const MatrixR&, const MatrixR&) const;

        void normalize(MatrixR&) const;

        bool update_utopian_from_points(const MatrixR&);
    };

    explicit ResponseOptimization(NeuralNetwork* = nullptr);

    ~ResponseOptimization();

    void set(NeuralNetwork* = nullptr);

    void clear_constraints();
    void clear_constraints(const string&);

    void clear_objectives();
    void clear_objectives(const string&);

    void clear_time_roles();
    void clear_time_roles(const string&);

    void set_constraint(const string&, const ComparisonOperator comparison = ComparisonOperator::None, float low = 0.0f, float up = 0.0f);

    void set_constraint(const string&, const vector<float>&);

    void set_cardinality_constraint(const vector<string>&, Index, bool force_nonzero = true);
    void clear_cardinality_constraints();

    void set_objective(const string&, const Sense, const float value = 0.0f);

    void set_time_role(const string&, const TimeType);

    void set_formula_constraint(const string&,
                                ComparisonOperator,
                                float low = 0.0f, float up = 0.0f);

    void set_formula_constraint(function<float(const VectorR&, const VectorR&)>,
                                ComparisonOperator,
                                float low = 0.0f, float up = 0.0f);

    void set_formula_constraint(const string&, const vector<float>&);

    void clear_formula_constraints();

    void set_min_feasible_ratio(float);
    void set_max_oversample_factor(Index);
    void set_exploration_ratio(float);

    void set_fixed_history(const Tensor3&);
    void clear_fixed_history();

    void set_iterations(const int);
    void set_zoom_factor(float);
    void set_evaluations_number(const int);
    void set_relative_tolerance(float);
    void set_max_pareto_number(const Index);
    void set_max_total_evaluations(const Index);
    void set_initial_sampling_factor(const Index);

    void set_branch_mode(const BranchMode);

    void set_deformation_domain_factor(float);
    float get_deformation_domain_factor();

    vector<Descriptives> get_descriptives(const string&) const;

    const pair<vector<Variable>, vector<Descriptives>>& get_variables_and_descriptives(const string&) const;

    vector<float> get_utopian_point() const;

    const map<string, vector<Index>>& get_category_frequencies() const noexcept { return sampling_memory.category_frequencies; }

    const vector<Cardinality>& get_cardinality_constraints() const noexcept { return constraint_set.cardinality; }

    pair<Index, VectorR> get_advised_point(const MatrixR&,
                                                         const VectorR& importance_scale = VectorR()) const;

    pair<Index, VectorR> get_robust_point(const MatrixR&, float balance = 0.5f) const;

    Domain get_original_domain(string_view role) const;

    MatrixR calculate_random_inputs(const Domain&, Index evaluations_count = -1) const;

    Lattice build_input_lattice(const vector<Variable>&,
                                const vector<Index>&,
                                const Domain&,
                                map<string, Index>&) const;

    vector<vector<Index>> resolve_cardinality_columns(const Domain&,
                                                      const map<string, Index>&,
                                                      const vector<char>&,
                                                      float,
                                                      MatrixR&) const;

    Tensor3 combine_input(const MatrixR&) const;

    MatrixR calculate_outputs(const MatrixR&) const;

    pair<MatrixR, MatrixR> filter_feasible_points(const MatrixR&,
                                                                const MatrixR&,
                                                                const Domain&) const;

    pair<MatrixR, MatrixR> sample_feasible_points(const Domain&,
                                                                const Domain&,
                                                                const Index evaluations_multiplier = 1) const;

    pair<MatrixR, MatrixR> calculate_optimal_points(const MatrixR&,
                                                                  const MatrixR&,
                                                                  const Objectives&) const;

    pair<MatrixR, MatrixR> calculate_optimal_points(const MatrixR&,
                                                                  const MatrixR&,
                                                                  const Objectives&,
                                                                  const MatrixR&) const;

    pair<MatrixR, MatrixR> calculate_pareto(const MatrixR&, const MatrixR&, const MatrixR&) const;

    pair<float, float> calculate_quality_metrics(const MatrixR&,
                                                               const MatrixR&,
                                                               const Objectives&) const;

    MatrixR perform_single_objective_optimization() const;

    MatrixR perform_multiobjective_optimization() const;

    MatrixR solve_once() const;

    MatrixR perform_response_optimization();

    Index get_objectives_number() const;

    Index get_optimizing_objectives_number() const;

    Index get_evaluations_used() const;

    UnivariateConstraint get_constraint(const string&) const;
    bool is_objective(const string&) const;
    Sense get_sense(const string&) const;
    bool is_history(const string&) const;
    static bool is_past(const TimeType);

    bool is_forecasting() const { return fixed_history.size() > 0; }

    bool row_satisfies_formula_constraints(const VectorR&,
                                                         const VectorR&) const;

    pair<MatrixR, MatrixR> generate_feasible_points(const Domain&,
                                                                  const Domain&,
                                                                  Index) const;

    void initialize_network_differential() const;

    void restore_cardinality_columns(Domain&, const Domain&) const;

    void promote_single_variable_constraints();

    void expand_fixed_objectives();

    vector<char> discrete_column_mask(const vector<Variable>&) const;
       
private:

    NeuralNetwork* neural_network = nullptr;

    ConstraintSet constraint_set;

    map<string, Sense> objectives;

    map<string, float> fixed_values;

    map<string, TimeType> time_roles;

    Tensor3 fixed_history;


    Index evaluations_number = 2000;
    Index max_iterations = 20;
    Index min_iterations = 4;
    Index initial_sampling_factor = 1;
    Index max_pareto_number = 2000;
    Index max_total_evaluations = 0;
    Index max_oversample_factor = 8;
    float zoom_factor = 0.85f;
    float relative_tolerance = 1e-6f;
    float min_feasible_ratio = 0.01f;
    float exploration_ratio = 0.1f;
    float deformation_domain_factor = 1.0f;
    BranchMode branch_mode = BranchMode::Budgeted;


    mutable map<string, pair<vector<Variable>, vector<Descriptives>>> variables_descriptives;

    mutable NetworkJacobian network_jacobian; // @todo not a member but an argument somewhere

    mutable SamplingMemory sampling_memory;
	

    mutable Index evaluations_used = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
