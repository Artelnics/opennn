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
#include "constraints_utilities.h"

namespace opennn
{

class NeuralNetwork;
class Dataset;

// Analytic snapshot of a feedforward surrogate (Scaling / Dense / Unscaling):
// reproduces forward(x) and the exact reverse-mode input-VJP (df/dx)^T v from
// the network's public getters, so the output-constraint repair gets an exact,
// cheap Jacobian. Defined in the .cpp; rebuilt per optimization run.
struct NetworkDifferential;

class ResponseOptimization
{
public:

    enum class ConditionType {None, Between, EqualTo, LessEqualTo, GreaterEqualTo, LessThan, GreaterThan, Minimize, Maximize, Past};

    struct Condition
    {
        ConditionType condition;
        float low_bound;
        float up_bound;

        Condition(ConditionType new_type = ConditionType::None, float new_low_bound = 0.0f, float new_up_bound = 0.0f)
            : condition(new_type), low_bound(new_low_bound), up_bound(new_up_bound) {}
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

        void bound(const vector<Variable>& variables, const vector<Condition>& conditions);

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

        MatrixR objective_sources;

        MatrixR utopian_and_senses;

        MatrixR objective_normalizer;

        MatrixR extract(const MatrixR& inputs, const MatrixR& output) const;

        void normalize(MatrixR& objective_matrix) const;

        bool update_utopian_from_points(const MatrixR& unnormalized_objective_values);
    };

    ResponseOptimization(NeuralNetwork* = nullptr);

    ~ResponseOptimization();

    void set(NeuralNetwork* = nullptr);

    void clear_conditions();
    void clear_conditions(const string& name);

    Condition get_condition(const string& name) const;

    void set_condition(const string& name, const ConditionType condition = ConditionType::None, float low = 0.0f, float up = 0.0f);

    void set_formula_constraint(const string& expression,
                                ConditionType op,
                                float low = 0.0f, float up = 0.0f);

    void set_formula_constraint(function<float(const VectorR&, const VectorR&)> callback,
                                ConditionType op,
                                float low = 0.0f, float up = 0.0f);

    void clear_formula_constraints();

    void set_min_feasible_ratio(float new_ratio);
    void set_max_oversample_factor(Index new_factor);

    void set_fixed_history(const Tensor3& history);

    void set_iterations(const int iterations);
    void set_zoom_factor(float new_zoom_factor);
    void set_evaluations_number(const int new_evaluations_number);
    void set_relative_tolerance(float new_relative_tolerance);
    void set_max_pareto_number(const Index new_max_pareto_number);
    void set_max_total_evaluations(const Index new_max_total_evaluations);
    void set_initial_sampling_factor(const Index new_initial_sampling_factor);

    void set_deformation_domain_factor(float new_deformation_domain_factor);
    float get_deformation_domain_factor();

    vector<Descriptives> get_descriptives(const string& role) const;

    pair<vector<Variable>, vector<Descriptives>> get_variables_and_descriptives(const string& role) const;

    vector<float> get_utopian_point() const;

    pair<Index, VectorR> get_advised_point(const MatrixR& pareto_front,
                                                         const VectorR& importance_scale = VectorR()) const;

    Domain get_original_domain(const string role) const;

    MatrixR calculate_random_inputs(const Domain& input_domain, Index evaluations_count = -1) const;

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

    MatrixR perform_response_optimization() const;

    Index get_objectives_number() const;

private:

    vector<NamedColumn> build_input_columns_for_formula() const;
    vector<NamedColumn> build_output_columns_for_formula() const;

    bool row_satisfies_formula_constraints(const VectorR& input_row,
                                                         const VectorR& output_row) const;

    pair<MatrixR, MatrixR> generate_feasible_points(const Domain& input_domain,
                                                                  const Domain& output_domain,
                                                                  Index evaluations_count) const;

    // Rebuilds + self-validates the analytic surrogate Jacobian for the current
    // network. Leaves network_differential null (finite-difference fallback) when
    // there is no output constraint, the network is forecasting/unsupported, or
    // the analytic forward fails to match calculate_outputs.
    void initialize_network_differential() const;

    NeuralNetwork* neural_network = nullptr;

    mutable unique_ptr<NetworkDifferential> network_differential;

    map<string, Condition> conditions;

    vector<FormulaConstraint> formula_constraints;

    float min_feasible_ratio = 0.01f;
    Index max_oversample_factor = 8;

    Index evaluations_number = 2000;

    Index max_iterations = 20;

    Index min_iterations = 4;

    float zoom_factor = 0.85f;

    float relative_tolerance = 1e-6f;

    Index max_pareto_number = 10000;

    // Optional hard cap on the TOTAL number of surrogate evaluations spent
    // across the whole run (initial sampling + every per-Pareto-point local
    // sampling in every iteration). 0 = unlimited (default; preserves the
    // original behaviour exactly). When > 0, the MO/SO loop stops launching
    // new sampling calls once `evaluations_used` reaches this budget, then
    // returns the best front found so far. Used to run IDC under a matched
    // surrogate-evaluation budget against population-based baselines.
    Index max_total_evaluations = 0;
    mutable Index evaluations_used = 0;

    // Multiplier on the candidate count of the FIRST (initial, full-domain)
    // multi-objective sampling only: the initial pass draws
    // evaluations_number * initial_sampling_factor candidates, while every
    // per-Pareto-point local sampling keeps the base evaluations_number. A
    // larger initial set gives a broader domain to seed the contraction from.
    // 1 = unchanged (default; preserves the original behaviour exactly). The
    // extra initial cost is counted against max_total_evaluations like any
    // other sampling, so the matched budget still holds.
    Index initial_sampling_factor = 1;

    float deformation_domain_factor = 1.0f;

    Tensor3 fixed_history;

    bool is_forecasting = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
