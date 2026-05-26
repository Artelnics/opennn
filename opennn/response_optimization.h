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
#include "formula_expression.h"

namespace opennn
{

class NeuralNetwork;
class Dataset;

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

    struct FormulaConstraint
    {
        string expression;
        function<float(const VectorR&, const VectorR&)> callback;
        bool uses_callback = false;

        ConditionType op = ConditionType::None;
        float low_bound = 0.0f;
        float up_bound = 0.0f;

        CompiledFormula compiled;
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

    ResponseOptimization(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void clear_conditions();
    void clear_conditions(const string& name);

    [[nodiscard]] Condition get_condition(const string& name) const;

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

    void set_deformation_domain_factor(float new_deformation_domain_factor);
    [[nodiscard]] float get_deformation_domain_factor();

    [[nodiscard]] vector<Descriptives> get_descriptives(const string& role) const;

    [[nodiscard]] pair<vector<Variable>, vector<Descriptives>> get_variables_and_descriptives(const string& role) const;

    [[nodiscard]] vector<float> get_utopian_point() const;

    [[nodiscard]] pair<Index, VectorR> get_advised_point(const MatrixR& pareto_front,
                                                         const VectorR& importance_scale = VectorR()) const;

    [[nodiscard]] Domain get_original_domain(const string role) const;

    [[nodiscard]] MatrixR calculate_random_inputs(const Domain& input_domain, Index evaluations_count = -1) const;

    [[nodiscard]] Tensor3 combine_input(const MatrixR& present_random_values) const;

    [[nodiscard]] MatrixR calculate_outputs(const MatrixR& input) const;

    [[nodiscard]] pair<MatrixR, MatrixR> filter_feasible_points(const MatrixR& inputs,
                                                                const MatrixR& outputs,
                                                                const Domain& output_domain) const;

    [[nodiscard]] pair<MatrixR, MatrixR> sample_feasible_points(const Domain& input_domain,
                                                                const Domain& output_domain) const;

    [[nodiscard]] pair<MatrixR, MatrixR> calculate_optimal_points(const MatrixR& feasible_inputs,
                                                                  const MatrixR& feasible_outputs,
                                                                  const Objectives& objectives) const;

    [[nodiscard]] MatrixR assemble_results(const MatrixR& inputs, const MatrixR& outputs) const;

    [[nodiscard]] pair<MatrixR, MatrixR> calculate_pareto(const MatrixR& inputs, const MatrixR& outputs, const MatrixR& objective_matrix) const;

    [[nodiscard]] pair<float, float> calculate_quality_metrics(const MatrixR& inputs,
                                                               const MatrixR& outputs,
                                                               const Objectives& objectives) const;

    [[nodiscard]] MatrixR perform_single_objective_optimization() const;

    [[nodiscard]] MatrixR perform_multiobjective_optimization() const;

    [[nodiscard]] MatrixR perform_response_optimization() const;

    [[nodiscard]] Index get_objectives_number() const;

private:

    [[nodiscard]] vector<NamedColumn> build_input_columns_for_formula() const;
    [[nodiscard]] vector<NamedColumn> build_output_columns_for_formula() const;

    void apply_affine_input_swap(MatrixR& random_inputs,
                                 const FormulaConstraint& formula_constraint,
                                 const Domain& input_domain) const;

    [[nodiscard]] bool row_satisfies_formula_constraints(const VectorR& input_row,
                                                         const VectorR& output_row) const;

    [[nodiscard]] pair<MatrixR, MatrixR> generate_feasible_points(const Domain& input_domain,
                                                                  const Domain& output_domain,
                                                                  Index evaluations_count) const;

    NeuralNetwork* neural_network = nullptr;

    Dataset* dataset = nullptr;

    map<string, Condition> conditions;

    vector<FormulaConstraint> formula_constraints;

    float min_feasible_ratio = 0.01f;
    Index max_oversample_factor = 8;

    // Defaults set 2026-05-25 to the best 40k-budget config from the
    // IDC_benchmark grid search (e2000_i20_z85). The 160k-budget grid
    // winner (e8000_i20_z85) was the briefly-promoted canonical 2026-05-22,
    // but matched-budget pymoo comparison at 160k is wall-clock-prohibitive
    // (~17h for the full 23-entry holdout sweep), so we settled on the
    // best 40k shape — still uses the grid-confirmed gentle zoom 0.85 and
    // tightened tolerance 1e-6, just with a budget that lets pymoo keep up.
    Index evaluations_number = 2000;

    Index max_iterations = 20;

    Index min_iterations = 4;

    float zoom_factor = 0.85f;

    float relative_tolerance = 1e-6f;

    float deformation_domain_factor = 1.0f;

    Tensor3 fixed_history;

    bool is_forecasting = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
