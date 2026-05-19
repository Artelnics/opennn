//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "dataset.h"
#include "statistics.h"
#include "variable.h"

namespace opennn
{

class NeuralNetwork;

/// @brief Optimizes input values so that a network's outputs satisfy user-defined conditions and objectives.
class ResponseOptimization
{
public:

    /// @brief Types of constraint or objective applied to an input or output variable.
    enum class ConditionType {None, Between, EqualTo, LessEqualTo, GreaterEqualTo, LessThan, GreaterThan, Minimize, Maximize};

    /// @brief Constraint or objective imposed on a single variable, with optional bounds.
    struct Condition
    {
        ConditionType condition;
        float low_bound;
        float up_bound;

        /// @brief Builds a condition of the given type with the given bounds.
        Condition(ConditionType new_type = ConditionType::None, float new_low_bound = 0.0, float new_up_bound = 0.0)
            : condition(new_type), low_bound(new_low_bound), up_bound(new_up_bound) {}
    };

    /// @brief Bounded domain in input or output space defined by inferior and superior frontiers.
    struct Domain
    {
        Domain() = default;
        virtual ~Domain() = default;

        /// @brief Builds a domain from feature dimensions and per-feature descriptives.
        Domain(const vector<Index>& feature_dimensions, const vector<Descriptives>& descriptives)
        {
            set(feature_dimensions, descriptives);
        }

        /// @brief Initializes the frontiers from feature dimensions and per-feature descriptives.
        void set(const vector<Index>& feature_dimensions, const vector<Descriptives>& descriptives);

        /// @brief Tightens the frontiers using the supplied per-variable conditions.
        void bound(const vector<Index>& feature_dimensions, const vector<Condition>& conditions);

        /// @brief Zooms and recenters the domain around the supplied optimal points for the next iteration.
        void reshape(const float zoom_factor,
                     const VectorR& center,
                     const MatrixR& subset_optimal_points,
                     const vector<Index>& input_feature_dimensions,
                     const vector<VariableType>& input_variable_types);

        VectorR inferior_frontier;
        VectorR superior_frontier;
    };

    /// @brief Encodes the objectives extracted from the response optimization configuration.
    struct Objectives
    {
        /// @brief Builds the objective matrices from the parent response optimization.
        Objectives(const ResponseOptimization& response_optimization);

        MatrixR objective_sources;

        MatrixR utopian_and_senses;

        MatrixR objective_normalizer;

        /// @brief Extracts the objective values for the given inputs and outputs.
        MatrixR extract(const MatrixR& inputs, const MatrixR& output) const;

        /// @brief Normalizes the objective matrix in place using the precomputed normalizer.
        void normalize(MatrixR& objective_matrix) const;
    };

    /// @brief Builds the Objectives helper from the current conditions.
    Objectives build_objectives() const;

    /// @brief Constructs the optimizer bound to an optional neural network and dataset.
    ResponseOptimization(NeuralNetwork* = nullptr, Dataset* = nullptr);

    /// @brief Binds the optimizer to a neural network and a dataset.
    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);

    /// @brief Removes all conditions previously configured on input and output variables.
    void clear_conditions();

    /// @brief Adds or replaces the condition applied to the variable with the given name.
    /// @param name Variable name as defined in the dataset.
    /// @param condition Condition type to enforce.
    /// @param low_bound Lower bound used by bounded condition types.
    /// @param up_bound Upper bound used by bounded condition types.
    void set_condition(const string& name, const ConditionType condition, float low_bound = 0.0, float up_bound = 0.0);

    void set_iterations(const int iterations);
    void set_zoom_factor(float new_zoom_factor);
    void set_evaluations_number(const int new_evaluations_number);
    void set_relative_tolerance(float new_relative_tolerance);

    /// @brief Returns the coordinates of the utopian point used as reference for multiobjective optimization.
    vector<float> get_utopian_point() const;

    /// @brief Returns the original (untrimmed) domain for variables playing the given role.
    Domain get_original_domain(const string role) const;

    /// @brief Returns the configured condition at the given index.
    Condition get_condition(const Index index) const;

    /// @brief Draws a random sample of input points within the given input domain.
    MatrixR calculate_random_inputs(const Domain& input_domain) const;

    /// @brief Filters the (inputs, outputs) pairs whose outputs fall inside the feasible output domain.
    pair<MatrixR, MatrixR> filter_feasible_points(const MatrixR& inputs,
                                                  const MatrixR& outputs,
                                                  const Domain& output_domain) const;

    /// @brief Selects the optimal points among feasible candidates according to the supplied objectives.
    pair<MatrixR, MatrixR> calculate_optimal_points(const MatrixR& feasible_inputs,
                                                    const MatrixR& feasible_outputs,
                                                    const Objectives& objectives) const;

    /// @brief Assembles the final results matrix concatenating inputs and outputs.
    MatrixR assemble_results(const MatrixR& inputs, const MatrixR& outputs) const;

    /// @brief Performs single-objective optimization and returns the best input/output pair.
    MatrixR perform_single_objective_optimization(const Objectives& objectives) const;

    /// @brief Computes the Pareto front of the supplied input/output samples.
    pair<MatrixR, MatrixR> calculate_pareto(const MatrixR& inputs, const MatrixR& outputs, const MatrixR& objective_matrix) const;

    /// @brief Computes quality metrics of the optimization (e.g. distance to utopian point).
    pair<float, float> calculate_quality_metrics(const MatrixR& inputs, const MatrixR& outputs,const Objectives& objectives) const;

    /// @brief Performs multiobjective optimization and returns the Pareto-optimal input/output samples.
    MatrixR perform_multiobjective_optimization(const Objectives& objectives) const;

    /// @brief Runs the response optimization using the configured conditions and returns the optimal results.
    MatrixR perform_response_optimization() const;

private:

    NeuralNetwork* neural_network = nullptr;

    Dataset* dataset = nullptr;

    vector<Condition> conditions;

    Index evaluations_number = 2000;

    Index max_iterations = 10;

    Index min_iterations = 4;

    float zoom_factor = 0.45f;

    float relative_tolerance = 0.001f;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
