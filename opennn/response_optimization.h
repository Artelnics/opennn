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

class ResponseOptimization
{
public:

    enum class ConditionType {None, Between, EqualTo, LessEqualTo, GreaterEqualTo, LessThan, GreaterThan, Minimize, Maximize};

    struct Condition
    {
        ConditionType condition;
        float low_bound;
        float up_bound;

        Condition(ConditionType new_type = ConditionType::None, float new_low_bound = 0.0, float new_up_bound = 0.0)
            : condition(new_type), low_bound(new_low_bound), up_bound(new_up_bound) {}
    };

    struct Domain
    {
        Domain() = default;
        virtual ~Domain() = default;

        Domain(const vector<Index>& feature_dimensions, const vector<Descriptives>& descriptives)
        {
            set(feature_dimensions, descriptives);
        }

        void set(const vector<Index>& feature_dimensions, const vector<Descriptives>& descriptives);

        void bound(const vector<Index>& feature_dimensions, const vector<Condition>& conditions);

        void reshape(const float zoom_factor,
                     const VectorR& center,
                     const MatrixR& subset_optimal_points,
                     const vector<Index>& input_feature_dimensions,
                     const vector<VariableType>& input_variable_types);

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
    };

    Objectives build_objectives() const;

    ResponseOptimization(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void clear_conditions();
    void set_condition(const string& name, const ConditionType condition, float low_bound = 0.0, float up_bound = 0.0);

    void set_iterations(const int iterations);
    void set_zoom_factor(float new_zoom_factor);
    void set_evaluations_number(const int new_evaluations_number);
    void set_relative_tolerance(float new_relative_tolerance);

    vector<float> get_utopian_point() const;

    Domain get_original_domain(const string role) const;

    Condition get_condition(const Index index) const;

    MatrixR calculate_random_inputs(const Domain& input_domain) const;

    pair<MatrixR, MatrixR> filter_feasible_points(const MatrixR& inputs,
                                                  const MatrixR& outputs,
                                                  const Domain& output_domain) const;

    pair<MatrixR, MatrixR> calculate_optimal_points(const MatrixR& feasible_inputs,
                                                    const MatrixR& feasible_outputs,
                                                    const Objectives& objectives) const;

    MatrixR assemble_results(const MatrixR& inputs, const MatrixR& outputs) const;

    MatrixR perform_single_objective_optimization(const Objectives& objectives) const;

    pair<MatrixR, MatrixR> calculate_pareto(const MatrixR& inputs, const MatrixR& outputs, const MatrixR& objective_matrix) const;

    pair<float, float> calculate_quality_metrics(const MatrixR& inputs, const MatrixR& outputs,const Objectives& objectives) const;

    MatrixR perform_multiobjective_optimization(const Objectives& objectives) const;

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
