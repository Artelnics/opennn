//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S   H E A D E R   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef RESPONSEOPTIMIZATION_H
#define RESPONSEOPTIMIZATION_H

#pragma once

#include "pch.h"
#include "dataset.h"
#include "statistics.h"

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
        type low_bound;
        type up_bound;

        Condition(ConditionType new_type = ConditionType::None, type low = 0.0, type up = 0.0)
            : condition(new_type), low_bound(low), up_bound(up) {}
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

        void reshape(const type zoom_factor,
                     const VectorR& center,
                     const MatrixR& subset_optimal_points,
                     const vector<Index>& input_feature_dimensions,
                     const vector<Dataset::VariableType>& input_variable_types);

        VectorR inferior_frontier;
        VectorR superior_frontier;
    };

    struct Objectives
    {
        Objectives(const ResponseOptimization& response_optimization);

        MatrixR objective_sources; //Row 0: if is input or not, Row 1 : feature index in input or target subsets

        MatrixR utopian_and_senses; // Row 0: Utopian point, Row 1: Senses of optimization (1 for max, -1 for min)

        MatrixR objective_normalizer; // Row 0: Multipliers (1/range), Row 1: Offsets (-inferior/range)

        MatrixR extract(const MatrixR& inputs, const MatrixR& output) const;

        void normalize(MatrixR& objective_matrix) const;
    };

    Objectives build_objectives() const;

    ResponseOptimization(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void clear_conditions();
    void set_condition(const string& name, const ConditionType condition, type low = 0.0, type up = 0.0);

    void set_iterations(const int iterations);
    void set_zoom_factor(type new_zoom_factor);
    void set_evaluations_number(const int new_evaluations_number);
    void set_relative_tolerance(type new_relative_tolerance);

    vector<type> get_utopian_point() const;

    Domain get_original_domain(const string role) const;

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

    pair<type, type> calculate_quality_metrics(const MatrixR& inputs, const MatrixR& outputs,const Objectives& objectives) const;

    MatrixR perform_multiobjective_optimization(const Objectives& objectives) const;

    MatrixR perform_response_optimization() const;

private:

    NeuralNetwork* neural_network = nullptr;

    Dataset* dataset = nullptr;

    vector<Condition> conditions;

    Index evaluations_number = 1000;

    Index max_iterations = 5;

    Index min_iterations = 3;

    type zoom_factor = type(0.45);

    type relative_tolerance = type(0.001);

    //minimum number of points?
};

}
#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
