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
#include "statistics.h"
#include "variable.h"

namespace opennn
{

class NeuralNetwork;

class ResponseOptimization
{
public:

    enum class ConditionType {None, Between, EqualTo, LessEqualTo, GreaterEqualTo, LessThan, GreaterThan, Minimize, Maximize, Past};

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

        Domain(const vector<Variable>& variables,
               const vector<Descriptives>& descriptives,
               const type deformation_domain_factor = type(1))
        {
            set(variables, descriptives, deformation_domain_factor);
        }

        void set(const vector<Variable>& variables,
                 const vector<Descriptives>& descriptives,
                 const type deformation_domain_factor = type(1));

        void bound(const vector<Variable>& variables, const vector<Condition>& conditions);

        void reshape(const type zoom_factor, const VectorR& center, const MatrixR& points_inputs, const vector<Variable>& vars);

        VectorR inferior_frontier;
        VectorR superior_frontier;

        //MatrixR allowed_values;
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

    ResponseOptimization(NeuralNetwork* = nullptr);

    void set(NeuralNetwork* = nullptr);

    void clear_conditions();
    void clear_conditions(const string& name);

    Condition get_condition(const string& name) const;

    void set_condition(const string& name, const ConditionType condition = ConditionType::None, type low = 0.0, type up = 0.0);

    void set_fixed_history(const Tensor3& history);

    void set_iterations(const int iterations);
    void set_zoom_factor(type new_zoom_factor);
    void set_evaluations_number(const int new_evaluations_number);
    void set_relative_tolerance(type new_relative_tolerance);

    void set_deformation_domain_factor(type new_deformation_domain_factor);
    type get_deformation_domain_factor();

    vector<Descriptives> get_descriptives(const string& ) const;

    pair<vector<Variable>, vector<Descriptives>> get_variables_and_descriptives(const string& role) const;

    vector<type> get_utopian_point() const;

    Domain get_original_domain(const string role) const;

    MatrixR calculate_random_inputs(const Domain& input_domain) const;

    Tensor3 input_constructor(const MatrixR& present_random_values) const;

    MatrixR calculate_outputs(const MatrixR& optimized_variables) const;

    pair<MatrixR, MatrixR> filter_feasible_points(const MatrixR& inputs,
                                                  const MatrixR& outputs,
                                                  const Domain& output_domain) const;

    pair<MatrixR, MatrixR> calculate_optimal_points(const MatrixR& feasible_inputs,
                                                    const MatrixR& feasible_outputs,
                                                    const Objectives& objectives) const;

    MatrixR assemble_results(const MatrixR& inputs, const MatrixR& outputs) const;

    pair<MatrixR, MatrixR> calculate_pareto(const MatrixR& inputs, const MatrixR& outputs, const MatrixR& objective_matrix) const;

    pair<type, type> calculate_quality_metrics(const MatrixR& inputs,
                                               const MatrixR& outputs,
                                               const Objectives& objectives) const;

    MatrixR perform_single_objective_optimization() const;

    MatrixR perform_multiobjective_optimization() const;

    MatrixR perform_response_optimization() const;

    Index get_objectives_number() const
    {
        Index objectives_number = 0;

        for (const auto& [_, constraints] : conditions)
            if (constraints.condition == ConditionType::Maximize || constraints.condition == ConditionType::Minimize)
                objectives_number++;

        return objectives_number;
    }


private:

    NeuralNetwork* neural_network = nullptr;

    map<string, Condition> conditions;

    Index evaluations_number = 2000;

    Index max_iterations = 10;

    Index min_iterations = 4;

    type zoom_factor = type(0.45);

    type relative_tolerance = type(0.001);

    type deformation_domain_factor = type(1);

    //minimum number of points?

    Tensor3 fixed_history; //(1 matrix, time_steps,  features_dimentions )
    //@simone @todo, forecasting start from here
    bool is_forecasting = false;   
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
