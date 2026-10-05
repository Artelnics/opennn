// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#pragma once

#include "opennn/core/statistics.h"
#include "opennn/core/variable.h"
#include "opennn/core/random_utilities.h"
#include "opennn/network/network.h"
#include "opennn/network/standard_networks.h"
#include "opennn/network/layers/scaling_layer.h"
#include "opennn/network/layers/unscaling_layer.h"
#include "opennn/response_optimization/response_optimization.h"

using namespace opennn;

using Sense = ResponseOptimization::Objective::Sense;
using Condition = ResponseOptimization::Constraint::Condition;


inline vector<Descriptives> make_descriptives(const Index count, const float minimum, const float maximum)
{
    return vector<Descriptives>(size_t(count),
                                Descriptives(minimum,
                                             maximum,
                                             0.5f*(minimum + maximum),
                                             0.25f*(maximum - minimum)));
}


struct MinimalApproximation
{
    unique_ptr<ApproximationNetwork> network;

    MinimalApproximation(const vector<string>& input_names,
                         const vector<string>& output_names,
                         const float input_minimum = 0.0f,
                         const float input_maximum = 10.0f,
                         const float output_minimum = -1.0f,
                         const float output_maximum = 1.0f)
    {
        const Index inputs_number = Index(input_names.size());
        const Index outputs_number = Index(output_names.size());

        network = make_unique<ApproximationNetwork>(Shape{inputs_number}, Shape{4}, Shape{outputs_number});

        vector<Variable> input_variables(static_cast<size_t>(inputs_number));

        for (Index i = 0; i < inputs_number; i++)
        {
            input_variables[size_t(i)].name = input_names[size_t(i)];
            input_variables[size_t(i)].set_role("Input");
            input_variables[size_t(i)].type = VariableType::Numeric;
        }

        network->set_input_variables(input_variables);

        vector<Variable> output_variables(static_cast<size_t>(outputs_number));

        for (Index i = 0; i < outputs_number; i++)
        {
            output_variables[size_t(i)].name = output_names[size_t(i)];
            output_variables[size_t(i)].set_role("Target");
            output_variables[size_t(i)].type = VariableType::Numeric;
        }

        network->set_output_variables(output_variables);

        static_cast<Scaling*>(network->get_first("Scaling"))
            ->set_descriptives(make_descriptives(inputs_number, input_minimum, input_maximum));

        static_cast<Unscaling*>(network->get_first("Unscaling"))
            ->set_descriptives(make_descriptives(outputs_number, output_minimum, output_maximum));
    }
};


struct CategoricalApproximation
{
    unique_ptr<ApproximationNetwork> network;

    CategoricalApproximation(const vector<string>& numeric_names,
                             const string& categorical_name,
                             const vector<string>& categories,
                             const Index outputs_number = 1,
                             const float input_minimum = 0.0f,
                             const float input_maximum = 10.0f)
    {
        const Index numeric_number = Index(numeric_names.size());
        const Index categories_number = Index(categories.size());

        network = make_unique<ApproximationNetwork>(Shape{numeric_number + categories_number},
                                                    Shape{4},
                                                    Shape{outputs_number});

        vector<Variable> input_variables(size_t(numeric_number) + 1);

        for (Index i = 0; i < numeric_number; i++)
        {
            input_variables[size_t(i)].name = numeric_names[size_t(i)];
            input_variables[size_t(i)].set_role("Input");
            input_variables[size_t(i)].type = VariableType::Numeric;
        }

        input_variables.back().name = categorical_name;
        input_variables.back().set_role("Input");
        input_variables.back().type = VariableType::Categorical;
        input_variables.back().set_categories(categories);

        network->set_input_variables(input_variables);

        vector<Variable> output_variables(static_cast<size_t>(outputs_number));

        for (Index i = 0; i < outputs_number; i++)
        {
            output_variables[size_t(i)].name = "y" + to_string(i + 1);
            output_variables[size_t(i)].set_role("Target");
            output_variables[size_t(i)].type = VariableType::Numeric;
        }

        network->set_output_variables(output_variables);

        vector<Descriptives> input_descriptives = make_descriptives(numeric_number,
                                                                    input_minimum,
                                                                    input_maximum);

        const vector<Descriptives> category_descriptives = make_descriptives(categories_number,
                                                                             0.0f,
                                                                             1.0f);

        input_descriptives.insert(input_descriptives.end(),
                                  category_descriptives.begin(),
                                  category_descriptives.end());

        static_cast<Scaling*>(network->get_first("Scaling"))->set_descriptives(input_descriptives);

        static_cast<Unscaling*>(network->get_first("Unscaling"))
            ->set_descriptives(make_descriptives(outputs_number, -1.0f, 1.0f));
    }
};


inline vector<float> scan_categories(Network& network,
                                     const Index numeric_number,
                                     const Index categories_number,
                                     const float input_minimum,
                                     const float input_maximum,
                                     const Index samples_number = 4096)
{
    const Index features_number = numeric_number + categories_number;

    vector<float> best_values(size_t(categories_number), -numeric_limits<float>::max());

    MatrixR inputs(samples_number, features_number);

    for (Index category = 0; category < categories_number; category++)
    {
        set_random_uniform(inputs, input_minimum, input_maximum);

        inputs.rightCols(categories_number).setZero();
        inputs.col(numeric_number + category).setOnes();

        const MatrixR outputs = network.calculate_outputs(inputs);

        for (Index i = 0; i < samples_number; i++)
            best_values[size_t(category)] = max(best_values[size_t(category)], -outputs(i, 0));
    }

    return best_values;
}


inline Index read_category(const MatrixR& results,
                           const Index row,
                           const Index numeric_number,
                           const Index categories_number)
{
    Index category = -1;

    for (Index j = 0; j < categories_number; j++)
    {
        const float value = results(row, numeric_number + j);

        if (value == 0.0f) continue;

        if (value != 1.0f || category >= 0) return -1;

        category = j;
    }

    return category;
}


inline pair<float, float> sample_response(Network& network,
                                          const Index inputs_number,
                                          const float input_minimum,
                                          const float input_maximum,
                                          const Index samples_number = 512)
{
    MatrixR inputs(samples_number, inputs_number);

    set_random_uniform(inputs, input_minimum, input_maximum);

    const MatrixR outputs = network.calculate_outputs(inputs);

    vector<float> values(static_cast<size_t>(samples_number));

    for (Index i = 0; i < samples_number; i++)
        values[size_t(i)] = outputs(i, 0);

    ranges::sort(values);

    return {values[values.size()/2], values.back() - values.front()};
}
