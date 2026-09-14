//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S Y N T H E T I C   F I X T U R E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// Small untrained networks built on the spot, and the handful of ways the response tests
// read them. Nothing here is trained: the weights are whatever the constructor gives, so a
// test may ask where the response goes but never what value it takes.
//
// Shared so that the expression tests and the condition tests state a network the same
// way. The trained network lives in concrete_fixture.h instead.

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


// The same range for every variable of a layer, which is all the box these tests need.

inline vector<Descriptives> make_descriptives(const Index count, const float minimum, const float maximum)
{
    return vector<Descriptives>(size_t(count),
                                Descriptives(minimum,
                                             maximum,
                                             0.5f*(minimum + maximum),
                                             0.25f*(maximum - minimum)));
}


// Numeric inputs in one box, numeric outputs in another.

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


// The same, with one categorical input last, so its categories occupy the trailing columns.

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


// The best response the network gives inside each category, found by scanning. What a
// search over the categories has to match.

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


// Which category a result row holds, or -1 if its columns are not one hot.

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


// The median response of an untrained network, and the spread around it. A test that has
// to name a reachable value asks for it here rather than assuming one.

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

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
