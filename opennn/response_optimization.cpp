//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "response_optimization.h"
#include "statistics.h"
#include "scaling_layer_2d.h"
#include "bounding_layer.h"

namespace opennn
{

ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : data_set(new_data_set)
{
    set(new_neural_network, new_data_set);
}


void ResponseOptimization::set(NeuralNetwork* new_neural_network, DataSet* new_data_set)
{   
    neural_network = new_neural_network;
    data_set = new_data_set;

    if(!neural_network) return;

    const Index inputs_number = neural_network->get_inputs_number();
    const Index outputs_number = neural_network->get_outputs_number();

    input_conditions.resize(inputs_number);
    input_conditions.setConstant(Condition::None);

    output_conditions.resize(outputs_number);
    output_conditions.setConstant(Condition::None);

    if(neural_network->has(Layer::Type::Scaling2D))
    {
        ScalingLayer2D* scaling_layer_2d = static_cast<ScalingLayer2D*>(neural_network->get_first(Layer::Type::Scaling2D));

        input_minimums = scaling_layer_2d->get_minimums();
        input_maximums = scaling_layer_2d->get_maximums();
    }

    if(neural_network->get_model_type() == NeuralNetwork::ModelType::Classification || neural_network->get_model_type() == NeuralNetwork::ModelType::Default)
    {
        output_minimums.resize(outputs_number);
        output_minimums.setZero();

        output_maximums.resize(outputs_number);
        output_maximums.setConstant(type(1));
    }

    if(neural_network->has(Layer::Type::Bounding))
    {
        BoundingLayer* bounding_layer = static_cast<BoundingLayer*>(neural_network->get_first(Layer::Type::Bounding));

        output_minimums = bounding_layer->get_lower_bounds();
        output_maximums = bounding_layer->get_upper_bounds();
    }
}


void ResponseOptimization::set_evaluations_number(const Index& new_evaluations_number)
{
    evaluations_number = new_evaluations_number;
}


Tensor<ResponseOptimization::Condition, 1> ResponseOptimization::get_input_conditions() const
{
    return input_conditions;
}


Tensor<ResponseOptimization::Condition, 1> ResponseOptimization::get_output_conditions() const
{
    return output_conditions;
}


Index ResponseOptimization::get_evaluations_number() const
{
    return evaluations_number;
}


Tensor<type, 1> ResponseOptimization::get_input_minimums() const
{
    return input_minimums;
}


Tensor<type, 1> ResponseOptimization::get_input_maximums() const
{
    return input_maximums;
}


Tensor<type, 1> ResponseOptimization::get_outputs_minimums() const
{
    return output_minimums;
}


Tensor<type, 1> ResponseOptimization::get_outputs_maximums() const
{
    return output_maximums;
}

void ResponseOptimization::set_input_condition(const string& name,
                                               const ResponseOptimization::Condition& condition,
                                               const Tensor<type, 1>& values)
{
    const Index index = neural_network->get_input_index(name);

    set_input_condition(index, condition, values);
}


void ResponseOptimization::set_output_condition(const string& name,
                                                const ResponseOptimization::Condition& condition,
                                                const Tensor<type, 1>& values)
{
    const Index index = neural_network->get_output_index(name);

    set_output_condition(index, condition, values);
}


void ResponseOptimization::set_input_condition(const Index& index,
                                               const ResponseOptimization::Condition& condition,
                                               const Tensor<type, 1>& values)
{
    input_conditions[index] = condition;

    switch(condition)
    {
    case Condition::Minimum:
        if(values.size() != 0)
            throw runtime_error("For Minimum condition, size of values must be 0.\n");
        return;

    case Condition::Maximum:
        if(values.size() != 0)
            throw runtime_error("For Maximum condition, size of values must be 0.\n");
        return;

    case Condition::EqualTo:
        if(values.size() != 1)
            throw runtime_error("For LessEqualTo condition, size of values must be 1.\n");

        input_minimums[index] = values[0];
        input_maximums[index] = values[0];

        return;

    case Condition::LessEqualTo:
        if(values.size() != 1)
            throw runtime_error("For LessEqualTo condition, size of values must be 1.\n");

        input_maximums[index] = values[0];

        return;

    case Condition::GreaterEqualTo:
        if(values.size() != 1)
            throw runtime_error("For LessEqualTo condition, size of values must be 1.\n");

        input_minimums[index] = values[0];

        return;

    case Condition::Between:
        if(values.size() != 2)
            throw runtime_error("For Between condition, size of values must be 2.\n");

        input_minimums[index] = values[0];
        input_maximums[index] = values[1];

        return;

    default:
        return;
    }
}


void ResponseOptimization::set_output_condition(const Index& index,
                                                const ResponseOptimization::Condition& condition,
                                                const Tensor<type, 1>& values)
{
    output_conditions[index] = condition;

    switch(condition)
    {
    case Condition::Minimum:
        if(values.size() != 0)
            throw runtime_error("For Minimum condition, size of values must be 0.\n");

        return;

    case Condition::Maximum:
        if(values.size() != 0)
            throw runtime_error("For Maximum condition, size of values must be 0.\n");

        return;

    case Condition::EqualTo:
        throw runtime_error("EqualTo condition is only available for inputs.\n");

    case Condition::LessEqualTo:
        throw runtime_error("For LessEqualTo condition, size of values must be 1.\n");

        output_maximums[index] = values[0];

        return;

    case Condition::GreaterEqualTo:

        if(values.size() != 1)
            throw runtime_error("For GreaterEqualTo condition, size of values must be 1.\n");

        output_minimums[index] = values[0];

        return;

    case Condition::Between:

        if(values.size() != 2)
            throw runtime_error("For Between condition, size of values must be 2.\n");

        output_minimums[index] = values[0];
        output_maximums[index] = values[1];

        return;

    case Condition::None:
        return;

    default:
        return;
    }
}


Tensor<ResponseOptimization::Condition, 1> ResponseOptimization::get_conditions(const vector<string>& conditions_string) const
{
    const Index conditions_number = conditions_string.size();

    Tensor<Condition, 1> conditions(conditions_number);

    for(Index i = 0; i < conditions_number; i++)
        if(conditions_string[i] == "Minimize" || conditions_string[i] == "Minimum")
            conditions[i] = Condition::Minimum;
        else if(conditions_string[i] == "Maximize" || conditions_string[i] == "Maximum")
            conditions[i] = Condition::Maximum;
        else if(conditions_string[i] == "="|| conditions_string[i] == "EqualTo")
            conditions[i] = Condition::EqualTo;
        else if(conditions_string[i] == "Between")
            conditions[i] = Condition::Between;
        else if(conditions_string[i] == ">="
             || conditions_string[i] == ">"
             || conditions_string[i] == "GreaterEqualTo"
             || conditions_string[i] == "GreaterThan")
            conditions[i] = Condition::GreaterEqualTo;
        else if(conditions_string[i] == "<="
             || conditions_string[i] == "<"
             || conditions_string[i] == "LessEqualTo"
             || conditions_string[i] == "LessThan")
            conditions[i] = Condition::LessEqualTo;
        else
            conditions[i] = Condition::None;

    return conditions;
}


Tensor<Tensor<type, 1>, 1> ResponseOptimization::get_values_conditions(const Tensor<ResponseOptimization::Condition, 1>& conditions, 
                                                                       const Tensor<type, 1>& values) const
{
    const Index conditions_size = conditions.size();

    Tensor<Tensor<type, 1>, 1> values_conditions(conditions_size);

    Index index = 0;
    for(Index i = 0; i < conditions_size; i++)
    {
        Tensor<type, 1> current_values;

        const Condition current_condition = conditions[i];

        switch(current_condition)
        {
        case Condition::Minimum:
        case Condition::Maximum:
            values_conditions[i].resize(0);
            index++;
            break;

        case Condition::EqualTo:
        case Condition::LessEqualTo:
        case Condition::GreaterEqualTo:
            current_values.resize(1);
            current_values[0] = values[index++];
            values_conditions[i] = current_values;
            break;

        case Condition::Between:
            current_values.resize(2);
            current_values[0] = values[index++];
            current_values[1] = values[index++];
            values_conditions[i] = current_values;
            break;

        case Condition::None:

            current_values.resize(2);

            if(i < input_minimums.size())
            {
                current_values[0] = input_minimums(i);
                current_values[1] = input_maximums(i);
            }
            else
            {
                current_values[0] = output_minimums(i);
                current_values[1] = output_maximums(i);
            }

            values_conditions[i] = current_values;
            index++;
            break;
        }
    }

    return values_conditions;
}


Tensor<type, 2> ResponseOptimization::calculate_inputs() const
{
    const Index inputs_number = neural_network->get_inputs_number();

    Tensor<type, 2> inputs(evaluations_number, inputs_number);
    inputs.setZero();

    const int input_raw_variables_number = data_set->get_raw_variables_number(DataSet::VariableUse::Input);

    vector<Index> used_raw_variables_indices = data_set->get_used_raw_variables_indices();

    for(Index i = 0; i < evaluations_number; i++)
    {
        Index used_raw_variable_index = 0;

        Index index = 0;

        for(Index j = 0; j < input_raw_variables_number; j++)
        {
            used_raw_variable_index = used_raw_variables_indices[j];

            const DataSet::RawVariableType raw_variable_type = data_set->get_raw_variable_type(used_raw_variable_index);

            if(raw_variable_type == DataSet::RawVariableType::Numeric
            || raw_variable_type == DataSet::RawVariableType::Constant)
            {
                inputs(i, index) = get_random_type(input_minimums[index], input_maximums[index]);
                index++;
            }
            else if(raw_variable_type == DataSet::RawVariableType::Binary)
            {
                inputs(i, index) = (input_conditions(index) == ResponseOptimization::Condition::EqualTo)
                    ? input_minimums[index]
                    : type(rand() % 2);

                index++;
            }
            else if(raw_variable_type == DataSet::RawVariableType::Categorical)
            {
                const Index categories_number = data_set->get_raw_variables()[used_raw_variable_index].get_categories_number();
                Index equal_index = -1;

                for(Index k = 0; k < categories_number; k++)
                {
                    inputs(i,index + k) = type(0);

                    if(input_conditions(index + k) == ResponseOptimization::Condition::EqualTo)
                    {
                        inputs(i,index + k) = input_minimums(index +k);

                        if(inputs(i, index + k) == 1)
                            equal_index = k;
                    }
                }

                if(equal_index == -1)
                    inputs(i, index + rand() % categories_number) = type(1);

                index += categories_number;
            }
            else
            {
                inputs(i, index) = get_random_type(input_minimums[index], input_maximums[index]);
                index++;
            }
        }
    }

    return inputs;
}


Tensor<type, 2> ResponseOptimization::calculate_envelope(const Tensor<type, 2>& inputs, const Tensor<type, 2>& outputs) const
{
    const Index inputs_number = neural_network->get_inputs_number();
    const Index outputs_number = neural_network->get_outputs_number();

    Tensor<type, 2> envelope = assemble_matrix_matrix(inputs,outputs);

    for(Index i = 0; i < outputs_number; i++)
        if(envelope.size() != 0)
            envelope = filter_column_minimum_maximum(envelope, inputs_number + i, output_minimums(i), output_maximums(i));
        else
            return Tensor<type,2>();

    return envelope;
}


ResponseOptimizationResults* ResponseOptimization::perform_optimization() const
{
    ResponseOptimizationResults* results = new ResponseOptimizationResults(neural_network);

    const Tensor<type, 2> inputs = calculate_inputs();

    const Tensor<type, 2> outputs = neural_network->calculate_outputs(inputs);

    const Tensor<type, 2> envelope = calculate_envelope(inputs, outputs);

    const Index samples_number = envelope.dimension(0);

    const Index inputs_number = neural_network->get_inputs_number();
    const Index outputs_number = neural_network->get_outputs_number();

    Tensor<type, 1> objective(samples_number);
    objective.setZero();

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < inputs_number; j++)
            if(input_conditions[j] == Condition::Minimum)
                objective[i] += envelope(i, j);
            else if(input_conditions[j] == Condition::Maximum)
                objective[i] += -envelope(i, j);

        for(Index j = 0; j < outputs_number; j++)
            if(output_conditions[j] == Condition::Minimum)
                objective[i] += envelope(i, inputs_number+j);
            else if(output_conditions[j] == Condition::Maximum)
                objective[i] += -envelope(i, inputs_number+j);
    }

    const Index optimal_index = minimal_index(objective);

    results->optimal_variables = (envelope.size() != 0)
        ? envelope.chip(optimal_index, 0)
        : Tensor<type, 1>();

    return results;
}


ResponseOptimizationResults::ResponseOptimizationResults(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
}


void ResponseOptimizationResults::print() const
{
    const Index inputs_number = neural_network->get_inputs_number();
    const Index outputs_number = neural_network->get_outputs_number();

    const vector<string> input_names = neural_network->get_input_names();
    const vector<string> output_names = neural_network->get_output_names();

    cout << "\nResponse optimization results: " << endl;

    if(optimal_variables.size() == 0)
        throw runtime_error("Optimal variables vector is empty.\n");

    for(Index i = 0; i < inputs_number; i++)
        cout << input_names[i] << ": " << optimal_variables[i] << endl;

    for(Index i = 0; i < outputs_number; i++)
        cout << output_names[i] << ": " << optimal_variables[inputs_number + i] << endl;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
