//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "response_optimization.h"

namespace opennn
{

/// Default constructor.
/// It creates a scaling layer object with no scaling neurons.

ResponseOptimization::ResponseOptimization()
{
}

ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : data_set_pointer(new_data_set_pointer)
{
    set(new_neural_network_pointer);
}


ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network_pointer)
    : neural_network_pointer(new_neural_network_pointer)
{
    const Index inputs_number = neural_network_pointer->get_inputs_number();
    const Index outputs_number = neural_network_pointer->get_outputs_number();

    inputs_conditions.resize(inputs_number);
    inputs_conditions.setConstant(Condition::None);

    outputs_conditions.resize(outputs_number);
    outputs_conditions.setConstant(Condition::None);

    inputs_minimums = neural_network_pointer->get_scaling_layer_pointer()->get_minimums();
    inputs_maximums = neural_network_pointer->get_scaling_layer_pointer()->get_maximums();

    if(neural_network_pointer->get_last_trainable_layer_pointer()->get_type() == Layer::Type::Probabilistic) // Classification case
    {

        outputs_minimums.resize(outputs_number);
        outputs_minimums.setZero();

        outputs_maximums.resize(outputs_number);
        outputs_maximums.setConstant({type(1)});
    }
    else // Approximation and forecasting
    {
        outputs_minimums = neural_network_pointer->get_bounding_layer_pointer()->get_lower_bounds();
        outputs_maximums = neural_network_pointer->get_bounding_layer_pointer()->get_upper_bounds();
    }
}


void ResponseOptimization::set(NeuralNetwork* new_neural_network_pointer)
{
    neural_network_pointer = new_neural_network_pointer;

    const Index inputs_number = neural_network_pointer->get_inputs_number();
    const Index outputs_number = neural_network_pointer->get_outputs_number();

    inputs_conditions.resize(inputs_number);
    inputs_conditions.setConstant(Condition::None);

    outputs_conditions.resize(outputs_number);
    outputs_conditions.setConstant(Condition::None);

    inputs_minimums = neural_network_pointer->get_scaling_layer_pointer()->get_minimums();
    inputs_maximums = neural_network_pointer->get_scaling_layer_pointer()->get_maximums();

    if(neural_network_pointer->get_last_trainable_layer_pointer()->get_type() == Layer::Type::Probabilistic) // Classification case
    {

        outputs_minimums.resize(outputs_number);
        outputs_minimums.setZero();

        outputs_maximums.resize(outputs_number);
        outputs_maximums.setConstant({type(1)});
    }
    else // Approximation and forecasting
    {
        outputs_minimums = neural_network_pointer->get_bounding_layer_pointer()->get_lower_bounds();
        outputs_maximums = neural_network_pointer->get_bounding_layer_pointer()->get_upper_bounds();
    }
}


void ResponseOptimization::set_evaluations_number(const Index& new_evaluations_number)
{
    evaluations_number = new_evaluations_number;
}


Tensor<ResponseOptimization::Condition, 1> ResponseOptimization::get_inputs_conditions() const
{
    return inputs_conditions;
}


Tensor<ResponseOptimization::Condition, 1> ResponseOptimization::get_outputs_conditions() const
{
    return outputs_conditions;
}


Index ResponseOptimization::get_evaluations_number() const
{
    return evaluations_number;
}

Tensor<type, 1> ResponseOptimization::get_inputs_minimums() const
{
    return inputs_minimums;
}


Tensor<type, 1> ResponseOptimization::get_inputs_maximums() const
{
    return inputs_maximums;
}


Tensor<type, 1> ResponseOptimization::get_outputs_minimums() const
{
    return outputs_minimums;
}


Tensor<type, 1> ResponseOptimization::get_outputs_maximums() const
{
    return outputs_maximums;
}

void ResponseOptimization::set_input_condition(const string& name,
                                               const ResponseOptimization::Condition& condition,
                                               const Tensor<type, 1>& values)
{
    const Index index = neural_network_pointer->get_input_index(name);

    set_input_condition(index, condition, values);
}


void ResponseOptimization::set_output_condition(const string& name, const ResponseOptimization::Condition& condition, const Tensor<type, 1>& values)
{
    const Index index = neural_network_pointer->get_output_index(name);

    set_output_condition(index, condition, values);
}


void ResponseOptimization::set_input_condition(const Index& index, const ResponseOptimization::Condition& condition, const Tensor<type, 1>& values)
{
    inputs_conditions[index] = condition;

    ostringstream buffer;

    switch(condition)
    {
    case Condition::Minimum:

        if(values.size() != 0)
        {
            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_input_condition() method.\n"
                   << "For Minimum condition, size of values must be 0.\n";

            throw invalid_argument(buffer.str());
        }

        return;

    case Condition::Maximum:

        if(values.size() != 0)
        {
            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_input_condition() method.\n"
                   << "For Maximum condition, size of values must be 0.\n";

            throw invalid_argument(buffer.str());
        }

        return;

    case Condition::EqualTo:

        if(values.size() != 1)
        {
            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_input_condition() method.\n"
                   << "For LessEqualTo condition, size of values must be 1.\n";

            throw invalid_argument(buffer.str());
        }

        inputs_minimums[index] = values[0];
        inputs_maximums[index] = values[0];

        return;

    case Condition::LessEqualTo:

        if(values.size() != 1)
        {
            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_input_condition() method.\n"
                   << "For LessEqualTo condition, size of values must be 1.\n";

            throw invalid_argument(buffer.str());
        }

        inputs_maximums[index] = values[0];

        return;

    case Condition::GreaterEqualTo:

        if(values.size() != 1)
        {
            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_input_condition() method.\n"
                   << "For LessEqualTo condition, size of values must be 1.\n";

            throw invalid_argument(buffer.str());
        }

        inputs_minimums[index] = values[0];

        return;

    case Condition::Between:

        if(values.size() != 2)
        {
            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_input_condition() method.\n"
                   << "For Between condition, size of values must be 2.\n";

            throw invalid_argument(buffer.str());
        }

        inputs_minimums[index] = values[0];
        inputs_maximums[index] = values[1];

        return;
    default:
        return;
    }
}


void ResponseOptimization::set_output_condition(const Index& index, const ResponseOptimization::Condition& condition, const Tensor<type, 1>& values)
{
    outputs_conditions[index] = condition;

    ostringstream buffer;

    switch(condition)
    {
    case Condition::Minimum:

        if(values.size() != 0)
        {
            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_output_condition() method.\n"
                   << "For Minimum condition, size of values must be 0.\n";

            throw invalid_argument(buffer.str());
        }

        return;

    case Condition::Maximum:

        if(values.size() != 0)
        {
            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_output_condition() method.\n"
                   << "For Maximum condition, size of values must be 0.\n";

            throw invalid_argument(buffer.str());
        }

        return;

    case Condition::EqualTo:

        buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_output_condition() method.\n"
                   << "EqualTo condition is only available for inputs.\n";

        throw invalid_argument(buffer.str());

    case Condition::LessEqualTo:

        if(values.size() != 1)
        {
            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_output_condition() method.\n"
                   << "For LessEqualTo condition, size of values must be 1.\n";

            throw invalid_argument(buffer.str());
        }

        outputs_maximums[index] = values[0];

        return;

    case Condition::GreaterEqualTo:

        if(values.size() != 1)
        {
            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_output_condition() method.\n"
                   << "For GreaterEqualTo condition, size of values must be 1.\n";

            throw invalid_argument(buffer.str());
        }

        outputs_minimums[index] = values[0];

        return;

    case Condition::Between:

        if(values.size() != 2)
        {
            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void set_output_condition() method.\n"
                   << "For Between condition, size of values must be 2.\n";

            throw invalid_argument(buffer.str());
        }

        outputs_minimums[index] = values[0];
        outputs_maximums[index] = values[1];

        return;

    case Condition::None:
        return;

    default:
        return;
    }
}


void ResponseOptimization::set_inputs_outputs_conditions(const Tensor<string, 1>& names,
                                                         const Tensor<string, 1>& conditions_string,
                                                         const Tensor<type, 1>& values)
{
    const Tensor<Condition, 1> conditions = get_conditions(conditions_string);

    const Tensor<Tensor<type, 1>, 1> values_conditions = get_values_conditions(conditions, values);

    const Index variables_number = conditions_string.size();

    const Tensor<string, 1> inputs_names = data_set_pointer->get_input_variables_names();

    const Tensor<string, 1> outputs_names = data_set_pointer->get_target_variables_names();

    Index index;

    for(Index i = 0; i < variables_number; i ++)
    {
        if(contains(inputs_names,names[i]))
        {
            index = neural_network_pointer->get_input_index(names[i]);

            set_input_condition(index, conditions[i], values_conditions[i]);
        }
        else if(contains(outputs_names,names[i]))
        {
            index = neural_network_pointer->get_output_index(names[i]);

            set_output_condition(index, conditions[i], values_conditions[i]);
        }
    }
}


Tensor<ResponseOptimization::Condition, 1> ResponseOptimization::get_conditions(const Tensor<string, 1>& conditions_string) const
{
    const Index conditions_size = conditions_string.size();

    Tensor<Condition, 1> conditions(conditions_size);

    for(Index i = 0; i < conditions_size; i++)
    {
        if(conditions_string[i] == "Minimize" || conditions_string[i] == "Minimum")
        {
            conditions[i] = Condition::Minimum;
        }
        else if(conditions_string[i] == "Maximize" || conditions_string[i] == "Maximum")
        {
            conditions[i] = Condition::Maximum;
        }
        else if(conditions_string[i] == "="|| conditions_string[i] == "EqualTo")
        {
            conditions[i] = Condition::EqualTo;
        }
        else if(conditions_string[i] == "Between")
        {
            conditions[i] = Condition::Between;
        }
        else if(conditions_string[i] == ">="
                || conditions_string[i] == ">"
                || conditions_string[i] == "GreaterEqualTo"
                || conditions_string[i] == "GreaterThan")
        {
            conditions[i] = Condition::GreaterEqualTo;
        }
        else if(conditions_string[i] == "<="
                || conditions_string[i] == "<"
                || conditions_string[i] == "LessEqualTo"
                || conditions_string[i] == "LessThan")
        {
            conditions[i] = Condition::LessEqualTo;
        }
        else
        {
            conditions[i] = Condition::None;
        }
    }

    return conditions;
}


Tensor<Tensor<type, 1>, 1> ResponseOptimization::get_values_conditions(const Tensor<ResponseOptimization::Condition, 1>& conditions, const Tensor<type, 1>& values) const
{
    const Index conditions_size = conditions.size();

    Tensor<Tensor<type, 1>, 1> values_conditions(conditions_size);

    Index index = 0;

    ostringstream buffer;

    for(Index i = 0; i < conditions_size; i++)
    {
        Tensor<type, 1> current_values;

        const Condition current_condition = conditions[i];

        switch(current_condition)
        {
        case Condition::Minimum:

            values_conditions[i].resize(0);

            index++;
            break;

        case Condition::Maximum:

            values_conditions[i].resize(0);

            index++;
            break;

        case Condition::EqualTo:

            current_values.resize(1);
            current_values[0] = values[index];
            index++;

            values_conditions[i] = current_values;

            break;

        case Condition::LessEqualTo:

            current_values.resize(1);
            current_values[0] = values[index];
            index++;

            values_conditions[i] = current_values;

            break;

        case Condition::GreaterEqualTo:

            current_values.resize(1);
            current_values[0] = values[index];

            index++;

            values_conditions[i] = current_values;

            break;

        case Condition::Between:

            current_values.resize(2);
            current_values[0] = values[index];
            index++;
            current_values[1] = values[index];
            index++;

            values_conditions[i] = current_values;

            break;

        case Condition::None:

            current_values.resize(2);

            if(i<inputs_minimums.size())
            {
                current_values[0] = inputs_minimums(i);
                current_values[1] = inputs_maximums(i);
            }else
            {
                current_values[0] = outputs_minimums(i);
                current_values[1] = outputs_maximums(i);
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
    const Index inputs_number = neural_network_pointer->get_inputs_number();

    Tensor<type, 2> inputs(evaluations_number, inputs_number);
    inputs.setZero();

    const int input_columns_number = data_set_pointer->get_input_columns_number();

    Tensor<Index, 1> used_columns_indices = data_set_pointer->get_used_columns_indices();

    for(Index i = 0; i < evaluations_number; i++)
    {
        Index used_column_index = 0;

        Index index = 0;

        for(Index j = 0; j < input_columns_number; j++)
        {
            used_column_index = used_columns_indices(j);

            DataSet::ColumnType column_type = data_set_pointer->get_column_type(used_column_index);

            if(column_type == DataSet::ColumnType::Numeric || column_type == DataSet::ColumnType::Constant)
            {
                inputs(i,index) = calculate_random_uniform(inputs_minimums[index], inputs_maximums[index]);
                index++;
            }

            else if(column_type == DataSet::ColumnType::Binary)
            {
                if(inputs_conditions(index) == ResponseOptimization::Condition::EqualTo)
                {
                    inputs(i,index) = inputs_minimums[index];
                }
                else
                {
                    inputs(i,index) = rand() % 2;
                }
                index++;
            }

            else if(column_type == DataSet::ColumnType::Categorical)
            {
                Index categories_number = data_set_pointer->get_columns()(used_column_index).get_categories_number();
                Index equal_index = -1;

                for(Index k = 0; k < categories_number; k++)
                {
                    inputs(i,index + k) = 0;
                    if(inputs_conditions(index + k) == ResponseOptimization::Condition::EqualTo)
                    {
                        inputs(i,index + k) = inputs_minimums(index +k);
                        if(inputs(i, index + k) == 1)
                        {
                            equal_index = k;
                        }
                    }
                }

                if(equal_index == -1)
                {
                    Index random =  rand() % categories_number ;
                    random =  rand() % categories_number ;
                    inputs(i, index + random) = 1;
                }
                index+=(categories_number);
            }
            else
            {
                inputs(i,index) = calculate_random_uniform(inputs_minimums[index], inputs_maximums[index]);
                index++;
            }
        }
    }

    return inputs;
}


Tensor<type, 2> ResponseOptimization::calculate_envelope(const Tensor<type, 2>& inputs, const Tensor<type, 2>& outputs) const
{
    const Index inputs_number = neural_network_pointer->get_inputs_number();
    const Index outputs_number = neural_network_pointer->get_outputs_number();

    Tensor<type, 2> envelope = assemble_matrix_matrix(inputs,outputs);

    for(Index i = 0; i < outputs_number; i++)
    {
        if(envelope.size() != 0)
        {
            envelope = filter_column_minimum_maximum(envelope, inputs_number + i, outputs_minimums(i), outputs_maximums(i));
        }
        else
        {
            return Tensor<type,2>();
        }
    }

    return envelope;

}


ResponseOptimizationResults* ResponseOptimization::perform_optimization() const
{
    ResponseOptimizationResults* results = new ResponseOptimizationResults(neural_network_pointer);

    Tensor<type, 2> inputs = calculate_inputs();

    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

    Tensor<type, 2> outputs;

    outputs = neural_network_pointer->calculate_outputs(inputs.data(), inputs_dimensions);

    const Tensor<type, 2> envelope = calculate_envelope(inputs, outputs);

    const Index samples_number = envelope.dimension(0);

    const Index inputs_number = neural_network_pointer->get_inputs_number();
    const Index outputs_number = neural_network_pointer->get_outputs_number();

    Tensor<type, 1> objective(samples_number);
    objective.setZero();

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < inputs_number; j++)
        {
            if(inputs_conditions[j] == Condition::Minimum)
            {
                objective[i] += envelope(i,j);
            }
            else if(inputs_conditions[j] == Condition::Maximum)
            {
                objective[i] += -envelope(i,j);
            }
        }

        for(Index j = 0; j < outputs_number; j++)
        {
            if(outputs_conditions[j] == Condition::Minimum)
            {
                objective[i] += envelope(i, inputs_number+j);
            }
            else if(outputs_conditions[j] == Condition::Maximum)
            {
                objective[i] += -envelope(i, inputs_number+j);
            }
        }
    }

    const Index optimal_index = minimal_index(objective);

    if(envelope.size() != 0 )
    {
        results->optimal_variables = envelope.chip(optimal_index, 0);
    }
    else
    {
        results->optimal_variables = Tensor<type,1>();
    }

    return results;
}


type ResponseOptimization::calculate_random_uniform(const type& minimum, const type& maximum) const
{
    const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

    const type random_uniform = minimum + (maximum - minimum) * random;

    return random_uniform;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
