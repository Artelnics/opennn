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


ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network_pointer)
    : neural_network_pointer(new_neural_network_pointer)
{
    const Index inputs_number = neural_network_pointer->get_inputs_number();
    const Index outputs_number = neural_network_pointer->get_outputs_number();

    inputs_conditions.resize(inputs_number);
    inputs_conditions.setConstant(Condition::Between);

    outputs_conditions.resize(outputs_number);
    outputs_conditions.setConstant(Condition::Minimum);

    inputs_minimums = neural_network_pointer->get_scaling_layer_pointer()->get_minimums();
    inputs_maximums = neural_network_pointer->get_scaling_layer_pointer()->get_maximums();

    outputs_minimums = neural_network_pointer->get_bounding_layer_pointer()->get_lower_bounds();
    outputs_maximums = neural_network_pointer->get_bounding_layer_pointer()->get_upper_bounds();
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

    const Tensor<string, 1> inputs_names = neural_network_pointer->get_inputs_names();

    Index index;

    for(Index i = 0; i < variables_number; i ++)
    {
        if(contains(inputs_names,names[i]))
        {
            index = neural_network_pointer->get_input_index(names[i]);

            set_input_condition(index, conditions[i], values_conditions[i]);
        }
        else
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
        if(conditions_string[i] == "Minimize")
        {
            conditions[i] = Condition::Minimum;
        }
        else if(conditions_string[i] == "Maximize")
        {
            conditions[i] = Condition::Maximum;
        }
        else if(conditions_string[i] == "=")
        {
            conditions[i] = Condition::EqualTo;
        }
        else if(conditions_string[i] == "Between")
        {
            conditions[i] = Condition::Between;
        }
        else if(conditions_string[i] == ">="
                || conditions_string[i] == ">")
        {
            conditions[i] = Condition::GreaterEqualTo;
        }
        else if(conditions_string[i] == "<="
                || conditions_string[i] == "<")
        {
            conditions[i] = Condition::LessEqualTo;
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
        }
    }

    return values_conditions;
}


Tensor<type, 2> ResponseOptimization::calculate_inputs() const
{
    const Index inputs_number = neural_network_pointer->get_inputs_number();

    Tensor<type, 2> inputs(evaluations_number, inputs_number);

    for(Index i = 0; i < evaluations_number; i++)
    {
        for(Index j = 0; j < inputs_number; j++)
        {
            inputs(i,j) = calculate_random_uniform(inputs_minimums[j], inputs_maximums[j]);
        }
    }

    return inputs;
}


Tensor<type, 2> ResponseOptimization::calculate_envelope(const Tensor<type, 2>& inputs, const Tensor<type, 2>& outputs) const
{
    const Index inputs_number = neural_network_pointer->get_inputs_number();
    const Index outputs_number = neural_network_pointer->get_outputs_number();

//    Tensor<type, 2> envelope = (inputs.to_matrix()).assemble_columns((outputs.to_matrix())); old line
    Tensor<type, 2> envelope = assemble_matrix_matrix(inputs,outputs);

    for(Index i = 0; i < outputs_number; i++)
    {
//        envelope = envelope.filter_column_minimum_maximum(inputs_number+i, outputs_minimums[i], outputs_maximums[i]); old line
        envelope = filter_column_minimum_maximum(envelope, inputs_number + i, outputs_minimums(i), outputs_maximums(i));
    }

    return envelope;

}


ResponseOptimizationResults* ResponseOptimization::perform_optimization() const
{
    ResponseOptimizationResults* results = new ResponseOptimizationResults(neural_network_pointer);

    const Tensor<type, 2> inputs = calculate_inputs();

    const Tensor<type, 2> outputs = neural_network_pointer->calculate_outputs(inputs);

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

    results->optimal_variables = envelope.chip(optimal_index, 0);

    results->optimum_objective = objective[optimal_index];

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
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
