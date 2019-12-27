//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S                 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "response_optimization.h"

namespace OpenNN
{

/// Default constructor. 
/// It creates a scaling layer object with no scaling neurons. 

ResponseOptimization::ResponseOptimization()
{   
}


ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network_pointer)
{
    neural_network_pointer = new_neural_network_pointer;

    const size_t inputs_number = neural_network_pointer->get_inputs_number();
    const size_t outputs_number = neural_network_pointer->get_outputs_number();

    inputs_conditions.set(inputs_number, Between);
    outputs_conditions.set(outputs_number, Minimum);

    inputs_minimums = neural_network_pointer->get_scaling_layer_pointer()->get_minimums();
    inputs_maximums = neural_network_pointer->get_scaling_layer_pointer()->get_maximums();

    outputs_minimums = neural_network_pointer->get_bounding_layer_pointer()->get_lower_bounds();
    outputs_maximums = neural_network_pointer->get_bounding_layer_pointer()->get_upper_bounds();
}


/// Destructor.

ResponseOptimization::~ResponseOptimization()
{
}


void ResponseOptimization::set_evaluations_number(const size_t& new_evaluations_number)
{
    evaluations_number = new_evaluations_number;
}


Vector<ResponseOptimization::Condition> ResponseOptimization::get_inputs_conditions()
{
    return inputs_conditions;
}


Vector<ResponseOptimization::Condition> ResponseOptimization::get_outputs_conditions()
{
    return outputs_conditions;
}


Vector<double> ResponseOptimization::get_inputs_minimums()
{
    return inputs_minimums;
}


Vector<double> ResponseOptimization::get_inputs_maximums()
{
    return inputs_maximums;
}


Vector<double> ResponseOptimization::get_outputs_minimums()
{
    return outputs_minimums;
}


Vector<double> ResponseOptimization::get_outputs_maximums()
{
    return outputs_maximums;
}

void ResponseOptimization::set_input_condition(const string& name, const ResponseOptimization::Condition& condition, const Vector<double>& values)
{
    const size_t index = neural_network_pointer->get_input_index(name);

    set_input_condition(index, condition, values);
}


void ResponseOptimization::set_output_condition(const string& name, const ResponseOptimization::Condition& condition, const Vector<double>& values)
{
    const size_t index = neural_network_pointer->get_output_index(name);

    set_output_condition(index, condition, values);
}


void ResponseOptimization::set_input_condition(const size_t& index, const ResponseOptimization::Condition& condition, const Vector<double>& values)
{
    inputs_conditions[index] = condition;

    ostringstream buffer;

    switch(condition)
    {
        case Minimum:

            if(values.size() != 0)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_input_condition() method.\n"
                       << "For Minimum condition, size of values must be 0.\n";

                throw logic_error(buffer.str());
            }

        return;

        case Maximum:

            if(values.size() != 0)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_input_condition() method.\n"
                       << "For Maximum condition, size of values must be 0.\n";

                throw logic_error(buffer.str());
            }

        return;

        case EqualTo:

            if(values.size() != 1)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_input_condition() method.\n"
                       << "For LessEqualTo condition, size of values must be 1.\n";

                throw logic_error(buffer.str());
            }

            inputs_minimums[index] = values[0];
            inputs_maximums[index] = values[0];

        return;

        case LessEqualTo:

            if(values.size() != 1)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_input_condition() method.\n"
                       << "For LessEqualTo condition, size of values must be 1.\n";

                throw logic_error(buffer.str());
            }

            inputs_maximums[index] = values[0];

        return;

        case GreaterEqualTo:

            if(values.size() != 1)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_input_condition() method.\n"
                       << "For LessEqualTo condition, size of values must be 1.\n";

                throw logic_error(buffer.str());
            }

            inputs_minimums[index] = values[0];

        return;

        case Between:

            if(values.size() != 2)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_input_condition() method.\n"
                       << "For Between condition, size of values must be 2.\n";

                throw logic_error(buffer.str());
            }

            inputs_minimums[index] = values[0];
            inputs_maximums[index] = values[1];

        return;
    }
}


void ResponseOptimization::set_output_condition(const size_t& index, const ResponseOptimization::Condition& condition, const Vector<double>& values)
{
    outputs_conditions[index] = condition;

    ostringstream buffer;

    switch(condition)
    {
        case Minimum:

            if(values.size() != 0)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_output_condition() method.\n"
                       << "For Minimum condition, size of values must be 0.\n";

                throw logic_error(buffer.str());
            }

        return;

        case Maximum:

            if(values.size() != 0)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_output_condition() method.\n"
                       << "For Maximum condition, size of values must be 0.\n";

                throw logic_error(buffer.str());
            }

        return;

        case EqualTo:

            if(values.size() != 1)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_output_condition() method.\n"
                       << "For LessEqualTo condition, size of values must be 1.\n";

                throw logic_error(buffer.str());
            }

            outputs_minimums[index] = values[0];
            outputs_maximums[index] = values[0];

        return;

        case LessEqualTo:

            if(values.size() != 1)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_output_condition() method.\n"
                       << "For LessEqualTo condition, size of values must be 1.\n";

                throw logic_error(buffer.str());
            }

            outputs_maximums[index] = values[0];

        return;

        case GreaterEqualTo:

            if(values.size() != 1)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_output_condition() method.\n"
                       << "For LessEqualTo condition, size of values must be 1.\n";

                throw logic_error(buffer.str());
            }

            outputs_minimums[index] = values[0];

        return;

        case Between:

            if(values.size() != 2)
            {
                buffer << "OpenNN Exception: ResponseOptimization class.\n"
                       << "void set_output_condition() method.\n"
                       << "For Between condition, size of values must be 2.\n";

                throw logic_error(buffer.str());
            }

            outputs_minimums[index] = values[0];
            outputs_maximums[index] = values[1];

        return;        
    }
}


void ResponseOptimization::set_inputs_outputs_conditions(const Vector<string>& names, const Vector<string>& conditions_string, const Vector<double>& values)
{
    Vector<Condition> conditions = get_conditions(conditions_string);
    Vector<Vector<double>> values_conditions = get_values_conditions(conditions, values);

    const size_t variables_number = conditions_string.size();

    const Vector<string> inputs_names = neural_network_pointer->get_inputs_names();

    size_t index;

    for(size_t i = 0; i < variables_number; i ++)
    {
        if(inputs_names.contains(names[i]))
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


Vector<ResponseOptimization::Condition> ResponseOptimization::get_conditions(const Vector<string>& conditions_string) const
{
    const size_t conditions_size = conditions_string.size();

    Vector<Condition> conditions(conditions_size);

    for(size_t i = 0; i < conditions_size; i++)
    {
        if(conditions_string[i] == "Minimize")
        {
            conditions[i] = Minimum;
        }
        else if(conditions_string[i] == "Maximize")
        {
            conditions[i] = Maximum;
        }
        else if(conditions_string[i] == "=")
        {
            conditions[i] = EqualTo;
        }
        else if(conditions_string[i] == "Between")
        {
            conditions[i] = Between;
        }
        else if(conditions_string[i] == ">="
             || conditions_string[i] == ">")
        {
            conditions[i] = GreaterEqualTo;
        }
        else if(conditions_string[i] == "<="
             || conditions_string[i] == "<")
        {
            conditions[i] = LessEqualTo;
        }
    }

    return conditions;
}


Vector<Vector<double>> ResponseOptimization::get_values_conditions(const Vector<ResponseOptimization::Condition>& conditions, const Vector<double>& values) const
{
    const size_t conditions_size = conditions.size();

    Vector<Vector<double>> values_conditions(conditions_size);

    size_t index = 0;

    ostringstream buffer;

    for(size_t i = 0; i < conditions_size; i++)
    {
        Vector<double> current_values;

        const Condition current_condition = conditions[i];

        switch(current_condition)
        {
            case Minimum:

                values_conditions[i] = Vector<double>();

                index++;
            break;

            case Maximum:

                values_conditions[i] = Vector<double>();

                index++;
            break;

            case EqualTo:

                current_values.resize(1);
                current_values[0] = values[index];
                index++;

                values_conditions[i] = current_values;

            break;

            case LessEqualTo:

                current_values.resize(1);
                current_values[0] = values[index];
                index++;

                values_conditions[i] = current_values;

            break;

            case GreaterEqualTo:

                current_values.resize(1);
                current_values[0] = values[index];

                index++;

                values_conditions[i] = current_values;

            break;

            case Between:

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


Tensor<double> ResponseOptimization::calculate_inputs() const
{
    const size_t inputs_number = neural_network_pointer->get_inputs_number();

    Tensor<double> inputs(evaluations_number, inputs_number);

    for(size_t i = 0; i < evaluations_number; i++)
    {
        for(size_t j = 0; j < inputs_number; j++)
        {
            inputs(i,j) = calculate_random_uniform(inputs_minimums[j], inputs_maximums[j]);
        }
    }

    return inputs;
}


Matrix<double> ResponseOptimization::calculate_envelope(const Tensor<double>& inputs, const Tensor<double>& outputs) const
{
    const size_t inputs_number = neural_network_pointer->get_inputs_number();
    const size_t outputs_number = neural_network_pointer->get_outputs_number();

    Matrix<double> envelope = (inputs.to_matrix()).assemble_columns((outputs.to_matrix()));

    for(size_t i = 0; i < outputs_number; i++)
    {
        envelope = envelope.filter_column_minimum_maximum(inputs_number+i, outputs_minimums[i], outputs_maximums[i]);
    }

    return envelope;
}


ResponseOptimization::Results* ResponseOptimization::perform_optimization() const
{
    Results* results = new Results(neural_network_pointer);

    const Tensor<double> inputs = calculate_inputs();

    const Tensor<double> outputs = neural_network_pointer->calculate_outputs(inputs);

    const Matrix<double> envelope = calculate_envelope(inputs, outputs);

    const size_t samples_number = envelope.get_rows_number();

    const size_t inputs_number = neural_network_pointer->get_inputs_number();
    const size_t outputs_number = neural_network_pointer->get_outputs_number();

    Vector<double> objective(samples_number, 0.0);

    for(size_t i = 0; i < samples_number; i++)
    {
        for(size_t j = 0; j < inputs_number; j++)
        {
            if(inputs_conditions[j] == Minimum)
            {
                objective[i] += envelope(i,j);
            }
            else if(inputs_conditions[j] == Maximum)
            {
                objective[i] += -envelope(i,j);
            }
        }

        for(size_t j = 0; j < outputs_number; j++)
        {
            if(outputs_conditions[j] == Minimum)
            {
                objective[i] += envelope(i, inputs_number+j);
            }
            else if(outputs_conditions[j] == Maximum)
            {
                objective[i] += -envelope(i, inputs_number+j);
            }
        }
    }

    const size_t optimal_index = minimal_index(objective);

    results->optimal_variables = envelope.get_row(optimal_index);

    results->optimum_objective = objective[optimal_index];

    return results;
}


double ResponseOptimization::calculate_random_uniform(const double& minimum, const double& maximum) const
{
  const double random = static_cast<double>(rand()/(RAND_MAX + 1.0));

  const double random_uniform = minimum + (maximum - minimum) * random;

  return random_uniform;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
