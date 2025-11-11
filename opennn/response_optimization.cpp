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
#include "dataset.h"
#include "neural_network.h"

namespace opennn
{

ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network, Dataset* new_dataset)
    : dataset(new_dataset)
{
    set(new_neural_network, new_dataset);
}


void ResponseOptimization::set(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{   
    neural_network = new_neural_network;

    dataset = new_dataset;

    if(!neural_network) return;

    const Index inputs_number = neural_network->get_inputs_number();

    const Index outputs_number = neural_network->get_outputs_number();

    input_conditions.resize(inputs_number);

    input_conditions.setConstant(Condition::None);

    output_conditions.resize(outputs_number);

    output_conditions.setConstant(Condition::None);

    if(neural_network->has("Scaling2d"))
    {
        Scaling2d* scaling_layer_2d = static_cast<Scaling2d*>(neural_network->get_first("Scaling2d"));

        input_minimums = scaling_layer_2d->get_minimums();

        input_maximums = scaling_layer_2d->get_maximums();
    }

    if(neural_network->has("Bounding"))
    {
        Bounding* bounding_layer = static_cast<Bounding*>(neural_network->get_first("Bounding"));

        output_minimums = bounding_layer->get_lower_bounds();

        output_maximums = bounding_layer->get_upper_bounds();
    }
    else
    {
        output_minimums.resize(outputs_number);

        output_minimums.setZero();

        output_maximums.resize(outputs_number);

        output_maximums.setConstant(type(1));
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

    case Condition::None:

    case Condition::Minimum:

    case Condition::Maximum:

        break;
    }
}


void ResponseOptimization::set_output_condition(const Index& index,
                                                const ResponseOptimization::Condition& condition,
                                                const Tensor<type, 1>& values)
{
    output_conditions[index] = condition;

    switch(condition)
    {        
    case Condition::EqualTo:

        throw runtime_error("EqualTo condition is only available for inputs.\n");

        return;

    case Condition::LessEqualTo:

        if(values.size() != 1)
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

    case Condition::Minimum:

    case Condition::Maximum:

        break;
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


Tensor<type, 2> ResponseOptimization::calculate_inputs() const
{
    const Index inputs_number = neural_network->get_inputs_number();

    Tensor<type, 2> inputs(evaluations_number, inputs_number);
    inputs.setZero();

    const Index input_raw_variables_number = dataset->get_raw_variables_number("Input");

    vector<Index> used_raw_variables_indices = dataset->get_used_raw_variables_indices();

    for(Index i = 0; i < evaluations_number; i++)
    {
        Index index = 0;

        for(Index j = 0; j < input_raw_variables_number; j++)
        {
            const Index used_raw_variable_index = used_raw_variables_indices[j];

            const Dataset::RawVariableType raw_variable_type = dataset->get_raw_variable_type(used_raw_variable_index);

            if(raw_variable_type == Dataset::RawVariableType::Numeric
            || raw_variable_type == Dataset::RawVariableType::Constant)
            {
                inputs(i, index) = get_random_type(input_minimums[index], input_maximums[index]);
                index++;
            }
            else if(raw_variable_type == Dataset::RawVariableType::Binary)
            {
                inputs(i, index) = (input_conditions(index) == ResponseOptimization::Condition::EqualTo)
                    ? input_minimums[index]
                    : type(rand() % 2);

                index++;
            }
            else if(raw_variable_type == Dataset::RawVariableType::Categorical)
            {
                const Index categories_number = dataset->get_raw_variables()[used_raw_variable_index].get_categories_number();
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


Tensor<type,2> ResponseOptimization::calculate_envelope(const Tensor<type,2>& inputs, const Tensor<type,2>& outputs) const
{
    const Index inputs_number = neural_network->get_inputs_number();
    const Index outputs_number = neural_network->get_outputs_number();

    Tensor<type, 2> envelope = assemble_matrix_matrix(inputs, outputs);

    if(envelope.size() == 0)
        return Tensor<type,2>();


    struct Constraint { Index col; type min; type max; };

    vector<Constraint> constraints;

    constraints.reserve(inputs_number + outputs_number);

    for(Index j = 0; j < outputs_number; ++j)
        constraints.push_back({ inputs_number + j, output_minimums(j), output_maximums(j) });

    const Index rows_number = envelope.dimension(0);

    const Index columns_number = envelope.dimension(1);

    vector<bool> rows_to_keep_mask;

    rows_to_keep_mask.resize(static_cast<size_t>(rows_number));

    Index kept_count = 0;

    for(Index i = 0; i < rows_number; ++i)
    {
        bool fits_the_constrain = true;

        for(const auto &current_constrain : constraints)
        {
            const type v = envelope(i, current_constrain.col);

            if(v < current_constrain.min || v > current_constrain.max)
            {
                fits_the_constrain = false;

                break;
            }
        }
        rows_to_keep_mask[static_cast<size_t>(i)] = fits_the_constrain ? 1 : 0;

        if(fits_the_constrain)
            ++kept_count;
    }

    if(kept_count == 0)
        return Tensor<type,2>();


    Tensor<type,2> filtered(kept_count, columns_number);

    Index adding_row = 0;

    for(Index i = 0; i < rows_number; ++i)
    {
        if(rows_to_keep_mask[static_cast<size_t>(i)])
        {
            const Tensor<type,1> row = envelope.chip(i, 0);

            set_row(filtered, row, adding_row);

            ++adding_row;
        }
    }

    return filtered;
}

ResponseOptimizationResults* ResponseOptimization::perform_optimization() const
{
    //@simone try to make one funcion for tabling the optimal point

    ResponseOptimizationResults* results = new ResponseOptimizationResults(neural_network);

    const Tensor<type, 2> inputs = calculate_inputs();

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

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

// -------------------- PARETO helpers --------------------
bool ResponseOptimization::dominates_row(const Tensor<type,1>& first_row_to_compare,
                                         const Tensor<type,1>& second_row_to_compare,
                                         const Tensor<type,1>& sense)
{
    bool strictly_better = false;

    for(Index j = 0; j < first_row_to_compare.size(); ++j)
    {
        const type first_row_oriented = first_row_to_compare(j) * sense(j);

        const type second_row_oriented = second_row_to_compare(j) * sense(j);

        if(first_row_oriented > second_row_oriented)
            return false;
        else
            strictly_better = true;
    }
    return strictly_better;
}

void ResponseOptimization::build_objectives_from_envelope(const Tensor<type,2>& envelope,
                                                          Tensor<type,2>& objectives,
                                                          Tensor<type,1>& sense,
                                                          Tensor<Index,1>& objective_indices) const
{
    const Index inputs_number  = neural_network->get_inputs_number();

    const Index outputs_number = neural_network->get_outputs_number();

    Index objectives_number = 0;

    for (Index i = 0; i < inputs_number; ++i)
        if (input_conditions(i) == Condition::Minimum || input_conditions(i) == Condition::Maximum)
            ++objectives_number;

    for(Index j = 0; j<outputs_number; ++j)
        if(output_conditions(j) == Condition::Minimum || output_conditions(j) == Condition::Maximum)
            ++objectives_number;

    objectives.resize(envelope.dimension(0), objectives_number);

    sense.resize(objectives_number);

    objective_indices.resize(objectives_number);

    Index counter_objectives = 0;

    for (Index i = 0; i < inputs_number; ++i)
    {
        if (input_conditions(i) != Condition::Minimum && input_conditions(i) != Condition::Maximum)
            continue;

        for (Index j = 0; j < envelope.dimension(0); ++j)
            objectives(j, counter_objectives) = envelope(j, i);

        sense(counter_objectives) = (input_conditions(i) == Condition::Maximum) ? type(-1) : type(1);

        objective_indices(counter_objectives) = i;

        ++counter_objectives;
    }


    for(Index j = 0; j<outputs_number; ++j)
    {
        if(output_conditions(j) != Condition::Minimum && output_conditions(j) != Condition::Maximum)
            continue;

        for(Index i = 0; i < envelope.dimension(0); ++i)
            objectives(i, counter_objectives) = envelope(i, inputs_number + j);

        sense(counter_objectives) = (output_conditions(j) == Condition::Maximum) ? type(-1) : type(1);

        objective_indices(counter_objectives) =inputs_number + j;

        ++counter_objectives;
    }
}


ResponseOptimization::ParetoResult ResponseOptimization::perform_pareto_analysis(const Tensor<type, 2>& objectives,
                                                                                 const Tensor<type, 1>& sense,
                                                                                 const Tensor<type, 2>& inputs,
                                                                                 const Tensor<type, 2>& envelope) const
{
    const Index rows_number = envelope.dimension(0);

    const Index objectives_number = objectives.dimension(1);

    Tensor<char,1> is_dominated(rows_number);

    is_dominated.setZero();

    for(Index i = 0; i < rows_number; ++i)
    {
        if(is_dominated(i))
            continue;

        for(Index k = 0; k < rows_number; ++k)
        {
            if(i == k || is_dominated(k))
                continue;

            Tensor<type,1> row_i(objectives_number);

            Tensor<type,1> row_k(objectives_number);

            for(Index j=0; j < objectives_number; ++j)
            {
                row_i(j)=objectives(i,j);

                row_k(j)=objectives(k,j);
            }

            if(dominates_row(row_k, row_i, sense))
            {
                is_dominated(i)=1;
                break;
            }
            if(dominates_row(row_i, row_k, sense))
                is_dominated(k)=1;
        }
    }

    vector<Index> idexes_non_dominate_rows;

    idexes_non_dominate_rows.reserve(rows_number);

    for(Index i = 0; i < rows_number; ++i)
        if(!is_dominated(i))
            idexes_non_dominate_rows.push_back(i);

    Tensor<Index,1> pareto_indices(idexes_non_dominate_rows.size());

    for(Index i = 0; i < pareto_indices.size(); ++i)
        pareto_indices(i)=idexes_non_dominate_rows[i];

    const Index pareto_points_number = pareto_indices.size();

    const Index envelope_variables_number = envelope.dimension(1);

    Tensor<type,2> pareto_objective_values(pareto_points_number, objectives_number);

    Tensor<type,2> pareto_candidate_variables(pareto_points_number, envelope_variables_number);

    Tensor<type,2> pareto_input_vectors(pareto_points_number, inputs.dimension(1));

    for(Index current_pareto_index = 0; current_pareto_index < pareto_points_number; ++current_pareto_index)
    {
        const Index i = pareto_indices(current_pareto_index);

        for(Index j = 0; j < objectives_number; ++j)
            pareto_objective_values(current_pareto_index,j)  = objectives(i,j);

        for(Index j = 0; j < envelope_variables_number; ++j)
            pareto_candidate_variables(current_pareto_index,j) = envelope(i,j);

        for(Index j = 0; j < inputs.dimension(1); ++j)
            pareto_input_vectors(current_pareto_index,j) = inputs(i,j);
    }

    ParetoResult res;

    res.pareto_indices   = std::move(pareto_indices);

    res.pareto_objectives= std::move(pareto_objective_values);

    res.pareto_variables = std::move(pareto_candidate_variables);

    res.pareto_inputs    = std::move(pareto_input_vectors);

    res.envelope = std::move(envelope);

    return res;
}

// -------------------- Public entry: build, then Pareto --------------------
ResponseOptimization::ParetoResult ResponseOptimization::perform_pareto() const
{
    const Tensor<type, 2> inputs  = calculate_inputs();

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

    const Tensor<type, 2> envelope = calculate_envelope(inputs, outputs);

    if(envelope.size() == 0)
        //message more evaluation number or less constraints
        return ParetoResult{};

    Tensor<type,2> objectives;

    Tensor<type,1> sense;

    Tensor<Index,1> objectives_indices;

    build_objectives_from_envelope(envelope, objectives, sense, objectives_indices);

    if(objectives_indices.dimension(1) < 1)
        return ParetoResult{};

    const Index Pareto_rows_dimension = envelope.dimension(0);

    const Index input_number_dimension  = neural_network->get_inputs_number();

    Tensor<type,2> inputs_filtered(Pareto_rows_dimension, input_number_dimension);

    for(Index r = 0; r < Pareto_rows_dimension; ++r)
        for(Index c = 0; c < input_number_dimension; ++c)
            inputs_filtered(r,c) = envelope(r,c);

    return perform_pareto_analysis(objectives, sense, inputs_filtered, envelope);
}

Tensor<type, 1> ResponseOptimization::get_nearest_point_to_utopian(const ResponseOptimization::ParetoResult& pareto_result) const
{
    const Index num_pareto_points = pareto_result.pareto_objectives.dimension(0);

    const Index num_objectives = pareto_result.pareto_objectives.dimension(1);

    // Calculate the utopian point (ideal best outcomes for each objective)

    Tensor<type, 1> utopian_point(num_objectives);

    Tensor<type, 1> objectives_maximums(num_objectives);

    Tensor<type, 1> objectives_minimums(num_objectives);

    utopian_point.setZero();

    int output_number = neural_network->get_outputs_number();

    int input_number = neural_network->get_inputs_number();

    int j = 0;

    for (Index i = 0; i < input_number; ++i)
    {
        if (get_input_conditions()(i) == ResponseOptimization::Condition::Minimum)
        {
            objectives_minimums(j)=input_minimums[i];

            utopian_point(j) = 0;

            j++;
        }
        else if (get_input_conditions()(i) == ResponseOptimization::Condition::Maximum)
        {
            objectives_maximums(j)=input_maximums[i];

            utopian_point(j) = 1;

            j++;
        }
    }

    for (Index i = 0; i < output_number; ++i)
    {
        if (get_output_conditions()(i) == ResponseOptimization::Condition::Minimum)
        {
            objectives_minimums(j) = output_minimums[i];

            utopian_point(j) = 0;

            j++;
        }
        else if (get_output_conditions()(i) == ResponseOptimization::Condition::Maximum)
        {
            objectives_maximums(j) = output_maximums[i];

            utopian_point(j) = 1;

            j++;
        }
    }

    Tensor<type,2> scaled_pareto_points = pareto_result.pareto_objectives;

    #pragma omp parallel for
    for (Index j = 0; j < num_objectives; j++)
    {
        for(Index i = 0; i < num_pareto_points; i++)
           scaled_pareto_points(i, j) = (pareto_result.pareto_objectives(i, j) - objectives_minimums(j)) / (objectives_maximums(j) - objectives_minimums(j));
    }

    Tensor<type, 1> distances(num_pareto_points);

    for (Index i = 0; i < num_pareto_points; ++i)
    {
        type distance = 0;

        for (Index j = 0; j < num_objectives; ++j)
            distance += pow(scaled_pareto_points(i, j) - utopian_point(j), 2);

        distances(i) = sqrt(distance);
    }

    Index nearest_point_index = minimal_index(distances);

    Tensor<type, 1> best_point;

    best_point = pareto_result.envelope.chip(pareto_result.pareto_indices(nearest_point_index),0) ;

    return best_point;
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
