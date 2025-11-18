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
#include "scaling.h"
#include "scaling_layer_2d.h"
#include "bounding_layer.h"
#include "dataset.h"
#include "neural_network.h"
#include <variant>

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

    for(Index j = 0; j < inputs_number; ++j)
        constraints.push_back({ j, input_minimums(j), input_maximums(j) });

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

//Pareto helpers
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

ResponseOptimization::ParetoResult ResponseOptimization::perform_pareto() const
{
    const Tensor<type, 2> inputs  = calculate_inputs();

    const Tensor<type, 2> outputs = neural_network->calculate_outputs<2,2>(inputs);

    const Tensor<type, 2> envelope = calculate_envelope(inputs, outputs);

    if(envelope.size() == 0)
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

    for(Index i = 0; i < Pareto_rows_dimension; ++i)
        for(Index j = 0; j < input_number_dimension; ++j)
            inputs_filtered(i,j) = envelope(i,j);

    return perform_pareto_analysis(objectives, sense, inputs_filtered, envelope);
}

ResponseOptimization::SingleOrPareto ResponseOptimization::iterative_optimization(int objective_count)
{
    const Index max_iterations      = iterative_max_iterations;

    const type  zoom_factor         = iterative_zoom_factor;

    const type  min_span_eps        = iterative_min_span_eps;

    const type  improvement_tolerance = iterative_improvement_tolerance;

    const Index inputs_number  = neural_network->get_inputs_number();
    const Index outputs_number = neural_network->get_outputs_number();

    // Save original bounds (RAII guard)
    Tensor<type,1> original_input_minimums  = input_minimums;
    Tensor<type,1>  original_input_maximums  = input_maximums;
    Tensor<type,1> original_output_minimums = output_minimums;
    Tensor<type,1> original_output_maximums = output_maximums;

    struct BoundsGuard
    {
        ResponseOptimization* self;

        Tensor<type,1> saved_input_minimums;

        Tensor<type,1> saved_input_maximums;

        Tensor<type,1> saved_output_minimums;

        Tensor<type,1> saved_output_maximums;

        ~BoundsGuard(){
            self->input_minimums  = saved_input_minimums;
            self->input_maximums  = saved_input_maximums;
            self->output_minimums = saved_output_minimums;
            self->output_maximums = saved_output_maximums;
        }
    } guard{ this, original_input_minimums,  original_input_maximums, original_output_minimums, original_output_maximums };

    auto clampv = [&](const type& current_limit, const type& low_limit, const type& high_limit)
    {
        return current_limit < low_limit ? low_limit : (current_limit > high_limit ? high_limit : current_limit);
    };

    auto append_rows = [](Tensor<type,2>& acc, const Tensor<type,2>& block)
    {
        if(block.size() == 0) return;

        if(acc.size() == 0)
        {
            acc = block;
            return;
        }

        const Index old_rows = acc.dimension(0);
        const Index cols     = acc.dimension(1);
        const Index new_rows = block.dimension(0);

        Tensor<type,2> tmp(old_rows + new_rows, cols);

        for(Index i = 0; i < old_rows; ++i)
            for(Index j = 0; j < cols; ++j)
                tmp(i,j) = acc(i,j);

        for(Index i = 0; i < new_rows; ++i)
            for(Index j = 0; j < cols; ++j)
                tmp(old_rows + i, j) = block(i,j);

        acc = std::move(tmp);
    };

    if (objective_count <= 1)
    {
        auto search_best = [&](Tensor<type,1>& best_row, type& best_val_signed)->bool
        {
            const Tensor<type,2> inputs  = calculate_inputs();

            const Tensor<type,2> outputs = neural_network->calculate_outputs<2,2>(inputs);

            const Tensor<type,2> envelope     = calculate_envelope(inputs, outputs);

            if (envelope.size() == 0)
                return false;

            int objective_column = -1;

            type sign = type(1);

            int count = 0;

            for (Index j = 0; j < inputs_number; ++j)
            {
                if      (input_conditions(j) == Condition::Minimum)
                {
                    objective_column = int(j);

                    sign =  type(1);

                    ++count;
                }
                else if (input_conditions(j) == Condition::Maximum)
                {
                    objective_column = int(j);

                    sign = -type(1);

                    ++count;
                }
            }
            for (Index j = 0; j < outputs_number; ++j)
            {
                if      (output_conditions(j) == Condition::Minimum)
                {
                    objective_column = int(inputs_number + j);

                    sign =  type(1);

                    ++count;
                }
                else if (output_conditions(j) == Condition::Maximum)
                {
                    objective_column = int(inputs_number + j);

                    sign = -type(1);

                    ++count;
                }
            }

            if (count != 1)
                throw runtime_error("iterative_optimization: expected single objective.");

            Tensor<type,1> objective_column_from_envelope = envelope.chip(objective_column, 1).eval();

            const type orientation = (sign > type(0)) ? type(1) : type(-1);

            Tensor<type,1> objective_column_oriented = (objective_column_from_envelope * orientation).eval();

            const Index best_idx = minimal_index(objective_column_oriented);

            best_val_signed = objective_column_oriented(best_idx);

            best_row = envelope.chip(best_idx, 0);

            return true;
        };

        Tensor<type,1> best_row = Tensor<type,1>();

        type best_value = std::numeric_limits<double>::infinity();
        {
            Tensor<type,1> vacuum_row;

            type vacuum_value;

            if (!search_best(vacuum_row, vacuum_value))
                return Tensor<type,1>();

            best_row = vacuum_row;

            best_value = vacuum_value;
        }

        Tensor<type,1> current_input_minimums = input_minimums;
        Tensor<type,1> current_input_maximums = input_maximums;

        for (Index iteration = 0; iteration < max_iterations; ++iteration)
        {
            // Compute new box centered at best_row (inputs part only)
            bool collapsed = true;

            for (Index j = 0; j < inputs_number; ++j)
            {
                const type span = current_input_maximums(j) - current_input_minimums(j);

                const type new_span = span * zoom_factor;

                const type center = best_row(j);

                type new_min = center - new_span * type(0.5);

                type new_max = center + new_span * type(0.5);

                // Respect original global bounds
                new_min = clampv(new_min, original_input_minimums(j),  original_input_maximums(j));
                new_max = clampv(new_max, original_input_minimums(j),  original_input_maximums(j));

                // Avoid inverted intervals
                if (new_max <= new_min)
                {
                    const type epsilon = max(type(1e-12), span * type(1e-6));

                    new_min = clampv(center - epsilon, original_input_minimums(j),  original_input_maximums(j));
                    new_max = clampv(center + epsilon, original_input_minimums(j),  original_input_maximums(j));
                }

                current_input_minimums(j) = new_min;

                current_input_maximums(j) = new_max;

                if ((new_max - new_min) > min_span_eps)
                    collapsed = false;
            }
            if (collapsed) break;

            // Temporarily set class bounds to the shrunken box
            input_minimums  = current_input_minimums;

            input_maximums  = current_input_maximums;

            // Re-run pass
            Tensor<type,1> candidate_row;

            type candidate_value;

            if (!search_best(candidate_row, candidate_value))
                break;

            const type gain = best_value - candidate_value;

            const type module_biggest_value = max(std::abs(best_value), type(1));

            const type real_improvement = gain / module_biggest_value;

            if (real_improvement > improvement_tolerance)
            {
                best_value = candidate_value;

                best_row = candidate_row;
            }
            else
            {
                // Little to gain: one last micro-zoom and exit
                for (Index j = 0; j < inputs_number; ++j)
                {
                    const type span = current_input_maximums(j) - current_input_minimums(j);

                    const type new_span = max(span * type(0.2), type(1e-12));

                    const type center = best_row(j);

                    current_input_minimums(j) = clampv(center - new_span*type(0.5), original_input_minimums(j),  original_input_maximums(j));

                    current_input_maximums(j) = clampv(center + new_span*type(0.5), original_input_minimums(j),  original_input_maximums(j));
                }
                input_minimums = current_input_minimums;

                input_maximums = current_input_maximums;

                Tensor<type,1> final_row;

                type final_val;
                if (search_best(final_row, final_val) && final_val < best_value)
                {
                    best_value = final_val;

                    best_row = final_row;
                }
                break;
            }
        }

        return best_row;
    }

    ParetoResult first = perform_pareto();

    if(first.pareto_variables.size() == 0)
        first = perform_pareto();

    if(first.pareto_variables.size() == 0)
        return first;

    Tensor<type,2> all_envelope = first.pareto_variables;

    const Index pareto_points_number = first.pareto_variables.dimension(0);

    for(Index current_pareto_point = 0; current_pareto_point < pareto_points_number; ++current_pareto_point)
    {
        Tensor<type,1> center_row = first.pareto_variables.chip(current_pareto_point, 0);

        Tensor<type,1> current_input_minimums = original_input_minimums;

        Tensor<type,1> current_input_maximums = original_input_maximums;


        for(Index current_iteration = 0; current_iteration < max_iterations; ++current_iteration)
        {
            bool collapsed = true;

            for(Index j = 0; j < inputs_number; ++j)
            {
                const type span     = current_input_maximums(j) - current_input_minimums(j);
                const type new_span = span * zoom_factor;

                const type center   = center_row(j);

                type new_min = center - new_span * type(0.5);
                type new_max = center + new_span * type(0.5);

                new_min = clampv(new_min, original_input_minimums(j), original_input_maximums(j));
                new_max = clampv(new_max, original_input_minimums(j), original_input_maximums(j));

                if(new_max <= new_min)
                {
                    const type epsilon = max(type(1e-12), span * type(1e-6));

                    new_min = clampv(center - epsilon, original_input_minimums(j), original_input_maximums(j));
                    new_max = clampv(center + epsilon, original_input_minimums(j), original_input_maximums(j));
                }

                current_input_minimums(j) = new_min;
                current_input_maximums(j) = new_max;

                if((new_max - new_min) > min_span_eps)
                    collapsed = false;
            }

            if(collapsed)
                break;

            input_minimums = current_input_minimums;
            input_maximums = current_input_maximums;

            ParetoResult local_pareto = perform_pareto();

            if(local_pareto.pareto_variables.size() != 0)
                append_rows(all_envelope, local_pareto.pareto_variables);
        }
    }

    Tensor<type,2> all_objectives;
    Tensor<type,1> sense;
    Tensor<Index,1> objective_indices;

    build_objectives_from_envelope(all_envelope, all_objectives, sense, objective_indices);

    if(objective_indices.size() < 1)
        return ParetoResult{};

    const Index all_rows = all_envelope.dimension(0);

    Tensor<type,2> inputs_filtered(all_rows, inputs_number);

    for(Index i = 0; i < all_rows; ++i)
        for(Index j = 0; j < inputs_number; ++j)
            inputs_filtered(i,j) = all_envelope(i,j);

    ParetoResult final_res = perform_pareto_analysis(all_objectives, sense, inputs_filtered, all_envelope);

    return final_res;

}





}
// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute iteration and/or
// modify iteration under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that iteration will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
