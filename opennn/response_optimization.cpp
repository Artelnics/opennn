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

    const Index inputs_number = neural_network->get_features_number();

    const Index outputs_number = neural_network->get_outputs_number();

    const Index raw_inputs_number  = dataset->get_raw_variables_number("Input");
    const Index raw_outputs_number = dataset->get_raw_variables_number("Target");

    input_conditions.resize(raw_inputs_number);
    input_conditions.setConstant(Condition::None);

    output_conditions.resize(raw_outputs_number);
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

    //@simone DIAGNOSTIC PRINT
    cout << "SETTER CALL: Name=" << name << " Condition=" << (int)condition << endl;

    const Index raw_index = dataset->get_raw_variable_index(name);

    if(raw_index < 0)
        throw runtime_error("Unknown input variable name: " + name);

    const vector<Index> input_indices = dataset->get_raw_variable_indices("Input");

    Index relative_index = -1;
    for(size_t i = 0; i < input_indices.size(); ++i)
    {
        if(input_indices[i] == raw_index)
        {
            relative_index = static_cast<Index>(i);
            break;
        }
    }

    //@simone tutto l'if è diagnostico
    if (relative_index != -1) {
        input_conditions(relative_index) = condition;
        cout << "  -> Set input_conditions(" << relative_index << ") to " << (int)condition << endl;
    } else {
        cout << "  -> FAILED to find relative index for " << name << endl;
    }


    if(relative_index == -1)
    {
        // This happens if the variable exists but its role is not "Input"
        throw runtime_error("Variable " + name + " is not set as Input.");
    }


    input_conditions(relative_index) = condition;

    const vector<Dataset::RawVariable> raw_inputs = dataset->get_raw_variables("Input");

    const Dataset::RawVariable& raw_var = raw_inputs[relative_index];
    const Index raw_inputs_number = static_cast<Index>(raw_inputs.size());


    vector<Index> raw_input_categoricals;
    vector<Index> raw_input_categories_sizes;

    dataset->get_categorical_info("Input", raw_input_categoricals, raw_input_categories_sizes);

    vector<Index> raw_input_feature_start(raw_inputs_number);
    vector<Index> raw_input_feature_size(raw_inputs_number);

    {
        Index feature_index = 0;
        size_t category_position = 0;

        for(Index current_variable = 0; current_variable < raw_inputs_number; ++current_variable)
        {
            if(category_position < raw_input_categoricals.size()
                && raw_input_categoricals[category_position] == current_variable)
            {
                const Index one_hot_size = raw_input_categories_sizes[category_position];

                raw_input_feature_start[current_variable] = feature_index;
                raw_input_feature_size[current_variable]  = one_hot_size;

                feature_index += one_hot_size;
                ++category_position;
            }
            else
            {
                raw_input_feature_start[current_variable] = feature_index;
                raw_input_feature_size[current_variable]  = 1;

                ++feature_index;
            }
        }
    }

    const Index start = raw_input_feature_start[relative_index];
    const Index size  = raw_input_feature_size[relative_index];

    if(!raw_var.is_categorical())
    {
        if(size != 1)
            throw std::runtime_error("Non-categorical input with feature_size != 1 for " + name);

        set_input_condition(start, condition, values);

        return;
    }

    if(condition == Condition::EqualTo)
    {
        if(values.size() != 1)
            throw runtime_error("EqualTo for categorical input expects 1 numeric value (category index).");

        const Index category_index = static_cast<Index>(llround(values(0)));

        if(category_index < 0 || category_index >= size)
            throw runtime_error("Category index out of range for input " + name);

        for(Index j = 0; j < size; ++j)
        {
            Tensor<type,1> value(1);
            value(0) = (j == category_index) ? type(1) : type(0);

            set_input_condition(start + j, Condition::EqualTo, value);
        }
        return;
    }

    if(condition == Condition::GreaterEqualTo
        || condition == Condition::LessEqualTo
        || condition == Condition::Between)
    {
        throw runtime_error("Inequality conditions are not supported for categorical input " + name);
    }
}



void ResponseOptimization::set_output_condition(const string& variable_name,
                                                const ResponseOptimization::Condition& condition,
                                                const Tensor<type, 1>& values)
{
    const Index raw_index = dataset->get_raw_variable_index(variable_name);

    if(raw_index < 0)
        throw runtime_error("Unknown output variable name: " + variable_name);

    const vector<Index> target_indices = dataset->get_raw_variable_indices("Target");

    Index relative_index = -1;
    for(size_t i = 0; i < target_indices.size(); ++i)
    {
        if(target_indices[i] == raw_index)
        {
            relative_index = static_cast<Index>(i);
            break;
        }
    }

    if(relative_index == -1)
        throw runtime_error("Variable " + variable_name + " is not set as Target.");

    output_conditions(relative_index) = condition;

    const vector<Dataset::RawVariable> raw_outputs = dataset->get_raw_variables("Target");

    const Dataset::RawVariable& raw_var = raw_outputs[relative_index];


    const Index raw_outputs_number  = static_cast<Index>(raw_outputs.size());

    vector<Index> raw_output_categoricals;

    vector<Index> raw_output_categories_sizes;

    dataset->get_categorical_info("Target", raw_output_categoricals, raw_output_categories_sizes);

    vector<Index> raw_output_feature_start(raw_outputs_number);
    vector<Index> raw_output_feature_size(raw_outputs_number);

    {
        Index feature_index      = 0;
        size_t category_position = 0;

        for(Index current_variable = 0; current_variable < raw_outputs_number; ++current_variable)
        {
            if(category_position < raw_output_categoricals.size()
                && raw_output_categoricals[category_position] == current_variable)
            {
                const Index one_hot_size = raw_output_categories_sizes[category_position];

                raw_output_feature_start[current_variable] = feature_index;
                raw_output_feature_size[current_variable]  = one_hot_size;

                feature_index += one_hot_size;
                ++category_position;
            }
            else
            {
                raw_output_feature_start[current_variable] = feature_index;
                raw_output_feature_size[current_variable]  = 1;

                ++feature_index;
            }
        }
    }

    const Index start = raw_output_feature_start[relative_index];

    const Index size  = raw_output_feature_size[relative_index];

    if(!raw_var.is_categorical())
    {
        if(size != 1)
            throw runtime_error("Non-categorical output with feature_size != 1 for " + variable_name);

        set_output_condition(start, condition, values);

        return;
    }

    if(condition == Condition::EqualTo)
    {
        if(values.size() != 1)
            throw runtime_error("EqualTo for categorical output expects 1 numeric value (category index).");

        const Index category_index = static_cast<Index>(llround(values(0)));

        if(category_index < 0 || category_index >= size)
            throw runtime_error("Category index out of range for output " + variable_name);

        for(Index j = 0; j < size; ++j)
        {
            Tensor<type,1> values(1);
            values(0) = (j == category_index) ? type(1) : type(0);

            set_output_condition(start + j, Condition::EqualTo, values);
        }
        return;
    }

    if(condition == Condition::GreaterEqualTo
        || condition == Condition::LessEqualTo
        || condition == Condition::Between)
    {
        throw runtime_error("Inequality conditions are not supported for categorical output " + variable_name);
    }
}


void ResponseOptimization::set_input_condition(const Index& index,
                                               const ResponseOptimization::Condition& condition,
                                               const Tensor<type, 1>& values)
{
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
    switch(condition)
    {        
    case Condition::EqualTo:
        output_minimums[index] = values[0];

        output_maximums[index] = values[0];

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
    const Index inputs_number = neural_network->get_features_number();

    Tensor<type, 2> inputs(evaluations_number, inputs_number);
    inputs.setZero();

    const Index input_raw_variables_number = dataset->get_raw_variables_number("Input");

    const vector<Index> input_raw_variables_indices = dataset->get_raw_variable_indices("Input");

    const auto& raw_vars = dataset->get_raw_variables("Input");

    const type tiny_span = type(1e-9);

    for(Index i = 0; i < evaluations_number; i++)
    {
        Index index = 0;

        for(Index j = 0; j < input_raw_variables_number; j++)
        {
            const Index used_raw_variable_index = input_raw_variables_indices[j];

            const Dataset::RawVariableType raw_variable_type = dataset->get_raw_variable_type(used_raw_variable_index);

            if(raw_variable_type == Dataset::RawVariableType::Numeric
            || raw_variable_type == Dataset::RawVariableType::Constant)
            {
                inputs(i, index) = get_random_type(input_minimums[index], input_maximums[index]);
                ++index;
            }
            else if(raw_variable_type == Dataset::RawVariableType::Binary)
            {
                const type mn  = input_minimums[index];
                const type mx  = input_maximums[index];
                const type span = mx - mn;

                type value;

                if(fabs(static_cast<double>(span)) < static_cast<double>(tiny_span))
                {
                    value = mn;
                }
                else
                {
                    value = type(rand() % 2);
                }

                inputs(i, index) = value;
                ++index;
            }
            else if(raw_variable_type == Dataset::RawVariableType::Categorical)
            {
                const Index categories_number = dataset->get_raw_variables()[used_raw_variable_index].get_categories_number();

                Index fixed_category = -1;

                for(Index category_index = 0; category_index < categories_number; ++category_index)
                {
                    const Index current_category = index + category_index;

                    const type mn   = input_minimums[current_category];
                    const type mx   = input_maximums[current_category];
                    const type span = mx - mn;

                    if(fabs(static_cast<double>(span)) < static_cast<double>(tiny_span)
                        && mn > type(0.5))
                    {
                        fixed_category = category_index;
                        break;
                    }
                }

                for(Index category_index = 0; category_index < categories_number; ++category_index)
                    inputs(i, index + category_index) = type(0);

                if(fixed_category != -1)
                {
                    inputs(i, index + fixed_category) = type(1);
                }
                else
                {
                    const Index random_category = static_cast<Index>(rand() % categories_number);
                    inputs(i, index + random_category) = type(1);
                }

                index += categories_number;
            }
            else
            {
                inputs(i, index) =
                    get_random_type(input_minimums[index],
                                    input_maximums[index]);
                ++index;
            }
        }
    }

    return inputs;
}

Tensor<type,2> ResponseOptimization::calculate_envelope(const Tensor<type,2>& inputs, const Tensor<type,2>& outputs) const
{
    const Index inputs_number = neural_network->get_features_number();
    const Index outputs_number = neural_network->get_outputs_number();

    Tensor<type, 2> envelope = assemble_matrix_matrix(inputs, outputs);

    if(envelope.size() == 0)
        return Tensor<type,2>();

    const Index raw_inputs_number  = dataset->get_raw_variables_number("Input");
    const Index raw_outputs_number = dataset->get_raw_variables_number("Target");

    // --- Inputs ---
    vector<Index> raw_input_categoricals;
    vector<Index> raw_input_categories_sizes;
    dataset->get_categorical_info("Input", raw_input_categoricals, raw_input_categories_sizes);

    const bool present_input_categorical = !raw_input_categoricals.empty();
    vector<bool> is_categorical_input(inputs_number, false);

    if(present_input_categorical)
    {
        Index feature_index      = 0;
        size_t category_position = 0;

        for(Index current_variable = 0; current_variable < raw_inputs_number; ++current_variable)
        {
            if(category_position < raw_input_categoricals.size()
                && raw_input_categoricals[category_position] == current_variable)
            {
                const Index one_hot_size = raw_input_categories_sizes[category_position];

                for(Index current_category = 0; current_category < one_hot_size; ++current_category)
                {
                    if(feature_index < inputs_number)
                        is_categorical_input[feature_index] = true;

                    ++feature_index;
                }

                ++category_position;
            }
            else
            {
                if(feature_index < inputs_number)
                    is_categorical_input[feature_index] = false;

                ++feature_index;
            }
        }
    }

    // --- Outputs ---
    vector<Index> raw_output_categoricals;
    vector<Index> raw_output_categories_sizes;
    dataset->get_categorical_info("Target", raw_output_categoricals, raw_output_categories_sizes);

    const bool present_output_categorical = !raw_output_categoricals.empty();
    vector<bool> is_categorical_output(outputs_number, false);

    if(present_output_categorical)
    {
        Index output_index      = 0;
        size_t category_position = 0;

        for(Index current_variable = 0; current_variable < raw_outputs_number; ++current_variable)
        {
            if(category_position < raw_output_categoricals.size()
                && raw_output_categoricals[category_position] == current_variable)
            {
                const Index one_hot_size = raw_output_categories_sizes[category_position];

                for(Index current_category = 0; current_category < one_hot_size; ++current_category)
                {
                    if(output_index < outputs_number)
                        is_categorical_output[output_index] = true;

                    ++output_index;
                }

                ++category_position;
            }
            else
            {
                if(output_index < outputs_number)
                    is_categorical_output[output_index] = false;

                ++output_index;
            }
        }
    }



    struct Constraint
    {
        Index col;

        type min;

        type max;

        bool  is_categorical;
    };

    vector<Constraint> constraints;

    constraints.reserve(inputs_number + outputs_number);

    const type tiny_span = type(1e-12);  // to detect "fixed" bounds
    const type equality_tol    = type(1e-9);  // tolerance when checking equality

    for(Index j = 0; j < inputs_number; ++j)
    {
        const bool is_it_categorical = present_input_categorical && is_categorical_input[j];
        const type minimum     = input_minimums(j);
        const type maximum     = input_maximums(j);

        if(!is_it_categorical)
        {
            constraints.push_back({ j, minimum, maximum, false });
        }
        else
        {
            const type span = maximum - minimum;

            if(fabs(static_cast<double>(span)) < static_cast<double>(tiny_span))
            {
                // Fixed categorical feature
                constraints.push_back({ j, minimum, maximum, true });
            }
        }
    }

    for(Index j = 0; j < outputs_number; ++j)
    {
        const bool is_it_categorical = present_output_categorical && is_categorical_output[j];
        const type minimum     = output_minimums(j);
        const type maximum     = output_maximums(j);

        if(!is_it_categorical)
        {
            constraints.push_back({ inputs_number + j, minimum, maximum, false });
        }
        else
        {
            const type span = maximum - minimum;

            if(fabs(static_cast<double>(span)) < static_cast<double>(tiny_span))
            {
                constraints.push_back({ inputs_number + j, minimum, maximum, true });
            }
        }
    }

    const Index rows_number = envelope.dimension(0);

    const Index columns_number = envelope.dimension(1);

    vector<bool> rows_to_keep_mask;

    rows_to_keep_mask.resize(static_cast<size_t>(rows_number));

    Index kept_count = 0;

    for(Index i = 0; i < rows_number; ++i)
    {
        bool fits_the_constraint = true;

        for(const auto &current_constraint : constraints)
        {
            const type value = envelope(i, current_constraint.col);

            if(!current_constraint.is_categorical)
            {
                if(value < current_constraint.min || value > current_constraint.max)
                {
                    fits_the_constraint = false;

                    break;
                }
            }
            else
            {
                if(fabs(static_cast<double>(value - current_constraint.min)) > static_cast<double>(equality_tol))
                {
                    fits_the_constraint = false;
                    break;
                }
            }
        }
        rows_to_keep_mask[static_cast<size_t>(i)] = fits_the_constraint ? 1 : 0;

        if(fits_the_constraint)
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

/* @simone versione originale
void ResponseOptimization::build_objectives_from_envelope(const Tensor<type,2>& envelope,
                                                          Tensor<type,2>& objectives,
                                                          Tensor<type,1>& sense,
                                                          Tensor<Index,1>& objective_indices) const
{
    const Index inputs_number  = neural_network->get_features_number();

    const Index outputs_number = neural_network->get_outputs_number();

    const Index raw_inputs_number  = dataset->get_raw_variables_number("Input");
    const Index raw_outputs_number = dataset->get_raw_variables_number("Target");

    Index objectives_number = 0;

    for (Index i = 0; i < raw_inputs_number; ++i)
        if (input_conditions(i) == Condition::Minimum || input_conditions(i) == Condition::Maximum)
            ++objectives_number;

    for(Index j = 0; j < raw_outputs_number; ++j)
        if(output_conditions(j) == Condition::Minimum || output_conditions(j) == Condition::Maximum)
            ++objectives_number;

    objectives.resize(envelope.dimension(0), objectives_number);

    sense.resize(objectives_number);

    objective_indices.resize(objectives_number);

    Index counter_objectives = 0;

    for (Index i = 0; i < raw_inputs_number; ++i)
    {
        if (input_conditions(i) != Condition::Minimum && input_conditions(i) != Condition::Maximum)
            continue;

        for (Index j = 0; j < envelope.dimension(0); ++j)
            objectives(j, counter_objectives) = envelope(j, i);

        sense(counter_objectives) = (input_conditions(i) == Condition::Maximum) ? type(-1) : type(1);

        objective_indices(counter_objectives) = i;

        ++counter_objectives;
    }


    for(Index j = 0; j < raw_outputs_number; ++j)
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
*/

//Versione google AI
void ResponseOptimization::build_objectives_from_envelope(const Tensor<type,2>& envelope,
                                                          Tensor<type,2>& objectives,
                                                          Tensor<type,1>& sense,
                                                          Tensor<Index,1>& objective_indices) const
{
    const Index inputs_number  = neural_network->get_features_number();

    // 1. Get counts
    const Index raw_inputs_number  = dataset->get_raw_variables_number("Input");
    const Index raw_outputs_number = dataset->get_raw_variables_number("Target");

    const vector<Index> global_input_indices = dataset->get_raw_variable_indices("Input");
    const vector<Index> global_output_indices = dataset->get_raw_variable_indices("Target");

    // 2. Re-calculate Feature Offsets (Mapping Variables -> Matrix Columns)
    // --- INPUTS ---
    vector<Index> raw_input_categoricals;
    vector<Index> raw_input_categories_sizes;
    dataset->get_categorical_info("Input", raw_input_categoricals, raw_input_categories_sizes);

    vector<Index> raw_input_feature_start(raw_inputs_number);

    {
        Index feature_index = 0;
        size_t category_position = 0;
        for(Index i = 0; i < raw_inputs_number; ++i)
        {
            raw_input_feature_start[i] = feature_index;

            if(category_position < raw_input_categoricals.size() && raw_input_categoricals[category_position] == i)
            {
                feature_index += raw_input_categories_sizes[category_position];
                ++category_position;
            }
            else
            {
                feature_index++;
            }
        }
    }

    // --- OUTPUTS ---
    vector<Index> raw_output_categoricals;
    vector<Index> raw_output_categories_sizes;
    dataset->get_categorical_info("Target", raw_output_categoricals, raw_output_categories_sizes);

    vector<Index> raw_output_feature_start(raw_outputs_number);

    {
        Index feature_index = 0;
        size_t category_position = 0;
        for(Index i = 0; i < raw_outputs_number; ++i)
        {
            raw_output_feature_start[i] = feature_index;

            if(category_position < raw_output_categoricals.size() && raw_output_categoricals[category_position] == i)
            {
                feature_index += raw_output_categories_sizes[category_position];
                ++category_position;
            }
            else
            {
                feature_index++;
            }
        }
    }

    // 3. Count Objectives
    Index objectives_number = 0;

    for (Index i = 0; i < raw_inputs_number; ++i)
        if (input_conditions(i) == Condition::Minimum || input_conditions(i) == Condition::Maximum)
            ++objectives_number;

    for(Index j = 0; j < raw_outputs_number; ++j)
        if(output_conditions(j) == Condition::Minimum || output_conditions(j) == Condition::Maximum)
            ++objectives_number;

    // 4. Build Objectives Matrix
    objectives.resize(envelope.dimension(0), objectives_number);
    sense.resize(objectives_number);
    objective_indices.resize(objectives_number);

    Index counter_objectives = 0;

    // --- Fill Inputs Objectives ---
    for (Index i = 0; i < raw_inputs_number; ++i)
    {
        if (input_conditions(i) != Condition::Minimum && input_conditions(i) != Condition::Maximum)
            continue;

        const Index column_idx = raw_input_feature_start[i];

        for (Index row = 0; row < envelope.dimension(0); ++row)
            objectives(row, counter_objectives) = envelope(row, column_idx);

        sense(counter_objectives) = (input_conditions(i) == Condition::Maximum) ? type(-1) : type(1);

        // Store Global Raw Index (0 to N)
        objective_indices(counter_objectives) = global_input_indices[i];

        ++counter_objectives;
    }

    // --- Fill Outputs Objectives ---
    for(Index j = 0; j < raw_outputs_number; ++j)
    {
        if(output_conditions(j) != Condition::Minimum && output_conditions(j) != Condition::Maximum)
            continue;

        // FIX 2: Use calculated start index + inputs_number offset
        const Index column_idx = inputs_number + raw_output_feature_start[j];

        for(Index row = 0; row < envelope.dimension(0); ++row)
            objectives(row, counter_objectives) = envelope(row, column_idx);

        sense(counter_objectives) = (output_conditions(j) == Condition::Maximum) ? type(-1) : type(1);

        // FIX 3: Store Global Raw Index (Input Count + Output Index)
        objective_indices(counter_objectives) = global_output_indices[j];

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

    const Index input_number_dimension  = neural_network->get_features_number();

    Tensor<type,2> inputs_filtered(Pareto_rows_dimension, input_number_dimension);

    for(Index i = 0; i < Pareto_rows_dimension; ++i)
        for(Index j = 0; j < input_number_dimension; ++j)
            inputs_filtered(i,j) = envelope(i,j);

    return perform_pareto_analysis(objectives, sense, inputs_filtered, envelope);
}

ResponseOptimization::SingleOrPareto ResponseOptimization::iterative_optimization(int objective_count)
{
    //@simone
     cout << "\n--- DIAGNOSTIC: iterative_optimization START ---" << endl;

    const Index max_iterations      = iterative_max_iterations;

    const type  zoom_factor         = iterative_zoom_factor;

    const type  min_span_eps        = iterative_min_span_eps;

    const type  improvement_tolerance = iterative_improvement_tolerance;

    const vector<string> feature_names  = dataset->get_raw_variable_names("Input");

    const vector<string> output_names = dataset->get_raw_variable_names("Target");

    const Index raw_inputs_number  = feature_names.size();

    const Index raw_outputs_number = output_names.size();

    const Index inputs_number  = neural_network->get_features_number();

    const Index outputs_number = neural_network->get_outputs_number();

    vector<Index> raw_input_categoricals;

    vector<Index> raw_input_categories_sizes;

    dataset->get_categorical_info("Input", raw_input_categoricals, raw_input_categories_sizes);

    const bool present_input_categorical = !raw_input_categoricals.empty();

    vector<Index> raw_output_categoricals;

    vector<Index> raw_output_categories_sizes;

    dataset->get_categorical_info("Target", raw_output_categoricals, raw_output_categories_sizes);

    const bool present_output_categorical = !raw_output_categoricals.empty();

    vector<Index> raw_input_feature_start(raw_inputs_number);
    vector<Index> raw_input_feature_size(raw_inputs_number);

    vector<Index> raw_output_feature_start(raw_outputs_number);
    vector<Index> raw_output_feature_size(raw_outputs_number);

    vector<bool> is_categorical_input(inputs_number,  false);
    vector<bool> is_categorical_output(outputs_number, false);

    //@simone
    cout << "NN Features: " << inputs_number << ", NN Outputs: " << outputs_number << endl;
     cout << "Raw Inputs: " << raw_inputs_number << ", Raw Targets: " << raw_outputs_number << endl;

    {
        Index feature_index      = 0;
        size_t category_position = 0;

        for(Index raw = 0; raw < raw_inputs_number; ++raw)
        {
            if(category_position < raw_input_categoricals.size()
                && raw_input_categoricals[category_position] == raw)
            {
                const Index one_hot_size = raw_input_categories_sizes[category_position];

                raw_input_feature_start[raw] = feature_index;
                raw_input_feature_size[raw]  = one_hot_size;

                for(Index k = 0; k < one_hot_size; ++k)
                {
                    if(feature_index < inputs_number)
                        is_categorical_input[feature_index] = true;
                    ++feature_index;
                }

                ++category_position;
            }
            else
            {
                raw_input_feature_start[raw] = feature_index;
                raw_input_feature_size[raw]  = 1;

                if(feature_index < inputs_number)
                    is_categorical_input[feature_index] = false;

                ++feature_index;
            }
        }
    }
    {
        Index feature_index      = 0;
        size_t category_position = 0;

        for(Index raw = 0; raw < raw_outputs_number; ++raw)
        {
            if(category_position < raw_output_categoricals.size()
                && raw_output_categoricals[category_position] == raw)
            {
                const Index one_hot_size = raw_output_categories_sizes[category_position];

                raw_output_feature_start[raw] = feature_index;
                raw_output_feature_size[raw]  = one_hot_size;

                for(Index k = 0; k < one_hot_size; ++k)
                {
                    if(feature_index < outputs_number)
                        is_categorical_output[feature_index] = true;
                    ++feature_index;
                }

                ++category_position;
            }
            else
            {
                raw_output_feature_start[raw] = feature_index;
                raw_output_feature_size[raw]  = 1;

                if(feature_index < outputs_number)
                    is_categorical_output[feature_index] = false;

                ++feature_index;
            }
        }
    }
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

    int  objective_column = -1;
    type sign             = type(1);
    int  numeric_count    = 0;

    //@simone
    string objective_name = "None";
    cout << "Scanning Inputs for objectives..." << endl;


    // Inputs: raw space [0, raw_inputs_number)
    for (Index raw = 0; raw < raw_inputs_number; ++raw)
    {
        const auto cond = input_conditions(raw);
        if (cond != Condition::Minimum && cond != Condition::Maximum)
            continue;

        const Index start = raw_input_feature_start[raw];
        const Index size  = raw_input_feature_size[raw];

        //@simone
        cout << "  Found Input Objective at Raw Index " << raw
             << " | Feature Start: " << start << " | Size: " << size << endl;

        // only numeric/binary objectives (size == 1)
        if (size != 1)
            continue;

        objective_column = static_cast<int>(start);
        sign             = (cond == Condition::Minimum) ? type(1) : -type(1);
        ++numeric_count;

        //@simone
        objective_name   = "Input_" + to_string(raw);

    }

    //@simone
    cout << "Scanning Outputs for objectives..." << endl;

    // Outputs: raw space [0, raw_outputs_number)
    for (Index raw = 0; raw < raw_outputs_number; ++raw)
    {
        const auto cond = output_conditions(raw);
        if (cond != Condition::Minimum && cond != Condition::Maximum)
            continue;

        const Index start = raw_output_feature_start[raw];
        const Index size  = raw_output_feature_size[raw];

        //@simone
        cout << "  Found Output Objective at Raw Index " << raw
             << " | Feature Start: " << start << " | Size: " << size << endl;

        if (size != 1)
            continue;

        objective_column = static_cast<int>(inputs_number + start);
        sign             = (cond == Condition::Minimum) ? type(1) : -type(1);
        ++numeric_count;

        //@simone
        objective_name   = "Output_" + to_string(raw);
    }
    //@simone
    cout << "Numeric Objectives Count: " << numeric_count << endl;
    cout << "Selected Objective Column: " << objective_column << " (" << objective_name << ")" << endl;

    if (objective_count <= 1 && numeric_count == 1)
    {

        //@simone
        int total_columns = inputs_number+outputs_number;
        if(objective_column < 0 || objective_column >= total_columns) {
            cerr << "CRITICAL ERROR: Objective column " << objective_column
                 << " is out of bounds! (Total Cols: " << total_columns << ")" << endl;
            throw std::runtime_error("Objective column out of bounds");
        }
        ////////////
        auto search_best = [&](Tensor<type,1>& best_row,
                               type& best_val_signed)->bool
        {

            //@simone guarda i cout
            cout << "  -> Generating inputs..." << endl;
            const Tensor<type,2> inputs  = calculate_inputs();
            cout << "  -> Calculating outputs..." << endl;
            const Tensor<type,2> outputs = neural_network->calculate_outputs<2,2>(inputs);
            cout << "  -> Calculating envelope..." << endl;
            const Tensor<type,2> envelope = calculate_envelope(inputs, outputs);

            if (envelope.size() == 0)
                return false;

            Tensor<type,1> objective_column_from_envelope =
                envelope.chip(objective_column, 1).eval();

            const type orientation = (sign > type(0)) ? type(1) : type(-1);

            Tensor<type,1> objective_column_oriented =
                (objective_column_from_envelope * orientation).eval();

            const Index best_idx = minimal_index(objective_column_oriented);

            best_val_signed = objective_column_oriented(best_idx);
            best_row        = envelope.chip(best_idx, 0);

            return true;
        };


        Tensor<type,1> best_row = Tensor<type,1>();

        type best_value = numeric_limits<double>::infinity();
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
            bool collapsed = true;

            for (Index j = 0; j < inputs_number; ++j)
            {
                if(present_input_categorical && is_categorical_input[j])
                {
                    //@simone qui devi cambiare la logica di zoom per i categiric con una strategia
                    // Freeze categorical features to best_row(j)
                    const type center = best_row(j);
                    const type fixed  = clampv(center,
                                              original_input_minimums(j),
                                              original_input_maximums(j));

                    current_input_minimums(j) = fixed;
                    current_input_maximums(j) = fixed;
                    continue;
                }

                const type span = current_input_maximums(j) - current_input_minimums(j);

                const type new_span = span * zoom_factor;

                const type center = best_row(j);

                type new_min = center - new_span * type(0.5);

                type new_max = center + new_span * type(0.5);

                // Respect original global bounds
                new_min = clampv(new_min, original_input_minimums(j),  original_input_maximums(j));
                new_max = clampv(new_max, original_input_minimums(j),  original_input_maximums(j));

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

            input_minimums  = current_input_minimums;

            input_maximums  = current_input_maximums;

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
                    //@simone anche qui la logica e strategia
                    if(present_input_categorical && is_categorical_input[j])
                    {
                        const type center = best_row(j);
                        const type fixed  = clampv(center,
                                                  original_input_minimums(j),
                                                  original_input_maximums(j));

                        current_input_minimums(j) = fixed;
                        current_input_maximums(j) = fixed;
                        continue;
                    }

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

        //@simone
cout << "--- DIAGNOSTIC: iterative_optimization END (Single) ---" << endl;
        return best_row;
    }

cout << "Entering Pareto Branch..." << endl;

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
                if(present_input_categorical && is_categorical_input[j])
                {
                    const type center = center_row(j);
                    const type fixed  = clampv(center,
                                              original_input_minimums(j),
                                              original_input_maximums(j));

                    current_input_minimums(j) = fixed;
                    current_input_maximums(j) = fixed;
                    continue;
                }

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
