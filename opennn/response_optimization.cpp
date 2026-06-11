//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "response_optimization.h"
#include "tensor_operations.h"
#include "statistics.h"
#include "neural_network.h"
#include "variable.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"

namespace opennn
{

namespace
{

inline ComparisonOperator to_comparison_operator(const ResponseOptimization::ConditionType condition)
{
    switch (condition)
    {
    case ResponseOptimization::ConditionType::EqualTo:        return ComparisonOperator::EqualTo;
    case ResponseOptimization::ConditionType::Between:        return ComparisonOperator::Between;
    case ResponseOptimization::ConditionType::GreaterEqualTo: return ComparisonOperator::GreaterEqualTo;
    case ResponseOptimization::ConditionType::LessEqualTo:    return ComparisonOperator::LessEqualTo;
    case ResponseOptimization::ConditionType::GreaterThan:    return ComparisonOperator::GreaterThan;
    case ResponseOptimization::ConditionType::LessThan:       return ComparisonOperator::LessThan;
    default:                                                  return ComparisonOperator::None;
    }
}

}

ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network)
{
    set(new_neural_network);
}


ResponseOptimization::~ResponseOptimization() = default;


void ResponseOptimization::set(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
}


void ResponseOptimization::set_condition(const string& name, const ConditionType condition, float low, float up)
{
   conditions[name] = Condition(condition, low, up);
}


vector<NamedColumn> ResponseOptimization::build_input_columns_for_formula() const
{
    const vector<Variable>& input_variables = neural_network->get_input_variables();

    vector<NamedColumn> input_columns;
    input_columns.reserve(input_variables.size());

    Index input_column = 0;

    for (const Variable& variable : input_variables)
    {
        const Index dimension = variable.get_feature_count();
        const bool is_past = (get_condition(variable.name).condition == ConditionType::Past);

        if (variable.get_role() != "Input" || is_past)
            continue;

        if (dimension == 1)
            input_columns.push_back({variable.name, input_column});

        input_column += dimension;
    }

    return input_columns;
}


vector<NamedColumn> ResponseOptimization::build_output_columns_for_formula() const
{
    const vector<Variable>& output_variables = neural_network->get_output_variables();

    vector<NamedColumn> output_columns;
    output_columns.reserve(output_variables.size());

    Index output_column = 0;

    for (const Variable& variable : output_variables)
    {
        const Index dimension = variable.get_feature_count();

        if (dimension == 1)
            output_columns.push_back({variable.name, output_column});

        output_column += dimension;
    }

    return output_columns;
}


void ResponseOptimization::set_formula_constraint(const string& expression,
                                                  const ConditionType op,
                                                  const float low,
                                                  const float up)
{
    throw_if(!neural_network,
             "ResponseOptimization: set_formula_constraint requires a neural network to be set first");

    FormulaConstraint formula_constraint;
    formula_constraint.expression = expression;
    formula_constraint.comparison_operator = to_comparison_operator(op);
    formula_constraint.low_bound = low;
    formula_constraint.up_bound = up;
    formula_constraint.uses_callback = false;

    const vector<NamedColumn> input_columns = build_input_columns_for_formula();
    const vector<NamedColumn> output_columns = build_output_columns_for_formula();

    formula_constraint.compiled = compile_formula(expression, input_columns, output_columns);

    formula_constraints.push_back(move(formula_constraint));
}


void ResponseOptimization::set_formula_constraint(function<float(const VectorR&, const VectorR&)> callback,
                                                  const ConditionType op,
                                                  const float low,
                                                  const float up)
{
    FormulaConstraint formula_constraint;
    formula_constraint.callback = move(callback);
    formula_constraint.uses_callback = true;
    formula_constraint.comparison_operator = to_comparison_operator(op);
    formula_constraint.low_bound = low;
    formula_constraint.up_bound = up;

    formula_constraint.compiled.shape = FormulaShape::Nonlinear;
    formula_constraint.compiled.scope = FormulaScope::Mixed;

    formula_constraints.push_back(move(formula_constraint));
}


void ResponseOptimization::clear_formula_constraints()
{
    formula_constraints.clear();
}


void ResponseOptimization::set_min_feasible_ratio(float new_ratio)
{
    min_feasible_ratio = new_ratio;
}


void ResponseOptimization::set_max_oversample_factor(Index new_factor)
{
    max_oversample_factor = new_factor;
}


void ResponseOptimization::clear_conditions()
{
    conditions.clear();
}


void ResponseOptimization::clear_conditions(const string& name)
{
    conditions.erase(name);
}


ResponseOptimization::Condition ResponseOptimization::get_condition(const string& name) const
{
    map<string, Condition>::const_iterator it = conditions.find(name);

    if (it != conditions.end())
        return it->second;

    return Condition(ConditionType::None);
}


void ResponseOptimization::set_fixed_history(const Tensor3& history)
{
    fixed_history = history;
    is_forecasting = true;
}


void ResponseOptimization::set_evaluations_number(const int new_evaluations_number)
{
    evaluations_number = new_evaluations_number;
}


void ResponseOptimization::set_iterations(const int new_max_iterations)
{
    max_iterations = new_max_iterations;
}


void ResponseOptimization::set_zoom_factor(float new_zoom_factor)
{
    zoom_factor = new_zoom_factor;
}


void ResponseOptimization::set_relative_tolerance(float new_relative_tolerance)
{
    relative_tolerance = new_relative_tolerance;
}


void ResponseOptimization::set_max_pareto_number(const Index new_max_pareto_number)
{
    max_pareto_number = new_max_pareto_number;
}


void ResponseOptimization::set_max_total_evaluations(const Index new_max_total_evaluations)
{
    max_total_evaluations = new_max_total_evaluations;
}


void ResponseOptimization::set_initial_sampling_factor(const Index new_initial_sampling_factor)
{
    initial_sampling_factor = max(Index(1), new_initial_sampling_factor);
}


void ResponseOptimization::set_deformation_domain_factor(float new_deformation_domain_factor)
{
    deformation_domain_factor = new_deformation_domain_factor;
}


float ResponseOptimization::get_deformation_domain_factor()
{
    return deformation_domain_factor;
}

Index ResponseOptimization::get_objectives_number() const
{
    Index objectives_number = 0;

    for (const auto& [_, constraints] : conditions)
        if (constraints.condition == ConditionType::Maximize || constraints.condition == ConditionType::Minimize)
            objectives_number++;

    return objectives_number;
}

vector<Descriptives> ResponseOptimization::get_descriptives(const string& role) const
{
    if (role == "Input")
    {
        if (neural_network->has("Scaling"))
            return static_cast<Scaling*>(neural_network->get_first("Scaling"))->get_descriptives();
    }
    else if (role == "Target")
        if (neural_network->has("Unscaling"))
            return static_cast<Unscaling*>(neural_network->get_first("Unscaling"))->get_descriptives();

    throw runtime_error("ResponseOptimization: Required Scaling/Unscaling layer for role '" + role + "' not found.");
}

pair<vector<Variable>, vector<Descriptives>> ResponseOptimization::get_variables_and_descriptives(const string& role) const
{
    const bool is_input_request = (role == "Input");

    const vector<Variable>& variables_uncheked = is_input_request ? neural_network->get_input_variables()
                                                        : neural_network->get_output_variables();

    const vector<Descriptives> descriptives_uncheked = get_descriptives(is_input_request ? "Input" : "Target");

    throw_if(variables_uncheked.size() != descriptives_uncheked.size(),
             "ResponseOptimization: Variable count and Descriptives count mismatch.");

    vector<Variable> filtered_vars;
    vector<Descriptives> filtered_desc;
    filtered_vars.reserve(variables_uncheked.size());
    filtered_desc.reserve(descriptives_uncheked.size());

    for (size_t i = 0; i < variables_uncheked.size(); ++i)
    {
        const string& var_role = variables_uncheked[i].get_role();

        const Condition current_cond = get_condition(variables_uncheked[i].name);

        if (current_cond.condition == ConditionType::Past)
            continue; // Skip this variable entirely for optimization purposes

        const bool keep = is_input_request ? (var_role == "Input")
                                           : (var_role == "Target" || var_role == "InputTarget");
        if (keep)
        {
            filtered_vars.push_back(variables_uncheked[i]);
            filtered_desc.push_back(descriptives_uncheked[i]);
        }
    }

    return {filtered_vars, filtered_desc};
}

void ResponseOptimization::Domain::set(const vector<Variable>& variables, const vector<Descriptives>& descriptives, const float deformation_domain_factor)
{
    const Index variables_number = static_cast<Index>(variables.size());

    const vector<Index> feature_dimensions = get_feature_dimensions(variables);

    const Index total_feature_dimensions = accumulate(feature_dimensions.begin(), feature_dimensions.end(), Index(0));

    inferior_frontier.resize(total_feature_dimensions);
    superior_frontier.resize(total_feature_dimensions);


    Index feature_index = 0;

    for(Index variable = 0; variable < variables_number; ++variable)
    {
        const Index feature_dimension = feature_dimensions[variable];

        if (feature_dimension > 1)
        {
            inferior_frontier.segment(feature_index, feature_dimension).setConstant(0.0f);
            superior_frontier.segment(feature_index, feature_dimension).setConstant(1.0f);
        }
        else
        {
            const float original_minimum = static_cast<float>(descriptives[variable].minimum);
            const float original_maximum = static_cast<float>(descriptives[variable].maximum);

            const float center = (original_maximum + original_minimum) * 0.5f;
            const float half_range = (original_maximum - original_minimum) * 0.5f * deformation_domain_factor;

            inferior_frontier(feature_index) = center - half_range;
            superior_frontier(feature_index) = center + half_range;
        }

        feature_index += feature_dimension;
    }
}


ResponseOptimization::Domain ResponseOptimization::get_original_domain(const string role) const
{
    const auto [variables, descriptives] = get_variables_and_descriptives(role);

    const size_t variables_number = variables.size();

    throw_if(descriptives.size() != variables_number,
             "ResponseOptimization: Descriptives count (" + to_string(descriptives.size()) +
             ") does not match variables count (" + to_string(variables_number) + ") for " + role);

    const vector<Index> feature_dimensions = get_feature_dimensions(variables);


    vector<Condition> applicable_conditions;
    applicable_conditions.reserve(variables_number);

    for (const Variable& variable : variables)
        applicable_conditions.push_back(get_condition(variable.name));

    Domain original_domain(variables, descriptives, deformation_domain_factor);

    original_domain.bound(variables, applicable_conditions);

    return original_domain;
}


ResponseOptimization::Objectives::Objectives(const ResponseOptimization& response_optimization)
{
    const Index objectives_number = response_optimization.get_objectives_number();

    throw_if(objectives_number == 0,
             "No Objectives found, make sure to set Minimize or Maximize to any variable");

    objective_sources.resize(2, objectives_number);

    objective_normalizer.resize(2, objectives_number);

    utopian_and_senses.resize(2, objectives_number);

    Index current_objective_index = 0;

    auto process_role = [&](const string& role)
    {
        const bool is_input = (role == "Input");

        const vector<Variable> variables = response_optimization.get_variables_and_descriptives(role).first;

        const vector<Index> feature_dimensions_by_role = get_feature_dimensions(variables);

        const Domain domain = response_optimization.get_original_domain(role);

        Index feature_pointer = 0;

        for (Index i = 0; i < static_cast<Index>(variables.size()); ++i)
        {
            const Condition current_condition = response_optimization.get_condition(variables[i].name);

            if (current_condition.condition == ConditionType::Maximize
            || current_condition.condition == ConditionType::Minimize)
            {
                objective_sources(0, current_objective_index) = is_input ? 1.0f : 0.0f;

                objective_sources(1, current_objective_index) = static_cast<float>(feature_pointer);

                const float inferior_frontier = domain.inferior_frontier(feature_pointer);
                const float superior_frontier = domain.superior_frontier(feature_pointer);
                const float range = superior_frontier - inferior_frontier;

                objective_normalizer(0, current_objective_index) = 1.0 / (range < EPSILON ? EPSILON : range);

                objective_normalizer(1, current_objective_index) = -inferior_frontier / (range < EPSILON ? EPSILON : range);

                if (current_condition.condition == ConditionType::Maximize)
                {
                    utopian_and_senses(0, current_objective_index) = superior_frontier;
                    utopian_and_senses(1, current_objective_index) = 1.0;
                }
                else
                {
                    utopian_and_senses(0, current_objective_index) = inferior_frontier;
                    utopian_and_senses(1, current_objective_index) = -1.0;
                }

                current_objective_index++;
            }

            feature_pointer += feature_dimensions_by_role[i];
        }
    };

    process_role("Input");
    process_role("Target");
}


void ResponseOptimization::Domain::bound(const vector<Variable>& variables, const vector<Condition>& conditions)
{
    const vector<Index> feature_dimensions = get_feature_dimensions(variables);
    Index feature_index = 0;

    for(size_t variable_index = 0; variable_index < variables.size(); ++variable_index)
    {
        const Index feature_dimension = feature_dimensions[variable_index];

        const Condition& condition = conditions[variable_index];

        if(feature_dimension == 1)
        {
            float& inferior = inferior_frontier(feature_index);
            float& superior = superior_frontier(feature_index);

            switch(condition.condition)
            {
            case ConditionType::EqualTo:
                inferior = max(inferior, condition.low_bound);
                superior = min(superior, condition.low_bound);
                break;
            case ConditionType::Between:
                inferior = max(inferior, condition.low_bound);
                superior = min(superior, condition.up_bound);
                break;
            case ConditionType::GreaterEqualTo:
                inferior = max(inferior, condition.low_bound);
                break;
            case ConditionType::LessEqualTo:
                superior = min(superior, condition.up_bound);
                break;
            case ConditionType::LessThan:
                superior = min(superior, condition.up_bound);
                break;
            case ConditionType::GreaterThan:
                inferior = max(inferior, condition.low_bound);
                break;
            default:
                break;
            }
        }
        else if(condition.condition == ConditionType::EqualTo)
        {
            const Index category_index = static_cast<Index>(llround(condition.low_bound));

            for(Index j = 0; j < feature_dimension; ++j)
            {
                inferior_frontier(feature_index + j) = (j == category_index) ? 1.0 : 0.0;
                superior_frontier(feature_index + j) = (j == category_index) ? 1.0 : 0.0;
            }
        }

        feature_index += feature_dimension;
    }
}


MatrixR ResponseOptimization::calculate_random_inputs(const Domain& input_domain, const Index evaluations_count) const
{
    const vector<Variable> variables = get_variables_and_descriptives("Input").first;

    const vector<Index> input_feature_dimensions = get_feature_dimensions(variables);

    const Index inputs_features_number = get_features_number(variables);

    const Index effective_evaluations = (evaluations_count > 0) ? evaluations_count : evaluations_number;

    MatrixR random_inputs(effective_evaluations, inputs_features_number);
    set_random_uniform(random_inputs, 0, 1);

    Index current_feature_index = 0;

    for(size_t input_variable = 0; input_variable < variables.size(); ++input_variable)
    {
        const Index categories_number = input_feature_dimensions[input_variable];

        if(categories_number == 1)
        {
            if (variables[input_variable].type == VariableType::Binary)
                random_inputs.col(current_feature_index).array() = random_inputs.col(current_feature_index).array().round();
            else
            {
                const float inferior = input_domain.inferior_frontier(current_feature_index);
                const float superior = input_domain.superior_frontier(current_feature_index);
                const float range = superior - inferior;

                random_inputs.col(current_feature_index).array() = random_inputs.col(current_feature_index).array() * range + inferior;
            }
            current_feature_index++;
        }
        else
        {
            random_inputs.block(0, current_feature_index, effective_evaluations, categories_number).setZero();

            vector<Index> allowed_categories;
            allowed_categories.reserve(categories_number);

            for(Index i = 0; i < categories_number; ++i)
                if(input_domain.superior_frontier(current_feature_index + i) > 0.5)
                    allowed_categories.push_back(i);

            throw_if(allowed_categories.empty(),
                     "ResponseOptimization: variable '"
                     + variables[input_variable].name +
                     "' has every category constrained out — cannot generate inputs.");

            for(Index i = 0; i < effective_evaluations; ++i)
                random_inputs(i, current_feature_index + allowed_categories[random_integer(0, allowed_categories.size()-1)]) = 1.0;

            current_feature_index += categories_number;
        }
    }

    repair_inputs(random_inputs,
                  input_domain.inferior_frontier,
                  input_domain.superior_frontier,
                  formula_constraints);

    return random_inputs;
}


Tensor3 ResponseOptimization::combine_input(const MatrixR& input_control) const
{
    const vector<Variable> input_variables = neural_network->get_input_variables();
    const Index batch_size = input_control.rows(); // 1000
    const Shape input_shape = neural_network->get_input_shape();
    const Index total_lags = input_shape[0]; // 2
    const Index total_features = input_shape[1]; // 8

    Tensor3 input_combined(batch_size, total_lags, total_features);

    input_combined.device(get_device()) = fixed_history.broadcast(array_3(batch_size, 1, 1));

    Index feature_cursor = 0;    
    Index candidate_col_cursor = 0; 

    for (const Variable& variable : input_variables)
    {
        const Index dim = variable.get_feature_count();

        const Condition current_cond = get_condition(variable.name);

        if (variable.get_role() == "Input" && current_cond.condition != ConditionType::Past)
        {
           const MatrixR block_data = input_control.block(0, candidate_col_cursor, batch_size, dim);

            TensorMap<const Tensor<float, 3, Layout>> block_tensor(block_data.data(), batch_size, 1, dim);

            input_combined.slice(array_3(0, total_lags - 1, feature_cursor), array_3(batch_size, 1, dim)).device(get_device()) = block_tensor;

            candidate_col_cursor += dim;
        }

        feature_cursor += dim;
    }

    return input_combined;
}


MatrixR ResponseOptimization::calculate_outputs(const MatrixR& input) const
{
    if (is_forecasting)
    {
        throw_if(fixed_history.size() == 0,
                 "ResponseOptimization: Model is forecasting but fixed_history is empty. Call set_fixed_history() first.");

        const Tensor3 formatted_input = combine_input(input);

        return neural_network->calculate_outputs(formatted_input);
    }

    return neural_network->calculate_outputs(input);
}


void ResponseOptimization::Domain::reshape(const float zoom_factor,
                                           const VectorR& center,
                                           const MatrixR& points_inputs,
                                           const vector<Variable>& variables)
{
    const vector<Index> feature_dimensions = get_feature_dimensions(variables);

    VectorR categories_to_save = points_inputs.colwise().maxCoeff();

    for(Index i = 0; i < categories_to_save.size(); ++i)
        if(center(i) > categories_to_save(i))
            categories_to_save(i) = center(i);

    Index current_feature_index = 0;

    for(size_t input_variable = 0; input_variable < variables.size(); ++input_variable)
    {
        const Index categories_number = feature_dimensions[input_variable];

        if(categories_number == 1 && variables[input_variable].type != VariableType::Binary)
        {
            const float half_span = (superior_frontier(current_feature_index) - inferior_frontier(current_feature_index)) * zoom_factor / 2;
            inferior_frontier(current_feature_index) = max(center(current_feature_index) - half_span, inferior_frontier(current_feature_index));
            superior_frontier(current_feature_index) = min(center(current_feature_index) + half_span, superior_frontier(current_feature_index));

        }
        else
        {
            for(Index category_index = 0; category_index < categories_number; ++category_index)
            {
                const Index current_category = current_feature_index + category_index;
                inferior_frontier(current_category) = max(categories_to_save(current_category), inferior_frontier(current_category));
                superior_frontier(current_category) = min(categories_to_save(current_category), superior_frontier(current_category));
            }
        }

        current_feature_index += categories_number;
    }
}


bool ResponseOptimization::row_satisfies_formula_constraints(const VectorR& input_row,
                                                             const VectorR& output_row) const
{
    for (const FormulaConstraint& formula_constraint : formula_constraints)
    {
        const float evaluated_value = formula_constraint.uses_callback
            ? formula_constraint.callback(input_row, output_row)
            : formula_constraint.compiled.evaluate(input_row, output_row);

        const float low_bound = formula_constraint.low_bound;
        const float up_bound = formula_constraint.up_bound;

        bool constraint_satisfied = true;

        switch (formula_constraint.comparison_operator)
        {
        case ComparisonOperator::EqualTo:
            constraint_satisfied = abs(evaluated_value - low_bound) <= bound_tolerance(low_bound);
            break;
        case ComparisonOperator::Between:
            constraint_satisfied = (evaluated_value >= low_bound - bound_tolerance(low_bound))
                                && (evaluated_value <= up_bound + bound_tolerance(up_bound));
            break;
        case ComparisonOperator::GreaterEqualTo:
            constraint_satisfied = evaluated_value >= low_bound - bound_tolerance(low_bound);
            break;
        case ComparisonOperator::LessEqualTo:
            constraint_satisfied = evaluated_value <= up_bound + bound_tolerance(up_bound);
            break;
        case ComparisonOperator::GreaterThan:
            constraint_satisfied = evaluated_value > low_bound;
            break;
        case ComparisonOperator::LessThan:
            constraint_satisfied = evaluated_value < up_bound;
            break;
        default:
            constraint_satisfied = true;
            break;
        }

        if (!constraint_satisfied)
            return false;
    }

    return true;
}


pair<MatrixR, MatrixR> ResponseOptimization::filter_feasible_points(const MatrixR& inputs,
                                                                    const MatrixR& outputs,
                                                                    const Domain& output_domain) const
{
    const vector<Variable>& all_target_variables = neural_network->get_output_variables();
    const Index rows_number = outputs.rows();

    vector<Index> feasible_indices(rows_number);
    iota(feasible_indices.begin(), feasible_indices.end(), 0);

    Index domain_index = 0;

    for (Index column_index = 0; column_index < static_cast<Index>(all_target_variables.size()); ++column_index)
    {
        const Condition current_condition = get_condition(all_target_variables[column_index].name);

        if (current_condition.condition == ConditionType::Past)
            continue; // not in domain — skip without advancing domain_index

        if (!(current_condition.condition == ConditionType::Maximize ||
              current_condition.condition == ConditionType::Minimize ||
              current_condition.condition == ConditionType::None))
        {
            feasible_indices = filter_selected_indices_by_column(outputs,
                                                                 feasible_indices,
                                                                 column_index,
                                                                 output_domain.inferior_frontier(domain_index),
                                                                 output_domain.superior_frontier(domain_index));
        }

        ++domain_index;

        if (feasible_indices.empty())
            break;
    }

    if (!formula_constraints.empty() && !feasible_indices.empty())
    {
        vector<Index> formula_feasible_indices;
        formula_feasible_indices.reserve(feasible_indices.size());

        if (all_formula_constraints_are_linear(formula_constraints))
        {
            // All-linear shortcut: one matrix product evaluates every constraint for every surviving candidate.
            const Index k     = static_cast<Index>(feasible_indices.size());
            const Index n_in  = inputs.cols();
            const Index n_out = outputs.cols();

            MatrixR X(k, n_in);
            MatrixR Y(k, n_out);
            for (Index i = 0; i < k; ++i)
            {
                X.row(i) = inputs.row(feasible_indices[i]);
                Y.row(i) = outputs.row(feasible_indices[i]);
            }

            const LinearConstraintSet linear_set = build_linear_constraint_set(formula_constraints, n_in, n_out);

            const MatrixR values = X * linear_set.A.leftCols(n_in).transpose()
                                 + Y * linear_set.A.rightCols(n_out).transpose();   // (k, m)

            for (Index i = 0; i < k; ++i)
            {
                const bool feasible = ((values.row(i).transpose().array() >= linear_set.lower.array()) &&
                                       (values.row(i).transpose().array() <= linear_set.upper.array())).all();

                if (feasible)
                    formula_feasible_indices.push_back(feasible_indices[i]);
            }
        }
        else
        {
            for (const Index row_index : feasible_indices)
            {
                const VectorR input_row = inputs.row(row_index).transpose();
                const VectorR output_row = outputs.row(row_index).transpose();

                if (row_satisfies_formula_constraints(input_row, output_row))
                    formula_feasible_indices.push_back(row_index);
            }
        }

        feasible_indices = move(formula_feasible_indices);
    }

    if (feasible_indices.empty())
        return {MatrixR(), MatrixR()};

    const Index feasible_count = static_cast<Index>(feasible_indices.size());

    MatrixR feasible_inputs(feasible_count, inputs.cols());
    MatrixR feasible_outputs(feasible_count, outputs.cols());

    for (Index i = 0; i < feasible_count; ++i)
    {
        feasible_inputs.row(i) = inputs.row(feasible_indices[i]);
        feasible_outputs.row(i) = outputs.row(feasible_indices[i]);
    }

    return {feasible_inputs, feasible_outputs};
}


pair<MatrixR, MatrixR> ResponseOptimization::sample_feasible_points(const Domain& input_domain,
                                                                    const Domain& output_domain,
                                                                    const Index evaluations_multiplier) const
{
    const Index multiplier = max(Index(1), evaluations_multiplier);

    if (formula_constraints.empty())
        return generate_feasible_points(input_domain, output_domain, evaluations_number * multiplier);

    const Index base_evaluations = evaluations_number * multiplier;
    const Index evaluations_cap = base_evaluations * max_oversample_factor;
    const float low_ratio_threshold = min_feasible_ratio * float(0.25);

    Index current_evaluations = base_evaluations;
    Index consecutive_low_ratio_attempts = 0;

    pair<MatrixR, MatrixR> feasible_result;

    while (true)
    {
        feasible_result = generate_feasible_points(input_domain, output_domain, current_evaluations);

        const Index feasible_count = feasible_result.first.rows();
        const float feasible_ratio = (current_evaluations > 0)
            ? float(feasible_count) / float(current_evaluations)
            : float(0);

        if (feasible_count > 0 && feasible_ratio >= min_feasible_ratio)
            return feasible_result;

        if (feasible_ratio < low_ratio_threshold)
            ++consecutive_low_ratio_attempts;
        else
            consecutive_low_ratio_attempts = 0;

        if (current_evaluations >= evaluations_cap)
            break;
        if (consecutive_low_ratio_attempts >= 2)
            break;

        const Index next_evaluations = min(current_evaluations * Index(2), evaluations_cap);

        cout << "> Adaptive oversampling: feasible ratio " << feasible_ratio
             << " below threshold " << min_feasible_ratio
             << ", increasing evaluations from " << current_evaluations
             << " to " << next_evaluations << endl;

        current_evaluations = next_evaluations;
    }

    const bool early_stop_triggered = (consecutive_low_ratio_attempts >= 2);
    const Index final_feasible_count = feasible_result.first.rows();

    throw_if(final_feasible_count == 0,
             "ResponseOptimization: formula constraints appear infeasible — "
             "no feasible points found after adaptive oversampling up to "
             + to_string(current_evaluations) + " evaluations.");

    throw_if(early_stop_triggered,
             "ResponseOptimization: formula constraints are too tight — "
             "feasibility ratio stayed below "
             + to_string(low_ratio_threshold)
             + " over consecutive oversampling attempts (up to "
             + to_string(current_evaluations) + " evaluations, "
             + to_string(final_feasible_count) + " feasible points found).");

    return feasible_result;
}


pair<MatrixR, MatrixR> ResponseOptimization::generate_feasible_points(const Domain& input_domain,
                                                         const Domain& output_domain,
                                                         const Index evaluations_count) const
{
    MatrixR random_inputs = calculate_random_inputs(input_domain, evaluations_count);

    // Repair constraints that reference the network outputs (no-op when none):
    // Gauss-Newton onto g(x, f(x)) = c. Prefer the exact analytic Jacobian
    // (NetworkDifferential); otherwise fall back to a finite-difference VJP over
    // calculate_outputs. Either way the exact forward drives the residual to
    // zero, so the repaired points are exact to tolerance.
    if (network_differential)
    {
        repair_output_constraints(random_inputs,
                                  input_domain.inferior_frontier,
                                  input_domain.superior_frontier,
                                  formula_constraints,
                                  [this](const VectorR& x) { return network_differential->forward(x); },
                                  [this](const VectorR& x, const VectorR& cotangent) { return network_differential->vjp(x, cotangent); });
    }
    else
    {
        const SurrogateForward forward = [this](const VectorR& x) -> VectorR
        {
            MatrixR single(1, x.size());
            single.row(0) = x.transpose();
            return calculate_outputs(single).row(0).transpose();
        };

        repair_output_constraints(random_inputs,
                                  input_domain.inferior_frontier,
                                  input_domain.superior_frontier,
                                  formula_constraints,
                                  forward);
    }

    const MatrixR outputs = calculate_outputs(random_inputs);
    evaluations_used += evaluations_count;   // total surrogate-evaluation budget tracking
    return filter_feasible_points(random_inputs, outputs, output_domain);
}


MatrixR ResponseOptimization::Objectives::extract(const MatrixR& inputs, const MatrixR& outputs) const
{
    const Index objectives_number = objective_sources.cols();

    MatrixR objective_matrix(inputs.rows(), objectives_number);

    for (Index j = 0; j < objectives_number; ++j)
        objective_matrix.col(j)= (objective_sources(0, j) > 0.5)
              ? inputs.col(static_cast<Index>(objective_sources(1, j)))
              : outputs.col(static_cast<Index>(objective_sources(1, j)));

    return objective_matrix;
}


void ResponseOptimization::Objectives::normalize(MatrixR& objective_matrix) const
{
    const auto combined_scale = objective_normalizer.row(0).array() * utopian_and_senses.row(1).array();
    const auto combined_offset = objective_normalizer.row(1).array() * utopian_and_senses.row(1).array();

    objective_matrix.array().rowwise() *= combined_scale;
    objective_matrix.array().rowwise() += combined_offset;
}


bool ResponseOptimization::Objectives::update_utopian_from_points(const MatrixR& unnormalized_objective_values)
{
    if (unnormalized_objective_values.rows() == 0)
        return false;

    const Index objectives_number = utopian_and_senses.cols();

    if (unnormalized_objective_values.cols() != objectives_number)
        return false;

    bool any_updated = false;

    for (Index j = 0; j < objectives_number; ++j)
    {
        const float sense = utopian_and_senses(1, j);
        const float current_utopian = utopian_and_senses(0, j);

        const float best = (sense > 0)
            ? unnormalized_objective_values.col(j).maxCoeff()
            : unnormalized_objective_values.col(j).minCoeff();

        if (sense * (best - current_utopian) <= float(0))
            continue;

        const float scale = objective_normalizer(0, j);
        const float offset = objective_normalizer(1, j);

        if (abs(scale) < EPSILON)
            continue;

        const float old_inferior = -offset / scale;
        const float old_superior = old_inferior + float(1) / scale;

        const float new_inferior = (sense > 0) ? old_inferior : best;
        const float new_superior = (sense > 0) ? best : old_superior;
        const float new_range = new_superior - new_inferior;

        if (new_range < EPSILON)
            continue;

        utopian_and_senses(0, j) = best;
        objective_normalizer(0, j) = float(1) / new_range;
        objective_normalizer(1, j) = -new_inferior / new_range;

        any_updated = true;
    }

    return any_updated;
}


pair<MatrixR, MatrixR> ResponseOptimization::calculate_optimal_points(const MatrixR& feasible_inputs,
                                                                      const MatrixR& feasible_outputs,
                                                                      const Objectives& objectives) const
{
    const Index subset_dimension = clamp<Index>(llround(zoom_factor * evaluations_number), 1, feasible_outputs.rows());

    MatrixR objective_matrix = objectives.extract(feasible_inputs, feasible_outputs);

    objectives.normalize(objective_matrix);

    const VectorR normalized_utopian_point = (objectives.utopian_and_senses.row(1).array() + (float)1.0) / (float)2.0;

    const VectorI nearest_rows = get_nearest_points(objective_matrix, normalized_utopian_point , (int)subset_dimension);

    MatrixR nearest_inputs(subset_dimension, feasible_inputs.cols());
    MatrixR nearest_outputs(subset_dimension, feasible_outputs.cols());

    for(Index i = 0; i < subset_dimension; ++i)
    {
        nearest_inputs.row(i) = feasible_inputs.row(nearest_rows(i));
        nearest_outputs.row(i) = feasible_outputs.row(nearest_rows(i));
    }

    return {nearest_inputs, nearest_outputs};
}


MatrixR ResponseOptimization::perform_single_objective_optimization() const
{
    const Objectives objectives(*this);

    const vector<Variable> input_variables = get_variables_and_descriptives("Input").first;

    const Domain original_input_domain = get_original_domain("Input");
    const Domain original_output_domain = get_original_domain("Target");
    Domain input_domain = original_input_domain;

    pair<MatrixR, MatrixR> optimal_set;

    float optimal_point;

    float previous_optimal_point = 0;

    cout << "> Optimization loop starting with zoom factor: " << zoom_factor << endl;

    for (Index i = 0; i < max_iterations; i++)
    {
        auto [feasible_inputs, feasible_outputs] = sample_feasible_points(input_domain, original_output_domain);

        if (feasible_outputs.size() > 0 && !feasible_outputs.allFinite())
        {
            cout << "Model produced NaN — aborting optimization loop." << endl;
            break;
        }

        if (feasible_inputs.rows() == 0)
        {
            cout << "!!! [Critical] Zero feasible points found. "
                 << "Check if your constraints are too strict. "
                 << "Aborting optimization loop." << endl;
            break;
        }

        cout << "\n> [Iteration " << i + 1 << " / " << max_iterations << "]" << endl;
        cout << "  - Feasible points: " << feasible_inputs.rows() << endl;

        optimal_set = calculate_optimal_points(feasible_inputs, feasible_outputs, objectives);

        if (optimal_set.first.rows() == 0 || optimal_set.second.rows() == 0)
        {
            cout << "!!! [Critical] calculate_optimal_points returned empty. "
                 << "Aborting optimization loop." << endl;
            break;
        }

        optimal_point = (objectives.objective_sources(0, 0) > 0.5f
            ? optimal_set.first
            : optimal_set.second)(0, static_cast<Index>(objectives.objective_sources(1, 0)));

        const float relative_error = abs((optimal_point - previous_optimal_point) / (objectives.utopian_and_senses(0,0) + 1e-6f));

        cout << "  - Relative error: " << relative_error << endl;

        if (relative_error < relative_tolerance && i > min_iterations)
        {
            cout << "> Optimization loop stopped for reaching the relative tolerance desired: " << relative_tolerance << endl;
            break;
        }

        previous_optimal_point = optimal_point;

        input_domain.reshape(zoom_factor, optimal_set.first.row(0), optimal_set.first, input_variables);
    }

    return optimal_set.first.rows() == 0
        ? MatrixR()
        : append_columns(optimal_set.first, optimal_set.second);
}


pair<MatrixR, MatrixR> ResponseOptimization::calculate_pareto(const MatrixR& inputs,
                                                              const MatrixR& outputs,
                                                              const MatrixR& objective_matrix) const
{
    const Index rows_number = inputs.rows();

    if (rows_number == 0)
        return {};

    vector<int> non_dominated(static_cast<size_t>(rows_number), 1);

    for (Index i = 0; i < rows_number; ++i)
    {
        const auto row_i = objective_matrix.row(i);

        if(!row_i.allFinite())
        {
            non_dominated[i] = 0;
            continue;
        }

        for (Index j = 0; j < rows_number; ++j)
        {
            if (i == j)
                continue;

            const auto row_j = objective_matrix.row(j);

            if ((row_j.array() >= row_i.array()).all() && (row_j.array() > row_i.array()).any())
            {
                non_dominated[i] = 0;
                break;
            }
        }
    }

    vector<Index> non_dominated_indices;
    non_dominated_indices.reserve(rows_number);

    for (Index i = 0; i < rows_number; ++i)
        if (non_dominated[i] == 1)
            non_dominated_indices.push_back(i);

    const Index pareto_size = static_cast<Index>(non_dominated_indices.size());

    MatrixR pareto_inputs(pareto_size, inputs.cols());
    MatrixR pareto_outputs(pareto_size, outputs.cols());

    for (Index i = 0; i < (Index)non_dominated_indices.size(); ++i)
    {       
        pareto_inputs.row(i) = inputs.row(non_dominated_indices[i]);
        pareto_outputs.row(i) = outputs.row(non_dominated_indices[i]);
    }

    return {pareto_inputs, pareto_outputs};
}


pair<float, float> ResponseOptimization::calculate_quality_metrics(const MatrixR& inputs,
                                                                 const MatrixR& outputs,
                                                                 const Objectives& objectives) const
{
    const Index points_number = inputs.rows();

    if (points_number == 0)
        return {static_cast<float>(1e6), static_cast<float>(1e6)};

    MatrixR objective_matrix = objectives.extract(inputs, outputs);

    objectives.normalize(objective_matrix);

    const Index objectives_number = objective_matrix.cols();

    const float hypercube_diagonal = sqrt(static_cast<float>(objectives_number));

    float maximum_internal_gap = 0.0;

    for (Index i = 0; i < points_number; ++i)
    {
        const auto current_point = objective_matrix.row(i);

        VectorR  distances = (objective_matrix.rowwise() - current_point).rowwise().squaredNorm();

        distances(i) = MAX;

        const float minimum_neighbor_distance = sqrt(distances.minCoeff());

        maximum_internal_gap = max(maximum_internal_gap, minimum_neighbor_distance);
    }

    if (points_number == 1)
        maximum_internal_gap = 1.0;

    maximum_internal_gap /= hypercube_diagonal;

    const VectorR max_objectives = objective_matrix.colwise().maxCoeff();

    const float sum_boundary_gaps = (1.0 - max_objectives.array()).abs().sum();

    const float average_boundary_gap = sum_boundary_gaps / static_cast<float>(objectives_number);
    const float normalized_boundary_gap = average_boundary_gap / hypercube_diagonal;

    return {maximum_internal_gap, normalized_boundary_gap};
}


MatrixR ResponseOptimization::perform_multiobjective_optimization() const
{
    Objectives objectives(*this);

    const vector<Variable> input_variables = get_variables_and_descriptives("Input").first;

    const Domain original_input_domain = get_original_domain("Input");
    const Domain original_output_domain = get_original_domain("Target");

    // The initial, full-domain pass draws a broader candidate set
    // (initial_sampling_factor x evaluations_number) to seed the contraction
    // from a wider sample; per-Pareto-point sampling below keeps the base count.
    auto [first_feasible_inputs, first_feasible_outputs] = sample_feasible_points(original_input_domain, original_output_domain, initial_sampling_factor);

    if (first_feasible_inputs.rows() == 0)
    {
        cout << "!!! [Critical] Zero feasible points found. "
             << "Check if your constraints are too strict." << endl;
        return MatrixR();
    }

    MatrixR first_objective_matrix  = objectives.extract(first_feasible_inputs, first_feasible_outputs);
    objectives.normalize(first_objective_matrix);

    auto [global_pareto_inputs, global_pareto_outputs] = calculate_pareto(first_feasible_inputs, first_feasible_outputs, first_objective_matrix);

    if (global_pareto_inputs.rows() > 0)
    {
        const MatrixR initial_pareto_unnormalized = objectives.extract(global_pareto_inputs, global_pareto_outputs);
        if (objectives.update_utopian_from_points(initial_pareto_unnormalized))
            cout << "> Utopian promoted from initial Pareto front." << endl;
    }

    cout << "> Initial Pareto front size: " << global_pareto_inputs.rows() << " points." << endl;

    vector<Domain> input_domains(static_cast<size_t>(global_pareto_inputs.rows()), original_input_domain);

    float current_zoom = zoom_factor;

    float previous_holes_magnitude = 0.0;
    float previous_area_covered = 0.0;

    cout << "> Optimization loop starting with zoom factor: " << current_zoom << endl;

    for (Index i = 0; i < max_iterations; i++)
    {
        cout << "\n> [Iteration " << i + 1 << " / " << max_iterations << "]" << endl;

        MatrixR candidate_inputs = global_pareto_inputs;
        MatrixR candidate_outputs = global_pareto_outputs;

        for (Index j = 0; j < global_pareto_inputs.rows(); j++)
        {
            // Matched-budget stop: once the total surrogate-evaluation budget
            // is spent, stop launching new local-sampling calls and use the
            // candidates aggregated so far this iteration.
            if (max_total_evaluations > 0 && evaluations_used >= max_total_evaluations)
                break;

            auto [local_feasible_inputs, local_feasible_outputs] = sample_feasible_points(input_domains[j], original_output_domain);

            MatrixR local_objective_matrix  = objectives.extract(local_feasible_inputs, local_feasible_outputs);
            objectives.normalize(local_objective_matrix);

            auto [local_pareto_input, local_pareto_output] = calculate_pareto(local_feasible_inputs, local_feasible_outputs, local_objective_matrix );

            candidate_inputs = append_rows(candidate_inputs, local_pareto_input);
            candidate_outputs = append_rows(candidate_outputs, local_pareto_output);
        }

        cout << "  - Aggregated local Pareto candidates: " << candidate_inputs.rows() << endl;

        if (candidate_inputs.rows() == 0)
            break;

        pair<MatrixR, MatrixR> optimal_set = calculate_optimal_points(candidate_inputs, candidate_outputs, objectives);

        MatrixR objective_matrix = objectives.extract(candidate_inputs, candidate_outputs);
        objectives.normalize(objective_matrix);

        const auto pareto_pair = calculate_pareto(candidate_inputs, candidate_outputs, objective_matrix);

        global_pareto_inputs = pareto_pair.first;
        global_pareto_outputs = pareto_pair.second;

        cout << "  - New Pareto front size: " << global_pareto_inputs.rows()  << endl;

        if (max_pareto_number > 0 && global_pareto_inputs.rows() >= max_pareto_number)
        {
            cout << "> [Pareto cap] Front reached " << global_pareto_inputs.rows()
                 << " points (cap=" << max_pareto_number
                 << "). Stopping at iteration " << i + 1 << "." << endl;
            break;
        }

        if (max_total_evaluations > 0 && evaluations_used >= max_total_evaluations)
        {
            cout << "> [Budget cap] Reached " << evaluations_used
                 << " surrogate evaluations (budget=" << max_total_evaluations
                 << "). Stopping at iteration " << i + 1 << "." << endl;
            break;
        }

        if (global_pareto_inputs.rows() > 0)
        {
            const MatrixR pareto_objective_unnormalized = objectives.extract(global_pareto_inputs, global_pareto_outputs);
            if (objectives.update_utopian_from_points(pareto_objective_unnormalized))
            {
                cout << "  - Utopian promoted to better Pareto coordinate." << endl;
                previous_holes_magnitude = 0.0;
                previous_area_covered = 0.0;
            }
        }

        const pair<float, float> quality = calculate_quality_metrics(global_pareto_inputs, global_pareto_outputs, objectives);

        const float current_hole = quality.first;
        const float current_boundary = quality.second;

        cout << "  - Internal Hole: " << current_hole << " | Boundary Gap: " << current_boundary << endl;

        const float delta_hole = abs(current_hole - previous_holes_magnitude);
        const float delta_boundary = abs(current_boundary - previous_area_covered);

        if (i > min_iterations && delta_hole < relative_tolerance && delta_boundary < relative_tolerance)
        {
            cout << "> [Convergence] Quality metrics stabilized. Stopping at iteration " << i + 1 << endl;
            break;
        }

        previous_holes_magnitude = current_hole;
        previous_area_covered = current_boundary;

        input_domains.reserve(static_cast<size_t>(global_pareto_inputs.rows()));
        input_domains.assign(static_cast<size_t>(global_pareto_inputs.rows()), original_input_domain);

        const MatrixR best_and_pareto = append_rows(optimal_set.first,global_pareto_inputs);

        for (Index j = 0; j < global_pareto_inputs.rows(); j++)
            input_domains[j].reshape(current_zoom, global_pareto_inputs.row(j), best_and_pareto , input_variables);

        current_zoom *= zoom_factor;
    }
    cout << "\n> [Optimization Complete] Assembling final results..." << endl;
    cout << "> Total surrogate evaluations used: " << evaluations_used << endl;

    return append_columns(global_pareto_inputs, global_pareto_outputs);
}


vector<float> ResponseOptimization::get_utopian_point() const
{
    const Objectives objectives(*this);

    const Index objectives_number = objectives.utopian_and_senses.cols();

    vector<float> utopian_point(static_cast<size_t>(objectives_number));

    for (Index j = 0; j < objectives_number; ++j)
        utopian_point[static_cast<size_t>(j)] = objectives.utopian_and_senses(0, j);

    return utopian_point;
}


pair<Index, VectorR> ResponseOptimization::get_advised_point(const MatrixR& pareto_front,
                                                             const VectorR& importance_scale) const
{
    if (pareto_front.rows() == 0)
        return {-1, VectorR()};

    if (pareto_front.rows() == 1)
        return {0, pareto_front.row(0).transpose()};

    const Index objectives_number = get_objectives_number();

    VectorR scale = (importance_scale.size() == 0)
        ? VectorR::Ones(objectives_number)
        : importance_scale;

    throw_if(scale.size() != objectives_number, "Importance scale size must match objectives number.\n");

    throw_if(scale.minCoeff() < float(0), "Importance scale must be non-negative.\n");

    throw_if(scale.maxCoeff() == float(0), "Importance scale must contain at least one non-zero entry.\n");

    const Index inputs_number = neural_network->get_inputs_number();

    throw_if(pareto_front.cols() < inputs_number,
             "Pareto front has fewer columns than the number of input features.\n");

    const MatrixR pareto_inputs  = pareto_front.leftCols(inputs_number);
    const MatrixR pareto_outputs = pareto_front.rightCols(pareto_front.cols() - inputs_number);

    const Objectives objectives(*this);

    MatrixR objective_matrix = objectives.extract(pareto_inputs, pareto_outputs);

    VectorR normalized_utopian(objectives_number);

    for (Index j = 0; j < objectives_number; ++j)
    {
        const float col_min = objective_matrix.col(j).minCoeff();
        const float col_max = objective_matrix.col(j).maxCoeff();
        const float col_range = col_max - col_min;

        if (col_range > EPSILON)
            objective_matrix.col(j).array() = (objective_matrix.col(j).array() - col_min) / col_range;
        else
            objective_matrix.col(j).setZero();

        const float sense = objectives.utopian_and_senses(1, j);

        normalized_utopian(j) = (sense > float(0)) ? float(1) : float(0);
    }

    for (Index j = 0; j < objectives_number; ++j)
    {
        objective_matrix.col(j) *= scale(j);
        normalized_utopian(j)   *= scale(j);
    }

    const VectorI nearest = get_nearest_points(objective_matrix, normalized_utopian, 1);

    const Index advised_row_index = nearest(0);

    return {advised_row_index, pareto_front.row(advised_row_index).transpose()};
}


void ResponseOptimization::initialize_network_differential() const
{
    network_differential.reset();

    if (!neural_network || is_forecasting)
        return;

    // Only needed when some constraint references the network outputs.
    bool has_output_constraint = false;
    for (const FormulaConstraint& constraint : formula_constraints)
        if (!constraint.uses_callback
            && constraint.comparison_operator != ComparisonOperator::None
            && constraint.compiled.scope != FormulaScope::InputsOnly)
            has_output_constraint = true;

    if (!has_output_constraint)
        return;

    auto candidate = make_unique<NetworkDifferential>();

    try { candidate->build(*neural_network); }
    catch (const exception& error)
    {
        cout << "!!! [Warning] Analytic surrogate Jacobian unavailable (" << error.what()
             << "); falling back to the finite-difference VJP." << endl;
        return;
    }

    // Self-validation ("stay in sync"): the analytic forward must match
    // calculate_outputs over the input domain. Sample within the scaling
    // descriptives' range when a scaling layer is present.
    const Index inputs_number = neural_network->get_inputs_number();

    VectorR lower = VectorR::Constant(inputs_number, -1.0f);
    VectorR upper = VectorR::Constant(inputs_number,  1.0f);
    if (!candidate->layers.empty() && candidate->layers.front().kind == NetworkDifferential::Kind::Scale)
    {
        lower = candidate->layers.front().minimum;
        upper = candidate->layers.front().maximum;
    }

    // Validation runs ONCE per optimization (not per evaluation), so it is
    // negligible against the run. The cheap forward check uses more probes (it
    // is the drift guard); the costlier VJP check (2*n_in forwards per probe)
    // uses a few, enough to catch a broken Jacobian.
    const Index forward_probes = 16;
    const Index vjp_probes = 4;

    MatrixR probe(forward_probes, inputs_number);
    set_random_uniform(probe, 0, 1);
    for (Index j = 0; j < inputs_number; ++j)
        probe.col(j) = (lower(j) + probe.col(j).array() * (upper(j) - lower(j))).matrix();

    const MatrixR reference = calculate_outputs(probe);
    const Index outputs_number = reference.cols();

    VectorR fd_step(inputs_number);
    for (Index j = 0; j < inputs_number; ++j)
        fd_step(j) = max(1e-4f, 1e-3f * (upper(j) - lower(j)));

    const VectorR cotangent = VectorR::Ones(outputs_number);

    float worst_forward_error = 0.0f;
    float worst_vjp_error = 0.0f;

    for (Index i = 0; i < forward_probes; ++i)
    {
        const VectorR x = probe.row(i).transpose();

        // (a) forward must match the network (cheap; every probe).
        const VectorR analytic_output = candidate->forward(x);
        const VectorR truth = reference.row(i).transpose();
        worst_forward_error = max(worst_forward_error,
            (analytic_output - truth).cwiseAbs().maxCoeff() / (1.0f + truth.cwiseAbs().maxCoeff()));

        // (b) analytic VJP must match a central-difference VJP (a few probes).
        if (i >= vjp_probes) continue;

        const VectorR analytic_gradient = candidate->vjp(x, cotangent);
        VectorR finite_difference_gradient = VectorR::Zero(inputs_number);
        for (Index k = 0; k < inputs_number; ++k)
        {
            MatrixR plus(1, inputs_number), minus(1, inputs_number);
            plus.row(0) = x.transpose();   minus.row(0) = x.transpose();
            plus(0, k) += fd_step(k);      minus(0, k) -= fd_step(k);
            const VectorR forward_plus  = calculate_outputs(plus).row(0).transpose();
            const VectorR forward_minus = calculate_outputs(minus).row(0).transpose();
            finite_difference_gradient(k) = cotangent.dot(forward_plus - forward_minus) / (2.0f * fd_step(k));
        }
        worst_vjp_error = max(worst_vjp_error,
            (analytic_gradient - finite_difference_gradient).cwiseAbs().maxCoeff()
            / (1.0f + analytic_gradient.cwiseAbs().maxCoeff()));
    }

    if (worst_forward_error < 1e-3f && worst_vjp_error < 2e-2f)
    {
        network_differential = move(candidate);
        cout << "> Analytic surrogate Jacobian active (forward error " << worst_forward_error
             << ", VJP-vs-finite-difference error " << worst_vjp_error << ")." << endl;
    }
    else
        cout << "!!! [Warning] Analytic surrogate Jacobian failed validation (forward error "
             << worst_forward_error << ", VJP error " << worst_vjp_error
             << "); falling back to the finite-difference VJP." << endl;
}


MatrixR ResponseOptimization::perform_response_optimization() const
{
    const Index objectives_number = get_objectives_number();

    throw_if(objectives_number == 0, "No objectives found\n");

    evaluations_used = 0;   // reset the total surrogate-evaluation budget counter

    initialize_network_differential();   // exact analytic VJP if possible, else finite differences

    return (objectives_number == 1)
        ? perform_single_objective_optimization()
        : perform_multiobjective_optimization();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
