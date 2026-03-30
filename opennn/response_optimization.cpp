//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "response_optimization.h"

#include "pch.h"
#include "tensor_utilities.h"
#include "statistics.h"
#include "neural_network.h"
#include "variable.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"

namespace opennn
{

ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network)
{
    set(new_neural_network);
}


void ResponseOptimization::set(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;

    if(!neural_network)
        return;
}


void ResponseOptimization::set_condition(const string& name, const ConditionType condition, type low, type up)
{
   conditions[name] = Condition(condition, low, up);
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
    auto it = conditions.find(name);

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


void ResponseOptimization::set_zoom_factor(type new_zoom_factor)
{
    zoom_factor = new_zoom_factor;
}


void ResponseOptimization::set_relative_tolerance(type new_relative_tolerance)
{
    relative_tolerance = new_relative_tolerance;
}


void ResponseOptimization::set_deformation_domain_factor(type new_deformation_domain_factor)
{
    deformation_domain_factor = new_deformation_domain_factor;
}


type ResponseOptimization::get_deformation_domain_factor()
{
    return deformation_domain_factor;
}


vector<Descriptives> ResponseOptimization::get_descriptives(const string& role) const
{
    if (role == "Input")
    {
        if (neural_network->has("Scaling2d"))
            return static_cast<Scaling<2>*>(neural_network->get_first("Scaling2d"))->get_descriptives();
        if (neural_network->has("Scaling3d"))
            return static_cast<Scaling<3>*>(neural_network->get_first("Scaling3d"))->get_descriptives();
        if (neural_network->has("Scaling4d"))
            return static_cast<Scaling<4>*>(neural_network->get_first("Scaling4d"))->get_descriptives();
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

    if (variables_uncheked.size() != descriptives_uncheked.size())
        throw runtime_error("ResponseOptimization: Variable count and Descriptives count mismatch.");

    vector<Variable> filtered_vars;
    vector<Descriptives> filtered_desc;

    for (size_t i = 0; i < variables_uncheked.size(); ++i)
    {
        const string& var_role = variables_uncheked[i].role;

        const Condition current_cond = get_condition(variables_uncheked[i].name);

        if (current_cond.condition == ConditionType::Past)
            continue; // Skip this variable entirely for optimization purposes

        if (is_input_request)
        {
            if (var_role == "Input")
            {
                filtered_vars.push_back(variables_uncheked[i]);
                filtered_desc.push_back(descriptives_uncheked[i]);
            }
        }
        else
            if (var_role == "Target" || var_role == "InputTarget")
            {
                filtered_vars.push_back(variables_uncheked[i]);
                filtered_desc.push_back(descriptives_uncheked[i]);
            }

    }

    return {filtered_vars, filtered_desc};
}

void ResponseOptimization::Domain::set(const vector<Variable>& variables, const vector<Descriptives>& descriptives, const type deformation_domain_factor)
{
    const Index variables_number = static_cast<Index>(variables.size());

    const vector<Index> feature_dimensions = get_feature_dimensions(variables);

    const Index total_feature_dimensions = accumulate(feature_dimensions.begin(), feature_dimensions.end(), Index(0));

    inferior_frontier.resize(total_feature_dimensions);
    superior_frontier.resize(total_feature_dimensions);

    //allowed_values.resize(total_feature_dimensions, 1);
    //allowed_values.setConstant(numeric_limits<type>::quiet_NaN());

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
            const type original_minimum = static_cast<type>(descriptives[variable].minimum);
            const type original_maximum = static_cast<type>(descriptives[variable].maximum);

            const type center = (original_maximum + original_minimum) * 0.5f;
            const type half_range = (original_maximum - original_minimum) * 0.5f * deformation_domain_factor;

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

    if (descriptives.size() != variables_number)
        throw runtime_error("ResponseOptimization: Descriptives count (" + to_string(descriptives.size()) +
                            ") does not match variables count (" + to_string(variables_number) + ") for " + role);

    const vector<Index> feature_dimensions = get_feature_dimensions(variables);

    vector<Condition> applicable_conditions;
    applicable_conditions.reserve(variables_number);

    for (const auto& variable : variables)
        applicable_conditions.push_back(get_condition(variable.name));

    Domain original_domain(variables, descriptives, deformation_domain_factor);

    original_domain.bound(variables, applicable_conditions);

    return original_domain;
}


ResponseOptimization::Objectives::Objectives(const ResponseOptimization& response_optimization)
{
    const Index objectives_number = response_optimization.get_objectives_number();

    if (objectives_number == 0)
        throw runtime_error("No Objectives found, make sure to set Minimize or Maximize to any variable");

    objective_sources.resize(2, objectives_number);

    objective_normalizer.resize(2, objectives_number);

    utopian_and_senses.resize(2, objectives_number);

    Index current_objective_index = 0;

    auto process_role = [&](const string& role)
    {
        const bool is_input = (role == "Input");

        const auto [variables, descriptives] = response_optimization.get_variables_and_descriptives(role);

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

                objective_sources(1, current_objective_index) = static_cast<type>(feature_pointer);

                const type inferior_frontier = domain.inferior_frontier(feature_pointer);
                const type superior_frontier = domain.superior_frontier(feature_pointer);
                const type range = superior_frontier - inferior_frontier;

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
            type& inferior = inferior_frontier(feature_index);
            type& superior = superior_frontier(feature_index);

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


MatrixR ResponseOptimization::calculate_random_inputs(const Domain& input_domain) const
{
    const auto [variables, descriptives] = get_variables_and_descriptives("Input");

    const vector<Index> input_feature_dimensions = get_feature_dimensions(variables);

    const Index inputs_features_number = get_features_number(variables);

    MatrixR random_inputs(evaluations_number, inputs_features_number);
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
                const type inf = input_domain.inferior_frontier(current_feature_index);
                const type sup = input_domain.superior_frontier(current_feature_index);
                const type range = sup - inf;

                random_inputs.col(current_feature_index).array() = random_inputs.col(current_feature_index).array() * range + inf;
            }
            current_feature_index++;
        }
        else
        {
            random_inputs.block(0, current_feature_index, evaluations_number, categories_number).setZero();

            vector<Index> allowed_categories;

            for(Index i = 0; i < categories_number; ++i)
                if(input_domain.superior_frontier(current_feature_index + i) > 0.5)
                    allowed_categories.push_back(i);

            for(Index i = 0; i < evaluations_number; ++i)
                random_inputs(i, current_feature_index + allowed_categories[random_integer(0, allowed_categories.size()-1)]) = 1.0;

            current_feature_index += categories_number;
        }
    }

    return random_inputs;
}


// @todo bad name constructor
Tensor3 ResponseOptimization::input_constructor(const MatrixR& controllable_candidates) const
{
    // Get all 8 input features

    const vector<Variable> all_input_vars = neural_network->get_input_variables();
    const Index batch_size = controllable_candidates.rows(); // 1000
    const Shape input_shape = neural_network->get_input_shape();
    const Index total_lags = input_shape[0]; // 2
    const Index total_features = input_shape[1]; // 8

    // Create the target tensor [1000, 2, 8]
    Tensor3 constructed_input(batch_size, total_lags, total_features);

    // Step 1: Copy the "Perfect Rectangle" (History) to all 1000 samples
    // This fills [1000, 2, 8] with the state and previous settings
    constructed_input.device(get_device()) = fixed_history.broadcast(array_3(batch_size, 1, 1));

    // Step 2: Paste the 5 "Line on Top" candidate values
    // into the last lag (current time step)
    Index feature_cursor = 0;      // Moves 0 to 7 (8 total)
    Index candidate_col_cursor = 0; // Moves 0 to 4 (5 total)

    for (const auto& variable : all_input_vars)
    {
        const Index dim = variable.is_categorical() ? variable.get_categories_number() : 1;

        const Condition current_cond = get_condition(variable.name);

        if (variable.role == "Input" && current_cond.condition != ConditionType::Past)
        {
            // Only overwrite the last time step (total_lags - 1)
            for (Index i = 0; i < batch_size; ++i)
            {
                for(Index d = 0; d < dim; ++d)
                {
                    constructed_input(i, total_lags - 1, feature_cursor + d) =
                        controllable_candidates(i, candidate_col_cursor + d);
                }
            }
            candidate_col_cursor += dim;
        }
        // If role is "InputTarget", we do nothing.
        // The value from 'fixed_history' remains there.

        feature_cursor += dim;
    }

    return constructed_input;
}

// @todo change name optimized_variables
MatrixR ResponseOptimization::calculate_outputs(const MatrixR& optimized_variables) const
{
    if (is_forecasting)
    {
        if(fixed_history.size() == 0)
            throw runtime_error("ResponseOptimization: Model is forecasting but fixed_history is empty. Call set_fixed_history() first.");

        const Tensor3 formatted_input = input_constructor(optimized_variables);

        return neural_network->calculate_outputs(formatted_input);
    }

    return neural_network->calculate_outputs(optimized_variables);
}


void ResponseOptimization::Domain::reshape(const type zoom_factor,
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
            const type half_span = (superior_frontier(current_feature_index) - inferior_frontier(current_feature_index)) * zoom_factor / 2;
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
/*
pair<MatrixR, MatrixR> ResponseOptimization::filter_feasible_points(const MatrixR& inputs, const MatrixR& outputs, const Domain& output_domain) const
{
    const vector<Index> feasible_rows = build_feasible_rows_mask(outputs, output_domain.inferior_frontier, output_domain.superior_frontier);

    if(feasible_rows.empty())
        return {};

    MatrixR feasible_inputs((Index)feasible_rows.size(), inputs.cols());
    MatrixR feasible_outputs((Index)feasible_rows.size(), outputs.cols());

    for(Index j = 0; j < (Index)feasible_rows.size(); ++j)
    {
        feasible_inputs.row(j) = inputs.row(feasible_rows[j]);
        feasible_outputs.row(j) = outputs.row(feasible_rows[j]);
    }

    return {feasible_inputs, feasible_outputs};
}

*/

pair<MatrixR, MatrixR> ResponseOptimization::filter_feasible_points(const MatrixR& inputs, const MatrixR& outputs, const Domain& output_domain) const
{
    const vector<Variable>& target_vars = neural_network->get_output_variables();
    const Index rows_count = outputs.rows();

    // 1. Start with all row indices as "potentially feasible"
    vector<Index> feasible_indices(rows_count);
    iota(feasible_indices.begin(), feasible_indices.end(), 0);

    // 2. Loop through columns and apply filters ONLY for Hard Constraints
    for (size_t j = 0; j < target_vars.size(); ++j)
    {
        const Condition current_condition = get_condition(target_vars[j].name);

        if (!(current_condition.condition == ConditionType::Maximize ||
              current_condition.condition == ConditionType::Minimize ||
              current_condition.condition == ConditionType::None))
        {
            feasible_indices = filter_selected_indices_by_column(outputs,
                                                        feasible_indices,
                                                        static_cast<Index>(j),
                                                        output_domain.inferior_frontier(j),
                                                        output_domain.superior_frontier(j));
        }

        if (feasible_indices.empty())
            break;
    }

    if (feasible_indices.empty())
        return {MatrixR(), MatrixR()};

    MatrixR feasible_inputs((Index)feasible_indices.size(), inputs.cols());
    MatrixR feasible_outputs((Index)feasible_indices.size(), outputs.cols());

    for (Index i = 0; i < (Index)feasible_indices.size(); ++i)
    {
        feasible_inputs.row(i) = inputs.row(feasible_indices[i]);
        feasible_outputs.row(i) = outputs.row(feasible_indices[i]);
    }

    return {feasible_inputs, feasible_outputs};
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


pair<MatrixR, MatrixR> ResponseOptimization::calculate_optimal_points(const MatrixR& feasible_inputs,
                                                                      const MatrixR& feasible_outputs,
                                                                      const Objectives& objectives) const
{
    const Index subset_dimension = clamp<Index>(llround(zoom_factor * evaluations_number), 1, feasible_outputs.rows());

    MatrixR objective_matrix = objectives.extract(feasible_inputs, feasible_outputs);

    objectives.normalize(objective_matrix);

    const VectorR normalized_utopian_point = (objectives.utopian_and_senses.row(1).array() + (type)1.0) / (type)2.0;

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

// @todo to tensor
MatrixR ResponseOptimization::assemble_results(const MatrixR& inputs, const MatrixR& outputs) const
{
    MatrixR result(inputs.rows(), inputs.cols() + outputs.cols());
    result.leftCols(inputs.cols()) = inputs;
    result.rightCols(outputs.cols()) = outputs;
    return result;
}


MatrixR ResponseOptimization::perform_single_objective_optimization() const
{
    const Objectives objectives(*this);

    const auto [input_variables, descriptives] = get_variables_and_descriptives("Input");

    // @todo too many domains?
    const Domain original_input_domain = get_original_domain("Input");
    const Domain original_output_domain = get_original_domain("Target");
    Domain input_domain = original_input_domain;

    pair<MatrixR, MatrixR> optimal_set;

    type optimal_point;

    type previous_optimal_point = 0;

    cout << "> Optimization loop starting with zoom factor: " << zoom_factor << endl;

    for (Index i = 0; i < max_iterations; i++)
    {
        const MatrixR random_inputs = calculate_random_inputs(input_domain);

        const MatrixR outputs = calculate_outputs(random_inputs);

        if (outputs.hasNaN())
            cout << "Model produced NaN" << endl;

        auto [feasible_inputs, feasible_outputs] = filter_feasible_points(random_inputs, outputs, original_output_domain);

        if (feasible_inputs.rows() == 0)
            cout << "!!! [Critical] Zero feasible points found. "
                 << "Check if your constraints are too strict." << endl;

        optimal_set = calculate_optimal_points(feasible_inputs, feasible_outputs, objectives);

        optimal_point = (objectives.objective_sources(0, 0) > 0.5f
            ? optimal_set.first
            : optimal_set.second)(0, static_cast<Index>(objectives.objective_sources(1, 0)));

        const type relative_error = abs((optimal_point - previous_optimal_point) / (objectives.utopian_and_senses(0,0) + 1e-6f));

        cout <<  i << "-th " << "> loop " << "with relative error" << relative_error << endl;

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
        : assemble_results(optimal_set.first, optimal_set.second);
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


pair<type, type> ResponseOptimization::calculate_quality_metrics(const MatrixR& inputs,
                                                                 const MatrixR& outputs,
                                                                 const Objectives& objectives) const
{
    const Index points_number = inputs.rows();

    if (points_number == 0)
        return {static_cast<type>(1e6), static_cast<type>(1e6)};

    MatrixR objective_matrix = objectives.extract(inputs, outputs);

    objectives.normalize(objective_matrix);

    const Index objectives_number = objective_matrix.cols();

    const type hypercube_diagonal = sqrt(static_cast<type>(objectives_number));

    type maximum_internal_gap = 0.0;

    for (Index i = 0; i < points_number; ++i)
    {
        const auto current_point = objective_matrix.row(i);

        VectorR  distances = (objective_matrix.rowwise() - current_point).rowwise().squaredNorm();

        distances(i) = MAX;

        const type minimum_neighbor_distance = sqrt(distances.minCoeff());

        maximum_internal_gap = max(maximum_internal_gap, minimum_neighbor_distance);
    }

    if (points_number == 1)
        maximum_internal_gap = 1.0;

    maximum_internal_gap /= hypercube_diagonal;

    const VectorR max_objectives = objective_matrix.colwise().maxCoeff();

    const type sum_boundary_gaps = (1.0 - max_objectives.array()).abs().sum();

    const type average_boundary_gap = sum_boundary_gaps / static_cast<type>(objectives_number);
    const type normalized_boundary_gap = average_boundary_gap / hypercube_diagonal;

    return {maximum_internal_gap, normalized_boundary_gap};
}


MatrixR ResponseOptimization::perform_multiobjective_optimization() const
{
    const Objectives objectives(*this);

    const auto [input_variables, input_descriptives] = get_variables_and_descriptives("Input");

    const auto [output_variables, output_descriptives] = get_variables_and_descriptives("Target");

    const Domain original_input_domain = get_original_domain("Input");
    const Domain original_output_domain = get_original_domain("Target");

    const MatrixR random_inputs = calculate_random_inputs(original_input_domain);

    const MatrixR outputs = calculate_outputs(random_inputs);

    auto [first_feasible_inputs, first_feasible_outputs] = filter_feasible_points(random_inputs, outputs, original_output_domain);

    if (first_feasible_inputs.rows() == 0)
    {
        cout << "!!! [Critical] Zero feasible points found. "
             << "Check if your constraints are too strict." << endl;
        return MatrixR();
    }

    MatrixR first_objective_matrix  = objectives.extract(first_feasible_inputs, first_feasible_outputs);
    objectives.normalize(first_objective_matrix);

    auto [global_pareto_inputs, global_pareto_outputs] = calculate_pareto(first_feasible_inputs, first_feasible_outputs, first_objective_matrix);

    cout << "> Initial Pareto front size: " << global_pareto_inputs.rows() << " points." << endl;

    vector<Domain> input_domains(static_cast<size_t>(global_pareto_inputs.rows()), original_input_domain);

    type current_zoom = zoom_factor;

    type previous_holes_magnitude = 0.0;
    type previous_area_covered = 0.0;

    cout << "> Optimization loop starting with zoom factor: " << current_zoom << endl;

    for (Index i = 0; i < max_iterations; i++)
    {
        cout << "\n> [Iteration " << i + 1 << " / " << max_iterations << "]" << endl;

        MatrixR candidate_inputs = global_pareto_inputs;
        MatrixR candidate_outputs = global_pareto_outputs;

        for (Index j = 0; j < global_pareto_inputs.rows(); j++)
        {
            const MatrixR local_random_inputs = calculate_random_inputs(input_domains[j]);

            const MatrixR local_outputs = calculate_outputs(local_random_inputs);

            auto [local_feasible_inputs, local_feasible_outputs] = filter_feasible_points(local_random_inputs, local_outputs, original_output_domain);

            MatrixR local_objective_matrix  = objectives.extract(local_feasible_inputs, local_feasible_outputs);
            objectives.normalize(local_objective_matrix);

            auto [local_pareto_input, local_pareto_output] = calculate_pareto(local_feasible_inputs, local_feasible_outputs, local_objective_matrix );

            candidate_inputs = append_rows(candidate_inputs, local_pareto_input);
            candidate_outputs = append_rows(candidate_outputs, local_pareto_output);
        }

        cout << "  - Aggregated local Pareto candidates: " << candidate_inputs.rows() << endl;

        if (candidate_inputs.rows() == 0)
            break;

        auto optimal_set = calculate_optimal_points(candidate_inputs, candidate_outputs, objectives);

        MatrixR objective_matrix = objectives.extract(candidate_inputs, candidate_outputs);
        objectives.normalize(objective_matrix);

        const auto pareto_pair = calculate_pareto(candidate_inputs, candidate_outputs, objective_matrix);

        global_pareto_inputs = pareto_pair.first;
        global_pareto_outputs = pareto_pair.second;

        cout << "  - New Pareto front size: " << global_pareto_inputs.rows()  << endl;

        const pair<type, type> quality = calculate_quality_metrics(global_pareto_inputs, global_pareto_outputs, objectives);

        const type current_hole = quality.first;
        const type current_boundary = quality.second;

        cout << "  - Internal Hole: " << current_hole << " | Boundary Gap: " << current_boundary << endl;

        const type delta_hole = abs(current_hole - previous_holes_magnitude);
        const type delta_boundary = abs(current_boundary - previous_area_covered);

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

    return assemble_results(global_pareto_inputs, global_pareto_outputs);
}


MatrixR ResponseOptimization::perform_response_optimization() const
{
    const Index objectives_number = get_objectives_number();

    if (objectives_number == 0)
        throw runtime_error("No objectives found\n");

    return (objectives_number == 1)
        ? perform_single_objective_optimization()
        : perform_multiobjective_optimization();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute iteration and/or
// modify iteration under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that iteration will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
