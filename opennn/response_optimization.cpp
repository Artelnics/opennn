//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "tensors.h"
#include "statistics.h"
#include "dataset.h"
#include "neural_network.h"
#include "response_optimization.h"

namespace opennn
{

ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    set(new_neural_network, new_dataset);
}


void ResponseOptimization::set(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    neural_network = new_neural_network;
    dataset = new_dataset;

    if(!neural_network || !dataset)
        return;

    const Index variables_number = dataset->get_variables_number();

    conditions.assign(static_cast<size_t>(variables_number), Condition(ConditionType::None));
}

void ResponseOptimization::set_condition(const string& name, const ConditionType condition, type low, type up)
{
    if(!dataset)
        throw runtime_error("Dataset not set.");

    const Index index = dataset->get_variable_index(name);

    conditions[index] = Condition(condition, low, up);
}

void ResponseOptimization::clear_conditions()
{
    if(dataset)
        conditions.assign(static_cast<size_t>(dataset->get_variables_number()), Condition(ConditionType::None));
    else
        conditions.clear();
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


void ResponseOptimization::Domain::set(const vector<Index>& feature_dimensions, const vector<Descriptives>& descriptives)
{
    const Index variables_number = static_cast<Index>(feature_dimensions.size());

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
            inferior_frontier(feature_index) = static_cast<type>(descriptives[variable].minimum);
            superior_frontier(feature_index) = static_cast<type>(descriptives[variable].maximum);
        }

        feature_index += feature_dimension;
    }
}


ResponseOptimization::Domain ResponseOptimization::get_original_domain(const string role) const
{
    const vector<Index> feature_dimensions = dataset->get_feature_dimensions();

    const vector<Index> variable_indices = dataset->get_variable_indices(role);

    cout << "puo essere in gather?" << endl;

    const vector<Index> feature_by_role_dimensions = gather_by_index(feature_dimensions, variable_indices);

    const vector<Descriptives> feture_descriptives = dataset->calculate_feature_descriptives(role);

    const vector<Condition> conditions_by_role = gather_by_index(conditions, variable_indices);

    cout << "o in original domain?" << endl;

    Domain original_domain(feature_by_role_dimensions, feture_descriptives);

    cout << "o in bound?" << endl;

    original_domain.bound(feature_by_role_dimensions, conditions_by_role);

    return original_domain;
}


ResponseOptimization::Objectives::Objectives(const ResponseOptimization& response_optimization)
{
    const vector<Index> feature_dimensions = response_optimization.dataset->get_feature_dimensions();

    Index objectives_number = 0;

    for (const auto& constraints : response_optimization.conditions)
        if (constraints.condition == ConditionType::Maximize || constraints.condition == ConditionType::Minimize)
            objectives_number++;

    if (objectives_number == 0)
        throw runtime_error("No Objectives found, make sure to set Minimize or Maximize to any variable");

    cout << "DEBUG: Number of objectives: " << objectives_number << endl;

    objective_sources.resize(2,objectives_number);

    objective_normalizer.resize(2, objectives_number);

    utopian_and_senses.resize(2, objectives_number);

    Index current_objective_index = 0;

    auto process_role = [&](const string& role)
    {
        cout << "DEBUG: Processing role: " << role << endl;

        const vector<Index> variable_indices = response_optimization.dataset->get_variable_indices(role);
        const vector<Index> feature_dimensions_by_role = gather_by_index(feature_dimensions, variable_indices);

        cout << "DEBUG: Getting domain for " << role << endl;

        const Domain domain = response_optimization.get_original_domain(role);

        cout << "DEBUG: Got domain for " << role << endl;

        const bool is_input = (role == "Input");

        Index feature_pointer = 0;

        for (Index i = 0; i < static_cast<Index>(variable_indices.size()); ++i)
        {
            const Condition& current_condition = response_optimization.conditions[variable_indices[i]];

            if (current_condition.condition == ConditionType::Maximize || current_condition.condition == ConditionType::Minimize)
            {
                objective_sources(0, current_objective_index) = is_input ? 1.0 : 0.0;

                objective_sources(1, current_objective_index) = static_cast<type>(feature_pointer);

                const type inferior_frontier = domain.inferior_frontier(feature_pointer);
                const type superior_frontier = domain.superior_frontier(feature_pointer);
                const type range = superior_frontier - inferior_frontier;

                const type epsilon = 1e-9;

                objective_normalizer(0, current_objective_index) = 1.0 / (range < epsilon ? epsilon : range);

                objective_normalizer(1, current_objective_index) = -inferior_frontier / (range < epsilon ? epsilon : range);

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

ResponseOptimization::Objectives ResponseOptimization::build_objectives() const
{
    return  Objectives(*this);
}

void ResponseOptimization::Domain::bound(const vector<Index>& feature_dimensions, const vector<Condition>& conditions)
{
    Index feature_index = 0;

    for(size_t variable_index = 0; variable_index < feature_dimensions.size(); ++variable_index)
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
    const Index inputs_features_number = input_domain.inferior_frontier.size();

    const vector<Index> input_indices = dataset->get_variable_indices("Input");
    const vector<Dataset::VariableType> input_variable_types = dataset->get_variable_types(input_indices);

    const vector<Index> feature_dimensions = dataset->get_feature_dimensions();
    const vector<Index> input_feature_dimensions = gather_by_index(feature_dimensions, input_indices);

    MatrixR random_inputs(evaluations_number, inputs_features_number);
    set_random_uniform(random_inputs, 0, 1);

    Index current_feature_index = 0;

    for(size_t input_variable = 0; input_variable < input_feature_dimensions.size(); ++input_variable)
    {
        const Index categories_number = input_feature_dimensions[input_variable];

        if(categories_number == 1)
        {
            if (input_variable_types[input_variable] != Dataset::VariableType::Binary)
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

            for(Index row = 0; row < evaluations_number; ++row)
                random_inputs(row, current_feature_index + allowed_categories[random_integer(0, allowed_categories.size()-1)]) = 1.0;

            current_feature_index += categories_number;
        }
    }

    return random_inputs;
}

void ResponseOptimization::Domain::reshape(const type zoom_factor,
                                           const VectorR& center,
                                           const MatrixR& optimal_points_inputs,
                                           const vector<Index>& input_feature_dimensions,
                                           const vector<Dataset::VariableType>& input_variable_types)
{
    VectorR categories_to_save = optimal_points_inputs.colwise().maxCoeff();

    for(Index i = 0; i < categories_to_save.size(); ++i)
        if(center(i) > categories_to_save(i))
            categories_to_save(i) = center(i);

    Index current_feature_index = 0;

    for(size_t input_variable = 0; input_variable < input_feature_dimensions.size(); ++input_variable)
    {
        const Index categories_number = input_feature_dimensions[input_variable];

        if(categories_number == 1 && input_variable_types[input_variable] != Dataset::VariableType::Binary)
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

pair<MatrixR, MatrixR> ResponseOptimization::filter_feasible_points(const MatrixR& inputs, const MatrixR& outputs, const Domain& output_domain) const
{
    const vector<Index> feasible_rows = build_feasible_rows_mask(outputs, output_domain.inferior_frontier, output_domain.superior_frontier);

    if(feasible_rows.empty())
        return {};

    MatrixR feasible_inputs((Index)feasible_rows.size(), inputs.cols());
    MatrixR feasible_outputs((Index)feasible_rows.size(), outputs.cols());

    for(Index j = 0; j < (Index)feasible_rows.size(); ++j)
    {
        set_row(feasible_inputs, inputs.row(feasible_rows[j]), j);
        set_row(feasible_outputs, outputs.row(feasible_rows[j]), j);
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
    const auto combined_scale = (objective_normalizer.row(0).array() * utopian_and_senses.row(1).array());
    const auto combined_offset = (objective_normalizer.row(1).array() * utopian_and_senses.row(1).array());

    objective_matrix.array().rowwise() *= combined_scale;
    objective_matrix.array().rowwise() += combined_offset;
}


pair<MatrixR, MatrixR> ResponseOptimization::calculate_optimal_points(const MatrixR& feasible_inputs, const MatrixR& feasible_outputs, const Objectives& objectives) const
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


MatrixR ResponseOptimization::assemble_results(const MatrixR& inputs, const MatrixR& outputs) const
{
    const vector<Index> feature_dimensions = dataset->get_feature_dimensions();

    const Index total_variables_number = (Index)feature_dimensions.size();

    vector<Index> global_starts_blocks(total_variables_number, 0);

    for (Index i = 1; i < total_variables_number; ++i)
        global_starts_blocks[i] = global_starts_blocks[i - 1] + feature_dimensions[i - 1];

    MatrixR result(inputs.rows(), global_starts_blocks.back() + feature_dimensions.back());

    auto copy_blocks = [&](const vector<Index>& indices_in_out, const vector<Index>& hot_encoded_dimensions, const MatrixR& source_to_copy)
    {
        Index start_source_feature_columns = 0;

        for (size_t i = 0; i < indices_in_out.size(); ++i)
        {
            result.block(0, global_starts_blocks[indices_in_out[i]], inputs.rows(), hot_encoded_dimensions[i])
                = source_to_copy.block(0, start_source_feature_columns, inputs.rows(), hot_encoded_dimensions[i]);
                  start_source_feature_columns += hot_encoded_dimensions[i];
        }
    };

    const vector<Index> output_indices = dataset->get_variable_indices("Target");
    const vector<Index> output_feature_dimensions = gather_by_index(feature_dimensions, output_indices);

    const vector<Index> input_indices = dataset->get_variable_indices("Input");
    const vector<Index> input_feature_dimensions = gather_by_index(feature_dimensions, input_indices);

    copy_blocks(input_indices, input_feature_dimensions, inputs);
    copy_blocks(output_indices, output_feature_dimensions, outputs);

    return result;
}

MatrixR ResponseOptimization::perform_single_objective_optimization(const Objectives& objectives) const
{
    const vector<Index> input_indices = dataset->get_variable_indices("Input");

    const vector<Index> feature_dimensions = dataset->get_feature_dimensions();

    const vector<Index> input_feature_dimensions = gather_by_index(feature_dimensions, input_indices);

    const vector<Dataset::VariableType> input_variable_types = dataset->get_variable_types(input_indices);

    const Domain original_input_domain = get_original_domain("Input");

    const Domain original_output_domain = get_original_domain("Target");

    Domain input_domain_to_iterate = original_input_domain;

    pair<MatrixR, MatrixR> optimal_set;

    type optimal_point;

    type previous_optimal_point = 0;

    cout << "> Optimization loop starting with zoom factor: " << zoom_factor << endl;

    for (Index i = 0; i < max_iterations; i++)
    {
        const MatrixR random_inputs = calculate_random_inputs(input_domain_to_iterate);

        auto [feasible_inputs, feasible_outputs] = filter_feasible_points(random_inputs, neural_network->calculate_outputs(random_inputs), original_output_domain);

        if(feasible_inputs.rows() == 0)
            break;

        optimal_set = calculate_optimal_points(feasible_inputs, feasible_outputs, objectives);

        optimal_point = (objectives.objective_sources(0, 0) > 0.5f
                             ? optimal_set.first
                             : optimal_set.second)(0, static_cast<Index>(objectives.objective_sources(1, 0)));

        const type relative_error = abs((optimal_point - previous_optimal_point) / (objectives.utopian_and_senses(0,0) + 1e-6f));

        if (relative_error < relative_tolerance)
        {
            cout << "> Optimization loop stopped for reaching the relative tolerance desired: " << relative_tolerance << endl;
            break;
        }

        previous_optimal_point = optimal_point;

        input_domain_to_iterate.reshape(zoom_factor, optimal_set.first.row(0), optimal_set.first, input_feature_dimensions, input_variable_types);
    }

    return optimal_set.first.rows() == 0
        ? MatrixR()
        : assemble_results(optimal_set.first, optimal_set.second);
}


pair<MatrixR, MatrixR> ResponseOptimization::calculate_pareto(const MatrixR& inputs, const MatrixR& outputs, const MatrixR& objective_matrix) const
{
    const Index rows_number = objective_matrix.rows();

    if (rows_number == 0)
        return {};

    VectorB non_dominated(static_cast<size_t>(rows_number), true);

    #pragma omp parallel for

    for (Index i = 0; i < rows_number; ++i)
    {
        const auto row_i = objective_matrix.row(i);

        for (Index j = 0; j < rows_number; ++j)
        {
            if (i == j)
                continue;

            const auto row_j = objective_matrix.row(j);

            if ((row_j.array() >= row_i.array()).all())
            {
                non_dominated[i] = false;
                break;
            }
        }
    }

    vector<Index> non_dominated_indices;
    non_dominated_indices.reserve(rows_number);

    for (Index i = 0; i < rows_number; ++i)
        if (non_dominated[i])
            non_dominated_indices.push_back(i);

    MatrixR pareto_inputs((Index)non_dominated_indices.size(), inputs.cols());
    MatrixR pareto_outputs((Index)non_dominated_indices.size(), outputs.cols());

    for (Index i = 0; i < (Index)non_dominated_indices.size(); ++i)
    {       
        pareto_inputs.row(i) = inputs.row(non_dominated_indices[i]);
        pareto_outputs.row(i) = outputs.row(non_dominated_indices[i]);
    }

    return {pareto_inputs, pareto_outputs};
}


pair<type, type> ResponseOptimization::calculate_quality_metrics(const MatrixR& inputs, const MatrixR& outputs, const Objectives& objectives) const
{
    const Index points_number = inputs.rows();

    if (points_number == 0)
        return {static_cast<type>(1e6), static_cast<type>(1e6)};

    MatrixR objective_matrix = objectives.extract(inputs, outputs);
    objectives.normalize(objective_matrix);

    const Index objectives_number = objective_matrix.cols();

    const type hypercube_diagonal = sqrt(static_cast<type>(objectives_number));
    const type compromise_distance = hypercube_diagonal / 2.0;

    type maximum_internal_gap = 0.0;

    for (Index i = 0; i < points_number; ++i)
    {
        const auto current_point = objective_matrix.row(i);

        VectorR  distances = (objective_matrix.rowwise() - current_point).rowwise().squaredNorm();

        distances(i) = numeric_limits<type>::max();

        const type minimum_neighbor_distance = sqrt(distances.minCoeff());

        maximum_internal_gap = max(maximum_internal_gap, minimum_neighbor_distance);
    }

    if (points_number == 1)
        maximum_internal_gap = 1.0;

    maximum_internal_gap /= hypercube_diagonal;

    VectorR max_objectives = objective_matrix.colwise().maxCoeff();

    const type sum_boundary_gaps = (1.0 - max_objectives.array()).abs().sum();

    const type average_boundary_gap = sum_boundary_gaps / static_cast<type>(objectives_number);
    const type normalized_boundary_gap = average_boundary_gap / compromise_distance;

    return {maximum_internal_gap, normalized_boundary_gap};
}


MatrixR ResponseOptimization::perform_multiobjective_optimization(const Objectives& objectives) const
{
    const vector<Index> input_indices = dataset->get_variable_indices("Input");

    const vector<Index> feature_dimensions = dataset->get_feature_dimensions();
    const vector<Index> input_feature_dimensions = gather_by_index(feature_dimensions, input_indices);

    const vector<Dataset::VariableType> input_variable_types = dataset->get_variable_types(input_indices);

    const Domain original_input_domain = get_original_domain("Input");

    const Domain original_output_domain = get_original_domain("Target");

    const MatrixR first_random_inputs = calculate_random_inputs(original_input_domain);

    auto [first_feasible_inputs, first_feasible_outputs] = filter_feasible_points(first_random_inputs, neural_network->calculate_outputs(first_random_inputs), original_output_domain);

    if (first_feasible_inputs.rows() == 0)
    {
        cout << "!!! [Critical] Zero feasible points found. "
             << "Check if your constraints are too strict." << endl;
        return MatrixR();
    }

    auto [global_pareto_inputs, global_pareto_outputs] = calculate_pareto(first_feasible_inputs, first_feasible_outputs, objectives.extract(first_feasible_inputs, first_feasible_outputs));

    cout << "> Initial Pareto front size: " << global_pareto_inputs.rows() << " points." << endl;

    vector<Domain> local_input_domains(static_cast<size_t>(global_pareto_inputs.rows()), original_input_domain);

    type current_zoom = zoom_factor;

    type previous_holes_magnitude = 0.0;
    type previous_area_covered = 0.0;

    cout << "> Optimization loop starting with zoom factor: " << current_zoom << endl;

    for (Index i = 0; i < max_iterations; i++)
    {
        cout << "\n> [Iteration " << i + 1 << " / " << max_iterations << "]" << endl;

        MatrixR union_inputs;
        MatrixR union_outputs;

        for (Index j = 0; j < global_pareto_inputs.rows(); j++)
        {
            const MatrixR local_random_inputs = calculate_random_inputs(local_input_domains[j]);

            auto [local_feasible_inputs, local_feasible_outputs] = filter_feasible_points(local_random_inputs, neural_network->calculate_outputs(local_random_inputs), original_output_domain);
            auto [local_pareto_input, local_pareto_output] = calculate_pareto(local_feasible_inputs, local_feasible_outputs, objectives.extract(local_feasible_inputs, local_feasible_outputs));

            union_inputs = append_rows(union_inputs, local_pareto_input);
            union_outputs = append_rows(union_outputs, local_pareto_output);
        }

        cout << "  - Aggregated local Pareto candidates: " << union_inputs.rows() << endl;

        const MatrixR candidate_inputs = append_rows(global_pareto_inputs, union_inputs);
        const MatrixR candidate_outputs = append_rows(global_pareto_outputs, union_outputs);

        if (candidate_inputs.rows() == 0)
            break;

        auto optimal_set = calculate_optimal_points(candidate_inputs, candidate_outputs, objectives);

        auto pareto_pair = calculate_pareto(candidate_inputs, candidate_outputs, objectives.extract(candidate_inputs, candidate_outputs));

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

        local_input_domains.assign(static_cast<size_t>(global_pareto_inputs.rows()), original_input_domain);

        for (Index j = 0; j < global_pareto_inputs.rows(); j++)
            local_input_domains[j].reshape(current_zoom, global_pareto_inputs.row(j), optimal_set.first, input_feature_dimensions, input_variable_types);

        current_zoom *= zoom_factor;
    }
    cout << "\n> [Optimization Complete] Assembling final results..." << endl;

    return assemble_results(global_pareto_inputs, global_pareto_outputs);
}

MatrixR ResponseOptimization::perform_response_optimization() const
{
    if(!dataset)
        throw runtime_error("Dataset not set\n");

    cout << "DEBUG: Building objectives..." << endl;
    const Objectives objectives = build_objectives();
    cout << "DEBUG: Objectives built." << endl;

    if (objectives.objective_sources.cols() == 0)
        throw runtime_error("No objectives found\n");

    return  (objectives.objective_sources.cols() == 1)
               ? perform_single_objective_optimization(objectives)
               : perform_multiobjective_optimization(objectives);;
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
