//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   R E S P O N S E   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/response_optimization/genetic_response.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/statistics.h"
#include "opennn/core/tensor_operations.h"

namespace opennn
{

namespace
{

vector<vector<Index>> non_dominated_sort(const MatrixR& objective_values)
{
    const Index individuals_number = objective_values.rows();

    vector<vector<Index>> dominated_individuals(individuals_number);
    vector<Index> domination_counts(individuals_number, 0);

    for (Index i = 0; i < individuals_number; i++)
        for (Index j = i + 1; j < individuals_number; j++)
            if (row_dominates(objective_values, i, j))
            {
                dominated_individuals[i].push_back(j);
                domination_counts[j]++;
            }
            else if (row_dominates(objective_values, j, i))
            {
                dominated_individuals[j].push_back(i);
                domination_counts[i]++;
            }

    vector<vector<Index>> fronts(1);

    for (Index i = 0; i < individuals_number; i++)
        if (domination_counts[i] == 0) fronts.front().push_back(i);

    while (!fronts.back().empty())
    {
        vector<Index> next_front;

        for (const Index i : fronts.back())
            for (const Index j : dominated_individuals[i])
                if (--domination_counts[j] == 0)
                    next_front.push_back(j);

        fronts.push_back(next_front);
    }

    fronts.pop_back();

    return fronts;
}


VectorR calculate_crowding_distances(const MatrixR& front_values)
{
    const Index points_number = front_values.rows();

    if (points_number < 3)
        return VectorR::Constant(points_number, MAX);

    return neighbor_distances(front_values, Index(sqrt(float(points_number))));
}

}


GeneticResponse::GeneticResponse(NeuralNetwork* new_neural_network)
    : ResponseOptimization(new_neural_network)
{
}


pair<MatrixR, MatrixR> GeneticResponse::initialize_population(const pair<VectorR, VectorR>& domain) const
{
    const Index attempts_number = iterations_number*points_number;

    MatrixR inputs(points_number, domain.first.size());
    MatrixR outputs(points_number, neural_network->get_outputs_number());

    Index feasible_number = 0;

    for (Index attempt = 0; attempt < attempts_number && feasible_number < points_number; attempt++)
    {
        const auto [input, output] = get_feasible_point(calculate_random_input(domain), domain);

        if (input.size() == 0) continue;

        inputs.row(feasible_number) = input.transpose();
        outputs.row(feasible_number) = output.transpose();

        feasible_number++;
    }

    throw_if(feasible_number < points_number,
             "Only " + to_string(feasible_number) + " of " + to_string(points_number)
             + " individuals could be made feasible in " + to_string(attempts_number)
             + " attempts. The constraints may be impossible to satisfy.");

    return {inputs, outputs};
}


MatrixR GeneticResponse::multi_optimization()
{
    const pair<VectorR, VectorR> domain = calculate_domain();

    pair<MatrixR, MatrixR> population = initialize_population(domain);

    for (Index generation = 0; generation < iterations_number; generation++)
    {
        const vector<Index> ranking = calculate_fitness(population.first, population.second);

        const pair<MatrixR, MatrixR> children = recombinate_population(population.first, ranking, domain);

        population = append_rows(population, mutate_population(children.first, domain));

        const vector<Index> survivors = calculate_fitness(population.first, population.second);

        population = slice_rows(population,
                                vector<Index>(survivors.begin(),
                                              survivors.begin() + min(points_number, Index(survivors.size()))));
    }

    vector<Index> front = clean_front(population.first, population.second);

    for (Index i = 0; i < iterations_number && Index(front.size()) < requested_front_size; i++)
    {
        const vector<Index> parents =
            calculate_pareto_front(evaluate_objectives(population.first, population.second));

        const pair<MatrixR, MatrixR> children = recombinate_population(population.first, parents, domain);

        const pair<MatrixR, MatrixR> offspring = mutate_population(children.first, domain);

        if (offspring.first.rows() == 0) break;

        population = append_rows(population, offspring);

        front = clean_front(population.first, population.second);
    }

    if (Index(front.size()) < requested_front_size)
        cerr << "Warning: the front holds " << front.size() << " of the " << requested_front_size
             << " points requested. The feasible set may be too small to spread them over.\n";

    return append_columns(slice_rows(population, front));
}


MatrixR GeneticResponse::single_optimization()
{
    const pair<VectorR, VectorR> domain = calculate_domain();

    pair<MatrixR, MatrixR> population = initialize_population(domain);

    for (Index generation = 0; generation < iterations_number; generation++)
    {
        const vector<Index> ranking = calculate_fitness(population.first, population.second);

        const pair<MatrixR, MatrixR> children = recombinate_population(population.first, ranking, domain);

        population = append_rows(population, mutate_population(children.first, domain));

        const vector<Index> survivors = calculate_fitness(population.first, population.second);

        population = slice_rows(population,
                                vector<Index>(survivors.begin(),
                                              survivors.begin() + min(points_number, Index(survivors.size()))));
    }

    const MatrixR objective_values = evaluate_objectives(population.first, population.second);

    Index best_point = 0;

    objective_values.col(0).maxCoeff(&best_point);

    return append_columns(population.first.row(best_point), population.second.row(best_point));
}


vector<Index> GeneticResponse::calculate_fitness(const MatrixR& inputs, const MatrixR& outputs) const
{
    const MatrixR objective_values = evaluate_objectives(inputs, outputs);

    if (objective_values.cols() == 1)
        return ranked_indices(objective_values.col(0));

    vector<Index> ranking;
    ranking.reserve(size_t(inputs.rows()));

    const vector<vector<Index>> fronts = non_dominated_sort(objective_values);

    for (const vector<Index>& front : fronts)
    {
        const MatrixR front_values = minmax_score(slice_rows(objective_values, front));

        VectorR crowding_distances = calculate_crowding_distances(front_values);

        for (const Index extreme : extreme_indices(front_values))
            crowding_distances(extreme) = MAX;

        const VectorI positions = maximal_indices(crowding_distances, Index(front.size()));

        for (Index i = 0; i < positions.size(); i++)
            ranking.push_back(front[size_t(positions(i))]);
    }

    return ranking;
}


pair<MatrixR, MatrixR> GeneticResponse::recombinate_population(const MatrixR& parent_inputs,
                                                               const vector<Index>& ranking,
                                                               const pair<VectorR, VectorR>& domain) const
{
    const auto select_parent = [&ranking]()
    {
        const Index individuals_number = Index(ranking.size());

        return ranking[size_t(min(random_integer(0, individuals_number - 1),
                                  random_integer(0, individuals_number - 1)))];
    };

    const Index attempts_number = iterations_number*points_number;

    MatrixR inputs(points_number, parent_inputs.cols());
    MatrixR outputs(points_number, neural_network->get_outputs_number());

    Index feasible_number = 0;

    for (Index i = 0; i < attempts_number && feasible_number < points_number; i += 2)
    {
        VectorR first_child = parent_inputs.row(select_parent()).transpose();
        VectorR second_child = parent_inputs.row(select_parent()).transpose();

        if (random_uniform(0.0f, 1.0f) < crossover_probability)
            crossover(first_child, second_child, domain);

        const auto [first_input, first_output] = get_feasible_point(first_child, domain);

        if (first_input.size() > 0)
        {
            inputs.row(feasible_number) = first_input.transpose();
            outputs.row(feasible_number) = first_output.transpose();

            feasible_number++;
        }

        if (feasible_number == points_number) break;

        const auto [second_input, second_output] = get_feasible_point(second_child, domain);

        if (second_input.size() > 0)
        {
            inputs.row(feasible_number) = second_input.transpose();
            outputs.row(feasible_number) = second_output.transpose();

            feasible_number++;
        }
    }

    throw_if(feasible_number < points_number,
             "Only " + to_string(feasible_number) + " of " + to_string(points_number)
             + " children could be recombined into feasible points in " + to_string(attempts_number)
             + " attempts. The constraints may be impossible to satisfy.");

    return {inputs, outputs};
}


pair<MatrixR, MatrixR> GeneticResponse::mutate_population(const MatrixR& offspring_inputs,
                                                          const pair<VectorR, VectorR>& domain) const
{
    if (offspring_inputs.rows() == 0) return {};

    const Index attempts_number = iterations_number*points_number;

    MatrixR inputs(points_number, offspring_inputs.cols());
    MatrixR outputs(points_number, neural_network->get_outputs_number());

    Index feasible_number = 0;

    for (Index attempt = 0; attempt < attempts_number && feasible_number < points_number; attempt++)
    {
        VectorR child = offspring_inputs.row(attempt % offspring_inputs.rows()).transpose();

        mutate_individual(child, domain);

        const auto [input, output] = get_feasible_point(child, domain);

        if (input.size() == 0) continue;

        inputs.row(feasible_number) = input.transpose();
        outputs.row(feasible_number) = output.transpose();

        feasible_number++;
    }

    throw_if(feasible_number < points_number,
             "Only " + to_string(feasible_number) + " of " + to_string(points_number)
             + " children survived mutation feasibly in " + to_string(attempts_number)
             + " attempts. The constraints may be impossible to satisfy.");

    return {inputs, outputs};
}


void GeneticResponse::crossover(VectorR& first_child,
                                VectorR& second_child,
                                const pair<VectorR, VectorR>& domain) const
{
    const float exponent = 1.0f/(crossover_distribution_index + 1.0f);

    for (Index j = 0; j < first_child.size(); j++)
    {
        if (random_bool()) continue;

        const float u = random_uniform(0.0f, 1.0f);

        const float spread = (u <= 0.5f) ? pow(2.0f*u, exponent)
                                         : pow(1.0f/(2.0f*(1.0f - u)), exponent);

        const float mean = 0.5f*(first_child(j) + second_child(j));
        const float half_difference = 0.5f*spread*(first_child(j) - second_child(j));

        first_child(j) = clamp(mean + half_difference, domain.first(j), domain.second(j));

        second_child(j) = clamp(mean - half_difference, domain.first(j), domain.second(j));
    }
}


void GeneticResponse::mutate_individual(VectorR& candidate, const pair<VectorR, VectorR>& domain) const
{
    const vector<pair<Index, Index>> categorical_blocks = get_categorical_blocks(neural_network->get_input_variables());

    vector<char> categorical_columns(size_t(candidate.size()), 0);

    Index variables_number = candidate.size();

    for (const auto& [first_column, categories_number] : categorical_blocks)
    {
        fill_n(categorical_columns.begin() + first_column, categories_number, 1);

        variables_number -= categories_number - 1;
    }

    const float probability = 1.0f/float(variables_number);

    for (Index j = 0; j < candidate.size(); j++)
    {
        if (categorical_columns[size_t(j)] || random_uniform(0.0f, 1.0f) >= probability) continue;

        const float range = domain.second(j) - domain.first(j);

        if (range <= 0.0f) continue;

        candidate(j) = clamp(candidate(j) + random_normal(0.0f, mutation_deviation)*range,
                             domain.first(j),
                             domain.second(j));
    }

    vector<char> closed_categories;
    vector<float> block;

    for (const auto& [first_column, categories_number] : categorical_blocks)
    {
        if (random_uniform(0.0f, 1.0f) >= probability) continue;

        closed_categories.resize(size_t(categories_number));

        for (Index j = 0; j < categories_number; j++)
            closed_categories[size_t(j)] = (domain.second(first_column + j) <= 0.0f) ? 1 : 0;

        if (!draw_k_hot(categories_number, 1, {}, closed_categories, block)) continue;

        for (Index j = 0; j < categories_number; j++)
            candidate(first_column + j) = block[size_t(j)];
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
