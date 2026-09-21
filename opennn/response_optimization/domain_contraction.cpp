// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/network/network.h"
#include "opennn/core/tensor_operations.h"

namespace opennn
{

namespace
{

pair<VectorR, VectorR> local_domain(const VectorR& center,
                                    const VectorR& half_interval,
                                    const pair<VectorR, VectorR>& initial_domain)
{
    return {(center - half_interval).cwiseMax(initial_domain.first),
            (center + half_interval).cwiseMin(initial_domain.second)};
}


VectorR initial_half_interval(const pair<VectorR, VectorR>& domain,
                              const vector<pair<Index, Index>>& blocks)
{
    VectorR half_interval = (domain.second - domain.first)/2.0f;

    for (const pair<Index, Index>& block : blocks)
        half_interval.segment(block.first, block.second).setConstant(numeric_limits<float>::infinity());

    return half_interval;
}


Index category_column(const VectorR& input, const pair<Index, Index>& block)
{
    Index category = 0;

    input.segment(block.first, block.second).maxCoeff(&category);

    return block.first + category;
}


vector<pair<Index, Index>> category_columns(const MatrixR& inputs,
                                            const vector<pair<Index, Index>>& blocks)
{
    vector<pair<Index, Index>> columns;

    columns.reserve(size_t(inputs.rows())*blocks.size());

    for (Index i = 0; i < inputs.rows(); i++)
        for (const pair<Index, Index>& block : blocks)
            columns.emplace_back(i, category_column(inputs.row(i).transpose(), block));

    return columns;
}


vector<pair<VectorR, VectorR>> local_domains_around(const MatrixR& centers,
                                                    const VectorR& half_interval,
                                                    const pair<VectorR, VectorR>& initial_domain)
{
    vector<pair<VectorR, VectorR>> domains(size_t(centers.rows()));

    for (Index i = 0; i < centers.rows(); i++)
        domains[size_t(i)] = local_domain(centers.row(i).transpose(), half_interval, initial_domain);

    return domains;
}

}


DomainContraction::DomainContraction(Network* new_network)
    : ResponseOptimization(new_network)
{
}


void DomainContraction::set_contraction_factor(const float new_contraction_factor)
{
    contraction_factor = clamp(new_contraction_factor, EPSILON, 1.0f);
}


pair<VectorR, VectorR> DomainContraction::contract_categories(pair<VectorR, VectorR> domain,
                                                              const VectorR& category_scores,
                                                              const Index iteration) const
{
    for (const pair<Index, Index>& block : get_categorical_blocks(network->get_input_variables()))
    {
        vector<Index> live_columns;

        for (Index j = 0; j < block.second; j++)
            if (domain.second(block.first + j) > 0.0f)
                live_columns.push_back(block.first + j);

        ranges::sort(live_columns, {},
                     [&category_scores](const Index column) { return category_scores(column); });

        const Index survivors_number =
            max(Index(1), Index(ceil(pow(contraction_factor, float(iteration + 1))*float(block.second))));

        for (Index i = 0; i < Index(live_columns.size()) - survivors_number; i++)
            domain.second(live_columns[size_t(i)]) = 0.0f;
    }

    return domain;
}


pair<MatrixR, MatrixR> DomainContraction::sample_local_domains(
    const vector<pair<VectorR, VectorR>>& local_domains)
{
    const Index sample_size = max(Index(1), points_number/Index(local_domains.size()));

    pair<MatrixR, MatrixR> points;

    Index starved_domains = 0;

    for (const pair<VectorR, VectorR>& domain : local_domains)
    {
        feasibility_system.borders = domain;

        Index sampled = 0;

        for (Index attempt_feasibility = 0;
             attempt_feasibility < iterations_number && sampled < sample_size;
             attempt_feasibility++)
        {
            const Index batch = sample_size - sampled;

            MatrixR inputs(batch, domain.first.size());
            MatrixR outputs(batch, network->get_outputs_number());

            Index feasible_number = 0;

            for (Index i = 0; i < batch; i++)
            {
                const auto [input, output] = feasibility_system.solve(calculate_random_input(domain));

                if (input.size() == 0) continue;

                inputs.row(feasible_number) = input.transpose();
                outputs.row(feasible_number) = output.transpose();

                feasible_number++;
            }

            sampled += feasible_number;

            points = append_rows(points, {inputs.topRows(feasible_number),
                                          outputs.topRows(feasible_number)});
        }

        if (sampled < sample_size) starved_domains++;
    }

    throw_if(points.first.rows() == 0,
             "No feasible point could be drawn in " + to_string(iterations_number)
             + " attempts. The constraints may be impossible to satisfy.");

    if (starved_domains > 0)
        logging::warning() << "Warning: " << starved_domains << " of " << local_domains.size()
             << " local domains yielded fewer than " << sample_size << " feasible points.\n";

    return points;
}


MatrixR DomainContraction::single_optimization()
{
    const vector<pair<Index, Index>> blocks =
        get_categorical_blocks(network->get_input_variables());

    pair<VectorR, VectorR> allowed_domain = calculate_domain();

    VectorR half_interval = initial_half_interval(allowed_domain, blocks);

    pair<VectorR, VectorR> domain = allowed_domain;

    VectorR category_scores = VectorR::Constant(allowed_domain.first.size(), -MAX);

    VectorR best_input;
    VectorR best_output;

    float best_value = -MAX;

    bool finite_value_seen = false;

    for (Index iteration = 0; iteration < iterations_number; iteration++)
    {
        const auto [feasible_inputs, feasible_outputs] = sample_local_domains({domain});

        const VectorR values = evaluate_objectives(feasible_inputs, feasible_outputs).col(0);

        finite_value_seen = finite_value_seen || values.array().isFinite().any();

        for (const auto [row, column] : category_columns(feasible_inputs, blocks))
            category_scores(column) = max(category_scores(column), values(row));

        Index best_row = 0;

        if (values.maxCoeff(&best_row) > best_value)
        {
            best_value = values(best_row);

            best_input = feasible_inputs.row(best_row).transpose();
            best_output = feasible_outputs.row(best_row).transpose();
        }

        if (best_input.size() == 0) continue;

        half_interval *= contraction_factor;

        allowed_domain = contract_categories(allowed_domain, category_scores, iteration);

        domain = local_domain(best_input, half_interval, allowed_domain);
    }

    // Feasible points were drawn (sample_local_domains throws otherwise), so an
    // empty result means the objective itself never produced a usable value.
    throw_if(best_input.size() == 0 && !finite_value_seen,
             "Objective '" + objectives.front().expression.text + "' has no finite value at any feasible "
             "point. Check the expression for divisions by zero, logarithms or square roots of negative "
             "values, and exponents that overflow.");

    throw_if(best_input.size() == 0, "No feasible point was found.");

    return append_columns(best_input.transpose(), best_output.transpose());
}


MatrixR DomainContraction::multi_optimization()
{
    const vector<pair<Index, Index>> blocks =
        get_categorical_blocks(network->get_input_variables());

    pair<VectorR, VectorR> allowed_domain = calculate_domain();

    const VectorR initial_superior = allowed_domain.second;

    VectorR half_interval = initial_half_interval(allowed_domain, blocks);

    vector<pair<VectorR, VectorR>> local_domains(1, allowed_domain);

    pair<MatrixR, MatrixR> candidates;

    for (Index iteration = 0; iteration < iterations_number; iteration++)
    {
        candidates = append_rows(candidates, sample_local_domains(local_domains));

        candidates = slice_rows(candidates,
                                clean_front(candidates.first, candidates.second));

        VectorR category_scores = VectorR::Zero(allowed_domain.first.size());

        for (const auto [row, column] : category_columns(candidates.first, blocks))
            category_scores(column) += 1.0f;

        half_interval *= contraction_factor;

        allowed_domain = contract_categories(allowed_domain, category_scores, iteration);

        local_domains = local_domains_around(candidates.first, half_interval, allowed_domain);
    }

    vector<Index> front = clean_front(candidates.first, candidates.second);

    for (Index attempt_front = 0;
         attempt_front < iterations_number && Index(front.size()) < requested_front_size;
         attempt_front++)
    {
        candidates = slice_rows(candidates, front);

        half_interval /= contraction_factor;

        allowed_domain.second = initial_superior;

        local_domains = local_domains_around(candidates.first, half_interval, allowed_domain);

        candidates = append_rows(candidates, sample_local_domains(local_domains));

        front = clean_front(candidates.first, candidates.second);
    }

    if (Index(front.size()) < requested_front_size)
        logging::warning() << "Warning: the front holds " << front.size() << " of the " << requested_front_size
             << " points requested. The feasible set may be too small to spread them over.\n";

    return append_columns(slice_rows(candidates, front));
}

}
