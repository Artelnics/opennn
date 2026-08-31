//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D O M A I N   C O N T R A C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/neural_network/neural_network.h"
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

}


DomainContraction::DomainContraction(NeuralNetwork* new_neural_network)
    : ResponseOptimization(new_neural_network)
{
}


pair<VectorR, VectorR> DomainContraction::contract_categories(pair<VectorR, VectorR> domain,
                                                              const VectorR& category_scores,
                                                              const Index iteration) const
{
    for (const pair<Index, Index>& block : get_categorical_blocks(neural_network->get_input_variables()))
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
    const vector<pair<VectorR, VectorR>>& local_domains) const
{
    const Index sample_size = max(Index(1), points_number/Index(local_domains.size()));

    pair<MatrixR, MatrixR> points;

    Index starved_domains = 0;

    for (const pair<VectorR, VectorR>& domain : local_domains)
    {
        Index sampled = 0;

        for (Index attempt_feasibility = 0;
             attempt_feasibility < iterations_number && sampled < sample_size;
             attempt_feasibility++)
        {
            const Index batch = sample_size - sampled;

            MatrixR inputs(batch, domain.first.size());
            MatrixR outputs(batch, neural_network->get_outputs_number());

            Index feasible_number = 0;

            for (Index i = 0; i < batch; i++)
            {
                const auto [input, output] = get_feasible_point(calculate_random_input(domain), domain);

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
        cerr << "Warning: " << starved_domains << " of " << local_domains.size()
             << " local domains yielded fewer than " << sample_size << " feasible points.\n";

    return points;
}


MatrixR DomainContraction::single_optimization()
{
    const vector<pair<Index, Index>> blocks =
        get_categorical_blocks(neural_network->get_input_variables());

    pair<VectorR, VectorR> allowed_domain = calculate_domain();

    VectorR half_interval = initial_half_interval(allowed_domain, blocks);

    pair<VectorR, VectorR> domain = allowed_domain;

    VectorR category_scores = VectorR::Constant(allowed_domain.first.size(), -MAX);

    VectorR best_input;
    VectorR best_output;

    float best_value = -MAX;

    for (Index iteration = 0; iteration < iterations_number; iteration++)
    {
        const auto [feasible_inputs, feasible_outputs] = sample_local_domains({domain});

        const MatrixR objective_values = evaluate_objectives(feasible_inputs, feasible_outputs);

        for (Index i = 0; i < feasible_inputs.rows(); i++)
        {
            const VectorR input = feasible_inputs.row(i).transpose();
            const VectorR output = feasible_outputs.row(i).transpose();

            const float value = objective_values(i, 0);

            for (const pair<Index, Index>& block : blocks)
            {
                const Index column = category_column(input, block);

                category_scores(column) = max(category_scores(column), value);
            }

            if (value <= best_value) continue;

            best_value = value;

            best_input = input;
            best_output = output;
        }

        if (best_input.size() == 0) continue;

        half_interval *= contraction_factor;

        allowed_domain = contract_categories(allowed_domain, category_scores, iteration);

        domain = local_domain(best_input, half_interval, allowed_domain);
    }

    throw_if(best_input.size() == 0, "No feasible point was found.");

    return append_columns(best_input.transpose(), best_output.transpose());
}


MatrixR DomainContraction::multi_optimization()
{
    const vector<pair<Index, Index>> blocks =
        get_categorical_blocks(neural_network->get_input_variables());

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

        for (Index i = 0; i < candidates.first.rows(); i++)
        {
            const VectorR input = candidates.first.row(i).transpose();

            for (const pair<Index, Index>& block : blocks)
                category_scores(category_column(input, block)) += 1.0f;
        }

        half_interval *= contraction_factor;

        allowed_domain = contract_categories(allowed_domain, category_scores, iteration);

        local_domains.resize(size_t(candidates.first.rows()));

        for (Index i = 0; i < candidates.first.rows(); i++)
            local_domains[size_t(i)] =
                local_domain(candidates.first.row(i).transpose(), half_interval, allowed_domain);
    }

    vector<Index> front = clean_front(candidates.first, candidates.second);

    for (Index attempt_front = 0;
         attempt_front < iterations_number && Index(front.size()) < requested_front_size;
         attempt_front++)
    {
        candidates = slice_rows(candidates, front);

        half_interval /= contraction_factor;

        allowed_domain.second = initial_superior;

        local_domains.resize(size_t(candidates.first.rows()));

        for (Index i = 0; i < candidates.first.rows(); i++)
            local_domains[size_t(i)] =
                local_domain(candidates.first.row(i).transpose(), half_interval, allowed_domain);

        candidates = append_rows(candidates, sample_local_domains(local_domains));

        front = clean_front(candidates.first, candidates.second);
    }

    if (Index(front.size()) < requested_front_size)
        cerr << "Warning: the front holds " << front.size() << " of the " << requested_front_size
             << " points requested. The feasible set may be too small to spread them over.\n";

    return append_columns(slice_rows(candidates, front));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
