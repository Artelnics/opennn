// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/response_optimization/response_optimization.h"

namespace opennn
{

class GeneticResponse : public ResponseOptimization
{
public:

    explicit GeneticResponse(Network* = nullptr);

private:

    MatrixR single_optimization() override;
    MatrixR multi_optimization() override;

    pair<MatrixR, MatrixR> initialize_population(const pair<VectorR, VectorR>&) const;
    pair<MatrixR, MatrixR> evolve_population(const pair<VectorR, VectorR>&) const;

    vector<Index> calculate_fitness(const MatrixR&, const MatrixR&) const;

    pair<MatrixR, MatrixR> recombinate_population(const MatrixR&,
                                                  const vector<Index>&,
                                                  const pair<VectorR, VectorR>&) const;

    pair<MatrixR, MatrixR> mutate_population(const MatrixR&, const pair<VectorR, VectorR>&) const;

    void crossover(VectorR&, VectorR&, const pair<VectorR, VectorR>&) const;

    void mutate_individual(VectorR&, const pair<VectorR, VectorR>&) const;

    float crossover_probability = 0.9f;
    float crossover_distribution_index = 20.0f;
    float mutation_deviation = 0.1f;
};

}
