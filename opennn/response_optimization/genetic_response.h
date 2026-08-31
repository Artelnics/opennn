//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G E N E T I C   R E S P O N S E   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/response_optimization/response_optimization.h"

namespace opennn
{

class GeneticResponse : public ResponseOptimization
{
public:

    explicit GeneticResponse(NeuralNetwork* = nullptr);

private:

    MatrixR single_optimization() override;
    MatrixR multi_optimization() override;

    pair<VectorR, VectorR> initialize_individual(const pair<VectorR, VectorR>&) const;

    pair<MatrixR, MatrixR> initialize_population(const pair<VectorR, VectorR>&) const;

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

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
