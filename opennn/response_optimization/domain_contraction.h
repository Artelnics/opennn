//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D O M A I N   C O N T R A C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/response_optimization/response_optimization.h"

namespace opennn
{

class DomainContraction : public ResponseOptimization
{
public:

    explicit DomainContraction(NeuralNetwork* = nullptr);

private:

    MatrixR single_optimization() override;
    MatrixR multi_optimization() override;

    pair<MatrixR, MatrixR> sample_local_domains(const vector<pair<VectorR, VectorR>>&) const;

    pair<VectorR, VectorR> contract_categories(pair<VectorR, VectorR>, const VectorR&, Index) const;

    float contraction_factor = 0.85f;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
