// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/response_optimization/response_optimization.h"

namespace opennn
{

class DomainContraction : public ResponseOptimization
{
public:

    explicit DomainContraction(Network* = nullptr);

private:

    MatrixR single_optimization() override;
    MatrixR multi_optimization() override;

    pair<MatrixR, MatrixR> sample_local_domains(const vector<pair<VectorR, VectorR>>&) const;

    pair<VectorR, VectorR> contract_categories(pair<VectorR, VectorR>, const VectorR&, Index) const;

    float contraction_factor = 0.85f;
};

}
