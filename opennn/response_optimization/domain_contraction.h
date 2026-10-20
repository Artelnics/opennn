// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/response_optimization/response_optimization.h"

namespace opennn
{

class DomainContraction : public ResponseOptimization
{
public:

    explicit DomainContraction(Network* = nullptr);

    void set_contraction_factor(float);

private:

    MatrixR single_optimization() override;
    MatrixR multi_optimization() override;

    pair<MatrixR, MatrixR> sample_local_domains(const vector<pair<VectorR, VectorR>>&, bool require_points = true);

    void contract_categories(pair<VectorR, VectorR>&, const vector<pair<Index, Index>>&, const VectorR&, Index) const;

    float contraction_factor = 0.85f;
};

}
