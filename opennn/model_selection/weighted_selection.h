// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#pragma once

#include "opennn/core/opennn_types.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace opennn::detail
{

// Independent exponential priorities implement successive weighted sampling
// without replacement. See Efraimidis, arXiv:1012.0256v2, sections 2 and 3.2.
// Existing selections (elites) are retained, even when they exceed the quota.
template<typename Uniform>
void fill_weighted_selection(const VectorR& weights, Index quota, VectorB& selected, Uniform&& uniform)
{
    if (selected.size() != weights.size() || quota < 0 || quota > weights.size())
        throw std::invalid_argument("Weighted selection: invalid mask or quota.");

    Index available = 0;
    for (Index i = 0; i < weights.size(); ++i)
    {
        if (!std::isfinite(weights(i)) || weights(i) < 0.0f)
            throw std::invalid_argument("Weighted selection: weights must be finite and nonnegative.");
        if (!selected(i) && weights(i) > 0.0f) ++available;
    }

    const Index needed = quota - selected.count();
    if (needed <= 0) return;
    if (needed > available)
        throw std::invalid_argument("Weighted selection: insufficient positive weights.");

    std::vector<std::pair<double, Index>> priorities;
    priorities.reserve(size_t(available));
    for (Index i = 0; i < weights.size(); ++i)
    {
        if (selected(i) || weights(i) == 0.0f) continue;
        const double draw = double(uniform());
        if (!std::isfinite(draw) || draw < 0.0 || draw > 1.0)
            throw std::invalid_argument("Weighted selection: uniform draw must be in [0,1].");
        const double interior = std::min(draw, std::nextafter(1.0, 0.0));
        priorities.emplace_back(-std::log1p(-interior) / double(weights(i)), i);
    }

    // Index order resolves the rare ties from finite-precision random draws.
    std::partial_sort(priorities.begin(), priorities.begin() + needed, priorities.end());
    for (Index rank = 0; rank < needed; ++rank)
        selected(priorities[size_t(rank)].second) = true;
}

}
