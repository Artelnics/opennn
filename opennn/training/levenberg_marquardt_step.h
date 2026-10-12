// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/core/opennn_types.h"

#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace opennn::detail
{

// Solve (curvature + damping*I) step = -gradient, then ask the caller whether
// to accept it. See Madsen, Nielsen and Tingleff (2004), section 3.2, eq. (3.13).
// Inputs stay immutable across trials; only an accepted trial may be committed
// by the caller. OpenNN's multiplicative damping schedule is preserved.
template<typename Accept>
bool find_damped_step(const MatrixR& curvature, const VectorR& gradient,
                      float& damping, float factor, float lower, float upper,
                      Eigen::Ref<VectorR> step, Accept&& accept)
{
    if (!std::isfinite(factor) || factor <= 1.0f)
        throw std::invalid_argument("LevenbergMarquardt: damping factor must be finite and greater than one.");

    if (!curvature.allFinite() || !gradient.allFinite()) return false;

    MatrixR system(curvature.rows(), curvature.cols());
    for (;;)
    {
        system = curvature;
        system.diagonal().array() += damping;
        step = system.colPivHouseholderQr().solve(-gradient);

        if (step.allFinite() && accept(step))
        {
            damping = std::clamp(damping / factor, lower, upper);
            return true;
        }

        const float next = std::clamp(damping * factor, lower, upper);
        const bool exhausted = next >= upper || next <= damping;
        damping = next;
        if (exhausted) return false;
    }
}

}
