// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

#pragma once

#include "opennn/core/opennn_types.h"

namespace opennn::detail
{

// Internal CPU kernel. Matrices are symmetric n-by-n, vectors have length n,
// and output/work storage must not overlap the inputs or each other.
// Nocedal (1980), "Updating Quasi-Newton Matrices With Limited Storage",
// equations (1) and (3): H+ = (I-rho*s*yT) H (I-rho*y*sT) + rho*s*sT.
inline bool bfgs_inverse_update(const Eigen::Ref<const MatrixR>& previous,
                                const Eigen::Ref<const VectorR>& step,
                                const Eigen::Ref<const VectorR>& gradient_change,
                                Eigen::Ref<MatrixR> updated,
                                Eigen::Ref<VectorR> work)
{
    updated = previous;

    const float curvature = step.dot(gradient_change);
    if (!(curvature > EPSILON) || !std::isfinite(curvature)) return false;

    work.noalias() = previous * gradient_change;
    const float quadratic = gradient_change.dot(work);
    if (!(quadratic > EPSILON) || !std::isfinite(quadratic)) return false;

    const double rho = 1.0 / double(curvature);
    const double coefficient = 0.5 * rho * (1.0 + rho * double(quadratic));
    if (coefficient > double(std::numeric_limits<float>::max())) return false;

    // H+ = H + s*qT + q*sT, q = rho*((1+rho*yT*H*y)*s/2 - H*y).
    work = float(coefficient) * step - float(rho) * work;
    if (!work.allFinite()) return false;

    updated.selfadjointView<Eigen::Lower>().rankUpdate(step, work);
    updated.triangularView<Eigen::Upper>() = updated.triangularView<Eigen::Lower>().transpose();

    if (!updated.allFinite())
    {
        updated = previous;
        return false;
    }
    return true;
}

}
