//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/statistics.h"
#include "opennn/core/variable.h"

namespace opennn
{
    template<typename X>
    auto scale_minimum_maximum_formula(const X& x, const Descriptives& d, float min_range, float max_range)
    {
        return (x - d.minimum) / (d.maximum - d.minimum) * (max_range - min_range) + min_range;
    }

    template<typename X>
    auto scale_mean_standard_deviation_formula(const X& x, const Descriptives& d)
    {
        return (x - d.mean) / d.standard_deviation;
    }

    // Inverses of the two above. Every caller that unscales used to write these
    // out by hand, so the forward and backward directions could drift apart.
    template<typename X>
    auto unscale_minimum_maximum_formula(const X& x, const Descriptives& d, float min_range, float max_range)
    {
        return (x - min_range) / (max_range - min_range) * (d.maximum - d.minimum) + d.minimum;
    }

    template<typename X>
    auto unscale_mean_standard_deviation_formula(const X& x, const Descriptives& d)
    {
        return d.mean + x * d.standard_deviation;
    }

    inline float scale_value(ScalerMethod method, const Descriptives& desc, float value,
                             float min_range = -1.0f, float max_range = 1.0f)
    {
        using enum ScalerMethod;
        switch (method)
        {
        case None:
        case ImageMinMax:
            return value;
        case MinimumMaximum:
            return desc.maximum - desc.minimum < EPSILON
                ? 0.0f
                : scale_minimum_maximum_formula(value, desc, min_range, max_range);
        case MeanStandardDeviation:
            return desc.standard_deviation > EPSILON ? scale_mean_standard_deviation_formula(value, desc) : 0.0f;
        case StandardDeviation:
            return desc.standard_deviation > EPSILON ? value / desc.standard_deviation : 0.0f;
        case Logarithm:
            return log(max(value, EPSILON));
        }

        return value;
    }

    inline pair<float, float> scaling_affine(ScalerMethod scaler,
                                             const Descriptives& descriptives,
                                             float min_range,
                                             float max_range)
    {
        // Guards must match scale_value above: a feature with no spread scales to
        // zero. Adding EPSILON to the denominator instead would turn a constant
        // image channel into a ~1.7e7 multiplier, and nothing downstream checks it.
        using enum ScalerMethod;
        switch (scaler)
        {
        case MinimumMaximum:
        {
            const float range = descriptives.maximum - descriptives.minimum;
            if (range < EPSILON) return {0.0f, 0.0f};

            const float scale = (max_range - min_range) / range;
            return {scale, min_range - descriptives.minimum * scale};
        }
        case MeanStandardDeviation:
        {
            if (descriptives.standard_deviation <= EPSILON) return {0.0f, 0.0f};

            const float scale = 1.0f / descriptives.standard_deviation;
            return {scale, -descriptives.mean * scale};
        }
        case StandardDeviation:
            if (descriptives.standard_deviation <= EPSILON) return {0.0f, 0.0f};
            return {1.0f / descriptives.standard_deviation, 0.0f};
        case ImageMinMax:
            return {1.0f / 255.0f, 0.0f};
        case None:
        case Logarithm:
            return {1.0f, 0.0f};
        }

        throw runtime_error("scaling_affine: invalid scaler method.");
    }

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
