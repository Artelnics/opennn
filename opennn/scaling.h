//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "statistics.h"
#include "variable.h"

namespace opennn
{
    // Shared ScalerMethod formulas; x is a float or an Eigen array expression.
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

    // Scalar core: degenerate denominators guard to zero; minmax maps to [min_range, max_range].
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

    // Affine (scale, offset) core with +EPSILON denominators instead of degenerate guards (image pipeline).
    inline pair<float, float> scaling_affine(ScalerMethod scaler,
                                             const Descriptives& descriptives,
                                             float min_range,
                                             float max_range)
    {
        switch (scaler)
        {
        case ScalerMethod::MinimumMaximum:
        {
            const float scale = (max_range - min_range)
                              / ((descriptives.maximum - descriptives.minimum) + EPSILON);
            return {scale, min_range - descriptives.minimum * scale};
        }
        case ScalerMethod::MeanStandardDeviation:
        {
            const float scale = 1.0f / (descriptives.standard_deviation + EPSILON);
            return {scale, -descriptives.mean * scale};
        }
        case ScalerMethod::StandardDeviation:
            return {1.0f / (descriptives.standard_deviation + EPSILON), 0.0f};
        case ScalerMethod::ImageMinMax:
            return {1.0f / 255.0f, 0.0f};
        case ScalerMethod::None:
        case ScalerMethod::Logarithm:
            return {1.0f, 0.0f};
        }

        throw runtime_error("ImageDataset: invalid scaler method.");
    }

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
