//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#pragma once

#include "statistics.h"

namespace opennn
{
    /// @brief Standardises a column of the matrix in place using its descriptives' mean and standard deviation.
    /// @param matrix Data matrix to modify in place.
    /// @param column Column index to scale.
    /// @param descriptives Pre-computed descriptives of the column.
    void scale_mean_standard_deviation(MatrixMap, Index, const Descriptives&);

    /// @brief Divides a column of the matrix by its standard deviation in place.
    void scale_standard_deviation(MatrixMap, Index, const Descriptives&);

    /// @brief Rescales a column to the [min_range, max_range] interval using its descriptives.
    void scale_minimum_maximum(MatrixMap, Index, const Descriptives&, float = -1.0f, float = 1.0f);

    /// @brief Applies an element-wise logarithm to the given column.
    void scale_logarithmic(MatrixMap, Index);

    /// @brief Inverse of scale_minimum_maximum(): reconstructs original values for the given column.
    void unscale_minimum_maximum(MatrixMap, Index, const Descriptives&, float = -1.0f, float = 1.0f);

    /// @brief Inverse of scale_mean_standard_deviation() for the given column.
    void unscale_mean_standard_deviation(MatrixMap, Index, const Descriptives&);

    /// @brief Inverse of scale_standard_deviation() for the given column.
    void unscale_standard_deviation(MatrixMap, Index, const Descriptives&);

    /// @brief Inverse of scale_logarithmic() for the given column.
    void unscale_logarithmic(MatrixMap, Index);

    /// @brief Maps a column back from [-1, 1] to the [0, 255] image-pixel range.
    void unscale_image_minimum_maximum(MatrixMap, Index);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
