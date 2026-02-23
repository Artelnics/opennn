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
    //enum class string{None, MinimumMaximum, MeanStandardDeviation, StandardDeviation, Logarithm, ImageMinMax};

    void scale_mean_standard_deviation(MatrixR&, Index, const Descriptives&);
    void scale_standard_deviation(MatrixR&, Index, const Descriptives&);
    void scale_minimum_maximum(MatrixR&, Index, const Descriptives&, type = type(-1), type = type(1));
    void scale_logarithmic(MatrixR&, Index);

    void scale_mean_standard_deviation(MatrixMap, Index, const Descriptives&);
    void scale_standard_deviation(MatrixMap, Index, const Descriptives&);
    void scale_minimum_maximum(MatrixMap, Index, const Descriptives&, type = type(-1), type = type(1));
    void scale_logarithmic(MatrixMap, Index);

    void scale_mean_standard_deviation_3d(Tensor3&, Index, const Descriptives&);
    void scale_standard_deviation_3d(Tensor3&, Index, const Descriptives&);
    void scale_minimum_maximum_3d(Tensor3&, Index, const Descriptives&, type, type);
    void scale_logarithmic_3d(Tensor3&, Index);

    void unscale_minimum_maximum(MatrixMap, Index, const Descriptives&, type = type(-1), type = type(1));
    void unscale_mean_standard_deviation(MatrixMap, Index, const Descriptives&);
    void unscale_standard_deviation(MatrixMap, Index, const Descriptives&);
    void unscale_logarithmic(MatrixMap, Index);
    void unscale_image_minimum_maximum(MatrixMap, Index);

    void unscale_minimum_maximum(MatrixR&, Index, const Descriptives&, type = type(-1), type = type(1));
    void unscale_mean_standard_deviation(MatrixR&, Index, const Descriptives&);
    void unscale_standard_deviation(MatrixR&, Index, const Descriptives&);
    void unscale_logarithmic(MatrixR&, Index);
    void unscale_image_minimum_maximum(MatrixR&, Index);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
