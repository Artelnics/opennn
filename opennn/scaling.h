//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#ifndef SCALING_H
#define SCALING_H

// System includes

#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <vector>

// OpenNN includes

#include "config.h"
#include "statistics.h"

namespace opennn
{
/// Enumeration of the available methods for scaling and unscaling the data.

    enum class Scaler{NoScaling, MinimumMaximum, MeanStandardDeviation, StandardDeviation, Logarithm};

    void scale_mean_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void scale_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void scale_minimum_maximum(Tensor<type, 2>&, const Index&, const Descriptives&, const type& = type(-1), const type& = type(1));
    void scale_image_pixel_minimum_maximum(Tensor<type, 4>&, const Descriptives&, const type& = type(-1), const type& = type(1));

    Tensor<type, 1> scale_minimum_maximum(const Tensor<type, 1>&);
    Tensor<type, 2> scale_minimum_maximum(const Tensor<type, 2>&);

    void scale_logarithmic(Tensor<type, 2>&, const Index&);
    void scale_minimum_maximum_binary(Tensor<type, 2>&, const type&, const type&, const Index&);

    void unscale_minimum_maximum(Tensor<type, 2>&, const Index&, const Descriptives&, const type& = type(-1), const type& = type(1));
    void unscale_mean_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void unscale_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void unscale_logarithmic(Tensor<type, 2>&, const Index&);

}

#endif // STATISTICS_H
