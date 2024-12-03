//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#ifndef SCALING_H
#define SCALING_H

#include "descriptives.h"

namespace opennn
{
    enum class Scaler{None, MinimumMaximum, MeanStandardDeviation, StandardDeviation, Logarithm, ImageMinMax};

    void scale_mean_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void scale_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void scale_minimum_maximum(Tensor<type, 2>&, const Index&, const Descriptives&, const type& = type(-1), const type& = type(1));

    // Tensor<type, 1> scale_minimum_maximum(const Tensor<type, 1>&);
    // Tensor<type, 2> scale_minimum_maximum(const Tensor<type, 2>&);

    void scale_logarithmic(Tensor<type, 2>&, const Index&);

    //void scale_minimum_maximum_binary(Tensor<type, 2>&, const type&, const type&, const Index&);

    void unscale_minimum_maximum(Tensor<type, 2>&, const Index&, const Descriptives&, const type& = type(-1), const type& = type(1));
    void unscale_mean_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void unscale_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void unscale_logarithmic(Tensor<type, 2>&, const Index&);
    void unscale_image_minimum_maximum(Tensor<type, 2>&, const Index&);
}

#endif // STATISTICS_H
