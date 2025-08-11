//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#ifndef SCALING_H
#define SCALING_H

#include "statistics.h"

namespace opennn
{
    enum class Scaler{None, MinimumMaximum, MeanStandardDeviation, StandardDeviation, Logarithm, ImageMinMax};

    string scaler_to_string(const Scaler& scaler);

    Scaler string_to_scaler(const string& new_scaler);

    void scale_mean_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void scale_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void scale_minimum_maximum(Tensor<type, 2>&, const Index&, const Descriptives&, const type& = type(-1), const type& = type(1));
    void scale_logarithmic(Tensor<type, 2>&, const Index&);

    void scale_mean_standard_deviation_3d(Tensor<type, 3>&, const Index&, const Descriptives&);
    void scale_standard_deviation_3d(Tensor<type, 3>&, const Index&, const Descriptives&);
    void scale_minimum_maximum_3d(Tensor<type, 3>&, const Index&, const Descriptives&, const type&, const type&);
    void scale_logarithmic_3d(Tensor<type, 3>&, const Index&);

    void unscale_minimum_maximum(Tensor<type, 2>&, const Index&, const Descriptives&, const type& = type(-1), const type& = type(1));
    void unscale_mean_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void unscale_standard_deviation(Tensor<type, 2>&, const Index&, const Descriptives&);
    void unscale_logarithmic(Tensor<type, 2>&, const Index&);
    void unscale_image_minimum_maximum(Tensor<type, 2>&, const Index&);
}

#endif // STATISTICS_H
