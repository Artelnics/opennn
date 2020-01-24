//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TRANSFORMATIONS_H
#define TRANSFORMATIONS_H

// System includes

#include <math.h>

// OpenNN includes

#include "config.h"
#include "statistics.h"

using namespace std;

namespace OpenNN
{
   /// Scaling methods

    // Minimum-maximum vector scaling

     void scale_minimum_maximum(Tensor<type, 1>&, const type &, const type &);

     void scale_minimum_maximum(Tensor<type, 1>&, const Descriptives&);

     Descriptives scale_minimum_maximum(Tensor<type, 1>&);

     // Minimum-maximum matrix scaling

     void scale_minimum_maximum(Tensor<type, 2>&, const Tensor<Descriptives, 1>&);

     Tensor<Descriptives, 1> scale_minimum_maximum(Tensor<type, 2>&);

     // Minimum-maximum scaling

     void scale_rows_minimum_maximum(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);

     void scale_columns_minimum_maximum(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);

     // Mean-std vector scaling

     void scale_mean_standard_deviation(Tensor<type, 1>&, const Descriptives &);
     void scale_mean_standard_deviation(Tensor<type, 1>&, const type &, const type &);

     Descriptives scale_mean_standard_deviation(Tensor<type, 1>&);

     // Mean-std matrix scaling

     void scale_mean_standard_deviation(Tensor<type, 2>&, const Tensor<Descriptives, 1>&);
     Tensor<Descriptives, 1> scale_mean_standard_deviation(Tensor<type, 2>&);
     void scale_rows_mean_standard_deviation(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);
     void scale_columns_mean_standard_deviation(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);

     // Standard deviation vector scaling

     void scale_standard_deviation(Tensor<type, 1>&, const type &);
     void scale_standard_deviation(Tensor<type, 1>&, const Tensor<type, 1>&);
     void scale_standard_deviation(Tensor<type, 1>&, const Descriptives &);

     Descriptives scale_standard_deviation(Tensor<type, 1>&);

     // Range matrix scaling

     void scale_range(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const type& minimum, const type& maximum);

     Tensor<Descriptives, 1> scale_range(Tensor<type, 2>&, const type&, const type&);

     // Logarithmic matrix scaling

     void scale_logarithmic(Tensor<type, 2>&, const Tensor<Descriptives, 1>&);

     Tensor<Descriptives, 1> scale_logarithmic(Tensor<type, 2>&);

     void scale_rows_logarithmic(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);

     void scale_columns_logarithmic(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);

    ///Unscaling Methods

     // Minimum-maximum vector unscaling

     void unscale_minimum_maximum(Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

     // Minimum-maximum matrix unscaling

     void unscale_minimum_maximum(Tensor<type, 2>&, const Tensor<Descriptives, 1>&);

     void unscale_rows_minimum_maximum(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);

     void unscale_columns_minimum_maximum(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);

     // Mean-std vector unscaling

     void unscale_mean_standard_deviation(Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

     // Mean-std matrix unscaling

     void unscale_mean_standard_deviation(Tensor<type, 2>&, const Tensor<Descriptives, 1>&);

     void unscale_rows_mean_standard_deviation(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);

     void unscale_columns_mean_standard_deviation(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);

     // Logarighmic matrix unscaling

     void unscale_logarithmic(Tensor<type, 2>&, const Tensor<Descriptives, 1>&);

     void unscale_rows_logarithmic(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);

     void unscale_columns_logarithmic(Tensor<type, 2>&, const Tensor<Descriptives, 1>&, const Tensor<Index, 1>&);

     // Association

     void transform_association(Tensor<type, 2>&);

     // Bounding methods

     void apply_lower_bound(Tensor<type, 1>&, const type&);

     void apply_lower_bound(Tensor<type, 1>&, const Tensor<type, 1>&);

     void apply_upper_bound(Tensor<type, 1>&, const type&);

     void apply_upper_bound(Tensor<type, 1>&, const Tensor<type, 1>&);

     void apply_lower_upper_bounds(Tensor<type, 1>&, const type &, const type &);

     void apply_lower_upper_bounds(Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

     void transform_time_series(Tensor<type, 2>&, const Index&, const Index&, const Index&);
     void transform_time_series(Tensor<type, 2>&, const Index&, const Index&);
}

#endif
