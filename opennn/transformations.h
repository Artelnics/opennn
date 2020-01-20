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

#include "statistics.h"

using namespace std;

namespace OpenNN
{
   /// Scaling methods

    // Minimum-maximum vector scaling

     void scale_minimum_maximum(Tensor<double, 1>&, const double &, const double &);

     void scale_minimum_maximum(Tensor<double, 1>&, const Descriptives&);

     Descriptives scale_minimum_maximum(Tensor<double, 1>&);

     // Minimum-maximum matrix scaling

     void scale_minimum_maximum(Tensor<double, 2>&, const vector<Descriptives>&);

     vector<Descriptives> scale_minimum_maximum(Tensor<double, 2>&);

     // Minimum-maximum scaling

     void scale_rows_minimum_maximum(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);

     void scale_columns_minimum_maximum(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);

     // Mean-std vector scaling

     void scale_mean_standard_deviation(Tensor<double, 1>&, const Descriptives &);
     void scale_mean_standard_deviation(Tensor<double, 1>&, const double &, const double &);

     Descriptives scale_mean_standard_deviation(Tensor<double, 1>&);

     // Mean-std matrix scaling

     void scale_mean_standard_deviation(Tensor<double, 2>&, const vector<Descriptives>&);
     vector<Descriptives> scale_mean_standard_deviation(Tensor<double, 2>&);
     void scale_rows_mean_standard_deviation(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);
     void scale_columns_mean_standard_deviation(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);

     // Standard deviation vector scaling

     void scale_standard_deviation(Tensor<double, 1>&, const double &);
     void scale_standard_deviation(Tensor<double, 1>&, const Tensor<double, 1>&);
     void scale_standard_deviation(Tensor<double, 1>&, const Descriptives &);

     Descriptives scale_standard_deviation(Tensor<double, 1>&);

     // Range matrix scaling

     void scale_range(Tensor<double, 2>&, const vector<Descriptives>&, const double& minimum, const double& maximum);

     vector<Descriptives> scale_range(Tensor<double, 2>&, const double&, const double&);

     // Logarithmic matrix scaling

     void scale_logarithmic(Tensor<double, 2>&, const vector<Descriptives>&);

     vector<Descriptives> scale_logarithmic(Tensor<double, 2>&);

     void scale_rows_logarithmic(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);

     void scale_columns_logarithmic(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);

    ///Unscaling Methods

     // Minimum-maximum vector unscaling

     void unscale_minimum_maximum(Tensor<double, 1>&, const Tensor<double, 1>&, const Tensor<double, 1>&);

     // Minimum-maximum matrix unscaling

     void unscale_minimum_maximum(Tensor<double, 2>&, const vector<Descriptives>&);

     void unscale_rows_minimum_maximum(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);

     void unscale_columns_minimum_maximum(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);

     // Mean-std vector unscaling

     void unscale_mean_standard_deviation(Tensor<double, 1>&, const Tensor<double, 1>&, const Tensor<double, 1>&);

     // Mean-std matrix unscaling

     void unscale_mean_standard_deviation(Tensor<double, 2>&, const vector<Descriptives>&);

     void unscale_rows_mean_standard_deviation(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);

     void unscale_columns_mean_standard_deviation(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);

     // Logarighmic matrix unscaling

     void unscale_logarithmic(Tensor<double, 2>&, const vector<Descriptives>&);

     void unscale_rows_logarithmic(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);

     void unscale_columns_logarithmic(Tensor<double, 2>&, const vector<Descriptives>&, const vector<int>&);

     // Association

     void transform_association(Tensor<double, 2>&);

     // Bounding methods

     void apply_lower_bound(Tensor<double, 1>&, const double&);

     void apply_lower_bound(Tensor<double, 1>&, const Tensor<double, 1>&);

     void apply_upper_bound(Tensor<double, 1>&, const double&);

     void apply_upper_bound(Tensor<double, 1>&, const Tensor<double, 1>&);

     void apply_lower_upper_bounds(Tensor<double, 1>&, const double &, const double &);

     void apply_lower_upper_bounds(Tensor<double, 1>&, const Tensor<double, 1>&, const Tensor<double, 1>&);

     void transform_time_series(Tensor<double, 2>&, const int&, const int&, const int&);
     void transform_time_series(Tensor<double, 2>&, const int&, const int&);
}

#endif
