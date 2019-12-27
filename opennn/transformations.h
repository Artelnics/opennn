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

#include "vector.h"
#include "matrix.h"
#include "tensor.h"
#include "statistics.h"

using namespace std;

namespace OpenNN
{
   /// Scaling methods

    // Minimum-maximum vector scaling

     void scale_minimum_maximum(Vector<double>&, const double &, const double &);

     void scale_minimum_maximum(Vector<double>&, const Descriptives&);

     Descriptives scale_minimum_maximum(Vector<double>&);

     // Minimum-maximum matrix scaling

     void scale_minimum_maximum(Matrix<double>&, const Vector<Descriptives>&);

     Vector<Descriptives> scale_minimum_maximum(Matrix<double>&);

     // Minimum-maximum scaling

     void scale_rows_minimum_maximum(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&);

     void scale_columns_minimum_maximum(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&);

     // Mean-std vector scaling

     void scale_mean_standard_deviation(Vector<double>&, const Descriptives &);
     void scale_mean_standard_deviation(Vector<double>&, const double &, const double &);

     Descriptives scale_mean_standard_deviation(Vector<double>&);

     // Mean-std matrix scaling

     void scale_mean_standard_deviation(Matrix<double>&, const Vector<Descriptives>&);
     Vector<Descriptives> scale_mean_standard_deviation(Matrix<double>&);
     void scale_rows_mean_standard_deviation(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&);
     void scale_columns_mean_standard_deviation(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&);

     // Standard deviation vector scaling

     void scale_standard_deviation(Vector<double>&, const double &);
     void scale_standard_deviation(Vector<double>&, const Vector<double>&);
     void scale_standard_deviation(Vector<double>&, const Descriptives &);

     Descriptives scale_standard_deviation(Vector<double>&);

     // Range matrix scaling

     void scale_range(Matrix<double>&, const Vector<Descriptives>&, const double& minimum, const double& maximum);

     Vector<Descriptives> scale_range(Matrix<double>&, const double&, const double&);

     // Logarithmic matrix scaling

     void scale_logarithmic(Matrix<double>&, const Vector<Descriptives>&);

     Vector<Descriptives> scale_logarithmic(Matrix<double>&);

     void scale_rows_logarithmic(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&);

     void scale_columns_logarithmic(Matrix<double>&, const Vector<Descriptives >&, const Vector<size_t>&);

    ///Unscaling Methods

     // Minimum-maximum vector unscaling

     void unscale_minimum_maximum(Vector<double>&, const Vector<double>&, const Vector<double>&);

     // Minimum-maximum matrix unscaling

     void unscale_minimum_maximum(Matrix<double>&, const Vector<Descriptives>&);

     void unscale_rows_minimum_maximum(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&);

     void unscale_columns_minimum_maximum(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&);

     // Mean-std vector unscaling

     void unscale_mean_standard_deviation(Vector<double>&, const Vector<double>&, const Vector<double>&);

     // Mean-std matrix unscaling

     void unscale_mean_standard_deviation(Matrix<double>&, const Vector<Descriptives>&);

     void unscale_rows_mean_standard_deviation(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&);

     void unscale_columns_mean_standard_deviation(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&);

     // Logarighmic matrix unscaling

     void unscale_logarithmic(Matrix<double>&, const Vector<Descriptives>&);

     void unscale_rows_logarithmic(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&);

     void unscale_columns_logarithmic(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&);

     // Association

     void transform_association(Matrix<double>&);

     // Bounding methods

     void apply_lower_bound(Vector<double>&, const double&);

     void apply_lower_bound(Vector<double>&, const Vector<double>&);

     void apply_upper_bound(Vector<double>&, const double&);

     void apply_upper_bound(Vector<double>&, const Vector<double>&);

     void apply_lower_upper_bounds(Vector<double>&, const double &, const double &);

     void apply_lower_upper_bounds(Vector<double>&, const Vector<double>&, const Vector<double>&);

     void transform_time_series(Matrix<double>&, const size_t&, const size_t&, const size_t&);
     void transform_time_series(Matrix<double>&, const size_t&, const size_t&);
}

#endif
