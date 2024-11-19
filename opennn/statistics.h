//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#ifndef STATISTICS_H
#define STATISTICS_H

#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <vector>

#include "config.h"
#include "descriptives.h"
#include "box_plot.h"
#include "histogram.h"

namespace opennn
{

// Minimum

 type minimum(const Tensor<type, 1>&);
 type minimum(const Tensor<type, 1>&, const vector<Index>&);
 Index minimum(const Tensor<Index, 1>&);
 type minimum(const Tensor<type, 2>&);
 Tensor<type, 1> column_minimums(const Tensor<type, 2>&, const vector<Index>& = vector<Index>(), const vector<Index>& = vector<Index>());

 // Maximum

 type maximum(const Tensor<type, 1>&);
 type maximum(const Tensor<type, 1>&, const vector<Index>&);
 Index maximum(const Tensor<Index, 1>&);
 //type maximum(const Tensor<type, 2>&);
 Tensor<type, 1> column_maximums(const Tensor<type, 2>&, 
	                             const vector<Index>& = vector<Index>(), 
	                             const vector<Index>& = vector<Index>());

 // Range
 type range(const Tensor<type, 1>&);

 // Mean
 type mean(const Tensor<type, 1>&);
 type mean(const Tensor<type, 1>&, const Index&, const Index&);
 type mean(const Tensor<type, 2>&,  const Index&);
 Tensor<type, 1> mean(const Tensor<type, 2>&);
 Tensor<type, 1> mean(const Tensor<type, 2>&, const vector<Index>&);
 Tensor<type, 1> mean(const Tensor<type, 2>&, const vector<Index>&, const vector<Index>&);

 // Median
 type median(const Tensor<type, 1>&);
 type median(const Tensor<type, 2>&, const Index&);
 Tensor<type, 1> median(const Tensor<type, 2>&);
 Tensor<type, 1> median(const Tensor<type, 2>&, const vector<Index>&);
 Tensor<type, 1> median(const Tensor<type, 2>&, const vector<Index>&, const vector<Index>&);

 // Variance
 type variance(const Tensor<type, 1>&);
 type variance(const Tensor<type, 1>&, const Tensor<Index, 1>&);

 // Standard deviation
 type standard_deviation(const Tensor<type, 1>&);
 type standard_deviation(const Tensor<type, 1>&, const Tensor<Index, 1>&);
 Tensor<type, 1> standard_deviation(const Tensor<type, 1>&, const Index&);

 // Assymetry
 type asymmetry(const Tensor<type, 1>&);

 // Kurtosis
 type kurtosis(const Tensor<type, 1>&);

 // Quartiles
 Tensor<type, 1> quartiles(const Tensor<type, 1>&);
 Tensor<type, 1> quartiles(const Tensor<type, 1>&, const vector<Index>&);

 // Box plot
 BoxPlot box_plot(const Tensor<type, 1>&);
 BoxPlot box_plot(const Tensor<type, 1>&, const vector<Index>&);

 // Descriptives vector
 Descriptives vector_descriptives(const Tensor<type, 1>&);

 // Descriptives matrix
 vector<Descriptives> descriptives(const Tensor<type, 2>&);
 vector<Descriptives> descriptives(const Tensor<type, 2>&, const vector<Index>&, const vector<Index>&);

 // Histograms
 Histogram histogram(const Tensor<type, 1>&, const Index&  = 10);
 Histogram histogram_centered(const Tensor<type, 1>&, const type& = type(0), const Index&  = 10);
 Histogram histogram(const Tensor<bool, 1>&);
 Histogram histogram(const Tensor<Index, 1>&, const Index&  = 10);
 Tensor<Histogram, 1> histograms(const Tensor<type, 2>&, const Index& = 10);
 Tensor<Index, 1> total_frequencies(const Tensor<Histogram, 1>&);


 // Minimal indices
 Index minimal_index(const Tensor<type, 1>&);
 Tensor<Index, 1> minimal_indices(const Tensor<type, 1>&, const Index&);
 Tensor<Index, 1> minimal_indices(const Tensor<type, 2>&);

 // Maximal indices
 Index maximal_index(const Tensor<type, 1>&);
 Index maximal_index_from_indices(const Tensor<type, 1>&, const Tensor<Index, 1>&);
 Tensor<Index, 1> maximal_indices(const Tensor<type, 1>&, const Index&);
 Tensor<Index, 1> maximal_indices(const Tensor<type, 2>&);
 Tensor<Index, 2> maximal_column_indices(const Tensor<type, 2>&, const Index&);
 
 // Percentiles
 Tensor<type, 1> percentiles(const Tensor<type, 1>&);

 // NAN
 Index count_nan(const Tensor<type, 1>&);
}

#endif // STATISTICS_H
