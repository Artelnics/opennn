#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "pch.h"

namespace opennn
{

struct Histogram
{

  explicit Histogram(const Index& = 0);

  explicit Histogram(const Tensor<type, 1>&, const Tensor<Index, 1>&);

  explicit Histogram(const Tensor<Index, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&);

  explicit Histogram(const Tensor<type, 1>&, const Index&);

  explicit Histogram(const Tensor<type, 1>&);

  // Methods

  Index get_bins_number() const;

  Index count_empty_bins() const;

  Index calculate_minimum_frequency() const;

  Index calculate_maximum_frequency() const;

  Index calculate_most_populated_bin() const;

  Tensor<type, 1> calculate_minimal_centers() const;

  Tensor<type, 1> calculate_maximal_centers() const;

  Index calculate_bin(const type&) const;

  Index calculate_frequency(const type&) const;

  void save(const string&) const;

  Tensor<type, 1> minimums;

  Tensor<type, 1> maximums;

  Tensor<type, 1> centers;

  Tensor<Index, 1> frequencies;
};


}
#endif
