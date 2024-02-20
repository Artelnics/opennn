#ifndef DESCRIPTIVES_H
#define DESCRIPTIVES_H

#include <string>

#include "config.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

/// This structure contains the simplest Descriptives for a set, variable, etc. It includes :

///
/// <ul>
/// <li> Minimum.
/// <li> Maximum.
/// <li> Mean.
/// <li> Standard Deviation.
/// </ul>

struct Descriptives {

  // Default constructor.

  explicit Descriptives();

  // Values constructor.

  explicit Descriptives(const type&, const type&, const type&, const type&);

  explicit Descriptives(const Tensor<type, 1>&);

  // Set methods

  void set(const type&, const type&, const type&, const type&);

  void set_minimum(const type&);

  void set_maximum(const type&);

  void set_mean(const type&);

  void set_standard_deviation(const type&);

  Tensor<type, 1> to_vector() const;

  bool has_minimum_minus_one_maximum_one();

  bool has_mean_zero_standard_deviation_one();

  void save(const string &file_name) const;

  void print(const string& = "Descriptives:") const;

  /// Name of variable

  string name = "Descriptives";

  /// Smallest value of a set, function, etc.

  type minimum = type(-1.0);

  /// Biggest value of a set, function, etc.

  type maximum = type(1);

  /// Mean value of a set, function, etc.

  type mean = type(0);

  /// Standard deviation value of a set, function, etc.

  type standard_deviation = type(1);

};



}
#endif
