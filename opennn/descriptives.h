#ifndef DESCRIPTIVES_H
#define DESCRIPTIVES_H

#include <string>

#include "config.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct Descriptives {

  explicit Descriptives(const type& = type(NAN), const type& = type(NAN), const type& = type(NAN), const type& = type(NAN));

  Tensor<type, 1> to_tensor() const;

  void set(const type& = type(NAN), const type& = type(NAN), const type& = type(NAN), const type& = type(NAN));

  void set_minimum(const type&);

  void set_maximum(const type&);

  void set_mean(const type&);

  void set_standard_deviation(const type&);

  Tensor<type, 1> to_vector() const;

//  bool has_minimum_minus_one_maximum_one();

//  bool has_mean_zero_standard_deviation_one();

  void save(const string &file_name) const;

  void print(const string& = "Descriptives:") const;

  string name = "Descriptives";

  type minimum = type(-1.0);

  type maximum = type(1);

  type mean = type(0);

  type standard_deviation = type(1);

};



}
#endif
