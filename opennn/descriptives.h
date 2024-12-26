#ifndef DESCRIPTIVES_H
#define DESCRIPTIVES_H

#include "pch.h"

namespace opennn
{

struct Descriptives
{
  Descriptives(const type& = type(NAN), const type& = type(NAN), const type& = type(NAN), const type& = type(NAN));

  Tensor<type, 1> to_tensor() const;

  void set(const type& = type(NAN), const type& = type(NAN), const type& = type(NAN), const type& = type(NAN));

  void save(const filesystem::path&) const;

  void print(const string& = "Descriptives:") const;

  string name = "Descriptives";

  type minimum = type(-1.0);

  type maximum = type(1);

  type mean = type(0);

  type standard_deviation = type(1);

};



}
#endif
