//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   P E R C E P T R O N   L A Y E R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef FLATTENPERCEPTRONLAYERTEST_H
#define FLATTENPERCEPTRONLAYERTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class FlattenPerceptronLayerTest : public UnitTesting
{

public:

  FlattenPerceptronLayerTest();

  virtual ~FlattenPerceptronLayerTest();

  //Forward propagate
  void test_flatten_perceptron_forward_propagate();

  //Backward pass
  void test_flatten_perceptron_backward_pass();

  // Unit testing methods

  void run_test_case() final override;

private:

};

#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
