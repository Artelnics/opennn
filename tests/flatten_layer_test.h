//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef FLATTENLAYERTEST_H
#define FLATTENLAYERTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class FlattenLayerTest : public UnitTesting
{

public:

   explicit FlattenLayerTest();

   virtual ~FlattenLayerTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();
   void test_calculate_flatten_outputs();

   // Unit testing methods

   void run_test_case();

private:

   FlattenLayer flatten_layer;

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
