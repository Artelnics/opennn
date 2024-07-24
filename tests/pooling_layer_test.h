//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef POOLINGLAYERTEST_H
#define POOLINGLAYERTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class PoolingLayerTest : public UnitTesting
{

public:

   explicit PoolingLayerTest();

   virtual ~PoolingLayerTest();

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor();
//   void test_calculate_max_pooling_outputs();
//   void test_calculate_average_pooling_outputs();
   void test_forward_propagate_average_pooling();
   void test_forward_propagate_max_pooling();

   // Unit testing methods

   void run_test_case();

private:

   PoolingLayer pooling_layer;

};

#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
