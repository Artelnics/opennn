//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   T E S T   C L A S S   H E A D E R       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef BOUNDINGLAYERTEST_H
#define BOUNDINGLAYERTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/bounding_layer.h"

namespace opennn
{

class BoundingLayerTest : public UnitTesting
{

public: 

   explicit BoundingLayerTest();

   virtual ~BoundingLayerTest();

   // Constructor and destructor methods

   void test_constructor();

   void test_destructor();

   // Output methods

   void test_forward_propagate();

   // Unit testing methods

   void run_test_case();

private:

   BoundingLayer bounding_layer;

   BoundingLayerForwardPropagation bounding_layer_forward_propagation;
};

}

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
