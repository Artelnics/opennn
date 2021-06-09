//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   T E S T   C L A S S   H E A D E R         
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SCALINGTEST_H
#define SCALINGTEST_H

// Unit testing includes

#include "unit_testing.h"

class ScalingTest : public UnitTesting
{

public:  

   explicit ScalingTest();

    // Scaling

    void test_scale_mean_standard_deviation();
    void test_scale_minimum_maximum();

    // Unscaling

    void test_unscale_data_mean_standard_deviation();
    void test_unscale_data_minimum_maximum();



   virtual ~ScalingTest();

   void run_test_case();
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
