//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N S   S E L E C T I O N   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#ifndef NEURONSSELECTIONALGORITHMTEST_H
#define NEURONSSELECTIONALGORITHMTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class NeuronsSelectionTest : public UnitTesting
{

public:

    // CONSTRUCTOR

    explicit NeuronsSelectionTest();

    virtual ~NeuronsSelectionTest();

    // Constructor and destructor methods

    void test_constructor();

    void test_destructor();

    // Unit testing methods

    void run_test_case();

private:

    DataSet data_set;

    NeuralNetwork neural_network;

    TrainingStrategy training_strategy;

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
