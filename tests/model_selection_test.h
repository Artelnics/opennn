//   OpenNN: An Open Source Neural Networks C++ Library                    
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   T E S T   C L A S S   H E A D E R     
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#ifndef MODELSELECTIONTEST_H
#define MODELSELECTIONTEST_H

#include "../opennn/unit_testing.h"
#include "../opennn/data_set.h"
#include "../opennn/neural_network.h"
#include "../opennn/training_strategy.h"
#include "../opennn/model_selection.h"

namespace opennn
{

class ModelSelectionTest : public UnitTesting 
{

public:

   explicit ModelSelectionTest();

   virtual ~ModelSelectionTest();

   // Constructor and destructor methods

   void test_constructor();

   void test_destructor();

   // Model selection methods

   void test_perform_neurons_selection();

   // Serialization methods

   void test_to_XML();   
   void test_from_XML();
   void test_save();
   void test_load();

   // Unit testing methods

   void run_test_case();

private:

   DataSet data_set;

   NeuralNetwork neural_network;

   TrainingStrategy training_strategy;

   ModelSelection model_selection;

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
