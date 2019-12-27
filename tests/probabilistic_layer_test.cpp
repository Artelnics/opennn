//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   T E S T   C L A S S           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "probabilistic_layer_test.h"


ProbabilisticLayerTest::ProbabilisticLayerTest() : UnitTesting()
{

}


ProbabilisticLayerTest::~ProbabilisticLayerTest()
{

}


void ProbabilisticLayerTest::test_constructor()
{
   cout << "test_constructor\n";

   // Default constructor

   ProbabilisticLayer probabilistic_layer_l1;

   assert_true(probabilistic_layer_l1.get_neurons_number() == 0, LOG);

   // Probabilistic neurons number constructor

   ProbabilisticLayer probabilistic_layer_l2;

   probabilistic_layer_l2.set_neurons_number(3);

   assert_true(probabilistic_layer_l2.get_neurons_number() == 3, LOG);

   // Copy constructor

   ProbabilisticLayer probabilistic_layer_l3(probabilistic_layer_l2);

   probabilistic_layer_l3.set_neurons_number(probabilistic_layer_l2.get_neurons_number());

   assert_true(probabilistic_layer_l3.get_neurons_number() == 3, LOG);

}

void ProbabilisticLayerTest::test_destructor()
{
   cout << "test_destructor\n";
}


void ProbabilisticLayerTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

   ProbabilisticLayer probabilistic_layer_l1;
   ProbabilisticLayer probabilistic_layer_l2;

   // Test

   probabilistic_layer_l1.set_neurons_number(2);

   probabilistic_layer_l2.set_neurons_number(probabilistic_layer_l1.get_neurons_number());

   assert_true(probabilistic_layer_l2.get_neurons_number() == 2, LOG);

}


void ProbabilisticLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   ProbabilisticLayer probabilistic_layer;

   // Test

   probabilistic_layer.set();
   assert_true(probabilistic_layer.get_neurons_number() == 0, LOG);

   // Test

   probabilistic_layer.set_neurons_number(1);
   assert_true(probabilistic_layer.get_neurons_number() == 1, LOG);

}


void ProbabilisticLayerTest::test_set()
{
   cout << "test_set\n";
}


void ProbabilisticLayerTest::test_set_default()
{
   cout << "test_set_default\n";
}


void ProbabilisticLayerTest::test_get_display()
{
   cout << "test_get_display\n";
}


void ProbabilisticLayerTest::test_set_display()
{
   cout << "test_set_display\n";
}


void ProbabilisticLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   ProbabilisticLayer probabilistic_layer;

   Tensor<double> inputs;
   Tensor<double> outputs;
   Vector<double> biases;
   Matrix<double> synaptic_weights;

   // Test

   synaptic_weights.set(1,1,1.0);
   probabilistic_layer.set_synaptic_weights(synaptic_weights);

   biases.set(1,1.0);
   probabilistic_layer.set_synaptic_weights(synaptic_weights);

   probabilistic_layer.set_neurons_number(1);

   probabilistic_layer.set_activation_function(ProbabilisticLayer::Binary);

   inputs.set({1,1}, 0.0);

   outputs = probabilistic_layer.calculate_outputs(inputs);

   assert_true(outputs.size() == 1, LOG);
   assert_true(outputs[0] == 0.0, LOG);

   // Test

   probabilistic_layer.set_neurons_number(1);
   probabilistic_layer.set_activation_function(ProbabilisticLayer::Logistic);

   inputs.set({1,1}, 0.0);
   outputs = probabilistic_layer.calculate_outputs(inputs);

   assert_true(outputs.size() == 1, LOG);
   assert_true(outputs[0] >= 0.0, LOG);
}


void ProbabilisticLayerTest::test_to_XML()
{
   cout << "test_to_XML\n";

   ProbabilisticLayer  probabilistic_layer;
   tinyxml2::XMLDocument* pld;

   // Test

   probabilistic_layer.set();

   pld = probabilistic_layer.to_XML();

   assert_true(pld != nullptr, LOG);

   // Test

   probabilistic_layer.set_neurons_number(2);
   probabilistic_layer.set_activation_function(ProbabilisticLayer::Competitive);
   probabilistic_layer.set_display(false);

   pld = probabilistic_layer.to_XML();

   probabilistic_layer.from_XML(*pld);

   assert_true(probabilistic_layer.get_neurons_number() == 2, LOG);
   assert_true(probabilistic_layer.get_activation_function() == ProbabilisticLayer::Competitive, LOG);
   assert_true(probabilistic_layer.get_display() == false, LOG);

   delete pld;
}


void ProbabilisticLayerTest::test_from_XML()
{
   cout << "test_from_XML\n";

   ProbabilisticLayer  probabilistic_layer;
   tinyxml2::XMLDocument* pld;

   // Test

   pld = probabilistic_layer.to_XML();

   probabilistic_layer.from_XML(*pld);

   delete pld;
}


void ProbabilisticLayerTest::test_calculate_activation_derivatives()
{
    cout << "test_calculate_activation_derivatives\n";

    ProbabilisticLayer probabilistic_layer;

    Tensor<double> combinations;
    Tensor<double> derivatives;

    NumericalDifferentiation numerical_differentiation;

    Tensor<double> activations_derivatives;
    Tensor<double> numerical_activation_derivative;

    // Test

    probabilistic_layer.set(1,1);
    probabilistic_layer.set_activation_function(ProbabilisticLayer::Logistic);

    combinations.set({1, 1}, 0.0);

    derivatives = probabilistic_layer.calculate_activations_derivatives(combinations);
    assert_true(abs(derivatives(0,0) - 0.25) < numeric_limits<double>::min(), LOG);

    // Test numerical differentiation

    if(numerical_differentiation_tests)
    {
       probabilistic_layer.set(2, 4);

       combinations.set({1,4}, 1.0);

       probabilistic_layer.set_activation_function(ProbabilisticLayer::Softmax);

       activations_derivatives = probabilistic_layer.calculate_activations_derivatives(combinations);

       numerical_differentiation.calculate_derivatives(probabilistic_layer,
                                                       &ProbabilisticLayer::calculate_activations,
                                                       combinations);

       assert_true((absolute_value(activations_derivatives - numerical_activation_derivative)) < 1.0e-3, LOG);
    }
}


void ProbabilisticLayerTest::run_test_case()
{
   cout << "Running probabilistic layer test case...\n";

   // Constructor and destructor methods

   test_constructor();

   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Layer architecture

   test_get_neurons_number();

   // Display messages

   test_get_display();

   // Set methods

   test_set();

   test_set_default();

   // Display messages

   test_set_display();

   // Probabilistic post-processing

   test_calculate_outputs();

   // Serialization methods

   test_to_XML();

   test_from_XML();

   // Activation derivatives

   test_calculate_activation_derivatives();

   cout << "End of probabilistic layer test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
