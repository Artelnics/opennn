/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N E U R A L   N E T W O R K   T E S T   C L A S S                                                          */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/


// Unit testing includes

#include "neural_network_test.h"


using namespace OpenNN;


// GENERAL CONSTRUCTOR

NeuralNetworkTest::NeuralNetworkTest() : UnitTesting()
{
}


// DESTRUCTOR

NeuralNetworkTest::~NeuralNetworkTest()
{
}


// METHODS

void NeuralNetworkTest::test_constructor()
{
   message += "test_constructor\n";

   Vector<size_t> multilayer_perceptron_architecture;

   string file_name = "../data/neural_network.xml";

   // Default constructor

   NeuralNetwork nn1;

   assert_true(nn1.has_multilayer_perceptron() == false, LOG);

   // Multilayer perceptron architecture constructor 

   multilayer_perceptron_architecture.set(3, 1);

   NeuralNetwork nn2(multilayer_perceptron_architecture);

   assert_true(nn2.has_multilayer_perceptron() == true, LOG);

   // One layer constructor 
   
   NeuralNetwork nn3(1, 2);

   assert_true(nn3.has_multilayer_perceptron() == true, LOG);

   // Two layers constructor 

   NeuralNetwork nn5(1, 2, 3);

   assert_true(nn5.has_multilayer_perceptron() == true, LOG);

   // File constructor

   nn1.save(file_name);
   NeuralNetwork nn6(file_name);

   // Copy constructor

}


void NeuralNetworkTest::test_destructor()
{
   message += "test_destructor\n";
}


void NeuralNetworkTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

   NeuralNetwork nn1;
   NeuralNetwork nn2 = nn1;

   assert_true(nn2.has_multilayer_perceptron() == false, LOG);
}


void NeuralNetworkTest::test_get_multilayer_perceptron_pointer()
{
   message += "test_get_multilayer_perceptron_pointer\n";

   NeuralNetwork nn;

   // Test

   nn.set(1, 1);
   assert_true(nn.get_multilayer_perceptron_pointer() != nullptr, LOG);
}


void NeuralNetworkTest::test_get_inputs_pointer()
{
   message += "test_get_inputs_pointer\n";

   NeuralNetwork nn(1, 1);

   // Test

   assert_true(nn.get_inputs_pointer() != nullptr, LOG);
}


void NeuralNetworkTest::test_get_outputs_pointer()
{
   message += "test_get_outputs_pointer\n";

   NeuralNetwork nn(1, 1);

   // Test

   assert_true(nn.get_outputs_pointer() != nullptr, LOG);
}


void NeuralNetworkTest::test_get_scaling_layer_pointer()
{
   message += "test_get_scaling_layer_pointer\n";

   NeuralNetwork nn;

   nn.construct_scaling_layer();

   // Test

   assert_true(nn.get_scaling_layer_pointer() != nullptr, LOG);
}


void NeuralNetworkTest::test_get_unscaling_layer_pointer()  
{
   message += "test_get_unscaling_layer_pointer\n";

   NeuralNetwork nn;

   nn.construct_unscaling_layer();

   // Test

   assert_true(nn.get_unscaling_layer_pointer() != nullptr, LOG);
}


void NeuralNetworkTest::test_get_bounding_layer_pointer()
{
   message += "test_get_bounding_layer_pointer\n";

   NeuralNetwork nn;

   nn.construct_bounding_layer();

   // Test

   assert_true(nn.get_bounding_layer_pointer() != nullptr, LOG);
}


void NeuralNetworkTest::test_get_probabilistic_layer_pointer()
{
   message += "test_get_probabilistic_layer_pointer\n";

   NeuralNetwork nn;

   nn.construct_probabilistic_layer();

   // Test

   assert_true(nn.get_probabilistic_layer_pointer() != nullptr, LOG);
}


void NeuralNetworkTest::test_get_display()
{
   message += "test_get_display\n";
}


void NeuralNetworkTest::test_set()
{
   message += "test_set\n";

   NeuralNetwork nn;

   // Test
}


void NeuralNetworkTest::test_set_default()
{
   message += "test_set_default\n";
}


void NeuralNetworkTest::test_set_variables()
{
   message += "test_set_variables\n";
}


void NeuralNetworkTest::test_set_variables_statistics()
{
   message += "test_set_variables_statistics\n";
}


void NeuralNetworkTest::test_set_display_inputs_warning()
{
   message += "test_set_display_inputs_warning\n";
}


void NeuralNetworkTest::test_set_display()
{
   message += "test_set_display\n";
}


void NeuralNetworkTest::test_prune_input()
{
   message += "test_prune_input\n";

   NeuralNetwork nn;

   // Test

   nn.set(1, 1);

   nn.prune_input(0);

   assert_true(nn.get_inputs_pointer()->get_inputs_number() == 0, LOG);
   assert_true(nn.get_outputs_pointer()->get_outputs_number() == 1, LOG);

   // Test

   nn.set(2, 2, 2);

   nn.prune_input(1);

   assert_true(nn.get_inputs_pointer()->get_inputs_number() == 1, LOG);
   assert_true(nn.get_outputs_pointer()->get_outputs_number() == 2, LOG);

}


void NeuralNetworkTest::test_prune_output()
{
   message += "test_prune_output\n";

   NeuralNetwork nn;

   // Test

   nn.set(1, 1);

   nn.prune_output(0);

   assert_true(nn.get_inputs_pointer()->get_inputs_number() == 1, LOG);
   assert_true(nn.get_outputs_pointer()->get_outputs_number() == 0, LOG);

   // Test

   nn.set(2, 2, 2);

   nn.prune_output(1);

   assert_true(nn.get_inputs_pointer()->get_inputs_number() == 2, LOG);
   assert_true(nn.get_outputs_pointer()->get_outputs_number() == 1, LOG);

}


void NeuralNetworkTest::test_get_parameters_number()
{
   message += "test_get_parameters_number\n";

   NeuralNetwork nn;

   // Test

   nn.set();
   assert_true(nn.get_parameters_number() == 0, LOG);

   // Test

   nn.set(1, 1, 1);
   assert_true(nn.get_parameters_number() == 4, LOG);

   // Test

   nn.set(1, 1, 1);

   assert_true(nn.get_parameters_number() == 5, LOG);
}


void NeuralNetworkTest::test_get_parameters()   
{
   message += "test_get_parameters\n";

   NeuralNetwork nn;
   Vector<double> parameters;

   // Test

   nn.set();
   parameters = nn.get_parameters();

   assert_true(parameters.size() == 0, LOG);

   // Test

   nn.set(1, 1, 1);
   nn.initialize_parameters(0.0);
   parameters = nn.get_parameters();

   assert_true(parameters.size() == 5, LOG);
   assert_true(parameters == 0.0, LOG);

   // Test

   nn.set();
   nn.initialize_parameters(0.0);
   parameters = nn.get_parameters();
   assert_true(parameters.size() == 1, LOG);
   assert_true(parameters == 0.0, LOG);

   // Test

   nn.set(1,1,1);
   nn.initialize_parameters(0.0);
   parameters = nn.get_parameters();
   assert_true(parameters.size() == 5, LOG);
   assert_true(parameters == 0.0, LOG);
}


void NeuralNetworkTest::test_set_parameters()
{
   message += "test_set_parameters\n";

   Vector<size_t> multilayer_perceptron_architecture;
   NeuralNetwork nn;

   size_t parameters_number;
   Vector<double> parameters;

   // Test

   nn.set_parameters(parameters);   

   parameters = nn.get_parameters();
   assert_true(parameters.size() == 0, LOG);

   // Test

   multilayer_perceptron_architecture.set(2, 2);
   nn.set(multilayer_perceptron_architecture);

   parameters_number = nn.get_parameters_number();

   parameters.set(0.0, 1.0, (double)parameters_number-1);
   nn.set_parameters(parameters);   
   parameters = nn.get_parameters();

   assert_true(parameters.size() == parameters_number, LOG);
   assert_true(parameters[0] == 0.0, LOG);
   assert_true(parameters[parameters_number-1] == parameters_number-1.0, LOG); 
}


// @todo Unknown scaling method: 3

void NeuralNetworkTest::test_initialize_random()
{
   message += "test_initialize_random\n";

//   NeuralNetwork nn;

    // Test

//   nn.set();

//   nn.initialize_random();
}


void NeuralNetworkTest::test_initialize_parameters()
{
   message += "test_initialize_parameters\n";

   NeuralNetwork nn;
   Vector<double> parameters;

   // Test

   nn.set(1, 1, 1);

   nn.randomize_parameters_normal(1.0, 0.0);
   parameters = nn.get_parameters();
   assert_true(parameters == 1.0, LOG);
}


void NeuralNetworkTest::test_randomize_parameters_uniform()
{
   message += "test_randomize_parameters_uniform\n";

   NeuralNetwork nn;
   Vector<double> parameters;

   // Test

   nn.set(1,1,1);
   nn.randomize_parameters_uniform();
   parameters = nn.get_parameters();
   assert_true(parameters >= -1.0, LOG);
   assert_true(parameters <=  1.0, LOG);   
}


void NeuralNetworkTest::test_randomize_parameters_normal()
{
   message += "test_randomize_parameters_normal\n";

   NeuralNetwork nn;
   Vector<double> network_parameters;

   // Test

   nn.set(1,1,1);      
   nn.randomize_parameters_normal(1.0, 0.0);
   network_parameters = nn.get_parameters();
   assert_true(network_parameters == 1.0, LOG);
}


void NeuralNetworkTest::test_calculate_parameters_norm()
{
   message += "test_calculate_parameters_norm\n";

   NeuralNetwork nn;
   double parameters_norm;

   // Test 

   nn.set();

   parameters_norm = nn.calculate_parameters_norm();

   assert_true(parameters_norm == 0.0, LOG);
}


void NeuralNetworkTest::test_calculate_outputs()
{
   message += "test_calculate_outputs\n";
/*
   NeuralNetwork nn;

   size_t inputs_number;
   size_t outputs_number;

   Vector<size_t> architecture;

   Vector<double> inputs;
   Vector<double> outputs;

   size_t parameters_number;

   Vector<double> parameters;

   double time;

   // Test

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   inputs.set(3, 0.0);

   outputs = nn.calculate_outputs(inputs);

   assert_true(outputs == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);
   nn.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   outputs = nn.calculate_outputs(inputs);

   assert_true(outputs == 0.0, LOG);

   // Test
/*
   nn.set(1, 1);

   inputs.set(1);
   inputs.randomize_normal();

   parameters = nn.get_parameters();

   assert_true(nn.calculate_outputs(inputs) == nn.calculate_outputs(inputs, parameters), LOG);

   // Test

   nn.set(4, 3, 5);

   inputs.set(4, 0.0);

   parameters_number = nn.get_parameters_number();

   parameters.set(parameters_number, 0.0);

   outputs = nn.calculate_outputs(inputs, parameters);

   assert_true(outputs.size() == 5, LOG);
   assert_true(outputs == 0.0, LOG);

   // Test

   architecture.set(5);

   architecture.randomize_uniform(5, 10);

   nn.set(architecture);

   inputs_number = nn.get_inputs_pointer()->get_inputs_number();
   outputs_number = nn.get_outputs_pointer()->get_outputs_number();

   inputs.set(inputs_number, 0.0);

   parameters_number = nn.get_parameters_number();

   parameters.set(parameters_number, 0.0);

   outputs = nn.calculate_outputs(inputs, parameters);

   assert_true(outputs.size() == outputs_number, LOG);
   assert_true(outputs == 0.0, LOG);

   //Test

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);

   inputs.set(3, 0.0);
   time = 0.0;

   outputs = nn.calculate_outputs(inputs, time);

   assert_true(outputs == 0.0, LOG);

   // Test

   nn.set(1, 1, 1);
   nn.initialize_parameters(0.0);

   inputs.set(1, 0.0);
   time = 0.0;

   outputs = nn.calculate_outputs(inputs, time);

   assert_true(outputs == 0.0, LOG);

   // Test

   nn.set(4, 3, 5);

   inputs.set(4, 0.0);
   time = 0.0;

   parameters_number = nn.get_parameters_number();

   parameters.set(parameters_number, 0.0);

   outputs = nn.calculate_outputs(inputs, parameters, time);

   assert_true(outputs.size() == 5, LOG);
   assert_true(outputs == 0.0, LOG);

   // Test

   architecture.set(5);

   architecture.randomize_uniform(5, 10);

   nn.set(architecture);

   inputs_number = nn.get_inputs_pointer()->get_inputs_number();
   outputs_number = nn.get_outputs_pointer()->get_outputs_number();

   inputs.set(inputs_number, 0.0);
   time = 0.0;

   parameters_number = nn.get_parameters_number();

   parameters.set(parameters_number, 0.0);

   outputs = nn.calculate_outputs(inputs, parameters, time);

   assert_true(outputs.size() == outputs_number, LOG);
   assert_true(outputs == 0.0, LOG);
*/
}


void NeuralNetworkTest::test_calculate_Jacobian()
{
   message += "test_calculate_Jacobian\n";
/*
   // One layer

   NeuralNetwork nn;

   Vector<size_t> multilayer_perceptron_architecture;

   Vector<double> inputs;
   Matrix<double> Jacobian;

//   Vector<double> inputs_minimum;
//   Vector<double> inputs_maximum;

//   Vector<double> inputs_mean;
//   Vector<double> inputs_standard_deviation;

//   Vector<double> outputs_minimum;
//   Vector<double> outputs_maximum;

//   Vector<double> outputs_mean;
//   Vector<double> outputs_standard_deviation;

//   mmlp.set_display(false);


   NumericalDifferentiation nd;
   Matrix<double> numerical_Jacobian;

   // Test

   nn.set(1, 1, 1);
   nn.initialize_parameters(0.0);
   inputs.set(1, 0.0);
   Jacobian = nn.calculate_Jacobian(inputs);
   assert_true(Jacobian == 0.0, LOG);    

   // Test

   nn.set(3, 4, 2);
   nn.initialize_parameters(0.0);
   inputs.set(3, 0.0);
   Jacobian = nn.calculate_Jacobian(inputs);
   assert_true(Jacobian == 0.0, LOG);    

   // Test

   if(numerical_differentiation_tests)
   {
      nn.set(3, 4, 2);
      nn.initialize_parameters(0.0);
      inputs.set(3, 0.0);
      Jacobian = nn.calculate_Jacobian(inputs);
      numerical_Jacobian = nd.calculate_Jacobian(nn, &NeuralNetwork::calculate_outputs, inputs);
      assert_true((Jacobian-numerical_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   multilayer_perceptron_architecture.set(3, 1);   
   nn.set(multilayer_perceptron_architecture);
   nn.initialize_parameters(0.0);
   inputs.set(1, 0.0);
   Jacobian = nn.calculate_Jacobian(inputs);
   assert_true(Jacobian == 0.0, LOG);    

   // Test

   multilayer_perceptron_architecture.set(3);   
   multilayer_perceptron_architecture[0] = 3;
   multilayer_perceptron_architecture[1] = 4;
   multilayer_perceptron_architecture[2] = 1;

   nn.set(multilayer_perceptron_architecture);

   nn.initialize_parameters(0.0);
   inputs.set(3, 0.0);
   Jacobian = nn.calculate_Jacobian(inputs);
   assert_true(Jacobian == 0.0, LOG);    

   // Test

   if(numerical_differentiation_tests)
   {
      multilayer_perceptron_architecture.set(3);   
      multilayer_perceptron_architecture[0] = 3;
      multilayer_perceptron_architecture[1] = 4;
      multilayer_perceptron_architecture[2] = 1;

      nn.set(multilayer_perceptron_architecture);

      inputs.set(3);
      inputs[0] = 0.0;
      inputs[1] = 1.0;
      inputs[2] = 2.0;

      Jacobian = nn.calculate_Jacobian(inputs);
      numerical_Jacobian = nd.calculate_Jacobian(nn, &NeuralNetwork::calculate_outputs, inputs);
      assert_true((Jacobian-numerical_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Scaling and unscaling test

//   if(numerical_differentiation_tests)
//   {
//      nn.set(2, 3);

//      nn.set_variables_scaling_method(NeuralNetwork::MinimumMaximum);

//      nn.set_input_minimum(0, -0.3);
//      nn.set_input_minimum(1, -0.2);
     
//      nn.set_input_maximum(0, 0.0);
//      nn.set_input_maximum(1, 0.1);

//      nn.set_output_minimum(0, -1.0);
//      nn.set_output_minimum(1, -4.1);
//      nn.set_output_minimum(2, -8.2);
     
//      nn.set_output_maximum(0, 1.0);
//      nn.set_output_maximum(1, 7.2);
//      nn.set_output_maximum(2, 6.0);

//      inputs.set(2);
//      inputs.randomize_normal();

//      Jacobian = nn.calculate_Jacobian(inputs);
//      numerical_Jacobian = nd.calculate_Jacobian(nn, &NeuralNetwork::calculate_outputs, inputs);

//      assert_true((Jacobian-numerical_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
//   }

   // Scaling and unscaling test

//   if(numerical_differentiation_tests)
//   {
//      nn.set(2, 3);

//      nn.set_variables_scaling_method(NeuralNetwork::MeanStandardDeviation);

//      nn.set_input_mean(0, -0.3);
//      nn.set_input_mean(1, -0.2);
     
//      nn.set_input_standard_deviation(0, 0.2);
//      nn.set_input_standard_deviation(1, 0.1);

//      nn.set_output_mean(0, -1.0);
//      nn.set_output_mean(1, -4.1);
//      nn.set_output_mean(2, -8.2);
     
//      nn.set_output_standard_deviation(0, 1.0);
//      nn.set_output_standard_deviation(1, 7.2);
//      nn.set_output_standard_deviation(2, 6.0);

//      inputs.set(2);
//      inputs.randomize_normal();

//      Jacobian = nn.calculate_Jacobian(inputs);
//      numerical_Jacobian = nd.calculate_Jacobian(nn, &NeuralNetwork::calculate_outputs, inputs);

//      assert_true((Jacobian-numerical_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
//   }

   // Conditions test

//   mmlp.set(1, 1, 1);

//   mmlp.initialize_parameters(0.0);
//   inputs.set(1, 0.0);
//   Jacobian = mmlp.calculate_Jacobian(inputs);

//   assert_true(Jacobian == 0.0, LOG);    

   // Conditions test

   // Lower and upper bounds test

   // Probabilistic postprocessing test
*/
}


void NeuralNetworkTest::test_calculate_Jacobian_data()
{
   message += "test_calculate_Jacobian_data\n";
}


// @todo

void NeuralNetworkTest::test_calculate_parameters_Jacobian()
{
   message += "test_calculate_parameters_Jacobian\n";

   NumericalDifferentiation nd;

   NeuralNetwork nn;

   Vector<size_t> multilayer_perceptron_architecture;
//   size_t outputs_number;

//   size_t parameters_number;
  
   Vector<double> inputs;

   Vector<double> parameters;
   Matrix<double> parameters_Jacobian;
   Matrix<double> numerical_parameters_Jacobian;

   // Test

   nn.set(1, 1); 

//   outputs_number = nn.get_outputs_number();

//   parameters_number = nn.get_parameters_number();

   nn.initialize_parameters(0.0);

   inputs.set(1, 0.0);

//   parameters_Jacobian = nn.calculate_parameters_Jacobian(inputs);

//   assert_true(parameters_Jacobian.get_rows_number() == outputs_number, LOG);
//   assert_true(parameters_Jacobian.get_columns_number() == parameters_number, LOG);

   if(numerical_differentiation_tests)
   {
//      parameters = nn.get_parameters();

//      numerical_parameters_Jacobian = nd.calculate_Jacobian(nn, &NeuralNetwork::calculate_parameters_output, inputs, parameters);

//      assert_true((parameters_Jacobian-numerical_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
//      nn.set(2, 1); 

//      nn.initialize_parameters(0.0);

//      inputs.set(2, 0.0);

//      parameters_Jacobian = nn.calculate_parameters_Jacobian(inputs);

//      parameters = nn.get_parameters();
//      numerical_parameters_Jacobian = nd.calculate_Jacobian(nn, &NeuralNetwork::calculate_parameters_output, inputs, parameters);

 //     assert_true((parameters_Jacobian-numerical_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
//      nn.set(1, 2); 

//      nn.initialize_parameters(0.0);

//      inputs.set(1, 0.0);

//      parameters_Jacobian = nn.calculate_parameters_Jacobian(inputs);

//      parameters = nn.get_parameters();
//      numerical_parameters_Jacobian = nd.calculate_Jacobian(nn, &NeuralNetwork::calculate_parameters_output, inputs, parameters);

//      assert_true((parameters_Jacobian-numerical_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
//      nn.set(2, 3); 

//      nn.initialize_parameters(0.0);

//      inputs.set(2, 0.0);

//      parameters_Jacobian = nn.calculate_parameters_Jacobian(inputs);

//      parameters = nn.get_parameters();
//      numerical_parameters_Jacobian = nd.calculate_Jacobian(nn, &NeuralNetwork::calculate_parameters_output, inputs, parameters);

//      assert_true((parameters_Jacobian-numerical_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

//   nn.set(3, 4, 2); 

//   outputs_number = nn.get_outputs_number();
//   parameters_number = nn.get_parameters_number();

//   inputs.set(3, 3.1415927);
   
//   parameters_Jacobian = nn.calculate_parameters_Jacobian(inputs);

//   assert_true(parameters_Jacobian.get_rows_number() == outputs_number, LOG);
//   assert_true(parameters_Jacobian.get_columns_number() == parameters_number, LOG);

   if(numerical_differentiation_tests)
   {
//      parameters = nn.get_parameters();
//      numerical_parameters_Jacobian = nd.calculate_Jacobian(nn, &NeuralNetwork::calculate_parameters_output, inputs, parameters);

//      assert_true((parameters_Jacobian-numerical_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

//   multilayer_perceptron_architecture.set(12, -3, 2);

//   nn.set(7, multilayer_perceptron_architecture);

//   outputs_number = nn.get_outputs_number();
//   parameters_number = nn.get_parameters_number();

//   inputs.set(7, 3.1415927);
   
//   parameters_Jacobian = nn.calculate_parameters_Jacobian(inputs);

//   assert_true(parameters_Jacobian.get_rows_number() == outputs_number, LOG);
//   assert_true(parameters_Jacobian.get_columns_number() == parameters_number, LOG);

   if(numerical_differentiation_tests)
   {
//    parameters = nn.get_parameters();
//      numerical_parameters_Jacobian = nd.calculate_Jacobian(nn, &NeuralNetwork::calculate_parameters_output, inputs, parameters);

//     assert_true((parameters_Jacobian-numerical_parameters_Jacobian).calculate_absolute_value() < 1.0e-5, LOG);
   }

}


// @todo

void NeuralNetworkTest::test_calculate_parameters_Jacobian_data()
{
   message += "test_calculate_parameters_Jacobian_data\n";
}


void NeuralNetworkTest::test_to_XML()
{
   message += "test_to_XML\n";

   NeuralNetwork  nn;

   tinyxml2::XMLDocument* document;
   
   // Test
   
   document = nn.to_XML();

   assert_true(document != nullptr, LOG);

   delete document;

   // Test

//   nn.initialize_random();

   document = nn.to_XML();

   assert_true(document != nullptr, LOG);

   delete document;
}


void NeuralNetworkTest::test_from_XML()
{
   message += "test_from_XML\n";

   NeuralNetwork  nn;

   tinyxml2::XMLDocument* document;
   
   // Test

//   nn.initialize_random();

   document = nn.to_XML();

   nn.from_XML(*document);

   delete document;
}


void NeuralNetworkTest::test_print()
{
   message += "test_print\n";

   // Empty neural network
 
   NeuralNetwork nn1;
   //nn1.print();

   // Only network architecture

   NeuralNetwork nn2(2, 4, 3);
   //nn2.print();

   //nn4.print();

   //nn5.print();

   // Units

   // Description
}


void NeuralNetworkTest::test_save()
{
   message += "test_save\n";

   string file_name = "../data/neural_network.xml";

   NeuralNetwork nn;

   // Empty multilayer perceptron
 
   nn.set();
   nn.save(file_name);

   // Only network architecture

   nn.set(2, 4, 3);
   nn.save(file_name);

   nn.set(1, 1, 1);

   nn.save(file_name);
}


void NeuralNetworkTest::test_load()
{
   message += "test_load\n";

   string file_name = "../data/neural_network.xml";

   // Empty neural network

   NeuralNetwork nn1;
   nn1.save(file_name);
   nn1.load(file_name);
}


// @todo

void NeuralNetworkTest::test_write_expression()
{
   message += "test_write_expression\n";

   NeuralNetwork nn;
   string expression;

   // Test

//   expression = nn.write_expression();

   // Test

   nn.set(1, 1, 1);
   nn.initialize_parameters(-1.0);
//   expression = nn.write_expression();

   // Test

   nn.set(2, 1, 1);
   nn.initialize_parameters(-1.0);
//   expression = nn.write_expression();

   // Test

   nn.set(1, 2, 1);
   nn.initialize_parameters(-1.0);
//   expression = nn.write_expression();

   // Test

   nn.set(1, 1, 2);
   nn.initialize_parameters(-1.0);
//   expression = nn.write_expression();

   // Test

   nn.set(2, 2, 2);
   nn.initialize_parameters(-1.0);
//   expression = nn.write_expression();

}


void NeuralNetworkTest::test_get_Hinton_diagram_XML()
{
   message += "test_get_Hinton_diagram_XML\n";
}


void NeuralNetworkTest::test_save_Hinton_diagram()
{
   message += "test_save_Hinton_diagram\n";
}


void NeuralNetworkTest::run_test_case()
{
   message += "Running neural network test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   test_get_inputs_pointer();
   test_get_outputs_pointer();

   test_get_multilayer_perceptron_pointer();
   test_get_scaling_layer_pointer();
   test_get_unscaling_layer_pointer();   
   test_get_bounding_layer_pointer();
   test_get_probabilistic_layer_pointer();

   // Parameters methods

   test_get_parameters_number();
   test_get_parameters();   

   // Display messages

   test_get_display();

   // Set methods

   test_set();
   test_set_default();

   // Variables

   test_set_variables();

   // Variables statistics

   test_set_variables_statistics();

   // Parameters methods

   test_set_parameters();

   // Display messages

   test_set_display();

   // Growing and pruning

   test_prune_input();
   test_prune_output();

   // Neural network initialization methods

   test_initialize_random();

   // Parameters initialization methods

   test_initialize_parameters();
   test_randomize_parameters_uniform();
   test_randomize_parameters_normal();

   // Parameters norm 

   test_calculate_parameters_norm();

   // Output 

   test_calculate_outputs();
   test_calculate_outputs();

//   test_calculate_Jacobian();
//   test_calculate_Jacobian_data();

//   test_calculate_parameters_Jacobian();
//   test_calculate_parameters_Jacobian_data();

   // Expression methods

   test_write_expression();

   // Hinton diagram methods

   test_get_Hinton_diagram_XML();

   test_save_Hinton_diagram();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   test_print();
   test_save();

   test_load();

   message += "End of neural network test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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
