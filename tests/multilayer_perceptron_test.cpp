/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M U L T I L A Y E R   P E R C E P T R O N   T E S T   C L A S S                                            */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "multilayer_perceptron_test.h"

using namespace OpenNN;


// GENERAL CONSTRUCTOR

MultilayerPerceptronTest::MultilayerPerceptronTest() : UnitTesting()
{
}


// DESTRUCTOR

MultilayerPerceptronTest::~MultilayerPerceptronTest()
{
}


// METHODS


void MultilayerPerceptronTest::test_constructor()
{
   message += "test_constructor\n";

   Vector<size_t> architecture;

   // Default constructor

   MultilayerPerceptron n1;

   assert_true(n1.get_layers_number() == 0, LOG);

   // Network architecture constructor

   architecture.set(2);
   architecture[0] = 1;
   architecture[1] = 2;

   MultilayerPerceptron n2(architecture);

   assert_true(n2.get_inputs_number() == 1, LOG);
   assert_true(n2.get_layers_number() == 1, LOG);
   assert_true(n2.get_outputs_number() == 2, LOG);

   // One layer constructor 

   // Two layers constructor 

   MultilayerPerceptron n3(1, 2, 3);

   assert_true(n3.get_inputs_number() == 1, LOG);
   assert_true(n3.get_layers_number() == 2, LOG);
   assert_true(n3.get_outputs_number() == 3, LOG);

   // Copy constructor

}


void MultilayerPerceptronTest::test_destructor()
{
   message += "test_destructor\n";
}


void MultilayerPerceptronTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

   MultilayerPerceptron mlp_1;
   MultilayerPerceptron mlp_2 = mlp_1;

   assert_true(mlp_2.get_inputs_number() == 0, LOG);
   assert_true(mlp_2.get_layers_number() == 0, LOG);
   assert_true(mlp_2.get_outputs_number() == 0, LOG);

}


void MultilayerPerceptronTest::test_get_inputs_number()
{
   message += "test_get_inputs_number\n";

   MultilayerPerceptron n;

   // Test

   n.set();
   assert_true(n.get_inputs_number() == 0, LOG);

   // Test

   n.set(1, 1, 1);
   assert_true(n.get_inputs_number() == 1, LOG);
}


void MultilayerPerceptronTest::test_count_layers_perceptrons_number()
{
   message += "test_count_layers_perceptrons_number\n";

   MultilayerPerceptron n;
   Vector<size_t> architecture;

   // Test

   n.set();
   architecture = n.get_layers_perceptrons_numbers();
   assert_true(architecture.size() == 0, LOG);

   // Test

   n.set(1, 1, 1);
   architecture = n.get_layers_perceptrons_numbers();
   assert_true(architecture.size() == 2, LOG);
   assert_true(architecture[0] == 1, LOG);
   assert_true(architecture[1] == 1, LOG);

}


void MultilayerPerceptronTest::test_get_outputs_number()
{
   message += "test_get_outputs_number\n";

   MultilayerPerceptron n;

   assert_true(n.get_outputs_number() == 0, LOG);

}


void MultilayerPerceptronTest::test_get_perceptrons_number()
{
   message += "test_get_perceptrons_number\n";

   MultilayerPerceptron n;
   Vector<size_t> architecture;

   // Test

   n.set();
   assert_true(n.get_perceptrons_number() == 0, LOG);

   // Test

   n.set(1,1,1);
   assert_true(n.get_perceptrons_number() == 2, LOG);

   // Test

   n.set(1,2,3);
   assert_true(n.get_perceptrons_number() == 5, LOG);

   // Test

   architecture.set(3,1);
   n.set(architecture);
   assert_true(n.get_perceptrons_number() == 2, LOG);
}


void MultilayerPerceptronTest::test_count_cumulative_perceptrons_number()
{
   message += "test_count_cumulative_perceptrons_number\n";

   MultilayerPerceptron n;
   Vector<size_t> architecture;

   Vector<size_t> cumulative_neurons_number;

   // Test

   n.set();
   cumulative_neurons_number = n.count_cumulative_perceptrons_number();
   assert_true(cumulative_neurons_number.size() == 0, LOG);

   // Test

   n.set(1,1,1);

   cumulative_neurons_number = n.count_cumulative_perceptrons_number();

   assert_true(cumulative_neurons_number[0] == 1, LOG);
   assert_true(cumulative_neurons_number[1] == 2, LOG);

   // Test

   n.set(1,2,3);

   cumulative_neurons_number = n.count_cumulative_perceptrons_number();

   assert_true(cumulative_neurons_number[0] == 2, LOG);
   assert_true(cumulative_neurons_number[1] == 5, LOG);

   // Test

   architecture.set(3,1);
   n.set(architecture);

   cumulative_neurons_number = n.count_cumulative_perceptrons_number();

   assert_true(cumulative_neurons_number[0] == 1, LOG);
   assert_true(cumulative_neurons_number[1] == 2, LOG);

}


void MultilayerPerceptronTest::test_get_layers_activation_function()
{
   message += "test_get_layers_activation_function\n";

   MultilayerPerceptron n;

   Vector<PerceptronLayer::ActivationFunction> layers_activation_function;

   n.set_layers_activation_function(layers_activation_function);

   layers_activation_function = n.get_layers_activation_function();

   assert_true(layers_activation_function.size() == 0, LOG);
}


void MultilayerPerceptronTest::test_get_layers_activation_function_name()
{
   message += "test_get_layers_activation_function_name\n";
}


void MultilayerPerceptronTest::test_get_layers()
{
   message += "test_get_layers\n";

   MultilayerPerceptron n;

   Vector<PerceptronLayer> layers;
   
   layers = n.get_layers();

   assert_true(layers.size() == 0, LOG);
}


void MultilayerPerceptronTest::test_get_layer()
{
   message += "test_get_layer\n";

   MultilayerPerceptron n;
   Vector<size_t> architecture;

   // Test

   architecture.set(3);
   architecture[0] = 4;
   architecture[1] = 2;
   architecture[2] = 3;

   n.set(architecture);

   const PerceptronLayer& layer_0 = n.get_layer(0);
   assert_true(layer_0.get_perceptrons_number() == 2, LOG);

   const PerceptronLayer& layer_1 = n.get_layer(1);
   assert_true(layer_1.get_perceptrons_number() == 3, LOG);
}


void MultilayerPerceptronTest::test_get_layers_number()
{
   message += "test_get_layers_number\n";

   MultilayerPerceptron n;

   Vector<size_t> architecture;

   // Test

   n.set();
   assert_true(n.get_layers_number() == 0, LOG);

   // Test

   architecture.set(3);
   architecture[0] = 4;
   architecture[1] = 2;
   architecture[2] = 3;

   n.set(architecture);

   assert_true(n.get_layers_number() == 2, LOG);
}


void MultilayerPerceptronTest::test_get_parameters_number()
{
   message += "test_get_parameters_number\n";

   MultilayerPerceptron n;

   // Test

   n.set();
   assert_true(n.get_parameters_number() == 0, LOG);

   // Test

   n.set(2, 4, 3);
   assert_true(n.get_parameters_number() == 27, LOG);
}


void MultilayerPerceptronTest::test_get_cumulative_parameters_number()
{
   message += "test_get_cumulative_parameters_number\n";

   MultilayerPerceptron n;

   // Test
}


void MultilayerPerceptronTest::test_get_layers_parameters_number()
{   
   message += "test_get_layers_parameters_number\n";
 /*
   MultilayerPerceptron n;
   Vector<size_t> layers_parameters_number;
   Vector<size_t> architecture(3, 1);

   // Test

   n.set();
   layers_parameters_number = n.get_layers_parameters_number();

   assert_true(layers_parameters_number.size() == 0, LOG);

   // Test

   n.set(1, 1, 1);
   layers_parameters_number = n.get_layers_parameters_number();

   assert_true(layers_parameters_number.size() == 2, LOG);
   assert_true(layers_parameters_number[0] == 2, LOG);
   assert_true(layers_parameters_number[1] == 2, LOG);

   // Test

   architecture.set(4, 1);
   n.set(architecture);

   layers_parameters_number = n.get_layers_parameters_number();

   assert_true(layers_parameters_number.size() == 3, LOG);
   assert_true(layers_parameters_number[0] == 2, LOG);
   assert_true(layers_parameters_number[1] == 2, LOG);
   assert_true(layers_parameters_number[2] == 2, LOG);
   */
}


void MultilayerPerceptronTest::test_set()
{
   message += "test_set\n";
}


void MultilayerPerceptronTest::test_set_default()
{
   message += "test_set_default\n";
}


void MultilayerPerceptronTest::test_set_parameters()
{
	message += "test_set_parameters\n";

	MultilayerPerceptron n(1, 1, 1);

    size_t parameters_number = n.get_parameters_number();

	Vector<double> parameters(parameters_number);
	
	parameters.randomize_uniform();

	n.set_parameters(parameters);

    Vector<double> new_parameters = n.get_parameters();

    assert_true(new_parameters == parameters, LOG);
}


void MultilayerPerceptronTest::test_get_layers_biases()
{
   message += "test_get_layers_biases\n";

   MultilayerPerceptron n;
   Vector< Vector<double> > layers_biases;

   // Test

   n.set(1, 1, 1);
   n.initialize_parameters(0.0);

   layers_biases = n.get_layers_biases();

   assert_true(layers_biases.size() == 2, LOG);
   assert_true(layers_biases[0].size() == 1, LOG);
   assert_true(layers_biases[1].size() == 1, LOG);
}


void MultilayerPerceptronTest::test_get_layers_synaptic_weights()
{
   message += "test_get_layers_synaptic_weights\n";

   MultilayerPerceptron n;

   Vector< Matrix<double> > layers_synaptic_weights;

   // Test

   n.set(1, 1, 1);

   n.initialize_parameters(0.0);

   layers_synaptic_weights = n.get_layers_synaptic_weights();

   assert_true(layers_synaptic_weights.size() == 2, LOG);

   assert_true(layers_synaptic_weights[0].get_rows_number() == 1, LOG);
   assert_true(layers_synaptic_weights[0].get_columns_number() == 1, LOG);
   assert_true(layers_synaptic_weights[0] == 0.0, LOG);

   assert_true(layers_synaptic_weights[1].get_rows_number() == 1, LOG);
   assert_true(layers_synaptic_weights[1].get_columns_number() == 1, LOG);
   assert_true(layers_synaptic_weights[1] == 0.0, LOG);
}


void MultilayerPerceptronTest::test_get_layers_parameters()
{
   message += "test_get_layers_parameters\n";

   Vector<size_t> architecture;

   MultilayerPerceptron n;

   Vector< Vector<double> > layers_parameters;

   // Test

   architecture.set(3, 1);

   n.set(architecture);

//   layers_parameters = n.get_layers_parameters();
//   assert_true(layers_parameters.size() == 2, LOG);
}


void MultilayerPerceptronTest::test_get_parameter_indices()
{
   message += "test_get_parameter_indices\n";

   MultilayerPerceptron n;

   Vector<size_t> architecture;

   size_t parameters_number;

   Vector<double> parameters;

   Vector<size_t> parameter_indices(3);

   // Test

   n.set(1, 1);
   parameters_number = n.get_parameters_number();

   parameter_indices = n.get_parameter_indices(0);
   assert_true(parameter_indices[0] == 0, LOG);
   assert_true(parameter_indices[1] == 0, LOG);
   assert_true(parameter_indices[2] == 0, LOG);

   parameter_indices = n.get_parameter_indices(1);
   assert_true(parameter_indices[0] == 0, LOG);
   assert_true(parameter_indices[1] == 0, LOG);
   assert_true(parameter_indices[2] == 1, LOG);

   // Test 

   n.set(1, 1, 1);

   parameter_indices = n.get_parameter_indices(0);
   assert_true(parameter_indices[0] == 0, LOG);
   assert_true(parameter_indices[1] == 0, LOG);
   assert_true(parameter_indices[2] == 0, LOG);

   parameter_indices = n.get_parameter_indices(1);
   assert_true(parameter_indices[0] == 0, LOG);
   assert_true(parameter_indices[1] == 0, LOG);
   assert_true(parameter_indices[2] == 1, LOG);

   parameter_indices = n.get_parameter_indices(2);
   assert_true(parameter_indices[0] == 1, LOG);
   assert_true(parameter_indices[1] == 0, LOG);
   assert_true(parameter_indices[2] == 0, LOG);

   parameter_indices = n.get_parameter_indices(3);
   assert_true(parameter_indices[0] == 1, LOG);
   assert_true(parameter_indices[1] == 0, LOG);
   assert_true(parameter_indices[2] == 1, LOG);

   // Test

   n.set(2, 4, 3);
   parameters_number = n.get_parameters_number();

   parameter_indices = n.get_parameter_indices(0);
   assert_true(parameter_indices[0] == 0, LOG);
   assert_true(parameter_indices[1] == 0, LOG);
   assert_true(parameter_indices[2] == 0, LOG);

   parameter_indices = n.get_parameter_indices(parameters_number-1);
   assert_true(parameter_indices[0] == 1, LOG);
   assert_true(parameter_indices[1] == 2, LOG);
   assert_true(parameter_indices[2] == 4, LOG);

   // Test

   // Test

   n.set(2, 4, 3);

   parameters.set(27);

   parameters[0]  =  0.85;
   parameters[1]  = -0.04;
   parameters[2]  =  0.87;
   parameters[3]  = -0.25;
   parameters[4]  =  0.25;
   parameters[5]  = -0.27;
   parameters[6]  =  0.29;
   parameters[7]  = -0.57;
   parameters[8]  =  0.15;
   parameters[9]  = -0.77;
   parameters[10] =  0.96;
   parameters[11] = -0.48;
   parameters[12] =  0.08;
   parameters[13] = -0.06;
   parameters[14] =  0.26;
   parameters[15] = -0.15;
   parameters[16] =  0.96;
   parameters[17] = -0.33;
   parameters[18] =  0.63;
   parameters[19] = -0.32; 
   parameters[20] =  0.89;
   parameters[21] = -0.80;
   parameters[22] =  0.80; 
   parameters[23] = -0.03;
   parameters[24] =  0.32;
   parameters[25] =  0.06; 
   parameters[26] = -0.38;

   n.set_parameters(parameters);   

   parameter_indices = n.get_parameter_indices(0);
   assert_true(parameter_indices[0] == 0, LOG);
   assert_true(parameter_indices[1] == 0, LOG);
   assert_true(parameter_indices[2] == 0, LOG);

   parameter_indices = n.get_parameter_indices(13);
   assert_true(parameter_indices[0] == 1, LOG);
   assert_true(parameter_indices[1] == 0, LOG);
   assert_true(parameter_indices[2] == 1, LOG);

   parameter_indices = n.get_parameter_indices(26);
   assert_true(parameter_indices[0] == 1, LOG);
   assert_true(parameter_indices[1] == 2, LOG);
   assert_true(parameter_indices[2] == 4, LOG);

}


void MultilayerPerceptronTest::test_get_parameters_indices()
{
   message += "test_get_parameters_indices\n";

   MultilayerPerceptron n;

   Vector<size_t> architecture;

   size_t parameters_number;

   Vector<double> parameters;

   Matrix<size_t> parameters_indices;

   // Test

   n.set(1, 1);
   parameters_number = n.get_parameters_number();

   parameters_indices = n.get_parameters_indices();

   assert_true(parameters_indices.get_rows_number() == parameters_number, LOG);
   assert_true(parameters_indices.get_columns_number() == 3, LOG);

   assert_true(parameters_indices(0,0) == 0, LOG);
   assert_true(parameters_indices(0,1) == 0, LOG);
   assert_true(parameters_indices(0,2) == 0, LOG);

   assert_true(parameters_indices(1,0) == 0, LOG);
   assert_true(parameters_indices(1,1) == 0, LOG);
   assert_true(parameters_indices(1,2) == 1, LOG);

   // Test 

   n.set(1, 1, 1);
   parameters_number = n.get_parameters_number();

   parameters_indices = n.get_parameters_indices();

   assert_true(parameters_indices.get_rows_number() == parameters_number, LOG);
   assert_true(parameters_indices.get_columns_number() == 3, LOG);

   assert_true(parameters_indices(0,0) == 0, LOG);
   assert_true(parameters_indices(0,1) == 0, LOG);
   assert_true(parameters_indices(0,2) == 0, LOG);

   assert_true(parameters_indices(1,0) == 0, LOG);
   assert_true(parameters_indices(1,1) == 0, LOG);
   assert_true(parameters_indices(1,2) == 1, LOG);

   assert_true(parameters_indices(2,0) == 1, LOG);
   assert_true(parameters_indices(2,1) == 0, LOG);
   assert_true(parameters_indices(2,2) == 0, LOG);

   assert_true(parameters_indices(3,0) == 1, LOG);
   assert_true(parameters_indices(3,1) == 0, LOG);
   assert_true(parameters_indices(3,2) == 1, LOG);
}



void MultilayerPerceptronTest::test_set_layers_biases()
{
   message += "test_set_layers_biases\n";

   MultilayerPerceptron n;
   Vector< Vector<double> > true_layers_biases;
   Vector< Vector<double> > layers_biases;

   // Test

   n.set(1, 1);

   true_layers_biases.set(1);
   true_layers_biases[0].set(1, 0.0);

   n.set_layers_biases(true_layers_biases);

   layers_biases = n.get_layers_biases();

   assert_true(layers_biases[0] == true_layers_biases[0], LOG);
}


void MultilayerPerceptronTest::test_set_layers_synaptic_weights()
{
   message += "test_set_layers_synaptic_weights\n";

   MultilayerPerceptron n;
   Vector< Matrix<double> > layers_synaptic_weights;

   // Test

   n.set(1, 1, 1);
   n.initialize_synaptic_weights(0.0);

   layers_synaptic_weights = n.get_layers_synaptic_weights();

   assert_true(layers_synaptic_weights.size() == 2, LOG);
   assert_true(layers_synaptic_weights[0].get_rows_number() == 1, LOG);
   assert_true(layers_synaptic_weights[0].get_columns_number() == 1, LOG);
   assert_true(layers_synaptic_weights[0] == 0.0, LOG);
   assert_true(layers_synaptic_weights[1].get_rows_number() == 1, LOG);
   assert_true(layers_synaptic_weights[1].get_columns_number() == 1, LOG);
   assert_true(layers_synaptic_weights[1] == 0.0, LOG);
}


void MultilayerPerceptronTest::test_set_layers_parameters()
{
   message += "test_set_layers_parameters\n";

   MultilayerPerceptron n;

   Vector< Vector<double> > layers_parameters;
   Vector< Vector<double> > new_layers_parameters;

   // Test

   n.set(1, 1, 1);

   new_layers_parameters.set(2);

   new_layers_parameters[0].set(2, 0.0);
   new_layers_parameters[1].set(2, 1.0);

   n.set_layers_parameters(new_layers_parameters);

//   layers_parameters = n.get_layers_parameters();

//   assert_true(layers_parameters[0] == new_layers_parameters[0], LOG);
}


void MultilayerPerceptronTest::test_get_display()
{
   message += "test_get_display\n";
}


void MultilayerPerceptronTest::test_set_layers_activation_function()
{
   message += "test_set_layers_activation_function\n";
}


void MultilayerPerceptronTest::test_set_display()
{
   message += "test_set_display\n";
}


void MultilayerPerceptronTest::test_is_empty()
{
   message += "test_is_empty\n";

   MultilayerPerceptron mlp;

   // Test

   mlp.set();

   assert_true(mlp.is_empty(), LOG);

   // Test

   mlp.set(1, 1);

   assert_true(!mlp.is_empty(), LOG);
}


void MultilayerPerceptronTest::test_grow_input()
{
   message += "test_grow_input\n";

//   MultilayerPerceptron mlp;

   // Test

//   mlp.set();

//   mlp.grow_input();

//   assert_true(mlp.is_empty(), LOG);
}


void MultilayerPerceptronTest::test_grow_layer()
{
   message += "test_grow_layer\n";

   MultilayerPerceptron mlp;

   // Test
}


void MultilayerPerceptronTest::test_prune_input()
{
   message += "test_prune_input\n";

   MultilayerPerceptron mlp;

   // Test

   mlp.set(1, 1);

   mlp.prune_input(0);

   assert_true(mlp.get_layers_number() == 1, LOG);
   assert_true(mlp.get_inputs_number() == 0, LOG);

   // Test

   mlp.set(2, 2, 2);

   mlp.prune_input(0);

   assert_true(mlp.get_layers_number() == 2, LOG);
   assert_true(mlp.get_inputs_number() == 1, LOG);
}


void MultilayerPerceptronTest::test_prune_output()
{
   message += "test_prune_output\n";

   MultilayerPerceptron mlp;

   // Test

   mlp.set(1, 1);

   mlp.prune_output(0);

   assert_true(mlp.get_layers_number() == 1, LOG);
   assert_true(mlp.get_inputs_number() == 0, LOG);
   assert_true(mlp.get_outputs_number() == 0, LOG);

   // Test

   mlp.set(2, 2, 2);

   mlp.prune_output(1);

   assert_true(mlp.get_inputs_number() == 2, LOG);
   assert_true(mlp.get_layers_number() == 2, LOG);
   assert_true(mlp.get_outputs_number() == 1, LOG);
}


void MultilayerPerceptronTest::test_prune_layer()
{
   message += "test_prune_layer\n";

   MultilayerPerceptron mlp;

   // Test
}


void MultilayerPerceptronTest::test_get_parameters()   
{
   message += "test_get_parameters\n";

   MultilayerPerceptron mlp;
   Vector<double> parameters;

   // Test

   mlp.set();

   parameters = mlp.get_parameters();

   assert_true(parameters.size() == 0, LOG);

   // Test

   mlp.set(1, 1, 1);
   mlp.initialize_parameters(0.0);
   parameters = mlp.get_parameters();

   assert_true(parameters.size() == 4, LOG);
   assert_true(parameters == 0.0, LOG);

   // Test

   mlp.set(1, 1, 1);
   mlp.initialize_parameters(0.0);
   parameters = mlp.get_parameters();

   assert_true(parameters.size() == 4, LOG);
   assert_true(parameters == 0.0, LOG);

   // Test

   mlp.initialize_random();

   parameters = mlp.get_parameters();
   mlp.set_parameters(parameters);
   assert_true(parameters == mlp.get_parameters(), LOG);
}


void MultilayerPerceptronTest::test_initialize_random()
{
   message += "test_initialize_random\n";

   MultilayerPerceptron n;

   size_t inputs_number;
   size_t layers_number;
   Vector<size_t> architecture;

   // Test

   n.initialize_random();

   inputs_number = n.get_inputs_number();

   assert_true(inputs_number >= 1 && inputs_number <= 10, LOG); 

   layers_number = n.get_layers_number();

   assert_true(layers_number >= 1 && layers_number <= 10, LOG); 

   architecture = n.get_layers_perceptrons_numbers();

   assert_true(architecture >= 1 && architecture <= 10, LOG); 

}


void MultilayerPerceptronTest::test_initialize_parameters()
{
   message += "test_initialize_parameters\n";

   MultilayerPerceptron n;

   Vector<double> parameters;
   n.set(1,1,1);
   n.initialize_parameters(0.0);

   parameters = n.get_parameters();

   assert_true(parameters == 0.0, LOG);

}


void MultilayerPerceptronTest::test_initialize_biases()
{
   message += "test_initialize_biases\n";
}


void MultilayerPerceptronTest::test_initialize_synaptic_weights()
{
   message += "test_initialize_synaptic_weights\n";
}


void MultilayerPerceptronTest::test_randomize_parameters_uniform()
{
   message += "test_randomize_parameters_uniform\n";

   MultilayerPerceptron n;
   Vector<double> parameters;

   // Test

   n.set(1,1,1);      
   n.randomize_parameters_uniform();
   parameters = n.get_parameters();
   assert_true(parameters >= -1.0, LOG);
   assert_true(parameters <=  1.0, LOG);   
}


void MultilayerPerceptronTest::test_randomize_parameters_normal()
{
   message += "test_randomize_parameters_normal\n";

   MultilayerPerceptron n;
   Vector<double> parameters;

   // Test

   n.set(1,1,1);      
   n.randomize_parameters_normal(1.0, 0.0);
   parameters = n.get_parameters();
   assert_true(parameters == 1.0, LOG);
}


void MultilayerPerceptronTest::test_calculate_parameters_norm()
{
   message += "test_calculate_parameters_norm\n";

   MultilayerPerceptron n;

   Vector< Vector<double> > layers_biases;
   Vector< Matrix<double> > layers_synaptic_weights;

   Vector<double> parameters;

   double parameters_norm;

   // Test 

   n.set();

   parameters_norm = n.calculate_parameters_norm();

   assert_true(parameters_norm == 0.0, LOG);

   // Test

   n.set(1, 1, 1);
   n.initialize_parameters(0.0);

   parameters_norm = n.calculate_parameters_norm();

   assert_true(parameters_norm == 0.0, LOG);

   // Test

   n.set(2, 4, 3);

   layers_biases.set(2);

   layers_biases[0].set(4);
   layers_biases[0][0] =  0.85;
   layers_biases[0][1] = -0.25;
   layers_biases[0][2] =  0.29;
   layers_biases[0][3] = -0.77;

   layers_biases[1].set(3);
   layers_biases[1][0] =  0.08;
   layers_biases[1][1] = -0.33;
   layers_biases[1][2] =  0.80;

   n.set_layers_biases(layers_biases);

   layers_synaptic_weights.set(2);

   layers_synaptic_weights[0].set(2, 4);

//   layers_synaptic_weights[0](0,0) = -0.04;
//   layers_synaptic_weights[0](0,1) =  0.87;

//   layers_synaptic_weights[0](1,0) =  0.25;
//   layers_synaptic_weights[0](1,1) = -0.27;

//   layers_synaptic_weights[0](2,0) = -0.57;
//   layers_synaptic_weights[0](2,1) =  0.15;

//   layers_synaptic_weights[0](3,0) =  0.96;
//   layers_synaptic_weights[0](3,1) = -0.48;

   layers_synaptic_weights[1].set(4, 3);

//   layers_synaptic_weights[1](0,0) = -0.06;
//   layers_synaptic_weights[1](0,1) =  0.26;
//   layers_synaptic_weights[1](0,2) = -0.15;
//   layers_synaptic_weights[1](0,3) =  0.96;

//   layers_synaptic_weights[1](1,0) =  0.63;
//   layers_synaptic_weights[1](1,1) = -0.32;
//   layers_synaptic_weights[1](1,2) =  0.89;
//   layers_synaptic_weights[1](1,3) = -0.80;

//   layers_synaptic_weights[1](2,0) = -0.03;
//   layers_synaptic_weights[1](2,1) =  0.32;
//   layers_synaptic_weights[1](2,2) =  0.06;
//   layers_synaptic_weights[1](2,3) = -0.38;

   n.set_layers_synaptic_weights(layers_synaptic_weights);

   parameters = n.get_parameters();

   parameters_norm = n.calculate_parameters_norm();

   assert_true(fabs(parameters_norm - parameters.calculate_L2_norm()) < 1.0e-6, LOG);
}


void MultilayerPerceptronTest::test_calculate_outputs()
{
   message += "test_calculate_outputs\n";
/*
   MultilayerPerceptron mlp;

   Vector<double> parameters;

   Vector<double> inputs;
   Vector<double> outputs;
   Vector<double> potential_outputs;

   Vector<double> hidden_layer_output;
   Vector<double> output_layer_output;

   // Test

   mlp.set(2, 4, 3);

   parameters.set(27);

   parameters[0]  =  0.85;
   parameters[1]  = -0.04;
   parameters[2]  =  0.87;
   parameters[3]  = -0.25;
   parameters[4]  =  0.25;
   parameters[5]  = -0.27;
   parameters[6]  =  0.29;
   parameters[7]  = -0.57;
   parameters[8]  =  0.15;
   parameters[9]  = -0.77;
   parameters[10] =  0.96;
   parameters[11] = -0.48;
   parameters[12] =  0.08;
   parameters[13] = -0.06;
   parameters[14] =  0.26;
   parameters[15] = -0.15;
   parameters[16] =  0.96;
   parameters[17] = -0.33;
   parameters[18] =  0.63;
   parameters[19] = -0.32; 
   parameters[20] =  0.89;
   parameters[21] = -0.80;
   parameters[22] =  0.80; 
   parameters[23] = -0.03;
   parameters[24] =  0.32;
   parameters[25] =  0.06; 
   parameters[26] = -0.38;

   mlp.set_parameters(parameters);

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

   const PerceptronLayer& hidden_layer = mlp.get_layer(0);

   hidden_layer_output = hidden_layer.calculate_outputs(inputs);

   const PerceptronLayer& output_layer = mlp.get_layer(1);

   output_layer_output = output_layer.calculate_outputs(hidden_layer_output);

   outputs = mlp.calculate_outputs(inputs);
   assert_true(outputs.size() == 3, LOG);
   assert_true(outputs == output_layer_output, LOG);

   // Test

   inputs.set(1, 3.0);

   mlp.set(1, 1);

   mlp.initialize_parameters(2.0);

   outputs = mlp.calculate_outputs(inputs);

   parameters.set(2, 1.0);

   potential_outputs = mlp.calculate_outputs(inputs, parameters);

   assert_true(outputs != potential_outputs, LOG);

   // Test

   mlp.set(1, 1);

   inputs.set(1);
   inputs.randomize_normal();

   parameters = mlp.get_parameters();

   assert_true(mlp.calculate_outputs(inputs) == mlp.calculate_outputs(inputs, parameters), LOG);

   // Test

   inputs.set(1, 3.0);

   mlp.set(1, 2, 1);

   mlp.initialize_parameters(2.0);

   outputs = mlp.calculate_outputs(inputs);

   parameters = mlp.get_parameters();

   potential_outputs = mlp.calculate_outputs(inputs, parameters*2.0);

   assert_true(outputs != potential_outputs, LOG);

   // Test

   mlp.set(2, 2);

   inputs.set(2);
   inputs.randomize_normal();

   parameters = mlp.get_parameters();

//   const MultilayerPerceptron::LayersParameters layers_parameters = mlp.get_layers_parameters(parameters);

//   cout << "parameters: " << parameters << endl;

//   cout << "Weights: " << layers_parameters.synaptic_weights << endl;
//   cout << "Biases: " << layers_parameters.biases << endl;

//   cout << "mlp weights: " << mlp.get_layers_synaptic_weights() << endl;
//   cout << "mlp biases: " << mlp.get_layers_biases() << endl;

//   assert_true(mlp.calculate_outputs(inputs) == mlp.calculate_outputs(inputs, parameters), LOG);
*/
}


void MultilayerPerceptronTest::test_calculate_Jacobian()
{
   message += "test_calculate_Jacobian\n";
/*
   NumericalDifferentiation nd;

   MultilayerPerceptron mlp;

   size_t inputs_number;
   Vector<size_t> architecture;
   Vector<double> parameters;

   Vector<double> inputs;

   Matrix<double> Jacobian;
   Matrix<double> numerical_Jacobian;

   Vector<double> hidden_layer_output;

   Matrix<double> hidden_layer_Jacobian;
   Matrix<double> output_layer_Jacobian;

   // Test

   mlp.set(1,1,1);
   mlp.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   Jacobian = mlp.calculate_Jacobian(inputs);
   assert_true(Jacobian == 0.0, LOG);    

   // Test

   mlp.set(3,4,2);
   mlp.initialize_parameters(0.0);
   inputs.set(3, 0.0);
   Jacobian = mlp.calculate_Jacobian(inputs);
   assert_true(Jacobian == 0.0, LOG);    

   // Test

   mlp.set(2, 4, 3);

   parameters.set(27);

   parameters[0]  =  0.85;
   parameters[1]  = -0.04;
   parameters[2]  =  0.87;
   parameters[3]  = -0.25;
   parameters[4]  =  0.25;
   parameters[5]  = -0.27;
   parameters[6]  =  0.29;
   parameters[7]  = -0.57;
   parameters[8]  =  0.15;
   parameters[9]  = -0.77;
   parameters[10] =  0.96;
   parameters[11] = -0.48;
   parameters[12] =  0.08;
   parameters[13] = -0.06;
   parameters[14] =  0.26;
   parameters[15] = -0.15;
   parameters[16] =  0.96;
   parameters[17] = -0.33;
   parameters[18] =  0.63;
   parameters[19] = -0.32; 
   parameters[20] =  0.89;
   parameters[21] = -0.80;
   parameters[22] =  0.80; 
   parameters[23] = -0.03;
   parameters[24] =  0.32;
   parameters[25] =  0.06; 
   parameters[26] = -0.38;

   mlp.set_parameters(parameters);

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

   const PerceptronLayer& hidden_layer = mlp.get_layer(0);

   hidden_layer_output = hidden_layer.calculate_outputs(inputs);
   hidden_layer_Jacobian = hidden_layer.calculate_Jacobian(inputs);

   const PerceptronLayer& output_layer = mlp.get_layer(1);

   output_layer_Jacobian = output_layer.calculate_Jacobian(hidden_layer_output);

   Jacobian = mlp.calculate_Jacobian(inputs);
   assert_true(Jacobian.get_rows_number() == 3, LOG);
   assert_true(Jacobian.get_columns_number() == 2, LOG);
   assert_true((Jacobian - output_layer_Jacobian.dot(hidden_layer_Jacobian)).calculate_absolute_value() < 1.0e-3, LOG);

   if(numerical_differentiation_tests)
   {
//      numerical_Jacobian = mlp.calculate_Jacobian(mlp, &MultilayerPerceptron::calculate_outputs, inputs);
//      assert_true((Jacobian-numerical_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   for(size_t i = 0; i < random_tests_number; i++)
   {
      mlp.initialize_random();

      inputs_number = mlp.get_inputs_number();

	  inputs.set(inputs_number);
	  inputs.randomize_uniform();

      Jacobian = mlp.calculate_Jacobian(inputs);

      if(numerical_differentiation_tests)
      {
         numerical_Jacobian = nd.calculate_Jacobian(mlp, &MultilayerPerceptron::calculate_outputs, inputs);
         assert_true((Jacobian-numerical_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
      }
   }
*/
}


// @todo

void MultilayerPerceptronTest::test_calculate_Hessian()
{
   message += "test_calculate_Hessian\n";
/*
   cout.precision(2);

   NumericalDifferentiation nd;

   MultilayerPerceptron n;

//   size_t inputs_number;
   Vector<size_t> architecture;
//   size_t outputs_number;

   Vector<double> parameters;

   Vector<double> inputs;
   Matrix<double> Jacobian;
   Vector< Matrix<double> > Hessian;
   Vector< Matrix<double> > numerical_Hessian;

   Matrix<double> Hessian;

   Vector<double> hidden_layer_output;

   Matrix<double> hidden_layer_Jacobian;
   Vector< Matrix<double> > hidden_layer_Hessian;

   Matrix<double> output_layer_Jacobian;
   Vector< Matrix<double> > output_layer_Hessian;

   // Test

   n.set(1, 1);

   n.initialize_parameters(0.0);

   inputs.set(1, 0.0);

//   Hessian = n.calculate_Hessian(inputs);

//   assert_true(Hessian.size() == 1, LOG);    
//   assert_true(Hessian[0].get_rows_number() == 1, LOG);    
//   assert_true(Hessian[0].get_columns_number() == 1, LOG);    
//   assert_true(Hessian[0] == 0.0, LOG);    

   // Test

   n.set(1, 1, 1);

   inputs.set(1, 0.0);

//   Hessian = n.calculate_Hessian(inputs);
//   assert_true(Hessian.size() == 1, LOG);    

//   assert_true(Hessian[0].get_rows_number() == 1, LOG);    
//   assert_true(Hessian[0].get_columns_number() == 1, LOG);    

   if(numerical_differentiation_tests)
   {
//      numerical_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_outputs, inputs);

//      assert_true((Hessian[0]-numerical_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(3, 2, 1);

   inputs.set(3, 3.1415927);

//   Hessian = n.calculate_Hessian(inputs);

//   assert_true(Hessian.size() == 1, LOG);    
//   assert_true(Hessian[0].is_symmetric(), LOG);    
//   assert_true(Hessian[0].get_rows_number() == 3, LOG);    
//   assert_true(Hessian[0].get_columns_number() == 3, LOG);    

   if(numerical_differentiation_tests)
   {
//      numerical_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_outputs, inputs);

//    assert_true((Hessian[0]-numerical_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(3, 4, 2);

   inputs.set(3, 3.1415927);

//   Hessian = n.calculate_Hessian(inputs);

//   assert_true(Hessian.size() == 2, LOG);    
//   assert_true(Hessian[0].get_rows_number() == 3, LOG);    
//   assert_true(Hessian[0].get_columns_number() == 3, LOG);    

//   assert_true(Hessian[1].get_rows_number() == 3, LOG);    
//   assert_true(Hessian[1].get_columns_number() == 3, LOG);    

   if(numerical_differentiation_tests)
   {
//      numerical_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_outputs, inputs);

//    assert_true((Hessian[0]-numerical_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//    assert_true((Hessian[1]-numerical_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   architecture.set(5, 1);

   n.set(architecture);

   inputs.set(1, 3.1415927);

//   Hessian = n.calculate_Hessian(inputs);
//   assert_true(Hessian.size() == 1, LOG);    

//   assert_true(Hessian[0].get_rows_number() == 1, LOG);    
//   assert_true(Hessian[0].get_columns_number() == 1, LOG);    

   if(numerical_differentiation_tests)
   {
//      numerical_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_outputs, inputs);

//    assert_true((Hessian[0]-numerical_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
      architecture.set(3, 2, 9);

      n.set(architecture);

//      inputs.set(4);
//      inputs[0] = -1.025;
//      inputs[1] = 0.575;
//      inputs[2] = -7.256;
//      inputs[3] = 3.012;

//    Hessian = n.calculate_Hessian(inputs);
//    numerical_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_outputs, inputs);

//    assert_true((Hessian[0]-numerical_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(2, 4, 3);

   parameters.set(27);

   parameters[0]  =  0.85;
   parameters[1]  = -0.04;
   parameters[2]  =  0.87;
   parameters[3]  = -0.25;
   parameters[4]  =  0.25;
   parameters[5]  = -0.27;
   parameters[6]  =  0.29;
   parameters[7]  = -0.57;
   parameters[8]  =  0.15;
   parameters[9]  = -0.77;
   parameters[10] =  0.96;
   parameters[11] = -0.48;
   parameters[12] =  0.08;
   parameters[13] = -0.06;
   parameters[14] =  0.26;
   parameters[15] = -0.15;
   parameters[16] =  0.96;
   parameters[17] = -0.33;
   parameters[18] =  0.63;
   parameters[19] = -0.32; 
   parameters[20] =  0.89;
   parameters[21] = -0.80;
   parameters[22] =  0.80; 
   parameters[23] = -0.03;
   parameters[24] =  0.32;
   parameters[25] =  0.06; 
   parameters[26] = -0.38;

   n.set_parameters(parameters);   

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

//   hidden_layer_output = n.calculate_layer_output(0, inputs);
//   hidden_layer_Jacobian = n.calculate_layer_Jacobian(0, inputs);
//   hidden_layer_Hessian = n.calculate_layer_Hessian(0, inputs);

//   output_layer_Jacobian = n.calculate_layer_Jacobian(1, hidden_layer_output);
//   output_layer_Hessian = n.calculate_layer_Hessian(1, hidden_layer_output);

//   Jacobian = n.calculate_Jacobian(inputs);
//   Hessian = n.calculate_Hessian(inputs);

//   assert_true(Hessian.size() == 3, LOG);
//   assert_true(Hessian[0].get_rows_number() == 2, LOG);
//   assert_true(Hessian[0].get_columns_number() == 2, LOG);
//   assert_true(Hessian[1].get_rows_number() == 2, LOG);
//   assert_true(Hessian[1].get_columns_number() == 2, LOG);
//   assert_true(Hessian[2].get_rows_number() == 2, LOG);
//   assert_true(Hessian[2].get_columns_number() == 2, LOG);

//   Hessian = hidden_layer_Jacobian.calculate_transpose().dot(output_layer_Hessian[0]).dot(hidden_layer_Jacobian)
//	       + hidden_layer_Hessian[0]*output_layer_Jacobian(0,0)
//	       + hidden_layer_Hessian[1]*output_layer_Jacobian(0,1)
//	       + hidden_layer_Hessian[2]*output_layer_Jacobian(0,2)
//	       + hidden_layer_Hessian[3]*output_layer_Jacobian(0,3);

//   assert_true(Hessian[0] == Hessian, LOG);

//   Hessian = hidden_layer_Jacobian.calculate_transpose().dot(output_layer_Hessian[1]).dot(hidden_layer_Jacobian)
//	       + hidden_layer_Hessian[0]*output_layer_Jacobian(1,0)
//	       + hidden_layer_Hessian[1]*output_layer_Jacobian(1,1)
//	       + hidden_layer_Hessian[2]*output_layer_Jacobian(1,2)
//	       + hidden_layer_Hessian[3]*output_layer_Jacobian(1,3);

//   assert_true(Hessian[1] == Hessian, LOG);

//   Hessian = hidden_layer_Jacobian.calculate_transpose().dot(output_layer_Hessian[2]).dot(hidden_layer_Jacobian)
//	       + hidden_layer_Hessian[0]*output_layer_Jacobian(2,0)
//	       + hidden_layer_Hessian[1]*output_layer_Jacobian(2,1)
//	       + hidden_layer_Hessian[2]*output_layer_Jacobian(2,2)
//	       + hidden_layer_Hessian[3]*output_layer_Jacobian(2,3);

//   assert_true(Hessian[2] == Hessian, LOG);

//   if(numerical_differentiation_tests)
//   {
//      numerical_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_outputs, inputs);

//    assert_true((Hessian[0]-numerical_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//    assert_true((Hessian[1]-numerical_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
//   }

   // Test

//   for(size_t i = 0; i < random_tests_number; i++)
//   {
//      n.initialize_random();

//	  inputs_number = n.get_inputs_number();

//	  inputs.set(inputs_number);
//	  inputs.randomize_uniform();

//      Hessian = n.calculate_Hessian(inputs);

//      if(numerical_differentiation_tests)
//      {
//         numerical_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_outputs, inputs);

//	     outputs_number = n.get_outputs_number();

//	     for(size_t j = 0; j < outputs_number; j++)
//	     {
//         assert_true((Hessian[j]-numerical_Hessian[j]).calculate_absolute_value() < 1.0e-3, LOG);
//	     }
//      }
//   }
*/
}


void MultilayerPerceptronTest::test_calculate_layer_combination_combination()
{
   message += "test_calculate_layer_combination_combination\n";   
/*
   MultilayerPerceptron n;

   Vector<double> parameters;

   Vector<double> inputs;

   Vector<double> hidden_layer_combination;
   Vector<double> hidden_layer_activation;
   Vector<double> output_layer_combination;

   Vector<double> previous_layer_combination;   
   Vector<double> layer_combination_combination;   

   // Test

   n.set(1, 1, 1);
   n.initialize_parameters(0.0);

   previous_layer_combination.set(1, 0.0);

   layer_combination_combination = n.calculate_layer_combination_combination(1, previous_layer_combination);
   assert_true(layer_combination_combination.size() == 1, LOG);      
   assert_true(layer_combination_combination == 0.0, LOG);

   // Test

   n.set(2, 4, 3);

   parameters.set(27);

   parameters[0]  =  0.85;
   parameters[1]  = -0.04;
   parameters[2]  =  0.87;
   parameters[3]  = -0.25;
   parameters[4]  =  0.25;
   parameters[5]  = -0.27;
   parameters[6]  =  0.29;
   parameters[7]  = -0.57;
   parameters[8]  =  0.15;
   parameters[9]  = -0.77;
   parameters[10] =  0.96;
   parameters[11] = -0.48;
   parameters[12] =  0.08;
   parameters[13] = -0.06;
   parameters[14] =  0.26;
   parameters[15] = -0.15;
   parameters[16] =  0.96;
   parameters[17] = -0.33;
   parameters[18] =  0.63;
   parameters[19] = -0.32; 
   parameters[20] =  0.89;
   parameters[21] = -0.80;
   parameters[22] =  0.80; 
   parameters[23] = -0.03;
   parameters[24] =  0.32;
   parameters[25] =  0.06; 
   parameters[26] = -0.38;

   n.set_parameters(parameters);   

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

   const PerceptronLayer& hl = n.get_layer(0);

   hidden_layer_combination = hl.calculate_combinations(inputs);
   hidden_layer_activation = hl.calculate_activations(hidden_layer_combination);

   const PerceptronLayer& ol = n.get_layer(1);
   
   output_layer_combination = ol.calculate_combinations(hidden_layer_activation);

   layer_combination_combination = n.calculate_layer_combination_combination(1, hidden_layer_combination);

   assert_true((layer_combination_combination - output_layer_combination).calculate_absolute_value() < 1.0e-3, LOG);
*/
}


void MultilayerPerceptronTest::test_calculate_layer_combination_combination_Jacobian()
{
   message += "test_calculate_layer_combination_combination_Jacobian\n";
/*
   cout.precision(2);

   NumericalDifferentiation nd;

   MultilayerPerceptron mp;

   Vector<size_t> architecture;

   Vector<double> parameters;

   Matrix<double> output_layer_synaptic_weights;

   Vector<double> inputs;

   Vector<double> hidden_layer_combination;
   Vector<double> hidden_layer_activation;
   Vector<double> hidden_layer_activation_derivative;
   Vector<double> output_layer_combination;

   Vector<double> previous_layer_combination;
   Vector<double> previous_layer_activation_derivative;

   Matrix<double> layer_combination_combination_Jacobian;
   Matrix<double> numerical_layer_combination_combination_Jacobian;

   // Test

   mp.set(1, 1, 1);
   mp.initialize_parameters(0.0);
   inputs.set(1, 0.0);   

   previous_layer_combination = mp.get_layer(0).calculate_combinations(inputs);

   layer_combination_combination_Jacobian = mp.calculate_layer_combination_combination_Jacobian(1, previous_layer_combination);
   
   assert_true(layer_combination_combination_Jacobian == 0.0, LOG);

   // Test

   mp.set(1, 2, 3);
   mp.randomize_parameters_normal();

   inputs.set(1);
   inputs.randomize_normal();

   const PerceptronLayer& previous_layer = mp.get_layer(0);

   previous_layer_combination = previous_layer.calculate_combinations(inputs);
   previous_layer_activation_derivative = previous_layer.calculate_activations_derivatives(previous_layer_combination);

   layer_combination_combination_Jacobian = mp.calculate_layer_combination_combination_Jacobian(1, previous_layer_activation_derivative);

//   if(numerical_differentiation_tests)
//   {
      numerical_layer_combination_combination_Jacobian = nd.calculate_Jacobian(mp, &MultilayerPerceptron::calculate_layer_combination_combination, 1, previous_layer_combination);

      assert_true((layer_combination_combination_Jacobian - numerical_layer_combination_combination_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
//   }


   // Test

  mp.set(5, 10, 2);
  mp.randomize_parameters_normal();

  inputs.set(5);
  inputs.randomize_normal();

  const PerceptronLayer& previous_layer2 = mp.get_layer(0);

  previous_layer_combination = previous_layer2.calculate_combinations(inputs);
  previous_layer_activation_derivative = previous_layer2.calculate_activations_derivatives(previous_layer_combination);

  layer_combination_combination_Jacobian = mp.calculate_layer_combination_combination_Jacobian(1, previous_layer_activation_derivative);

  //   if(numerical_differentiation_tests)
  //   {
  numerical_layer_combination_combination_Jacobian = nd.calculate_Jacobian(mp, &MultilayerPerceptron::calculate_layer_combination_combination, 1, previous_layer_combination);

  assert_true((layer_combination_combination_Jacobian - numerical_layer_combination_combination_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
  //   }

   // Test

   mp.set(2, 4, 3);

   parameters.set(27);

   parameters[0]  =  0.85;
   parameters[1]  = -0.04;
   parameters[2]  =  0.87;
   parameters[3]  = -0.25;
   parameters[4]  =  0.25;
   parameters[5]  = -0.27;
   parameters[6]  =  0.29;
   parameters[7]  = -0.57;
   parameters[8]  =  0.15;
   parameters[9]  = -0.77;
   parameters[10] =  0.96;
   parameters[11] = -0.48;
   parameters[12] =  0.08;
   parameters[13] = -0.06;
   parameters[14] =  0.26;
   parameters[15] = -0.15;
   parameters[16] =  0.96;
   parameters[17] = -0.33;
   parameters[18] =  0.63;
   parameters[19] = -0.32; 
   parameters[20] =  0.89;
   parameters[21] = -0.80;
   parameters[22] =  0.80; 
   parameters[23] = -0.03;
   parameters[24] =  0.32;
   parameters[25] =  0.06; 
   parameters[26] = -0.38;

   mp.set_parameters(parameters);

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

   const PerceptronLayer& hidden_layer = mp.get_layer(0);

   hidden_layer_combination = hidden_layer.calculate_combinations(inputs);
   hidden_layer_activation = hidden_layer.calculate_activations(hidden_layer_combination);
   hidden_layer_activation_derivative = hidden_layer.calculate_activations_derivatives(hidden_layer_combination);

   const PerceptronLayer& output_layer = mp.get_layer(1);

   output_layer_combination = output_layer.calculate_combinations(hidden_layer_activation);

   layer_combination_combination_Jacobian = mp.calculate_layer_combination_combination_Jacobian(1, hidden_layer_activation_derivative);

//   if(numerical_differentiation_tests)
//   {
      numerical_layer_combination_combination_Jacobian = nd.calculate_Jacobian(mp, &MultilayerPerceptron::calculate_layer_combination_combination, 1, hidden_layer_combination);

      assert_true((layer_combination_combination_Jacobian - numerical_layer_combination_combination_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
//   }
*/
}


void MultilayerPerceptronTest::test_calculate_interlayer_combination_combination()
{
   message += "test_calculate_interlayer_combination_combination\n";
/*
   MultilayerPerceptron n;

   Vector<size_t> architecture;
   Vector<double> parameters;
   Vector<double> inputs;

   Vector<double> hidden_layer_combination;
   Vector<double> hidden_layer_activation;
   Vector<double> output_layer_combination;

   Vector<double> domain_layer_combination;
   Vector<double> interlayer_combination_combination;
  
   // Test

   n.set(2, 4, 3);

   parameters.set(27);

   parameters[0]  =  0.85;
   parameters[1]  = -0.04;
   parameters[2]  =  0.87;
   parameters[3]  = -0.25;
   parameters[4]  =  0.25;
   parameters[5]  = -0.27;
   parameters[6]  =  0.29;
   parameters[7]  = -0.57;
   parameters[8]  =  0.15;
   parameters[9]  = -0.77;
   parameters[10] =  0.96;
   parameters[11] = -0.48;
   parameters[12] =  0.08;
   parameters[13] = -0.06;
   parameters[14] =  0.26;
   parameters[15] = -0.15;
   parameters[16] =  0.96;
   parameters[17] = -0.33;
   parameters[18] =  0.63;
   parameters[19] = -0.32; 
   parameters[20] =  0.89;
   parameters[21] = -0.80;
   parameters[22] =  0.80; 
   parameters[23] = -0.03;
   parameters[24] =  0.32;
   parameters[25] =  0.06; 
   parameters[26] = -0.38;

   n.set_parameters(parameters);   

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

   const PerceptronLayer& hidden_layer = n.get_layer(0);

   hidden_layer_combination = hidden_layer.calculate_combinations(inputs);
   hidden_layer_activation = hidden_layer.calculate_activations(hidden_layer_combination);

   const PerceptronLayer& output_layer = n.get_layer(1);

   output_layer_combination = output_layer.calculate_combinations(hidden_layer_activation);

   interlayer_combination_combination = n.calculate_interlayer_combination_combination(0, 0, hidden_layer_combination);

   assert_true((interlayer_combination_combination - hidden_layer_combination).calculate_absolute_value() < 1.0e-3, LOG);

   interlayer_combination_combination = n.calculate_interlayer_combination_combination(0, 1, hidden_layer_combination);

   assert_true((interlayer_combination_combination - output_layer_combination).calculate_absolute_value() < 1.0e-3, LOG);

   interlayer_combination_combination = n.calculate_interlayer_combination_combination(1, 1, output_layer_combination);

   assert_true((interlayer_combination_combination - output_layer_combination).calculate_absolute_value() < 1.0e-3, LOG);
*/
}


// @todo 

void MultilayerPerceptronTest::test_calculate_interlayer_combination_combination_Jacobian()
{
   message += "test_calculate_interlayer_combination_combination_Jacobian\n";
/*
   NumericalDifferentiation nd;

   MultilayerPerceptron n;

   size_t inputs_number;
   size_t layers_number;
   Vector<size_t> architecture;

   Vector<double> parameters;
   Vector<double> inputs;

   Vector<double> hidden_layer_combination;
   Vector<double> hidden_layer_activation;
   Vector<double> hidden_layer_activation_derivative;
   Vector<double> output_layer_combination;
   Vector<double> domain_layer_combination;

   Vector<double> interlayer_combination_combination;

   Vector< Vector<double> > layers_combination;

   Matrix<double> layer_combination_combination_Jacobian;
   Matrix<double> interlayer_combination_combination_Jacobian;

   Matrix<double> numerical_interlayer_combination_combination_Jacobian;

   // Test

   n.set(2, 4, 3);

   parameters.set(27);

   parameters[0]  =  0.85;
   parameters[1]  = -0.04;
   parameters[2]  =  0.87;
   parameters[3]  = -0.25;
   parameters[4]  =  0.25;
   parameters[5]  = -0.27;
   parameters[6]  =  0.29;
   parameters[7]  = -0.57;
   parameters[8]  =  0.15;
   parameters[9]  = -0.77;
   parameters[10] =  0.96;
   parameters[11] = -0.48;
   parameters[12] =  0.08;
   parameters[13] = -0.06;
   parameters[14] =  0.26;
   parameters[15] = -0.15;
   parameters[16] =  0.96;
   parameters[17] = -0.33;
   parameters[18] =  0.63;
   parameters[19] = -0.32; 
   parameters[20] =  0.89;
   parameters[21] = -0.80;
   parameters[22] =  0.80; 
   parameters[23] = -0.03;
   parameters[24] =  0.32;
   parameters[25] =  0.06; 
   parameters[26] = -0.38;

   n.set_parameters(parameters);   

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

   const PerceptronLayer& hidden_layer = n.get_layer(0);

   hidden_layer_combination = hidden_layer.calculate_combinations(inputs);
   hidden_layer_activation = hidden_layer.calculate_activations(hidden_layer_combination);

   hidden_layer_activation_derivative = hidden_layer.calculate_activations_derivatives(hidden_layer_combination);

   const PerceptronLayer& output_layer = n.get_layer(1);

   output_layer_combination = output_layer.calculate_combinations(hidden_layer_activation);

   interlayer_combination_combination_Jacobian = n.calculate_interlayer_combination_combination_Jacobian(0, 0, hidden_layer_combination);
   assert_true(interlayer_combination_combination_Jacobian.get_rows_number() == 4, LOG);
   assert_true(interlayer_combination_combination_Jacobian.get_columns_number() == 4, LOG);
   assert_true(interlayer_combination_combination_Jacobian.is_identity(), LOG);

   //if(numerical_differentiation_tests)
   //{
      numerical_interlayer_combination_combination_Jacobian = nd.calculate_Jacobian(n, &MultilayerPerceptron::calculate_interlayer_combination_combination, 0, 0, hidden_layer_combination);
      assert_true((interlayer_combination_combination_Jacobian-numerical_interlayer_combination_combination_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);			    
   //}

   interlayer_combination_combination_Jacobian = n.calculate_interlayer_combination_combination_Jacobian(0, 1, hidden_layer_combination);
   assert_true(interlayer_combination_combination_Jacobian.get_rows_number() == 3, LOG);
   assert_true(interlayer_combination_combination_Jacobian.get_columns_number() == 4, LOG);

   layer_combination_combination_Jacobian = n.calculate_layer_combination_combination_Jacobian(1, hidden_layer_activation_derivative);
   assert_true((interlayer_combination_combination_Jacobian - layer_combination_combination_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);

  // if(numerical_differentiation_tests)
   //{
      numerical_interlayer_combination_combination_Jacobian = nd.calculate_Jacobian(n, &MultilayerPerceptron::calculate_interlayer_combination_combination, 0, 1, hidden_layer_combination);
      assert_true((interlayer_combination_combination_Jacobian-numerical_interlayer_combination_combination_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);			    
   //}

   interlayer_combination_combination_Jacobian = n.calculate_interlayer_combination_combination_Jacobian(1, 0, hidden_layer_combination);
   assert_true(interlayer_combination_combination_Jacobian.get_rows_number() == 4, LOG);
   assert_true(interlayer_combination_combination_Jacobian.get_columns_number() == 3, LOG);
   assert_true(interlayer_combination_combination_Jacobian == 0.0, LOG);
   
   interlayer_combination_combination_Jacobian = n.calculate_interlayer_combination_combination_Jacobian(1, 1, output_layer_combination);
   assert_true(interlayer_combination_combination_Jacobian.get_rows_number() == 3, LOG);
   assert_true(interlayer_combination_combination_Jacobian.get_columns_number() == 3, LOG);
   assert_true(interlayer_combination_combination_Jacobian.is_identity(), LOG);

  // if(numerical_differentiation_tests)
  // {
      numerical_interlayer_combination_combination_Jacobian = nd.calculate_Jacobian(n, &MultilayerPerceptron::calculate_interlayer_combination_combination, 1, 1, output_layer_combination);
      assert_true((interlayer_combination_combination_Jacobian-numerical_interlayer_combination_combination_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);			    
  // }

   // Test

   for(size_t i = 0; i < random_tests_number; i++)
   {
      n.initialize_random();

	  inputs_number = n.get_inputs_number();

      layers_number = n.get_layers_number();

	  inputs.set(inputs_number);
	  inputs.randomize_normal();

      layers_combination = n.calculate_layers_combinations(inputs);

      for(size_t domain_index = 0; domain_index < layers_number; domain_index++)
	  {
         for(size_t image_index = 0; image_index < layers_number; image_index++)
	     {              
            interlayer_combination_combination_Jacobian = n.calculate_interlayer_combination_combination_Jacobian(domain_index, image_index, layers_combination[domain_index]);

	        if(domain_index <= image_index)
		    {
               //if(numerical_differentiation_tests)
               //{
                  numerical_interlayer_combination_combination_Jacobian = nd.calculate_Jacobian(n, &MultilayerPerceptron::calculate_interlayer_combination_combination, domain_index, image_index, layers_combination[domain_index]);
                  assert_true((interlayer_combination_combination_Jacobian-numerical_interlayer_combination_combination_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
               //}
			}
			else
			{
               assert_true(interlayer_combination_combination_Jacobian == 0.0, LOG);			    
		    }
	     }	     
      }   
   }
*/
}


// @todo

void MultilayerPerceptronTest::test_calculate_parameters_Jacobian()
{
   message += "test_calculate_parameters_Jacobian\n";

   cout.precision(2);

   NumericalDifferentiation nd;

   MultilayerPerceptron n;

   size_t inputs_number;

//   size_t parameters_number;
   Vector<double> parameters;
 
   Vector<double> inputs;

   Vector<double> hidden_layer_combination;
   Vector<double> hidden_layer_activation;
   Vector<double> hidden_layer_activation_derivative;
   Vector<double> hidden_layer_activation_second_derivative;

   Vector<double> hidden_layer_output;
   Matrix<double> hidden_layer_combination_parameters_Jacobian;

   Matrix<double> output_layer_combination_parameters_Jacobian;

   Matrix<double> parameters_Jacobian;
   Matrix<double> numerical_parameters_Jacobian;

   // Test

   n.set(2, 4, 3);

   parameters.set(27);

   parameters[0]  =  0.85;
   parameters[1]  = -0.04;
   parameters[2]  =  0.87;
   parameters[3]  = -0.25;
   parameters[4]  =  0.25;
   parameters[5]  = -0.27;
   parameters[6]  =  0.29;
   parameters[7]  = -0.57;
   parameters[8]  =  0.15;
   parameters[9]  = -0.77;
   parameters[10] =  0.96;
   parameters[11] = -0.48;
   parameters[12] =  0.08;
   parameters[13] = -0.06;
   parameters[14] =  0.26;
   parameters[15] = -0.15;
   parameters[16] =  0.96;
   parameters[17] = -0.33;
   parameters[18] =  0.63;
   parameters[19] = -0.32; 
   parameters[20] =  0.89;
   parameters[21] = -0.80;
   parameters[22] =  0.80; 
   parameters[23] = -0.03;
   parameters[24] =  0.32;
   parameters[25] =  0.06; 
   parameters[26] = -0.38;

   n.set_parameters(parameters);   

//   parameters_number = n.get_parameters_number();   

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

//   hidden_layer_combination = hidden_layer.calculate_combination(inputs);
//   hidden_layer_activation = n.calculate_layer_activation(0, hidden_layer_combination);
//   hidden_layer_activation_derivative = n.calculate_layer_activation_derivatives(0, hidden_layer_combination);
//   hidden_layer_activation_second_derivative = n.calculate_layer_activation_second_derivatives(0, hidden_layer_combination);

//   hidden_layer_output = n.calculate_layer_output(0, inputs);

//   hidden_layer_combination_parameters_Jacobian = n.calculate_layer_combination_parameters_Jacobian(0, inputs);

//   output_layer_combination_parameters_Jacobian = n.calculate_layer_combination_parameters_Jacobian(1, hidden_layer_output);

//   parameters_Jacobian = n.calculate_parameters_Jacobian(inputs);

//   assert_true(parameters_Jacobian.get_rows_number() == 3, LOG);
//   assert_true(parameters_Jacobian.get_columns_number() == parameters_number, LOG);

   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Jacobian = nd.calculate_Jacobian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

      assert_true((parameters_Jacobian-numerical_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
      for(size_t i = 0; i < random_tests_number; i++)
      {
         n.initialize_random();

	     inputs_number = n.get_inputs_number();

	     parameters = n.get_parameters();

	     inputs.set(inputs_number);
	     inputs.randomize_normal();

//         parameters_Jacobian = n.calculate_parameters_Jacobian(inputs);

//         numerical_parameters_Jacobian = nd.calculate_Jacobian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

         assert_true((parameters_Jacobian-numerical_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG); 
      }
   }
}


// @todo

void MultilayerPerceptronTest::test_calculate_parameters_Hessian()
{
   message += "test_calculate_parameters_Hessian\n";

   NumericalDifferentiation nd;

   MultilayerPerceptron n;

   size_t inputs_number;
   Vector<size_t> architecture;
   size_t outputs_number;

//   size_t parameters_number;
   Vector<double> parameters;
 
   Vector<double> inputs;

   Vector< Matrix<double> > parameters_Hessian;

   Vector< Matrix<double> > layer_parameters_Hessian;
   Vector< Matrix<double> > numerical_parameters_Hessian;

   // Test

   n.set(1, 1); 

   n.initialize_parameters(0.0);
   
//   parameters_number = n.get_parameters_number();
   parameters = n.get_parameters();

   inputs.set(1, 0.0);

//   parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//   assert_true(parameters_Hessian.size() == 1, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0] == 0.0, LOG);

//   layer_parameters_Hessian = n.calculate_layer_parameters_Hessian(0, inputs);

//   assert_true((parameters_Hessian[0]-layer_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//   assert_true((parameters_Hessian[0] - parameters_Hessian[0].calculate_transpose()).calculate_absolute_value() < 1.0e-3, LOG);

//   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//   assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(1, 1); 
 
//   parameters_number = n.get_parameters_number();
   parameters = n.get_parameters();

   inputs.set(1, 3.1415927);

//   parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//   assert_true(parameters_Hessian.size() == 1, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0] == 0.0, LOG);

//   layer_parameters_Hessian = n.calculate_layer_parameters_Hessian(0, inputs);

//   assert_true((parameters_Hessian[0]-layer_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//   assert_true((parameters_Hessian[0] - parameters_Hessian[0].calculate_transpose()).calculate_absolute_value() < 1.0e-3, LOG);

   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//   assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(2, 1); 
 
//   parameters_number = n.get_parameters_number();
   parameters = n.get_parameters();

   inputs.set(2);
   inputs.randomize_uniform();

//   parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//   assert_true(parameters_Hessian.size() == 1, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0] == 0.0, LOG);

//   layer_parameters_Hessian = n.calculate_layer_parameters_Hessian(0, inputs);

//   assert_true((parameters_Hessian[0]-layer_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//   assert_true((parameters_Hessian[0] - parameters_Hessian[0].calculate_transpose()).calculate_absolute_value() < 1.0e-3, LOG);

   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//    assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(2, 2); 
 
//   parameters_number = n.get_parameters_number();
   parameters = n.get_parameters();

   inputs.set(2);
   inputs.randomize_normal();

//   parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//   assert_true(parameters_Hessian.size() == 2, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0] == 0.0, LOG);
//   assert_true((parameters_Hessian[0] - parameters_Hessian[0].calculate_transpose()).calculate_absolute_value() < 1.0e-3, LOG);

//   assert_true(parameters_Hessian[1].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[1].get_columns_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[1] == 0.0, LOG);
//   assert_true((parameters_Hessian[1] - parameters_Hessian[1].calculate_transpose()).calculate_absolute_value() < 1.0e-3, LOG);

//   layer_parameters_Hessian = n.calculate_layer_parameters_Hessian(0, inputs);

//   assert_true((parameters_Hessian[0]-layer_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//   assert_true((parameters_Hessian[1]-layer_parameters_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);

   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//    assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//    assert_true((parameters_Hessian[1]-numerical_parameters_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(2, 2); 

//   n.set_layer_activation_function(0, Perceptron::HyperbolicTangent);

//   parameters_number = n.get_parameters_number();
   parameters = n.get_parameters();

   inputs.set(2);
   inputs.randomize_normal();

//   parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//   assert_true(parameters_Hessian.size() == 2, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
//   assert_true((parameters_Hessian[0] - parameters_Hessian[0].calculate_transpose()).calculate_absolute_value() < 1.0e-3, LOG);

//   assert_true(parameters_Hessian[1].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[1].get_columns_number() == parameters_number, LOG);
//   assert_true((parameters_Hessian[1] - parameters_Hessian[1].calculate_transpose()).calculate_absolute_value() < 1.0e-3, LOG);

//   layer_parameters_Hessian = n.calculate_layer_parameters_Hessian(0, inputs);

//   assert_true((parameters_Hessian[0]-layer_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//   assert_true((parameters_Hessian[1]-layer_parameters_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);

   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//    assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//    assert_true((parameters_Hessian[1]-numerical_parameters_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(3, 2); 

//   n.set_layer_activation_function(0, Perceptron::Logistic);

//   parameters_number = n.get_parameters_number();
   parameters = n.get_parameters();

   inputs.set(3);
   inputs.randomize_normal();

//   parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//   assert_true(parameters_Hessian.size() == 2, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
//   assert_true((parameters_Hessian[0] - parameters_Hessian[0].calculate_transpose()).calculate_absolute_value() < 1.0e-3, LOG);

//   assert_true(parameters_Hessian[1].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[1].get_columns_number() == parameters_number, LOG);
//   assert_true((parameters_Hessian[1] - parameters_Hessian[1].calculate_transpose()).calculate_absolute_value() < 1.0e-3, LOG);

//   layer_parameters_Hessian = n.calculate_layer_parameters_Hessian(0, inputs);

//   assert_true((parameters_Hessian[0]-layer_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//   assert_true((parameters_Hessian[1]-layer_parameters_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);

   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//    assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//    assert_true((parameters_Hessian[1]-numerical_parameters_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(1, 1, 1); 

//   n.set_layer_activation_function(0, Perceptron::Linear);
//   n.set_layer_activation_function(1, Perceptron::Linear);

//   n.initialize_parameters(0.0);
   
//   parameters_number = n.get_parameters_number();
   parameters = n.get_parameters();

   inputs.set(1, 0.0);

//   parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//   assert_true(parameters_Hessian.size() == 1, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == parameters_number, LOG);

   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//    assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//    assert_true((parameters_Hessian[0] - parameters_Hessian[0].calculate_transpose()).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(1, 1, 1); 
  
//   parameters_number = n.get_parameters_number();
   parameters = n.get_parameters();

   inputs.set(1, 3.1415927);

//   parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//   assert_true(parameters_Hessian.size() == 1, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
   
   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//   assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(2, 1, 1); 
  
//   parameters_number = n.get_parameters_number();
   parameters = n.get_parameters();

   inputs.set(2);
   inputs.randomize_uniform();

//   parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//   assert_true(parameters_Hessian.size() == 1, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
   
   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);
   }

//   assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   n.set(1, 2, 1); 
  
//   parameters_number = n.get_parameters_number();
   parameters = n.get_parameters();

   inputs.set(1, 3.1415927);

//   parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//   assert_true(parameters_Hessian.size() == 1, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
   
   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//   assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   n.set(1, 1, 2); 
  
//   parameters_number = n.get_parameters_number();
   parameters = n.get_parameters();

   inputs.set(1, 3.1415927);

//   parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//   assert_true(parameters_Hessian.size() == 2, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[1].get_rows_number() == parameters_number, LOG);
//   assert_true(parameters_Hessian[1].get_columns_number() == parameters_number, LOG);
   
   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//   assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test
   
   if(numerical_differentiation_tests)
   {
      n.set(2, 4, 3);

      parameters.set(27);

      parameters[0]  =  0.85;
      parameters[1]  = -0.04;
      parameters[2]  =  0.87;
      parameters[3]  = -0.25;
      parameters[4]  =  0.25;
      parameters[5]  = -0.27;
      parameters[6]  =  0.29;
      parameters[7]  = -0.57;
      parameters[8]  =  0.15;
      parameters[9]  = -0.77;
      parameters[10] =  0.96;
      parameters[11] = -0.48;
      parameters[12] =  0.08;
      parameters[13] = -0.06;
      parameters[14] =  0.26;
      parameters[15] = -0.15;
      parameters[16] =  0.96;
      parameters[17] = -0.33;
      parameters[18] =  0.63;
      parameters[19] = -0.32; 
      parameters[20] =  0.89;
      parameters[21] = -0.80;
      parameters[22] =  0.80; 
      parameters[23] = -0.03;
      parameters[24] =  0.32;
      parameters[25] =  0.06; 
      parameters[26] = -0.38;
   
      n.set_parameters(parameters);   

      inputs.set(2);
      inputs[0] = -0.88;
      inputs[1] =  0.78;

//    parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//    assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//    assert_true((parameters_Hessian[1]-numerical_parameters_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
      architecture.set(4, 1);

      n.set(architecture);

//      n.set_layer_activation_function(0, Perceptron::Logistic);
//      n.set_layer_activation_function(1, Perceptron::Logistic);
//      n.set_layer_activation_function(2, Perceptron::Logistic);

      inputs.set(1, 3.1415927);

//    parameters_Hessian = n.calculate_parameters_Hessian(inputs);

      parameters = n.get_parameters();

//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//    assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
      architecture.set(4);

      architecture[0] = 2;
      architecture[1] = 3;
      architecture[2] = 5;
      architecture[3] = 4;

      n.set(architecture);

//      n.set_layer_activation_function(0, Perceptron::Logistic);
//      n.set_layer_activation_function(1, Perceptron::Logistic);
//      n.set_layer_activation_function(2, Perceptron::Logistic);

      inputs.set(2);
      inputs.randomize_normal();

//    parameters_Hessian = n.calculate_parameters_Hessian(inputs);

      parameters = n.get_parameters();

//      numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

//    assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//    assert_true((parameters_Hessian[1]-numerical_parameters_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
//    assert_true((parameters_Hessian[2]-numerical_parameters_Hessian[2]).calculate_absolute_value() < 1.0e-3, LOG);
//    assert_true((parameters_Hessian[3]-numerical_parameters_Hessian[3]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test 

   if(numerical_differentiation_tests)
   {
      for(size_t i = 0; i < random_tests_number; i++)
      {
         n.initialize_random();

         parameters = n.get_parameters();

         inputs_number = n.get_inputs_number();
         outputs_number = n.get_outputs_number();

         inputs.set(inputs_number);
         inputs.randomize_normal();

//       parameters_Hessian = n.calculate_parameters_Hessian(inputs);

//         numerical_parameters_Hessian = nd.calculate_Hessian(n, &MultilayerPerceptron::calculate_parameters_output, inputs, parameters);

         for(size_t j = 0; j < outputs_number; j++)
         {      
//          assert_true((parameters_Hessian[j]-numerical_parameters_Hessian[j]).calculate_absolute_value() < 1.0e-3, LOG);
         }
      }
   }

}


void MultilayerPerceptronTest::test_calculate_layers_combination()
{
   message += "test_calculate_layers_combination\n";
/*
   MultilayerPerceptron n(1,1,1);
   n.initialize_parameters(0.0);

   // Test

   Vector<double> inputs(1, 0.0);

   Vector< Vector<double> > forward_propagation_combination = n.calculate_layers_combinations(inputs);

   assert_true(forward_propagation_combination.size() == 2, LOG);
   assert_true(forward_propagation_combination[0] == 0, LOG);
   assert_true(forward_propagation_combination[1] == 0, LOG);
*/
}


void MultilayerPerceptronTest::test_calculate_layers_combination_Jacobian()
{
   message += "test_calculate_layers_combination_Jacobian\n";
}


void MultilayerPerceptronTest::test_calculate_layers_combination_parameters_Jacobian()
{
   message += "test_calculate_layers_combination_parameters_Jacobian\n";
}


void MultilayerPerceptronTest::test_calculate_perceptrons_combination_parameters_gradient()
{
   message += "test_calculate_perceptrons_combination_parameters_gradient\n";
/*
   MultilayerPerceptron n;

   Vector<double> inputs;

   Vector< Vector<double> > layers_inputs;

   Vector< Vector< Vector<double> > > perceptrons_combination_parameters_gradient;

   // Test

   n.set(1, 1);
   n.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   layers_inputs =  n.calculate_layers_input(inputs);

   perceptrons_combination_parameters_gradient = n.calculate_perceptrons_combination_parameters_gradient(layers_inputs);
*/
}


void MultilayerPerceptronTest::test_calculate_layers_activation()
{
   message += "test_calculate_layers_activation\n";
/*
   MultilayerPerceptron n;
   size_t layers_number;
   Vector<size_t> architecture;

   Vector<double> inputs;

   Vector< Vector<double> > forward_propagation_activation;

   // Test

   architecture.set(4, 1);
   n.set(architecture);

   layers_number = n.get_layers_number();

   n.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   forward_propagation_activation = n.calculate_layers_activations(inputs);

   assert_true(forward_propagation_activation.size() == layers_number, LOG);
   assert_true(forward_propagation_activation[0].size() == 1, LOG);
   assert_true(forward_propagation_activation[0] == 0.0, LOG);
   assert_true(forward_propagation_activation[1].size() == 1, LOG);
   assert_true(forward_propagation_activation[1] == 0.0, LOG);
   assert_true(forward_propagation_activation[2].size() == 1, LOG);
   assert_true(forward_propagation_activation[2] == 0.0, LOG);
*/
}


void MultilayerPerceptronTest::test_calculate_layers_activation_derivatives()
{
   message += "test_calculate_layers_activation_derivative\n";
/*
   MultilayerPerceptron n;
   size_t layers_number;
   Vector<size_t> architecture;

   Vector<double> inputs;
   Vector< Vector<double> > layers_activation_derivative;

   // Test

   architecture.set(3, 1);
   n.set(architecture);

   layers_number = n.get_layers_number();

   n.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   layers_activation_derivative = n.calculate_layers_activations_derivatives(inputs);

   assert_true(layers_activation_derivative.size() == layers_number, LOG);
   assert_true(layers_activation_derivative[0].size() == 1, LOG);
   assert_true(layers_activation_derivative[1].size() == 1, LOG);
*/
}


void MultilayerPerceptronTest::test_calculate_layers_activation_second_derivatives()
{
   message += "test_calculate_layers_activation_second_derivative\n";
/*
   MultilayerPerceptron n;
   size_t layers_number;
   Vector<size_t> architecture;

   Vector<double> inputs;
   Vector< Vector<double> > layers_activation_second_derivative;

   // Test

   architecture.set(3, 1);

   n.set(architecture);

   layers_number = n.get_layers_number();

   n.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   layers_activation_second_derivative = n.calculate_layers_activation_second_derivatives(inputs);

   assert_true(layers_activation_second_derivative.size() == layers_number, LOG);
   assert_true(layers_activation_second_derivative[0].size() == 1, LOG);
   assert_true(layers_activation_second_derivative[1].size() == 1, LOG);
*/
}


void MultilayerPerceptronTest::test_calculate_layers_Jacobian()
{
   message += "test_calculate_layers_Jacobian\n";
/*
   MultilayerPerceptron n;

   Vector<double> inputs;

   Vector< Matrix<double> > layers_Jacobian;

   // Test

   n.set(1,1,1);  
   n.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   layers_Jacobian = n.calculate_layers_Jacobian(inputs);

   assert_true(layers_Jacobian.size() == 2, LOG);
   assert_true(layers_Jacobian[0] == 0.0, LOG);
   assert_true(layers_Jacobian[1] == 0.0, LOG);

   // Test

   n.set(2,3,4);
   n.initialize_parameters(0.0);

   inputs.set(2, 0.0);

   layers_Jacobian = n.calculate_layers_Jacobian(inputs);

   assert_true(layers_Jacobian.size() == 2, LOG);

   assert_true(layers_Jacobian[0].get_rows_number() == 3, LOG);
   assert_true(layers_Jacobian[0].get_columns_number() == 2, LOG);
   assert_true(layers_Jacobian[0] == 0.0, LOG);

   assert_true(layers_Jacobian[1].get_rows_number() == 4, LOG);
   assert_true(layers_Jacobian[1].get_columns_number() == 3, LOG);
   assert_true(layers_Jacobian[1] == 0.0, LOG);

   // Test

   Vector<size_t> architecture(4);
   architecture[0] = 1;
   architecture[1] = 2;
   architecture[2] = 3;
   architecture[3] = 4;

   n.set(architecture);

   n.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   layers_Jacobian = n.calculate_layers_Jacobian(inputs);

   assert_true(layers_Jacobian.size() == 3, LOG);

   assert_true(layers_Jacobian[0].get_rows_number() == 2, LOG);
   assert_true(layers_Jacobian[0].get_columns_number() == 1, LOG);
   assert_true(layers_Jacobian[0] == 0.0, LOG);

   assert_true(layers_Jacobian[1].get_rows_number() == 3, LOG);
   assert_true(layers_Jacobian[1].get_columns_number() == 2, LOG);
   assert_true(layers_Jacobian[1] == 0.0, LOG);

   assert_true(layers_Jacobian[2].get_rows_number() == 4, LOG);
   assert_true(layers_Jacobian[2].get_columns_number() == 3, LOG);
   assert_true(layers_Jacobian[2] == 0.0, LOG);  
*/
}


void MultilayerPerceptronTest::test_calculate_layers_Hessian()
{
   message += "test_calculate_layers_Hessian\n";
/*
   MultilayerPerceptron n;

   Vector<double> inputs;

   Vector< Vector< Matrix<double> > > layers_Hessian;

   // Test

   n.set(1, 1);  
   n.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   layers_Hessian = n.calculate_layers_Hessian(inputs);

   assert_true(layers_Hessian.size() == 1, LOG);
   assert_true(layers_Hessian[0].size() == 1, LOG);
   assert_true(layers_Hessian[0][0] == 0.0, LOG);

   // Test

   n.set(2,3,4);
   n.initialize_parameters(0.0);

   inputs.set(2, 0.0);

   layers_Hessian = n.calculate_layers_Hessian(inputs);

   assert_true(layers_Hessian.size() == 2, LOG);

   assert_true(layers_Hessian[0].size() == 3, LOG);
   assert_true(layers_Hessian[1].size() == 4, LOG);

   // Test

   Vector<size_t> architecture(4);
   architecture[0] = 1;
   architecture[1] = 2;
   architecture[2] = 3;
   architecture[3] = 4;

   n.set(architecture);

   n.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   layers_Hessian = n.calculate_layers_Hessian(inputs);

   assert_true(layers_Hessian.size() == 3, LOG);
   assert_true(layers_Hessian[0].size() == 2, LOG);
   assert_true(layers_Hessian[1].size() == 3, LOG);
   assert_true(layers_Hessian[2].size() == 4, LOG);
*/
}


void MultilayerPerceptronTest::test_calculate_first_order_forward_propagation()
{
   message += "test_calculate_first_order_forward_propagation\n";
/*
   MultilayerPerceptron n;

   Vector<double> inputs;

   Vector< Vector< Vector<double> > > first_order_forward_propagation;

   // Test

   n.set(2, 4, 3);
   n.initialize_parameters(0.0);

   inputs.set(2, 0.0);

   first_order_forward_propagation = n.calculate_first_order_forward_propagation(inputs);

   assert_true(first_order_forward_propagation.size() == 2, LOG);
   assert_true(first_order_forward_propagation[0].size() == 2, LOG);
   assert_true(first_order_forward_propagation[1].size() == 2, LOG);
*/
}


void MultilayerPerceptronTest::test_calculate_second_order_forward_propagation()
{
   message += "test_calculate_second_order_forward_propagation\n";
/*
   MultilayerPerceptron n;

   Vector<double> inputs;

   Vector< Vector< Vector<double> > > second_order_forward_propagation;

   // Test

   n.set(2, 4, 3);
   n.initialize_parameters(0.0);

   inputs.set(2, 0.0);

   second_order_forward_propagation = n.calculate_second_order_forward_propagation(inputs);

   assert_true(second_order_forward_propagation.size() == 3, LOG);
   assert_true(second_order_forward_propagation[0].size() == 2, LOG);
   assert_true(second_order_forward_propagation[1].size() == 2, LOG);
   assert_true(second_order_forward_propagation[2].size() == 2, LOG);
*/
}


// @todo

void MultilayerPerceptronTest::test_write_expression()
{
   message += "test_write_expression\n";

   Vector<size_t> architecture;

   MultilayerPerceptron n;
   string expression;

   // Test

//   expression = n.write_expression();

   // Test

   n.set(1, 1, 1);
   n.initialize_parameters(-1.0);
//   expression = n.write_expression();

   // Test

   n.set(2, 1, 1);
   n.initialize_parameters(-1.0);
//   expression = n.write_expression();

   // Test

   n.set(1, 2, 1);
   n.initialize_parameters(-1.0);
//   expression = n.write_expression();

   // Test

   n.set(1, 1, 2);
   n.initialize_parameters(-1.0);
//   expression = n.write_expression();

   // Test

   n.set(2, 2, 2);
   n.initialize_parameters(-1.0);
//   expression = n.write_expression();

   // Test

   architecture.set(3, 2);
   n.set(architecture);
   n.initialize_parameters(-1.0);
//   expression = n.write_expression();
}


void MultilayerPerceptronTest::test_to_XML()
{
   message += "test_to_XML\n";

   MultilayerPerceptron mlp;

   tinyxml2::XMLDocument* mlpe;

   // Test

   mlpe = mlp.to_XML();

   assert_true(mlpe != nullptr, LOG);

   // Test

   mlp.initialize_random();

   mlpe = mlp.to_XML();

   assert_true(mlpe != nullptr, LOG);
}


void MultilayerPerceptronTest::test_from_XML()
{
   message += "test_from_XML\n";

   MultilayerPerceptron mlp;

   tinyxml2::XMLDocument* document;

   // Test

   mlp.initialize_random();

   document = mlp.to_XML();
   
   mlp.from_XML(*document);
}


void MultilayerPerceptronTest::run_test_case()
{
   message += "Running multilayer perceptron test case...\n";

   // Constructor and destructor methods
/*
   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // Architecture

   test_get_inputs_number();

   test_get_layers_number();
   test_count_layers_perceptrons_number();

   test_get_outputs_number();

   test_get_layers();
   test_get_layer();

   // Neurons number

   test_get_perceptrons_number();
   test_count_cumulative_perceptrons_number();

   // Layers parameters

   test_get_layers_parameters_number();

   test_get_layers_biases();
   test_get_layers_synaptic_weights();
   test_get_layers_parameters();

   // Multilayer perceptron parameters

   test_get_parameters_number();
   test_get_cumulative_parameters_number();

   test_get_parameters();

   // Multilayer perceptron parameters indices

   test_get_parameter_indices();
   test_get_parameters_indices();

   // Activation functions

   test_get_layers_activation_function();
   test_get_layers_activation_function_name();

   // Parameters methods

   test_get_parameters_number();
   test_get_parameters();   

   // Display messages

   test_get_display();

   // Set methods

   test_set();
   test_set_default();

   // Multilayer perceptron architecture

   test_set();

   // Multilayer perceptron parameters

   test_set_layers_biases();
   test_set_layers_synaptic_weights();      
   test_set_layers_parameters();
   test_set_parameters();

   // Activation functions

   test_set_layers_activation_function();

   // Parameters methods

   test_set_parameters();

   // Display messages

   test_set_display();

   // Check methods

   test_is_empty();

   // Growing and pruning

   test_grow_input();
   test_grow_layer();

   test_prune_input();
   test_prune_output();

   test_prune_layer();

   // Initialization methods

   test_initialize_random();

   // Parameters initialization methods

   test_initialize_parameters();
   test_initialize_biases(); 
   test_initialize_synaptic_weights();
   test_randomize_parameters_uniform();
   test_randomize_parameters_normal();

   // Parameters initialization methods

   test_initialize_parameters();
   test_randomize_parameters_uniform();
   test_randomize_parameters_normal();

   // Parameters norm 

   test_calculate_parameters_norm();

   // Multilayer perceptron outputs
*/
   test_calculate_outputs();
/*
   test_calculate_Jacobian();
//   test_calculate_Hessian();

   // Layer combination combination

   test_calculate_layer_combination_combination();
   test_calculate_layer_combination_combination_Jacobian();

   // Interlayers combination combination

   test_calculate_interlayer_combination_combination();
   test_calculate_interlayer_combination_combination_Jacobian();

   // Multilayer perceptron parameters outputs

   test_calculate_parameters_Jacobian();
   test_calculate_parameters_Hessian();

   // Forward propagation

   test_calculate_layers_combination();

   test_calculate_layers_combination_Jacobian();
   test_calculate_layers_combination_parameters_Jacobian();
   test_calculate_perceptrons_combination_parameters_gradient();

   test_calculate_layers_activation();
   test_calculate_layers_activation_derivatives();
   test_calculate_layers_activation_second_derivatives();

   test_calculate_layers_Jacobian();
   test_calculate_layers_Hessian();

   test_calculate_output_layers_delta();
   test_calculate_output_interlayers_Delta();

   test_calculate_interlayers_combination_combination_Jacobian();

   test_calculate_first_order_forward_propagation();
   test_calculate_second_order_forward_propagation();

   // Expression methods

   test_write_expression();

   // Serialization methods

   test_to_XML();
   test_from_XML();
*/
   message += "End of multilayer perceptron test case.\n";
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
