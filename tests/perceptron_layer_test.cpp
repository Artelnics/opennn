/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P E R C E P T R O N   L A Y E R   T E S T   C L A S S                                                      */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "perceptron_layer_test.h"


using namespace OpenNN;


// GENERAL CONSTRUCTOR

PerceptronLayerTest::PerceptronLayerTest() : UnitTesting()
{
}


// DESTRUCTOR

PerceptronLayerTest::~PerceptronLayerTest()
{
}


// METHODS

void PerceptronLayerTest::test_constructor()
{
   message += "test_constructor\n";

   // Default constructor

   PerceptronLayer l1;

   assert_true(l1.get_inputs_number() == 0, LOG);
   assert_true(l1.get_perceptrons_number() == 0, LOG);

   // Copy constructor

   l1.set(1, 2);

   PerceptronLayer l2(l1);

   assert_true(l2.get_inputs_number() == 1, LOG);
   assert_true(l2.get_perceptrons_number() == 2, LOG);
}


void PerceptronLayerTest::test_destructor()
{
   message += "test_destructor\n";

}


void PerceptronLayerTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

   PerceptronLayer l_1;
   PerceptronLayer l_2 = l_1;

   assert_true(l_2.get_inputs_number() == 0, LOG);
   assert_true(l_2.get_perceptrons_number() == 0, LOG);
   
}


void PerceptronLayerTest::test_count_inputs_number()
{
   message += "test_count_inputs_number\n";

   PerceptronLayer pl;

   // Test

   pl.set();
   assert_true(pl.get_inputs_number() == 0, LOG);

   // Test

   pl.set(1, 1);
   assert_true(pl.get_inputs_number() == 1, LOG);
}


void PerceptronLayerTest::test_get_perceptrons_number()
{
   message += "test_get_size\n";

   PerceptronLayer pl(1, 1);

   assert_true(pl.get_perceptrons_number() == 1, LOG);
}


void PerceptronLayerTest::test_get_activation_function()
{
   message += "test_get_activation_function\n";

   PerceptronLayer pl(1, 1);
   
   pl.set_activation_function(Perceptron::Logistic);
   assert_true(pl.get_activation_function() == Perceptron::Logistic, LOG);

   pl.set_activation_function(Perceptron::HyperbolicTangent);
   assert_true(pl.get_activation_function() == Perceptron::HyperbolicTangent, LOG);

   pl.set_activation_function(Perceptron::Threshold);
   assert_true(pl.get_activation_function() == Perceptron::Threshold, LOG);

   pl.set_activation_function(Perceptron::SymmetricThreshold);
   assert_true(pl.get_activation_function() == Perceptron::SymmetricThreshold, LOG);

   pl.set_activation_function(Perceptron::Linear);
   assert_true(pl.get_activation_function() == Perceptron::Linear, LOG);

}


void PerceptronLayerTest::test_get_activation_function_name()
{
   message += "test_get_activation_function_name\n";
}


void PerceptronLayerTest::test_count_parameters_number()
{      
   message += "test_count_parameters_number\n";

   PerceptronLayer pl;

   // Test

   pl.set(1, 1);

   assert_true(pl.count_parameters_number() == 2, LOG);

   // Test

   pl.set(3, 1);

   assert_true(pl.count_parameters_number() == 4, LOG);

   // Test

   pl.set(2, 4);

   assert_true(pl.count_parameters_number() == 12, LOG);

   // Test

   pl.set(4, 2);

   assert_true(pl.count_parameters_number() == 10, LOG);

}


void PerceptronLayerTest::test_count_cumulative_parameters_number()
{      
   message += "test_count_cumulative_parameters_number\n";

   PerceptronLayer pl;
}


void PerceptronLayerTest::test_set()
{
   message += "test_set\n";
}


void PerceptronLayerTest::test_set_default()
{
   message += "test_set_default\n";
}


void PerceptronLayerTest::test_arrange_biases()
{
   message += "test_arrange_biases\n";

   PerceptronLayer pl;
   Vector<double> biases;

   // Test

   pl.set(1, 1);
   pl.initialize_parameters(0.0);

   biases = pl.arrange_biases();

   assert_true(biases.size() == 1, LOG);
   assert_true(biases[0] == 0.0, LOG);
}


void PerceptronLayerTest::test_arrange_synaptic_weights()
{
   message += "test_arrange_synaptic_weights\n";

   PerceptronLayer pl;

   Matrix<double> synaptic_weights;

   // Test

   pl.set(1, 1);

   pl.initialize_parameters(0.0);

   synaptic_weights = pl.arrange_synaptic_weights();

   assert_true(synaptic_weights.get_rows_number() == 1, LOG);
   assert_true(synaptic_weights.get_columns_number() == 1, LOG);
   assert_true(synaptic_weights == 0.0, LOG);
}


void PerceptronLayerTest::test_arrange_parameters()
{
   message += "test_arrange_parameters\n";

   PerceptronLayer pl;
   Vector<double> biases;
   Matrix<double> synaptic_weights;
   Vector<double> parameters;

   // Test

   pl.set(1, 1);
   pl.initialize_parameters(1.0);

   parameters = pl.arrange_parameters();

   assert_true(parameters.size() == 2, LOG);
   assert_true(parameters == 1.0, LOG);

   // Test

   pl.set(2, 4);

   biases.set(4);
   biases[0] =  0.85;
   biases[1] = -0.25;
   biases[2] =  0.29;
   biases[3] = -0.77;

   pl.set_biases(biases);

   synaptic_weights.set(4, 2);

   synaptic_weights(0,0) = -0.04;
   synaptic_weights(0,1) =  0.87;

   synaptic_weights(1,0) =  0.25;
   synaptic_weights(1,1) = -0.27;

   synaptic_weights(2,0) = -0.57;
   synaptic_weights(2,1) =  0.15;

   synaptic_weights(3,0) =  0.96;
   synaptic_weights(3,1) = -0.48;

   pl.set_synaptic_weights(synaptic_weights);

   parameters = pl.arrange_parameters();

   assert_true(parameters.size() == 12, LOG);
   assert_true(parameters[0] == 0.85, LOG);
   assert_true(parameters[11] == -0.48, LOG);
}


void PerceptronLayerTest::test_set_biases()
{
   message += "test_set_biases\n";

   PerceptronLayer pl;

   Vector<double> biases;

   // Test

   pl.set(1, 1);

   biases.set(1, 0.0);

   pl.set_biases(biases);

   assert_true(pl.arrange_biases() == biases, LOG);
}


void PerceptronLayerTest::test_set_synaptic_weights()
{
   message += "test_set_synaptic_weights\n";

   PerceptronLayer pl(1, 1);

   Matrix<double> synaptic_weights(1, 1, 0.0);

   pl.set_synaptic_weights(synaptic_weights);

   assert_true(pl.arrange_synaptic_weights() == synaptic_weights, LOG);
}


void PerceptronLayerTest::test_set_parameters()
{
   message += "test_set_parameters\n";

   PerceptronLayer pl(1, 1);

   Vector<double> parameters(2, 0.0);

   pl.set_parameters(parameters);

   assert_true(pl.arrange_parameters() == parameters, LOG);
}


void PerceptronLayerTest::test_get_display()
{
   message += "test_get_display\n";
}


void PerceptronLayerTest::test_set_size()
{
   message += "test_set_size\n";
}


void PerceptronLayerTest::test_set_activation_function()
{
   message += "test_set_activation_function\n";
}


void PerceptronLayerTest::test_set_display()
{
   message += "test_set_display\n";
}


void PerceptronLayerTest::test_grow_inputs()
{
   message += "test_grow_inputs\n";

   PerceptronLayer pl;

    // Test

//    pl.set();
//    pl.grow_inputs(1);

//    assert_true(pl.get_inputs_number() == 0, LOG);
//    assert_true(pl.get_perceptrons_number() == 0, LOG);

//    // Test

//    pl.set(1, 1);
//    pl.grow_inputs(1);

//    assert_true(pl.get_inputs_number() == 2, LOG);
//    assert_true(pl.get_perceptrons_number() == 1, LOG);
}


void PerceptronLayerTest::test_grow_perceptrons()
{
   message += "test_grow_perceptrons\n";

   PerceptronLayer pl;

   // Test

   pl.set(1, 1);
   pl.grow_perceptrons(1);

   assert_true(pl.get_inputs_number() == 1, LOG);
   assert_true(pl.get_perceptrons_number() == 2, LOG);
}


void PerceptronLayerTest::test_prune_input()
{
   message += "test_prune_input\n";

   PerceptronLayer pl;

   // Test

   pl.set(1, 1);
   pl.prune_input(0);

   assert_true(pl.get_inputs_number() == 0, LOG);
   assert_true(pl.get_perceptrons_number() == 1, LOG);
}


void PerceptronLayerTest::test_prune_perceptron()
{
   message += "test_prune_perceptron\n";

   PerceptronLayer pl;

   // Test

   pl.set(1, 1);
   pl.prune_perceptron(0);

   assert_true(pl.get_inputs_number() == 0, LOG);
   assert_true(pl.get_perceptrons_number() == 0, LOG);
}


void PerceptronLayerTest::test_initialize_random()
{
   message += "test_initialize_random\n";

   PerceptronLayer pl;

   size_t inputs_number;
   size_t perceptrons_number;

   // Test

   pl.initialize_random();

   inputs_number = pl.get_inputs_number();

   assert_true(inputs_number >= 1 && inputs_number <= 10, LOG); 

   perceptrons_number = pl.get_perceptrons_number();

   assert_true(perceptrons_number >= 1 && perceptrons_number <= 10, LOG); 
}


void PerceptronLayerTest::test_initialize_parameters()
{
   message += "test_initialize_parameters\n";

   PerceptronLayer pl;

   Vector<double> parameters;

   // Test

   pl.set(1, 1);
   pl.initialize_parameters(0.0);

   parameters = pl.arrange_parameters();

   assert_true(parameters == 0.0, LOG);
}


void PerceptronLayerTest::test_initialize_biases()
{
   message += "test_initialize_biases\n";
}


void PerceptronLayerTest::test_initialize_synaptic_weights()
{
   message += "test_initialize_synaptic_weights\n";
}


void PerceptronLayerTest::test_randomize_parameters_uniform()
{
   message += "test_randomize_parameters_uniform\n";

   PerceptronLayer pl;
   Vector<double> parameters;

   // Test

   pl.set(1, 1);

   pl.randomize_parameters_uniform();
   parameters = pl.arrange_parameters();
   
   assert_true(parameters >= -1.0, LOG);
   assert_true(parameters <=  1.0, LOG);   

}


void PerceptronLayerTest::test_randomize_parameters_normal()
{
   message += "test_randomize_parameters_normal\n";

   PerceptronLayer pl;
   Vector<double> parameters;

   // Test

   pl.set(1, 1);

   pl.randomize_parameters_normal(1.0, 0.0);
   parameters = pl.arrange_parameters();

   assert_true(parameters == 1.0, LOG);
}


void PerceptronLayerTest::test_calculate_parameters_norm()
{
   message += "test_calculate_parameters_norm\n";

   PerceptronLayer pl;
   Vector<double> biases;
   Matrix<double> synaptic_weights;
   Vector<double> parameters;

   double parameters_norm;

   // Test

   pl.set(1, 1);
   pl.initialize_parameters(0.0);

   parameters_norm = pl.calculate_parameters_norm();

   assert_true(parameters_norm == 0.0, LOG);

   // Test

   pl.set(2, 4);

   biases.set(4);
   biases[0] =  0.85;
   biases[1] = -0.25;
   biases[2] =  0.29;
   biases[3] = -0.77;

   pl.set_biases(biases);

   synaptic_weights.set(4, 2);

   synaptic_weights(0,0) = -0.04;
   synaptic_weights(0,1) =  0.87;

   synaptic_weights(1,0) =  0.25;
   synaptic_weights(1,1) = -0.27;

   synaptic_weights(2,0) = -0.57;
   synaptic_weights(2,1) =  0.15;

   synaptic_weights(3,0) =  0.96;
   synaptic_weights(3,1) = -0.48;

   pl.set_synaptic_weights(synaptic_weights);

   parameters = pl.arrange_parameters();

   parameters_norm = pl.calculate_parameters_norm();

   assert_true(fabs(parameters_norm - parameters.calculate_norm()) < 1.0e-6, LOG);

   // Test

   pl.set(4, 2);

   parameters.set(10);
   parameters[0] =  0.41;
   parameters[1] = -0.68; 
   parameters[2] =  0.14; 
   parameters[3] = -0.50; 
   parameters[4] =  0.52; 
   parameters[5] = -0.70; 
   parameters[6] =  0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] =  0.05; 

   pl.set_parameters(parameters);

   parameters_norm = pl.calculate_parameters_norm();

   assert_true(fabs(parameters_norm - parameters.calculate_norm()) < 1.0e-6, LOG);
}


void PerceptronLayerTest::test_calculate_combination()
{
   message += "test_calculate_combination\n";

   PerceptronLayer pl;

   Vector<double> biases;
   Matrix<double> synaptic_weights;
   Vector<double> parameters;

   Vector<double> inputs;   

   Vector<double> combination;

   // Test
 
   pl.set(1, 2);
   pl.initialize_parameters(0.0);
   inputs.set(1, 0.0);   

   combination = pl.calculate_combinations(inputs);

   assert_true(combination.size() == 2, LOG);      
   assert_true(combination == 0.0, LOG);

   // Test

   pl.set(2, 4);

   biases.set(4);
   biases[0] =  0.85;
   biases[1] = -0.25;
   biases[2] =  0.29;
   biases[3] = -0.77;

   pl.set_biases(biases);

   synaptic_weights.set(4, 2);

   synaptic_weights(0,0) = -0.04;
   synaptic_weights(0,1) =  0.87;

   synaptic_weights(1,0) =  0.25;
   synaptic_weights(1,1) = -0.27;

   synaptic_weights(2,0) = -0.57;
   synaptic_weights(2,1) =  0.15;

   synaptic_weights(3,0) =  0.96;
   synaptic_weights(3,1) = -0.48;

   pl.set_synaptic_weights(synaptic_weights);

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

   combination = pl.calculate_combinations(inputs);

   assert_true(combination - (biases + synaptic_weights.dot(inputs)) < 1.0e-3, LOG);

   // Test

   pl.set(4, 2);

   parameters.set(10);
   parameters[0] =  0.41;
   parameters[1] = -0.68; 
   parameters[2] =  0.14; 
   parameters[3] = -0.50; 
   parameters[4] =  0.52; 
   parameters[5] = -0.70; 
   parameters[6] =  0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] =  0.05; 

   pl.set_parameters(parameters);

   inputs.set(4);
   inputs[0] =  0.85;
   inputs[1] = -0.25;
   inputs[2] =  0.29;
   inputs[3] = -0.77;

   combination = pl.calculate_combinations(inputs);

   biases = pl.arrange_biases();
   synaptic_weights = pl.arrange_synaptic_weights();

   assert_true(combination - (biases + synaptic_weights.dot(inputs)).calculate_absolute_value() < 1.0e-6, LOG);

   // Test

   pl.set(1, 1);

   inputs.set(1);
   inputs.randomize_normal();

   parameters = pl.arrange_parameters();

   assert_true(pl.calculate_combinations(inputs) == pl.calculate_combinations(inputs, parameters), LOG);

}


void PerceptronLayerTest::test_calculate_combination_Jacobian()
{
   message += "test_calculate_combination_Jacobian\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Matrix<double> synaptic_weights;
   Vector<double> parameters;
   Vector<double> inputs;

   Matrix<double> combination_Jacobian;
   Matrix<double> numerical_combination_Jacobian;

   // Test

   pl.set(3, 2);

   inputs.set(3);
   inputs.randomize_normal();

   combination_Jacobian = pl.calculate_combinations_Jacobian(inputs);

   if(numerical_differentiation_tests)
   {
      numerical_combination_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_combinations, inputs);

      assert_true((combination_Jacobian-numerical_combination_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   pl.set(4, 2);

   parameters.set(10);
   parameters[0] =  0.41;
   parameters[1] = -0.68; 
   parameters[2] =  0.14; 
   parameters[3] = -0.50; 
   parameters[4] =  0.52; 
   parameters[5] = -0.70; 
   parameters[6] =  0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] =  0.05; 

   pl.set_parameters(parameters);

   inputs.set(4);
   inputs[0] =  0.85;
   inputs[1] = -0.25;
   inputs[2] =  0.29;
   inputs[3] = -0.77;

   combination_Jacobian = pl.calculate_combinations_Jacobian(inputs);

   synaptic_weights = pl.arrange_synaptic_weights();

   assert_true(combination_Jacobian == synaptic_weights, LOG);

   if(numerical_differentiation_tests)
   {
      numerical_combination_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_combinations, inputs);
      assert_true((combination_Jacobian-numerical_combination_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }
}


void PerceptronLayerTest::test_calculate_combination_Hessian_form()
{
   message += "test_calculate_combination_Hessian_form\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;
   Vector<double> inputs;

   Vector< Matrix<double> > combination_Hessian_form;
   Vector< Matrix<double> > numerical_combination_Hessian_form;

   // Test

   pl.set(2, 4);

   inputs.set(2);
   inputs.randomize_normal();

   combination_Hessian_form = pl.calculate_combinations_Hessian_form(inputs);

   assert_true(combination_Hessian_form.size() == 4, LOG);

   if(numerical_differentiation_tests)
   {
      numerical_combination_Hessian_form = nd.calculate_Hessian_form(pl, &PerceptronLayer::calculate_combinations, inputs);

      assert_true((combination_Hessian_form[0]-numerical_combination_Hessian_form[0]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_Hessian_form[1]-numerical_combination_Hessian_form[1]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_Hessian_form[2]-numerical_combination_Hessian_form[2]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_Hessian_form[3]-numerical_combination_Hessian_form[3]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   pl.set(4, 2);

   parameters.set(10);
   parameters[0] =  0.41;
   parameters[1] = -0.68; 
   parameters[2] =  0.14; 
   parameters[3] = -0.50; 
   parameters[4] =  0.52; 
   parameters[5] = -0.70; 
   parameters[6] =  0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] =  0.05; 

   pl.set_parameters(parameters);

   inputs.set(4);
   inputs[0] =  0.85;
   inputs[1] = -0.25;
   inputs[2] =  0.29;
   inputs[3] = -0.77;

   combination_Hessian_form = pl.calculate_combinations_Hessian_form(inputs);

   assert_true(combination_Hessian_form.size() == 2, LOG);

   assert_true(combination_Hessian_form[0].get_rows_number() == 4, LOG);
   assert_true(combination_Hessian_form[0].get_columns_number() == 4, LOG);
   assert_true(combination_Hessian_form[0] == 0.0, LOG);
   assert_true(combination_Hessian_form[0].is_symmetric(), LOG);

   assert_true(combination_Hessian_form[1].get_rows_number() == 4, LOG);
   assert_true(combination_Hessian_form[1].get_columns_number() == 4, LOG);
   assert_true(combination_Hessian_form[1] == 0.0, LOG);
   assert_true(combination_Hessian_form[1].is_symmetric(), LOG);

   if(numerical_differentiation_tests)
   {
      numerical_combination_Hessian_form = nd.calculate_Hessian_form(pl, &PerceptronLayer::calculate_combinations, inputs);

      assert_true((combination_Hessian_form[0]-numerical_combination_Hessian_form[0]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_Hessian_form[1]-numerical_combination_Hessian_form[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }
}


void PerceptronLayerTest::test_calculate_combination_parameters_Jacobian()
{
   message += "test_calculate_combination_parameters_Jacobian\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;

   Vector<double> inputs;

   Matrix<double> combination_parameters_Jacobian;
   Matrix<double> numerical_combination_parameters_Jacobian;

   // Test

   pl.set(2, 4);

   parameters = pl.arrange_parameters();

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

   combination_parameters_Jacobian = pl.calculate_combinations_Jacobian(inputs, parameters);

   if(numerical_differentiation_tests)
   {
      numerical_combination_parameters_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_combinations, inputs, parameters);

      assert_true((combination_parameters_Jacobian-numerical_combination_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   pl.set(4, 2);

   parameters.set(10);
   parameters[0] =  0.41;
   parameters[1] = -0.68; 
   parameters[2] =  0.14; 
   parameters[3] = -0.50; 
   parameters[4] =  0.52; 
   parameters[5] = -0.70; 
   parameters[6] =  0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] =  0.05; 

   pl.set_parameters(parameters);

   inputs.set(4);
   inputs[0] =  0.85;
   inputs[1] = -0.25;
   inputs[2] =  0.29;
   inputs[3] = -0.77;

   combination_parameters_Jacobian = pl.calculate_combinations_Jacobian(inputs, parameters);

   if(numerical_differentiation_tests)
   {
      numerical_combination_parameters_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_combinations, inputs, parameters);

      assert_true((combination_parameters_Jacobian-numerical_combination_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }
}


void PerceptronLayerTest::test_calculate_combination_parameters_Hessian_form()
{
   message += "test_calculate_combination_parameters_Hessian_form\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   size_t parameters_number;
   Vector<double> parameters;

   Vector<double> inputs;

   Vector< Matrix<double> > combination_parameters_Hessian_form;
   Vector< Matrix<double> > numerical_combination_parameters_Hessian_form;

   // Test

   pl.set(2, 4);

   parameters_number = pl.count_parameters_number();
   parameters = pl.arrange_parameters();

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

   combination_parameters_Hessian_form = pl.calculate_combinations_Hessian_form(inputs, parameters);

   assert_true(combination_parameters_Hessian_form.size() == 4, LOG);
   assert_true(combination_parameters_Hessian_form[0].get_rows_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[0].get_columns_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[0].calculate_absolute_value() < 1.0e-6 , LOG);

   assert_true(combination_parameters_Hessian_form[1].get_rows_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[1].get_columns_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[1].calculate_absolute_value() < 1.0e-6 , LOG);

   assert_true(combination_parameters_Hessian_form[2].get_rows_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[2].get_columns_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[2].calculate_absolute_value() < 1.0e-6 , LOG);

   assert_true(combination_parameters_Hessian_form[3].get_rows_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[3].get_columns_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[3].calculate_absolute_value() < 1.0e-6 , LOG);

   if(numerical_differentiation_tests)
   {
      numerical_combination_parameters_Hessian_form = nd.calculate_Hessian_form(pl, &PerceptronLayer::calculate_combinations, inputs, parameters);

      assert_true((combination_parameters_Hessian_form[0]-numerical_combination_parameters_Hessian_form[0]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_parameters_Hessian_form[1]-numerical_combination_parameters_Hessian_form[1]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_parameters_Hessian_form[2]-numerical_combination_parameters_Hessian_form[2]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_parameters_Hessian_form[3]-numerical_combination_parameters_Hessian_form[3]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   pl.set(4, 2);

   parameters.set(10);
   parameters[0] =  0.41;
   parameters[1] = -0.68; 
   parameters[2] =  0.14; 
   parameters[3] = -0.50; 
   parameters[4] =  0.52; 
   parameters[5] = -0.70; 
   parameters[6] =  0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] =  0.05; 

   pl.set_parameters(parameters);

   inputs.set(4);
   inputs[0] =  0.85;
   inputs[1] = -0.25;
   inputs[2] =  0.29;
   inputs[3] = -0.77;

   parameters_number = pl.count_parameters_number();

   combination_parameters_Hessian_form = pl.calculate_combinations_Hessian_form(inputs, parameters);

   assert_true(combination_parameters_Hessian_form.size() == 2, LOG);
   assert_true(combination_parameters_Hessian_form[0].get_rows_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[0].get_columns_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[0].calculate_absolute_value() < 1.0e-6 , LOG);

   assert_true(combination_parameters_Hessian_form[1].get_rows_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[1].get_columns_number() == parameters_number, LOG);
   assert_true(combination_parameters_Hessian_form[1].calculate_absolute_value() < 1.0e-6 , LOG);

   if(numerical_differentiation_tests)
   {
      numerical_combination_parameters_Hessian_form = nd.calculate_Hessian_form(pl, &PerceptronLayer::calculate_combinations, inputs, parameters);

      assert_true((combination_parameters_Hessian_form[0]-numerical_combination_parameters_Hessian_form[0]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_parameters_Hessian_form[1]-numerical_combination_parameters_Hessian_form[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }
}


void PerceptronLayerTest::test_calculate_activation()
{
   message += "test_calculate_activation\n";

   PerceptronLayer pl;

   Vector<double> parameters;
 
   Vector<double> inputs;   
   Vector<double> combination;   
   Vector<double> activation;

   // Test

   pl.set(1, 2);
   pl.initialize_parameters(0.0);

   combination.set(2, 0.0);

   pl.set_activation_function(Perceptron::Logistic);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);
   assert_true(activation == 0.5, LOG);

   // Test

   pl.set(1, 2);
   pl.initialize_parameters(0.0);

   combination.set(2, 0.0);

   pl.set_activation_function(Perceptron::HyperbolicTangent);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);
   assert_true(activation == 0.0, LOG);

   // Test

   pl.set(1, 2);
   pl.initialize_parameters(0.0);

   combination.set(2, 0.0);

   pl.set_activation_function(Perceptron::Threshold);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);
   assert_true(activation == 1.0, LOG);

   // Test

   pl.set(1, 2);
   pl.initialize_parameters(0.0);

   combination.set(2, 0.0);

   pl.set_activation_function(Perceptron::SymmetricThreshold);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);
   assert_true(activation == 1.0, LOG);

   // Test

   pl.set(1, 2);
   pl.initialize_parameters(0.0);

   combination.set(2, 0.0);

   pl.set_activation_function(Perceptron::Linear);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);
   assert_true(activation == 0.0, LOG);

   // Test

   pl.set(4, 2);

   parameters.set(10);
   parameters[0] =  0.41;
   parameters[1] = -0.68; 
   parameters[2] =  0.14; 
   parameters[3] = -0.50; 
   parameters[4] =  0.52; 
   parameters[5] = -0.70; 
   parameters[6] =  0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] =  0.05; 

   pl.set_parameters(parameters);

   inputs.set(4);
   inputs[0] =  0.85;
   inputs[1] = -0.25;
   inputs[2] =  0.29;
   inputs[3] = -0.77;

   combination = pl.calculate_combinations(inputs);

   pl.set_activation_function(Perceptron::Threshold);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);

   pl.set_activation_function(Perceptron::SymmetricThreshold);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);

   pl.set_activation_function(Perceptron::Logistic);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);

   pl.set_activation_function(Perceptron::HyperbolicTangent);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);

   pl.set_activation_function(Perceptron::Linear);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);
}


void PerceptronLayerTest::test_calculate_activation_derivative()
{
   message += "test_calculate_activation_derivative\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;
   Vector<double> parameters;         
   Vector<double> inputs;         
   Vector<double> combination;         
   Vector<double> activation_derivative; 
   Vector<double> numerical_activation_derivative; 

   numerical_differentiation_tests = true;

   // Test

   pl.set(1, 2);
   combination.set(2, 0.0);         

   pl.set_activation_function(Perceptron::Logistic);
   activation_derivative = pl.calculate_activations_derivatives(combination);
   assert_true(activation_derivative.size() == 2, LOG);
   assert_true(activation_derivative == 0.25, LOG);

   pl.set_activation_function(Perceptron::HyperbolicTangent);
   activation_derivative = pl.calculate_activations_derivatives(combination);
   assert_true(activation_derivative.size() == 2, LOG);
   assert_true(activation_derivative == 1.0, LOG);

   pl.set_activation_function(Perceptron::Linear);
   activation_derivative = pl.calculate_activations_derivatives(combination);
   assert_true(activation_derivative.size() == 2, LOG);
   assert_true(activation_derivative == 1.0, LOG);   

   // Test

   if(numerical_differentiation_tests)
   {
      pl.set(2, 4);

      combination.set(4);         
      combination[0] =  1.56;
      combination[1] = -0.68;
      combination[2] =  0.91;
      combination[3] = -1.99;

      pl.set_activation_function(Perceptron::Threshold);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::SymmetricThreshold);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::Logistic);
      activation_derivative = pl.calculate_activations_derivatives(combination);

      numerical_activation_derivative = nd.calculate_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::HyperbolicTangent);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::Linear);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
      pl.set(4, 2);

      parameters.set(10);
      parameters[0] =  0.41;
      parameters[1] = -0.68; 
      parameters[2] =  0.14; 
      parameters[3] = -0.50; 
      parameters[4] =  0.52; 
      parameters[5] = -0.70; 
      parameters[6] =  0.85; 
      parameters[7] = -0.18; 
      parameters[8] = -0.65; 
      parameters[9] =  0.05; 

      pl.set_parameters(parameters);

      inputs.set(4);
      inputs[0] =  0.85;
      inputs[1] = -0.25;
      inputs[2] =  0.29;
      inputs[3] = -0.77;

      combination = pl.calculate_combinations(inputs);

      pl.set_activation_function(Perceptron::Threshold);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::SymmetricThreshold);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::Logistic);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::HyperbolicTangent);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::Linear);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);
   }

}


void PerceptronLayerTest::test_calculate_activation_second_derivative()
{
   message += "test_calculate_activation_second_derivative\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;         

   Vector<double> inputs;         
   Vector<double> combination;         
   Vector<double> activation_second_derivative; 
   Vector<double> numerical_activation_second_derivative; 

   // Test

   pl.set(1, 2);
   pl.initialize_parameters(0.0);
   
   combination.set(2, 0.0);   

   pl.set_activation_function(Perceptron::Logistic);
   activation_second_derivative  = pl.calculate_activations_second_derivatives(combination);
   assert_true(activation_second_derivative.size() == 2, LOG);
   assert_true(activation_second_derivative == 0.0, LOG);

   pl.set_activation_function(Perceptron::HyperbolicTangent);
   activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
   assert_true(activation_second_derivative.size() == 2, LOG);
   assert_true(activation_second_derivative == 0.0, LOG);

   pl.set_activation_function(Perceptron::Linear);
   activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
   assert_true(activation_second_derivative.size() == 2, LOG);
   assert_true(activation_second_derivative == 0.0, LOG);

   // Test
   
   if(numerical_differentiation_tests)
   {
      pl.set(2, 4);

      combination.set(4);         
      combination[0] =  1.56;
      combination[1] = -0.68;
      combination[2] =  0.91;
      combination[3] = -1.99;

      pl.set_activation_function(Perceptron::Threshold);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::SymmetricThreshold);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::Logistic);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::HyperbolicTangent);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::Linear);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
      pl.set(4, 2);

      parameters.set(10);
      parameters[0] =  0.41;
      parameters[1] = -0.68; 
      parameters[2] =  0.14; 
      parameters[3] = -0.50; 
      parameters[4] =  0.52; 
      parameters[5] = -0.70; 
      parameters[6] =  0.85; 
      parameters[7] = -0.18; 
      parameters[8] = -0.65; 
      parameters[9] =  0.05; 

      pl.set_parameters(parameters);

      inputs.set(4);
      inputs[0] =  0.85;
      inputs[1] = -0.25;
      inputs[2] =  0.29;
      inputs[3] = -0.77;

      combination = pl.calculate_combinations(inputs);

      pl.set_activation_function(Perceptron::Threshold);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::SymmetricThreshold);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::Logistic);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::HyperbolicTangent);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(Perceptron::Linear);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivative(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);
   }

}


void PerceptronLayerTest::test_calculate_outputs()
{
   message += "test_calculate_outputs\n";

   PerceptronLayer pl;

   Vector<double> parameters;

   Vector<double> inputs;
   Vector<double> outputs;
   Vector<double> potential_outputs;

   // Test 

   pl.set(3, 2);
   pl.initialize_parameters(0.0);

   inputs.set(3, 0.0);

   outputs = pl.calculate_outputs(inputs);

   assert_true(outputs.size() == 2, LOG);
   assert_true(outputs == 0.0, LOG);

   // Test

   pl.set(4, 2);

   parameters.set(10);
   parameters[0] =  0.41;
   parameters[1] = -0.68; 
   parameters[2] =  0.14; 
   parameters[3] = -0.50; 
   parameters[4] =  0.52; 
   parameters[5] = -0.70; 
   parameters[6] =  0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] =  0.05; 

   pl.set_parameters(parameters);

   inputs.set(4);
   inputs[0] =  0.85;
   inputs[1] = -0.25;
   inputs[2] =  0.29;
   inputs[3] = -0.77;

   outputs = pl.calculate_outputs(inputs);

   assert_true(outputs.size() ==  2, LOG);

   // Test

   inputs.set(1, 3.0);

   pl.set(1, 1);

   pl.initialize_parameters(2.0);

   outputs = pl.calculate_outputs(inputs);

   parameters.set(2, 1.0);

   potential_outputs = pl.calculate_outputs(inputs, parameters);

   assert_true(outputs != potential_outputs, LOG);

   // Test

   pl.set(1, 1);

   inputs.set(1);
   inputs.randomize_normal();

   parameters = pl.arrange_parameters();

   assert_true(pl.calculate_outputs(inputs) == pl.calculate_outputs(inputs, parameters), LOG);

}


void PerceptronLayerTest::test_calculate_Jacobian()
{
   message += "test_calculate_Jacobian\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;

   Vector<double> inputs;

   Matrix<double> Jacobian;
   Matrix<double> numerical_Jacobian;

   // Test

   if(numerical_differentiation_tests)
   {
      pl.set(3, 2);

      inputs.set(3);
      inputs.randomize_normal();

      Jacobian = pl.calculate_Jacobian(inputs);

      numerical_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_outputs, inputs);

      assert_true((Jacobian-numerical_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   if(numerical_differentiation_tests)
   {
      pl.set(4, 2);

      parameters.set(10);
      parameters[0] =  0.41;
      parameters[1] = -0.68; 
      parameters[2] =  0.14; 
      parameters[3] = -0.50; 
      parameters[4] =  0.52; 
      parameters[5] = -0.70; 
      parameters[6] =  0.85; 
      parameters[7] = -0.18; 
      parameters[8] = -0.65; 
      parameters[9] =  0.05; 

      pl.set_parameters(parameters);

      inputs.set(4);
      inputs[0] =  0.85;
      inputs[1] = -0.25;
      inputs[2] =  0.29;
      inputs[3] = -0.77;

      Jacobian = pl.calculate_Jacobian(inputs);

      numerical_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_outputs, inputs);

      assert_true((Jacobian-numerical_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }
}


void PerceptronLayerTest::test_calculate_Hessian_form()
{
   message += "test_calculate_Hessian_form\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;

   Vector<double> inputs;

   Vector< Matrix<double> > Hessian_form;
   Vector< Matrix<double> > numerical_Hessian_form;

   Matrix<double> Hessian;

   // Test

   pl.set(1, 1);
   pl.initialize_parameters(0.0);

   inputs.set(1);
   inputs.initialize(0.0);

   Hessian_form = pl.calculate_Hessian_form(inputs);

   assert_true(Hessian_form.size() == 1, LOG);
   assert_true(Hessian_form[0].get_rows_number() == 1, LOG);
   assert_true(Hessian_form[0].get_columns_number() == 1, LOG);
   assert_true(Hessian_form[0] == 0.0, LOG);

   // Test

   if(numerical_differentiation_tests)
   {
      pl.set(2, 1);

      inputs.set(2);
      inputs.randomize_normal();

      Hessian_form = pl.calculate_Hessian_form(inputs);

      numerical_Hessian_form = nd.calculate_Hessian_form(pl, &PerceptronLayer::calculate_outputs, inputs);

      assert_true((Hessian_form[0]-numerical_Hessian_form[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   pl.set(2, 2);

   inputs.set(2);
   inputs.randomize_normal();

   Hessian_form = pl.calculate_Hessian_form(inputs);

   assert_true(Hessian_form.size() == 2, LOG);
   assert_true(Hessian_form[0].get_rows_number() == 2, LOG);
   assert_true(Hessian_form[0].get_columns_number() == 2, LOG);
   assert_true(Hessian_form[1].get_rows_number() == 2, LOG);
   assert_true(Hessian_form[1].get_columns_number() == 2, LOG);

   if(numerical_differentiation_tests)
   {
      numerical_Hessian_form = nd.calculate_Hessian_form(pl, &PerceptronLayer::calculate_outputs, inputs);

      assert_true((Hessian_form[0]-numerical_Hessian_form[0]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((Hessian_form[1]-numerical_Hessian_form[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   pl.set(4, 2);

   parameters.set(10);
   parameters[0] =  0.41;
   parameters[1] = -0.68; 
   parameters[2] =  0.14; 
   parameters[3] = -0.50; 
   parameters[4] =  0.52; 
   parameters[5] = -0.70; 
   parameters[6] =  0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] =  0.05; 

   pl.set_parameters(parameters);

   inputs.set(4);
   inputs[0] =  0.85;
   inputs[1] = -0.25;
   inputs[2] =  0.29;
   inputs[3] = -0.77;

   Hessian_form = pl.calculate_Hessian_form(inputs);

   assert_true((pl.get_perceptron(0).calculate_Hessian(inputs) - Hessian_form[0]).calculate_absolute_value() < 1.0e-3, LOG);
   assert_true((pl.get_perceptron(1).calculate_Hessian(inputs) - Hessian_form[1]).calculate_absolute_value() < 1.0e-3, LOG);

   if(numerical_differentiation_tests)
   {
      numerical_Hessian_form = nd.calculate_Hessian_form(pl, &PerceptronLayer::calculate_outputs, inputs);

      assert_true((Hessian_form[0]-numerical_Hessian_form[0]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((Hessian_form[1]-numerical_Hessian_form[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }

}


void PerceptronLayerTest::test_calculate_parameters_Jacobian()
{
   message += "test_calculate_parameters_Jacobian\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;

   Vector<double> inputs;

   Matrix<double> parameters_Jacobian;
   Matrix<double> numerical_parameters_Jacobian;

   // Test

   pl.set(4, 2);

   parameters.set(10);
   parameters[0] =  0.41;
   parameters[1] = -0.68; 
   parameters[2] =  0.14; 
   parameters[3] = -0.50; 
   parameters[4] =  0.52; 
   parameters[5] = -0.70; 
   parameters[6] =  0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] =  0.05; 

   pl.set_parameters(parameters);

   inputs.set(4);
   inputs[0] =  0.85;
   inputs[1] = -0.25;
   inputs[2] =  0.29;
   inputs[3] = -0.77;

   parameters_Jacobian = pl.calculate_Jacobian(inputs, parameters);

   assert_true(parameters_Jacobian.get_rows_number() == 2, LOG);
   assert_true(parameters_Jacobian.get_columns_number() == 10, LOG);
   
   if(numerical_differentiation_tests)
   {
      numerical_parameters_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_outputs, inputs, parameters);
      assert_true((parameters_Jacobian-numerical_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

}


void PerceptronLayerTest::test_calculate_parameters_Hessian_form()
{
   message += "test_calculate_parameters_Hessian_form\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;

   Vector<double> inputs;

   Vector< Matrix<double> > parameters_Hessian_form;
   Vector< Matrix<double> > numerical_parameters_Hessian_form;

   // Test

   pl.set(4, 2);

   parameters.set(10);
   parameters[0] =  0.41;
   parameters[1] = -0.68; 
   parameters[2] =  0.14; 
   parameters[3] = -0.50; 
   parameters[4] =  0.52; 
   parameters[5] = -0.70; 
   parameters[6] =  0.85; 
   parameters[7] = -0.18; 
   parameters[8] = -0.65; 
   parameters[9] =  0.05; 

   pl.set_parameters(parameters);

   inputs.set(4);
   inputs[0] =  0.85;
   inputs[1] = -0.25;
   inputs[2] =  0.29;
   inputs[3] = -0.77;

   parameters_Hessian_form = pl.calculate_Hessian_form(inputs, parameters);

   assert_true(parameters_Hessian_form.size() == 2, LOG);
   assert_true(parameters_Hessian_form[0].get_rows_number() == 10, LOG);
   assert_true(parameters_Hessian_form[0].get_columns_number() == 10, LOG);
   assert_true(parameters_Hessian_form[1].get_rows_number() == 10, LOG);
   assert_true(parameters_Hessian_form[1].get_columns_number() == 10, LOG);
   
   if(numerical_differentiation_tests)
   {
      numerical_parameters_Hessian_form = nd.calculate_Hessian_form(pl, &PerceptronLayer::calculate_outputs, inputs, parameters);

      assert_true((parameters_Hessian_form[0]-numerical_parameters_Hessian_form[0]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((parameters_Hessian_form[1]-numerical_parameters_Hessian_form[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }

}


void PerceptronLayerTest::test_write_expression()
{
   message += "test_write_expression\n";
}


void PerceptronLayerTest::run_test_case()
{
   message += "Running perceptron layer test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();

   // Get methods

   // PerceptronLayer arrangement

   test_count_inputs_number();
   test_get_perceptrons_number();

   // PerceptronLayer parameters

   test_count_parameters_number();
   test_count_cumulative_parameters_number();

   test_arrange_biases();
   test_arrange_synaptic_weights();
   test_arrange_parameters();

   // Activation functions

   test_get_activation_function();
   test_get_activation_function_name();

   test_get_activation_function();
   test_get_activation_function_name();

   // Display messages

   test_get_display();

   // Set methods

   test_set();
   test_set_default();

   // Perceptron layer parameters

   test_set_biases();
   test_set_parameters();

   test_set_synaptic_weights();      
   test_set_synaptic_weights();
   test_set_parameters();

   // Activation functions

   test_set_activation_function();
   test_set_activation_function();

   // Parameters methods

   test_set_parameters();

   // Display messages

   test_set_display();

   // Growing and pruning

   test_grow_inputs();
   test_grow_perceptrons();

   test_prune_input();
   test_prune_perceptron();

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

   // PerceptronLayer combination

   test_calculate_combination();

   test_calculate_combination_Jacobian();   
   test_calculate_combination_Hessian_form();

   // PerceptronLayer parameters combination

   test_calculate_combination_parameters_Jacobian();
   test_calculate_combination_parameters_Hessian_form();

   // PerceptronLayer activation 

   test_calculate_activation();
   test_calculate_activation_derivative();
   test_calculate_activation_second_derivative();

   // PerceptronLayer outputs 

   test_calculate_outputs();

   test_calculate_Jacobian();
   test_calculate_Hessian_form();

   // PerceptronLayer parameters outputs

   test_calculate_parameters_Jacobian();
   test_calculate_parameters_Hessian_form();

   // Expression methods

   test_write_expression();

   message += "End of perceptron layer test case.\n";
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
