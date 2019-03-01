/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P E R C E P T R O N   L A Y E R   T E S T   C L A S S                                                      */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
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


void PerceptronLayerTest::test_get_inputs_number()
{
   message += "test_get_inputs_number\n";

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
   
   pl.set_activation_function(PerceptronLayer::Logistic);
   assert_true(pl.get_activation_function() == PerceptronLayer::Logistic, LOG);

   pl.set_activation_function(PerceptronLayer::HyperbolicTangent);
   assert_true(pl.get_activation_function() == PerceptronLayer::HyperbolicTangent, LOG);

   pl.set_activation_function(PerceptronLayer::Threshold);
   assert_true(pl.get_activation_function() == PerceptronLayer::Threshold, LOG);

   pl.set_activation_function(PerceptronLayer::SymmetricThreshold);
   assert_true(pl.get_activation_function() == PerceptronLayer::SymmetricThreshold, LOG);

   pl.set_activation_function(PerceptronLayer::Linear);
   assert_true(pl.get_activation_function() == PerceptronLayer::Linear, LOG);

}


void PerceptronLayerTest::test_write_activation_function()
{
   message += "test_write_activation_function\n";

   PerceptronLayer pl(1, 1);

   pl.set_activation_function(PerceptronLayer::Logistic);
   assert_true(pl.write_activation_function() == "Logistic", LOG);

   pl.set_activation_function(PerceptronLayer::HyperbolicTangent);
   assert_true(pl.write_activation_function() == "HyperbolicTangent", LOG);

   pl.set_activation_function(PerceptronLayer::Threshold);
   assert_true(pl.write_activation_function() == "Threshold", LOG);

   pl.set_activation_function(PerceptronLayer::SymmetricThreshold);
   assert_true(pl.write_activation_function() == "SymmetricThreshold", LOG);

   pl.set_activation_function(PerceptronLayer::Linear);
   assert_true(pl.write_activation_function() == "Linear", LOG);
}


void PerceptronLayerTest::test_get_parameters_number()
{      
   message += "test_get_parameters_number\n";

   PerceptronLayer pl;

   // Test

   pl.set(1, 1);

   assert_true(pl.get_parameters_number() == 2, LOG);

   // Test

   pl.set(3, 1);

   assert_true(pl.get_parameters_number() == 4, LOG);

   // Test

   pl.set(2, 4);

   assert_true(pl.get_parameters_number() == 12, LOG);

   // Test

   pl.set(4, 2);

   assert_true(pl.get_parameters_number() == 10, LOG);

}


void PerceptronLayerTest::test_set()
{
   message += "test_set\n";
}


void PerceptronLayerTest::test_set_default()
{
   message += "test_set_default\n";
}


void PerceptronLayerTest::test_get_biases()
{
   message += "test_get_biases\n";

   PerceptronLayer pl;
   Vector<double> biases;

   // Test

   pl.set(1, 1);
   pl.initialize_parameters(0.0);

   biases = pl.get_biases();

   assert_true(biases.size() == 1, LOG);
   assert_true(biases[0] == 0.0, LOG);
}


void PerceptronLayerTest::test_get_synaptic_weights()
{
   message += "test_get_synaptic_weights\n";

   PerceptronLayer pl;

   Matrix<double> synaptic_weights;

   // Test

   pl.set(1, 1);

   pl.initialize_parameters(0.0);

   synaptic_weights = pl.get_synaptic_weights();

   assert_true(synaptic_weights.get_rows_number() == 1, LOG);
   assert_true(synaptic_weights.get_columns_number() == 1, LOG);
   assert_true(synaptic_weights == 0.0, LOG);
}


void PerceptronLayerTest::test_get_parameters()
{
   message += "test_get_parameters\n";

   PerceptronLayer pl;
   Vector<double> biases;
   Matrix<double> synaptic_weights;
   Vector<double> parameters;

   // Test

   pl.set(1, 1);
   pl.initialize_parameters(1.0);

   parameters = pl.get_parameters();

   assert_true(parameters.size() == 2, LOG);
   assert_true(parameters == 1.0, LOG);

   // Test

//   pl.set(2, 4);

//   biases.set(4);
//   biases[0] =  0.85;
//   biases[1] = -0.25;
//   biases[2] =  0.29;
//   biases[3] = -0.77;

//   pl.set_biases(biases);

//   synaptic_weights.set(4, 2);

//   synaptic_weights(0,0) = -0.04;
//   synaptic_weights(0,1) =  0.87;

//   synaptic_weights(1,0) =  0.25;
//   synaptic_weights(1,1) = -0.27;

//   synaptic_weights(2,0) = -0.57;
//   synaptic_weights(2,1) =  0.15;

//   synaptic_weights(3,0) =  0.96;
//   synaptic_weights(3,1) = -0.48;

//   pl.set_synaptic_weights(synaptic_weights);

//   parameters = pl.get_parameters();

//   assert_true(parameters.size() == 12, LOG);
//   assert_true(fabs(parameters[0] - 0.85) < numeric_limits<double>::epsilon(), LOG);
//   assert_true(fabs(parameters[11] - -0.48) < numeric_limits<double>::epsilon(), LOG);
}


void PerceptronLayerTest::test_get_perceptrons_parameters()
{
    message += "test_get_perceptrons_parameters\n";

//    PerceptronLayer pl;
//    Vector<double> biases;
//    Matrix<double> synaptic_weights;
//    Vector< Vector<double> > perceptrons_parameters;
//    Vector<double> v(3);

//    pl.set(2, 4);

//    biases.set(4);
//    biases[0] =  0.85;
//    biases[1] = -0.25;
//    biases[2] =  0.29;
//    biases[3] = -0.77;

//    pl.set_biases(biases);

//    synaptic_weights.set(4, 2);

//    synaptic_weights(0,0) = -0.04;
//    synaptic_weights(0,1) =  0.87;

//    synaptic_weights(1,0) =  0.25;
//    synaptic_weights(1,1) = -0.27;

//    synaptic_weights(2,0) = -0.57;
//    synaptic_weights(2,1) =  0.15;

//    synaptic_weights(3,0) =  0.96;
//    synaptic_weights(3,1) = -0.48;

//    pl.set_synaptic_weights(synaptic_weights);

//    perceptrons_parameters = pl.get_perceptrons_parameters();

//    v[0] = 0.85;
//    v[1] = -0.04;
//    v[2] = 0.87;

//    assert_true(perceptrons_parameters[0] == v , LOG);
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

   assert_true(pl.get_biases() == biases, LOG);
}


void PerceptronLayerTest::test_set_synaptic_weights()
{
   message += "test_set_synaptic_weights\n";

   PerceptronLayer pl(1, 1);

   Matrix<double> synaptic_weights(1, 1, 0.0);

   pl.set_synaptic_weights(synaptic_weights);

   assert_true(pl.get_synaptic_weights() == synaptic_weights, LOG);
}


void PerceptronLayerTest::test_set_inputs_number()
{
    message += "test_set_inputs_number\n";

    PerceptronLayer pl;
    Vector<double> biases;
    Matrix<double> synaptic_weights;
    Vector<double> new_biases;
    Vector<double> new_synaptic_weights;

    pl.set(3, 2);

    biases.set(2);
    biases[0] =  0.85;
    biases[1] = -0.25;

    pl.set_biases(biases);

    synaptic_weights.set(2, 3);

    synaptic_weights(0,0) = -0.04;
    synaptic_weights(0,1) =  0.87;
    synaptic_weights(0,2) =  0.25;

    synaptic_weights(1,0) = -0.27;
    synaptic_weights(1,1) = -0.57;
    synaptic_weights(1,2) =  0.15;

    pl.set_synaptic_weights(synaptic_weights);

    size_t new_inputs_number = 2;

    pl.set_inputs_number(new_inputs_number);

    new_biases = pl.get_biases();
    new_synaptic_weights = pl.get_synaptic_weights();

    assert_true(biases.size() == new_biases.size(), LOG);
    assert_true(synaptic_weights.size() != new_synaptic_weights.size(), LOG);
}


void PerceptronLayerTest::test_set_perceptrons_number()
{
    message += "test_set_perceptrons_number\n";

    PerceptronLayer pl;
    Vector<double> biases;
    Matrix<double> synaptic_weights;
    Vector<double> new_biases;
    Vector<double> new_synaptic_weights;

    pl.set(3, 2);

    biases.set(2);
    biases[0] =  0.85;
    biases[1] = -0.25;

    pl.set_biases(biases);

    synaptic_weights.set(2, 3);

    synaptic_weights(0,0) = -0.04;
    synaptic_weights(0,1) =  0.87;
    synaptic_weights(0,2) =  0.25;

    synaptic_weights(1,0) = -0.27;
    synaptic_weights(1,1) = -0.57;
    synaptic_weights(1,2) =  0.15;

    pl.set_synaptic_weights(synaptic_weights);

    size_t new_perceptrons_number = 1;

    pl.set_perceptrons_number(new_perceptrons_number);

    new_biases = pl.get_biases();
    new_synaptic_weights = pl.get_synaptic_weights();

    assert_true(biases.size() != new_biases.size(), LOG);
    assert_true(synaptic_weights.size() != new_synaptic_weights.size(), LOG);
}


void PerceptronLayerTest::test_set_parameters()
{
   message += "test_set_parameters\n";

   PerceptronLayer pl(1, 1);

   Vector<double> parameters(2.0,1,3.0);

   pl.set_parameters(parameters);

   assert_true(pl.get_parameters() == parameters, LOG);
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

//   PerceptronLayer pl;

   // Test

//   pl.set(1, 1);
//   pl.grow_perceptrons(1);

//   assert_true(pl.get_inputs_number() == 1, LOG);
//   assert_true(pl.get_perceptrons_number() == 2, LOG);
}


void PerceptronLayerTest::test_prune_input()
{
   message += "test_prune_input\n";

//   PerceptronLayer pl;

   // Test

//   pl.set(1, 1);
//   pl.prune_input(0);

//   assert_true(pl.get_inputs_number() == 0, LOG);
//   assert_true(pl.get_perceptrons_number() == 1, LOG);
}


void PerceptronLayerTest::test_prune_perceptron()
{
   message += "test_prune_perceptron\n";

   PerceptronLayer pl;

   // Test

//   pl.set(1, 1);
//   pl.prune_perceptron(0);

//   assert_true(pl.get_inputs_number() == 0, LOG);
//   assert_true(pl.get_perceptrons_number() == 0, LOG);
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

   parameters = pl.get_parameters();

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
   parameters = pl.get_parameters();
   
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
   parameters = pl.get_parameters();

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

   biases.set(4, 1.0);
   pl.set_biases(biases);

   synaptic_weights.set(2, 4, -1.0);
   pl.set_synaptic_weights(synaptic_weights);

   parameters = pl.get_parameters();

   parameters_norm = pl.calculate_parameters_norm();

   assert_true(fabs(parameters_norm - parameters.calculate_L2_norm()) < 1.0e-6, LOG);

   // Test

   pl.set(4, 2);

   parameters.set(10, 1.0);
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

   assert_true(fabs(parameters_norm - parameters.calculate_L2_norm()) < 1.0e-6, LOG);
}


void PerceptronLayerTest::test_calculate_combinations()
{
   message += "test_calculate_combinations\n";
/*
   PerceptronLayer pl;

   Vector<double> biases;
   Matrix<double> synaptic_weights;
   Vector<double> parameters;

   Vector<double> inputs;   

   Vector<double> combinations;

   // Test
 
   pl.set(1, 2);
   pl.initialize_parameters(0.0);

   inputs.set(1, 0.0);

   combinations = pl.calculate_combinations(inputs);

   assert_true(combinations.size() == 2, LOG);
   assert_true(combinations == 0.0, LOG);

   // Test

   pl.set(2, 4);

   biases.set(4);
   biases[0] =  0.85;
   biases[1] = -0.25;
   biases[2] =  0.29;
   biases[3] = -0.77;

   pl.set_biases(biases);

   synaptic_weights.set(2, 4, 1.0);

//   synaptic_weights(0,0) = -0.04;
//   synaptic_weights(0,1) =  0.87;

//   synaptic_weights(1,0) =  0.25;
//   synaptic_weights(1,1) = -0.27;

//   synaptic_weights(2,0) = -0.57;
//   synaptic_weights(2,1) =  0.15;

//   synaptic_weights(3,0) =  0.96;
//   synaptic_weights(3,1) = -0.48;

   pl.set_synaptic_weights(synaptic_weights);

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

   combinations = pl.calculate_combinations(inputs);

//   assert_true(combinations - (biases + synaptic_weights.dot(inputs)) < 1.0e-3, LOG);

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

   combinations = pl.calculate_combinations(inputs);

   biases = pl.get_biases();
   synaptic_weights = pl.get_synaptic_weights();

//   assert_true(combination - (biases + synaptic_weights.dot(inputs)).calculate_absolute_value() < 1.0e-6, LOG);

   // Test

   pl.set(1, 1);

   inputs.set(1);
   inputs.randomize_normal();

   parameters = pl.get_parameters();

//   assert_true(pl.calculate_combinations(inputs) == pl.calculate_combinations(inputs, parameters), LOG);
*/
}


void PerceptronLayerTest::test_calculate_combinations_Jacobian()
{
   message += "test_calculate_combinations_Jacobian\n";
/*
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

   synaptic_weights = pl.get_synaptic_weights();

   assert_true(combination_Jacobian == synaptic_weights, LOG);

   if(numerical_differentiation_tests)
   {
      numerical_combination_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_combinations, inputs);
      assert_true((combination_Jacobian-numerical_combination_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }
*/
}


void PerceptronLayerTest::test_calculate_combinations_Hessian()
{
   message += "test_calculate_combination_Hessian\n";
/*
   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;
   Vector<double> inputs;

   Vector< Matrix<double> > combination_Hessian;
   Vector< Matrix<double> > numerical_combination_Hessian;

   // Test

   pl.set(2, 4);

   inputs.set(2);
   inputs.randomize_normal();

   combination_Hessian = pl.calculate_combinations_Hessian(inputs);

   assert_true(combination_Hessian.size() == 4, LOG);

   if(numerical_differentiation_tests)
   {
      numerical_combination_Hessian = nd.calculate_Hessian(pl, &PerceptronLayer::calculate_combinations, inputs);

      assert_true((combination_Hessian[0]-numerical_combination_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_Hessian[1]-numerical_combination_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_Hessian[2]-numerical_combination_Hessian[2]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_Hessian[3]-numerical_combination_Hessian[3]).calculate_absolute_value() < 1.0e-3, LOG);
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

   combination_Hessian = pl.calculate_combinations_Hessian(inputs);

   assert_true(combination_Hessian.size() == 2, LOG);

   assert_true(combination_Hessian[0].get_rows_number() == 4, LOG);
   assert_true(combination_Hessian[0].get_columns_number() == 4, LOG);
   assert_true(combination_Hessian[0] == 0.0, LOG);
   assert_true(combination_Hessian[0].is_symmetric(), LOG);

   assert_true(combination_Hessian[1].get_rows_number() == 4, LOG);
   assert_true(combination_Hessian[1].get_columns_number() == 4, LOG);
   assert_true(combination_Hessian[1] == 0.0, LOG);
   assert_true(combination_Hessian[1].is_symmetric(), LOG);

   if(numerical_differentiation_tests)
   {
      numerical_combination_Hessian = nd.calculate_Hessian(pl, &PerceptronLayer::calculate_combinations, inputs);

      assert_true((combination_Hessian[0]-numerical_combination_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((combination_Hessian[1]-numerical_combination_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }
*/
}


void PerceptronLayerTest::test_calculate_combinations_parameters_Jacobian()
{
   message += "test_calculate_combination_parameters_Jacobian\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;

   Vector<double> inputs;

   Matrix<double> combination_parameters_Jacobian;
   Matrix<double> numerical_combination_parameters_Jacobian;

   // Test
/*
   pl.set(2, 4);

   parameters = pl.get_parameters();

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

   combination_parameters_Jacobian = pl.calculate_combinations_parameters_Jacobian(inputs);

   if(numerical_differentiation_tests)
   {
      numerical_combination_parameters_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_combinations, inputs, parameters);

      assert_true((combination_parameters_Jacobian-numerical_combination_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }
*/
   // Test

   pl.set(2, 2);
   pl.randomize_parameters_normal();
   parameters = pl.get_parameters();

   /*parameters.set(10);
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

   inputs.set(2);
   inputs.randomize_normal();
   inputs[0] =  0.85;
   inputs[1] = -0.25;
   inputs[2] =  0.29;
   inputs[3] = -0.77;

   combination_parameters_Jacobian = pl.calculate_combinations_parameters_Jacobian(inputs);

//   if(numerical_differentiation_tests)
   {
      numerical_combination_parameters_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_combinations, inputs, parameters);

      assert_true((combination_parameters_Jacobian-numerical_combination_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }
*/
}


void PerceptronLayerTest::test_calculate_combinations_parameters_Hessian()
{
   message += "test_calculate_combination_parameters_Hessian\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   size_t parameters_number;
   Vector<double> parameters;

   Vector<double> inputs;

   Vector< Matrix<double> > combination_parameters_Hessian;
   Vector< Matrix<double> > numerical_combination_parameters_Hessian;

   // Test

   pl.set(2, 4);

   parameters_number = pl.get_parameters_number();
   parameters = pl.get_parameters();

   inputs.set(2);
   inputs[0] = -0.88;
   inputs[1] =  0.78;

//   combination_parameters_Hessian = pl.calculate_combinations_Hessian(inputs, parameters);

//   assert_true(combination_parameters_Hessian.size() == 4, LOG);
//   assert_true(combination_parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[0].calculate_absolute_value() < 1.0e-6 , LOG);

//   assert_true(combination_parameters_Hessian[1].get_rows_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[1].get_columns_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[1].calculate_absolute_value() < 1.0e-6 , LOG);

//   assert_true(combination_parameters_Hessian[2].get_rows_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[2].get_columns_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[2].calculate_absolute_value() < 1.0e-6 , LOG);

//   assert_true(combination_parameters_Hessian[3].get_rows_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[3].get_columns_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[3].calculate_absolute_value() < 1.0e-6 , LOG);

//   if(numerical_differentiation_tests)
//   {
//      numerical_combination_parameters_Hessian = nd.calculate_Hessian(pl, &PerceptronLayer::calculate_combinations, inputs, parameters);

//      assert_true((combination_parameters_Hessian[0]-numerical_combination_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//      assert_true((combination_parameters_Hessian[1]-numerical_combination_parameters_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
//      assert_true((combination_parameters_Hessian[2]-numerical_combination_parameters_Hessian[2]).calculate_absolute_value() < 1.0e-3, LOG);
//      assert_true((combination_parameters_Hessian[3]-numerical_combination_parameters_Hessian[3]).calculate_absolute_value() < 1.0e-3, LOG);
//   }

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

   parameters_number = pl.get_parameters_number();

//   combination_parameters_Hessian = pl.calculate_combinations_Hessian(inputs, parameters);

//   assert_true(combination_parameters_Hessian.size() == 2, LOG);
//   assert_true(combination_parameters_Hessian[0].get_rows_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[0].get_columns_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[0].calculate_absolute_value() < 1.0e-6 , LOG);

//   assert_true(combination_parameters_Hessian[1].get_rows_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[1].get_columns_number() == parameters_number, LOG);
//   assert_true(combination_parameters_Hessian[1].calculate_absolute_value() < 1.0e-6 , LOG);

   if(numerical_differentiation_tests)
   {
//      numerical_combination_parameters_Hessian = nd.calculate_Hessian(pl, &PerceptronLayer::calculate_combinations, inputs, parameters);

//      assert_true((combination_parameters_Hessian[0]-numerical_combination_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//      assert_true((combination_parameters_Hessian[1]-numerical_combination_parameters_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }
}


void PerceptronLayerTest::test_calculate_activations()
{
   message += "test_calculate_activations\n";

   PerceptronLayer pl;

   Vector<double> parameters;
 
   Matrix<double> inputs;
   Matrix<double> combination;
   Matrix<double> activation;

   // Test

   pl.set(2,2);

   Matrix<double> weights(2,2,1.0);
   Vector<double> biases(2,1.0);

   pl.set_synaptic_weights(weights);
   pl.set_biases(biases);
//   pl.initialize_random();
   pl.set_activation_function(PerceptronLayer::RectifiedLinear);

   inputs.set(4,2);
   inputs.randomize_normal();

   combination = pl.calculate_combinations(inputs);

   activation = pl.calculate_activations(combination);

   cout << "combination: " << combination << endl;
   cout << "Activation: " << activation << endl;
   cout << "Absolute values: " << activation.calculate_sum() << endl;

   assert_true(activation.get_rows_number() == 4, LOG);
   assert_true(activation.get_columns_number() == 2, LOG);
   assert_true(fabs(activation.calculate_sum() - 0.0) < std::numeric_limits<double>::min(), LOG);

/*
   // Test

   pl.set(1, 2);
   pl.initialize_parameters(0.0);

   combination.set(2, 0.0);

   pl.set_activation_function(PerceptronLayer::Logistic);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);
   assert_true(activation == 0.5, LOG);

   // Test

   pl.set(1, 2);
   pl.initialize_parameters(0.0);

   combination.set(2, 0.0);

   pl.set_activation_function(PerceptronLayer::HyperbolicTangent);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);
   assert_true(activation == 0.0, LOG);

   // Test

   pl.set(1, 2);
   pl.initialize_parameters(0.0);

   combination.set(2, 0.0);

   pl.set_activation_function(PerceptronLayer::Threshold);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);
   assert_true(activation == 1.0, LOG);

   // Test

   pl.set(1, 2);
   pl.initialize_parameters(0.0);

   combination.set(2, 0.0);

   pl.set_activation_function(PerceptronLayer::SymmetricThreshold);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);
   assert_true(activation == 1.0, LOG);

   // Test

   pl.set(1, 2);
   pl.initialize_parameters(0.0);

   combination.set(2, 0.0);

   pl.set_activation_function(PerceptronLayer::Linear);
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

   pl.set_activation_function(PerceptronLayer::Threshold);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);

   pl.set_activation_function(PerceptronLayer::SymmetricThreshold);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);

   pl.set_activation_function(PerceptronLayer::Logistic);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);

   pl.set_activation_function(PerceptronLayer::HyperbolicTangent);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);

   pl.set_activation_function(PerceptronLayer::Linear);
   activation = pl.calculate_activations(combination);
   assert_true(activation.size() == 2, LOG);
*/
}


void PerceptronLayerTest::test_calculate_activations_derivatives()
{
   message += "test_calculate_activation_derivative\n";
/*
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

   pl.set_activation_function(PerceptronLayer::Logistic);
   activation_derivative = pl.calculate_activations_derivatives(combination);
   assert_true(activation_derivative.size() == 2, LOG);
   assert_true(activation_derivative == 0.25, LOG);

   pl.set_activation_function(PerceptronLayer::HyperbolicTangent);
   activation_derivative = pl.calculate_activations_derivatives(combination);
   assert_true(activation_derivative.size() == 2, LOG);
   assert_true(activation_derivative == 1.0, LOG);

   pl.set_activation_function(PerceptronLayer::Linear);
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

      pl.set_activation_function(PerceptronLayer::Threshold);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::SymmetricThreshold);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::Logistic);
      activation_derivative = pl.calculate_activations_derivatives(combination);

      numerical_activation_derivative = nd.calculate_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::HyperbolicTangent);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::Linear);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
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

      pl.set_activation_function(PerceptronLayer::Threshold);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::SymmetricThreshold);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::Logistic);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::HyperbolicTangent);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::Linear);
      activation_derivative = pl.calculate_activations_derivatives(combination);
      numerical_activation_derivative = nd.calculate_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_derivative - numerical_activation_derivative).calculate_absolute_value() < 1.0e-3, LOG);
   }
*/
}


void PerceptronLayerTest::test_calculate_activations_second_derivatives()
{
   message += "test_calculate_activation_second_derivative\n";
/*
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

   pl.set_activation_function(PerceptronLayer::Logistic);
   activation_second_derivative  = pl.calculate_activations_second_derivatives(combination);
   assert_true(activation_second_derivative.size() == 2, LOG);
   assert_true(activation_second_derivative == 0.0, LOG);

   pl.set_activation_function(PerceptronLayer::HyperbolicTangent);
   activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
   assert_true(activation_second_derivative.size() == 2, LOG);
   assert_true(activation_second_derivative == 0.0, LOG);

   pl.set_activation_function(PerceptronLayer::Linear);
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

      pl.set_activation_function(PerceptronLayer::Threshold);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::SymmetricThreshold);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::Logistic);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::HyperbolicTangent);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::Linear);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
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

      pl.set_activation_function(PerceptronLayer::Threshold);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::SymmetricThreshold);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::Logistic);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::HyperbolicTangent);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);

      pl.set_activation_function(PerceptronLayer::Linear);
      activation_second_derivative = pl.calculate_activations_second_derivatives(combination);
      numerical_activation_second_derivative = nd.calculate_second_derivatives(pl, &PerceptronLayer::calculate_activations, combination);
      assert_true((activation_second_derivative - numerical_activation_second_derivative).calculate_absolute_value() < 1.0e-3, LOG);
   }
*/
}


void PerceptronLayerTest::test_calculate_outputs()
{
/*
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

//   potential_outputs = pl.calculate_outputs(inputs, parameters);

//   assert_true(outputs != potential_outputs, LOG);

   // Test

   pl.set(1, 1);

   inputs.set(1);
   inputs.randomize_normal();

   parameters = pl.get_parameters();

//   assert_true(pl.calculate_outputs(inputs) == pl.calculate_outputs(inputs, parameters), LOG);
*/
}


void PerceptronLayerTest::test_calculate_Jacobian()
{
   message += "test_calculate_Jacobian\n";
/*
   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;

   Vector<double> inputs;

   Matrix<double> Jacobian;
   Matrix<double> numerical_Jacobian;

   // Test
numerical_differentiation_tests = true;
   if(numerical_differentiation_tests)
   {
      pl.set(3, 2);

      inputs.set(3);
      inputs.randomize_normal();

      Jacobian = pl.calculate_Jacobian(inputs);

      numerical_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_outputs, inputs);
      cout << "Jacobian: " << Jacobian << endl;
      cout << "numerical Jacobian: " << numerical_Jacobian << endl;
      assert_true((Jacobian-numerical_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test
/*
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
   */
}


void PerceptronLayerTest::test_calculate_Hessian()
{
   message += "test_calculate_Hessian\n";
/*
   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;

   Vector<double> inputs;

   Vector< Matrix<double> > Hessian;
   Vector< Matrix<double> > numerical_Hessian;

   Matrix<double> Hessian;

   // Test

   pl.set(1, 1);
   pl.initialize_parameters(0.0);

   inputs.set(1);
   inputs.initialize(0.0);

   Hessian = pl.calculate_Hessian(inputs);

   assert_true(Hessian.size() == 1, LOG);
   assert_true(Hessian[0].get_rows_number() == 1, LOG);
   assert_true(Hessian[0].get_columns_number() == 1, LOG);
   assert_true(Hessian[0] == 0.0, LOG);

   // Test

   if(numerical_differentiation_tests)
   {
      pl.set(2, 1);

      inputs.set(2);
      inputs.randomize_normal();

      Hessian = pl.calculate_Hessian(inputs);

      numerical_Hessian = nd.calculate_Hessian(pl, &PerceptronLayer::calculate_outputs, inputs);

      assert_true((Hessian[0]-numerical_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
   }

   // Test

   pl.set(2, 2);

   inputs.set(2);
   inputs.randomize_normal();

   Hessian = pl.calculate_Hessian(inputs);

   assert_true(Hessian.size() == 2, LOG);
   assert_true(Hessian[0].get_rows_number() == 2, LOG);
   assert_true(Hessian[0].get_columns_number() == 2, LOG);
   assert_true(Hessian[1].get_rows_number() == 2, LOG);
   assert_true(Hessian[1].get_columns_number() == 2, LOG);

   if(numerical_differentiation_tests)
   {
      numerical_Hessian = nd.calculate_Hessian(pl, &PerceptronLayer::calculate_outputs, inputs);

      assert_true((Hessian[0]-numerical_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((Hessian[1]-numerical_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
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

   Hessian = pl.calculate_Hessian(inputs);

//   assert_true((pl.get_perceptron(0).calculate_Hessian(inputs) - Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//   assert_true((pl.get_perceptron(1).calculate_Hessian(inputs) - Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);

   if(numerical_differentiation_tests)
   {
      numerical_Hessian = nd.calculate_Hessian(pl, &PerceptronLayer::calculate_outputs, inputs);

      assert_true((Hessian[0]-numerical_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
      assert_true((Hessian[1]-numerical_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }
*/
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

//   parameters_Jacobian = pl.calculate_Jacobian(inputs, parameters);

//   assert_true(parameters_Jacobian.get_rows_number() == 2, LOG);
//   assert_true(parameters_Jacobian.get_columns_number() == 10, LOG);
   
//   if(numerical_differentiation_tests)
//   {
//      numerical_parameters_Jacobian = nd.calculate_Jacobian(pl, &PerceptronLayer::calculate_outputs, inputs, parameters);
//      assert_true((parameters_Jacobian-numerical_parameters_Jacobian).calculate_absolute_value() < 1.0e-3, LOG);
//   }
}


void PerceptronLayerTest::test_calculate_parameters_Hessian()
{
   message += "test_calculate_parameters_Hessian\n";

   NumericalDifferentiation nd;

   PerceptronLayer pl;

   Vector<double> parameters;

   Vector<double> inputs;

   Vector< Matrix<double> > parameters_Hessian;
   Vector< Matrix<double> > numerical_parameters_Hessian;

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

//   parameters_Hessian = pl.calculate_Hessian(inputs, parameters);

//   assert_true(parameters_Hessian.size() == 2, LOG);
//   assert_true(parameters_Hessian[0].get_rows_number() == 10, LOG);
//   assert_true(parameters_Hessian[0].get_columns_number() == 10, LOG);
//   assert_true(parameters_Hessian[1].get_rows_number() == 10, LOG);
//   assert_true(parameters_Hessian[1].get_columns_number() == 10, LOG);
   
   if(numerical_differentiation_tests)
   {
//      numerical_parameters_Hessian = nd.calculate_Hessian(pl, &PerceptronLayer::calculate_outputs, inputs, parameters);

//      assert_true((parameters_Hessian[0]-numerical_parameters_Hessian[0]).calculate_absolute_value() < 1.0e-3, LOG);
//      assert_true((parameters_Hessian[1]-numerical_parameters_Hessian[1]).calculate_absolute_value() < 1.0e-3, LOG);
   }

}


void PerceptronLayerTest::test_write_expression()
{
   message += "test_write_expression\n";
}


void PerceptronLayerTest::run_test_case()
{
   message += "Running perceptron layer test case...\n";

   // Constructor and destructor
/*
   test_constructor();
   test_destructor();

   // Assignment operators

   test_assignment_operator();

   // Get methods

   // Inputs and perceptrons

   test_get_inputs_number();
   test_get_perceptrons_number();

   // Parameters

   test_get_parameters_number();

   test_get_biases();
   test_get_synaptic_weights();
   test_get_parameters();
   test_get_perceptrons_parameters();

   // Activation functions

   test_get_activation_function();
   test_write_activation_function();

   // Display messages

   test_get_display();

   // Set methods

   test_set();
   test_set_default();

   // Perceptron layer parameters

   test_set_biases();

   test_set_synaptic_weights();

   test_set_perceptrons_number();

   // Inputs

   test_set_inputs_number();

   // Activation functions

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

   // Combination

   test_calculate_combinations();

   test_calculate_combinations_Jacobian();
   test_calculate_combinations_Hessian();

   // Parameters combination

   test_calculate_combinations_parameters_Jacobian();
   test_calculate_combinations_parameters_Hessian();
*/
   // Activation

   test_calculate_activations();
 /*  test_calculate_activations_derivatives();
   test_calculate_activations_second_derivatives();

   // Outputs

   test_calculate_outputs();

   test_calculate_Jacobian();
   test_calculate_Hessian();

   // Parameters outputs

   test_calculate_parameters_Jacobian();
   test_calculate_parameters_Hessian();

   // Expression methods

   test_write_expression();
*/
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
