//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "neural_network_test.h"
#include <omp.h>

NeuralNetworkTest::NeuralNetworkTest() : UnitTesting()
{
}


NeuralNetworkTest::~NeuralNetworkTest()
{
}


void NeuralNetworkTest::test_constructor()
{
   cout << "test_constructor\n";

  // Test 0 / Default constructor

  NeuralNetwork neural_network_0;

  assert_true(neural_network_0.is_empty(), LOG);
  assert_true(neural_network_0.get_layers_number() == 0, LOG);

  // Test 1 / Model type constructorr

  Tensor<Index, 1> architecture(3);

  architecture.setValues({1,4,2});

  NeuralNetwork neural_network_1_1(NeuralNetwork::Approximation, architecture);

  assert_true(neural_network_1_1.get_layers_number() == 5, LOG);
  assert_true(neural_network_1_1.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);
  assert_true(neural_network_1_1.get_layer_pointer(1)->get_type() == Layer::Perceptron, LOG);
  assert_true(neural_network_1_1.get_layer_pointer(2)->get_type() == Layer::Perceptron, LOG);
  assert_true(neural_network_1_1.get_layer_pointer(3)->get_type() == Layer::Unscaling, LOG);
  assert_true(neural_network_1_1.get_layer_pointer(4)->get_type() == Layer::Bounding, LOG);

  NeuralNetwork neural_network_1_2(NeuralNetwork::Classification, architecture);

  assert_true(neural_network_1_2.get_layers_number() == 3, LOG);
  assert_true(neural_network_1_2.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);
  assert_true(neural_network_1_2.get_layer_pointer(1)->get_type() == Layer::Perceptron, LOG);
  assert_true(neural_network_1_2.get_layer_pointer(2)->get_type() == Layer::Probabilistic, LOG);

  NeuralNetwork neural_network_1_3(NeuralNetwork::Forecasting, architecture);

  assert_true(neural_network_1_3.get_layers_number() == 4, LOG);
  assert_true(neural_network_1_3.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);
  assert_true(neural_network_1_3.get_layer_pointer(1)->get_type() == Layer::LongShortTermMemory, LOG);
  assert_true(neural_network_1_3.get_layer_pointer(2)->get_type() == Layer::Perceptron, LOG);
  assert_true(neural_network_1_3.get_layer_pointer(3)->get_type() == Layer::Unscaling, LOG);

  NeuralNetwork neural_network_1_4(NeuralNetwork::ImageApproximation, architecture);

  assert_true(neural_network_1_4.get_layers_number() == 1, LOG);
  assert_true(neural_network_1_4.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);

  NeuralNetwork neural_network_1_5(NeuralNetwork::ImageClassification, architecture);

  assert_true(neural_network_1_5.get_layers_number() == 1, LOG);
  assert_true(neural_network_1_5.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);

  // Test 2 / Convolutional layer constructor
/*
  Tensor<Index, 1> new_inputs_dimensions(1);
  new_inputs_dimensions.setConstant(1);

  Index new_blocks_number = 1;

  Tensor<Index, 1> new_filters_dimensions(1);
  new_filters_dimensions.setConstant(1);

  Index new_outputs_number = 1;

  ConvolutionalLayer convolutional_layer(1,1); //CC -> cl(inputs_dim, filters_dim)

  NeuralNetwork neural_network_2(new_inputs_dimensions, new_blocks_number, new_filters_dimensions, new_outputs_number);

  assert_true(neural_network_2.is_empty(), LOG);
  assert_true(neural_network_2.get_layers_number() == 0, LOG);
*/

  // Test 3_1 / Layers constructor //CCH -> Test other type of layers

  Tensor<Layer*, 1> layers_2;

  NeuralNetwork neural_network_3_1(layers_2);

  assert_true(neural_network_3_1.is_empty(), LOG);
  assert_true(neural_network_3_1.get_layers_number() == 0, LOG);

  PerceptronLayer* perceptron_layer_3_1 = new PerceptronLayer(1, 1);

  neural_network_3_1.add_layer(perceptron_layer_3_1);

  assert_true(!neural_network_3_1.is_empty(), LOG);
  assert_true(neural_network_3_1.get_layer_pointer(0)->get_type() == Layer::Perceptron, LOG);

  // Test 3_2

  Tensor<Layer*, 1> layers_3(10);

  layers_3.setValues({new ScalingLayer, new ConvolutionalLayer, new PerceptronLayer,
                          new PoolingLayer, new ProbabilisticLayer, new LongShortTermMemoryLayer,
                          new RecurrentLayer, new UnscalingLayer, new PrincipalComponentsLayer, new BoundingLayer});

  NeuralNetwork neural_network_3_2(layers_3);

  assert_true(!neural_network_3_2.is_empty(), LOG);
  assert_true(neural_network_3_2.get_layers_number() == 10, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(1)->get_type() == Layer::Convolutional, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(2)->get_type() == Layer::Perceptron, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(3)->get_type() == Layer::Pooling, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(4)->get_type() == Layer::Probabilistic, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(5)->get_type() == Layer::LongShortTermMemory, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(6)->get_type() == Layer::Recurrent, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(7)->get_type() == Layer::Unscaling, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(8)->get_type() == Layer::PrincipalComponents, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(9)->get_type() == Layer::Bounding, LOG);

  // Copy constructor


//  NeuralNetwork neural_network_4(neural_network_1_1);
//  assert_true(neural_network_4.get_layers_number() == 5, LOG);


   // File constructor

//   neural_network.save(file_name);
//   NeuralNetwork nn6(file_name);

}

void NeuralNetworkTest::test_destructor()
{
   cout << "test_destructor\n";

   NeuralNetwork* neural_network_1 = new NeuralNetwork;

   delete neural_network_1;
}

void NeuralNetworkTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

   NeuralNetwork nn1;
   NeuralNetwork nn2 = nn1;
}

void NeuralNetworkTest::test_get_display()
{
   cout << "test_get_display\n";
}

void NeuralNetworkTest::test_add_layer()
{
   cout << "test_add_layer\n";

   NeuralNetwork neural_network_;

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer;
   neural_network_.add_layer(long_short_term_memory_layer);
   assert_true(neural_network_.get_layers_number() == 1, LOG);
   assert_true(neural_network_.get_layer_pointer(0)->get_type() == Layer::LongShortTermMemory, LOG);

   NeuralNetwork neural_network__;

   RecurrentLayer* recurrent_layer = new RecurrentLayer;
   neural_network__.add_layer(recurrent_layer);
   assert_true(neural_network__.get_layers_number() == 1, LOG);
   assert_true(neural_network__.get_layer_pointer(0)->get_type() == Layer::Recurrent, LOG);

   NeuralNetwork neural_network;

   ScalingLayer* scaling_layer = new ScalingLayer(1);
   neural_network.add_layer(scaling_layer);
   assert_true(neural_network.get_layers_number() == 1, LOG);
   assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);

   ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer;
   neural_network.add_layer(convolutional_layer);
   assert_true(neural_network.get_layers_number() == 2, LOG);
   assert_true(neural_network.get_layer_pointer(1)->get_type() == Layer::Convolutional, LOG);

   PerceptronLayer* perceptron_layer = new PerceptronLayer;
   neural_network.add_layer(perceptron_layer);
   assert_true(neural_network.get_layers_number() == 3, LOG);
   assert_true(neural_network.get_layer_pointer(2)->get_type() == Layer::Perceptron, LOG);

   PoolingLayer* pooling_layer = new PoolingLayer;

   neural_network.add_layer(pooling_layer);
   assert_true(neural_network.get_layers_number() == 4, LOG);
   assert_true(neural_network.get_layer_pointer(3)->get_type() == Layer::Pooling, LOG);

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer;

   neural_network.add_layer(probabilistic_layer);
   assert_true(neural_network.get_layers_number() == 5, LOG);
   assert_true(neural_network.get_layer_pointer(4)->get_type() == Layer::Probabilistic, LOG);

   UnscalingLayer* unscaling_layer = new UnscalingLayer;

   neural_network.add_layer(unscaling_layer);
   assert_true(neural_network.get_layers_number() == 6, LOG);
   assert_true(neural_network.get_layer_pointer(5)->get_type() == Layer::Unscaling, LOG);

   BoundingLayer* bounding_layer = new BoundingLayer;

   neural_network.add_layer(bounding_layer);
   assert_true(neural_network.get_layers_number() == 7, LOG);
   assert_true(neural_network.get_layer_pointer(6)->get_type() == Layer::Bounding, LOG);

   PrincipalComponentsLayer* principal_components_layer = new PrincipalComponentsLayer;

   neural_network.add_layer(principal_components_layer);
   assert_true(neural_network.get_layers_number() == 8, LOG);
   assert_true(neural_network.get_layer_pointer(7)->get_type() == Layer::PrincipalComponents, LOG);
}

void NeuralNetworkTest::check_layer_type()
{
   cout << "check_layer_type\n";

   // Test 1

   Tensor<Layer*, 1> layers_tensor(2);
   layers_tensor.setValues({new PerceptronLayer, new PerceptronLayer});

   NeuralNetwork neural_network(layers_tensor);

   assert_true(!neural_network.check_layer_type(Layer::LongShortTermMemory), LOG);
   assert_true(!neural_network.check_layer_type(Layer::Recurrent), LOG);

   assert_true(neural_network.check_layer_type(Layer::Scaling), LOG);
   assert_true(neural_network.check_layer_type(Layer::Convolutional), LOG);
   assert_true(neural_network.check_layer_type(Layer::Perceptron), LOG);
   assert_true(neural_network.check_layer_type(Layer::Pooling), LOG);
   assert_true(neural_network.check_layer_type(Layer::Probabilistic), LOG);
   assert_true(neural_network.check_layer_type(Layer::Unscaling), LOG);
   assert_true(neural_network.check_layer_type(Layer::Bounding), LOG);
   assert_true(neural_network.check_layer_type(Layer::PrincipalComponents), LOG);

   // Test 2

   Tensor<Layer*, 1> layers_tensor_1(1);
   layers_tensor_1.setValues({new PerceptronLayer});

   NeuralNetwork neural_network_1(layers_tensor_1);

   assert_true(!neural_network_1.check_layer_type(Layer::LongShortTermMemory), LOG);
   assert_true(!neural_network_1.check_layer_type(Layer::Recurrent), LOG);

   // Test 3

   Tensor<Layer*, 1> layers_tensor_2(1);
   layers_tensor_2.setValues({new ScalingLayer});

   NeuralNetwork neural_network_2(layers_tensor_2);

   assert_true(neural_network_2.check_layer_type(Layer::LongShortTermMemory), LOG);
   assert_true(neural_network_2.check_layer_type(Layer::Recurrent), LOG);
}

void NeuralNetworkTest::test_has_methods()
{
   cout << "test_had_methods\n";

   // Test 0

   Tensor<Layer*, 1> layer_tensor;

   NeuralNetwork neural_network(layer_tensor);

   assert_true(!neural_network.has_scaling_layer(), LOG);
   assert_true(!neural_network.has_principal_components_layer(), LOG);
   assert_true(!neural_network.has_long_short_term_memory_layer(), LOG);
   assert_true(!neural_network.has_recurrent_layer(), LOG);
   assert_true(!neural_network.has_unscaling_layer(), LOG);
   assert_true(!neural_network.has_bounding_layer(), LOG);
   assert_true(!neural_network.has_probabilistic_layer(), LOG);

   // Test 1

   Tensor<Layer*, 1> layer_tensor_1(7);
   layer_tensor_1.setValues({new ScalingLayer, new PrincipalComponentsLayer, new LongShortTermMemoryLayer,
                             new RecurrentLayer, new UnscalingLayer, new BoundingLayer, new ProbabilisticLayer});

   NeuralNetwork neural_network_1(layer_tensor_1);

   assert_true(neural_network_1.has_scaling_layer(), LOG);
   assert_true(neural_network_1.has_principal_components_layer(), LOG);
   assert_true(neural_network_1.has_long_short_term_memory_layer(), LOG);
   assert_true(neural_network_1.has_recurrent_layer(), LOG);
   assert_true(neural_network_1.has_unscaling_layer(), LOG);
   assert_true(neural_network_1.has_bounding_layer(), LOG);
   assert_true(neural_network_1.has_probabilistic_layer(), LOG);
}

void NeuralNetworkTest::test_get_inputs()
{
   cout << "test_get_inputs\n";

   NeuralNetwork neural_network;

   Tensor<string, 1> inputs_names(2);
   inputs_names.setValues({"in_1","in_2"});

   neural_network.set_inputs_names(inputs_names);
   assert_true(neural_network.get_inputs_names().size() == 2, LOG);
   assert_true(neural_network.get_input_name(0) == "in_1", LOG);
   assert_true(neural_network.get_input_name(1) == "in_2", LOG);
   assert_true(neural_network.get_input_index("in_1") == 0, LOG);
   assert_true(neural_network.get_input_index("in_2") == 1, LOG);
}

void NeuralNetworkTest::test_get_outputs()
{
   cout << "test_get_outputs\n";

   NeuralNetwork neural_network;

   Tensor<string, 1> outputs_names(2);
   outputs_names.setValues({"out_1","out_2"});

   neural_network.set_outputs_names(outputs_names);
   assert_true(neural_network.get_outputs_names().size() == 2, LOG);
   assert_true(neural_network.get_output_name(0) == "out_1", LOG);
   assert_true(neural_network.get_output_name(1) == "out_2", LOG);
   assert_true(neural_network.get_output_index("out_1") == 0, LOG);
   assert_true(neural_network.get_output_index("out_2") == 1, LOG);
}

void NeuralNetworkTest::test_get_trainable_layers()
{
   cout << "test_get_trainable_layers\n";

   Tensor<Layer*, 1> layer_tensor(10);
   layer_tensor.setValues({new ScalingLayer, new ConvolutionalLayer, new PerceptronLayer,
                           new PoolingLayer, new ProbabilisticLayer, new LongShortTermMemoryLayer,
                           new RecurrentLayer, new UnscalingLayer, new BoundingLayer, new PrincipalComponentsLayer});

   NeuralNetwork neural_network(layer_tensor);

   assert_true(neural_network.get_trainable_layers_pointers()(0)->get_type() == Layer::Convolutional, LOG);
   assert_true(neural_network.get_trainable_layers_pointers()(1)->get_type() == Layer::Perceptron, LOG);
   assert_true(neural_network.get_trainable_layers_pointers()(2)->get_type() == Layer::Pooling, LOG);
   assert_true(neural_network.get_trainable_layers_pointers()(3)->get_type() == Layer::Probabilistic, LOG);
   assert_true(neural_network.get_trainable_layers_pointers()(4)->get_type() == Layer::LongShortTermMemory, LOG);
   assert_true(neural_network.get_trainable_layers_pointers()(5)->get_type() == Layer::Recurrent, LOG);
   assert_true(neural_network.get_trainable_layers_pointers()(6)->get_type() == Layer::PrincipalComponents, LOG);

   assert_true(neural_network.get_trainable_layers_indices()(0) == 1, LOG);
   assert_true(neural_network.get_trainable_layers_indices()(1) == 2, LOG);
   assert_true(neural_network.get_trainable_layers_indices()(2) == 3, LOG);
   assert_true(neural_network.get_trainable_layers_indices()(3) == 4, LOG);
   assert_true(neural_network.get_trainable_layers_indices()(4) == 5, LOG);
   assert_true(neural_network.get_trainable_layers_indices()(5) == 6, LOG);
   assert_true(neural_network.get_trainable_layers_indices()(6) == 8, LOG);
}

void NeuralNetworkTest::test_get_layers_type_pointers()
{
   cout << "test_get_layers_type_pointers\n";

   // Test 1

   Tensor<Layer*, 1> layer_tensor(10);
   layer_tensor.setValues({new ScalingLayer, new ConvolutionalLayer, new PerceptronLayer,
                           new PoolingLayer, new ProbabilisticLayer, new LongShortTermMemoryLayer,
                           new RecurrentLayer, new UnscalingLayer, new BoundingLayer, new PrincipalComponentsLayer});

   NeuralNetwork neural_network(layer_tensor);

   assert_true(neural_network.get_scaling_layer_pointer()->get_type() == Layer::Scaling, LOG);
   assert_true(neural_network.get_unscaling_layer_pointer()->get_type() == Layer::Unscaling, LOG);
   assert_true(neural_network.get_bounding_layer_pointer()->get_type() == Layer::Bounding, LOG);
   assert_true(neural_network.get_probabilistic_layer_pointer()->get_type() == Layer::Probabilistic, LOG);/*
   assert_true(neural_network.get_principal_components_layer_pointer()->get_type() == Layer::PrincipalComponents, LOG);*/ //CCH
   assert_true(neural_network.get_long_short_term_memory_layer_pointer()->get_type() == Layer::LongShortTermMemory, LOG);
   assert_true(neural_network.get_recurrent_layer_pointer()->get_type() == Layer::Recurrent, LOG);

   assert_true(neural_network.get_first_perceptron_layer_pointer()->get_type() == Layer::Perceptron, LOG);
}

void NeuralNetworkTest::test_get_layer_pointer()
{
   cout << "test_get_layer_pointer\n";

   Tensor<Layer*, 1> layer_tensor(10);
   layer_tensor.setValues({new ScalingLayer, new ConvolutionalLayer, new PerceptronLayer,
                           new PoolingLayer, new ProbabilisticLayer, new LongShortTermMemoryLayer,
                           new RecurrentLayer, new UnscalingLayer, new PrincipalComponentsLayer, new BoundingLayer});

   NeuralNetwork neural_network(layer_tensor);

   assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);
   assert_true(neural_network.get_layer_pointer(1)->get_type() == Layer::Convolutional, LOG);
   assert_true(neural_network.get_layer_pointer(2)->get_type() == Layer::Perceptron, LOG);
   assert_true(neural_network.get_layer_pointer(3)->get_type() == Layer::Pooling, LOG);
   assert_true(neural_network.get_layer_pointer(4)->get_type() == Layer::Probabilistic, LOG);
   assert_true(neural_network.get_layer_pointer(5)->get_type() == Layer::LongShortTermMemory, LOG);
   assert_true(neural_network.get_layer_pointer(6)->get_type() == Layer::Recurrent, LOG);
   assert_true(neural_network.get_layer_pointer(7)->get_type() == Layer::Unscaling, LOG);
   assert_true(neural_network.get_layer_pointer(8)->get_type() == Layer::PrincipalComponents, LOG);
   assert_true(neural_network.get_layer_pointer(9)->get_type() == Layer::Bounding, LOG);

   assert_true(neural_network.get_output_layer_pointer()->get_type() == Layer::Bounding, LOG);
}

void NeuralNetworkTest::test_set()
{
   cout << "test_set\n";

   NeuralNetwork neural_network;

   // Test 0

   neural_network.set();

   assert_true(neural_network.get_inputs_names().size() == 0, LOG);
   assert_true(neural_network.get_outputs_names().size() == 0, LOG);
   assert_true(neural_network.get_layers_pointers().size() == 0, LOG);

   // Test 1

   Tensor<Index, 1> architecture(3);

   architecture.setValues({1,0,1}); //CC -> architecture = {inp_n, hddn_neurns_n, out_n}

   neural_network.set(NeuralNetwork::Approximation, architecture);
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
   assert_true(neural_network.get_layers_pointers().size() == 5, LOG);

   neural_network.set(NeuralNetwork::Classification, architecture);
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
   assert_true(neural_network.get_layers_pointers().size() == 3, LOG);

   neural_network.set(NeuralNetwork::Forecasting, architecture);
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
   assert_true(neural_network.get_layers_pointers().size() == 4, LOG);

   neural_network.set(NeuralNetwork::ImageApproximation, architecture);
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
   assert_true(neural_network.get_layers_pointers().size() == 1, LOG);

   neural_network.set(NeuralNetwork::ImageClassification, architecture);
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
   assert_true(neural_network.get_layers_pointers().size() == 1, LOG);

   // Test 2 / Convolutional layer set
   /*
   Tensor<Index, 1> new_inputs_dimensions(1);
   new_inputs_dimensions.setConstant(1);

   Index new_blocks_number = 1;

   Tensor<Index, 1> new_filters_dimensions(1);
   new_filters_dimensions.setConstant(1);

   Index new_outputs_number = 1;

   ConvolutionalLayer convolutional_layer(1,1); //CC -> cl(inputs_dim, filters_dim)

   neural_network.set(new_inputs_dimensions, new_blocks_number, new_filters_dimensions, new_outputs_number);

   assert_true(neural_network.is_empty(), LOG);
   assert_true(neural_network.get_layers_number() == 0, LOG);
   */

   // Test 3 / Copy layer set
/*
   NeuralNetwork neural_network_3_0;
   NeuralNetwork neural_network_3_1;

   Tensor<Index, 1> architecture_3_0(3);
   architecture_3_0.setValues({1,0,1});

   Tensor<Index, 1> architecture_3_1(3);
   architecture_3_1.setValues({5,0,5});

   neural_network_3_0.set(NeuralNetwork::Approximation, architecture_3_0);
   neural_network_3_1.set(NeuralNetwork::Approximation, architecture_3_1);

   neural_network_3_1.set(neural_network_3_0);

   assert_true(neural_network_3_1.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
   assert_true(neural_network_3_1.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
   assert_true(neural_network_3_1.get_layers_pointers().size() == 1, LOG);
   */
}

void NeuralNetworkTest::test_set_names()
{
   cout << "test_set_names\n";

   NeuralNetwork neural_network;

   // Test 0

   Tensor<string, 1> inputs_names;
   Tensor<string, 1> outputs_names;

   neural_network.set_inputs_names(inputs_names);
   neural_network.set_outputs_names(outputs_names);
   assert_true(neural_network.get_inputs_names().size() == 0, LOG);
   assert_true(neural_network.get_outputs_names().size() == 0, LOG);

   // Test 1

   Tensor<string, 1> inputs_names_1(2);
   inputs_names_1.setValues({"in_1","in_2"});
   Tensor<string, 1> outputs_names_1(2);
   outputs_names_1.setValues({"out_1","out_2"});

   neural_network.set_inputs_names(inputs_names_1);
   neural_network.set_outputs_names(outputs_names_1);
   assert_true(neural_network.get_inputs_names().size() == 2, LOG);
   assert_true(neural_network.get_inputs_names()(0) == "in_1", LOG);
   assert_true(neural_network.get_inputs_names()(1) == "in_2", LOG);
   assert_true(neural_network.get_outputs_names().size() == 2, LOG);
   assert_true(neural_network.get_outputs_names()(0) == "out_1", LOG);
   assert_true(neural_network.get_outputs_names()(1) == "out_2", LOG);
}

void NeuralNetworkTest::test_set_number()
{
   cout << "test_set_number\n";

   Tensor<Index, 1> architecture(3);
   architecture.setValues({1,0,1});

   NeuralNetwork neural_network(NeuralNetwork::Classification, architecture);

   // Test 0_0

   Index inputs_number;
   neural_network.set_inputs_number(inputs_number);

   assert_true(neural_network.get_inputs_number() == 0, LOG);
   assert_true(neural_network.get_layer_pointer(0)->get_inputs_number() == 0, LOG); //CC -> Scaling layer nmb assert

   // Test 0_1

   inputs_number = 3;

   neural_network.set_inputs_number(inputs_number);
   assert_true(neural_network.get_inputs_number() == 3, LOG);
   assert_true(neural_network.get_layer_pointer(0)->get_inputs_number() == 3, LOG); //CC -> Scaling layer nmb assert

   // Test 1_0
/*
   Tensor<bool, 1> inputs_names_1;
   neural_network.set_inputs_number(inputs_names_1);

   assert_true(neural_network.get_inputs_number() == 0, LOG);
*/

   // Test 1_1

   Tensor<bool, 1> inputs_names_1_1(2);
   inputs_names_1_1.setValues({true,false});

   neural_network.set_inputs_number(inputs_names_1_1);
   assert_true(neural_network.get_inputs_number() == 1, LOG);
}

void NeuralNetworkTest::test_set_default()
{
   cout << "test_set_default\n";

   NeuralNetwork neural_network;

   neural_network.set_default();

   assert_true(display == true, LOG);
}

void NeuralNetworkTest::test_set_pointers()
{
   cout << "test_set_pointers\n";

   const int n = omp_get_max_threads();
   NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
   ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

   Tensor<Index, 1> architecture(3);
   architecture.setValues({1,0,1});

   NeuralNetwork neural_network(NeuralNetwork::Classification, architecture);

   // Test 1 // Device

   neural_network.set_thread_pool_device(thread_pool_device);

   assert_true(neural_network.get_layers_number() == 3, LOG);
//   assert_true(neural_network.get_layer_pointer(0)->device_pointer->get_type() == Device::EigenThreadPool, LOG);
   //CCH -> Need get_device_pointer method?

   // Test 2 // Layers

   NeuralNetwork neural_network_2;

   Tensor<Layer*, 1> layers_tensor(3);
   layers_tensor.setValues({new ScalingLayer(1),new PerceptronLayer(1, 1), new UnscalingLayer(1)});

   neural_network_2.set_layers_pointers(layers_tensor);

   assert_true(!neural_network_2.is_empty(), LOG);
   assert_true(neural_network_2.get_layers_number() == 3, LOG);
   assert_true(neural_network_2.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);
   assert_true(neural_network_2.get_layer_pointer(1)->get_type() == Layer::Perceptron, LOG);
   assert_true(neural_network_2.get_layer_pointer(2)->get_type() == Layer::Unscaling, LOG);
}

void NeuralNetworkTest::test_set_display()
{
   cout << "test_set_display\n";

   NeuralNetwork neural_network;

   neural_network.set_display(true);
   assert_true(neural_network.get_display() == true, LOG);

   neural_network.set_display(false);
   assert_true(neural_network.get_display() == false, LOG);
}

void NeuralNetworkTest::test_get_layers_number()
{
   cout << "test_get_layers_number\n";

   // Test 0

   Tensor<Layer*, 1> layers_tensor;

   NeuralNetwork neural_network(layers_tensor);

   assert_true(neural_network.get_layers_number() == 0, LOG);
   assert_true(neural_network.get_trainable_layers_number() == 0, LOG);
   assert_true(neural_network.get_perceptron_layers_number() == 0, LOG);
   assert_true(neural_network.get_probabilistic_layers_number() == 0, LOG);
//   assert_true(neural_network.get_layers_neurons_numbers()(0) == 0, LOG);

   // Test 1

   Tensor<Layer*, 1> layers_tensor_1(10);

   layers_tensor_1.setValues({new ScalingLayer, new ConvolutionalLayer, new PerceptronLayer,
                           new PoolingLayer, new ProbabilisticLayer, new LongShortTermMemoryLayer,
                           new RecurrentLayer, new UnscalingLayer, new PrincipalComponentsLayer, new BoundingLayer});

   NeuralNetwork neural_network_1(layers_tensor_1);

   assert_true(neural_network_1.get_layers_number() == 10, LOG);
   assert_true(neural_network_1.get_trainable_layers_number() == 7, LOG);
   assert_true(neural_network_1.get_perceptron_layers_number() == 1, LOG); //CCH
   assert_true(neural_network_1.get_probabilistic_layers_number() == 1, LOG);
//   assert_true(neural_network.get_layers_neurons_numbers()(0) == 0, LOG);
}

void NeuralNetworkTest::test_inputs_outputs_number()
{
   cout << "test_inputs_outputs_number\n";

   // Test 0

   Tensor<Layer*, 1> layers_tensor;

   NeuralNetwork neural_network(layers_tensor);

   assert_true(neural_network.get_inputs_number() == 0, LOG);
   assert_true(neural_network.get_outputs_number() == 0, LOG);

   // Test 1

   Tensor<Layer*, 1> layers_tensor_1(2);
   layers_tensor_1.setValues({new ScalingLayer(5),new PerceptronLayer(5,2), new UnscalingLayer(2)});

   NeuralNetwork neural_network_1(layers_tensor_1);

   assert_true(neural_network_1.get_inputs_number() == 5, LOG);
   assert_true(neural_network_1.get_outputs_number() == 2, LOG);
}

void NeuralNetworkTest::test_get_architecture()
{
   cout << "test_get_architecture\n";

   // Test 0

   Tensor<Index, 1> architecture_0;

   NeuralNetwork neural_network_0(NeuralNetwork::Approximation, architecture_0);

   assert_true(neural_network_0.get_architecture().rank() == 1, LOG);
   assert_true(neural_network_0.get_architecture().dimension(0) == 0, LOG);

   // Test 1

   Tensor<Index, 1> architecture(3);
   architecture.setValues({1,4,2});

   NeuralNetwork neural_network_1_1(NeuralNetwork::Approximation, architecture);

   assert_true(neural_network_1_1.get_architecture().rank() == 1, LOG);
   assert_true(neural_network_1_1.get_architecture().dimension(0) == 5, LOG);
   assert_true(neural_network_1_1.get_architecture()(0) == 1, LOG);
   assert_true(neural_network_1_1.get_architecture()(1) == 4, LOG);
   assert_true(neural_network_1_1.get_architecture()(2) == 2, LOG);
   assert_true(neural_network_1_1.get_architecture()(3) == 2, LOG);
   assert_true(neural_network_1_1.get_architecture()(4) == 2, LOG);
}

void NeuralNetworkTest::test_get_parameters()
{
   cout << "test_get_parameters\n";

   NeuralNetwork neural_network;
   Tensor<type, 1> parameters;

   // Test 0

   neural_network.set();
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 0, LOG);
   assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);

   // Test 1

   Tensor<Index, 1> architecture(3);

   architecture.setValues({1,2,1});

   neural_network.set(NeuralNetwork::Approximation, architecture);

   PerceptronLayer* pl00 = dynamic_cast<PerceptronLayer*>(neural_network.get_layer_pointer(1));
   pl00->set_biases_constant(0);
   pl00->set_synaptic_weights_constant(1);

   PerceptronLayer* pl01 = dynamic_cast<PerceptronLayer*>(neural_network.get_layer_pointer(2));
   pl01->set_biases_constant(2);
   pl01->set_synaptic_weights_constant(3);

   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 7, LOG);
   assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
   assert_true(abs(parameters(1) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(5) - 3) < static_cast<type>(1e-5), LOG);

   // Test 2

   architecture.resize(4);

   architecture.setConstant(1);

   neural_network.set(NeuralNetwork::Approximation, architecture);

   PerceptronLayer* pl10 = dynamic_cast<PerceptronLayer*>(neural_network.get_layer_pointer(1));
   pl10->set_biases_constant(0);
   pl10->set_synaptic_weights_constant(1);

   PerceptronLayer* pl11 = dynamic_cast<PerceptronLayer*>(neural_network.get_layer_pointer(2));
   pl11->set_biases_constant(2);
   pl11->set_synaptic_weights_constant(3);

   PerceptronLayer* pl12 = dynamic_cast<PerceptronLayer*>(neural_network.get_layer_pointer(3));
   pl12->set_biases_constant(4);
   pl12->set_synaptic_weights_constant(5);

   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 6, LOG);
   assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
   assert_true(abs(parameters(0) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(1) - 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(2) - 2) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(3) - 3) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(4) - 4) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(5) - 5) < static_cast<type>(1e-5), LOG);
}

void NeuralNetworkTest::test_get_trainable_layers_parameters()
{
    cout << "test_get_trainable_layers_parameters\n";

    NeuralNetwork neural_network;
    Tensor<type, 1> parameters;
    Tensor<Tensor<type, 1>, 1> trainable_layers_parameters;

    // Test 0

    neural_network.set();
    parameters = neural_network.get_parameters();
    trainable_layers_parameters = neural_network.get_trainable_layers_parameters(parameters);

    assert_true(trainable_layers_parameters.rank() == 1, LOG);
    assert_true(trainable_layers_parameters.size() == 0, LOG);

    // Test 1

    Tensor<Index, 1> architecture(3);

    architecture.setValues({1,2,1});

    neural_network.set(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);
    parameters = neural_network.get_parameters();
    trainable_layers_parameters = neural_network.get_trainable_layers_parameters(parameters);

    assert_true(trainable_layers_parameters.rank() == 1, LOG);
    assert_true(trainable_layers_parameters.size() == 2, LOG);
    assert_true(neural_network.get_trainable_layers_parameters_numbers().size() == trainable_layers_parameters.size(), LOG);

    assert_true(neural_network.get_trainable_layers_parameters_numbers()(0) == 4, LOG);
    assert_true(neural_network.get_trainable_layers_parameters_numbers()(1) == 3, LOG);
    assert_true(trainable_layers_parameters(0).size() == neural_network.get_trainable_layers_parameters_numbers()(0) , LOG);
    assert_true(trainable_layers_parameters(1).size() == neural_network.get_trainable_layers_parameters_numbers()(1), LOG);

    assert_true(abs(trainable_layers_parameters(0)(0) - 0) < static_cast<type>(1e-5), LOG);
    assert_true(abs(trainable_layers_parameters(0)(3) - 0) < static_cast<type>(1e-5), LOG);
    assert_true(abs(trainable_layers_parameters(1)(0) - 0) < static_cast<type>(1e-5), LOG);
    assert_true(abs(trainable_layers_parameters(1)(2) - 0) < static_cast<type>(1e-5), LOG);

    // Test 2

    architecture.resize(4);
    architecture.setConstant(1);

    neural_network.set(NeuralNetwork::Approximation, architecture);

    PerceptronLayer* pl0 = dynamic_cast<PerceptronLayer*>(neural_network.get_layer_pointer(1));
    pl0->set_biases_constant(0);
    pl0->set_synaptic_weights_constant(1);

    PerceptronLayer* pl1 = dynamic_cast<PerceptronLayer*>(neural_network.get_layer_pointer(2));
    pl1->set_biases_constant(2);
    pl1->set_synaptic_weights_constant(3);

    PerceptronLayer* pl2 = dynamic_cast<PerceptronLayer*>(neural_network.get_layer_pointer(3));
    pl2->set_biases_constant(4);
    pl2->set_synaptic_weights_constant(5);

    parameters = neural_network.get_parameters();
    trainable_layers_parameters = neural_network.get_trainable_layers_parameters(parameters);

    assert_true(trainable_layers_parameters.rank() == 1, LOG);
    assert_true(trainable_layers_parameters.size() == 3, LOG);
    assert_true(neural_network.get_trainable_layers_parameters_numbers().size() == trainable_layers_parameters.size(), LOG);

    assert_true(trainable_layers_parameters(0).size() == neural_network.get_trainable_layers_parameters_numbers()(0), LOG);
    assert_true(trainable_layers_parameters(1).size() == neural_network.get_trainable_layers_parameters_numbers()(1), LOG);
    assert_true(trainable_layers_parameters(2).size() == neural_network.get_trainable_layers_parameters_numbers()(2), LOG);
    assert_true(neural_network.get_trainable_layers_parameters_numbers()(0) == 2, LOG);
    assert_true(neural_network.get_trainable_layers_parameters_numbers()(1) == 2, LOG);
    assert_true(neural_network.get_trainable_layers_parameters_numbers()(2) == 2, LOG);

    assert_true(abs(trainable_layers_parameters(0)(0) - 0) < static_cast<type>(1e-5), LOG);
    assert_true(abs(trainable_layers_parameters(0)(1) - 1) < static_cast<type>(1e-5), LOG);
    assert_true(abs(trainable_layers_parameters(1)(0) - 2) < static_cast<type>(1e-5), LOG);
    assert_true(abs(trainable_layers_parameters(1)(1) - 3) < static_cast<type>(1e-5), LOG);
    assert_true(abs(trainable_layers_parameters(2)(0) - 4) < static_cast<type>(1e-5), LOG);
    assert_true(abs(trainable_layers_parameters(2)(1) - 5) < static_cast<type>(1e-5), LOG);
}

void NeuralNetworkTest::test_set_parameters()
{
   cout << "test_set_parameters\n";

   Tensor<Index, 1> architecture;
   NeuralNetwork neural_network;

   Index parameters_number;
   Tensor<type, 1> parameters;

   // Test 0

   neural_network.set_parameters(parameters);
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 0, LOG);

   // Test 1

   architecture.resize(2);

   architecture.setConstant(2);

   neural_network.set(NeuralNetwork::Approximation, architecture);

   parameters_number = neural_network.get_parameters_number();
   parameters.resize(parameters_number);

   neural_network.set_parameters(parameters);
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 6, LOG);
   assert_true(parameters.size() == parameters_number, LOG);

   // Test 1

   parameters.setValues({1,2,3,4, 5,6});

   neural_network.set_parameters(parameters);
   parameters = neural_network.get_parameters();

   assert_true(abs(parameters(0) - 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(1) - 2) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(2) - 3) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(3) - 4) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(4) - 5) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(5) - 6) < static_cast<type>(1e-5), LOG);
}

void NeuralNetworkTest::test_set_parameters_constant()
{
   cout << "test_set_parameters_constant\n";

   NeuralNetwork neural_network;
   Tensor<type, 1> parameters;

   Tensor<Index, 1> architecture(3);

   architecture.setValues({1,2,1});

   neural_network.set(NeuralNetwork::Approximation, architecture);

   neural_network.set_parameters_constant(1);
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 7, LOG);
   assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
   assert_true(abs(parameters(1) - 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(5) - 1) < static_cast<type>(1e-5), LOG);
}

void NeuralNetworkTest::test_set_parameters_random()
{
   cout << "test_set_parameters_random\n";

   NeuralNetwork neural_network;
   Tensor<type, 1> parameters;

   Tensor<Index, 1> architecture(3);

   architecture.setValues({1,2,1});

   neural_network.set(NeuralNetwork::Approximation, architecture);

   neural_network.set_parameters_random();
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 7, LOG);
   assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
}

void NeuralNetworkTest::test_calculate_parameters_norm()
{
   cout << "test_calculate_parameters_norm\n";

   NeuralNetwork neural_network;
   type parameters_norm = 0;
   Tensor<Index, 1> architecture;

   // Test  0

   neural_network.set(NeuralNetwork::Approximation, architecture);

   parameters_norm = neural_network.calculate_parameters_norm();

   assert_true(abs(parameters_norm - 0) < static_cast<type>(1e-5), LOG);

   // Test  1

   architecture.resize(4);
   architecture.setConstant(1);

   neural_network.set(NeuralNetwork::Approximation, architecture);

   neural_network.set_parameters_constant(1.0);

   parameters_norm = neural_network.calculate_parameters_norm();

   assert_true(abs(parameters_norm - static_cast<type>(sqrt(6))) < static_cast<type>(1e-5), LOG);
}

void NeuralNetworkTest::test_calculate_parameters_descriptives()
{
   cout << "test_calculate_parameters_descriptives\n";

   NeuralNetwork neural_network;
   Descriptives parameters_descriptives;
   Tensor<Index, 1> architecture;

   // Test  0

   neural_network.set(NeuralNetwork::Approximation, architecture);

   parameters_descriptives = neural_network.calculate_parameters_descriptives();

   assert_true(parameters_descriptives.minimum > numeric_limits<type>::min(), LOG);
   assert_true(parameters_descriptives.maximum < numeric_limits<type>::max(), LOG);
//   assert_true(abs(parameters_descriptives.mean - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_descriptives.standard_deviation - 0) < static_cast<type>(1e-5), LOG);

   // Test 1

   architecture.resize(4);
   architecture.setConstant(1);

   neural_network.set(NeuralNetwork::Approximation, architecture);

   neural_network.set_parameters_constant(1.0);

   parameters_descriptives = neural_network.calculate_parameters_descriptives();

   assert_true(abs(parameters_descriptives.minimum - 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_descriptives.maximum - 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_descriptives.mean - 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_descriptives.standard_deviation - 0) < static_cast<type>(1e-5), LOG);
}

void NeuralNetworkTest::test_calculate_parameters_histogram()
{
   cout << "test_calculate_parameters_histogram\n";

   NeuralNetwork neural_network;
   Histogram parameters_histogram;
   Tensor<Index, 1> architecture;

   // Test  0
/*
   neural_network.set(NeuralNetwork::Approximation, architecture);

   parameters_histogram = neural_network.calculate_parameters_histogram(0);

   assert_true(parameters_descriptives.minimum > numeric_limits<type>::min(), LOG);
   assert_true(parameters_descriptives.maximum < numeric_limits<type>::max(), LOG);
//   assert_true(abs(parameters_descriptives.mean - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_descriptives.standard_deviation - 0) < static_cast<type>(1e-5), LOG);
*/
   // Test 1

   architecture.resize(4);
   architecture.setConstant(1);

   neural_network.set(NeuralNetwork::Approximation, architecture);

   Index parameters_number = neural_network.get_parameters_number();
/*
   cout << "parameters_number" << endl;
   cout << parameters_number << endl;
*/
   Tensor<type, 1> parameters(parameters_number);
   parameters.setValues({0,2, 0,4 ,0,6});

   neural_network.set_parameters(parameters);

   parameters_histogram = neural_network.calculate_parameters_histogram(3);
/*
   cout << "centers" << endl;
   cout << parameters_histogram.centers << endl;
   cout << "min" << endl;
   cout << parameters_histogram.minimums << endl;
   cout << "max" << endl;
   cout << parameters_histogram.maximums << endl;
   cout << "frec" << endl;
   cout << parameters_histogram.frequencies << endl;
*/
   assert_true(abs(parameters_histogram.centers(0) - 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_histogram.minimums(0) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_histogram.maximums(0) - 2) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_histogram.frequencies(0) - 3) < static_cast<type>(1e-5), LOG);

   assert_true(abs(parameters_histogram.centers(1) - 3) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_histogram.minimums(1) - 2) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_histogram.maximums(1) - 4) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_histogram.frequencies(1) - 1) < static_cast<type>(1e-5), LOG);

   assert_true(abs(parameters_histogram.centers(2) - 5) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_histogram.minimums(2) - 4) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_histogram.maximums(2) - 6) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters_histogram.frequencies(2) - 2) < static_cast<type>(1e-5), LOG);
}

void NeuralNetworkTest::test_perturbate_parameters()
{
   cout << "test_perturbate_parameters\n";

   Tensor<Index, 1> architecture(3);
   NeuralNetwork neural_network;

   Index parameters_number;
   Tensor<type, 1> parameters;

   architecture.setValues({1,1,1});

   neural_network.set(NeuralNetwork::Approximation, architecture);

   parameters_number = neural_network.get_parameters_number();
   parameters.resize(parameters_number);
   parameters.setConstant(1);

   neural_network.set_parameters(parameters);
   parameters = neural_network.get_parameters();

   neural_network.perturbate_parameters(0.5);
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 4, LOG);
   assert_true(parameters.size() == parameters_number, LOG);
   assert_true(abs(parameters(0) - static_cast<type>(1.5)) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(3) - static_cast<type>(1.5)) < static_cast<type>(1e-5), LOG);
}

void NeuralNetworkTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   const int n = omp_get_max_threads();
   NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
   ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

   NeuralNetwork neural_network;



   Index inputs_number;
   Index outputs_number;

   Tensor<Index, 1> architecture;

   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;

   Index parameters_number;

   Tensor<type, 1> parameters;

   // Test 1

   architecture.resize(2);

   architecture.setConstant(3);

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);
   neural_network.set_parameters_constant(0);

   inputs.resize(1,3);

   inputs.setConstant(1);

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(outputs.rank() == 2, LOG);
   assert_true(outputs.size() == 3, LOG);
   assert_true(abs(outputs(0,0) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(outputs(0,1) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(outputs(0,2) - 0) < static_cast<type>(1e-5), LOG);

   // Test 2

   architecture.resize(3);

   architecture.setValues({2, 1, 5});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);

   neural_network.set_parameters_constant(0);

   inputs.resize(1, 2);

   inputs.setConstant(0);

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(outputs.size() == 5, LOG);
   assert_true(abs(outputs(0,0) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(outputs(0,1) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(outputs(0,2) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(outputs(0,3) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(outputs(0,4) - 0) < static_cast<type>(1e-5), LOG);

   // Test 3

   architecture.resize(2);

   architecture.setValues({1, 2});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);

   inputs.resize(1, 1);

   inputs.setConstant(2);

   neural_network.set_parameters_constant(1);

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(outputs.size() == 2, LOG);
   assert_true(abs(outputs(0,0) - 3) < static_cast<type>(1e-5), LOG);
   assert_true(abs(outputs(0,1) - 3) < static_cast<type>(1e-5), LOG);

   // Test 4

   architecture.resize(3);
   architecture.setValues({4, 3, 3});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);


   inputs.resize(1, 4);

   inputs.setConstant(0);

   neural_network.set_parameters_constant(1);

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(neural_network.calculate_outputs(inputs).size() == 3, LOG);
   assert_true(abs(outputs(0,0) - static_cast<type>(3.2847)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0,1) - static_cast<type>(3.2847)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0,2) - static_cast<type>(3.2847)) < static_cast<type>(1e-3), LOG);

   // Test 5

   architecture.resize(2);

   architecture.setValues({1, 2});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);

   inputs_number = neural_network.get_inputs_number();
   parameters_number = neural_network.get_parameters_number();
   outputs_number = neural_network.get_outputs_number();

   inputs.resize(1,inputs_number);
   inputs.setConstant(0.0);

   parameters.resize(parameters_number);
   parameters.setConstant(0.0);

   neural_network.set_parameters(parameters);

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(outputs.size() == outputs_number, LOG);
   assert_true(abs(outputs(0,0) - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0,1) - 0) < static_cast<type>(1e-3), LOG);

   // Test 6

   architecture.resize(3);

   architecture.setConstant(1);

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);

   neural_network.set_parameters_constant(0);

   inputs.resize(1, 1);
   inputs.setConstant(0);

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(outputs.size() == 1, LOG);
   assert_true(abs(outputs(0,0) - 0) < static_cast<type>(1e-3), LOG);

   // Test 6_1

   architecture.resize(2);
   architecture.setConstant(1);

   neural_network.set(NeuralNetwork::Classification, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);

   neural_network.set_parameters_constant(0);

   inputs.resize(1, 1);
   inputs.setConstant(0);

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(outputs.size() == 1, LOG);
   assert_true(abs(outputs(0,0) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);

   inputs.setRandom();

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(outputs.size() == 1, LOG);

   // Test 7

   NeuralNetwork neural_network_7;

   const Index categories = 3;
   parameters_number = 5;

   inputs_number = 10;

   ScalingLayer* scaling_layer_3 = new ScalingLayer(inputs_number);
   PerceptronLayer* perceptron_layer_4 = new PerceptronLayer(inputs_number, categories);
   ProbabilisticLayer* probabilistic_layer_5 = new ProbabilisticLayer(categories,categories);

   neural_network_7.add_layer(scaling_layer_3);
   neural_network_7.add_layer(perceptron_layer_4);
   neural_network_7.add_layer(probabilistic_layer_5);
   neural_network_7.set_thread_pool_device(thread_pool_device);

   neural_network_7.set_parameters_constant(-5);

   inputs.resize(parameters_number, inputs_number);
   inputs.setConstant(-1);

   outputs = neural_network_7.calculate_outputs(inputs);

   assert_true(outputs.size() == parameters_number*categories, LOG);
   assert_true(abs(outputs(0,0) - static_cast<type>(0.2/3)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0,categories-1) - static_cast<type>(0.2/3)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(parameters_number-1,0) - static_cast<type>(0.2/3)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(parameters_number-1,categories-1) - static_cast<type>(0.2/3)) < static_cast<type>(1e-3), LOG);

   // Test 8

   architecture.resize(5);
   architecture.setValues({1,3,3,3,1});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);

   inputs_number = neural_network.get_inputs_number();
   outputs_number = neural_network.get_outputs_number();

   inputs.resize(2,inputs_number);
   inputs.setConstant(0);

   parameters_number = neural_network.get_parameters_number();
   parameters.resize(parameters_number);
   parameters.setConstant(0);

   neural_network.set_parameters(parameters);

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(outputs.dimension(1) == outputs_number, LOG);
   assert_true(abs(outputs(0,0) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(outputs(1,0) - 0) < static_cast<type>(1e-5), LOG);
}

void NeuralNetworkTest::test_calculate_trainable_outputs()
{
   cout << "test_calculate_trainable_outputs\n";

   const int n = omp_get_max_threads();
   NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
   ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

   NeuralNetwork neural_network;


   Tensor<Index, 1> architecture;

   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;
   Tensor<type, 2> trainable_outputs;

   Tensor<type, 1> parameters;

   // Test 1 //Unscaling

   architecture.resize(2);
   architecture.setConstant(3);

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);
   neural_network.set_parameters_constant(1);

   inputs.resize(1,3);
   inputs.setConstant(1);

   UnscalingLayer* ul = neural_network.get_unscaling_layer_pointer();
   ul->set_unscaling_method(UnscalingLayer::Logarithmic);

   outputs = neural_network.calculate_outputs(inputs);

   trainable_outputs = neural_network.calculate_trainable_outputs(inputs);

   assert_true(outputs.size() == 3, LOG);
   assert_true(abs(trainable_outputs(0,0) - 4) < static_cast<type>(1e-3)
               && abs(trainable_outputs(0,0) - outputs(0,0)) > static_cast<type>(1e-3), LOG);
   assert_true(abs(trainable_outputs(0,1) - 4) < static_cast<type>(1e-3)
               && abs(trainable_outputs(0,1) - outputs(0,1)) > static_cast<type>(1e-3), LOG);
   assert_true(abs(trainable_outputs(0,2) - 4) < static_cast<type>(1e-3)
               && abs(trainable_outputs(0,2) - outputs(0,2)) > static_cast<type>(1e-3), LOG);

   // Test 2 //Scaling

   architecture.resize(3);
   architecture.setValues({2, 1, 5});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);

   parameters.resize(neural_network.get_parameters_number());
   parameters.setValues({-1,-1,-3, -1,0,1, -1,0,1, 1,1,1, 1});

   neural_network.set_parameters(parameters);

   inputs.resize(1, 2);
   inputs.setValues({{1,-5}});

   UnscalingLayer* ul2 = neural_network.get_unscaling_layer_pointer();
   ul2->set_unscaling_method(UnscalingLayer::NoUnscaling);
   ScalingLayer* sl2 = neural_network.get_scaling_layer_pointer();
   sl2->set_scaling_methods(ScalingLayer::MinimumMaximum);

   Descriptives des_0(-10,10,1,2);
   Descriptives des_1(-20,20,2,3);
   Tensor<Descriptives, 1> descriptives(2);
   descriptives.setValues({des_0,des_1});
   sl2->set_descriptives(descriptives);

   outputs = neural_network.calculate_outputs(inputs);

   trainable_outputs = neural_network.calculate_trainable_outputs(inputs);

   assert_true(outputs.size() == 5, LOG);
   assert_true(abs(trainable_outputs(0,0) - 0) < static_cast<type>(1e-2)
               && abs(trainable_outputs(0,0) - outputs(0,0)) > static_cast<type>(1e-3), LOG);
   assert_true(abs(trainable_outputs(0,1) - 1) < static_cast<type>(1e-2)
               && abs(trainable_outputs(0,1) - outputs(0,1)) > static_cast<type>(1e-3), LOG);
   assert_true(abs(trainable_outputs(0,2) - 2) < static_cast<type>(1e-2)
               && abs(trainable_outputs(0,2) - outputs(0,2)) > static_cast<type>(1e-3), LOG);
   assert_true(abs(trainable_outputs(0,3) - 0) < static_cast<type>(1e-2)
               && abs(trainable_outputs(0,3) - outputs(0,3)) > static_cast<type>(1e-3), LOG);
   assert_true(abs(trainable_outputs(0,4) - 1) < static_cast<type>(1e-2)
               && abs(trainable_outputs(0,4) - outputs(0,4)) > static_cast<type>(1e-3), LOG);

   // Test 3 //Bounding

   architecture.resize(3);
   architecture.setValues({4, 3, 3});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);

   inputs.resize(1, 4);
   inputs.setValues({{1,3,-1,-5}});

   neural_network.set_parameters_constant(1);

   ScalingLayer* sl3 = neural_network.get_scaling_layer_pointer();
   sl3->set_scaling_methods(ScalingLayer::NoScaling);
   UnscalingLayer* ul3 = neural_network.get_unscaling_layer_pointer();
   ul3->set_unscaling_method(UnscalingLayer::NoUnscaling);
   BoundingLayer* bl3 = neural_network.get_bounding_layer_pointer();
   bl3->set_lower_bound(1,1);

   outputs = neural_network.calculate_outputs(inputs);

   trainable_outputs = neural_network.calculate_trainable_outputs(inputs);

   assert_true(outputs.size() == 3, LOG);
   assert_true(abs(trainable_outputs(0,0) + static_cast<type>(1.2847)) < static_cast<type>(1e-3)
               && abs(trainable_outputs(0,0) - outputs(0,0)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(trainable_outputs(0,1) + static_cast<type>(1.2847)) < static_cast<type>(1e-3)
               && abs(trainable_outputs(0,1) - outputs(0,1)) > static_cast<type>(1e-3), LOG);
   assert_true(abs(trainable_outputs(0,2) + static_cast<type>(1.2847)) < static_cast<type>(1e-3)
               && abs(trainable_outputs(0,2) - outputs(0,2)) < static_cast<type>(1e-3), LOG);

   // Test 4 //Parameters

   architecture.resize(3);
   architecture.setValues({4, 3, 3});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);

   inputs.resize(1, 4);
   inputs.setValues({{1,3,-1,-5}});

   neural_network.set_parameters_constant(1);

   ScalingLayer* sl4 = neural_network.get_scaling_layer_pointer();
   sl4->set_scaling_methods(ScalingLayer::NoScaling);
   UnscalingLayer* ul4 = neural_network.get_unscaling_layer_pointer();
   ul4->set_unscaling_method(UnscalingLayer::NoUnscaling);
   BoundingLayer* bl4 = neural_network.get_bounding_layer_pointer();
   bl4->set_lower_bound(1,1);

   outputs = neural_network.calculate_outputs(inputs);

   parameters.resize(neural_network.get_parameters_number());
   parameters = neural_network.get_parameters();

   trainable_outputs = neural_network.calculate_trainable_outputs(inputs, parameters);

   assert_true(outputs.size() == 3, LOG);
   assert_true(abs(trainable_outputs(0,0) + static_cast<type>(1.2847)) < static_cast<type>(1e-3)
               && abs(trainable_outputs(0,0) - outputs(0,0)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(trainable_outputs(0,1) + static_cast<type>(1.2847)) < static_cast<type>(1e-3)
               && abs(trainable_outputs(0,1) - outputs(0,1)) > static_cast<type>(1e-3), LOG);
   assert_true(abs(trainable_outputs(0,2) + static_cast<type>(1.2847)) < static_cast<type>(1e-3)
               && abs(trainable_outputs(0,2) - outputs(0,2)) < static_cast<type>(1e-3), LOG);
}

void NeuralNetworkTest::test_calculate_directional_inputs()
{
   cout << "test_calculate_directional_inputs\n";

   NeuralNetwork neural_network;

   Tensor<Index, 1> architecture;

   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;
   Tensor<type, 2> trainable_outputs;

   Tensor<type, 1> parameters;

   // Test 0

   architecture.resize(3);
   architecture.setValues({3, 4, 2});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   inputs.resize(2,3);
   inputs.setValues({{-5,-1,-3},{7,3,1}});

   Tensor<type, 1> point(3);
   point.setValues({0,0,0});

   Tensor<type, 2> directional_imputs = neural_network.calculate_directional_inputs(0,point,0,0,0);

   assert_true(directional_imputs.rank() == 2, LOG);
   assert_true(directional_imputs.dimension(0) == 0, LOG);

   // Test 1

   point.setValues({1, 2, 3});

   directional_imputs = neural_network.calculate_directional_inputs(2,point,-1,1,3);

   assert_true(directional_imputs.rank() == 2, LOG);
   assert_true(directional_imputs.dimension(0) == 3, LOG);
   assert_true(directional_imputs.dimension(1) == 3, LOG);
   assert_true(abs(directional_imputs(0,2) + 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_imputs(1,2) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_imputs(2,2) - 1) < static_cast<type>(1e-5), LOG);

   // Test 2

   point.setValues({1, 2, 3});

   directional_imputs = neural_network.calculate_directional_inputs(0, point, -4, 0, 5);

   assert_true(directional_imputs.rank() == 2, LOG);
   assert_true(directional_imputs.dimension(0) == 5, LOG);
   assert_true(directional_imputs.dimension(1) == 3, LOG);
   assert_true(abs(directional_imputs(0,0) + 4) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_imputs(1,0) + 3) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_imputs(2,0) + 2) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_imputs(3,0) + 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_imputs(4,0) + 0) < static_cast<type>(1e-5), LOG);
}

void NeuralNetworkTest::test_calculate_outputs_histograms()
{
   cout << "test_calculate_outputs_histograms\n";

   const int n = omp_get_max_threads();
   NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
   ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

   NeuralNetwork neural_network;


   Tensor<Index, 1> architecture;

   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;
   Tensor<Histogram, 1> outputs_histograms;

   Tensor<type, 1> parameters;

   // Test 1

   architecture.resize(2);
   architecture.setValues({1, 1});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);

   parameters.resize(neural_network.get_parameters_number());
   parameters.setConstant(1);
   neural_network.set_parameters(parameters);

   inputs.resize(1,1);
   inputs.setConstant(1);

   outputs_histograms = neural_network.calculate_outputs_histograms(inputs, 2);

   assert_true(outputs_histograms.rank() == 1, LOG);
   assert_true(outputs_histograms(0).minimums(0) - 2 < static_cast<type>(1e-5) &&
               abs(outputs_histograms(0).minimums(0) - outputs_histograms(0).maximums(0)) < static_cast<type>(1e-5) &&
               abs(outputs_histograms(0).minimums(0) - outputs_histograms(0).centers(0)) < static_cast<type>(1e-5), LOG);
   assert_true(outputs_histograms(0).frequencies(0) - 1 < static_cast<type>(1e-5), LOG);

   // Test 2

   architecture.resize(3);
   architecture.setValues({3, 4, 4});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_thread_pool_device(thread_pool_device);

   parameters.resize(neural_network.get_parameters_number());
   parameters.setValues({2,2,2,7, 3,3,3,3, 4,4,4,4, 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 0,0});
   neural_network.set_parameters(parameters);

   inputs.resize(2,3);
   inputs.setValues({{-5,-1,-3},{7,3,1}});

   outputs_histograms = neural_network.calculate_outputs_histograms(inputs, 2);

   assert_true(outputs_histograms.rank() == 1, LOG);
   assert_true(outputs_histograms(0).minimums(0) + 3 < static_cast<type>(1e-5) &&
               abs(outputs_histograms(0).minimums(0) - outputs_histograms(0).maximums(0)) < static_cast<type>(1e-5)  &&
               abs(outputs_histograms(0).minimums(0) - outputs_histograms(0).centers(0)) < static_cast<type>(1e-5)  , LOG);
   assert_true(outputs_histograms(0).frequencies(0) - 1 < static_cast<type>(1e-5), LOG);

   assert_true(outputs_histograms(0).minimums(1) - 9 < static_cast<type>(1e-5) &&
               abs(outputs_histograms(0).minimums(1) - outputs_histograms(0).maximums(1)) < static_cast<type>(1e-5)  &&
               abs(outputs_histograms(0).minimums(1) - outputs_histograms(0).centers(1)) < static_cast<type>(1e-5)  , LOG);
   assert_true(outputs_histograms(0).frequencies(1) - 1 < static_cast<type>(1e-5), LOG);

   assert_true(outputs_histograms(1).minimums(0) + 5 < static_cast<type>(1e-5) &&
               abs(outputs_histograms(1).minimums(0) - outputs_histograms(1).maximums(0)) < static_cast<type>(1e-5)  &&
               abs(outputs_histograms(1).minimums(0) - outputs_histograms(1).centers(0)) < static_cast<type>(1e-5)  , LOG);
   assert_true(outputs_histograms(1).frequencies(0) - 1 < static_cast<type>(1e-5), LOG);

   assert_true(outputs_histograms(1).minimums(1) - 13 < static_cast<type>(1e-5) &&
               abs(outputs_histograms(1).minimums(1) - outputs_histograms(1).maximums(1)) < static_cast<type>(1e-5)  &&
               abs(outputs_histograms(1).minimums(1) - outputs_histograms(1).centers(1)) < static_cast<type>(1e-5)  , LOG);
   assert_true(outputs_histograms(1).frequencies(1) - 1 < static_cast<type>(1e-5), LOG);
}

void NeuralNetworkTest::test_to_XML()
{
   cout << "test_to_XML\n";
/*
   NeuralNetwork neural_network;

   tinyxml2::XMLDocument* document;

   // Test

   document = neural_network.to_XML();

   assert_true(document != nullptr, LOG);

   delete document;
   */
}

void NeuralNetworkTest::test_from_XML()
{
   cout << "test_from_XML\n";
}

void NeuralNetworkTest::test_save()
{
   cout << "test_save\n";

   string file_name = "../data/neural_network.xml";

   NeuralNetwork neural_network;

   Tensor<Index, 1> architecture;

   // Empty multilayer perceptron

   neural_network.set();
   neural_network.save(file_name);

   // Only network architecture

   architecture.resize(3);

   architecture.setValues({2, 4, 3});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.save(file_name);

   architecture.resize(3);

   architecture.setValues({1, 1, 1});

   neural_network.set(NeuralNetwork::Approximation, architecture);

   neural_network.save(file_name);
}

void NeuralNetworkTest::test_load()
{
   cout << "test_load\n";

   string file_name = "../data/neural_network.xml";

   // Empty neural network

   NeuralNetwork neural_network;
   neural_network.save(file_name);
   neural_network.load(file_name);
}

void NeuralNetworkTest::test_print()
{
   cout << "test_print\n";

   // Empty neural network

   NeuralNetwork neural_network;

   Tensor<Index, 1> architecture;

   //neural_network.print();

   // Only network architecture

   architecture.resize(3);
   architecture.setValues({2, 4, 3});

   neural_network.set(NeuralNetwork::Approximation, architecture);

   //neural_network.print();
}

void NeuralNetworkTest::test_write_expression()
{
   cout << "test_write_expression\n";

   NeuralNetwork neural_network;
   string expression;

   Tensor<Index, 1> architecture;

   // Test

//   expression = neural_network.write_expression();

   // Test

   architecture.resize(3);

   architecture.setValues({1, 1, 1});


   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(-1.0);
//   expression = neural_network.write_expression();

   // Test

   architecture.resize(3);

   architecture.setValues({2, 1, 1});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(-1.0);
//   expression = neural_network.write_expression();

   // Test

   architecture.setValues({1, 2, 1});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(-1.0);
//   expression = neural_network.write_expression();

   // Test


   architecture.setValues({1, 1, 2});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(-1.0);
//   expression = neural_network.write_expression();

   // Test

   architecture.setValues({2, 2, 2});

   neural_network.set(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(-1.0);
//   expression = neural_network.write_expression();
}

void NeuralNetworkTest::test_forward_propagate()
{
    const int n = omp_get_max_threads();
    NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
    ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    // Test 1

    Index inputs_number = 2;
    Index target_number = 1;

    Tensor<Index, 1>architecture(2);

    architecture.setValues({inputs_number,target_number});

    Tensor<type,2> data(5, 3);

    data.setValues({{1,1,1},{2,2,2},{3,3,3},{0,0,0},{0,0,0}});

        //DataSet

    DataSet dataset(data);

    dataset.set_training();

    DataSet::Batch batch(5, &dataset);

    Tensor<Index,1> training_instances_indices = dataset.get_training_instances_indices();
    Tensor<Index,1> inputs_indices = dataset.get_input_variables_indices();
    Tensor<Index,1> targets_indices = dataset.get_target_variables_indices();

    batch.fill(training_instances_indices, inputs_indices, targets_indices);

        //NeuralNetwork

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
    neural_network.set_thread_pool_device(thread_pool_device);

    PerceptronLayer* perceptron_layer = dynamic_cast<PerceptronLayer*>(neural_network.get_layer_pointer(1));
    const Index neurons_number = perceptron_layer->get_neurons_number();
    perceptron_layer->set_activation_function(PerceptronLayer::Logistic);

    Tensor<type,2 > biases_perceptron(neurons_number, 1);
    biases_perceptron.setConstant(1);
    perceptron_layer->set_biases(biases_perceptron);
    Tensor<type,2 > synaptic_weights_perceptron(inputs_number, neurons_number);
    synaptic_weights_perceptron.setConstant(1);
    perceptron_layer->set_synaptic_weights(synaptic_weights_perceptron);

    NeuralNetwork::ForwardPropagation forward_propagation(dataset.get_training_instances_number(), &neural_network);

    neural_network.forward_propagate(batch, forward_propagation);

    Tensor<type, 2>perceptron_combinations = forward_propagation.layers[0].combinations_2d;

    Tensor<type, 2>perceptron_activations = forward_propagation.layers[0].activations_2d;

    assert_true(perceptron_combinations.dimension(0) == 5, LOG);
    assert_true(abs(perceptron_combinations(0,0) - 3) < static_cast<type>(1e-3)
             && abs(perceptron_combinations(1,0) - 5) < static_cast<type>(1e-3)
             && abs(perceptron_combinations(2,0) - 7) < static_cast<type>(1e-3)
             && abs(perceptron_combinations(3,0) - 1) < static_cast<type>(1e-3)
             && abs(perceptron_combinations(4,0) - 1) < static_cast<type>(1e-3), LOG);

    assert_true(perceptron_activations.dimension(0) == 5, LOG);
    assert_true(abs(perceptron_activations(0,0) - static_cast<type>(0.952)) < static_cast<type>(1e-3)
             && abs(perceptron_activations(1,0) - static_cast<type>(0.993)) < static_cast<type>(1e-3)
             && abs(perceptron_activations(2,0) - static_cast<type>(0.999)) < static_cast<type>(1e-3)
             && abs(perceptron_activations(3,0) - static_cast<type>(0.731)) < static_cast<type>(1e-3)
             && abs(perceptron_activations(4,0) - static_cast<type>(0.731)) < static_cast<type>(1e-3), LOG);

    // Test 2

    inputs_number = 4;
    target_number = 1;

    data.resize(3, 5);

    data.setValues({{-1,1,-1,1,0,0},{-2,2,3,1,0},{-3,3,5,1,0}});

        //DataSet

    dataset.set(data);

    dataset.set_training();

    DataSet::Batch batch_3(3, &dataset);

    training_instances_indices = dataset.get_training_instances_indices();
    inputs_indices = dataset.get_input_variables_indices();
    targets_indices = dataset.get_target_variables_indices();

    batch_3.fill(training_instances_indices, inputs_indices, targets_indices);

        //NeuralNetwork

    neural_network.set();
    Tensor<Layer*, 1> layers_tensor(2);
    layers_tensor.setValues({new PerceptronLayer(inputs_number,7), new ProbabilisticLayer(7,target_number)});
    neural_network.set_layers_pointers(layers_tensor);
    neural_network.set_thread_pool_device(thread_pool_device);

    PerceptronLayer* perceptron_layer_3 = dynamic_cast<PerceptronLayer*>(neural_network.get_layer_pointer(0));
    const Index neurons_number_3_0 = perceptron_layer_3->get_neurons_number();
    perceptron_layer_3->set_activation_function(PerceptronLayer::Logistic);

    ProbabilisticLayer* probabilistic_layer_3 = dynamic_cast<ProbabilisticLayer*>(neural_network.get_layer_pointer(1));
    const Index neurons_number_3_1 = probabilistic_layer_3->get_neurons_number();
    probabilistic_layer_3->set_activation_function(ProbabilisticLayer::Softmax);

    Tensor<type,2 > biases_pl(neurons_number, 1);
    biases_pl.setConstant(5);
    perceptron_layer_3->set_biases(biases_pl);
    Tensor<type,2 > synaptic_weights_pl(inputs_number, neurons_number_3_0);
    synaptic_weights_pl.setConstant(-1);
    perceptron_layer_3->set_synaptic_weights(synaptic_weights_pl);

    Tensor<type,2 > biases_pbl(neurons_number, 1);
    biases_pbl.setConstant(3);
    probabilistic_layer_3->set_biases(biases_pbl);
    Tensor<type,2 > synaptic_pbl(neurons_number_3_0, neurons_number_3_1);
    synaptic_pbl.setConstant(1);
    probabilistic_layer_3->set_synaptic_weights(synaptic_pbl);


    NeuralNetwork::ForwardPropagation forward_propagation_3(dataset.get_training_instances_number(), &neural_network);

    neural_network.forward_propagate(batch_3, forward_propagation_3);

    Tensor<type, 2>perceptron_combinations_3_0 = forward_propagation_3.layers[0].combinations_2d;
    Tensor<type, 2>perceptron_activations_3_0 = forward_propagation_3.layers[0].activations_2d;
    Tensor<type, 2>perceptron_combinations_3_1 = forward_propagation_3.layers[1].combinations_2d;
    Tensor<type, 2>perceptron_activations_3_1= forward_propagation_3.layers[1].activations_2d;

    assert_true(perceptron_combinations_3_0.dimension(0) == 3, LOG);
    assert_true(abs(perceptron_combinations_3_0(0,0) - 5) < static_cast<type>(1e-3)
             && abs(perceptron_combinations_3_0(1,0) - 1) < static_cast<type>(1e-3)
             && abs(perceptron_combinations_3_0(2,0) + 1) < static_cast<type>(1e-3), LOG);

    assert_true(perceptron_activations_3_0.dimension(0) == 3, LOG);
    assert_true(abs(perceptron_activations_3_0(0,0) - static_cast<type>(0.993)) < static_cast<type>(1e-3)
             && abs(perceptron_activations_3_0(1,0) - static_cast<type>(0.731)) < static_cast<type>(1e-3)
             && abs(perceptron_activations_3_0(2,0) - static_cast<type>(0.268)) < static_cast<type>(1e-3), LOG);

    assert_true(perceptron_combinations_3_1.dimension(0) == 3, LOG);
    assert_true(abs(perceptron_combinations_3_1(0,0) - static_cast<type>(3.993)) < static_cast<type>(1e-3)
             && abs(perceptron_combinations_3_1(1,0) - static_cast<type>(3.731)) < static_cast<type>(1e-3)
             && abs(perceptron_combinations_3_1(2,0) - static_cast<type>(3.268)) < static_cast<type>(1e-3), LOG);

    assert_true(perceptron_activations_3_1.dimension(0) == 3, LOG);
    assert_true(abs(perceptron_activations_3_1(0,0) - static_cast<type>(0.443)) < static_cast<type>(1e-3)
             && abs(perceptron_activations_3_1(1,0) - static_cast<type>(0.341)) < static_cast<type>(1e-3)
             && abs(perceptron_activations_3_1(2,0) - static_cast<type>(0.215)) < static_cast<type>(1e-3), LOG);
}


void NeuralNetworkTest::run_test_case()
{
   cout << "Running neural network test case...\n";

   // Constructor and destructor methods

   test_constructor();

   test_destructor();

   // Appending layers

   test_add_layer();
   check_layer_type();

   // Get methods

   test_has_methods();

   test_get_inputs();
   test_get_outputs();

   test_get_trainable_layers();
   test_get_layers_type_pointers();
   test_get_layer_pointer();

   // Set methods

   test_set();

   test_set_names();
   test_set_number();

   test_set_pointers();

   test_set_display();

   // Layers

   test_get_layers_number();

   // Architecture

   test_inputs_outputs_number();

   test_get_architecture();

   // Parameters methods

   test_get_parameters();
   test_get_trainable_layers_parameters();

   test_set_parameters();

   // Parameters initialization methods

   test_set_parameters_constant();
   test_set_parameters_random();

   // Parameters norm / descriptives / histogram

   test_calculate_parameters_norm();
   test_calculate_parameters_descriptives();
   test_calculate_parameters_histogram();

   test_perturbate_parameters();

   //Output

   test_calculate_outputs();
   test_calculate_trainable_outputs();

   test_calculate_directional_inputs();
   test_calculate_outputs_histograms();

   // Expression methods

   test_print();
   test_write_expression();

   //Forward propagate

   test_forward_propagate();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   test_save();

   test_load();

   cout << "End of neural network test case.\n";
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
