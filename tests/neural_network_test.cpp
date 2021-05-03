//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "neural_network_test.h"

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

  // Test 1 / Model type constructor

  NeuralNetwork neural_network_1_1(NeuralNetwork::Approximation, {1, 4, 2});

  assert_true(neural_network_1_1.get_layers_number() == 5, LOG);
  assert_true(neural_network_1_1.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);
  assert_true(neural_network_1_1.get_layer_pointer(1)->get_type() == Layer::Perceptron, LOG);
  assert_true(neural_network_1_1.get_layer_pointer(2)->get_type() == Layer::Perceptron, LOG);
  assert_true(neural_network_1_1.get_layer_pointer(3)->get_type() == Layer::Unscaling, LOG);
  assert_true(neural_network_1_1.get_layer_pointer(4)->get_type() == Layer::Bounding, LOG);

  NeuralNetwork neural_network_1_2(NeuralNetwork::Classification, {1, 4, 2});

  assert_true(neural_network_1_2.get_layers_number() == 3, LOG);
  assert_true(neural_network_1_2.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);
  assert_true(neural_network_1_2.get_layer_pointer(1)->get_type() == Layer::Perceptron, LOG);
  assert_true(neural_network_1_2.get_layer_pointer(2)->get_type() == Layer::Probabilistic, LOG);

  NeuralNetwork neural_network_1_3(NeuralNetwork::Forecasting, {1, 4, 2});

  assert_true(neural_network_1_3.get_layers_number() == 4, LOG);
  assert_true(neural_network_1_3.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);
  assert_true(neural_network_1_3.get_layer_pointer(1)->get_type() == Layer::LongShortTermMemory, LOG);
  assert_true(neural_network_1_3.get_layer_pointer(2)->get_type() == Layer::Perceptron, LOG);
  assert_true(neural_network_1_3.get_layer_pointer(3)->get_type() == Layer::Unscaling, LOG);

  NeuralNetwork neural_network_1_4(NeuralNetwork::ImageApproximation, {1, 4, 2});

  assert_true(neural_network_1_4.get_layers_number() == 1, LOG);
  assert_true(neural_network_1_4.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);

//  NeuralNetwork neural_network_1_5(NeuralNetwork::ImageClassification, architecture);

//  assert_true(neural_network_1_5.get_layers_number() == 1, LOG);
//  assert_true(neural_network_1_5.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);

  // Test 2 / Convolutional layer constructor

//  Tensor<Index, 1> new_inputs_dimensions(1);
//  new_inputs_dimensions.setConstant(1);

//  Index new_blocks_number = 1;

//  Tensor<Index, 1> new_filters_dimensions(1);
//  new_filters_dimensions.setConstant(1);

//  Index new_outputs_number = 1;

//  ConvolutionalLayer convolutional_layer(1,1); //CC -> cl(inputs_dim, filters_dim)

//  NeuralNetwork neural_network_2(new_inputs_dimensions, new_blocks_number, new_filters_dimensions, new_outputs_number);

//  assert_true(neural_network_2.is_empty(), LOG);
//  assert_true(neural_network_2.get_layers_number() == 0, LOG);


  // Test 3_1 / Layers constructor // Test other type of layers

  Tensor<Layer*, 1> layers_2;

  NeuralNetwork neural_network_3_1(layers_2);

  assert_true(neural_network_3_1.is_empty(), LOG);
  assert_true(neural_network_3_1.get_layers_number() == 0, LOG);

  PerceptronLayer* perceptron_layer_3_1 = new PerceptronLayer(1, 1);

  neural_network_3_1.add_layer(perceptron_layer_3_1);

  assert_true(!neural_network_3_1.is_empty(), LOG);
  assert_true(neural_network_3_1.get_layer_pointer(0)->get_type() == Layer::Perceptron, LOG);

  // Test 3_2

  Tensor<Layer*, 1> layers_3(7);

  layers_3.setValues({new ScalingLayer, new PerceptronLayer,
                          new PoolingLayer, new ProbabilisticLayer, new UnscalingLayer, new PerceptronLayer,
             new BoundingLayer});

  NeuralNetwork neural_network_3_2(layers_3);

  assert_true(!neural_network_3_2.is_empty(), LOG);
  assert_true(neural_network_3_2.get_layers_number() == 7, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(1)->get_type() == Layer::Perceptron, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(2)->get_type() == Layer::Pooling, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(3)->get_type() == Layer::Probabilistic, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(4)->get_type() == Layer::Unscaling, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(5)->get_type() == Layer::Perceptron, LOG);
  assert_true(neural_network_3_2.get_layer_pointer(6)->get_type() == Layer::Bounding, LOG);

   // File constructor @todo

//   neural_network.save(file_name);
//   NeuralNetwork nn6(file_name);

}

void NeuralNetworkTest::test_destructor()
{
   cout << "test_destructor\n";

   NeuralNetwork* neural_network_1 = new NeuralNetwork;

   delete neural_network_1;
}


void NeuralNetworkTest::test_get_display()
{
   cout << "test_get_display\n";
}


void NeuralNetworkTest::test_add_layer()
{
   cout << "test_add_layer\n";

    // LSTM

   neural_network.set();

   LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer;
   neural_network.add_layer(long_short_term_memory_layer);
   assert_true(neural_network.get_layers_number() == 1, LOG);
   assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::LongShortTermMemory, LOG);

    // RECURRENT

   neural_network.set();

   RecurrentLayer* recurrent_layer = new RecurrentLayer;
   neural_network.add_layer(recurrent_layer);
   assert_true(neural_network.get_layers_number() == 1, LOG);
   assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Recurrent, LOG);

   // SCALING

   neural_network.set();

   ScalingLayer* scaling_layer = new ScalingLayer(1);
   neural_network.add_layer(scaling_layer);
   assert_true(neural_network.get_layers_number() == 1, LOG);
   assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);

   // CONVOLUTIONAL

//   ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer;
//   neural_network.add_layer(convolutional_layer);
//   assert_true(neural_network.get_layers_number() == 2, LOG);
//   assert_true(neural_network.get_layer_pointer(1)->get_type() == Layer::Convolutional, LOG);

   // PERCEPTRON

   PerceptronLayer* perceptron_layer = new PerceptronLayer;
   neural_network.add_layer(perceptron_layer);
   assert_true(neural_network.get_layers_number() == 2, LOG);
   assert_true(neural_network.get_layer_pointer(1)->get_type() == Layer::Perceptron, LOG);

   // POOLING

   PoolingLayer* pooling_layer = new PoolingLayer;

   neural_network.add_layer(pooling_layer);
   assert_true(neural_network.get_layers_number() == 3, LOG);
   assert_true(neural_network.get_layer_pointer(2)->get_type() == Layer::Pooling, LOG);

   //PROBABILISTIC

   ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer;

   neural_network.add_layer(probabilistic_layer);
   assert_true(neural_network.get_layers_number() == 4, LOG);
   assert_true(neural_network.get_layer_pointer(3)->get_type() == Layer::Probabilistic, LOG);

   // UNSCALING

   UnscalingLayer* unscaling_layer = new UnscalingLayer;

   neural_network.add_layer(unscaling_layer);
   assert_true(neural_network.get_layers_number() == 5, LOG);
   assert_true(neural_network.get_layer_pointer(4)->get_type() == Layer::Unscaling, LOG);

   // BOUNDING

   BoundingLayer* bounding_layer = new BoundingLayer;

   neural_network.add_layer(bounding_layer);
   assert_true(neural_network.get_layers_number() == 6, LOG);
   assert_true(neural_network.get_layer_pointer(5)->get_type() == Layer::Bounding, LOG);

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

   // Test 2

   Tensor<Layer*, 1> layers_tensor_1(1);
   layers_tensor_1.setValues({new PerceptronLayer});

   NeuralNetwork neural_network_1(layers_tensor_1);

   assert_true(!neural_network_1.check_layer_type(Layer::LongShortTermMemory), LOG);
   assert_true(!neural_network_1.check_layer_type(Layer::Recurrent), LOG);

   assert_true(neural_network_1.check_layer_type(Layer::Convolutional),LOG);
   assert_true(neural_network_1.check_layer_type(Layer::Perceptron),LOG);

   // Test 3

   Tensor<Layer*, 1> layers_tensor_2(1);
   layers_tensor_2.setValues({new ScalingLayer});

   NeuralNetwork neural_network_2(layers_tensor_2);

   assert_true(neural_network_2.check_layer_type(Layer::LongShortTermMemory), LOG);
   assert_true(neural_network_2.check_layer_type(Layer::Recurrent), LOG);

   assert_true(neural_network_2.check_layer_type(Layer::Scaling), LOG);
   assert_true(neural_network_2.check_layer_type(Layer::Convolutional), LOG);
   assert_true(neural_network_2.check_layer_type(Layer::Perceptron), LOG);
   assert_true(neural_network_2.check_layer_type(Layer::Pooling), LOG);
   assert_true(neural_network_2.check_layer_type(Layer::Probabilistic), LOG);
   assert_true(neural_network_2.check_layer_type(Layer::Unscaling), LOG);
   assert_true(neural_network_2.check_layer_type(Layer::Bounding), LOG);

   // Test 4

   Tensor<Layer*, 1> layers_tensor_3(1);
   layers_tensor_3.setValues({new RecurrentLayer});

   NeuralNetwork neural_network_3(layers_tensor_3);

   assert_true(!neural_network_3.check_layer_type(Layer::Recurrent),LOG);
   assert_true(!neural_network_3.check_layer_type(Layer::LongShortTermMemory),LOG);
}

void NeuralNetworkTest::test_has_methods()
{
   cout << "test_had_methods\n";

   // Test 0

   Tensor<Layer*, 1> layer_tensor;

   NeuralNetwork neural_network(layer_tensor);

   assert_true(neural_network.is_empty(), LOG);

   assert_true(!neural_network.has_scaling_layer(), LOG);
   assert_true(!neural_network.has_long_short_term_memory_layer(), LOG);
   assert_true(!neural_network.has_recurrent_layer(), LOG);
   assert_true(!neural_network.has_unscaling_layer(), LOG);
   assert_true(!neural_network.has_bounding_layer(), LOG);
   assert_true(!neural_network.has_probabilistic_layer(), LOG);

   // Test 1

   Tensor<Layer*, 1> layer_tensor_1(6);
   layer_tensor_1.setValues({new ScalingLayer, new LongShortTermMemoryLayer,
                             new RecurrentLayer, new UnscalingLayer, new BoundingLayer, new ProbabilisticLayer});

   NeuralNetwork neural_network_1(layer_tensor_1);

   assert_true(neural_network_1.has_scaling_layer(), LOG);
   assert_true(neural_network_1.has_long_short_term_memory_layer(), LOG);
   assert_true(neural_network_1.has_recurrent_layer(), LOG);
   assert_true(neural_network_1.has_unscaling_layer(), LOG);
   assert_true(neural_network_1.has_bounding_layer(), LOG);
   assert_true(neural_network_1.has_probabilistic_layer(), LOG);

   // Test 2

   Tensor<Layer*, 1> layer_tensor_2(4);
   layer_tensor_2.setValues({new ScalingLayer, new LongShortTermMemoryLayer,
                             new RecurrentLayer, new UnscalingLayer});

   NeuralNetwork neural_network_2(layer_tensor_2);

   assert_true(neural_network_2.has_scaling_layer(), LOG);
   assert_true(neural_network_2.has_long_short_term_memory_layer(), LOG);
   assert_true(neural_network_2.has_recurrent_layer(), LOG);
   assert_true(neural_network_2.has_unscaling_layer(), LOG);
   assert_true(!neural_network_2.has_bounding_layer(), LOG);
   assert_true(!neural_network_2.has_probabilistic_layer(), LOG);
}


void NeuralNetworkTest::test_get_inputs()
{
   cout << "test_get_inputs\n";

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

   // Test 0

   Tensor<Layer*,1> layer_tensor_0(4);
   layer_tensor_0.setValues({new ScalingLayer, new PerceptronLayer, new UnscalingLayer, new ProbabilisticLayer});

   NeuralNetwork neural_network_0(layer_tensor_0);

   assert_true(neural_network_0.get_trainable_layers_pointers()(0)->get_type() == Layer::Perceptron, LOG);
   assert_true(neural_network_0.get_trainable_layers_pointers()(1)->get_type() == Layer::Probabilistic, LOG);

   assert_true(neural_network_0.get_trainable_layers_indices()(0) == 1, LOG);
   assert_true(neural_network_0.get_trainable_layers_indices()(1) == 3, LOG);


   // Test 1

   Tensor<Layer*, 1> layer_tensor(9);
   layer_tensor.setValues({new ScalingLayer, new ConvolutionalLayer, new PerceptronLayer,
                           new PoolingLayer, new ProbabilisticLayer, new LongShortTermMemoryLayer,
                           new RecurrentLayer, new UnscalingLayer, new BoundingLayer});

   NeuralNetwork neural_network(layer_tensor);

   assert_true(neural_network.get_trainable_layers_pointers()(0)->get_type() == Layer::Convolutional, LOG);
   assert_true(neural_network.get_trainable_layers_pointers()(1)->get_type() == Layer::Perceptron, LOG);
   assert_true(neural_network.get_trainable_layers_pointers()(2)->get_type() == Layer::Pooling, LOG);
   assert_true(neural_network.get_trainable_layers_pointers()(3)->get_type() == Layer::Probabilistic, LOG);
   assert_true(neural_network.get_trainable_layers_pointers()(4)->get_type() == Layer::LongShortTermMemory, LOG);
   assert_true(neural_network.get_trainable_layers_pointers()(5)->get_type() == Layer::Recurrent, LOG);

   assert_true(neural_network.get_trainable_layers_indices()(0) == 1, LOG);
   assert_true(neural_network.get_trainable_layers_indices()(1) == 2, LOG);
   assert_true(neural_network.get_trainable_layers_indices()(2) == 3, LOG);
   assert_true(neural_network.get_trainable_layers_indices()(3) == 4, LOG);
   assert_true(neural_network.get_trainable_layers_indices()(4) == 5, LOG);
   assert_true(neural_network.get_trainable_layers_indices()(5) == 6, LOG);
}


void NeuralNetworkTest::test_get_layers_type_pointers()
{
   cout << "test_get_layers_type_pointers\n";

   // Test 1

   Tensor<Layer*, 1> layer_tensor(9);
   layer_tensor.setValues({new ScalingLayer, new ConvolutionalLayer, new PerceptronLayer,
                           new PoolingLayer, new ProbabilisticLayer, new LongShortTermMemoryLayer,
                           new RecurrentLayer, new UnscalingLayer, new BoundingLayer});

   NeuralNetwork neural_network(layer_tensor);

   assert_true(neural_network.get_scaling_layer_pointer()->get_type() == Layer::Scaling, LOG);
   assert_true(neural_network.get_unscaling_layer_pointer()->get_type() == Layer::Unscaling, LOG);
   assert_true(neural_network.get_bounding_layer_pointer()->get_type() == Layer::Bounding, LOG);
   assert_true(neural_network.get_probabilistic_layer_pointer()->get_type() == Layer::Probabilistic, LOG);
   assert_true(neural_network.get_long_short_term_memory_layer_pointer()->get_type() == Layer::LongShortTermMemory, LOG);
   assert_true(neural_network.get_recurrent_layer_pointer()->get_type() == Layer::Recurrent, LOG);

   assert_true(neural_network.get_first_perceptron_layer_pointer()->get_type() == Layer::Perceptron, LOG);

   // Test 2

   Tensor<Layer*,1> layer_tensor_1(3);
   layer_tensor_1.setValues({new ScalingLayer, new PerceptronLayer, new UnscalingLayer});

   NeuralNetwork neural_network_1(layer_tensor_1);

   assert_true(neural_network_1.get_scaling_layer_pointer()->get_type() == Layer::Scaling, LOG);
   assert_true(neural_network_1.get_unscaling_layer_pointer()->get_type() == Layer::Unscaling, LOG);
   assert_true(neural_network_1.get_first_perceptron_layer_pointer()->get_type() == Layer::Perceptron, LOG);

}


void NeuralNetworkTest::test_get_layer_pointer()
{
   cout << "test_get_layer_pointer\n";

   Tensor<Layer*, 1> layer_tensor(9);
   layer_tensor.setValues({new ScalingLayer, new ConvolutionalLayer, new PerceptronLayer,
                           new PoolingLayer, new ProbabilisticLayer, new LongShortTermMemoryLayer,
                           new RecurrentLayer, new UnscalingLayer, new BoundingLayer});

   NeuralNetwork neural_network(layer_tensor);

   assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Scaling, LOG);
   assert_true(neural_network.get_layer_pointer(1)->get_type() == Layer::Convolutional, LOG);
   assert_true(neural_network.get_layer_pointer(2)->get_type() == Layer::Perceptron, LOG);
   assert_true(neural_network.get_layer_pointer(3)->get_type() == Layer::Pooling, LOG);
   assert_true(neural_network.get_layer_pointer(4)->get_type() == Layer::Probabilistic, LOG);
   assert_true(neural_network.get_layer_pointer(5)->get_type() == Layer::LongShortTermMemory, LOG);
   assert_true(neural_network.get_layer_pointer(6)->get_type() == Layer::Recurrent, LOG);
   assert_true(neural_network.get_layer_pointer(7)->get_type() == Layer::Unscaling, LOG);
   assert_true(neural_network.get_layer_pointer(8)->get_type() == Layer::Bounding, LOG);

//   assert_true(neural_network.get_output_layer_pointer()->get_type() == Layer::Bounding, LOG);
}


void NeuralNetworkTest::test_set()
{
   cout << "test_set\n";

   // Test 0

   neural_network.set();

   assert_true(neural_network.get_inputs_names().size() == 0, LOG);
   assert_true(neural_network.get_outputs_names().size() == 0, LOG);
   assert_true(neural_network.get_layers_pointers().size() == 0, LOG);

   // Test 1

   neural_network.set(NeuralNetwork::Approximation, {1,0,1});
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
   assert_true(neural_network.get_layers_pointers().size() == 5, LOG);

   neural_network.set(NeuralNetwork::Classification, {1,0,1});
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
   assert_true(neural_network.get_layers_pointers().size() == 3, LOG);

   neural_network.set(NeuralNetwork::Forecasting, {1,0,1});
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
   assert_true(neural_network.get_layers_pointers().size() == 4, LOG);

   neural_network.set(NeuralNetwork::ImageApproximation, {1,0,1});
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
   assert_true(neural_network.get_layers_pointers().size() == 1, LOG);

   neural_network.set(NeuralNetwork::ImageClassification, {1,0,1});
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
   assert_true(neural_network.get_layers_pointers().size() == 1, LOG);

   // Test 2 / Convolutional layer set

   Tensor<Index, 1> new_inputs_dimensions(1);
   new_inputs_dimensions.setConstant(1);

   Index new_blocks_number = 1;

   Tensor<Index, 1> new_filters_dimensions(1);
   new_filters_dimensions.setConstant(1);

   Index new_outputs_number = 1;

//   ConvolutionalLayer convolutional_layer(1,1); //CC -> cl(inputs_dim, filters_dim)

   neural_network.set(new_inputs_dimensions, new_blocks_number, new_filters_dimensions, new_outputs_number);

   assert_true(neural_network.is_empty(), LOG);
   assert_true(neural_network.get_layers_number() == 0, LOG);

   // Test 3

   neural_network.set(NeuralNetwork::Approximation, {1,1,1,1});
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);
   assert_true(neural_network.get_layers_pointers().size() == 6, LOG);

   neural_network.set(NeuralNetwork::Classification, {1,1,1,1});
   assert_true(neural_network.get_inputs_names().size() == 1, LOG);
   assert_true(neural_network.get_outputs_names().size() == 1, LOG);
   assert_true(neural_network.get_layers_pointers().size() == 4, LOG);
}


void NeuralNetworkTest::test_set_names()
{
   cout << "test_set_names\n";

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


void NeuralNetworkTest::test_set_inputs_number()
{
   cout << "test_set_inputs_number\n";

   Index inputs_number = 0;

   Tensor<bool, 1> inputs;

   // Test

   neural_network.set(NeuralNetwork::Classification, {1,0,1});

   inputs_number = 0;

   neural_network.set_inputs_number(inputs_number);

   assert_true(neural_network.get_inputs_number() == 0, LOG);
   assert_true(neural_network.get_layer_pointer(0)->get_inputs_number() == 0, LOG); //CC -> Scaling layer nmb assert

   // Test

   inputs_number = 3;

   neural_network.set_inputs_number(inputs_number);
   assert_true(neural_network.get_inputs_number() == 3, LOG);
   assert_true(neural_network.get_layer_pointer(0)->get_inputs_number() == 3, LOG); //CC -> Scaling layer nmb assert

   // Test 1_0

   Tensor<bool, 1> inputs_names_1;
   neural_network.set_inputs_number(inputs_names_1);

   assert_true(neural_network.get_inputs_number() == 0, LOG);

   // Test 1_1

   Tensor<bool, 1> inputs_names_1_1(2);
   inputs_names_1_1.setValues({true,false});

   neural_network.set_inputs_number(inputs_names_1_1);
   assert_true(neural_network.get_inputs_number() == 1, LOG);

   // Test 1_2

   Tensor<bool, 1> inputs_names_1_2(2);
   inputs_names_1_2.setValues({true,true});

   neural_network.set_inputs_number(inputs_names_1_2);
   assert_true(neural_network.get_inputs_number() ==2 , LOG);

}


void NeuralNetworkTest::test_set_default()
{
   cout << "test_set_default\n";

   neural_network.set_default();

   assert_true(neural_network.get_display(), LOG);
}


void NeuralNetworkTest::test_set_pointers()
{
   cout << "test_set_pointers\n";

   NeuralNetwork neural_network(NeuralNetwork::Classification, {1,0,1});

   // Test 1 // Device

   assert_true(neural_network.get_layers_number() == 3, LOG);

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

   neural_network.set_display(true);
   assert_true(neural_network.get_display(), LOG);

   neural_network.set_display(false);
   assert_true(!neural_network.get_display(), LOG);
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
   assert_true(neural_network.get_inputs_number() == 0, LOG);

   // Test 1

   Tensor<Layer*, 1> layers_tensor_1(9);

   layers_tensor_1.setValues({new ScalingLayer, new ConvolutionalLayer, new PerceptronLayer,
                           new PoolingLayer, new ProbabilisticLayer, new LongShortTermMemoryLayer,
                           new RecurrentLayer, new UnscalingLayer, new BoundingLayer});

   NeuralNetwork neural_network_1(layers_tensor_1);

   assert_true(neural_network_1.get_layers_number() == 9, LOG);
   assert_true(neural_network_1.get_trainable_layers_number() == 6, LOG);
   assert_true(neural_network_1.get_perceptron_layers_number() == 1, LOG);
   assert_true(neural_network_1.get_probabilistic_layers_number() == 1, LOG);
   assert_true(neural_network_1.get_layers_neurons_numbers()(0) == 0, LOG);
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

   Tensor<Layer*, 1> layers_tensor_1(3);
   layers_tensor_1.setValues({new ScalingLayer(5),new PerceptronLayer(5,2), new UnscalingLayer(2)});

   NeuralNetwork neural_network_1(layers_tensor_1);

   assert_true(neural_network_1.get_inputs_number() == 5, LOG);
   assert_true(neural_network_1.get_outputs_number() == 2, LOG);

   // Test 2

   Tensor<Layer*, 1> layers_tensor_2(4);
   layers_tensor_2.setValues({new ScalingLayer(1),new PerceptronLayer(1,2), new UnscalingLayer(2), new BoundingLayer(2)});

   NeuralNetwork neural_network_2(layers_tensor_2);

   assert_true(neural_network_2.get_inputs_number() == 1, LOG);
   assert_true(neural_network_2.get_outputs_number() == 2, LOG);

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

   NeuralNetwork neural_network_1_1(NeuralNetwork::Approximation, {1,4,2});

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

   Tensor<type, 1> parameters;

   // Test 0

   neural_network.set();
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 0, LOG);
   assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);

   // Test 1

   neural_network.set(NeuralNetwork::Approximation, {1,2,1});

   PerceptronLayer* pl00 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(1));
   pl00->set_biases_constant(0);
   pl00->set_synaptic_weights_constant(1);

   PerceptronLayer* pl01 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(2));
   pl01->set_biases_constant(2);
   pl01->set_synaptic_weights_constant(3);

   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 7, LOG);
   assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
   assert_true(abs(parameters(1) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(5) - 3) < static_cast<type>(1e-5), LOG);

   // Test 2

   neural_network.set(NeuralNetwork::Approximation, {1,1,1,1});

   PerceptronLayer* pl10 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(1));
   pl10->set_biases_constant(0);
   pl10->set_synaptic_weights_constant(1);

   PerceptronLayer* pl11 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(2));
   pl11->set_biases_constant(2);
   pl11->set_synaptic_weights_constant(3);

   PerceptronLayer* pl12 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(3));
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

   // Test 3

   neural_network.set(NeuralNetwork::Approximation, {1,1,1,1,1});

   PerceptronLayer* pl20 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(1));
   pl20->set_biases_constant(0);
   pl20->set_synaptic_weights_constant(1);

   PerceptronLayer* pl21 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(2));
   pl21->set_biases_constant(2);
   pl21->set_synaptic_weights_constant(3);

   PerceptronLayer* pl22 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(3));
   pl22->set_biases_constant(4);
   pl22->set_synaptic_weights_constant(5);

   PerceptronLayer* pl23 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(4));
   pl23->set_biases_constant(6);
   pl23->set_synaptic_weights_constant(7);


   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 8, LOG);
   assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
   assert_true(abs(parameters(0) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(1) - 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(2) - 2) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(3) - 3) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(4) - 4) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(5) - 5) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(6) - 6) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(7) - 7) < static_cast<type>(1e-5), LOG);

}

void NeuralNetworkTest::test_get_trainable_layers_parameters()
{
    cout << "test_get_trainable_layers_parameters\n";

    Tensor<type, 1> parameters;
    Tensor<Tensor<type, 1>, 1> trainable_layers_parameters;

    // Test 0

    neural_network.set();
    parameters = neural_network.get_parameters();
    trainable_layers_parameters = neural_network.get_trainable_layers_parameters(parameters);

    assert_true(trainable_layers_parameters.rank() == 1, LOG);
    assert_true(trainable_layers_parameters.size() == 0, LOG);

    // Test 1

    neural_network.set(NeuralNetwork::Approximation, {1,2,1});
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

    neural_network.set(NeuralNetwork::Approximation, {1,1,1,1});

    PerceptronLayer* pl0 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(1));
    pl0->set_biases_constant(0);
    pl0->set_synaptic_weights_constant(1);

    PerceptronLayer* pl1 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(2));
    pl1->set_biases_constant(2);
    pl1->set_synaptic_weights_constant(3);

    PerceptronLayer* pl2 = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(3));
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

   Index parameters_number;
   Tensor<type, 1> parameters;

   // Test 0

   neural_network.set_parameters(parameters);
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 0, LOG);

   // Test 1

   neural_network.set(NeuralNetwork::Approximation, {2,2});

   parameters_number = neural_network.get_parameters_number();
   parameters.resize(parameters_number);

   neural_network.set_parameters(parameters);
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 6, LOG);
   assert_true(parameters.size() == parameters_number, LOG);

   parameters.setValues({1,2,3,4,5,6});

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

   Tensor<type, 1> parameters;

   neural_network.set(NeuralNetwork::Approximation, {1,2,1});

   // Test 0

   neural_network.set_parameters_constant(1);
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 7, LOG);
   assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
   assert_true(abs(parameters(1) - 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(5) - 1) < static_cast<type>(1e-5), LOG);

   // Test 1

   neural_network.set_parameters_constant(3);
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 7, LOG);
   assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
   assert_true(abs(parameters(1) - 3) < static_cast<type>(1e-5), LOG);
   assert_true(abs(parameters(5) - 3) < static_cast<type>(1e-5), LOG);

}


void NeuralNetworkTest::test_set_parameters_random()
{
   cout << "test_set_parameters_random\n";

   Tensor<type, 1> parameters;

   Tensor<Index, 1> architecture(3);


   neural_network.set(NeuralNetwork::Approximation, {1,2,1});

   neural_network.set_parameters_random();
   parameters = neural_network.get_parameters();

   assert_true(parameters.size() == 7, LOG);
   assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
}


void NeuralNetworkTest::test_calculate_parameters_norm()
{
   cout << "test_calculate_parameters_norm\n";

   type parameters_norm = 0;
   type parameters_norm_1 = 0;
   

   // Test  0

   neural_network.set(NeuralNetwork::Approximation, {});

   parameters_norm = neural_network.calculate_parameters_norm();

   assert_true(abs(parameters_norm - 0) < static_cast<type>(1e-5), LOG);

   // Test  1

   neural_network.set(NeuralNetwork::Approximation, {1,1,1,1});

   neural_network.set_parameters_constant(1.0);

   parameters_norm = neural_network.calculate_parameters_norm();

   assert_true(abs(parameters_norm - static_cast<type>(sqrt(6))) < static_cast<type>(1e-5), LOG);

   // Test 2

   neural_network.set(NeuralNetwork::Approximation, {1,1,1,1,1});

   neural_network.set_parameters_constant(1.0);

   parameters_norm_1 = neural_network.calculate_parameters_norm();

   assert_true(abs(parameters_norm_1 - static_cast<type>(sqrt(8))) < static_cast<type>(1e-5), LOG);

}


void NeuralNetworkTest::test_perturbate_parameters()
{
   cout << "test_perturbate_parameters\n";

   // Test 1

   Tensor<Index, 1> architecture(3);

   Index parameters_number;
   Tensor<type, 1> parameters;

   neural_network.set(NeuralNetwork::Approximation, {1,1,1});

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

   // Test 2

   Tensor<Index, 1> architecture1;
   NeuralNetwork neural_network1;
   Tensor<type, 1> parameters1;

   parameters1.setConstant(0);

   neural_network1.set_parameters(parameters1);
   parameters1 = neural_network1.get_parameters();

   neural_network1.perturbate_parameters(0.5);
   parameters1 = neural_network1.get_parameters();

   assert_true(parameters1.size() == 0, LOG);
}


void NeuralNetworkTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   Index inputs_number;
   Index outputs_number;

   

   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;

   Index parameters_number;

   Tensor<type, 1> parameters;

   // Test 1

   neural_network.set(NeuralNetwork::Approximation, {3,3,3});
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

   neural_network.set(NeuralNetwork::Approximation, {2, 1, 5});

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

   neural_network.set(NeuralNetwork::Approximation, {1, 2});

   neural_network.set_parameters_constant(1);

   inputs.resize(1, 1);

   inputs.setConstant(2);

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(outputs.size() == 2, LOG);
   assert_true(abs(outputs(0,0) - 2) < static_cast<type>(1e-5), LOG);
   assert_true(abs(outputs(0,1) - tanh(3)) < static_cast<type>(1e-5), LOG);

   // Test 4

   neural_network.set(NeuralNetwork::Approximation, {4, 3, 3});

   inputs.resize(1, 4);

   inputs.setConstant(0);

   neural_network.set_parameters_constant(1);

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(neural_network.calculate_outputs(inputs).size() == 3, LOG);

   assert_true(abs(outputs(0,0) - tanh(3.2847)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0,1) - tanh(3.2847)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0,2) - tanh(3.2847)) < static_cast<type>(1e-3), LOG);

   // Test 5

   neural_network.set(NeuralNetwork::Approximation, {1, 2});

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

   neural_network.set(NeuralNetwork::Approximation, {1,1,1});

   neural_network.set_parameters_constant(0);

   inputs.resize(1, 1);
   inputs.setConstant(0);

   outputs = neural_network.calculate_outputs(inputs);

   assert_true(outputs.size() == 1, LOG);
   assert_true(abs(outputs(0,0) - 0) < static_cast<type>(1e-3), LOG);

   // Test 6_1

   neural_network.set(NeuralNetwork::Classification, {1,1});

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
   Index batch_size;
   batch_size = 5;

   inputs_number = 10;

   ScalingLayer* scaling_layer_3 = new ScalingLayer(inputs_number);
   PerceptronLayer* perceptron_layer_4 = new PerceptronLayer(inputs_number, categories);
   ProbabilisticLayer* probabilistic_layer_5 = new ProbabilisticLayer(categories,categories);

   neural_network_7.add_layer(scaling_layer_3);
   neural_network_7.add_layer(perceptron_layer_4);
   neural_network_7.add_layer(probabilistic_layer_5);

   neural_network_7.set_parameters_constant(-5);

   inputs.resize(batch_size, inputs_number);
   inputs.setConstant(-1);

   outputs = neural_network_7.calculate_outputs(inputs);

   assert_true(outputs.size() == batch_size*categories, LOG);

   assert_true(abs(outputs(0,0) - static_cast<type>(0.333)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(0,categories-1) - static_cast<type>(0.333)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(parameters_number-1,0) - static_cast<type>(0.333)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(parameters_number-1,categories-1) - static_cast<type>(0.333)) < static_cast<type>(1e-3), LOG);

   // Test 8

   neural_network.set(NeuralNetwork::Approximation, {1,3,3,3,1});

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


void NeuralNetworkTest::test_calculate_directional_inputs()
{
   cout << "test_calculate_directional_inputs\n";

   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;
   Tensor<type, 2> trainable_outputs;

   Tensor<type, 1> parameters;

   // Test 0

   neural_network.set(NeuralNetwork::Approximation, {3, 4, 2});
   neural_network.set_parameters_constant(0.0);

   inputs.resize(2,3);
   inputs.setValues({{-5,-1,-3},{7,3,1}});

   Tensor<type, 1> point(3);
   point.setValues({0,0,0});

   Tensor<type, 2> directional_inputs = neural_network.calculate_directional_inputs(0,point,0,0,0);

   assert_true(directional_inputs.rank() == 2, LOG);
   assert_true(directional_inputs.dimension(0) == 0, LOG);

   // Test 1

   point.setValues({1, 2, 3});

   directional_inputs = neural_network.calculate_directional_inputs(2,point,-1,1,3);

   assert_true(directional_inputs.rank() == 2, LOG);
   assert_true(directional_inputs.dimension(0) == 3, LOG);
   assert_true(directional_inputs.dimension(1) == 3, LOG);
   assert_true(abs(directional_inputs(0,2) + 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_inputs(1,2) - 0) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_inputs(2,2) - 1) < static_cast<type>(1e-5), LOG);

   // Test 2

   point.setValues({1, 2, 3});

   directional_inputs = neural_network.calculate_directional_inputs(0, point, -4, 0, 5);

   assert_true(directional_inputs.rank() == 2, LOG);
   assert_true(directional_inputs.dimension(0) == 5, LOG);
   assert_true(directional_inputs.dimension(1) == 3, LOG);
   assert_true(abs(directional_inputs(0,0) + 4) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_inputs(1,0) + 3) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_inputs(2,0) + 2) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_inputs(3,0) + 1) < static_cast<type>(1e-5), LOG);
   assert_true(abs(directional_inputs(4,0) + 0) < static_cast<type>(1e-5), LOG);
}


void NeuralNetworkTest::test_save()
{
   cout << "test_save\n";

   string file_name = "../data/neural_network.xml";

   // Empty neural network

   neural_network.set();
   neural_network.save(file_name);

   // Only network architecture

   neural_network.set(NeuralNetwork::Approximation, {2, 4, 3});
   neural_network.save(file_name);

   neural_network.set(NeuralNetwork::Approximation, {1, 1, 1});

   neural_network.save(file_name);
}


void NeuralNetworkTest::test_load()
{
   cout << "test_load\n";

   string file_name = "../data/neural_network.xml";

   // Empty neural network

   neural_network.save(file_name);
   neural_network.load(file_name);
}


void NeuralNetworkTest::test_print()
{
   cout << "test_print\n";

   // Empty neural network

   //neural_network.print();

   // Only network architecture

   neural_network.set(NeuralNetwork::Approximation, {2, 4, 3});

   //neural_network.print();
}


void NeuralNetworkTest::test_write_expression()
{
   cout << "test_write_expression\n";

   // Test 0

   //neural_network.print();

   // Only network architecture

   neural_network.set(NeuralNetwork::Approximation, {2, 4, 3});

   Tensor<string, 1> inputs_names = neural_network.get_inputs_names();
   Tensor<string, 1> outputs_names = neural_network.get_outputs_names();

   inputs_names.setValues({"x1", "x2"});
    outputs_names.setValues({"y1", "y2", "y3"});

   neural_network.set_inputs_names(inputs_names);
   neural_network.set_outputs_names(outputs_names);

   //cout << neural_network.write_expression(inputs_names, outputs_names);

   // Test 1

   //neural_network.print();

   // Only network architecture

   neural_network.set(NeuralNetwork::Classification, {2, 4, 3});
   Tensor<string, 1> inputs_names_2 = neural_network.get_inputs_names();
   Tensor<string, 1> outputs_names_2 = neural_network.get_outputs_names();

   inputs_names_2.setValues({"x1", "x2"});
   outputs_names_2.setValues({"y1", "y2", "y3"});

   neural_network.set_inputs_names(inputs_names_2);
   neural_network.set_outputs_names(outputs_names_2);

   //cout << neural_network_2.write_expression(inputs_names_2, outputs_names_2);

}

void NeuralNetworkTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    // Test 1

    Index inputs_number = 2;
    Index targets_number = 1;

    Tensor<Index, 1> architecture(2);

    Tensor<type, 2> data(5, 3);

    data.setValues({{1,1,1},{2,2,2},{3,3,3},{0,0,0},{0,0,0}});

    //DataSet

    DataSet dataset(data);

    dataset.set_training();

    DataSetBatch batch(5, &dataset);

    Tensor<Index,1> training_samples_indices = dataset.get_training_samples_indices();
    Tensor<Index,1> inputs_indices = dataset.get_input_variables_indices();
    Tensor<Index,1> targets_indices = dataset.get_target_variables_indices();

    batch.fill(training_samples_indices, inputs_indices, targets_indices);

    //NeuralNetwork

    NeuralNetwork neural_network(NeuralNetwork::Approximation, {inputs_number,targets_number});

    PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(1));
    const Index neurons_number = perceptron_layer->get_neurons_number();
    perceptron_layer->set_activation_function(PerceptronLayer::Logistic);

    Tensor<type,2 > biases_perceptron(neurons_number, 1);
    biases_perceptron.setConstant(1);
    perceptron_layer->set_biases(biases_perceptron);

    Tensor<type,2 > synaptic_weights_perceptron(inputs_number, neurons_number);
    synaptic_weights_perceptron.setConstant(1);
    perceptron_layer->set_synaptic_weights(synaptic_weights_perceptron);

    NeuralNetworkForwardPropagation forward_propagation(dataset.get_training_samples_number(), &neural_network);

    neural_network.forward_propagate(batch, forward_propagation);

    PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
        = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.layers[0]);

    Tensor<type, 2>perceptron_combinations = perceptron_layer_forward_propagation->combinations;

    Tensor<type, 2>perceptron_activations = perceptron_layer_forward_propagation->activations;

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
    targets_number = 1;

    data.resize(3, 5);

    data.setValues({{-1,1,-1,1,0},{-2,2,3,1,0},{-3,3,5,1,0}});

    //DataSet

    dataset.set(data);

    dataset.set_training();

    DataSetBatch batch_3(3, &dataset);

    training_samples_indices = dataset.get_training_samples_indices();
    inputs_indices = dataset.get_input_variables_indices();
    targets_indices = dataset.get_target_variables_indices();

    batch_3.fill(training_samples_indices, inputs_indices, targets_indices);

    //NeuralNetwork
    NeuralNetwork neural_network_2;

    neural_network_2.set();

    PerceptronLayer* perceptron_layer_3 = new PerceptronLayer(inputs_number, targets_number);
    const Index neurons_number_3_0 = perceptron_layer_3->get_neurons_number();
    perceptron_layer_3->set_activation_function(PerceptronLayer::Logistic);

    ProbabilisticLayer* probabilistic_layer_3 = new ProbabilisticLayer(targets_number, targets_number);

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

    Tensor<Layer*, 1> layers_tensor(2);

    layers_tensor.setValues({perceptron_layer_3, probabilistic_layer_3});
    neural_network_2.set_layers_pointers(layers_tensor);
    NeuralNetworkForwardPropagation forward_propagation_3(dataset.get_training_samples_number(), &neural_network_2);

    neural_network_2.forward_propagate(batch_3, forward_propagation_3);

//    Tensor<type, 2> perceptron_combinations_3_0 = forward_propagation_3.layers[0].combinations;
//    Tensor<type, 2> perceptron_activations_3_0 = forward_propagation_3.layers[0].activations;
//    Tensor<type, 2> probabilistic_combinations_3_1 = forward_propagation_3.layers[1].combinations;
//    Tensor<type, 2> probabilistic_activations_3_1= forward_propagation_3.layers[1].activations;

//    assert_true(perceptron_combinations_3_0.dimension(0) == 3, LOG);

//    assert_true(abs(perceptron_combinations_3_0(0,0) - 5) < static_cast<type>(1e-3)
//             && abs(perceptron_combinations_3_0(1,0) - 1) < static_cast<type>(1e-3)
//             && abs(perceptron_combinations_3_0(2,0) + 1) < static_cast<type>(1e-3), LOG);

//    assert_true(perceptron_activations_3_0.dimension(0) == 3, LOG);
//    assert_true(abs(perceptron_activations_3_0(0,0) - static_cast<type>(0.993)) < static_cast<type>(1e-3)
//             && abs(perceptron_activations_3_0(1,0) - static_cast<type>(0.731)) < static_cast<type>(1e-3)
//             && abs(perceptron_activations_3_0(2,0) - static_cast<type>(0.268)) < static_cast<type>(1e-3), LOG);

//    assert_true(probabilistic_combinations_3_1.dimension(0) == 3, LOG);
//    assert_true(abs(probabilistic_combinations_3_1(0,0) - static_cast<type>(3.993)) < static_cast<type>(1e-3)
//             && abs(probabilistic_combinations_3_1(1,0) - static_cast<type>(3.731)) < static_cast<type>(1e-3)
//             && abs(probabilistic_combinations_3_1(2,0) - static_cast<type>(3.268)) < static_cast<type>(1e-3), LOG);

//    assert_true(probabilistic_activations_3_1.dimension(0) == 3, LOG);
//    assert_true(abs(probabilistic_activations_3_1(0,0) - 1) < static_cast<type>(1e-3)
//             && abs(probabilistic_activations_3_1(1,0) - 1) < static_cast<type>(1e-3)
//             && abs(probabilistic_activations_3_1(2,0) - 1) < static_cast<type>(1e-3), LOG);
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
   test_set_inputs_number();

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
   test_perturbate_parameters();

   //Output

   test_calculate_outputs();

   test_calculate_directional_inputs();

   // Expression methods

   test_print();
   test_write_expression();

   //Forward propagate

   test_forward_propagate();

   // Serialization methods

   test_save();

   test_load();

   cout << "End of neural network test case.\n\n";
}


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
