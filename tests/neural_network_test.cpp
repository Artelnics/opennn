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

    Tensor<Layer*, 1> layers;

    // Default constructor

    NeuralNetwork neural_network_0;

    assert_true(neural_network_0.is_empty(), LOG);
    assert_true(neural_network_0.get_layers_number() == 0, LOG);

    // Model type constructors

    //  Approximation

    NeuralNetwork neural_network_1(NeuralNetwork::ProjectType::Approximation, {1, 4, 2});

    assert_true(neural_network_1.get_layers_number() == 5, LOG);
    assert_true(neural_network_1.get_layer_pointer(0)->get_type() == Layer::Type::Scaling, LOG);
    assert_true(neural_network_1.get_layer_pointer(1)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_1.get_layer_pointer(2)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_1.get_layer_pointer(3)->get_type() == Layer::Type::Unscaling, LOG);
    assert_true(neural_network_1.get_layer_pointer(4)->get_type() == Layer::Type::Bounding, LOG);

    // Classification

    NeuralNetwork neural_network_2(NeuralNetwork::ProjectType::Classification, {1, 4, 2});

    assert_true(neural_network_2.get_layers_number() == 3, LOG);
    assert_true(neural_network_2.get_layer_pointer(0)->get_type() == Layer::Type::Scaling, LOG);
    assert_true(neural_network_2.get_layer_pointer(1)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_2.get_layer_pointer(2)->get_type() == Layer::Type::Probabilistic, LOG);

    // Forecasting

    NeuralNetwork neural_network_3(NeuralNetwork::ProjectType::Forecasting, {1, 4, 2});

    assert_true(neural_network_3.get_layers_number() == 5, LOG);
    assert_true(neural_network_3.get_layer_pointer(0)->get_type() == Layer::Type::Scaling, LOG);
    assert_true(neural_network_3.get_layer_pointer(1)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_3.get_layer_pointer(2)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_3.get_layer_pointer(3)->get_type() == Layer::Type::Unscaling, LOG);

    ///@todo ImageClassification Project Type

    // ImageClassification

    // Inputs variables dimension = (channels, width, height)
//    Tensor<Index, 1> inputs_variables_dimension(3);
//    inputs_variables_dimension.setValues({1,28,28});
//    Index blocks_number = 0;
//    Index outputs_number = 10;
//    Tensor<Index, 1> filters_dimensions(3);
//    filters_dimensions.setValues({1,2,2});
//    NeuralNetwork neural_network_4(inputs_variables_dimension, blocks_number,filters_dimensions, outputs_number);

//    cout << "Layers number: " << neural_network_4.get_layers_number() << endl;
//    assert_true(neural_network_4.get_layers_number() == 6, LOG); // Scaling, 1Bloque (Conv, Pool), Flatten, 1 Perceptron, Probabilistic.
//    assert_true(neural_network_4.get_layer_pointer(0)->get_type() == Layer::Type::Scaling, LOG);
//    assert_true(neural_network_4.get_layer_pointer(1)->get_type() == Layer::Type::Convolutional, LOG);
//    assert_true(neural_network_4.get_layer_pointer(2)->get_type() == Layer::Type::Pooling, LOG);
//    assert_true(neural_network_4.get_layer_pointer(3)->get_type() == Layer::Type::Flatten, LOG);
//    assert_true(neural_network_4.get_layer_pointer(4)->get_type() == Layer::Type::Perceptron, LOG);
//    assert_true(neural_network_4.get_layer_pointer(5)->get_type() == Layer::Type::Probabilistic, LOG);

    //    Layers constructor

    // Default constructor

    NeuralNetwork neural_network_7(layers);

    assert_true(neural_network_7.is_empty(), LOG);
    assert_true(neural_network_7.get_layers_number() == 0, LOG);

    // Perceptron Layer

    PerceptronLayer* perceptron_layer_7 = new PerceptronLayer(1, 1);

    neural_network_7.add_layer(perceptron_layer_7);

    assert_true(!neural_network_7.is_empty(), LOG);
    assert_true(neural_network_7.get_layer_pointer(0)->get_type() == Layer::Type::Perceptron, LOG);

    //    Test 3_2

    layers.resize(7);

    layers.setValues({new ScalingLayer, new PerceptronLayer,
                      new PoolingLayer, new ProbabilisticLayer, new UnscalingLayer, new PerceptronLayer,
                      new BoundingLayer});

    NeuralNetwork neural_network_8(layers);

    assert_true(!neural_network_8.is_empty(), LOG);
    assert_true(neural_network_8.get_layers_number() == 7, LOG);

    assert_true(neural_network_8.get_layer_pointer(0)->get_type() == Layer::Type::Scaling, LOG);
    assert_true(neural_network_8.get_layer_pointer(1)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_8.get_layer_pointer(2)->get_type() == Layer::Type::Pooling, LOG);
    assert_true(neural_network_8.get_layer_pointer(3)->get_type() == Layer::Type::Probabilistic, LOG);
    assert_true(neural_network_8.get_layer_pointer(4)->get_type() == Layer::Type::Unscaling, LOG);
    assert_true(neural_network_8.get_layer_pointer(5)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_8.get_layer_pointer(6)->get_type() == Layer::Type::Bounding, LOG);

    ///@todo Convolutional layer
    /*
    // Convolutional layer constructor

    Tensor<Index, 1> new_inputs_dimensions(1);
    new_inputs_dimensions.setConstant(type(1));

    Index new_blocks_number = 1;

    Tensor<Index, 1> new_filters_dimensions(1);
    new_filters_dimensions.setConstant(type(1));

    Index new_outputs_number = 1;

    ConvolutionalLayer convolutional_layer(1,1); //CC -> cl(inputs_dim, filters_dim)

    NeuralNetwork neural_network_6(new_inputs_dimensions, new_blocks_number, new_filters_dimensions, new_outputs_number);

    assert_true(neural_network_6.is_empty(), LOG);
    assert_true(neural_network_6.get_layers_number() == 0, LOG);
*/
}


void NeuralNetworkTest::test_destructor()
{
    cout << "test_destructor\n";

    NeuralNetwork* neural_network_pointer = new NeuralNetwork;

    delete neural_network_pointer;
}


void NeuralNetworkTest::test_add_layer()
{
    cout << "test_add_layer\n";

    ScalingLayer* scaling_layer_pointer = new ScalingLayer;

    LongShortTermMemoryLayer* long_short_term_memory_layer_pointer = new LongShortTermMemoryLayer;

    RecurrentLayer* recurrent_layer_pointer = new RecurrentLayer;

    PerceptronLayer* perceptron_layer_pointer = new PerceptronLayer;

    ProbabilisticLayer* probabilistic_layer_pointer = new ProbabilisticLayer;

    UnscalingLayer* unscaling_layer_pointer = new UnscalingLayer;

    BoundingLayer* bounding_layer_pointer = new BoundingLayer;

    //   ConvolutionalLayer* convolutional_layer_pointer = new ConvolutionalLayer;

    //   PoolingLayer* pooling_layer_pointer = new PoolingLayer;

    // Scaling Layer

    neural_network.set();

    neural_network.add_layer(scaling_layer_pointer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Type::Scaling, LOG);

    // LSTM Layer

    neural_network.set();

    neural_network.add_layer(long_short_term_memory_layer_pointer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Type::LongShortTermMemory, LOG);

    // Recurrent Layer

    neural_network.set();

    neural_network.add_layer(recurrent_layer_pointer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Type::Recurrent, LOG);

    // Perceptron Layer

    neural_network.set();

    neural_network.add_layer(perceptron_layer_pointer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Type::Perceptron, LOG);

    // Probabilistic Layer

    neural_network.set();

    neural_network.add_layer(probabilistic_layer_pointer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Type::Probabilistic, LOG);

    // Unscaling Layer

    neural_network.set();

    neural_network.add_layer(unscaling_layer_pointer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Type::Unscaling, LOG);

    // Bounding Layer

    neural_network.set();

    neural_network.add_layer(bounding_layer_pointer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Type::Bounding, LOG);

    // Convolutional Layer

    //   neural_network.set();

    //   neural_network.add_layer(convolutional_layer_pointer);
    //   assert_true(neural_network.get_layers_number() == 1, LOG);
    //   assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Type::Convolutional, LOG);

    // Pooling Layer

    //   neural_network.set();

    //   neural_network.add_layer(pooling_layer_pointer);
    //   assert_true(neural_network.get_layers_number() == 1, LOG);
    //   assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Type::Pooling, LOG);

}


void NeuralNetworkTest::check_layer_type()
{
    cout << "check_layer_type\n";

    Tensor<Layer*, 1> layers_pointers;

    // Test

    layers_pointers.resize(2);
    layers_pointers.setValues({new PerceptronLayer, new PerceptronLayer});

    neural_network.set_layers_pointers(layers_pointers);

    assert_true(!neural_network.check_layer_type(Layer::Type::LongShortTermMemory), LOG);
    assert_true(!neural_network.check_layer_type(Layer::Type::Recurrent), LOG);

    assert_true(neural_network.check_layer_type(Layer::Type::Scaling), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Convolutional), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Perceptron), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Pooling), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Probabilistic), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Unscaling), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Bounding), LOG);

    // Test

    layers_pointers.resize(1);
    layers_pointers.setValues({new PerceptronLayer});

    neural_network.set_layers_pointers(layers_pointers);

    assert_true(!neural_network.check_layer_type(Layer::Type::LongShortTermMemory), LOG);
    assert_true(!neural_network.check_layer_type(Layer::Type::Recurrent), LOG);

    assert_true(neural_network.check_layer_type(Layer::Type::Convolutional),LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Perceptron),LOG);

    // Test

    layers_pointers.resize(1);
    layers_pointers.setValues({new ScalingLayer});

    neural_network.set_layers_pointers(layers_pointers);

    assert_true(neural_network.check_layer_type(Layer::Type::LongShortTermMemory), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Recurrent), LOG);

    assert_true(neural_network.check_layer_type(Layer::Type::Scaling), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Convolutional), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Perceptron), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Pooling), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Probabilistic), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Unscaling), LOG);
    assert_true(neural_network.check_layer_type(Layer::Type::Bounding), LOG);

    // Test

    layers_pointers.resize(1);
    layers_pointers.setValues({new RecurrentLayer});

    neural_network.set_layers_pointers(layers_pointers);

    assert_true(!neural_network.check_layer_type(Layer::Type::Recurrent),LOG);
    assert_true(!neural_network.check_layer_type(Layer::Type::LongShortTermMemory),LOG);
}


void NeuralNetworkTest::test_has_methods()
{
    cout << "test_has_methods\n";

    Tensor<Layer*, 1> layers_pointers;

    // Test

    neural_network.set_layers_pointers(layers_pointers);

    assert_true(neural_network.is_empty(), LOG);

    assert_true(!neural_network.has_scaling_layer(), LOG);
    assert_true(!neural_network.has_long_short_term_memory_layer(), LOG);
    assert_true(!neural_network.has_recurrent_layer(), LOG);
    assert_true(!neural_network.has_unscaling_layer(), LOG);
    assert_true(!neural_network.has_bounding_layer(), LOG);
    assert_true(!neural_network.has_probabilistic_layer(), LOG);

    // Test

    layers_pointers.resize(6);

    layers_pointers.setValues({new ScalingLayer, new LongShortTermMemoryLayer,
                               new RecurrentLayer, new UnscalingLayer, new BoundingLayer, new ProbabilisticLayer});

    neural_network.set_layers_pointers(layers_pointers);

    assert_true(neural_network.has_scaling_layer(), LOG);
    assert_true(neural_network.has_long_short_term_memory_layer(), LOG);
    assert_true(neural_network.has_recurrent_layer(), LOG);
    assert_true(neural_network.has_unscaling_layer(), LOG);
    assert_true(neural_network.has_bounding_layer(), LOG);
    assert_true(neural_network.has_probabilistic_layer(), LOG);

    // Test

    layers_pointers.resize(4);

    layers_pointers.setValues({new ScalingLayer, new LongShortTermMemoryLayer,
                               new RecurrentLayer, new UnscalingLayer});

    neural_network.set_layers_pointers(layers_pointers);

    assert_true(neural_network.has_scaling_layer(), LOG);
    assert_true(neural_network.has_long_short_term_memory_layer(), LOG);
    assert_true(neural_network.has_recurrent_layer(), LOG);
    assert_true(neural_network.has_unscaling_layer(), LOG);
    assert_true(!neural_network.has_bounding_layer(), LOG);
    assert_true(!neural_network.has_probabilistic_layer(), LOG);
}


void NeuralNetworkTest::test_set()
{
    cout << "test_set\n";

    // Empty NN

    neural_network.set();

    assert_true(neural_network.get_inputs_names().size() == 0, LOG);
    assert_true(neural_network.get_outputs_names().size() == 0, LOG);
    assert_true(neural_network.get_layers_pointers().size() == 0, LOG);

    // Approximation Project

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {1,0,1});
    assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
    assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
    assert_true(neural_network.get_layers_pointers().size() == 5, LOG);

    // Classification Project

    neural_network.set(NeuralNetwork::ProjectType::Classification, {1,0,1});
    assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
    assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
    assert_true(neural_network.get_layers_pointers().size() == 3, LOG);

    // Forecasting Project

    neural_network.set(NeuralNetwork::ProjectType::Forecasting, {1,0,1});
    assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
    assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
    assert_true(neural_network.get_layers_pointers().size() == 5, LOG);

    ///@todo Image treatment Projects

    //    neural_network.set(NeuralNetwork::ImageApproximation, {1,0,1});
    //    assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
    //    assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
    //    assert_true(neural_network.get_layers_pointers().size() == 1, LOG);

    //    neural_network.set(NeuralNetwork::ImageClassification, {1,0,1});
    //    assert_true(neural_network.get_inputs_names().size() == 1, LOG);  //CC -> architecture(0)
    //    assert_true(neural_network.get_outputs_names().size() == 1, LOG);  //CC -> architecture(architecture.size()-1)
    //    assert_true(neural_network.get_layers_pointers().size() == 1, LOG);

    //    Test / Convolutional layer set

    //            Tensor<Index, 1> new_inputs_dimensions(1);
    //    new_inputs_dimensions.setConstant(type(1));

    //    Index new_blocks_number = 1;

    //    Tensor<Index, 1> new_filters_dimensions(1);
    //    new_filters_dimensions.setConstant(type(1));

    //    Index new_outputs_number = 1;

    //    ConvolutionalLayer convolutional_layer(1,1); //CC -> cl(inputs_dim, filters_dim)

    //    neural_network.set(new_inputs_dimensions, new_blocks_number, new_filters_dimensions, new_outputs_number);

    //    assert_true(neural_network.is_empty(), LOG);
    //    assert_true(neural_network.get_layers_number() == 0, LOG);

}


void NeuralNetworkTest::test_set_names()
{
    cout << "test_set_names\n";

    Tensor<string, 1> inputs_names;
    Tensor<string, 1> outputs_names;

    // Test

    neural_network.set_inputs_names(inputs_names);
    neural_network.set_outputs_names(outputs_names);
    assert_true(neural_network.get_inputs_names().size() == 0, LOG);
    assert_true(neural_network.get_outputs_names().size() == 0, LOG);

    // Test

    inputs_names.resize(2);
    inputs_names.setValues({"in_1","in_2"});
    outputs_names.resize(2);
    outputs_names.setValues({"out_1","out_2"});

    neural_network.set_inputs_names(inputs_names);
    neural_network.set_outputs_names(outputs_names);
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

    inputs_number = 3;

    neural_network.set_inputs_number(inputs_number);
    assert_true(neural_network.get_inputs_number() == 3, LOG);
    assert_true(neural_network.get_layer_pointer(0)->get_inputs_number() == 3, LOG); //CC -> Scaling layer nmb assert

    // Test

    inputs.resize(2);
    inputs.setValues({true,false});

    neural_network.set_inputs_number(inputs);
    assert_true(neural_network.get_inputs_number() == 1, LOG);

    // Test

    inputs.resize(2);
    inputs.setValues({true,true});

    neural_network.set_inputs_number(inputs);
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

    Tensor<Layer*, 1> layers_pointers;

    // Test // Device

    neural_network.set(NeuralNetwork::ProjectType::Classification, {1,0,1});

    assert_true(neural_network.get_layers_number() == 3, LOG);

    // Test // Layers

    layers_pointers.resize(3);
    layers_pointers.setValues({new ScalingLayer(1), new PerceptronLayer(1, 1), new UnscalingLayer(1)});

    neural_network.set_layers_pointers(layers_pointers);

    assert_true(!neural_network.is_empty(), LOG);
    assert_true(neural_network.get_layers_number() == 3, LOG);
    assert_true(neural_network.get_layer_pointer(0)->get_type() == Layer::Type::Scaling, LOG);
    assert_true(neural_network.get_layer_pointer(1)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network.get_layer_pointer(2)->get_type() == Layer::Type::Unscaling, LOG);
}


void NeuralNetworkTest::test_set_parameters()
{
    cout << "test_set_parameters\n";

    Index parameters_number;
    Tensor<type, 1> parameters;

    // Test

    neural_network.set();

    neural_network.set_parameters(parameters);
    parameters = neural_network.get_parameters();

    assert_true(parameters.size() == 0, LOG);

    // Test

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {2,2});

    parameters_number = neural_network.get_parameters_number();
    parameters.resize(parameters_number);

    neural_network.set_parameters(parameters);
    parameters = neural_network.get_parameters();

    assert_true(parameters.size() == 6, LOG);
    assert_true(parameters.size() == parameters_number, LOG);

    // Test

    parameters.setValues({type(1),type(2),type(3),type(4),type(5),type(6)});

    neural_network.set_parameters(parameters);
    parameters = neural_network.get_parameters();

    assert_true(abs(parameters(0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(parameters(1) - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(parameters(2) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(parameters(3) - type(4)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(parameters(4) - type(5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(parameters(5) - type(6)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void NeuralNetworkTest::test_set_parameters_constant()
{
    cout << "test_set_parameters_constant\n";

    Tensor<type, 1> parameters;

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {1,2,1});

    // Test

    neural_network.set_parameters_constant(type(1));
    parameters = neural_network.get_parameters();

    assert_true(parameters.size() == 7, LOG);
    assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
    assert_true(abs(parameters(1) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(parameters(5) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    neural_network.set_parameters_constant(type(3));
    parameters = neural_network.get_parameters();

    assert_true(parameters.size() == 7, LOG);
    assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
    assert_true(abs(parameters(1) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(parameters(5) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void NeuralNetworkTest::test_set_parameters_random()
{
    cout << "test_set_parameters_random\n";

    Tensor<type, 1> parameters;

    // Test

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {1,2,1});

    neural_network.set_parameters_random();
    parameters = neural_network.get_parameters();

    assert_true(parameters.size() == 7, LOG);
    assert_true(neural_network.get_parameters_number() == parameters.size(), LOG);
}


void NeuralNetworkTest::test_calculate_parameters_norm()
{
    cout << "test_calculate_parameters_norm\n";

    type parameters_norm = type(0);

    // Test

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {});

    parameters_norm = neural_network.calculate_parameters_norm();

    assert_true(abs(parameters_norm) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {1,1,1,1});

    neural_network.set_parameters_constant(type(1));

    parameters_norm = neural_network.calculate_parameters_norm();

    assert_true(abs(parameters_norm - static_cast<type>(sqrt(6))) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {1,1,1,1,1});

    neural_network.set_parameters_constant(type(1));

    parameters_norm = neural_network.calculate_parameters_norm();

    assert_true(abs(parameters_norm - static_cast<type>(sqrt(8))) < type(NUMERIC_LIMITS_MIN), LOG);
}


void NeuralNetworkTest::test_perturbate_parameters()
{
    cout << "test_perturbate_parameters\n";

    Index inputs_number;
    Index neurons_number;
    Index outputs_number;

    Index parameters_number;
    Tensor<type, 1> parameters;

    // Test

    inputs_number = 1;
    neurons_number = 1;
    outputs_number = 1;

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});

    parameters_number = neural_network.get_parameters_number();
    parameters.resize(parameters_number);
    parameters.setConstant(type(1));

    neural_network.set_parameters(parameters);
    parameters = neural_network.get_parameters();

    neural_network.perturbate_parameters(type(0.5));
    parameters = neural_network.get_parameters();

    assert_true(parameters.size() == 4, LOG);
    assert_true(parameters.size() == parameters_number, LOG);
    assert_true(abs(parameters(0) - static_cast<type>(1.5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(parameters(3) - static_cast<type>(1.5)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    parameters.setConstant(type(0));

    neural_network.set_parameters(parameters);

    neural_network.perturbate_parameters(type(0.5));
    parameters = neural_network.get_parameters();

    assert_true(!is_zero(parameters), LOG);
}


void NeuralNetworkTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";

    Index inputs_number;
    Index neurons_number;
    Index outputs_number;
    Index batch_size;

    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;

    Tensor<Index, 1> inputs_dimensions;

    Index parameters_number;

    Tensor<type, 1> parameters;

    // Test

    batch_size = 1;
    inputs_number = 3;
    neurons_number = 2;
    outputs_number = 3;

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
    neural_network.set_parameters_constant(type(0));

    inputs.resize(batch_size, inputs_number);
    inputs.setConstant(type(1));
    inputs_dimensions = get_dimensions(inputs);

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.dimension(0) == batch_size, LOG);
    assert_true(outputs.dimension(1) == outputs_number, LOG);
    assert_true(abs(outputs(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,2)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    batch_size = 1;
    inputs_number = 2;
    neurons_number = 1;
    outputs_number = 5;

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});

    neural_network.set_parameters_constant(type(0));

    inputs.resize(batch_size, inputs_number);
    inputs.setConstant(type(0));
    inputs_dimensions = get_dimensions(inputs);

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == batch_size * outputs_number, LOG);
    assert_true(abs(outputs(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,4)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {1, 2});

    neural_network.set_parameters_constant(type(1));

    inputs.resize(1, 1);
    inputs.setConstant(2);

    inputs_dimensions = get_dimensions(inputs);

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == 2, LOG);
    assert_true(abs(outputs(0,0) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,1) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {4, 3, 3});

    inputs.resize(1, 4);
    inputs.setZero();

    neural_network.set_parameters_constant(type(1));

    inputs_dimensions = get_dimensions(inputs);

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == 3, LOG);

    assert_true(abs(outputs(0,0) - 3.2847) < static_cast<type>(1e-3), LOG);
    assert_true(abs(outputs(0,1) - 3.2847) < static_cast<type>(1e-3), LOG);
    assert_true(abs(outputs(0,2) - 3.2847) < static_cast<type>(1e-3), LOG);

    // Test

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {1, 2});

    inputs_number = neural_network.get_inputs_number();
    parameters_number = neural_network.get_parameters_number();
    outputs_number = neural_network.get_outputs_number();

    inputs.resize(1, inputs_number);
    inputs.setZero();

    parameters.resize(parameters_number);
    parameters.setConstant(type(0));

    neural_network.set_parameters(parameters);

    inputs_dimensions = get_dimensions(inputs);

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == outputs_number, LOG);
    assert_true(abs(outputs(0,0) - 0) < static_cast<type>(1e-3), LOG);
    assert_true(abs(outputs(0,1) - 0) < static_cast<type>(1e-3), LOG);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    outputs_number = 1;

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});

    neural_network.set_parameters_constant(type(0));

    inputs.resize(1, 1);
    inputs.setZero();

    inputs_dimensions = get_dimensions(inputs);

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == 1, LOG);
    assert_true(abs(outputs(0,0) - 0) < static_cast<type>(1e-3), LOG);

    // Test

    neural_network.set(NeuralNetwork::ProjectType::Classification, {1,1});

    neural_network.set_parameters_constant(type(0));

    inputs.resize(1, 1);
    inputs.setZero();

    inputs_dimensions = get_dimensions(inputs);

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == 1, LOG);
    assert_true(abs(outputs(0,0) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);

    // Test 7

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {1,3,3,3,1});

    batch_size = 2;
    inputs_number = neural_network.get_inputs_number();
    outputs_number = neural_network.get_outputs_number();

    inputs.resize(batch_size, inputs_number);
    inputs.setConstant(type(0));

    parameters_number = neural_network.get_parameters_number();
    parameters.resize(parameters_number);
    parameters.setConstant(type(0));

    neural_network.set_parameters(parameters);

    inputs_dimensions = get_dimensions(inputs);

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.dimension(1) == outputs_number, LOG);
    assert_true(abs(outputs(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(1,0)) < type(NUMERIC_LIMITS_MIN), LOG);

}


void NeuralNetworkTest::test_calculate_directional_inputs()
{
    cout << "test_calculate_directional_inputs\n";

    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;
    Tensor<type, 2> trainable_outputs;

    Tensor<type, 1> parameters;

    Tensor<type, 1> point;

    Tensor<type, 2> directional_inputs;

    // Test

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {3, 4, 2});
    neural_network.set_parameters_constant(type(0));

    inputs.resize(2,3);
    inputs.setValues(
                {{type(-5),type(-1),-type(3)},
                 {type(7),type(3),type(1)}});

    point.resize(3);
    point.setValues({type(0),type(0),type(0)});

    directional_inputs = neural_network.calculate_directional_inputs(0, point, type(0), type(0), 0);

    assert_true(directional_inputs.rank() == 2, LOG);
    assert_true(directional_inputs.dimension(0) == 0, LOG);

    // Test

    point.setValues({type(1), type(2), type(3)});

    directional_inputs = neural_network.calculate_directional_inputs(2, point, type(-1), type(1), 3);

    assert_true(directional_inputs.rank() == 2, LOG);
    assert_true(directional_inputs.dimension(0) == 3, LOG);
    assert_true(directional_inputs.dimension(1) == 3, LOG);
    assert_true(abs(directional_inputs(0,2) + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(1,2) - type(0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(2,2) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    point.setValues({type(1), type(2), type(3)});

    directional_inputs = neural_network.calculate_directional_inputs(0, point, type(-4), type(0), 5);

    assert_true(directional_inputs.rank() == 2, LOG);
    assert_true(directional_inputs.dimension(0) == 5, LOG);
    assert_true(directional_inputs.dimension(1) == 3, LOG);
    assert_true(abs(directional_inputs(0,0) + type(4)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(1,0) + type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(2,0) + type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(3,0) + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(directional_inputs(4,0) + type(0)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void NeuralNetworkTest::test_save()
{
    cout << "test_save\n";

    Index inputs_number;
    Index neurons_number;
    Index outputs_number;

    string file_name = "../data/neural_network.xml";

    // Empty neural network

    neural_network.set();
    neural_network.save(file_name);

    // Only network architecture

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {2, 4, 3});
    neural_network.save(file_name);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    outputs_number = 1;

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, outputs_number});
    neural_network.save(file_name);
}


void NeuralNetworkTest::test_load()
{
    cout << "test_load\n";

    const string file_name = "../data/neural_network.xml";

    // Test

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {2, 4, 3});
    neural_network.save(file_name);

    neural_network.set();
    neural_network.load(file_name);
}


void NeuralNetworkTest::test_forward_propagate()
{
    cout << "test_forward_propagate\n";

    // Test

    inputs_number = 2;
    outputs_number = 1;
    batch_size = 5;

    data.resize(batch_size, inputs_number + outputs_number);

    data.setValues({{1,1,1},
                    {2,2,2},
                    {3,3,3},
                    {0,0,0},
                    {0,0,0}});

    data_set.set(data);

    data_set.set_training();

    training_samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    batch.set(batch_size, &data_set);

    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    neural_network.set(NeuralNetwork::ProjectType::Approximation, {inputs_number, outputs_number});

    PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(neural_network.get_layer_pointer(1));

    const Index neurons_number = perceptron_layer->get_neurons_number();

    perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Logistic);

    Tensor<type, 2> biases_perceptron(neurons_number, 1);

    biases_perceptron.setConstant(type(1));

    perceptron_layer->set_biases(biases_perceptron);

    Tensor<type, 2> synaptic_weights_perceptron(inputs_number, neurons_number);

    synaptic_weights_perceptron.setConstant(type(1));

    perceptron_layer->set_synaptic_weights(synaptic_weights_perceptron);

    NeuralNetworkForwardPropagation forward_propagation(data_set.get_training_samples_number(), &neural_network);

    neural_network.forward_propagate(batch, forward_propagation);

    PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
            = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.layers[0]);

    Tensor<type, 2> perceptron_combinations = perceptron_layer_forward_propagation->combinations;

    TensorMap<Tensor<type, 2>> perceptron_activations(perceptron_layer_forward_propagation->outputs_data, perceptron_layer_forward_propagation->outputs_dimensions(0), perceptron_layer_forward_propagation->outputs_dimensions(1));

//    Tensor<type, 2> perceptron_activations = perceptron_layer_forward_propagation->outputs;

    assert_true(perceptron_combinations.dimension(0) == 5, LOG);
    assert_true(abs(perceptron_combinations(0,0) - 3) < static_cast<type>(1e-3), LOG);
    assert_true(abs(perceptron_combinations(1,0) - 5) < static_cast<type>(1e-3), LOG);
    assert_true(abs(perceptron_combinations(2,0) - 7) < static_cast<type>(1e-3), LOG);
    assert_true(abs(perceptron_combinations(3,0) - 1) < static_cast<type>(1e-3), LOG);
    assert_true(abs(perceptron_combinations(4,0) - 1) < static_cast<type>(1e-3), LOG);

    assert_true(perceptron_activations.dimension(0) == 5, LOG);
    assert_true(abs(perceptron_activations(0,0) - static_cast<type>(0.952)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(perceptron_activations(1,0) - static_cast<type>(0.993)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(perceptron_activations(2,0) - static_cast<type>(0.999)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(perceptron_activations(3,0) - static_cast<type>(0.731)) < static_cast<type>(1e-3), LOG);
    assert_true(abs(perceptron_activations(4,0) - static_cast<type>(0.731)) < static_cast<type>(1e-3), LOG);

    // Test

    inputs_number = 4;
    outputs_number = 2;

    data.resize(3, inputs_number + outputs_number);
    data.setValues({{-1,1,-1,1,1,0},{-2,2,3,1,1,0},{-3,3,5,1,1,0}});
    data_set.set(data);
    data_set.set_target();
    data_set.set_training();

    Tensor<Index, 1> input_columns_indices(4);
    input_columns_indices.setValues({0,1,2,3});

    Tensor<bool, 1> input_columns_use(4);
    input_columns_use.setConstant(true);

    data_set.set_input_columns(input_columns_indices, input_columns_use);

    training_samples_indices = data_set.get_training_samples_indices();
    input_variables_indices = data_set.get_input_variables_indices();
    target_variables_indices = data_set.get_target_variables_indices();

    batch.set(3, &data_set);
    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    neural_network.set();

    PerceptronLayer* perceptron_layer_3 = new PerceptronLayer(inputs_number, outputs_number);
    perceptron_layer_3->set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
    const Index neurons_number_3_0 = perceptron_layer_3->get_neurons_number();

    ProbabilisticLayer* probabilistic_layer_3 = new ProbabilisticLayer(outputs_number, outputs_number);
    probabilistic_layer_3->set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);
    const Index neurons_number_3_1 = probabilistic_layer_3->get_neurons_number();

    Tensor<type, 2> biases_pl(neurons_number, outputs_number);
    biases_pl.setConstant(5);
    perceptron_layer_3->set_biases(biases_pl);

    Tensor<type, 2> synaptic_weights_pl(inputs_number, neurons_number_3_0);
    synaptic_weights_pl.setConstant(type(-1));
    perceptron_layer_3->set_synaptic_weights(synaptic_weights_pl);

    Tensor<type, 2> biases_pbl(neurons_number, outputs_number);
    biases_pbl.setConstant(3);
    probabilistic_layer_3->set_biases(biases_pbl);

    Tensor<type, 2> synaptic_pbl(neurons_number_3_0, neurons_number_3_1);
    synaptic_pbl.setConstant(type(1));
    probabilistic_layer_3->set_synaptic_weights(synaptic_pbl);

    Tensor<Layer*, 1> layers_pointers(2);
    layers_pointers.setValues({perceptron_layer_3, probabilistic_layer_3});
    neural_network.set_layers_pointers(layers_pointers);

    NeuralNetworkForwardPropagation forward_propagation_3(data_set.get_training_samples_number(), &neural_network);

    neural_network.forward_propagate(batch, forward_propagation_3);

    PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation_3
            = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation_3.layers[0]);

    Tensor<type, 2> perceptron_combinations_3_0 = perceptron_layer_forward_propagation_3->combinations;

    TensorMap<Tensor<type, 2>> perceptron_activations_3_0(perceptron_layer_forward_propagation_3->outputs_data, perceptron_layer_forward_propagation_3->outputs_dimensions(0), perceptron_layer_forward_propagation_3->outputs_dimensions(1));

    ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation_3
            = static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation_3.layers[1]);

    Tensor<type, 2> probabilistic_combinations_3_1 = probabilistic_layer_forward_propagation_3->combinations;

    TensorMap<Tensor<type, 2>> probabilistic_activations_3_1(probabilistic_layer_forward_propagation_3->outputs_data, probabilistic_layer_forward_propagation_3->outputs_dimensions(0), probabilistic_layer_forward_propagation_3->outputs_dimensions(1));

    assert_true(perceptron_combinations_3_0.dimension(0) == 3, LOG);

    assert_true(abs(perceptron_combinations_3_0(0,0) - 5) < static_cast<type>(1e-3)
                && abs(perceptron_combinations_3_0(1,0) - 1) < static_cast<type>(1e-3)
                && abs(perceptron_combinations_3_0(2,0) + 1) < static_cast<type>(1e-3), LOG);

    assert_true(perceptron_activations_3_0.dimension(0) == 3, LOG);
    assert_true(abs(perceptron_activations_3_0(0,0) - static_cast<type>(0.993)) < static_cast<type>(1e-3)
                && abs(perceptron_activations_3_0(1,0) - static_cast<type>(0.731)) < static_cast<type>(1e-3)
                && abs(perceptron_activations_3_0(2,0) - static_cast<type>(0.268)) < static_cast<type>(1e-3), LOG);

    assert_true(probabilistic_combinations_3_1.dimension(0) == 3, LOG);
    assert_true(abs(probabilistic_combinations_3_1(0,0) - static_cast<type>(4.98661)) < static_cast<type>(1e-3)
                && abs(probabilistic_combinations_3_1(1,0) - static_cast<type>(4.46212)) < static_cast<type>(1e-3)
                && abs(probabilistic_combinations_3_1(2,0) - static_cast<type>(3.53788)) < static_cast<type>(1e-3), LOG);

    assert_true(probabilistic_activations_3_1.dimension(0) == 3, LOG);
    assert_true(abs(probabilistic_activations_3_1(0,0) - 0.5) < static_cast<type>(1e-3)
                && abs(probabilistic_activations_3_1(1,0) - 0.5) < static_cast<type>(1e-3)
                && abs(probabilistic_activations_3_1(2,0) - 0.5) < static_cast<type>(1e-3), LOG);
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

    // Set methods

    test_set();

    test_set_names();
    test_set_inputs_number();

    test_set_pointers();

    test_set_parameters();

    // Parameters initialization methods

    test_set_parameters_constant();
    test_set_parameters_random();

    // Parameters norm / descriptives / histogram

    test_calculate_parameters_norm();
    test_perturbate_parameters();

    // Output

    test_calculate_outputs();

    test_calculate_directional_inputs();

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
