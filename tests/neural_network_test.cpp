#include "pch.h"

#include "../opennn/neural_network.h"


TEST(NeuralNetworkTest, DefaultConstructor)
{
    EXPECT_EQ(1, 1);
}


TEST(NeuralNetworkTest, GeneralConstructor)
{
    EXPECT_EQ(1, 1);
}

/*
namespace opennn
{
void NeuralNetworkTest::test_constructor()
{
    cout << "test_constructor\n";

    Tensor<Layer*, 1> layers;    

    NeuralNetwork neural_network_0;

    assert_true(neural_network_0.is_empty(), LOG);
    assert_true(neural_network_0.get_layers_number() == 0, LOG);

    // Model type constructors

    //  Approximation

    NeuralNetwork neural_network_1(NeuralNetwork::ModelType::Approximation, {1}, {4}, {2});

    assert_true(neural_network_1.get_layers_number() == 5, LOG);
    assert_true(neural_network_1.get_layer(0)->get_type() == Layer::Type::Scaling2D, LOG);
    assert_true(neural_network_1.get_layer(1)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_1.get_layer(2)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_1.get_layer(3)->get_type() == Layer::Type::Unscaling, LOG);
    assert_true(neural_network_1.get_layer(4)->get_type() == Layer::Type::Bounding, LOG);

    // Classification

    NeuralNetwork neural_network_2(NeuralNetwork::ModelType::Classification, {1}, {4}, {2});

    assert_true(neural_network_2.get_layers_number() == 3, LOG);
    assert_true(neural_network_2.get_layer(0)->get_type() == Layer::Type::Scaling2D, LOG);
    assert_true(neural_network_2.get_layer(1)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_2.get_layer(2)->get_type() == Layer::Type::Probabilistic, LOG);

    // Forecasting

    NeuralNetwork neural_network_3(NeuralNetwork::ModelType::Forecasting, {1}, {4}, {2});

    assert_true(neural_network_3.get_layers_number() == 5, LOG);
    assert_true(neural_network_3.get_layer(0)->get_type() == Layer::Type::Scaling2D, LOG);
    assert_true(neural_network_3.get_layer(1)->get_type() == Layer::Type::Recurrent, LOG);
    assert_true(neural_network_3.get_layer(2)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_3.get_layer(3)->get_type() == Layer::Type::Unscaling, LOG);

    // Image classification

    Inputs variables dimension = (channels, width, height)
    Tensor<Index, 1> inputs_variables_dimension(3);
    inputs_variables_dimension.setValues({1,28,28});
    Index blocks_number = 0;
    Index outputs_number = 10;
    Tensor<Index, 1> filters_dimensions(3);
    filters_dimensions.setValues({1,2,2});
    NeuralNetwork neural_network_4(inputs_variables_dimension, blocks_number,filters_dimensions, outputs_number);

    cout << "Layers number: " << neural_network_4.get_layers_number() << endl;
    assert_true(neural_network_4.get_layers_number() == 6, LOG); // Scaling, 1Bloque (Conv, Pool), Flatten, 1 Perceptron, Probabilistic.
    assert_true(neural_network_4.get_layer(0)->get_type() == Layer::Type::Scaling2D, LOG);
    assert_true(neural_network_4.get_layer(1)->get_type() == Layer::Type::Convolutional, LOG);
    assert_true(neural_network_4.get_layer(2)->get_type() == Layer::Type::Pooling, LOG);
    assert_true(neural_network_4.get_layer(3)->get_type() == Layer::Type::Flatten, LOG);
    assert_true(neural_network_4.get_layer(4)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_4.get_layer(5)->get_type() == Layer::Type::Probabilistic, LOG);

    // Layers constructor

    NeuralNetwork neural_network_7(layers);

    assert_true(neural_network_7.is_empty(), LOG);
    assert_true(neural_network_7.get_layers_number() == 0, LOG);

    // Perceptron Layer

    unique_ptr<PerceptronLayer> perceptron_layer_7 = make_unique<PerceptronLayer>({1}, {1});

    neural_network_7.add_layer(perceptron_layer_7);

    assert_true(!neural_network_7.is_empty(), LOG);
    assert_true(neural_network_7.get_layer(0)->get_type() == Layer::Type::Perceptron, LOG);

    // Test 3_2

    layers.resize(7);

    layers.setValues({new ScalingLayer2D, new PerceptronLayer,
                      new PoolingLayer, new ProbabilisticLayer, new UnscalingLayer, new PerceptronLayer,
                      new BoundingLayer});

    NeuralNetwork neural_network_8(layers);

    assert_true(!neural_network_8.is_empty(), LOG);
    assert_true(neural_network_8.get_layers_number() == 7, LOG);

    assert_true(neural_network_8.get_layer(0)->get_type() == Layer::Type::Scaling2D, LOG);
    assert_true(neural_network_8.get_layer(1)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_8.get_layer(2)->get_type() == Layer::Type::Pooling, LOG);
    assert_true(neural_network_8.get_layer(3)->get_type() == Layer::Type::Probabilistic, LOG);
    assert_true(neural_network_8.get_layer(4)->get_type() == Layer::Type::Unscaling, LOG);
    assert_true(neural_network_8.get_layer(5)->get_type() == Layer::Type::Perceptron, LOG);
    assert_true(neural_network_8.get_layer(6)->get_type() == Layer::Type::Bounding, LOG);

    // Convolutional layer constructor

    Tensor<Index, 1> new_input_dimensions(1);
    new_input_dimensions.setConstant(type(1));

    Index new_blocks_number = 1;

    Tensor<Index, 1> new_filters_dimensions(1);
    new_filters_dimensions.setConstant(type(1));

    Index new_outputs_number = 1;

    ConvolutionalLayer convolutional_layer(1,1); //CC -> cl(inputs_dim, filters_dim)

    NeuralNetwork neural_network_6(new_input_dimensions, new_blocks_number, new_filters_dimensions, new_outputs_number);

    assert_true(neural_network_6.is_empty(), LOG);
    assert_true(neural_network_6.get_layers_number() == 0, LOG);

}


void NeuralNetworkTest::test_add_layer()
{
    cout << "test_add_layer\n";
/*
//    ScalingLayer2D* scaling_layer_2d = new ScalingLayer2D;

//    LongShortTermMemoryLayer* long_short_term_memory_layer = new LongShortTermMemoryLayer;

//    RecurrentLayer* recurrent_layer = new RecurrentLayer;

//    PerceptronLayer* perceptron_layer = new PerceptronLayer;

//    ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer;

//    UnscalingLayer* unscaling_layer = new UnscalingLayer;

//    BoundingLayer* bounding_layer = new BoundingLayer;

    ScalingLayer2D scaling_layer = ScalingLayer2D();

    LongShortTermMemoryLayer lstm_layer = LongShortTermMemoryLayer();

    RecurrentLayer recurrent_layer = RecurrentLayer();

    PerceptronLayer perceptron_layer = PerceptronLayer();

    ProbabilisticLayer probabilistic_layer = ProbabilisticLayer();

    UnscalingLayer unscaling_layer = UnscalingLayer();

    BoundingLayer bounding_layer = BoundingLayer();

    //   ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer;

    //   PoolingLayer* pooling_layer = new PoolingLayer;

    // Scaling Layer

    neural_network.set();

    neural_network.add_layer(&scaling_layer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer(0)->get_type() == Layer::Type::Scaling2D, LOG);

    // LSTM Layer

    neural_network.set();

    neural_network.add_layer(&lstm_layer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer(0)->get_type() == Layer::Type::LongShortTermMemory, LOG);

    // Recurrent Layer

    neural_network.set();

    neural_network.add_layer(&recurrent_layer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer(0)->get_type() == Layer::Type::Recurrent, LOG);

    // Perceptron Layer

    neural_network.set();

    neural_network.add_layer(&perceptron_layer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer(0)->get_type() == Layer::Type::Perceptron, LOG);

    // Probabilistic Layer

    neural_network.set();

    neural_network.add_layer(&probabilistic_layer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer(0)->get_type() == Layer::Type::Probabilistic, LOG);

    // Unscaling Layer

    neural_network.set();

    neural_network.add_layer(&unscaling_layer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer(0)->get_type() == Layer::Type::Unscaling, LOG);

    // Bounding Layer

    neural_network.set();

    neural_network.add_layer(&bounding_layer);
    assert_true(neural_network.get_layers_number() == 1, LOG);
    assert_true(neural_network.get_layer(0)->get_type() == Layer::Type::Bounding, LOG);

    // Convolutional Layer

    //   neural_network.set();

    //   neural_network.add_layer(convolutional_layer);
    //   assert_true(neural_network.get_layers_number() == 1, LOG);
    //   assert_true(neural_network.get_layer(0)->get_type() == Layer::Type::Convolutional, LOG);

    // Pooling Layer

    //   neural_network.set();

    //   neural_network.add_layer(pooling_layer);
    //   assert_true(neural_network.get_layers_number() == 1, LOG);
    //   assert_true(neural_network.get_layer(0)->get_type() == Layer::Type::Pooling, LOG);

}


void NeuralNetworkTest::test_calculate_outputs()
{
    cout << "test_calculate_outputs\n";

    Index inputs_number;
    Index neurons_number;
    Index outputs_number;
    Index batch_samples_number;

    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;

    Index parameters_number;

    Tensor<type, 1> parameters;

    // Test two layers perceptron with all zeros

    batch_samples_number = 1;
    inputs_number = 3;
    neurons_number = 2;
    outputs_number = 3;

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});
    neural_network.set_parameters_constant(type(0));

    inputs.resize(batch_samples_number, inputs_number);
    inputs.setConstant(type(0));
    
    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.dimension(0) == batch_samples_number, LOG);
    assert_true(outputs.dimension(1) == outputs_number, LOG);
    assert_true(abs(outputs(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,2)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    batch_samples_number = 3;
    inputs_number = 2;
    neurons_number = 4;
    outputs_number = 5;

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});

    neural_network.set_parameters_constant(type(0));

    inputs.resize(batch_samples_number, inputs_number);
    inputs.setConstant(type(0));  

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == batch_samples_number * outputs_number, LOG);
    assert_true(abs(outputs(0,0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,4)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    neural_network.set(NeuralNetwork::ModelType::Approximation, {1, 2});

    neural_network.set_parameters_constant(type(1));

    inputs.resize(1, 1);
    inputs.setConstant(2);    

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == 2, LOG);
    assert_true(abs(outputs(0,0) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(outputs(0,1) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    neural_network.set(NeuralNetwork::ModelType::Approximation, {4, 3, 3});

    inputs.resize(1, 4);
    inputs.setZero();

    neural_network.set_parameters_constant(type(1));    

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == 3, LOG);

    assert_true(abs(outputs(0,0) - 3.2847) < type(1e-3), LOG);
    assert_true(abs(outputs(0,1) - 3.2847) < type(1e-3), LOG);
    assert_true(abs(outputs(0,2) - 3.2847) < type(1e-3), LOG);

    // Test

    neural_network.set(NeuralNetwork::ModelType::Approximation, {1, 2});

    inputs_number = neural_network.get_inputs_number();
    parameters_number = neural_network.get_parameters_number();
    outputs_number = neural_network.get_outputs_number();

    inputs.resize(1, inputs_number);
    inputs.setZero();

    parameters.resize(parameters_number);
    parameters.setConstant(type(0));

    neural_network.set_parameters(parameters);

    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == outputs_number, LOG);
    assert_true(abs(outputs(0,0) - 0) < type(1e-3), LOG);
    assert_true(abs(outputs(0,1) - 0) < type(1e-3), LOG);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    outputs_number = 1;

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});

    neural_network.set_parameters_constant(type(0));

    inputs.resize(1, 1);
    inputs.setZero();
   
    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == 1, LOG);
    assert_true(abs(outputs(0,0) - 0) < type(1e-3), LOG);

    // Test

    neural_network.set(NeuralNetwork::ModelType::Classification, {1,1});

    neural_network.set_parameters_constant(type(0));

    inputs.resize(1, 1);
    inputs.setZero();
   
    outputs = neural_network.calculate_outputs(inputs);

    assert_true(outputs.size() == 1, LOG);
    assert_true(abs(outputs(0,0) - type(0.5)) < type(1e-3), LOG);

    // Test 7

    neural_network.set(NeuralNetwork::ModelType::Approximation, {1,3,3,3,1});

    batch_samples_number = 2;
    inputs_number = neural_network.get_inputs_number();
    outputs_number = neural_network.get_outputs_number();

    inputs.resize(batch_samples_number, inputs_number);
    inputs.setConstant(type(0));

    parameters_number = neural_network.get_parameters_number();
    parameters.resize(parameters_number);
    parameters.setConstant(type(0));

    neural_network.set_parameters(parameters);

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

    neural_network.set(NeuralNetwork::ModelType::Approximation, {3, 4, 2});
    neural_network.set_parameters_constant(type(0));

    inputs.resize(2,3);
    inputs.setValues({{type(-5),type(-1),-type(3)},
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

    neural_network.set(NeuralNetwork::ModelType::Approximation, {2, 4, 3});
    neural_network.save(file_name);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    outputs_number = 1;

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});
    neural_network.save(file_name); 
}


void NeuralNetworkTest::test_load()
{
    cout << "test_load\n";

    const string file_name = "../data/neural_network.xml";

    // Test

    neural_network.set(NeuralNetwork::ModelType::Approximation, {2, 4, 3});
    neural_network.save(file_name);

    neural_network.set();
    neural_network.load(file_name);

}


void NeuralNetworkTest::test_forward_propagate()
{

    cout << "test_forward_propagate\n";

    {
        // Test

        inputs_number = 2;
        outputs_number = 1;
        batch_samples_number = 5;
        bool is_training = false;

        data.resize(batch_samples_number, inputs_number + outputs_number);

        data.setValues({ {1,1,1},
                        {2,2,2},
                        {3,3,3},
                        {0,0,0},
                        {0,0,0} });

        data_set.set(data);

        data_set.set(DataSet::SampleUse::Training);

        training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(batch_samples_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        neural_network.set(NeuralNetwork::ModelType::Approximation, { inputs_number, outputs_number });

        PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(neural_network.get_layer(1));

        const Index neurons_number = perceptron_layer->get_neurons_number();

        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Logistic);

        Tensor<type, 1> biases_perceptron(neurons_number);

        biases_perceptron.setConstant(type(1));

        perceptron_layer->set_biases(biases_perceptron);

        Tensor<type, 2> synaptic_weights_perceptron(inputs_number, neurons_number);

        synaptic_weights_perceptron.setConstant(type(1));

        perceptron_layer->set_synaptic_weights(synaptic_weights_perceptron);

        ForwardPropagation forward_propagation(data_set.get_training_samples_number(), &neural_network);

        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
            = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.layers[1]);

        Tensor<type, 2> perceptron_activations = perceptron_layer_forward_propagation->outputs;

        assert_true(perceptron_activations.dimension(0) == 5, LOG);
        assert_true(abs(perceptron_activations(0, 0) - type(0.952)) < type(1e-3), LOG);
        assert_true(abs(perceptron_activations(1, 0) - type(0.993)) < type(1e-3), LOG);
        assert_true(abs(perceptron_activations(2, 0) - type(0.999)) < type(1e-3), LOG);
        assert_true(abs(perceptron_activations(3, 0) - type(0.731)) < type(1e-3), LOG);
        assert_true(abs(perceptron_activations(4, 0) - type(0.731)) < type(1e-3), LOG);
    }

    {
        // Test

        inputs_number = 4;
        outputs_number = 2;
        batch_samples_number = 3;
        bool is_training = false;

        data.resize(batch_samples_number, inputs_number + outputs_number);
        data.setValues({{-1,1,-1,1,1,0},{-2,2,3,1,1,0},{-3,3,5,1,1,0} });
        data_set.set(data);
        data_set.set_target();
        data_set.set(DataSet::SampleUse::Training);

        Tensor<Index, 1> input_raw_variables_indices(inputs_number);
        input_raw_variables_indices.setValues({ 0,1,2,3 });

        Tensor<bool, 1> input_raw_variables_use(4);
        input_raw_variables_use.setConstant(true);

        data_set.set_input_raw_variables(input_raw_variables_indices, input_raw_variables_use);

        training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(batch_samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

        neural_network.set();

        PerceptronLayer* perceptron_layer = new PerceptronLayer(inputs_number, outputs_number);
        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
        const Index neurons_number_perceptron = perceptron_layer->get_neurons_number();

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(outputs_number, outputs_number);
        probabilistic_layer->set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);
        const Index neurons_number_probabilistic = probabilistic_layer->get_neurons_number();

        Tensor<type, 1> biases_perceptron(outputs_number);
        biases_perceptron.setConstant(5);
        perceptron_layer->set_biases(biases_perceptron);

        Tensor<type, 2> synaptic_weights_perceptron(inputs_number, neurons_number_perceptron);
        synaptic_weights_perceptron.setConstant(type(-1));
        perceptron_layer->set_synaptic_weights(synaptic_weights_perceptron);

        Tensor<type, 1> biases_probabilistic(outputs_number);
        biases_probabilistic.setConstant(3);
        probabilistic_layer->set_biases(biases_probabilistic);

        Tensor<type, 2> synaptic_weights_probabilistic(neurons_number_perceptron, neurons_number_probabilistic);
        synaptic_weights_probabilistic.setConstant(type(1));
        probabilistic_layer->set_synaptic_weights(synaptic_weights_probabilistic);

        Tensor<unique_ptr<Layer>, 1> layers(2);
        layers.setValues({ perceptron_layer, probabilistic_layer });
        neural_network.set_layers(layers);

        ForwardPropagation forward_propagation(data_set.get_training_samples_number(), &neural_network);

        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
            = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.layers[0]);

        Tensor<type, 2> perceptron_activations = perceptron_layer_forward_propagation->outputs;

        ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation.layers[1]);

        Tensor<type, 2> probabilistic_activations = probabilistic_layer_forward_propagation->outputs;

        assert_true(perceptron_activations.dimension(0) == 3, LOG);
        assert_true(abs(perceptron_activations(0, 0) - type(0.993)) < type(1e-3)
            && abs(perceptron_activations(1, 0) - type(0.731)) < type(1e-3)
            && abs(perceptron_activations(2, 0) - type(0.268)) < type(1e-3), LOG);

        assert_true(probabilistic_activations.dimension(0) == 3, LOG);
        assert_true(abs(probabilistic_activations(0, 0) - 0.5) < type(1e-3)
            && abs(probabilistic_activations(1, 0) - 0.5) < type(1e-3)
            && abs(probabilistic_activations(2, 0) - 0.5) < type(1e-3), LOG);
    }

}

}
*/
