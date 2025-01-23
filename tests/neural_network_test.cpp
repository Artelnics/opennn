#include "pch.h"

#include "../opennn/forward_propagation.h"
#include "../opennn/neural_network.h"
#include "../opennn/perceptron_layer.h"

TEST(NeuralNetworkTest, DefaultConstructor)
{
    NeuralNetwork neural_network;

    EXPECT_EQ(neural_network.is_empty(), true);
    EXPECT_EQ(neural_network.get_layers_number(), 0);
}


TEST(NeuralNetworkTest, ApproximationConstructor)
{
    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, { 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling2D);
    EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Perceptron);
    EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Perceptron);
    EXPECT_EQ(neural_network.get_layer(3)->get_type(), Layer::Type::Unscaling);
    EXPECT_EQ(neural_network.get_layer(4)->get_type(), Layer::Type::Bounding);
}


TEST(NeuralNetworkTest, ClassificationConstructor)
{    
    NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification, { 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 3);
    EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling2D);
    EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Perceptron);
    EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Probabilistic);
}


TEST(NeuralNetworkTest, ForecastingConstructor)
{
    NeuralNetwork neural_network(NeuralNetwork::ModelType::Forecasting, { 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling2D);
//    EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Recurrent);
    EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Perceptron);
    EXPECT_EQ(neural_network.get_layer(3)->get_type(), Layer::Type::Unscaling);
}


TEST(NeuralNetworkTest, AutoAssociationConstructor)
{

}


TEST(NeuralNetworkTest, ImageClassificationConstructor)
{
    // Input dimensions {height, width, channels}

    const Index height = 3;
    const Index width = 3;
    const Index channels = 1;

    const Index blocks = 1;

    const Index outputs_number = 1;

    //NeuralNetwork neural_network(NeuralNetwork::ModelType::ImageClassification,
    //    {height, width, channels}, {blocks}, { outputs_number });
 
    //EXPECT_EQ(neural_network.get_layers_number(), 5); 
    //EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling4D);
    //EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Convolutional);
    //EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Pooling);
    //EXPECT_EQ(neural_network.get_layer(3)->get_type(), Layer::Type::Flatten);
    //EXPECT_EQ(neural_network.get_layer(4)->get_type(), Layer::Type::Perceptron);
    //EXPECT_EQ(neural_network.get_layer(5)->get_type(), Layer::Type::Probabilistic);

}


TEST(NeuralNetworkTest, ForwardPropagate)
{
    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<PerceptronLayer>(dimensions{1}, dimensions{1}));

    ForwardPropagation forward_propagation(1, &neural_network);
}


TEST(NeuralNetworkTest, CalculateOutputsEmpty)
{
    NeuralNetwork neural_network;

    Tensor<type, 2> inputs;

    const Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size(), 0);
}


TEST(NeuralNetworkTest, CalculateOutputsZero)
{
    const Index samples_number = get_random_index(1, 5);
    const Index inputs_number = get_random_index(1, 5);
    const Index neurons_number = get_random_index(1, 5);
    const Index outputs_number = get_random_index(1, 5);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, 
        {inputs_number}, {neurons_number}, {outputs_number});
    neural_network.set_parameters_constant(type(0));

    Tensor<type, 2> inputs(samples_number, inputs_number);
    inputs.setConstant(type(0));  
    /*
    const Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);
    
//    EXPECT_EQ(outputs.size(), batch_samples_number * outputs_number);
//    EXPECT_NEAR(outputs(0,0), 0, NUMERIC_LIMITS_MIN);
//    EXPECT_NEAR(outputs(0,1), 0, NUMERIC_LIMITS_MIN);
//    EXPECT_NEAR(outputs(0,2), 0, NUMERIC_LIMITS_MIN);
//    EXPECT_NEAR(outputs(0,3), 0, NUMERIC_LIMITS_MIN);
//    EXPECT_NEAR(outputs(0,4), 0, NUMERIC_LIMITS_MIN);

    // Test

    neural_network.set(NeuralNetwork::ModelType::Approximation, {1, 2});

    neural_network.set_parameters_constant(type(1));

    inputs.resize(1, 1);
    inputs.setConstant(2);    

    outputs = neural_network.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size() == 2);
    EXPECT_EQ(abs(outputs(0,0) - type(3)) < NUMERIC_LIMITS_MIN);
    EXPECT_EQ(abs(outputs(0,1) - type(3)) < NUMERIC_LIMITS_MIN);

    // Test

    neural_network.set(NeuralNetwork::ModelType::Approximation, {4, 3, 3});

    inputs.resize(1, 4);
    inputs.setZero();

    neural_network.set_parameters_constant(type(1));    

    outputs = neural_network.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size() == 3);

    EXPECT_EQ(outputs(0,0), 3.2847, type(1e-3));
    EXPECT_EQ(outputs(0,1), 3.2847, type(1e-3));
    EXPECT_EQ(outputs(0,2), 3.2847, type(1e-3));

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

    EXPECT_EQ(outputs.size() == outputs_number);
    EXPECT_EQ(abs(outputs(0,0) - 0) < type(1e-3));
    EXPECT_EQ(abs(outputs(0,1) - 0) < type(1e-3));

    // Test

    inputs_number = 1;
    neurons_number = 1;
    outputs_number = 1;

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});

    neural_network.set_parameters_constant(type(0));

    inputs.resize(1, 1);
    inputs.setZero();
   
    outputs = neural_network.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size() == 1);
    EXPECT_EQ(abs(outputs(0,0) - 0) < type(1e-3));

    // Test

    neural_network.set(NeuralNetwork::ModelType::Classification, {1,1});

    neural_network.set_parameters_constant(type(0));

    inputs.resize(1, 1);
    inputs.setZero();
   
    outputs = neural_network.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size() == 1);
    EXPECT_EQ(abs(outputs(0,0) - type(0.5)) < type(1e-3));

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

    EXPECT_EQ(outputs.dimension(1) == outputs_number);
    EXPECT_EQ(abs(outputs(0,0)) < NUMERIC_LIMITS_MIN);
    EXPECT_EQ(abs(outputs(1,0)) < NUMERIC_LIMITS_MIN);
*/
}


TEST(NeuralNetworkTest, calculate_directional_inputs)
{
/*
    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;
    Tensor<type, 2> trainable_outputs;

    Tensor<type, 1> parameters;

    Tensor<type, 1> point;

    Tensor<type, 2> directional_inputs;

    // Test

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {3, 4, 2});
    neural_network.set_parameters_constant(type(0));

    inputs.resize(2,3);
    inputs.setValues({{type(-5),type(-1),-type(3)},
                      {type(7),type(3),type(1)}});

    point.resize(3);
    point.setValues({type(0),type(0),type(0)});

    directional_inputs = neural_network.calculate_directional_inputs(0, point, type(0), type(0), 0);

    EXPECT_EQ(directional_inputs.rank(), 2);
    EXPECT_EQ(directional_inputs.dimension(0), 0);

    // Test

    point.setValues({type(1), type(2), type(3)});

    directional_inputs = neural_network.calculate_directional_inputs(2, point, type(-1), type(1), 3);

    EXPECT_EQ(directional_inputs.rank(), 2);
    EXPECT_EQ(directional_inputs.dimension(0), 3);
    EXPECT_EQ(directional_inputs.dimension(1), 3);
//    EXPECT_NEAR(directional_inputs(0,2), + type(1), NUMERIC_LIMITS_MIN);
//    EXPECT_NEAR(directional_inputs(1,2), - type(0), NUMERIC_LIMITS_MIN);
//    EXPECT_NEAR(directional_inputs(2,2), - type(1), NUMERIC_LIMITS_MIN);
/*
    // Test

    point.setValues({type(1), type(2), type(3)});

    directional_inputs = neural_network.calculate_directional_inputs(0, point, type(-4), type(0), 5);

    EXPECT_EQ(directional_inputs.rank() == 2);
    EXPECT_EQ(directional_inputs.dimension(0) == 5);
    EXPECT_EQ(directional_inputs.dimension(1) == 3);
    EXPECT_NEAR(abs(directional_inputs(0,0) + type(4)) < NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(directional_inputs(1,0) + type(3)) < NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(directional_inputs(2,0) + type(2)) < NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(directional_inputs(3,0) + type(1)) < NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(directional_inputs(4,0) + type(0)) < NUMERIC_LIMITS_MIN);
*/
}

/*
void NeuralNetworkTest::test_save()
{
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
    const string file_name = "../data/neural_network.xml";

    // Test

    neural_network.set(NeuralNetwork::ModelType::Approximation, {2, 4, 3});
    neural_network.save(file_name);

    neural_network.set();
    neural_network.load(file_name);

}


void NeuralNetworkTest::test_forward_propagate()
{
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

        data_set.set_data(data);

        data_set.set(DataSet::SampleUse::Training);

        training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(batch_samples_number, &data_set);

        batch.fill(training_samples_indices, input_variables_indices, {}, target_variables_indices);

        neural_network.set(NeuralNetwork::ModelType::Approximation, { inputs_number, outputs_number });

        PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(neural_network.get_layer(1));

        const Index neurons_number = perceptron_layer->get_neurons_number();

        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Logistic);

        ForwardPropagation forward_propagation(data_set.get_training_samples_number(), &neural_network);

        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
            = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.layers[1]);

        Tensor<type, 2> perceptron_activations = perceptron_layer_forward_propagation->outputs;

        EXPECT_EQ(perceptron_activations.dimension(0) == 5);
        EXPECT_EQ(abs(perceptron_activations(0, 0) - type(0.952)) < type(1e-3));
        EXPECT_EQ(abs(perceptron_activations(1, 0) - type(0.993)) < type(1e-3));
        EXPECT_EQ(abs(perceptron_activations(2, 0) - type(0.999)) < type(1e-3));
        EXPECT_EQ(abs(perceptron_activations(3, 0) - type(0.731)) < type(1e-3));
        EXPECT_EQ(abs(perceptron_activations(4, 0) - type(0.731)) < type(1e-3));
    }

    {
        // Test

        inputs_number = 4;
        outputs_number = 2;
        batch_samples_number = 3;
        bool is_training = false;

        data.resize(batch_samples_number, inputs_number + outputs_number);
        data.setValues({{-1,1,-1,1,1,0},{-2,2,3,1,1,0},{-3,3,5,1,1,0} });
        data_set.set_data(data);
        data_set.set_target();
        data_set.set(DataSet::SampleUse::Training);

        Tensor<Index, 1> input_raw_variable_indices(inputs_number);
        input_raw_variable_indices.setValues({ 0,1,2,3 });

        Tensor<bool, 1> input_raw_variables_use(4);
        input_raw_variables_use.setConstant(true);

        data_set.set_input_raw_variables(input_raw_variable_indices, input_raw_variables_use);

        training_samples_indices = data_set.get_sample_indices(SampleUse::Training);
        input_variables_indices = data_set.get_input_variables_indices();
        target_variables_indices = data_set.get_target_variables_indices();

        batch.set(batch_samples_number, &data_set);
        batch.fill(training_samples_indices, input_variables_indices, {}, target_variables_indices);

        neural_network.set();

        PerceptronLayer* perceptron_layer = new PerceptronLayer(inputs_number, outputs_number);
        perceptron_layer->set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
        const Index neurons_number_perceptron = perceptron_layer->get_neurons_number();

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(outputs_number, outputs_number);
        probabilistic_layer->set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);
        const Index neurons_number_probabilistic = probabilistic_layer->get_neurons_number();

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

        EXPECT_EQ(perceptron_activations.dimension(0) == 3);
        EXPECT_EQ(abs(perceptron_activations(0, 0) - type(0.993)) < type(1e-3)
            && abs(perceptron_activations(1, 0) - type(0.731)) < type(1e-3)
            && abs(perceptron_activations(2, 0) - type(0.268)) < type(1e-3));

        EXPECT_EQ(probabilistic_activations.dimension(0) == 3);
        EXPECT_EQ(abs(probabilistic_activations(0, 0) - 0.5) < type(1e-3)
            && abs(probabilistic_activations(1, 0) - 0.5) < type(1e-3)
            && abs(probabilistic_activations(2, 0) - 0.5) < type(1e-3));
    }
}
}
*/
