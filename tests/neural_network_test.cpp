#include "pch.h"

#include "../opennn/neural_network.h"
#include "../opennn/standard_networks.h"
#include "../opennn/dense_layer.h"
#include "../opennn/layer.h"
#include "../opennn/dataset.h"

using namespace opennn;

TEST(NeuralNetworkTest, DefaultConstructor)
{
    NeuralNetwork neural_network;

    EXPECT_EQ(neural_network.is_empty(), true);
    EXPECT_EQ(neural_network.get_layers_number(), 0); 
}


TEST(NeuralNetworkTest, ApproximationConstructor)
{
    ApproximationNetwork neural_network({ 1 }, { 4 }, { 2 });
    
    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling2d");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense2d");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense2d");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Bounding");
}


TEST(NeuralNetworkTest, ClassificationConstructor)
{   
    ClassificationNetwork neural_network({ 1 }, { 4 }, { 2 });
    
    EXPECT_EQ(neural_network.get_layers_number(), 3);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling2d");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense2d");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense2d");
}


TEST(NeuralNetworkTest, AproximationConstructor)
{
    ApproximationNetwork neural_network({ 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling2d");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense2d");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense2d");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Bounding");
}


TEST(NeuralNetworkTest, ForecastingConstructor)
{
    ForecastingNetwork neural_network({ 1,1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling3d");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Recurrent");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense2d");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Bounding");
}


TEST(NeuralNetworkTest, AutoAssociationConstructor)
{
    AutoAssociationNetwork neural_network({ 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 6);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling2d");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense2d");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense2d");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Dense2d");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Dense2d");
    EXPECT_EQ(neural_network.get_layer(5)->get_name(), "Unscaling");
}


TEST(NeuralNetworkTest, ImageClassificationConstructor)
{
    const Index height = 3;
    const Index width = 3;
    const Index channels = 1;

    const Index complexity = 1;

    const Index outputs_number = 1;

    ImageClassificationNetwork neural_network({height, width, channels}, { complexity }, { outputs_number });
 
    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling4d");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Convolutional");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Pooling");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Flatten4d");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Dense2d");
}


TEST(NeuralNetworkTest, ForwardPropagate)
{
    const Index samples_number = 5;
    const Index inputs_number = 2;
    const Index outputs_number = 1;
    const Index neurons_number = 1;

    bool is_training = true;

    Tensor<type, 2> data(samples_number, inputs_number + outputs_number);
    data.setValues({
        {0, 0, 1},
        {1, 1, 0},
        {2, 2, 1},
        {3, 3, 0},
        {4, 4, 1}
    });

    Dataset dataset(samples_number,
                    dimensions{inputs_number},
                    dimensions{outputs_number});
    dataset.set_data(data);
    dataset.set_sample_uses("Training");

    Batch batch(samples_number, &dataset);
    batch.fill(dataset.get_sample_indices("Training"),
               dataset.get_variable_indices("Input"),
               // dataset.get_variable_indices("Decoder"),
               dataset.get_variable_indices("Target"));
    
    // Test Logistic

    ApproximationNetwork neural_network_aproximation({inputs_number}, {neurons_number}, {outputs_number});

    ForwardPropagation forward_propagation(dataset.get_samples_number(), &neural_network_aproximation);

    neural_network_aproximation.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

    Dense2dForwardPropagation* perceptron_layer_forward_propagation
        = static_cast<Dense2dForwardPropagation*>(forward_propagation.layers[1].get());

    Tensor <type, 2> perceptron_activations = perceptron_layer_forward_propagation->outputs;

    EXPECT_EQ(perceptron_activations.dimension(0), 5);

    // Test Softmax
    
    ClassificationNetwork neural_network_classification({inputs_number}, {neurons_number}, {outputs_number});

    Dense2d* probabilistic_layer =static_cast<Dense2d*>(neural_network_classification.get_first("Dense2d"));
    probabilistic_layer->set_activation_function("Softmax");

    ForwardPropagation forward_propagation_0(dataset.get_samples_number(), &neural_network_classification);

    neural_network_classification.forward_propagate(batch.get_input_pairs(), forward_propagation_0, is_training);

    Dense2dForwardPropagation* probabilistic_layer_forward_propagation
        = static_cast<Dense2dForwardPropagation*>(forward_propagation_0.layers[2].get());

    Tensor <type, 2> probabilistic_activations = probabilistic_layer_forward_propagation->outputs;

    EXPECT_EQ(probabilistic_activations.dimension(0), 5);
}


TEST(NeuralNetworkTest, CalculateOutputsEmpty)
{
    NeuralNetwork neural_network;

    Tensor<type, 2> inputs;

    const Tensor<type, 2> outputs = neural_network.calculate_outputs<2,2>(inputs);

    EXPECT_EQ(outputs.size(), 0);

}


TEST(NeuralNetworkTest, CalculateDirectionalInputs)
{

    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;
    Tensor<type, 2> trainable_outputs;
    Tensor<type, 1> parameters;
    Tensor<type, 1> point;
    Tensor<type, 2> directional_inputs;

    // Test
        
    ApproximationNetwork neural_network({ 3 }, { 4 }, { 2 });
    neural_network.set_parameters_random();

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
    EXPECT_NEAR(directional_inputs(0,2), - type(1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(directional_inputs(1,2), type(0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(directional_inputs(2,2), type(1), NUMERIC_LIMITS_MIN);

    // Test

    point.setValues({type(1), type(2), type(3)});

    directional_inputs = neural_network.calculate_directional_inputs(0, point, type(-4), type(0), 5);

    EXPECT_EQ(directional_inputs.rank(), 2);
    EXPECT_EQ(directional_inputs.dimension(0), 5);
    EXPECT_EQ(directional_inputs.dimension(1), 3);
    EXPECT_NEAR(abs(directional_inputs(0,0)), type(4), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(directional_inputs(1,0)), type(3), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(directional_inputs(2,0)), type(2), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(directional_inputs(3,0)), type(1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(directional_inputs(4,0)), type(0), NUMERIC_LIMITS_MIN);
}


TEST(NeuralNetworkTest, TestSaveLoad)
{
    Index inputs_number;
    Index neurons_number;
    Index outputs_number;

    const string file_path_str = "../blank/data/neural_network.xml";
    const filesystem::path file_path(file_path_str);

    if (!filesystem::exists(file_path.parent_path()))
        filesystem::create_directories(file_path.parent_path());
  
    // Empty neural network

    NeuralNetwork empty_net;
    empty_net.save(file_path);

    NeuralNetwork loaded_empty_net(file_path);

    EXPECT_EQ(loaded_empty_net.get_features_number(), 0);
    EXPECT_EQ(loaded_empty_net.get_layers_number(), 0);
    EXPECT_EQ(loaded_empty_net.get_outputs_number(), 0);

    // Standard neural network

    ApproximationNetwork neural_approx1_network({ 5 }, { 1 }, { 3 });
    neural_approx1_network.save(file_path);

    NeuralNetwork loaded_neural_approx1_network(file_path);

    EXPECT_EQ(neural_approx1_network.get_features_number(), loaded_neural_approx1_network.get_features_number());

    EXPECT_EQ(neural_approx1_network.get_layers_number(), loaded_neural_approx1_network.get_layers_number());
    EXPECT_EQ(neural_approx1_network.get_layer(0)->get_name(), loaded_neural_approx1_network.get_layer(0)->get_name());
    EXPECT_EQ(neural_approx1_network.get_layer(1)->get_name(), loaded_neural_approx1_network.get_layer(1)->get_name());
    EXPECT_EQ(neural_approx1_network.get_layer(2)->get_name(), loaded_neural_approx1_network.get_layer(2)->get_name());
    EXPECT_EQ(neural_approx1_network.get_layer(3)->get_name(), loaded_neural_approx1_network.get_layer(3)->get_name());
    EXPECT_EQ(neural_approx1_network.get_layer(4)->get_name(), loaded_neural_approx1_network.get_layer(4)->get_name());

    EXPECT_EQ(neural_approx1_network.get_layer_input_indices().size(), loaded_neural_approx1_network.get_layer_input_indices().size());

    EXPECT_EQ(neural_approx1_network.get_outputs_number(), loaded_neural_approx1_network.get_outputs_number());
}