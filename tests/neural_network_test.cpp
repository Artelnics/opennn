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
    
    for (Index i = 0; i < neural_network.get_layers_number(); i++)
    {
        const string& name = neural_network.get_layer(i)->get_name();

        switch (i) {
        case 0:
            EXPECT_EQ(name, "Scaling2d");
            break;    
        case 1:
            EXPECT_EQ(name, "Dense2d");
            break;
        case 2:
            EXPECT_EQ(name, "Dense2d");
            break;
        case 3:
            EXPECT_EQ(name, "Unscaling");
            break;
        case 4:
            EXPECT_EQ(name, "Bounding");
            break;
        default:
            break;
        }
    }

    EXPECT_EQ(neural_network.get_layers_number(), 5);
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
    ForecastingNetwork neural_network({ 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 4);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling2d");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Recurrent");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Bounding");
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

    const Index blocks = 1;

    const Index outputs_number = 1;

    ImageClassificationNetwork neural_network({height, width, channels}, {blocks}, { outputs_number });
 
    EXPECT_EQ(neural_network.get_layers_number(), 3);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling4d");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Flatten");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense2d");
}


TEST(NeuralNetworkTest, ForwardPropagate)
{
    constexpr Index inputs_number = 2;
    constexpr Index outputs_number = 1;
    constexpr Index batch_size = 5;
    constexpr Index neurons_number = 1;

    bool is_training = true;

    Tensor<type, 2> data(batch_size, inputs_number + outputs_number);
    data.setValues({
        {1, 1, 1},
        {2, 2, 2},
        {3, 3, 3},
        {0, 0, 0},
        {0, 0, 0}
    });

    Dataset dataset(batch_size,
                    dimensions{inputs_number},
                    dimensions{outputs_number});
    dataset.set_data(data);
    dataset.set("Training");

    Batch batch(batch_size, &dataset);
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
    EXPECT_LE(perceptron_activations(0, 0) - type(0.5), type(1e-1));
    EXPECT_LE(perceptron_activations(1, 0) - type(0.5), type(1e-1));
    EXPECT_LE(perceptron_activations(2, 0) - type(0.5), type(1e-1));
    EXPECT_LE(perceptron_activations(3, 0) - type(0.5), type(1e-1));
    EXPECT_LE(perceptron_activations(4, 0) - type(0.5), type(1e-1));

    // Test Probabilistic

    ClassificationNetwork neural_network_classification({inputs_number}, {neurons_number}, {outputs_number});

    Dense2d* probabilistic_layer =static_cast<Dense2d*>(neural_network_classification.get_first("Dense2d"));
    probabilistic_layer->set_activation_function("Softmax");

    ForwardPropagation forward_propagation_0(dataset.get_samples_number(), &neural_network_classification);

    neural_network_classification.forward_propagate(batch.get_input_pairs(), forward_propagation_0, is_training);

    Dense2dForwardPropagation* probabilistic_layer_forward_propagation
        = static_cast<Dense2dForwardPropagation*>(forward_propagation_0.layers[2].get());

    Tensor <type, 2> probabilistic_activations = probabilistic_layer_forward_propagation->outputs;

    EXPECT_EQ(probabilistic_activations.dimension(0), 5);
    EXPECT_LE(probabilistic_activations(0, 0) - type(0.5), type(1e-1));
    EXPECT_LE(probabilistic_activations(1, 0) - type(0.5), type(1e-1));
    EXPECT_LE(probabilistic_activations(2, 0) - type(0.5), type(1e-1));
    EXPECT_LE(probabilistic_activations(3, 0) - type(0.5), type(1e-1));
    EXPECT_LE(probabilistic_activations(4, 0) - type(0.5), type(1e-1));
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

// @todo apparently save and load functions doesn't work
TEST(NeuralNetworkTest, TestSave)
{
    // NeuralNetwork neural_network;

    // Index inputs_number;
    // Index neurons_number;
    // Index outputs_number;

    // const string file_name = "../../blank/data/neural_network.xml";

    // // Empty neural network

    // neural_network.set();
    // neural_network.save(file_name);

    // // Only network architecture

    // neural_network.set(NeuralNetwork::ModelType::Approximation, { 2 }, { 4 }, { 3 });
    // neural_network.save(file_name);

    // EXPECT_EQ(neural_network.get_layers_number(), 5);
    // EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling2d);
    // EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Dense2d);
    // EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Dense2d);
    // EXPECT_EQ(neural_network.get_layer(3)->get_type(), Layer::Type::Unscaling);
    // EXPECT_EQ(neural_network.get_layer(4)->get_type(), Layer::Type::Bounding);

    // // Test

    // inputs_number = 1;
    // neurons_number = 1;
    // outputs_number = 1;

    // neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});
    // neural_network.save(file_name);

    // EXPECT_EQ(neural_network.get_layers_number(), 5);
    // EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling2d);
    // EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Dense2d);
    // EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Dense2d);
    // EXPECT_EQ(neural_network.get_layer(3)->get_type(), Layer::Type::Unscaling);
    // EXPECT_EQ(neural_network.get_layer(4)->get_type(), Layer::Type::Bounding);
}


TEST(NeuralNetworkTest, TestLoad)
{
    // const std::string file_name = "../../blank/data/neural_network.xml";

    // constexpr Index inputs_number = 2;
    // constexpr Index outputs_number = 1;
    // constexpr Index neurons_number = 1;

    // NeuralNetwork neural_network(
    //     NeuralNetwork::ModelType::Approximation,
    //     {inputs_number},
    //     {neurons_number},
    //     {outputs_number});

    // neural_network.save(file_name);

    // NeuralNetwork loaded_network;
    // loaded_network.load(file_name);

    // EXPECT_EQ(loaded_network.get_layers_number(), 5);
    // EXPECT_EQ(loaded_network.get_layer(0)->get_type(), Layer::Type::Scaling2d);
    // EXPECT_EQ(loaded_network.get_layer(1)->get_type(), Layer::Type::Dense2d);
    // EXPECT_EQ(loaded_network.get_layer(2)->get_type(), Layer::Type::Dense2d);
    // EXPECT_EQ(loaded_network.get_layer(3)->get_type(), Layer::Type::Unscaling);
    // EXPECT_EQ(loaded_network.get_layer(4)->get_type(), Layer::Type::Bounding);
}

