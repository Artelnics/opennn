#include "pch.h"

#include "../opennn/neural_network.h"
#include "../opennn/perceptron_layer.h"
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
    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, { 1 }, { 4 }, { 2 });
    
    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling2d);
    EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(3)->get_type(), Layer::Type::Unscaling);
    EXPECT_EQ(neural_network.get_layer(4)->get_type(), Layer::Type::Bounding);
    
    
    for (Index i = 0; i < neural_network.get_layers_number(); i++) {
        Layer::Type type = neural_network.get_layer(i)->get_type();

        switch (i) {
        case 0:
            EXPECT_EQ(type, Layer::Type::Scaling2d);
            break;    
        case 1:
            EXPECT_EQ(type, Layer::Type::Dense2d);
            break;
        case 2:
            EXPECT_EQ(type, Layer::Type::Dense2d);
            break;
        case 3:
            EXPECT_EQ(type, Layer::Type::Unscaling);
            break;
        case 4:
            EXPECT_EQ(type, Layer::Type::Bounding);
            break;
        default:
            break;
        }
    }

    EXPECT_EQ(neural_network.get_layers_number(), 5);
}


TEST(NeuralNetworkTest, ClassificationConstructor)
{   
    NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification, { 1 }, { 4 }, { 2 });
    
    EXPECT_EQ(neural_network.get_layers_number(), 3);
    EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling2d);
    EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Dense2d);
}


TEST(NeuralNetworkTest, ForecastingConstructor)
{
    NeuralNetwork neural_network(NeuralNetwork::ModelType::Forecasting, { 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling2d);
    EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Recurrent);
    EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(3)->get_type(), Layer::Type::Unscaling);
    EXPECT_EQ(neural_network.get_layer(4)->get_type(), Layer::Type::Bounding);
}


TEST(NeuralNetworkTest, AutoAssociationConstructor)
{
    NeuralNetwork neural_network(NeuralNetwork::ModelType::AutoAssociation, { 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 6);
    EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling2d);
    EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(3)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(4)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(5)->get_type(), Layer::Type::Unscaling);
}


TEST(NeuralNetworkTest, ImageClassificationConstructor)
{
    const Index height = 3;
    const Index width = 3;
    const Index channels = 1;

    const Index blocks = 1;

    const Index outputs_number = 1;

    NeuralNetwork neural_network(NeuralNetwork::ModelType::ImageClassification,
        {height, width, channels}, {blocks}, { outputs_number });
 
    EXPECT_EQ(neural_network.get_layers_number(), 5); 
    EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling4d);
    EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Convolutional);
    EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Pooling);
    EXPECT_EQ(neural_network.get_layer(3)->get_type(), Layer::Type::Flatten);
    EXPECT_EQ(neural_network.get_layer(4)->get_type(), Layer::Type::Dense2d);
}


TEST(NeuralNetworkTest, ForwardPropagate)
{
    //Test Dense2d

    Index inputs_number = 2;
    Index outputs_number = 1;
    Index batch_samples_number = 5;
    Index neurons_number = 1;

    bool is_training = true;

    Tensor <type, 2> data(batch_samples_number, inputs_number + outputs_number);

    data.setValues({{1, 1, 1},
                    {2, 2, 2},
                    {3, 3, 3},
                    {0, 0, 0},
                    {0, 0, 0}});

    Dataset dataset(batch_samples_number, {inputs_number}, {outputs_number});
    dataset.set_data(data);
    dataset.set(Dataset::SampleUse::Training);

    Batch batch(batch_samples_number, &dataset);
    batch.fill(dataset.get_sample_indices(Dataset::SampleUse::Training),
               dataset.get_variable_indices(Dataset::VariableUse::Input),
               dataset.get_variable_indices(Dataset::VariableUse::Decoder),
               dataset.get_variable_indices(Dataset::VariableUse::Target));

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});

    Dense2d* perceptron_layer = static_cast<Dense2d*>(neural_network.get_first(Layer::Type::Dense2d));
    perceptron_layer->set_activation_function(Dense2d::Activation::Logistic);

    ForwardPropagation forward_propagation(dataset.get_samples_number(), &neural_network);

    neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

    Dense2dForwardPropagation* perceptron_layer_forward_propagation
        = static_cast<Dense2dForwardPropagation*>(forward_propagation.layers[1].get());

    Tensor <type, 2> perceptron_activations = perceptron_layer_forward_propagation->outputs;

    EXPECT_EQ(perceptron_activations.dimension(0), 5);
    EXPECT_LE(perceptron_activations(0, 0) - type(0.5), type(1e-1));
    EXPECT_LE(perceptron_activations(1, 0) - type(0.5), type(1e-1));
    EXPECT_LE(perceptron_activations(2, 0) - type(0.5), type(1e-1));
    EXPECT_LE(perceptron_activations(3, 0) - type(0.5), type(1e-1));
    EXPECT_LE(perceptron_activations(4, 0) - type(0.5), type(1e-1));

    //Test Probabilistic

    NeuralNetwork neural_network_0(NeuralNetwork::ModelType::Classification, {inputs_number}, {neurons_number}, {outputs_number});

    Dense2d* probabilistic_layer =static_cast<Dense2d*>(neural_network_0.get_first(Layer::Type::Dense2d));
    probabilistic_layer->set_activation_function(Dense2d::Activation::Softmax);
/*
    ForwardPropagation forward_propagation_0(dataset.get_samples_number(), &neural_network_0);

    neural_network_0.forward_propagate(batch.get_input_pairs(), forward_propagation_0, is_training);

    ProbabilisticForwardPropagation* probabilistic_layer_forward_propagation
        = static_cast<ProbabilisticForwardPropagation*>(forward_propagation_0.layers[2].get());

    Tensor <type, 2> probabilistic_activations = probabilistic_layer_forward_propagation->outputs;

    EXPECT_EQ(probabilistic_activations.dimension(0), 5);
    EXPECT_LE(probabilistic_activations(0, 0) - type(0.5), type(1e-1));
    EXPECT_LE(probabilistic_activations(1, 0) - type(0.5), type(1e-1));
    EXPECT_LE(probabilistic_activations(2, 0) - type(0.5), type(1e-1));
    EXPECT_LE(probabilistic_activations(3, 0) - type(0.5), type(1e-1));
    EXPECT_LE(probabilistic_activations(4, 0) - type(0.5), type(1e-1));
*/
}


TEST(NeuralNetworkTest, CalculateOutputsEmpty)
{
    NeuralNetwork neural_network;

    Tensor<type, 2> inputs;
/*
    const Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size(), 0);
*/
}


TEST(NeuralNetworkTest, calculate_directional_inputs)
{

    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;
    Tensor<type, 2> trainable_outputs;

    Tensor<type, 1> parameters;

    Tensor<type, 1> point;

    Tensor<type, 2> directional_inputs;

    // Test
        
    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, { 3 }, { 4 }, { 2 });
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


TEST(NeuralNetworkTest, test_save)
{
    NeuralNetwork neural_network;

    Index inputs_number;
    Index neurons_number;
    Index outputs_number;

    string file_name = "../data/neural_network.xml";

    // Empty neural network

    neural_network.set();
    neural_network.save(file_name);

    // Only network architecture

    neural_network.set(NeuralNetwork::ModelType::Approximation, { 2 }, { 4 }, { 3 });
    neural_network.save(file_name);

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling2d);
    EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(3)->get_type(), Layer::Type::Unscaling);
    EXPECT_EQ(neural_network.get_layer(4)->get_type(), Layer::Type::Bounding);

    // Test

    inputs_number = 1;
    neurons_number = 1;
    outputs_number = 1;

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {outputs_number});
    neural_network.save(file_name); 

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_type(), Layer::Type::Scaling2d);
    EXPECT_EQ(neural_network.get_layer(1)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(2)->get_type(), Layer::Type::Dense2d);
    EXPECT_EQ(neural_network.get_layer(3)->get_type(), Layer::Type::Unscaling);
    EXPECT_EQ(neural_network.get_layer(4)->get_type(), Layer::Type::Bounding);
}

TEST(NeuralNetworkTest, test_load)
{
    /*
    NeuralNetwork neural_network;

    const string file_name = "../examples/rosenbrock/data/neural_network.xml";

    // Test

    neural_network.set(NeuralNetwork::ModelType::Approximation, { 2 }, { 4 }, { 3 });
    neural_network.save(file_name);

    neural_network.set();
    neural_network.load(file_name);
    */
}
/*
TEST(NeuralNetworkTest, test_forward_propagate)
{
    {
        // Test

        Index inputs_number = 2;
        Index outputs_number = 1;
        Index batch_size = 5;

        bool is_training = false;

        Tensor<type, 2> data;

        data.resize(batch_size, inputs_number + outputs_number);

        data.setValues({ {1,1,1},
                        {2,2,2},
                        {3,3,3},
                        {0,0,0},
                        {0,0,0} });

        Dataset dataset;

        dataset.set_data(data);

        dataset.set(Dataset::SampleUse::Training);

        //training_samples_indices = dataset.get_sample_indices(SampleUse::Training);
        //input_variables_indices = dataset.get_input_variables_indices();
        //target_variables_indices = dataset.get_target_variables_indices();

        vector<Index> training_samples_indices = dataset.get_sample_indices(Dataset::SampleUse::Training);
        vector<Index> input_variables_indices = dataset.get_used_variable_indices();
        vector<Index> target_variables_indices = dataset.get_used_raw_variables_indices();

        Batch batch;

        batch.set(batch_size, &dataset);

        batch.fill(training_samples_indices, input_variables_indices, {}, target_variables_indices);

        NeuralNetwork neural_network;

        neural_network.set(NeuralNetwork::ModelType::Approximation, { inputs_number }, {outputs_number }, {1});

        //Dense2d* perceptron_layer = static_cast<Dense2d*>(neural_network.get_layer(1));
        
        Layer::Type perceptron_layer = neural_network.get_layer(1)->get_type();

        const Index neurons_number = perceptron_layer->get_neurons_number();

        perceptron_layer=set_activation_function(Dense2d::Activation::Logistic);

        ForwardPropagation forward_propagation(dataset.get_training_samples_number(), &neural_network);

        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        Dense2dForwardPropagation* perceptron_layer_forward_propagation
            = static_cast<Dense2dForwardPropagation*>(forward_propagation.layers[1]);

        Tensor<type, 2> perceptron_activations = perceptron_layer_forward_propagation->outputs;

        EXPECT_EQ(perceptron_activations.dimension(0), 5);
        EXPECT_EQ(abs(perceptron_activations(0, 0)), type(0.952), type(1e-3));
        EXPECT_EQ(abs(perceptron_activations(1, 0)), type(0.993), type(1e-3));
        EXPECT_EQ(abs(perceptron_activations(2, 0)), type(0.999), type(1e-3));
        EXPECT_EQ(abs(perceptron_activations(3, 0)), type(0.731), type(1e-3));
        EXPECT_EQ(abs(perceptron_activations(4, 0)), type(0.731), type(1e-3));
    }

    {
        // Test

        Index inputs_number = 4;
        Index outputs_number = 2;
        Index batch_size = 3;

        bool is_training = false;

        Tensor<type, 2> data;

        data.resize(batch_size, inputs_number + outputs_number);
        data.setValues({{-1,1,-1,1,1,0},{-2,2,3,1,1,0},{-3,3,5,1,1,0} });

        Dataset dataset;

        dataset.set_data(data);
        dataset.set_target();
        dataset.set(Dataset::SampleUse::Training);

        Tensor<Index, 1> input_raw_variable_indices(inputs_number);
        input_raw_variable_indices.setValues({ 0,1,2,3 });

        Tensor<bool, 1> input_raw_variables_use(4);
        input_raw_variables_use.setConstant(true);

        dataset.set_input_raw_variables(input_raw_variable_indices, input_raw_variables_use);

        training_samples_indices = dataset.get_sample_indices(Dataset::SampleUse::Training);
        input_variables_indices = dataset.get_input_variables_indices();
        target_variables_indices = dataset.get_target_variables_indices();

        batch.set(batch_size, &dataset);
        batch.fill(training_samples_indices, input_variables_indices, {}, target_variables_indices);

        neural_network.set();

        Dense2d* perceptron_layer = new Dense2d(inputs_number, outputs_number);
        perceptron_layer->set_activation_function(Dense2d::Activation::Logistic);
        const Index neurons_number_perceptron = perceptron_layer->get_neurons_number();

        Probabilistic* probabilistic_layer = new Probabilistic(outputs_number, outputs_number);
        probabilistic_layer->set_activation_function(Probabilistic::Activation::Softmax);
        const Index neurons_number_probabilistic = probabilistic_layer->get_neurons_number();

        Tensor<unique_ptr<Layer>, 1> layers(2);
        layers.setValues({ perceptron_layer, probabilistic_layer });
        neural_network.set_layers(layers);

        ForwardPropagation forward_propagation(dataset.get_training_samples_number(), &neural_network);

        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, is_training);

        Dense2dForwardPropagation* perceptron_layer_forward_propagation
            = static_cast<Dense2dForwardPropagation*>(forward_propagation.layers[0]);

        Tensor<type, 2> perceptron_activations = perceptron_layer_forward_propagation->outputs;

        ProbabilisticForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticForwardPropagation*>(forward_propagation.layers[1]);

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
*/
