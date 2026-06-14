#include "pch.h"
#include "../opennn/tabular_dataset.h"
#include "numerical_derivatives.h"

#include "../opennn/loss.h"
#include "../opennn/tensor_types.h"
#include "../opennn/language_dataset.h"
#include "../opennn/dense_layer.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/json.h"


using namespace opennn;

TEST(CrossEntropyError2d, DefaultConstructor)
{
    Loss loss;

    EXPECT_TRUE(loss.get_neural_network() == nullptr);
    EXPECT_TRUE(loss.get_dataset() == nullptr);
}

TEST(CrossEntropyError2d, BackPropagate)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index samples_number = random_integer(2, 10);

    const Index inputs_number = random_integer(1, 10);
    const Index targets_number = 1;
    const Index neurons_number = random_integer(1, 10);

    TabularDataset dataset(samples_number, { inputs_number }, { targets_number });

    dataset.set_data_random();

    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{ inputs_number }, Shape{ targets_number }, "Sigmoid"));
    neural_network.compile();

    neural_network.set_parameters_random();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    const VectorR gradient = calculate_gradient(loss);

    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));

    type difference = 0;
    for (int i = 0; i < gradient.size(); i++)
    {
        difference += ((gradient[i] - numerical_gradient[i]) * (gradient[i] - numerical_gradient[i]));
    }
    type error = sqrt(difference);

    EXPECT_NEAR(error,0, type(1.0e-1));

    Configuration::instance().set();
}


// Batch class has been removed from the API
/*
TEST(CrossEntropyError2d, calculate_error)
{
    MatrixR data;
    TabularDataset dataset(5, { 3 }, { 1 });
    data.resize(5, 4);
    data << type(0), type(1), type(0), type(1),
            type(1), type(1), type(0), type(0),
            type(0), type(1), type(1), type(1),
            type(0), type(1), type(1), type(1),
            type(0), type(1), type(0), type(1);

    dataset.set_data(data);
    vector<Index> input_features_indices(3);
    input_features_indices = { 0,1,2 };

    vector<Index> target_features_indices(1);
    target_features_indices[0] = Index(3);

    dataset.set_variable_indices(input_features_indices, target_features_indices);
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{ 3 }, Shape{ 1 }, "Sigmoid"));

    Batch batch(5, &dataset, neural_network.get_config());

    const vector<Index> training_indices = dataset.get_sample_indices("Training");
    batch.fill(training_indices, input_features_indices, {}, target_features_indices);
    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);
    ForwardPropagation forward_propagation(5, &neural_network);

    BackPropagation back_propagation(5, &loss);

    const vector<TensorView> batch_input_pairs = batch.get_inputs();
    neural_network.forward_propagate(batch_input_pairs, forward_propagation, false);

    back_propagation.error = loss.calculate_error(batch, forward_propagation).error;

    const type calculate_error = back_propagation.error;

    EXPECT_FALSE(std::isnan(calculate_error));
    EXPECT_GE(abs(calculate_error), 0.0);
}


TEST(CrossEntropyError2d, calculate_output_gradients)
{
    MatrixR data;
    TabularDataset dataset(5, { 3 }, { 1 });

    data.resize(5, 4);
    data << type(2), type(5), type(6), type(0),
            type(2), type(9), type(1), type(0),
            type(2), type(9), type(1), type(1),
            type(6), type(5), type(6), type(1),
            type(0), type(1), type(0), type(1);

    dataset.set_data(data);

    vector<Index> input_features_indices(3);
    input_features_indices = { 0,1,2 };

    vector<Index> target_features_indices(1);
    target_features_indices[0] = Index(3);

    dataset.set_variable_indices(input_features_indices, target_features_indices);
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{ 3 }, Shape{ 1 }, "Sigmoid"));

    Batch batch(5, &dataset, neural_network.get_config());

    const vector<Index> training_indices = dataset.get_sample_indices("Training");
    batch.fill(training_indices, input_features_indices, {},  target_features_indices);


    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);
    ForwardPropagation forward_propagation(5, &neural_network);
    BackPropagation back_propagation(5, &loss);

    const vector<TensorView> batch_input_pairs = batch.get_inputs();
    neural_network.forward_propagate(batch_input_pairs, forward_propagation, true);

    loss.calculate_output_gradients(batch, forward_propagation, back_propagation);

    auto deltas = tensor_map<2>(back_propagation.get_output_gradients());
    auto outputs = tensor_map<2>(forward_propagation.get_last_trainable_layer_outputs());

    EXPECT_EQ(deltas.dimension(0), outputs.dimension(0));
    EXPECT_EQ(deltas.dimension(1), outputs.dimension(1));
}
*/


TEST(CrossEntropyError2d, get_name)
{
    MatrixR data;
    TabularDataset dataset(5, { 3 }, { 1 });

    data.resize(5, 4);
    data << type(2), type(5), type(0), type(0),
            type(2), type(9), type(1), type(0),
            type(2), type(9), type(1), type(1),
            type(6), type(5), type(0), type(1),
            type(0), type(1), type(0), type(1);

    dataset.set_data(data);

    vector<Index> input_features_indices(2);
    input_features_indices = { 0,1 };

    vector<Index> target_features_indices(2);
    target_features_indices = { 2,3 };

    dataset.set_variable_indices(input_features_indices, target_features_indices);
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{ 2 }, Shape{ 2 }, "Sigmoid"));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    string name = loss.get_name();

    EXPECT_EQ(name, "CrossEntropy");
}

TEST(CrossEntropyError2d, to_JSON)
{
    TabularDataset dataset(5, { 2 }, { 1 });
    MatrixR data;
    data.resize(5, 3);
    data << type(0.1), type(0.2), type(1),
            type(0.5), type(0.4), type(0),
            type(0.9), type(0.1), type(1),
            type(0.3), type(0.8), type(0),
            type(0.6), type(0.7), type(1);
    dataset.set_data(data);

    std::vector<Index> input_indices = { 0, 1 };
    std::vector<Index> target_indices = { 2 };
    dataset.set_variable_indices(input_indices, target_indices);
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{ 2 }, Shape{ 1 }, "Sigmoid"));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    JsonWriter writer;
    EXPECT_NO_THROW(loss.to_JSON(writer));

    std::string json_output = writer.c_str();

    // Verify we got non-empty JSON output
    EXPECT_FALSE(json_output.empty());
}

TEST(CrossEntropyError2d, from_JSON_valid_document)
{
    NeuralNetwork neural_network;
    TabularDataset dataset;
    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    JsonWriter writer;
    loss.to_JSON(writer);

    JsonDocument document;
    document.root = Json::parse(writer.c_str());

    Loss loss2;
    EXPECT_NO_THROW(loss2.from_JSON(document));
}

TEST(CrossEntropyError2d, from_JSON_invalid_document)
{
    const char* json_text = R"({"InvalidTag": {}})";

    JsonDocument document;
    document.root = Json::parse(json_text);

    Loss loss;
    EXPECT_THROW(loss.from_JSON(document), runtime_error);
}
