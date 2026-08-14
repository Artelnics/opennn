#include "tests/pch.h"

#include "opennn/core/json.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/layer.h"
#include "opennn/dataset/dataset.h"

using namespace opennn;

TEST(NeuralNetworkTest, DefaultConstructor)
{
    NeuralNetwork neural_network;

    EXPECT_EQ(neural_network.is_empty(), true);
    EXPECT_EQ(neural_network.get_layers_number(), 0);
    EXPECT_EQ(neural_network.get_task(), NetworkTask::Generic);
}

TEST(NeuralNetworkTest, RejectsWrongSourceCount)
{
    NeuralNetwork neural_network;

    EXPECT_THROW(neural_network.add_layer(
                     make_unique<opennn::Dense>(Shape{2}, Shape{2}, "Identity"),
                     {-1, -2}),
                 runtime_error);

    EXPECT_THROW(neural_network.add_layer(nullptr), runtime_error);
}

TEST(NeuralNetworkTest, SerializesNetworkTask)
{
    NeuralNetwork neural_network;
    neural_network.set_task(NetworkTask::TextClassification);

    JsonWriter writer;
    neural_network.to_JSON(writer);

    JsonDocument document;
    document.root = Json::parse(writer.c_str());

    NeuralNetwork loaded;
    loaded.from_JSON(document);

    EXPECT_EQ(loaded.get_task(), NetworkTask::TextClassification);
}

TEST(NeuralNetworkTest, ApproximationConstructor)
{
    ApproximationNetwork neural_network({ 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Bounding");
    EXPECT_EQ(neural_network.get_task(), NetworkTask::Approximation);
}

TEST(NeuralNetworkTest, ClassificationConstructor)
{
    ClassificationNetwork neural_network({ 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 3);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_task(), NetworkTask::Classification);
}

TEST(NeuralNetworkTest, AproximationConstructor)
{
    ApproximationNetwork neural_network({ 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Bounding");
}

TEST(NeuralNetworkTest, ForecastingConstructor)
{
    ForecastingNetwork neural_network({ 1,1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Recurrent");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Bounding");
    EXPECT_EQ(neural_network.get_task(), NetworkTask::Forecasting);
}

TEST(NeuralNetworkTest, AutoAssociationConstructor)
{
    AutoAssociationNetwork neural_network({ 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 6);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(5)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_task(), NetworkTask::AutoAssociation);
}

TEST(NeuralNetworkTest, AutoAssociationSymmetricEncoderConstructor)
{
    AutoAssociationNetwork neural_network({140}, {32, 16, 8}, "ReLU", "Sigmoid");

    ASSERT_EQ(neural_network.get_layers_number(), 8);
    EXPECT_EQ(neural_network.get_input_shape(), Shape({140}));
    EXPECT_EQ(neural_network.get_output_shape(), Shape({140}));

    const vector<Shape> expected_shapes = {
        {140}, {32}, {16}, {8}, {16}, {32}, {140}, {140}
    };

    for (Index i = 0; i < neural_network.get_layers_number(); ++i)
        EXPECT_EQ(neural_network.get_layer(i)->get_output_shape(), expected_shapes[size_t(i)]);

    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(3)->get_label(), "bottleneck_layer");

    for (Index i = 1; i <= 5; ++i)
    {
        const opennn::Dense* dense =
            dynamic_cast<const opennn::Dense*>(neural_network.get_layer(i).get());
        ASSERT_NE(dense, nullptr);
        EXPECT_EQ(dense->get_activation_function(), ActivationFunction::ReLU);
        EXPECT_TRUE(dense->get_use_bias());
    }

    const opennn::Dense* output =
        dynamic_cast<const opennn::Dense*>(neural_network.get_layer(6).get());
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->get_activation_function(), ActivationFunction::Sigmoid);
    EXPECT_TRUE(output->get_use_bias());
}

TEST(NeuralNetworkTest, AutoAssociationSymmetricEncoderRejectsEmptyEncoder)
{
    EXPECT_THROW(AutoAssociationNetwork({140}, {}, "ReLU", "Sigmoid"), runtime_error);
}

TEST(NeuralNetworkTest, ImageClassificationConstructor)
{
    const Index height = 3;
    const Index width = 3;
    const Index channels = 1;

    const Index complexity = 1;

    const Index outputs_number = 1;

    ImageClassificationNetwork neural_network({height, width, channels}, { complexity }, { outputs_number });

    EXPECT_EQ(neural_network.get_layers_number(), 6);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Convolutional");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Pooling");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Flatten");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(4)->get_label(), "dense_2d_layer_1");
    EXPECT_EQ(neural_network.get_layer(5)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(5)->get_label(), "classification_layer");
    EXPECT_EQ(neural_network.get_task(), NetworkTask::ImageClassification);
}

TEST(NeuralNetworkTest, ForwardPropagate)
{
    const Index samples_number = 5;
    const Index inputs_number = 2;
    const Index outputs_number = 1;
    const Index neurons_number = 1;

    ApproximationNetwork neural_network_aproximation({inputs_number}, {neurons_number}, {outputs_number});
    neural_network_aproximation.set_parameters_random();

    MatrixR input_data(samples_number, inputs_number);
    input_data << 0, 0,
                  1, 1,
                  2, 2,
                  3, 3,
                  4, 4;

    MatrixR result = neural_network_aproximation.calculate_outputs(input_data);

    EXPECT_EQ(result.rows(), samples_number);
    EXPECT_EQ(result.cols(), outputs_number);

    ClassificationNetwork neural_network_classification({inputs_number}, {neurons_number}, {outputs_number});

    MatrixR result_classification = neural_network_classification.calculate_outputs(input_data);

    EXPECT_EQ(result_classification.rows(), samples_number);
    EXPECT_EQ(result_classification.cols(), outputs_number);
}

TEST(NeuralNetworkTest, CalculateOutputsEmpty)
{
    NeuralNetwork neural_network;

    MatrixR inputs;

    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size(), 0);
}
