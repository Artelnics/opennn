#include "pch.h"
#include "numerical_derivatives.h"

#include "opennn/convolutional_layer.h"
#include "opennn/tensor_types.h"
#include "opennn/flatten_layer.h"
#include "opennn/dense_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"
#include "opennn/json.h"

using namespace opennn;

struct ConvolutionalLayerConfig {
    Shape input_shape;
    Shape kernel_shape;
    Shape stride_shape;
    string activation_function;
    string convolution_type;
    bool batch_normalization;
    string test_name;
};

class ConvolutionalLayerTest : public ::testing::TestWithParam<ConvolutionalLayerConfig> {};

namespace
{

Json convolution_json_body(const Shape& kernel_shape,
                           const Shape& stride_shape = {1, 1},
                           const string& convolution_type = "Valid",
                           bool batch_normalization = false,
                           bool residual = false)
{
    Json body = Json::make_object();
    body.set("KernelsHeight", kernel_shape[0]);
    body.set("KernelsWidth", kernel_shape[1]);
    body.set("KernelsChannels", kernel_shape[2]);
    body.set("KernelsNumber", kernel_shape[3]);
    body.set("StrideDimensions", shape_to_string(stride_shape));
    body.set("Convolution", convolution_type);
    body.set("BatchNormalization", batch_normalization);
    body.set("Residual", residual);
    return body;
}

}

INSTANTIATE_TEST_SUITE_P(ConvolutionalLayerTests, ConvolutionalLayerTest, ::testing::Values(

                                                                              ConvolutionalLayerConfig{
                                                                                  {28, 28, 1},
                                                                                  {3, 3, 1, 16},
                                                                                  {1, 1},
                                                                                  "Identity",
                                                                                  "Valid",
                                                                                  false,
                                                                                  "ValidPaddingWithoutBN"
                                                                              },

                                                                              ConvolutionalLayerConfig{
                                                                                  {32, 32, 3},
                                                                                  {5, 5, 3, 32},
                                                                                  {1, 1},
                                                                                  "ReLU",
                                                                                  "Same",
                                                                                  false,
                                                                                  "SamePaddingWithoutBN"
                                                                              },

                                                                              ConvolutionalLayerConfig{
                                                                                  {16, 16, 3},
                                                                                  {3, 3, 3, 8},
                                                                                  {1, 1},
                                                                                  "ReLU",
                                                                                  "Same",
                                                                                  true,
                                                                                  "SamePaddingWithBN"
                                                                              }
                                                                              ));

TEST_P(ConvolutionalLayerTest, Constructor) {

    ConvolutionalLayerConfig parameters = GetParam();

    Convolutional convolutional_layer(parameters.input_shape,
                                      parameters.kernel_shape,
                                      parameters.activation_function,
                                      parameters.stride_shape,
                                      parameters.convolution_type,
                                      parameters.batch_normalization,
                                      parameters.test_name);

    EXPECT_EQ(convolutional_layer.get_input_shape(), parameters.input_shape);
    EXPECT_EQ(convolutional_layer.get_kernel_height(), parameters.kernel_shape[0]);
    EXPECT_EQ(convolutional_layer.get_kernel_width(), parameters.kernel_shape[1]);
    EXPECT_EQ(convolutional_layer.get_kernel_channels(), parameters.kernel_shape[2]);
    EXPECT_EQ(convolutional_layer.get_kernels_number(), parameters.kernel_shape[3]);
    EXPECT_EQ(convolutional_layer.get_row_stride(), parameters.stride_shape[0]);
    EXPECT_EQ(convolutional_layer.get_column_stride(), parameters.stride_shape[1]);
    EXPECT_EQ(convolutional_layer.get_activation_function(), ActivationOperator::from_string(parameters.activation_function));
    EXPECT_EQ(convolutional_layer.get_batch_normalization(), parameters.batch_normalization);
}

TEST_P(ConvolutionalLayerTest, OutputShapeDerivedFromConfig) {

    ConvolutionalLayerConfig parameters = GetParam();

    Convolutional convolutional_layer(parameters.input_shape,
                                      parameters.kernel_shape,
                                      parameters.activation_function,
                                      parameters.stride_shape,
                                      parameters.convolution_type,
                                      parameters.batch_normalization,
                                      parameters.test_name);

    const bool same = (parameters.convolution_type == "Same");

    const Index expected_height = same
        ? (parameters.input_shape[0] + parameters.stride_shape[0] - 1) / parameters.stride_shape[0]
        : (parameters.input_shape[0] - parameters.kernel_shape[0]) / parameters.stride_shape[0] + 1;

    const Index expected_width = same
        ? (parameters.input_shape[1] + parameters.stride_shape[1] - 1) / parameters.stride_shape[1]
        : (parameters.input_shape[1] - parameters.kernel_shape[1]) / parameters.stride_shape[1] + 1;

    const Shape expected_output_shape{expected_height, expected_width, parameters.kernel_shape[3]};

    EXPECT_EQ(convolutional_layer.get_output_shape(), expected_output_shape);
    EXPECT_EQ(convolutional_layer.get_output_height(), expected_height);
    EXPECT_EQ(convolutional_layer.get_output_width(), expected_width);
}

TEST(ConvolutionalLayerTest, JsonRejectsMismatchedKernelChannels)
{
    Convolutional layer({8, 8, 3}, {3, 3, 3, 4});
    const Json body = convolution_json_body({3, 3, 1, 4});

    EXPECT_THROW(layer.read_JSON_body(&body), runtime_error);
}

TEST(ConvolutionalLayerTest, JsonRejectsOversizedKernelAndStride)
{
    Convolutional layer({4, 4, 1}, {3, 3, 1, 4});
    const Json oversized_kernel = convolution_json_body({5, 3, 1, 4});
    const Json oversized_stride = convolution_json_body({3, 3, 1, 4}, {5, 1});

    EXPECT_THROW(layer.read_JSON_body(&oversized_kernel), runtime_error);
    EXPECT_THROW(layer.read_JSON_body(&oversized_stride), runtime_error);
}

TEST(ConvolutionalLayerTest, JsonRejectsInvalidSameAndResidualConfigurations)
{
    Convolutional layer({8, 8, 3}, {3, 3, 3, 4});
    const Json even_same_kernel =
        convolution_json_body({4, 3, 3, 4}, {1, 1}, "Same");
    const Json residual_without_batch_norm =
        convolution_json_body({3, 3, 3, 4}, {1, 1}, "Same", false, true);

    EXPECT_THROW(layer.read_JSON_body(&even_same_kernel), runtime_error);
    EXPECT_THROW(layer.read_JSON_body(&residual_without_batch_norm), runtime_error);
}

TEST(ConvolutionalLayerTest, JsonAcceptsValidConfiguration)
{
    Convolutional layer({8, 8, 3}, {3, 3, 3, 4});
    const Json body = convolution_json_body({3, 3, 3, 4}, {2, 2}, "Same", true, true);

    EXPECT_NO_THROW(layer.read_JSON_body(&body));
    EXPECT_EQ(layer.get_kernel_channels(), 3);
    EXPECT_EQ(layer.get_row_stride(), 2);
    EXPECT_TRUE(layer.get_batch_normalization());
    EXPECT_TRUE(layer.get_residual());
}

TEST_P(ConvolutionalLayerTest, ForwardPropagate)
{
    ConvolutionalLayerConfig parameters = GetParam();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Convolutional>(
        parameters.input_shape,
        parameters.kernel_shape,
        parameters.activation_function,
        parameters.stride_shape,
        parameters.convolution_type,
        parameters.batch_normalization,
        parameters.test_name));
    neural_network.compile();

    neural_network.get_parameters_map().setConstant(type(0.5));

    const Index batch_size = 2;

    Tensor4 input_data(batch_size,
                       parameters.input_shape[0],
                       parameters.input_shape[1],
                       parameters.input_shape[2]);
    input_data.setConstant(1.0);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = { TensorView(input_data.data(),
                          { batch_size,
                           parameters.input_shape[0],
                           parameters.input_shape[1],
                           parameters.input_shape[2] }) };

    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    const Shape expected_output_dims = neural_network.get_layer(0)->get_output_shape();

    ASSERT_EQ(output_view.shape.rank, 4);
    EXPECT_EQ(output_view.shape[0], batch_size);
    EXPECT_EQ(output_view.shape[1], expected_output_dims[0]);
    EXPECT_EQ(output_view.shape[2], expected_output_dims[1]);
    EXPECT_EQ(output_view.shape[3], expected_output_dims[2]);

    const Index kernel_height = parameters.kernel_shape[0];
    const Index kernel_width = parameters.kernel_shape[1];
    const Index kernel_channels = parameters.kernel_shape[2];

    const type full_sum = type(kernel_height * kernel_width * kernel_channels) * type(1.0) * type(0.5) + type(0.5);

    const type* output_data = output_view.as<type>();

    if (!parameters.batch_normalization && parameters.activation_function == "Identity")
    {
        for (Index i = 0; i < output_view.size(); ++i)
            EXPECT_NEAR(output_data[i], full_sum, 1e-5);
    }
    else if (!parameters.batch_normalization && parameters.activation_function == "ReLU"
             && parameters.convolution_type == "Same")
    {
        const Index output_height = output_view.shape[1];
        const Index output_width = output_view.shape[2];
        const Index kernels_number = output_view.shape[3];

        const Index half_height = kernel_height / 2;
        const Index half_width = kernel_width / 2;

        const auto at = [&](Index sample, Index row, Index column, Index kernel)
        {
            return output_data[((sample * output_height + row) * output_width + column) * kernels_number + kernel];
        };

        const Index interior_row = half_height;
        const Index interior_column = half_width;
        ASSERT_LT(interior_row, output_height - half_height);
        ASSERT_LT(interior_column, output_width - half_width);

        EXPECT_NEAR(at(0, interior_row, interior_column, 0), full_sum, 1e-4);

        const type corner = at(0, 0, 0, 0);
        EXPECT_LT(corner, full_sum);

        for (Index i = 0; i < output_view.size(); ++i)
            EXPECT_GE(output_data[i], type(0));
    }
}

TEST_P(ConvolutionalLayerTest, BackwardGradientMatchesNumerical)
{
    ConvolutionalLayerConfig parameters = GetParam();

    const Index samples_number = 5;
    const Index targets_number = 1;

    const Shape input_shape{8, 8, parameters.input_shape[2]};
    const Shape kernel_shape{parameters.kernel_shape[0],
                             parameters.kernel_shape[1],
                             parameters.input_shape[2],
                             3};

    TabularDataset dataset(samples_number, input_shape, {targets_number});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<Convolutional>(input_shape,
                                                        kernel_shape,
                                                        parameters.activation_function,
                                                        parameters.stride_shape,
                                                        parameters.convolution_type,
                                                        parameters.batch_normalization,
                                                        parameters.test_name));

    const Shape conv_output_shape = neural_network.get_layer(0)->get_output_shape();
    neural_network.add_layer(make_unique<Flatten>(conv_output_shape));
    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_layer(1)->get_output_shape(),
                                                        dataset.get_target_shape()));
    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const type error = calculate_numerical_error(loss);
    EXPECT_GE(error, 0);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}

TEST(ConvolutionalLayerTest, ProjectionResidualReuseGradientMatchesNumerical)
{
    constexpr Index samples_number = 4;
    const Shape input_shape{2, 2, 2};

    TabularDataset dataset(samples_number, input_shape, Shape{1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork network;
    network.add_layer(make_unique<Convolutional>(
                          input_shape, Shape{1, 1, 2, 3}, "ReLU",
                          Shape{1, 1}, "Same", true, "stem"),
                      {-1});
    network.add_layer(make_unique<Convolutional>(
                          Shape{2, 2, 3}, Shape{1, 1, 3, 4}, "ReLU",
                          Shape{1, 1}, "Same", true, "main"),
                      {0});
    network.add_layer(make_unique<Convolutional>(
                          Shape{2, 2, 3}, Shape{1, 1, 3, 4}, "Identity",
                          Shape{1, 1}, "Same", true, "projection"),
                      {0});

    auto residual = make_unique<Convolutional>(
        Shape{2, 2, 4}, Shape{1, 1, 4, 4}, "ReLU",
        Shape{1, 1}, "Same", true, "residual");
    residual->set_residual(true);
    network.add_layer(move(residual), {1, 2});

    network.add_layer(make_unique<Convolutional>(
                          Shape{2, 2, 4}, Shape{1, 1, 4, 2}, "ReLU",
                          Shape{1, 1}, "Same", true, "later"),
                      {3});
    network.add_layer(make_unique<Flatten>(Shape{2, 2, 2}), {4});
    network.add_layer(make_unique<opennn::Dense>(
                          Shape{8}, Shape{1}, "Identity"),
                      {5});
    network.compile();
    network.set_training_activation_recomputation(true);

    Loss loss(&network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(),
              type(2.0e-3));
}
