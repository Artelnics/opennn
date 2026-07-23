#include "pch.h"
#include "numerical_derivatives.h"

#include <fstream>
#include <sstream>

#include "opennn/concatenation_layer.h"
#include "opennn/convolutional_layer.h"
#include "opennn/dense_layer.h"
#include "opennn/flatten_layer.h"
#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/loss.h"

using namespace opennn;

class ConcatenationLayerTest : public ::testing::Test
{
protected:
    const Index height = 3;
    const Index width = 2;
    const Index channels_a = 2;
    const Index channels_b = 3;
    const Shape input_shape{ height, width, channels_a };
    const vector<Index> per_input_channels{ channels_a, channels_b };
};


TEST_F(ConcatenationLayerTest, Constructor)
{
    Concatenation concatenation(input_shape, per_input_channels, "concat");

    EXPECT_EQ(concatenation.get_input_shape(), input_shape);
    EXPECT_EQ(concatenation.get_inputs_number(), 2);
    EXPECT_EQ(concatenation.get_name(), "Concatenation");
    EXPECT_EQ(concatenation.get_label(), "concat");

    const Shape output_shape = concatenation.get_output_shape();
    ASSERT_EQ(output_shape.rank, 3);
    EXPECT_EQ(output_shape[0], height);
    EXPECT_EQ(output_shape[1], width);
    EXPECT_EQ(output_shape[2], channels_a + channels_b);
}


TEST_F(ConcatenationLayerTest, OutputChannelsAreSumOfInputChannels)
{
    Concatenation concatenation(Shape{ 4, 5, 7 }, vector<Index>{ 1, 6, 2 }, "concat");

    EXPECT_EQ(concatenation.get_inputs_number(), 3);

    const Shape output_shape = concatenation.get_output_shape();
    ASSERT_EQ(output_shape.rank, 3);
    EXPECT_EQ(output_shape[0], 4);
    EXPECT_EQ(output_shape[1], 5);
    EXPECT_EQ(output_shape[2], 9);
}


TEST_F(ConcatenationLayerTest, DefaultConstructorHasEmptyOutput)
{
    Concatenation concatenation;

    EXPECT_TRUE(concatenation.get_output_shape().empty());
    EXPECT_EQ(concatenation.get_inputs_number(), 0);
}


TEST_F(ConcatenationLayerTest, ForwardPropagateConcatenatesAlongChannels)
{
    const Index batch_size = 2;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Concatenation>(input_shape, per_input_channels, "concat"),
                             vector<Index>{ -1, -2 });
    neural_network.compile();

    Tensor4 inputs_a(batch_size, height, width, channels_a);
    Tensor4 inputs_b(batch_size, height, width, channels_b);

    for (Index i = 0; i < inputs_a.size(); ++i)
        inputs_a.data()[i] = float(i);
    for (Index i = 0; i < inputs_b.size(); ++i)
        inputs_b.data()[i] = float(1000 + i);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = {
        TensorView(inputs_a.data(), { batch_size, height, width, channels_a }),
        TensorView(inputs_b.data(), { batch_size, height, width, channels_b })
    };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    const Shape& output_dims = output_view.shape;

    ASSERT_EQ(output_dims.rank, 4);
    EXPECT_EQ(output_dims[0], batch_size);
    EXPECT_EQ(output_dims[1], height);
    EXPECT_EQ(output_dims[2], width);
    EXPECT_EQ(output_dims[3], channels_a + channels_b);

    const Index total_channels = channels_a + channels_b;
    const float* output_data = output_view.as<type>();
    const float* a_data = inputs_a.data();
    const float* b_data = inputs_b.data();

    for (Index b = 0; b < batch_size; ++b)
        for (Index h = 0; h < height; ++h)
            for (Index w = 0; w < width; ++w)
            {
                const Index out_base = ((b * height + h) * width + w) * total_channels;
                const Index a_base = ((b * height + h) * width + w) * channels_a;
                const Index b_base = ((b * height + h) * width + w) * channels_b;

                for (Index c = 0; c < channels_a; ++c)
                    EXPECT_NEAR(output_data[out_base + c], a_data[a_base + c], 1e-6f);

                for (Index c = 0; c < channels_b; ++c)
                    EXPECT_NEAR(output_data[out_base + channels_a + c], b_data[b_base + c], 1e-6f);
            }
}


TEST_F(ConcatenationLayerTest, ForwardPropagateThreeInputs)
{
    const Index batch_size = 1;
    const Index c0 = 1;
    const Index c1 = 2;
    const Index c2 = 1;
    const Index total_channels = c0 + c1 + c2;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Concatenation>(Shape{ height, width, c0 },
                                                        vector<Index>{ c0, c1, c2 }, "concat"),
                             vector<Index>{ -1, -2, -3 });
    neural_network.compile();

    Tensor4 in0(batch_size, height, width, c0);
    Tensor4 in1(batch_size, height, width, c1);
    Tensor4 in2(batch_size, height, width, c2);
    in0.setConstant(10.0f);
    in1.setConstant(20.0f);
    in2.setConstant(30.0f);

    ForwardPropagation forward_propagation(batch_size, &neural_network);
    vector<TensorView> input_views = {
        TensorView(in0.data(), { batch_size, height, width, c0 }),
        TensorView(in1.data(), { batch_size, height, width, c1 }),
        TensorView(in2.data(), { batch_size, height, width, c2 })
    };
    neural_network.forward_propagate(input_views, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();

    ASSERT_EQ(output_view.shape.rank, 4);
    EXPECT_EQ(output_view.shape[3], total_channels);

    const float* output_data = output_view.as<type>();

    for (Index h = 0; h < height; ++h)
        for (Index w = 0; w < width; ++w)
        {
            const Index out_base = (h * width + w) * total_channels;
            EXPECT_NEAR(output_data[out_base + 0], 10.0f, 1e-6f);
            EXPECT_NEAR(output_data[out_base + 1], 20.0f, 1e-6f);
            EXPECT_NEAR(output_data[out_base + 2], 20.0f, 1e-6f);
            EXPECT_NEAR(output_data[out_base + 3], 30.0f, 1e-6f);
        }
}


TEST_F(ConcatenationLayerTest, LegacyConcatenateTagLoads)
{
    const Index batch_size = 2;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Concatenation>(input_shape, per_input_channels, "concat"),
                             vector<Index>{ -1, -2 });
    neural_network.compile();

    Tensor4 inputs_a(batch_size, height, width, channels_a);
    Tensor4 inputs_b(batch_size, height, width, channels_b);
    for (Index i = 0; i < inputs_a.size(); ++i)
        inputs_a.data()[i] = float(i);
    for (Index i = 0; i < inputs_b.size(); ++i)
        inputs_b.data()[i] = float(1000 + i);

    auto make_inputs = [&]() {
        return vector<TensorView>{
            TensorView(inputs_a.data(), { batch_size, height, width, channels_a }),
            TensorView(inputs_b.data(), { batch_size, height, width, channels_b })
        };
    };

    ForwardPropagation forward_before(batch_size, &neural_network);
    vector<TensorView> inputs_before = make_inputs();
    neural_network.forward_propagate(inputs_before, forward_before, false);
    const TensorView out_before = forward_before.get_outputs();
    const vector<float> expected(out_before.as<float>(), out_before.as<float>() + out_before.size());

    const string path = (filesystem::temp_directory_path() / "opennn_concat_legacy_tag.json").string();
    neural_network.save(path);

    {
        std::ifstream in(path);
        std::stringstream buffer;
        buffer << in.rdbuf();
        string text = buffer.str();
        const size_t at = text.find("\"Concatenation\"");
        ASSERT_NE(at, string::npos);
        text.replace(at, string("\"Concatenation\"").size(), "\"Concatenate\"");
        std::ofstream out(path);
        out << text;
    }

    NeuralNetwork loaded;
    loaded.load(path);

    const auto* concatenation = dynamic_cast<const Concatenation*>(loaded.get_layer(Index(0)).get());
    ASSERT_NE(concatenation, nullptr);
    EXPECT_EQ(concatenation->get_name(), "Concatenation");
    EXPECT_EQ(concatenation->get_inputs_number(), 2);

    ForwardPropagation forward_after(batch_size, &loaded);
    vector<TensorView> inputs_after = make_inputs();
    loaded.forward_propagate(inputs_after, forward_after, false);
    const TensorView out_after = forward_after.get_outputs();

    ASSERT_EQ(out_after.size(), Index(expected.size()));
    for (Index i = 0; i < out_after.size(); ++i)
        EXPECT_NEAR(out_after.as<float>()[i], expected[size_t(i)], 1.0e-6f);

    error_code file_error;
    filesystem::remove(path, file_error);
    filesystem::remove(filesystem::path(path).replace_extension(".bin"), file_error);
}


TEST_F(ConcatenationLayerTest, ConcatBackwardGradientMatchesNumerical)
{
    const Index samples_number = 4;
    const Index targets_number = 2;

    const Index in_height = 3;
    const Index in_width = 3;
    const Index in_channels = 2;
    const Shape data_input_shape{ in_height, in_width, in_channels };

    const Index branch_a_channels = 2;
    const Index branch_b_channels = 3;

    TabularDataset dataset(samples_number, data_input_shape, { targets_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;

    neural_network.add_layer(make_unique<Convolutional>(data_input_shape,
                                                        Shape{ 1, 1, in_channels, branch_a_channels },
                                                        "Identity", Shape{ 1, 1 }, "Same", true,
                                                        "branch_a"),
                             vector<Index>{ -1 });
    const Index branch_a_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Convolutional>(data_input_shape,
                                                        Shape{ 1, 1, in_channels, branch_b_channels },
                                                        "Identity", Shape{ 1, 1 }, "Same", true,
                                                        "branch_b"),
                             vector<Index>{ -1 });
    const Index branch_b_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Concatenation>(Shape{ in_height, in_width, branch_a_channels },
                                                        vector<Index>{ branch_a_channels, branch_b_channels },
                                                        "concat"),
                             vector<Index>{ branch_a_index, branch_b_index });
    const Index concat_index = neural_network.get_layers_number() - 1;

    neural_network.add_layer(make_unique<Flatten>(neural_network.get_layer(concat_index)->get_output_shape()),
                             vector<Index>{ concat_index });

    neural_network.add_layer(make_unique<opennn::Dense>(neural_network.get_output_shape(),
                                                        Shape{ targets_number }, "Identity"));

    neural_network.compile();

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    EXPECT_LT((gradient - numerical_gradient).array().abs().maxCoeff(), type(1.0e-3));
}
