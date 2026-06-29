#include "pch.h"
#include "../opennn/tensor_types.h"
#include "../opennn/concatenation_operator.h"
#include "../opennn/concatenation_layer.h"
#include "../opennn/forward_propagation.h"
#include "../opennn/back_propagation.h"
#include "../opennn/neural_network.h"
#include "../opennn/loss.h"

using namespace opennn;


static ConcatenationOperator make_operator(Index, Index, const vector<Index>& per_input_channels)
{
    ConcatenationOperator op;
    op.input_channels = per_input_channels;
    op.input_slots = {0};
    op.output_slots = {1};
    op.output_delta_slots = {0};
    op.input_delta_slots.resize(per_input_channels.size());
    iota(op.input_delta_slots.begin(), op.input_delta_slots.end(), size_t(1));
    return op;
}


TEST(ConcatenationOperatoreratorTest, SetStoresGeometry)
{
    ConcatenationOperator op;
    op.input_channels = {2, 3};

    ASSERT_EQ(op.input_channels.size(), size_t(2));
    EXPECT_EQ(op.input_channels[0], 2);
    EXPECT_EQ(op.input_channels[1], 3);
}


TEST(ConcatenationOperatoreratorTest, ForwardConcatenatesChannels)
{
    const Index batch_size = 2;
    const Index height = 1;
    const Index width = 1;
    const vector<Index> per_input_channels = {2, 3};
    const Index total_channels = 5;

    ConcatenationOperator op = make_operator(height, width, per_input_channels);

    Tensor4 input_a(batch_size, height, width, 2);
    Tensor4 input_b(batch_size, height, width, 3);

    for (Index i = 0; i < input_a.size(); ++i) input_a.data()[i] = float(i + 1);
    for (Index i = 0; i < input_b.size(); ++i) input_b.data()[i] = float(100 + i);

    Tensor4 output(batch_size, height, width, total_channels);
    output.setConstant(-1.0f);

    ForwardPropagation fp;
    fp.input_views.resize(1);
    fp.forward_slots.resize(1);

    fp.input_views[0] = {
        TensorView(input_a.data(), {batch_size, height, width, 2}),
        TensorView(input_b.data(), {batch_size, height, width, 3})
    };

    fp.forward_slots[0].resize(2);
    fp.forward_slots[0][1] = TensorView(output.data(), {batch_size, height, width, total_channels});

    op.forward_propagate(fp, 0, false);

    EXPECT_NEAR(output.data()[0], input_a.data()[0], 1e-6f);
    EXPECT_NEAR(output.data()[1], input_a.data()[1], 1e-6f);
    EXPECT_NEAR(output.data()[2], input_b.data()[0], 1e-6f);
    EXPECT_NEAR(output.data()[3], input_b.data()[1], 1e-6f);
    EXPECT_NEAR(output.data()[4], input_b.data()[2], 1e-6f);

    EXPECT_NEAR(output.data()[5], input_a.data()[2], 1e-6f);
    EXPECT_NEAR(output.data()[6], input_a.data()[3], 1e-6f);
    EXPECT_NEAR(output.data()[7], input_b.data()[3], 1e-6f);
    EXPECT_NEAR(output.data()[8], input_b.data()[4], 1e-6f);
    EXPECT_NEAR(output.data()[9], input_b.data()[5], 1e-6f);
}


TEST(ConcatenationOperatoreratorTest, ForwardSpatialLayout)
{
    const Index batch_size = 1;
    const Index height = 2;
    const Index width = 2;
    const vector<Index> per_input_channels = {1, 1};
    const Index total_channels = 2;

    ConcatenationOperator op = make_operator(height, width, per_input_channels);

    Tensor4 input_a(batch_size, height, width, 1);
    Tensor4 input_b(batch_size, height, width, 1);

    for (Index i = 0; i < input_a.size(); ++i) input_a.data()[i] = float(i);
    for (Index i = 0; i < input_b.size(); ++i) input_b.data()[i] = float(10 + i);

    Tensor4 output(batch_size, height, width, total_channels);
    output.setConstant(-1.0f);

    ForwardPropagation fp;
    fp.input_views.resize(1);
    fp.forward_slots.resize(1);

    fp.input_views[0] = {
        TensorView(input_a.data(), {batch_size, height, width, 1}),
        TensorView(input_b.data(), {batch_size, height, width, 1})
    };

    fp.forward_slots[0].resize(2);
    fp.forward_slots[0][1] = TensorView(output.data(), {batch_size, height, width, total_channels});

    op.forward_propagate(fp, 0, false);

    for (Index spatial = 0; spatial < height * width; ++spatial)
    {
        EXPECT_NEAR(output.data()[spatial * total_channels + 0], input_a.data()[spatial], 1e-6f);
        EXPECT_NEAR(output.data()[spatial * total_channels + 1], input_b.data()[spatial], 1e-6f);
    }
}


TEST(ConcatenationOperatoreratorTest, BackwardSplitsDelta)
{
    const Index batch_size = 2;
    const Index height = 1;
    const Index width = 1;
    const vector<Index> per_input_channels = {2, 3};
    const Index total_channels = 5;

    ConcatenationOperator op = make_operator(height, width, per_input_channels);

    Tensor4 output_delta(batch_size, height, width, total_channels);
    for (Index i = 0; i < output_delta.size(); ++i) output_delta.data()[i] = float(i + 1);

    Tensor4 delta_a(batch_size, height, width, 2);
    Tensor4 delta_b(batch_size, height, width, 3);
    delta_a.setConstant(0.0f);
    delta_b.setConstant(0.0f);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Concatenation>(Shape{height, width, total_channels}, per_input_channels, "concat"),
                             {-1, -2});
    neural_network.compile();

    Loss loss(&neural_network, nullptr);

    BackPropagation bp(batch_size, &loss);

    bp.layer_output_deltas[0] = TensorView(output_delta.data(), {batch_size, height, width, total_channels});

    bp.backward_slots[0].assign(3, TensorView());
    bp.backward_slots[0][1] = TensorView(delta_a.data(), {batch_size, height, width, 2});
    bp.backward_slots[0][2] = TensorView(delta_b.data(), {batch_size, height, width, 3});

    ForwardPropagation fp;
    op.back_propagate(fp, bp, 0);

    EXPECT_NEAR(delta_a.data()[0], output_delta.data()[0], 1e-6f);
    EXPECT_NEAR(delta_a.data()[1], output_delta.data()[1], 1e-6f);
    EXPECT_NEAR(delta_b.data()[0], output_delta.data()[2], 1e-6f);
    EXPECT_NEAR(delta_b.data()[1], output_delta.data()[3], 1e-6f);
    EXPECT_NEAR(delta_b.data()[2], output_delta.data()[4], 1e-6f);

    EXPECT_NEAR(delta_a.data()[2], output_delta.data()[5], 1e-6f);
    EXPECT_NEAR(delta_a.data()[3], output_delta.data()[6], 1e-6f);
    EXPECT_NEAR(delta_b.data()[3], output_delta.data()[7], 1e-6f);
    EXPECT_NEAR(delta_b.data()[4], output_delta.data()[8], 1e-6f);
    EXPECT_NEAR(delta_b.data()[5], output_delta.data()[9], 1e-6f);
}


TEST(ConcatenationOperatoreratorTest, BackwardSkipsEmptyInputDelta)
{
    const Index batch_size = 1;
    const Index height = 1;
    const Index width = 1;
    const vector<Index> per_input_channels = {2, 2};
    const Index total_channels = 4;

    ConcatenationOperator op = make_operator(height, width, per_input_channels);

    Tensor4 output_delta(batch_size, height, width, total_channels);
    for (Index i = 0; i < output_delta.size(); ++i) output_delta.data()[i] = float(i + 1);

    Tensor4 delta_b(batch_size, height, width, 2);
    delta_b.setConstant(0.0f);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Concatenation>(Shape{height, width, total_channels}, per_input_channels, "concat"),
                             {-1, -2});
    neural_network.compile();

    Loss loss(&neural_network, nullptr);

    BackPropagation bp(batch_size, &loss);

    bp.layer_output_deltas[0] = TensorView(output_delta.data(), {batch_size, height, width, total_channels});

    bp.backward_slots[0].assign(3, TensorView());
    bp.backward_slots[0][1] = TensorView();
    bp.backward_slots[0][2] = TensorView(delta_b.data(), {batch_size, height, width, 2});

    ForwardPropagation fp;
    op.back_propagate(fp, bp, 0);

    EXPECT_NEAR(delta_b.data()[0], output_delta.data()[2], 1e-6f);
    EXPECT_NEAR(delta_b.data()[1], output_delta.data()[3], 1e-6f);
}
