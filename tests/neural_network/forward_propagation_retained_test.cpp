// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

#include "tests/pch.h"

#include "opennn/core/configuration.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/random_utilities.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/neural_network/operators/tokenizer_operator.h"

using namespace opennn;

namespace
{

struct Seq2SeqLayout
{
    Index encoder_embedding = -1;
    Index encoder_last = -1;
    Index decoder_embedding = -1;
    Index decoder_first = -1;
    Index output_projection = -1;
};

Seq2SeqLayout find_layout(NeuralNetwork& network)
{
    Seq2SeqLayout layout;
    layout.encoder_embedding = network.get_layer_index("encoder_embedding");
    layout.decoder_embedding = network.get_layer_index("decoder_embedding");

    const auto& layers = network.get_layers();
    Index cross_attention = -1;
    for (Index i = 0; i < ssize(layers); ++i)
        if (layers[size_t(i)]->get_label().starts_with("cross_attention_"))
        {
            cross_attention = i;
            break;
        }

    layout.encoder_last =
        network.get_source_layers()[size_t(cross_attention)][1];
    layout.decoder_first = layout.encoder_last + 1;
    layout.output_projection = ssize(layers) - 1;
    return layout;
}

pair<ptrdiff_t, ptrdiff_t> slot_range(ForwardPropagation& propagation,
                                      const TensorView& view)
{
    const char* base = propagation.arena.as<char>();
    const ptrdiff_t low = static_cast<const char*>(view.get_data()) - base;
    return {low, low + view.byte_size()};
}

void expect_identical_plans(ForwardPropagation& left,
                            ForwardPropagation& right)
{
    ASSERT_EQ(left.arena.byte_size(), right.arena.byte_size());
    ASSERT_EQ(left.slots.size(), right.slots.size());

    for (size_t i = 0; i < left.slots.size(); ++i)
    {
        ASSERT_EQ(left.slots[i].size(), right.slots[i].size());

        for (size_t j = 0; j < left.slots[i].size(); ++j)
        {
            const TensorView& left_slot = left.slots[i][j];
            const TensorView& right_slot = right.slots[i][j];
            EXPECT_EQ(left_slot.byte_size(), right_slot.byte_size());
            if (!left_slot.get_data() || !right_slot.get_data()) continue;
            EXPECT_EQ(slot_range(left, left_slot),
                      slot_range(right, right_slot));
        }
    }
}

}

TEST(ForwardPropagationRetainedOutputsTest,
     RetainedEncoderOutputDoesNotOverlapDecoderSlots)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    Transformer network(4, 5, 12, 14, 8, 2, 16, 1);
    const Seq2SeqLayout layout = find_layout(network);

    vector<Index> rerun_layers = {layout.decoder_embedding};
    for (Index i = layout.decoder_first; i <= layout.output_projection; ++i)
        rerun_layers.push_back(i);

    InferenceShapePolicy policy;
    policy.retained_output_layers = {layout.encoder_last};
    ForwardPropagation propagation(
        1, &network, ForwardPropagationMode::Inference, policy);

    const TensorView& retained =
        propagation.slots[size_t(layout.encoder_last)].back();
    ASSERT_GT(retained.byte_size(), 0);
    const auto [retained_low, retained_high] = slot_range(propagation, retained);

    for (const Index i : rerun_layers)
        for (const TensorView& slot : propagation.slots[size_t(i)])
        {
            if (!slot.get_data() || slot.byte_size() == 0) continue;
            const auto [low, high] = slot_range(propagation, slot);
            EXPECT_TRUE(high <= retained_low || low >= retained_high)
                << "layer " << i << " slot [" << low << ", " << high
                << ") overlaps retained [" << retained_low << ", "
                << retained_high << ")";
        }

    Configuration::instance().set();
}

TEST(ForwardPropagationRetainedOutputsTest, EmptyPolicyKeepsDefaultPlan)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    Transformer network(5, 4, 8, 8, 8, 2, 16, 1);
    const Seq2SeqLayout layout = find_layout(network);

    ForwardPropagation default_propagation(
        1, &network, ForwardPropagationMode::Inference);
    ForwardPropagation explicit_propagation(
        1, &network, ForwardPropagationMode::Inference, InferenceShapePolicy{});

    expect_identical_plans(default_propagation, explicit_propagation);
    EXPECT_FALSE(default_propagation.needs_position_staging());

    InferenceShapePolicy retained_policy;
    retained_policy.retained_output_layers = {layout.encoder_last};
    ForwardPropagation retained_propagation(
        1, &network, ForwardPropagationMode::Inference, retained_policy);

    const TensorView& retained =
        retained_propagation.slots[size_t(layout.encoder_last)].back();
    EXPECT_LE(retained_propagation.arena.byte_size(),
              default_propagation.arena.byte_size()
                  + get_aligned_bytes(retained.byte_size()));

    Configuration::instance().set();
}

TEST(ForwardPropagationRetainedOutputsTest,
     DecoderOnlyPlanUnchangedByRetainedField)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    Qwen3 network(64, 32, 32, 1, 4, 2, 8, 64);

    ForwardPropagation default_propagation(
        1, &network, ForwardPropagationMode::Inference);
    ForwardPropagation explicit_propagation(
        1, &network, ForwardPropagationMode::Inference, InferenceShapePolicy{});

    expect_identical_plans(default_propagation, explicit_propagation);
    EXPECT_TRUE(default_propagation.needs_position_staging());

    Configuration::instance().set();
}

TEST(ForwardPropagationRetainedOutputsTest, RetainedOutputRejectsInvalidLayer)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    Transformer network(5, 4, 8, 8, 8, 2, 16, 1);

    InferenceShapePolicy policy;
    policy.retained_output_layers = {network.get_layers_number()};
    EXPECT_THROW(ForwardPropagation(
                     1, &network, ForwardPropagationMode::Inference, policy),
                 runtime_error);

    policy.retained_output_layers = {0};
    EXPECT_THROW(ForwardPropagation(
                     1, &network, ForwardPropagationMode::Training, policy),
                 runtime_error);

    Configuration::instance().set();
}

TEST(ForwardPropagationRetainedOutputsTest,
     IncrementalSequenceToSequenceMatchesFullForward)
{
    set_seed(42);
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index input_length = 4;
    const Index decoder_length = 5;
    const Index input_vocabulary = 12;
    const Index target_vocabulary = 14;

    Transformer network(input_length, decoder_length,
                        input_vocabulary, target_vocabulary, 8, 2, 16, 1);
    network.set_parameters_random();
    const Seq2SeqLayout layout = find_layout(network);

    Tensor3 encoder_inputs(1, input_length, 1);
    for (Index i = 0; i < input_length; ++i)
        encoder_inputs(0, i, 0) = float(1 + (i * 3) % (input_vocabulary - 1));

    Tensor3 decoder_inputs(1, decoder_length, 1);
    decoder_inputs.setZero();
    decoder_inputs(0, 0, 0) = float(TokenizerOperator::START_INDEX);

    InferenceShapePolicy policy;
    policy.retained_output_layers = {layout.encoder_last};
    ForwardPropagation propagation(
        1, &network, ForwardPropagationMode::Inference, policy);

    Tensor2 target(1, decoder_length);
    target.setZero();
    target(0, 0) = float(TokenizerOperator::START_INDEX);

    Tensor2 source(1, input_length);
    for (Index i = 0; i < input_length; ++i)
        source(0, i) = encoder_inputs(0, i, 0);

    const vector<TensorView> inputs = {
        TensorView(target.data(), {1, decoder_length}),
        TensorView(source.data(), {1, input_length})};

    network.forward_propagate(inputs, propagation, false,
                              layout.encoder_embedding, layout.encoder_last);

    for (Index position = 1; position < decoder_length; ++position)
    {
        const Tensor3 reference =
            network.calculate_outputs(decoder_inputs, encoder_inputs);

        network.forward_propagate(inputs, propagation, false,
                                  layout.decoder_embedding,
                                  layout.decoder_embedding);
        network.forward_propagate(inputs, propagation, false,
                                  layout.decoder_first,
                                  layout.output_projection);

        const TensorView outputs = propagation.get_outputs();
        const float* incremental =
            outputs.as<float>() + (position - 1) * target_vocabulary;

        Index best = 0;
        for (Index v = 0; v < target_vocabulary; ++v)
        {
            EXPECT_NEAR(incremental[v], reference(0, position - 1, v), 1.0e-5)
                << "position " << position << " vocabulary index " << v;
            if (reference(0, position - 1, v) > reference(0, position - 1, best))
                best = v;
        }

        decoder_inputs(0, position, 0) = float(best);
        target(0, position) = float(best);
    }

    Configuration::instance().set();
}

TEST(ForwardPropagationRetainedOutputsTest,
     SequenceCapacityDoesNotRequireCompactFinalOutput)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    Qwen3 network(64, 32, 32, 1, 4, 2, 8, 64);

    InferenceShapePolicy policy;
    policy.sequence_capacity = 4;

    ForwardPropagation propagation(
        1, &network, ForwardPropagationMode::Inference, policy);

    EXPECT_EQ(propagation.get_sequence_capacity(), 4);
    EXPECT_EQ(propagation.get_final_output_capacity(), 4);

    propagation.past_length = 3;
    propagation.valid_lengths.assign(1, vector<Index>{2});
    propagation.set(1, &network, nullptr,
                    ForwardPropagationMode::Inference, policy);

    EXPECT_EQ(propagation.past_length, 0);

    // Reset leaves a record per layer, every one of them empty: the lengths of
    // the batch just processed must not be read as the next batch's.
    EXPECT_EQ(Index(propagation.valid_lengths.size()), network.get_layers_number());
    EXPECT_TRUE(ranges::all_of(propagation.valid_lengths,
                               [](const vector<Index>& lengths) { return lengths.empty(); }));

    Configuration::instance().set();
}

TEST(ForwardPropagationRetainedOutputsTest, OutputWindowMatchesFullForwardForEverySample)
{
    set_seed(7);
    Configuration::instance().set(Device::CPU, Type::FP32);

    const Index batch_size = 3;
    const Index input_length = 4;
    const Index decoder_length = 5;
    const Index input_vocabulary = 12;
    const Index target_vocabulary = 14;

    Transformer network(input_length, decoder_length,
                        input_vocabulary, target_vocabulary, 8, 2, 16, 1);
    network.set_parameters_random();

    Tensor2 target(batch_size, decoder_length);
    Tensor2 source(batch_size, input_length);
    Tensor3 decoder_inputs(batch_size, decoder_length, 1);
    Tensor3 encoder_inputs(batch_size, input_length, 1);

    for (Index sample = 0; sample < batch_size; ++sample)
    {
        for (Index i = 0; i < decoder_length; ++i)
        {
            const float token = float(1 + (sample * 5 + i * 3) % (target_vocabulary - 1));
            target(sample, i) = token;
            decoder_inputs(sample, i, 0) = token;
        }
        for (Index i = 0; i < input_length; ++i)
        {
            const float token = float(1 + (sample * 7 + i * 2) % (input_vocabulary - 1));
            source(sample, i) = token;
            encoder_inputs(sample, i, 0) = token;
        }
    }

    const Tensor3 reference = network.calculate_outputs(decoder_inputs, encoder_inputs);

    InferenceShapePolicy policy;
    policy.sequence_capacity = decoder_length;
    policy.final_output_capacity = 1;

    ForwardPropagation windowed(
        batch_size, &network, ForwardPropagationMode::Inference, policy);

    const vector<TensorView> inputs = {
        TensorView(target.data(), {batch_size, decoder_length}),
        TensorView(source.data(), {batch_size, input_length})};

    network.forward_propagate(inputs, windowed, false);

    const TensorView outputs = windowed.get_outputs();
    ASSERT_EQ(outputs.size(), batch_size * target_vocabulary);

    for (Index sample = 0; sample < batch_size; ++sample)
        for (Index v = 0; v < target_vocabulary; ++v)
            EXPECT_NEAR(outputs.as<float>()[sample * target_vocabulary + v],
                        reference(sample, decoder_length - 1, v), 1.0e-5)
                << "sample " << sample << " vocabulary index " << v;

    Configuration::instance().set();
}
