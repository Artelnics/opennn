#include "tests/pch.h"

#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/standard_networks.h"

using namespace opennn;

TEST(Transformer, ConstructorCreatesNetwork)
{
    Transformer transformer(2, 3, 5, 6, 4, 4, 6, 1);

    EXPECT_EQ(transformer.is_empty(), false);
    EXPECT_EQ(transformer.get_layers_number(), 17);
}

// Attention can only mask padding it can find. Recovering it from the data --
// a padded token is an all-zero row -- holds exactly until something shifts
// that row, and every block here ends in a normalization whose bias does
// precisely that as soon as training moves it off zero. Measured before the
// Embeddings exported their lengths: of the six attention layers, only the two
// reading an Embedding directly could still see the padding. The other four,
// both cross-attentions among them, were attending over it.
//
// The encoder and the decoder are padded differently on purpose. Each
// attention layer has to end up with the lengths of the sequence its keys came
// from, which for cross-attention is the encoder's while its queries are the
// decoder's. One record for the whole forward pass would satisfy neither.
TEST(Transformer, EveryAttentionLayerKnowsWhereItsSourceSequenceEnds)
{
    const Index batch = 2;
    const Index input_sequence_length = 8;
    const Index decoder_sequence_length = 8;
    const Index vocabulary_size = 32;
    const Index embedding_dimension = 16;
    const Index heads_number = 2;
    const Index feed_forward_dimension = 32;
    const Index layers_number = 2;

    Configuration::instance().set(Device::CPU, Type::FP32);

    Transformer transformer(input_sequence_length, decoder_sequence_length,
                            vocabulary_size, vocabulary_size,
                            embedding_dimension, heads_number,
                            feed_forward_dimension, layers_number);
    transformer.set_parameters_random();

    // A normalization starts at gamma = 1, beta = 0, which maps a zero row back
    // to zero and leaves the padding visible after all. Training moves the
    // shift, and that is the state this test needs.
    for (Index i = 0; i < transformer.get_layers_number(); ++i)
    {
        auto& layer = transformer.get_layer(i);
        if (layer->get_name() != "Normalization3d") continue;

        auto views = layer->get_parameter_views();
        if (views.size() > 1) views[1].as_vector().setConstant(0.25f);
    }

    const vector<Index> encoder_lengths{3, 6};
    const vector<Index> decoder_lengths{5, 2};

    MatrixR encoder_ids(batch, input_sequence_length);
    MatrixR decoder_ids(batch, decoder_sequence_length);

    for (Index b = 0; b < batch; ++b)
        for (Index s = 0; s < input_sequence_length; ++s)
        {
            encoder_ids(b, s) = s < encoder_lengths[size_t(b)] ? float(1 + (b * 8 + s) % 20) : 0.0f;
            decoder_ids(b, s) = s < decoder_lengths[size_t(b)] ? float(1 + (b * 8 + s) % 20) : 0.0f;
        }

    ForwardPropagation forward_propagation(batch, &transformer);
    vector<TensorView> inputs{
        TensorView(decoder_ids.data(), {batch, decoder_sequence_length}),
        TensorView(encoder_ids.data(), {batch, input_sequence_length})};

    transformer.forward_propagate(inputs, forward_propagation, false);

    Index attention_layers = 0;

    for (Index i = 0; i < transformer.get_layers_number(); ++i)
    {
        const auto& layer = transformer.get_layer(i);
        if (layer->get_name() != "MultiHeadAttention") continue;

        ++attention_layers;

        const string label = layer->get_label();
        const size_t source_ordinal = forward_propagation.inputs[size_t(i)].size() - 1;

        const vector<Index>* lengths =
            forward_propagation.input_valid_lengths(size_t(i), source_ordinal);

        ASSERT_NE(lengths, nullptr) << label << " cannot tell padding from data";

        // Cross-attention takes its keys from the encoder while its queries
        // come from the decoder, so it is the encoder's lengths it needs.
        const bool reads_decoder = label.starts_with("decoder_self_attention");

        EXPECT_EQ(*lengths, reads_decoder ? decoder_lengths : encoder_lengths)
            << label << " has the wrong sequence's lengths";
    }

    EXPECT_EQ(attention_layers, 3 * layers_number);

    Configuration::instance().set();
}

TEST(Transformer, GeneralConstructor)
{
    const Index input_sequence_length = 5;
    const Index decoder_sequence_length = 4;
    const Index input_vocabulary_size = 100;
    const Index output_vocabulary_size = 120;
    const Index embedding_dimension = 8;
    const Index heads_number = 2;
    const Index feed_forward_dimension = 16;
    const Index layers_number = 1;

    Transformer transformer(input_sequence_length,
                            decoder_sequence_length,
                            input_vocabulary_size,
                            output_vocabulary_size,
                            embedding_dimension,
                            heads_number,
                            feed_forward_dimension,
                            layers_number);

    EXPECT_EQ(transformer.get_layers_number(), 17);

    EXPECT_EQ(transformer.get_layer_index("decoder"), -1);
    EXPECT_EQ(transformer.get_layer_index("input"), -2);

    EXPECT_EQ(transformer.get_layer_index("decoder_tokenizer"), 0);
    EXPECT_EQ(transformer.get_layer_index("decoder_embedding"), 1);
    EXPECT_EQ(transformer.get_layer_index("encoder_tokenizer"), 2);
    EXPECT_EQ(transformer.get_layer_index("encoder_embedding"), 3);

    EXPECT_EQ(transformer.get_layer_index("encoder_self_attention_1"), 4);
    EXPECT_THROW(transformer.get_layer_index("encoder_self_attention_addition_1"), runtime_error);
    EXPECT_EQ(transformer.get_layer_index("encoder_self_attention_normalization_1"), 5);
    EXPECT_EQ(transformer.get_layer_index("encoder_internal_dense_1"), 6);
    EXPECT_EQ(transformer.get_layer_index("encoder_external_dense_1"), 7);
    EXPECT_THROW(transformer.get_layer_index("encoder_dense_addition_1"), runtime_error);
    EXPECT_EQ(transformer.get_layer_index("encoder_dense_normalization_1"), 8);

    EXPECT_EQ(transformer.get_layer_index("decoder_self_attention_1"), 9);
    EXPECT_THROW(transformer.get_layer_index("decoder_self_attention_addition_1"), runtime_error);
    EXPECT_EQ(transformer.get_layer_index("decoder_self_attention_normalization_1"), 10);
    EXPECT_EQ(transformer.get_layer_index("cross_attention_1"), 11);
    EXPECT_THROW(transformer.get_layer_index("cross_attention_addition_1"), runtime_error);
    EXPECT_EQ(transformer.get_layer_index("cross_attention_normalization_1"), 12);
    EXPECT_EQ(transformer.get_layer_index("decoder_internal_dense_1"), 13);
    EXPECT_EQ(transformer.get_layer_index("decoder_external_dense_1"), 14);
    EXPECT_THROW(transformer.get_layer_index("decoder_dense_addition_1"), runtime_error);
    EXPECT_EQ(transformer.get_layer_index("decoder_dense_normalization_1"), 15);

    EXPECT_EQ(transformer.get_layer_index("output_projection"), 16);

    const vector<vector<Index>>& in = transformer.get_source_layers();

    ASSERT_EQ(in.size(), 17);

    EXPECT_EQ(in[0], (vector<Index>{-1}));
    EXPECT_EQ(in[1], (vector<Index>{0}));
    EXPECT_EQ(in[2], (vector<Index>{-2}));
    EXPECT_EQ(in[3], (vector<Index>{2}));

    EXPECT_EQ(in[4], (vector<Index>{3}));
    EXPECT_EQ(in[5], (vector<Index>{3, 4}));
    EXPECT_EQ(in[6], (vector<Index>{5}));
    EXPECT_EQ(in[7], (vector<Index>{6}));
    EXPECT_EQ(in[8], (vector<Index>{5, 7}));

    EXPECT_EQ(in[9],  (vector<Index>{1}));
    EXPECT_EQ(in[10], (vector<Index>{1, 9}));
    EXPECT_EQ(in[11], (vector<Index>{10, 8}));
    EXPECT_EQ(in[12], (vector<Index>{10, 11}));
    EXPECT_EQ(in[13], (vector<Index>{12}));
    EXPECT_EQ(in[14], (vector<Index>{13}));
    EXPECT_EQ(in[15], (vector<Index>{12, 14}));

    EXPECT_EQ(in[16], (vector<Index>{15}));
}

TEST(Transformer, TrainingArenaReusesResidualBranchOutputs)
{
    const Index batch_size = 2;
    Transformer transformer(4, 4, 16, 16, 8, 2, 16, 1);

    const auto specs = transformer.get_forward_specs(batch_size);
    const auto& layers = transformer.get_layers();

    Index chronological_bytes = 0;
    Index maximum_transient_bytes = 0;
    for (size_t i = 0; i < specs.size(); ++i)
        for (size_t j = 0; j < specs[i].size(); ++j)
        {
            if (specs[i][j].shape.empty()) continue;
            const Index bytes = get_aligned_bytes(specs[i][j]);
            if (layers[i]->get_forward_slot_kind(j) == ForwardSlotKind::Transient)
                maximum_transient_bytes =
                    max(maximum_transient_bytes, bytes);
            else
                chronological_bytes += bytes;
        }

    chronological_bytes += maximum_transient_bytes;

    ForwardPropagation forward_propagation(batch_size, &transformer);

    EXPECT_LT(forward_propagation.arena.bytes, chronological_bytes);
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
