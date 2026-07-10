#include "pch.h"

#include "opennn/standard_networks.h"

using namespace opennn;

TEST(Transformer, ConstructorCreatesNetwork)
{
    Transformer transformer(2, 3, 5, 6, 4, 4, 6, 1);

    EXPECT_EQ(transformer.is_empty(), false);
    EXPECT_EQ(transformer.get_layers_number(), 17);
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

    const vector<vector<Index>> out = transformer.get_consumer_layers();

    ASSERT_EQ(out.size(), 17);

    EXPECT_EQ(out[0],  (vector<Index>{1}));
    EXPECT_EQ(out[1],  (vector<Index>{10, 9}));
    EXPECT_EQ(out[2],  (vector<Index>{3}));
    EXPECT_EQ(out[3],  (vector<Index>{5, 4}));
    EXPECT_EQ(out[4],  (vector<Index>{5}));
    EXPECT_EQ(out[5],  (vector<Index>{8, 6}));
    EXPECT_EQ(out[6],  (vector<Index>{7}));
    EXPECT_EQ(out[7],  (vector<Index>{8}));
    EXPECT_EQ(out[8],  (vector<Index>{11}));
    EXPECT_EQ(out[9],  (vector<Index>{10}));
    EXPECT_EQ(out[10], (vector<Index>{12, 11}));
    EXPECT_EQ(out[11], (vector<Index>{12}));
    EXPECT_EQ(out[12], (vector<Index>{15, 13}));
    EXPECT_EQ(out[13], (vector<Index>{14}));
    EXPECT_EQ(out[14], (vector<Index>{15}));
    EXPECT_EQ(out[15], (vector<Index>{16}));
    EXPECT_EQ(out[16], (vector<Index>{-1}));
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
