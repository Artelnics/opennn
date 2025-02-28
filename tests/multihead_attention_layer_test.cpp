#include "pch.h"

#include "../opennn/multihead_attention_layer.h"

TEST(MultiheadAttentionLayer, DefaultConstructor) {
    MultiheadAttentionLayer multihead_attention_layer;

    EXPECT_EQ(multihead_attention_layer.get_heads_number(), 0);
    EXPECT_EQ(multihead_attention_layer.get_input_size(), 0);
    EXPECT_EQ(multihead_attention_layer.get_context_size(), 0);
    EXPECT_EQ(multihead_attention_layer.get_depth(), 0);
    EXPECT_EQ(multihead_attention_layer.get_weights_depth(), 0);

    EXPECT_EQ(multihead_attention_layer.get_input_dimensions(), dimensions{0});
    EXPECT_EQ(multihead_attention_layer.get_output_dimensions(), dimensions({multihead_attention_layer.get_input_size(), multihead_attention_layer.get_depth()}));
}


TEST(MultiheadAttentionLayer, GeneralConstructor)
{
    const Index heads_number = 8;
    const Index input_size = 64;
    const Index context_size = 128;
    const Index depth = 256;
    const bool use_causal_mask = true;

    MultiheadAttentionLayer multihead_attention_layer(input_size,context_size,depth,heads_number,use_causal_mask);

    EXPECT_EQ(multihead_attention_layer.get_heads_number(), heads_number);
    EXPECT_EQ(multihead_attention_layer.get_input_size(), input_size);
    EXPECT_EQ(multihead_attention_layer.get_context_size(), context_size);
    EXPECT_EQ(multihead_attention_layer.get_depth(), depth);

    EXPECT_EQ(multihead_attention_layer.get_input_dimensions(), dimensions{input_size});
    EXPECT_EQ(multihead_attention_layer.get_output_dimensions(), dimensions({input_size, depth}));
}


TEST(MultiheadAttentionLayer, ForwardPropagate)
{
    const Index batch_samples_number = 2;
    const Index input_size = 4;
    const Index context_size = 4;
    const Index depth = 8;
    const Index heads_number = 2;
    const bool is_training = true;

    const dimensions input_dimensions = {batch_samples_number, input_size};
    const dimensions context_dimensions = {batch_samples_number, context_size};

    MultiheadAttentionLayer multihead_attention_layer(input_size, context_size, depth, heads_number);

    unique_ptr<LayerForwardPropagation> multihead_attention_layer_forward_propagation
        = make_unique<MultiheadAttentionLayerForwardPropagation>(batch_samples_number, &multihead_attention_layer);

    Tensor<type, 3> input(batch_samples_number, input_size, depth);
    Tensor<type, 3> context(batch_samples_number, context_size, depth);

    input.setRandom();
    context.setRandom();

    pair<type*, dimensions> input_pair = {input.data(), {batch_samples_number, input_size, depth}};
    pair<type*, dimensions> context_pair = {context.data(), {batch_samples_number, context_size, depth}};

    multihead_attention_layer.forward_propagate({input_pair, context_pair},
                                                multihead_attention_layer_forward_propagation,
                                                is_training);

    pair<type*, dimensions> output_pair = multihead_attention_layer_forward_propagation->get_outputs_pair();

    EXPECT_EQ(output_pair.second[0], batch_samples_number);
    EXPECT_EQ(output_pair.second[1], input_size);
    EXPECT_EQ(output_pair.second[2], depth);
}

