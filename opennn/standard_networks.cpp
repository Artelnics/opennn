//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A N D A R D   N E T W O R K S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "standard_networks.h"
#include "activation_layer.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"
#include "dense_layer.h"
#include "bounding_layer.h"
#include "recurrent_layer.h"
#include "embedding_layer.h"
#include "convolutional_layer.h"
#include "detection_layer.h"
#include "pooling_layer.h"
#include "pooling_layer_3d.h"
#include "non_max_suppression_layer.h"
#include "flatten_layer.h"
#include "addition_layer.h"
#include "normalization_layer_3d.h"
#include "multihead_attention_layer.h"

namespace opennn
{

ApproximationNetwork::ApproximationNetwork(const Shape& input_shape,
                                           const Shape& complexity_dimensions,
                                           const Shape& output_shape) : NeuralNetwork()
{
    const Index complexity_size = complexity_dimensions.rank;

    add_layer(make_unique<Scaling>(input_shape));

    for (Index i = 0; i < complexity_size; ++i)
        add_layer(make_unique<Dense>(get_output_shape(),
                                       Shape{ complexity_dimensions[i] },
                                       "Tanh",
                                       false,
                                       format("dense2d_layer_{}", i + 1)));

    add_layer(make_unique<Dense>(get_output_shape(),
                                   output_shape,
                                   "Identity",
                                   false,
                                   "approximation_layer"));

    add_layer(make_unique<Unscaling>(output_shape));

    add_layer(make_unique<Bounding>(output_shape));

    compile();
    set_parameters_glorot();
}

ClassificationNetwork::ClassificationNetwork(const Shape& input_shape,
                                             const Shape& complexity_dimensions,
                                             const Shape& output_shape) : NeuralNetwork()
{
    const Index complexity_size = complexity_dimensions.rank;

    add_layer(make_unique<Scaling>(input_shape));

    for (Index i = 0; i < complexity_size; ++i)
        add_layer(make_unique<Dense>(get_output_shape(),
                                       Shape{complexity_dimensions[i]},
                                       "Tanh",
                                       false,
                                       format("dense2d_layer_{}", i + 1)));

    add_layer(make_unique<Dense>(get_output_shape(),
                                   output_shape,
                                   output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                                   false,
                                   "classification_layer"));

    compile();
    set_parameters_random();
}

ForecastingNetwork::ForecastingNetwork(const Shape& input_shape,
                                       const Shape& complexity_dimensions,
                                       const Shape& output_shape) : NeuralNetwork()
{
    clear();

    // Scaling supports rank 2 input ({time_steps, input_features}); the
    // optimizer auto-fills its descriptives from TimeSeriesDataset.
    add_layer(make_unique<Scaling>(input_shape));

    add_layer(make_unique<Recurrent>(get_output_shape(),
                                     complexity_dimensions,
                                     "Tanh",
                                     "recurrent_layer"));

    add_layer(make_unique<Dense>(get_output_shape(),
                                 output_shape,
                                 "Identity",
                                 false,
                                 "forecasting_layer"));

    // Unscaling rank-1 output back to original target range. Optimizer fills
    // descriptives from the dataset's target columns.
    add_layer(make_unique<Unscaling>(output_shape));

    compile();
    set_parameters_glorot();
}

AutoAssociationNetwork::AutoAssociationNetwork(const Shape& input_shape,
                                               const Shape& complexity_dimensions,
                                               const Shape& output_shape) : NeuralNetwork()
{
    add_layer(make_unique<Scaling>(input_shape));

    const Shape mapping_shape{ 10 };
    const Shape bottleneck_shape{ complexity_dimensions[0] };

    add_layer(make_unique<Dense>(input_shape,
                                 mapping_shape,
                                 "Tanh",
                                 false,
                                 "mapping_layer"));

    add_layer(make_unique<Dense>(mapping_shape,
                                 bottleneck_shape,
                                 "Identity",
                                 false,
                                 "bottleneck_layer"));

    add_layer(make_unique<Dense>(bottleneck_shape,
                                 mapping_shape,
                                 "Tanh",
                                 false,
                                 "demapping_layer"));

    add_layer(make_unique<Dense>(mapping_shape,
                                 Shape{ output_shape },
                                 "Identity",
                                 false,
                                 "output_layer"));

    add_layer(make_unique<Unscaling>(output_shape));

    compile();
    set_parameters_random();
}

#ifndef OPENNN_NO_VISION

ImageClassificationNetwork::ImageClassificationNetwork(const Shape& input_shape,
                                                       const Shape& complexity_dimensions,
                                                       const Shape& output_shape) : NeuralNetwork()
{
    if (input_shape.rank != 3)
        throw runtime_error("Input shape size is not 3.");

    auto scaling_layer = make_unique<Scaling>(input_shape);
    scaling_layer->set_scalers("ImageMinMax");
    add_layer(move(scaling_layer));

    const Index complexity_size = complexity_dimensions.rank;

    for (Index i = 0; i < complexity_size; ++i)
    {
        const Shape kernel_shape = { 3, 3, get_output_shape()[2], complexity_dimensions[i] };
        const Shape stride_shape = { 1, 1 };

        add_layer(make_unique<Convolutional>(get_output_shape(),
                                             kernel_shape,
                                             "ReLU",
                                             stride_shape,
                                             "Same",
                                             false,
                                             format("convolutional_layer_{}", i + 1)));

        const Shape pool_dimensions = { 2, 2 };
        const Shape pooling_stride_shape = { 2, 2 };
        const Shape padding_dimensions = { 0, 0 };

        add_layer(make_unique<Pooling>(get_output_shape(),
                                       pool_dimensions,
                                       pooling_stride_shape,
                                       padding_dimensions,
                                       "MaxPooling",
                                       format("pooling_layer_{}", i + 1)));
    }

    add_layer(make_unique<Flatten>(get_output_shape()));

    const Index flatten_size = get_output_shape()[0];
    const Shape hidden_shape = { min(flatten_size, Index(128)) };

    add_layer(make_unique<Dense>(get_output_shape(),
                                   hidden_shape,
                                   "ReLU",
                                   false,                     // batch_normalization
                                   "dense_2d_layer_1"));

    add_layer(make_unique<Dense>(get_output_shape(),
                                   output_shape,
                                   output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                                   false,
                                   "classification_layer"));

    compile();
    set_parameters_random();
}

ResNet::ResNet(const Shape& input_shape,
               const vector<Index>& blocks_per_stage,
               const Shape& initial_filters,
               const Shape& output_shape,
               bool use_bottleneck) : NeuralNetwork()
{
    if (input_shape.rank != 3)
        throw runtime_error("ResNet: input shape must be rank 3 (H, W, C).");
    if (Index(blocks_per_stage.size()) != Index(initial_filters.rank))
        throw runtime_error("ResNet: blocks_per_stage and initial_filters must have the same size.");
    if (blocks_per_stage.empty())
        throw runtime_error("ResNet: at least one stage is required.");

    // Channel expansion of a bottleneck block: the third (1x1) conv expands
    // the inner channel count back up by this factor. Standard ResNet-V1 uses 4.
    constexpr Index bottleneck_expansion = 4;

    // Convolutional with BatchNorm fused in (the 6th ctor flag). Explicit
    // input wiring lets us run multiple branches off the same index for skip
    // connections. Returns the new layer's global index.
    auto add_conv = [&](Index input_index,
                        const Shape& kernel_shape, const char* activation,
                        const Shape& stride, const string& name) -> Index {
        add_layer(make_unique<Convolutional>(
                      get_layer(input_index)->get_output_shape(),
                      kernel_shape, activation, stride, "Same",
                      /*batch_normalization=*/true, name),
                  {input_index});
        return get_layers_number() - 1;
    };

    // Add the skip path: identity when the main branch preserves shape,
    // otherwise a 1x1 projection conv (with BN) at the same stride.
    auto add_skip = [&](Index input_index, Index in_channels, Index out_channels,
                        Index stride, const string& prefix) -> Index {
        if (stride == 1 && in_channels == out_channels)
            return input_index;
        return add_conv(input_index,
                        Shape{1, 1, in_channels, out_channels}, "Identity",
                        Shape{stride, stride}, prefix + "_skip");
    };

    // Basic block (ResNet-18/34): two 3x3 convs, optional skip projection,
    // post-add ReLU via a standalone Activation layer.
    auto add_basic_block = [&](Index input_index, size_t stage, Index block,
                               Index filters) -> Index {
        const Shape in_shape  = get_layer(input_index)->get_output_shape();
        const Index in_chan   = in_shape[2];
        const Index stride    = (stage > 0 && block == 0) ? 2 : 1;
        const string prefix   = format("s{}b{}", stage, block);

        Index main_index = add_conv(input_index,
            Shape{3, 3, in_chan, filters}, "ReLU",
            Shape{stride, stride}, prefix + "_conv1");
        main_index = add_conv(main_index,
            Shape{3, 3, filters, filters}, "Identity",
            Shape{1, 1}, prefix + "_conv2");

        const Index skip_index = add_skip(input_index, in_chan, filters,
                                          stride, prefix);

        add_layer(make_unique<Addition>(get_layer(main_index)->get_output_shape(),
                                        prefix + "_add"),
                  {main_index, skip_index});
        const Index add_index = get_layers_number() - 1;

        add_layer(make_unique<Activation>(get_layer(add_index)->get_output_shape(),
                                          "ReLU", prefix + "_relu"),
                  {add_index});
        return get_layers_number() - 1;
    };

    // Bottleneck block (ResNet-50/101/152): 1x1 → 3x3 → 1x1 with 4× expansion
    // and post-add ReLU via a standalone Activation layer. `filters` is the
    // inner (compressed) channel count; output channels are filters *
    // bottleneck_expansion.
    auto add_bottleneck_block = [&](Index input_index, size_t stage, Index block,
                                    Index filters) -> Index {
        const Shape in_shape  = get_layer(input_index)->get_output_shape();
        const Index in_chan   = in_shape[2];
        const Index out_chan  = filters * bottleneck_expansion;
        const Index stride    = (stage > 0 && block == 0) ? 2 : 1;
        const string prefix   = format("s{}b{}", stage, block);

        Index main_index = add_conv(input_index,
            Shape{1, 1, in_chan, filters}, "ReLU",
            Shape{1, 1}, prefix + "_conv1");
        main_index = add_conv(main_index,
            Shape{3, 3, filters, filters}, "ReLU",
            Shape{stride, stride}, prefix + "_conv2");
        main_index = add_conv(main_index,
            Shape{1, 1, filters, out_chan}, "Identity",
            Shape{1, 1}, prefix + "_conv3");

        const Index skip_index = add_skip(input_index, in_chan, out_chan,
                                          stride, prefix);

        add_layer(make_unique<Addition>(get_layer(main_index)->get_output_shape(),
                                        prefix + "_add"),
                  {main_index, skip_index});
        const Index add_index = get_layers_number() - 1;

        add_layer(make_unique<Activation>(get_layer(add_index)->get_output_shape(),
                                          "ReLU", prefix + "_relu"),
                  {add_index});
        return get_layers_number() - 1;
    };

    // --- Network assembly ---

    add_layer(make_unique<Scaling>(input_shape));

    // Stem: 7x7 conv stride 2 + 3x3 MaxPool stride 2. Aggressive downsampling
    // turns a 224x224 input into 56x56 before the first stage.
    Index last_index = add_conv(0,
        Shape{7, 7, input_shape[2], initial_filters[0]}, "ReLU",
        Shape{2, 2}, "stem_conv");

    add_layer(make_unique<Pooling>(get_layer(last_index)->get_output_shape(),
                                   Shape{3, 3}, Shape{2, 2}, Shape{1, 1},
                                   "MaxPooling", "stem_pool"),
              {last_index});
    last_index = get_layers_number() - 1;

    // Residual stages.
    for (size_t stage = 0; stage < blocks_per_stage.size(); ++stage)
        for (Index block = 0; block < blocks_per_stage[stage]; ++block)
            last_index = use_bottleneck
                ? add_bottleneck_block(last_index, stage, block, initial_filters[stage])
                : add_basic_block(last_index, stage, block, initial_filters[stage]);

    // Head: global average pool → flatten → softmax classifier.
    const Shape pre_pool = get_layer(last_index)->get_output_shape();
    add_layer(make_unique<Pooling>(pre_pool,
                                   Shape{pre_pool[0], pre_pool[1]},
                                   Shape{1, 1}, Shape{0, 0},
                                   "AveragePooling", "global_avg_pool"),
              {last_index});
    last_index = get_layers_number() - 1;

    add_layer(make_unique<Flatten>(get_layer(last_index)->get_output_shape()),
              {last_index});
    last_index = get_layers_number() - 1;

    add_layer(make_unique<Dense>(get_layer(last_index)->get_output_shape(),
                                 output_shape, "Softmax", false, "classifier"),
              {last_index});

    compile();
    set_parameters_random();
}

YoloNetwork::YoloNetwork(const Shape& input_shape,
                         Index classes_number,
                         const vector<array<float, 2>>& anchors,
                         Index grid_size) : NeuralNetwork()
{
    if (input_shape.rank != 3)
        throw runtime_error("YoloNetwork: input shape must be rank 3 (H, W, C).");
    if (classes_number <= 0 || anchors.empty())
        throw runtime_error("YoloNetwork: classes_number and anchors must be valid.");
    if (input_shape[0] != grid_size * 32 || input_shape[1] != grid_size * 32)
        throw runtime_error("YoloNetwork: this minimal builder expects input H/W == grid_size * 32.");

    auto scaling_layer = make_unique<Scaling>(input_shape);
    scaling_layer->set_scalers("ImageMinMax");
    add_layer(move(scaling_layer));

    const Shape stride{1, 1};
    const Shape pool{2, 2};
    const Shape pool_stride{2, 2};
    const Shape no_padding{0, 0};

    const vector<Index> filters = {32, 64, 128, 256, 512};

    for (Index i = 0; i < ssize(filters); ++i)
    {
        add_layer(make_unique<Convolutional>(get_output_shape(),
                                             Shape{3, 3, get_output_shape()[2], filters[size_t(i)]},
                                             "ReLU", stride, "Same", false,
                                             format("yolo_conv_{}", i + 1)));

        add_layer(make_unique<Pooling>(get_output_shape(), pool, pool_stride,
                                       no_padding, "MaxPooling",
                                       format("yolo_pool_{}", i + 1)));
    }

    add_layer(make_unique<Convolutional>(get_output_shape(),
                                         Shape{3, 3, get_output_shape()[2], 1024},
                                         "ReLU", stride, "Same", false,
                                         "yolo_conv_6"));

    const Index detection_channels = ssize(anchors) * (5 + classes_number);

    add_layer(make_unique<Convolutional>(get_output_shape(),
                                         Shape{1, 1, get_output_shape()[2], detection_channels},
                                         "Identity", stride, "Same", false,
                                         "yolo_logits"));

    add_layer(make_unique<Detection>(get_output_shape(), anchors, "detection_layer"));

    add_layer(make_unique<NonMaxSuppression>(get_output_shape(),
                                             ssize(anchors),
                                             0.5f,
                                             0.4f,
                                             "non_max_suppression_layer"));

    compile();
    set_parameters_random();
}

TextClassificationNetwork::TextClassificationNetwork(const Shape& input_shape,
                                                     const Shape& complexity_dimensions,
                                                     const Shape& output_shape) : NeuralNetwork()
{
    const Index vocabulary_size = input_shape[0];
    const Index sequence_length = input_shape[1];
    const Index embedding_dimension = input_shape[2];
    const Index heads_number = complexity_dimensions[0];

    auto embedding_layer = make_unique<Embedding>(Shape({vocabulary_size, sequence_length}),
                                                  embedding_dimension,
                                                  "embedding_layer");
    embedding_layer->set_scale_embedding(true);
    embedding_layer->set_add_positional_encoding(true);
    add_layer(move(embedding_layer));

    add_layer(make_unique<MultiHeadAttention>(
        Shape({sequence_length, embedding_dimension}),
        heads_number,
        "multihead_attention_layer"));

    add_layer(make_unique<Pooling3d>(get_output_shape(), PoolingMethod::AveragePooling));

    add_layer(make_unique<Dense>(get_output_shape(), Shape({64}), "ReLU", false, "hidden_layer"));

    add_layer(make_unique<Dense>(get_output_shape(),
                                 output_shape,
                                 output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                                 false,
                                 "classification_layer"));

    compile();
    set_parameters_glorot();
}

Transformer::Transformer(Index input_sequence_length,
                         Index decoder_sequence_length,
                         Index input_vocabulary_size,
                         Index output_vocabulary_size,
                         Index embedding_dimension,
                         Index heads_number,
                         Index feed_forward_dimension,
                         Index layers_number)
    : NeuralNetwork()
{
    if (input_sequence_length == 0 ||
        decoder_sequence_length == 0 ||
        input_vocabulary_size == 0 ||
        output_vocabulary_size == 0 ||
        embedding_dimension == 0 ||
        heads_number == 0 ||
        feed_forward_dimension == 0 ||
        layers_number == 0)
        throw runtime_error("Transformer: all dimensions must be > 0.");

    if (embedding_dimension % heads_number != 0)
        throw runtime_error("Transformer: embedding_dimension must be divisible by heads_number.");

    // Adds: Addition(left_index, right_index) → Normalization3d. Returns the norm index.
    auto add_residual_and_norm = [&](const Shape& shape,
                                     const string& add_label,
                                     const string& norm_label,
                                     Index left_index, Index right_index) -> Index {
        add_layer(make_unique<Addition>(shape, add_label), {left_index, right_index});
        add_layer(make_unique<Normalization3d>(shape, norm_label));
        return get_layers_number() - 1;
    };

    // Adds two Dense layers (ReLU expand to ff_dim, Identity project back). Returns
    // the second dense's index. Each Dense chains to the previous layer by default.
    auto add_feed_forward = [&](const Shape& input_shape, Index ff_dim,
                                const string& internal_label,
                                const string& external_label) -> Index {
        const Index seq_len = input_shape[0];
        const Index emb_dim = input_shape[1];
        add_layer(make_unique<Dense>(input_shape, Shape{ff_dim},
                                     "ReLU", false, internal_label));
        add_layer(make_unique<Dense>(Shape{seq_len, ff_dim}, Shape{emb_dim},
                                     "Identity", false, external_label));
        return get_layers_number() - 1;
    };

    // Input embeddings

    auto decoder_embedding = make_unique<Embedding>(
        Shape{output_vocabulary_size, decoder_sequence_length},
        embedding_dimension, "decoder_embedding");
    decoder_embedding->set_scale_embedding(true);
    decoder_embedding->set_add_positional_encoding(true);
    add_layer(move(decoder_embedding), {-1});
    Index current_decoder_index = get_layers_number() - 1;

    auto encoder_embedding = make_unique<Embedding>(
        Shape{input_vocabulary_size, input_sequence_length},
        embedding_dimension, "encoder_embedding");
    encoder_embedding->set_scale_embedding(true);
    encoder_embedding->set_add_positional_encoding(true);
    add_layer(move(encoder_embedding), {-2});
    Index current_encoder_index = get_layers_number() - 1;

    // Encoder stack: self-attention block + feed-forward block, each with residual + post-norm.

    const Shape encoder_shape{input_sequence_length, embedding_dimension};

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = format("_{}", i + 1);

        add_layer(make_unique<MultiHeadAttention>(encoder_shape, heads_number,
                                                  "encoder_self_attention" + suffix),
                  {current_encoder_index});
        const Index attn_index = get_layers_number() - 1;

        const Index norm1_index = add_residual_and_norm(encoder_shape,
            "encoder_self_attention_addition" + suffix,
            "encoder_self_attention_normalization" + suffix,
            current_encoder_index, attn_index);

        const Index ff_index = add_feed_forward(encoder_shape, feed_forward_dimension,
            "encoder_internal_dense" + suffix,
            "encoder_external_dense" + suffix);

        current_encoder_index = add_residual_and_norm(encoder_shape,
            "encoder_dense_addition" + suffix,
            "encoder_dense_normalization" + suffix,
            norm1_index, ff_index);
    }

    const Index encoder_final_output_index = current_encoder_index;

    // Decoder stack: masked self-attention block + cross-attention block + feed-forward block.

    const Shape decoder_shape{decoder_sequence_length, embedding_dimension};

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = format("_{}", i + 1);

        // Masked self-attention.
        auto decoder_self_attention = make_unique<MultiHeadAttention>(
            decoder_shape, heads_number, "decoder_self_attention" + suffix);
        decoder_self_attention->set(decoder_sequence_length, decoder_sequence_length,
                                    embedding_dimension, heads_number,
                                    true,  // use_causal_mask
                                    "decoder_self_attention" + suffix);
        add_layer(move(decoder_self_attention), {current_decoder_index});
        const Index self_attn_index = get_layers_number() - 1;

        const Index norm1_index = add_residual_and_norm(decoder_shape,
            "decoder_self_attention_addition" + suffix,
            "decoder_self_attention_normalization" + suffix,
            current_decoder_index, self_attn_index);

        // Cross-attention against encoder output.
        add_layer(make_unique<MultiHeadAttention>(decoder_shape, encoder_shape,
                                                  heads_number,
                                                  "cross_attention" + suffix),
                  {norm1_index, encoder_final_output_index});
        const Index cross_attn_index = get_layers_number() - 1;

        const Index norm2_index = add_residual_and_norm(decoder_shape,
            "cross_attention_addition" + suffix,
            "cross_attention_normalization" + suffix,
            norm1_index, cross_attn_index);

        const Index ff_index = add_feed_forward(decoder_shape, feed_forward_dimension,
            "decoder_internal_dense" + suffix,
            "decoder_external_dense" + suffix);

        current_decoder_index = add_residual_and_norm(decoder_shape,
            "decoder_dense_addition" + suffix,
            "decoder_dense_normalization" + suffix,
            norm2_index, ff_index);
    }

    // Final token projection.

    add_layer(make_unique<Dense>(decoder_shape, Shape{output_vocabulary_size},
                                 "Softmax", false, "output_projection"));

    compile();
    set_parameters_random();
}

void Transformer::set_dropout_rate(const float new_dropout_rate)
{
    for (auto& layer : get_layers())
    {
        if (!layer) continue;

        const string& label = layer->get_label();
        const bool is_ffn_dense =
               label.starts_with("encoder_internal_dense")
            || label.starts_with("encoder_external_dense")
            || label.starts_with("decoder_internal_dense")
            || label.starts_with("decoder_external_dense");

        if (is_ffn_dense)
        {
            if (auto* dense = dynamic_cast<Dense*>(layer.get()))
                dense->set_dropout_rate(new_dropout_rate);
        }
        else if (auto* mha = dynamic_cast<MultiHeadAttention*>(layer.get()))
        {
            mha->set_dropout_rate(new_dropout_rate);
        }
    }
}

Index Transformer::get_input_sequence_length() const
{
    return get_layer("encoder_embedding")->get_input_shape()[0];
}

Index Transformer::get_decoder_sequence_length() const
{
    return get_layer("decoder_embedding")->get_input_shape()[0];
}

Index Transformer::get_embedding_dimension() const
{
    return get_layer(0)->get_output_shape().back();
}

Index Transformer::get_heads_number() const
{
    if (auto* mha = dynamic_cast<const MultiHeadAttention*>(get_first(LayerType::MultiHeadAttention)))
        return mha->get_heads_number();

    return 0;
}

#endif // OPENNN_NO_VISION

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
