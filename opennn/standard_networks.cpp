//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A N D A R D   N E T W O R K S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "standard_networks.h"
#include "registry.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"
#include "dense_layer.h"
#include "bounding_layer.h"
#include "recurrent_layer.h"
#include "embedding_layer.h"
#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "pooling_layer_3d.h"
#include "flatten_layer.h"
#include "addition_layer.h"
#include "normalization_layer_3d.h"
#include "multihead_attention_layer.h"
#include "string_utilities.h"

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
                                       "dense2d_layer_" + to_string(i + 1)));

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
                                       "dense2d_layer_" + to_string(i + 1)));

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
    set_default();

    add_layer(make_unique<Recurrent>(input_shape, complexity_dimensions));

    add_layer(make_unique<Dense>(complexity_dimensions, output_shape, "Identity", false, "dense_layer"));

    compile();
    set_parameters_random();
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
                                             "convolutional_layer_" + to_string(i + 1)));

        const Shape pool_dimensions = { 2, 2 };
        const Shape pooling_stride_shape = { 2, 2 };
        const Shape padding_dimensions = { 0, 0 };

        add_layer(make_unique<Pooling>(get_output_shape(),
                                       pool_dimensions,
                                       pooling_stride_shape,
                                       padding_dimensions,
                                       "MaxPooling",
                                       "pooling_layer_" + to_string(i + 1)));
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

SimpleResNet::SimpleResNet(const Shape& input_shape,
                           const vector<Index>& blocks_per_stage,
                           const Shape& initial_filters,
                           const Shape& output_shape) : NeuralNetwork()
{
    if (input_shape.rank != 3)
        throw runtime_error("Input shape size must be 3.");
    if (Index(blocks_per_stage.size()) != initial_filters.size())
        throw runtime_error("blocks_per_stage and initial_filters must have the same size.");

    // Adds a Convolutional with explicit input index. Returns the new layer's index.
    auto add_conv = [&](Index input_index,
                        const Shape& kernel_shape, const char* activation,
                        const Shape& stride, const string& name) -> Index {
        add_layer(make_unique<Convolutional>(
            get_layer(input_index)->get_output_shape(),
            kernel_shape, activation, stride, "Same", false, name),
            {input_index});
        return get_layers_number() - 1;
    };

    // Residual block: two 3x3 convs (the first applies `stride`, the second 1x1),
    // a 1x1 skip-conv when shape changes, an Addition, and a 1x1 ReLU "activation".
    auto add_residual_block = [&](Index input_index, size_t stage, Index block, Index filters) -> Index {
        const Shape input_shape  = get_layer(input_index)->get_output_shape();
        const Index stride       = (stage > 0 && block == 0) ? 2 : 1;
        const string prefix      = "s" + to_string(stage) + "b" + to_string(block);

        Index main_index = add_conv(input_index,
            Shape{3, 3, input_shape[2], filters}, "ReLU",
            Shape{stride, stride}, prefix + "_conv1");
        main_index = add_conv(main_index,
            Shape{3, 3, filters, filters}, "Identity",
            Shape{1, 1}, prefix + "_conv2");

        Index skip_index = input_index;
        if (stride != 1 || input_shape[2] != filters)
            skip_index = add_conv(input_index,
                Shape{1, 1, input_shape[2], filters}, "Identity",
                Shape{stride, stride}, prefix + "_skip");

        const Shape main_out = get_layer(main_index)->get_output_shape();
        add_layer(make_unique<Addition>(main_out, prefix + "_add"),
                  {main_index, skip_index});
        const Index sum_index = get_layers_number() - 1;

        return add_conv(sum_index,
            Shape{1, 1, filters, filters}, "ReLU",
            Shape{1, 1}, prefix + "_relu");
    };

    add_layer(make_unique<Scaling>(input_shape));

    Index last_index = add_conv(0,
        Shape{7, 7, input_shape[2], initial_filters[0]}, "ReLU",
        Shape{2, 2}, "stem_conv_1");

    add_layer(make_unique<Pooling>(get_layer(last_index)->get_output_shape(),
                                   Shape{3, 3}, Shape{2, 2}, Shape{1, 1},
                                   "MaxPooling", "stem_pool"),
              {last_index});
    last_index = get_layers_number() - 1;

    for (size_t stage = 0; stage < blocks_per_stage.size(); ++stage)
        for (Index block = 0; block < blocks_per_stage[stage]; ++block)
            last_index = add_residual_block(last_index, stage, block, initial_filters[stage]);

    const Shape pre_pool = get_layer(last_index)->get_output_shape();
    add_layer(make_unique<Pooling>(pre_pool,
                                   Shape{pre_pool[0], pre_pool[1]},
                                   Shape{1, 1}, Shape{0, 0},
                                   "AveragePooling", "global_avg_pool"),
              {last_index});
    last_index = get_layers_number() - 1;

    add_layer(make_unique<Flatten>(get_layer(last_index)->get_output_shape()), {last_index});
    last_index = get_layers_number() - 1;

    add_layer(make_unique<Dense>(get_layer(last_index)->get_output_shape(),
                                 output_shape, "Softmax", false, "dense_classifier"),
              {last_index});

    compile();
    set_parameters_random();
}

VGG16::VGG16(const Shape& new_input_shape, const Shape& new_target_shape)
    : NeuralNetwork()
{
    set(new_input_shape, new_target_shape);
}

void VGG16::set(const Shape& new_input_shape, const Shape& new_target_shape)
{
    add_layer(make_unique<Scaling>(new_input_shape));

    // 3x3 ReLU conv with stride 1, "Same" padding. In-channels read from the previous layer's output.
    auto add_conv = [&](Index out_channels, const string& name) {
        const Shape in = get_output_shape();
        add_layer(make_unique<Convolutional>(
            in, Shape{3, 3, in[2], out_channels}, "ReLU",
            Shape{1, 1}, "Same", false, name));
    };

    // 2x2 max pool with stride 2, no padding.
    auto add_max_pool = [&](const string& name) {
        add_layer(make_unique<Pooling>(
            get_output_shape(), Shape{2, 2}, Shape{2, 2}, Shape{0, 0},
            "MaxPooling", name));
    };

    add_conv(64,  "conv_1");
    add_conv(64,  "conv_2");
    add_max_pool("pool1");

    add_conv(128, "conv_3");
    add_conv(128, "conv_4");
    add_max_pool("pool2");

    add_conv(256, "conv_5");
    add_conv(256, "conv_6");
    add_conv(256, "conv_7");
    add_max_pool("pool3");

    add_conv(512, "conv_8");
    add_conv(512, "conv_9");
    add_conv(512, "conv_10");
    add_max_pool("pool4");

    add_conv(512, "conv_11");
    add_conv(512, "conv_12");
    add_conv(512, "conv_13");
    add_max_pool("pool5");

    const Shape pre_pool_shape = get_output_shape();
    add_layer(make_unique<Pooling>(
        pre_pool_shape,
        Shape{pre_pool_shape[0], pre_pool_shape[1]},
        Shape{1, 1}, Shape{0, 0},
        "AveragePooling", "global_avg_pool"));

    add_layer(make_unique<Flatten>(get_output_shape()));

    add_layer(make_unique<Dense>(get_output_shape(), new_target_shape,
                                 "Softmax", false, "dense_classifier"));

    compile();
    set_parameters_random();
}

VGG16::VGG16(const filesystem::path& file_name)
    : NeuralNetwork(file_name)
{

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

    add_layer(make_unique<Dense>(get_output_shape(), Shape({16}), "ReLU", false, "hidden_layer"));

    add_layer(make_unique<Dense>(get_output_shape(), output_shape, "Sigmoid", false, "classification_layer"));

    compile();
    set_parameters_glorot();
}

Transformer::Transformer(const Index input_sequence_length,
                         Index decoder_sequence_length,
                         Index input_vocabulary_size,
                         Index output_vocabulary_size,
                         Index embedding_dimension,
                         Index heads_number,
                         Index feed_forward_dimension,
                         Index layers_number)
{
    set(input_sequence_length,
        decoder_sequence_length,
        input_vocabulary_size,
        output_vocabulary_size,
        embedding_dimension,
        heads_number,
        feed_forward_dimension,
        layers_number);
}

void Transformer::set(const Index input_sequence_length,
                      Index decoder_sequence_length,
                      Index input_vocabulary_size,
                      Index output_vocabulary_size,
                      Index embedding_dimension,
                      Index heads_number,
                      Index feed_forward_dimension,
                      Index layers_number)
{
    name = "transformer";

    layers.clear();
    layer_input_indices.clear();

    if (input_sequence_length == 0 ||
        decoder_sequence_length == 0 ||
        input_vocabulary_size == 0 ||
        output_vocabulary_size == 0 ||
        embedding_dimension == 0 ||
        heads_number == 0 ||
        feed_forward_dimension == 0 ||
        layers_number == 0)
        return;

    if (embedding_dimension % heads_number != 0)
        throw runtime_error("embedding_dimension must be divisible by heads_number.");

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
        const string suffix = "_" + to_string(i + 1);

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
        const string suffix = "_" + to_string(i + 1);

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
               label.rfind("encoder_internal_dense", 0) == 0
            || label.rfind("encoder_external_dense", 0) == 0
            || label.rfind("decoder_internal_dense", 0) == 0
            || label.rfind("decoder_external_dense", 0) == 0;

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

void Transformer::set_input_vocabulary(const vector<string>& new_input_vocabulary)
{
    input_vocabulary = new_input_vocabulary;

    input_vocabulary_map.clear();
    input_vocabulary_map.reserve(input_vocabulary.size());

    for (size_t i = 0; i < input_vocabulary.size(); ++i)
        input_vocabulary_map[input_vocabulary[i]] = i;
}

void Transformer::set_output_vocabulary(const vector<string>& new_output_vocabulary)
{
    output_vocabulary = new_output_vocabulary;

    output_inverse_vocabulary_map.clear();
    output_inverse_vocabulary_map.reserve(output_vocabulary.size());

    for (size_t i = 0; i < output_vocabulary.size(); ++i)
        output_inverse_vocabulary_map[i] = output_vocabulary[i];
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

string Transformer::calculate_outputs(const string& source)
{
    if (input_vocabulary_map.empty() || output_inverse_vocabulary_map.empty())
        throw runtime_error("Transformer::calculate_outputs Error: Vocabularies not initialized.");

    constexpr float PAD   = 0.0f;
    constexpr float UNK   = 1.0f;
    constexpr float START = 2.0f;
    constexpr float END   = 3.0f;

    const Index input_sequence_length = get_input_sequence_length();
    const Index decoder_sequence_length = get_decoder_sequence_length();
    const Index batch_size = 1;

    const vector<string> source_tokens = tokenize(source);

    Tensor2 source_ids(batch_size, input_sequence_length);
    source_ids.setConstant(PAD);

    source_ids(0, 0) = START;

    Index write_index = 1;

    for (size_t i = 0; i < source_tokens.size() && write_index < input_sequence_length; ++i, ++write_index)
    {
        const auto it = input_vocabulary_map.find(source_tokens[i]);

        source_ids(0, write_index) = (it != input_vocabulary_map.end())
                                         ? static_cast<float>(it->second)
                                         : UNK;
    }

    if (write_index < input_sequence_length)
        source_ids(0, write_index) = END;

    Tensor2 target_ids(batch_size, decoder_sequence_length);
    target_ids.setConstant(PAD);
    target_ids(0, 0) = START;

    const bool was_gpu = is_gpu();
    if (was_gpu)
    {
        Configuration::instance().set(Device::CPU,
                                      Type::FP32,
                                      Type::FP32);
#ifdef OPENNN_HAS_CUDA
        copy_parameters_host();
        copy_states_host();
#endif
    }

    ForwardPropagation forward_propagation(batch_size, this);

    for (Index i = 1; i < decoder_sequence_length; ++i)
    {
        const vector<TensorView> inputs =
        {TensorView(target_ids.data(), {batch_size, decoder_sequence_length}),
         TensorView(source_ids.data(), {batch_size, input_sequence_length})};

        forward_propagate(inputs, forward_propagation, false);

        const TensorView output_view = forward_propagation.get_outputs();
        const Index vocabulary_size = output_view.shape[2];

        const float* distribution_ptr = output_view.as<float>() + (i-1)*vocabulary_size;

        const Map<const VectorR> current_distribution(distribution_ptr, vocabulary_size);

        const Index best_id = maximal_index(current_distribution);

        target_ids(0, i) = static_cast<float>(best_id);

        if (best_id == END)
            break;
    }

    if (was_gpu)
    {
        Configuration::instance().set(Device::Auto,
                                      Type::Auto,
                                      Type::Auto);
#ifdef OPENNN_HAS_CUDA
        copy_parameters_device();
        copy_states_device();
#endif
    }

    string result;

    for (Index i = 1; i < decoder_sequence_length; ++i)
    {
        const Index id = static_cast<Index>(target_ids(0, i));

        if (id == END || id == PAD)
            break;

        const auto it = output_inverse_vocabulary_map.find(id);

        if (it == output_inverse_vocabulary_map.end())
            continue;

        if (!result.empty())
            result += " ";

        result += it->second;
    }

    return result;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
