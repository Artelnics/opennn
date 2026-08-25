//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E   M O D E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/models/models.h"

#include <utility>

#include "opennn/core/string_utilities.h"
#include "opennn/neural_network/layers/activation_layer.h"
#include "opennn/neural_network/layers/addition_layer.h"
#include "opennn/neural_network/layers/clamping_layer.h"
#include "opennn/neural_network/layers/c2psa_layer.h"
#include "opennn/neural_network/layers/concatenation_layer.h"
#include "opennn/neural_network/layers/convolutional_layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/detection_layer.h"
#include "opennn/neural_network/layers/detection_v8_layer.h"
#include "opennn/neural_network/layers/embedding_layer.h"
#include "opennn/neural_network/layers/flatten_layer.h"
#include "opennn/neural_network/layers/grouped_query_attention_layer.h"
#include "opennn/neural_network/layers/long_short_term_memory_layer.h"
#include "opennn/neural_network/layers/multihead_attention_layer.h"
#include "opennn/neural_network/layers/non_max_suppression_layer.h"
#include "opennn/neural_network/layers/normalization_layer_3d.h"
#include "opennn/neural_network/layers/pooling_layer.h"
#include "opennn/neural_network/layers/pooling_layer_3d.h"
#include "opennn/neural_network/layers/recurrent_layer.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/neural_network/layers/tokenizer_layer.h"
#include "opennn/neural_network/layers/unscaling_layer.h"
#include "opennn/neural_network/layers/upsampling_layer.h"

namespace opennn
{

static void recompile_if_specs_changed(NeuralNetwork& network,
                                       const vector<vector<TensorSpec>>& forward_before,
                                       const vector<vector<TensorSpec>>& backward_before)
{
    if (forward_before == network.get_forward_specs(1)
        && backward_before == network.get_backward_specs(1))
        return;

    VectorR parameters_snapshot;
    if (network.get_parameters_buffer_size() > 0)
    {
        network.copy_parameters_host();
        parameters_snapshot = network.get_parameters_map();
    }

    network.compile();

    if (parameters_snapshot.size() > 0)
        network.set_parameters(parameters_snapshot);
}

static void finalize_build(NeuralNetwork& network)
{
    network.compile();
    network.set_parameters_glorot();
}

#ifndef OPENNN_NO_VISION

TextClassificationNetwork::TextClassificationNetwork(const Shape& input_shape,
                                                     const Shape& complexity_dimensions,
                                                     const Shape& output_shape,
                                                     PoolingMethod pooling_method)
    : NeuralNetwork(NetworkTask::TextClassification)
{
    throw_if(input_shape.get_rank() < 3,
             "TextClassificationNetwork: the input shape must be "
             "{{vocabulary_size, sequence_length, embedding_dimension}}, got rank {}.",
             input_shape.get_rank());
    throw_if(complexity_dimensions.get_rank() < 1,
             "TextClassificationNetwork: the complexity dimensions must name at least the "
             "number of heads.");

    const Index vocabulary_size = input_shape[0];
    const Index sequence_length = input_shape[1];
    const Index embedding_dimension = input_shape[2];
    const Index heads_number = complexity_dimensions[0];
    const Index hidden_neurons = complexity_dimensions.get_rank() > 1 ? complexity_dimensions[1] : 64;

    add_layer(make_unique<Tokenizer>(Shape{sequence_length}, "tokenizer"), {-1});

    auto embedding_layer = make_unique<Embedding>(Shape({vocabulary_size, sequence_length}),
                                                  embedding_dimension,
                                                  "embedding_layer");
    embedding_layer->set_scale_embedding(true);
    embedding_layer->set_add_positional_encoding(true);
    embedding_layer->set_export_valid_lengths(true);
    add_layer(std::move(embedding_layer));

    auto attention_layer = make_unique<MultiHeadAttention>(
        Shape({sequence_length, embedding_dimension}),
        heads_number,
        "multihead_attention_layer");

    add_layer(std::move(attention_layer));

    add_layer(make_unique<Pooling3d>(get_output_shape(), pooling_method));

    add_layer(make_unique<Dense>(get_output_shape(), Shape({hidden_neurons}), "ReLU", BatchNormalization::No, "dense_layer_1"));

    add_layer(make_unique<Dense>(get_output_shape(),
                                 output_shape,
                                 output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                                 BatchNormalization::No,
                                 "classification_layer"));

    finalize_build(*this);
}

static Index add_residual_and_norm(NeuralNetwork& network,
                                   const Shape& shape,
                                   const string& norm_label,
                                   Index left_index, Index right_index)
{
    auto norm = make_unique<Normalization3d>(shape, norm_label);
    norm->set_fuse_add(true);
    return network.add_layer(std::move(norm), {left_index, right_index});
}

static Index add_feed_forward(NeuralNetwork& network,
                              const Shape& input_shape, Index ff_dim,
                              const string& internal_label,
                              const string& external_label,
                              const string& internal_activation = "ReLU")
{
    const Index seq_len = input_shape[0];
    const Index emb_dim = input_shape[1];
    network.add_layer(make_unique<Dense>(input_shape, Shape{ff_dim},
                                         internal_activation, BatchNormalization::No, internal_label));
    return network.add_layer(make_unique<Dense>(Shape{seq_len, ff_dim}, Shape{emb_dim},
                                                "Identity", BatchNormalization::No, external_label));
}

Transformer::Transformer()
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
}

Transformer::Transformer(Index input_sequence_length,
                         Index decoder_sequence_length,
                         Index input_vocabulary_size,
                         Index output_vocabulary_size,
                         Index embedding_dimension,
                         Index heads_number,
                         Index feed_forward_dimension,
                         Index layers_number)
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
    throw_if(input_sequence_length == 0 ||
             decoder_sequence_length == 0 ||
             input_vocabulary_size == 0 ||
             output_vocabulary_size == 0 ||
             embedding_dimension == 0 ||
             heads_number == 0 ||
             feed_forward_dimension == 0 ||
             layers_number == 0,
             "Transformer: all dimensions must be > 0.");

    throw_if(embedding_dimension % heads_number != 0,
             "Transformer: embedding_dimension must be divisible by heads_number.");

    const Index decoder_tokenizer_index = add_layer(make_unique<Tokenizer>(Shape{decoder_sequence_length}, "decoder_tokenizer"), {-1});

    auto decoder_embedding = make_unique<Embedding>(
        Shape{output_vocabulary_size, decoder_sequence_length},
        embedding_dimension, "decoder_embedding");
    decoder_embedding->set_scale_embedding(true);
    decoder_embedding->set_add_positional_encoding(true);
    decoder_embedding->set_export_valid_lengths(true);
    Index current_decoder_index = add_layer(std::move(decoder_embedding), {decoder_tokenizer_index});

    const Index encoder_tokenizer_index = add_layer(make_unique<Tokenizer>(Shape{input_sequence_length}, "encoder_tokenizer"), {-2});

    auto encoder_embedding = make_unique<Embedding>(
        Shape{input_vocabulary_size, input_sequence_length},
        embedding_dimension, "encoder_embedding");
    encoder_embedding->set_scale_embedding(true);
    encoder_embedding->set_add_positional_encoding(true);
    encoder_embedding->set_export_valid_lengths(true);
    Index current_encoder_index = add_layer(std::move(encoder_embedding), {encoder_tokenizer_index});

    const Shape encoder_shape{input_sequence_length, embedding_dimension};

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = format("_{}", i + 1);

        const Index attn_index = add_layer(make_unique<MultiHeadAttention>(encoder_shape, heads_number,
                                                                           "encoder_self_attention" + suffix),
                                           {current_encoder_index});

        const Index norm1_index = add_residual_and_norm(*this, encoder_shape,
            "encoder_self_attention_normalization" + suffix,
            current_encoder_index, attn_index);

        const Index ff_index = add_feed_forward(*this, encoder_shape, feed_forward_dimension,
            "encoder_internal_dense" + suffix,
            "encoder_external_dense" + suffix);

        current_encoder_index = add_residual_and_norm(*this, encoder_shape,
            "encoder_dense_normalization" + suffix,
            norm1_index, ff_index);
    }

    const Index encoder_final_output_index = current_encoder_index;

    const Shape decoder_shape{decoder_sequence_length, embedding_dimension};

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = format("_{}", i + 1);

        auto decoder_self_attention = make_unique<MultiHeadAttention>(
            decoder_shape, heads_number, "decoder_self_attention" + suffix);
        decoder_self_attention->set(decoder_sequence_length, decoder_sequence_length,
                                    embedding_dimension, heads_number,
                                    CausalMask::Yes,
                                    "decoder_self_attention" + suffix);
        const Index self_attn_index = add_layer(std::move(decoder_self_attention), {current_decoder_index});

        const Index norm1_index = add_residual_and_norm(*this, decoder_shape,
            "decoder_self_attention_normalization" + suffix,
            current_decoder_index, self_attn_index);

        const Index cross_attn_index = add_layer(make_unique<MultiHeadAttention>(decoder_shape, encoder_shape,
                                                                                 heads_number,
                                                                                 "cross_attention" + suffix),
                                                 {norm1_index, encoder_final_output_index});

        const Index norm2_index = add_residual_and_norm(*this, decoder_shape,
            "cross_attention_normalization" + suffix,
            norm1_index, cross_attn_index);

        const Index ff_index = add_feed_forward(*this, decoder_shape, feed_forward_dimension,
            "decoder_internal_dense" + suffix,
            "decoder_external_dense" + suffix);

        current_decoder_index = add_residual_and_norm(*this, decoder_shape,
            "decoder_dense_normalization" + suffix,
            norm2_index, ff_index);
    }

    add_layer(make_unique<Dense>(decoder_shape, Shape{output_vocabulary_size},
                                 "Softmax", BatchNormalization::No, "output_projection"));

    finalize_build(*this);
}

template<typename Apply>
static void apply_and_recompile(NeuralNetwork& network, Apply apply)
{
    const auto forward_before = network.get_forward_specs(1);
    const auto backward_before = network.get_backward_specs(1);

    for (const auto& layer : network.get_layers())
        if (layer) apply(*layer);

    recompile_if_specs_changed(network, forward_before, backward_before);
}

static void set_attention_and_dense_dropout(NeuralNetwork& network, float new_dropout_rate,
                                            initializer_list<string_view> dense_prefixes)
{
    apply_and_recompile(network, [&](Layer& layer)
    {
        if (auto* mha = dynamic_cast<MultiHeadAttention*>(&layer))
            mha->set_dropout_rate(new_dropout_rate);
        else if (starts_with_any(layer.get_label(), dense_prefixes))
            if (auto* dense = dynamic_cast<Dense*>(&layer))
                dense->set_dropout_rate(new_dropout_rate);
    });
}

void Transformer::set_dropout_rate(const float new_dropout_rate)
{
    set_attention_and_dense_dropout(*this, new_dropout_rate,
                                    {"encoder_internal_dense", "encoder_external_dense",
                                     "decoder_internal_dense", "decoder_external_dense"});
}

void Transformer::set_attention_sdpa_min_sequence_length(Index new_threshold)
{
    apply_and_recompile(*this, [&](Layer& layer)
    {
        if (auto* mha = dynamic_cast<MultiHeadAttention*>(&layer))
            mha->set_sdpa_min_sequence_length(new_threshold);
    });
}

TextGenerationNetwork::TextGenerationNetwork()
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
}

TextGenerationNetwork::TextGenerationNetwork(Index sequence_length,
                                             Index vocabulary_size,
                                             Index embedding_dimension,
                                             Index heads_number,
                                             Index feed_forward_dimension,
                                             Index layers_number,
                                             bool pre_normalization,
                                             bool scale_embedding,
                                             bool learned_positional,
                                             const string& feed_forward_activation)
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
    throw_if(sequence_length == 0 ||
             vocabulary_size == 0 ||
             embedding_dimension == 0 ||
             heads_number == 0 ||
             feed_forward_dimension == 0 ||
             layers_number == 0,
             "TextGenerationNetwork: all dimensions must be > 0.");

    throw_if(embedding_dimension % heads_number != 0,
             "TextGenerationNetwork: embedding_dimension must be divisible by heads_number.");

    const Index tokenizer_index = add_layer(make_unique<Tokenizer>(Shape{sequence_length}, "tokenizer"), {-1});

    auto embedding = make_unique<Embedding>(
        Shape{vocabulary_size, sequence_length},
        embedding_dimension, "embedding");
    embedding->set_scale_embedding(scale_embedding);
    if (learned_positional)
        embedding->set_learned_positional(true);
    else
        embedding->set_add_positional_encoding(true);
    Index current_index = add_layer(std::move(embedding), {tokenizer_index});

    const Shape block_shape{sequence_length, embedding_dimension};

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = format("_{}", i + 1);

        Index attention_input_index = current_index;

        if (pre_normalization)
        {
            attention_input_index = add_layer(make_unique<Normalization3d>(block_shape,
                                                                           "attention_normalization" + suffix),
                                              {current_index});
        }

        auto self_attention = make_unique<MultiHeadAttention>(
            block_shape, heads_number, "self_attention" + suffix);
        self_attention->set(sequence_length, sequence_length,
                            embedding_dimension, heads_number,
                            CausalMask::Yes,
                            "self_attention" + suffix);
        const Index attn_index = add_layer(std::move(self_attention), {attention_input_index});

        if (pre_normalization)
        {
            const Index residual_index = add_layer(make_unique<Addition>(block_shape, "attention_addition" + suffix),
                                                   {current_index, attn_index});

            add_layer(make_unique<Normalization3d>(block_shape,
                                                   "dense_normalization" + suffix),
                      {residual_index});

            const Index ff_index = add_feed_forward(*this, block_shape, feed_forward_dimension,
                "internal_dense" + suffix,
                "external_dense" + suffix,
                feed_forward_activation);

            current_index = add_layer(make_unique<Addition>(block_shape, "dense_addition" + suffix),
                                      {residual_index, ff_index});
        }
        else
        {
            const Index norm1_index = add_residual_and_norm(*this, block_shape,
                "self_attention_normalization" + suffix,
                current_index, attn_index);

            const Index ff_index = add_feed_forward(*this, block_shape, feed_forward_dimension,
                "internal_dense" + suffix,
                "external_dense" + suffix,
                feed_forward_activation);

            current_index = add_residual_and_norm(*this, block_shape,
                "dense_normalization" + suffix,
                norm1_index, ff_index);
        }
    }

    if (pre_normalization)
        add_layer(make_unique<Normalization3d>(block_shape, "final_normalization"),
                  {current_index});

    add_layer(make_unique<Dense>(block_shape, Shape{vocabulary_size},
                                 "Softmax", BatchNormalization::No, "output_projection"));

    finalize_build(*this);
}

static Index add_bert_encoder(NeuralNetwork& net,
                              Index sequence_length, Index vocabulary_size, Index hidden_size,
                              Index heads_number, Index intermediate_size, Index layers_number,
                              Index type_vocabulary_size)
{
    throw_if(sequence_length == 0 || vocabulary_size == 0 || hidden_size == 0 ||
             heads_number == 0 || intermediate_size == 0 || layers_number == 0 ||
             type_vocabulary_size == 0,
             "BERT: all dimensions must be > 0.");

    throw_if(hidden_size % heads_number != 0,
             "BERT: hidden_size must be divisible by heads_number.");

    const Shape seq_hidden{sequence_length, hidden_size};

    auto word_embeddings = make_unique<Embedding>(
        Shape{vocabulary_size, sequence_length}, hidden_size, "word_embeddings");
    word_embeddings->set_learned_positional(true);
    word_embeddings->set_export_valid_lengths(true);
    const Index word_index = net.add_layer(std::move(word_embeddings), {-1});

    const Index type_index = net.add_layer(make_unique<Embedding>(
                                               Shape{type_vocabulary_size + 1, sequence_length}, hidden_size, "token_type_embeddings"),
                                           {-2});

    Index current = add_residual_and_norm(net, seq_hidden, "embeddings_layer_norm", word_index, type_index);

    for (Index i = 0; i < layers_number; ++i)
    {
        const string sfx = format("_{}", i + 1);

        const Index attention_index = net.add_layer(make_unique<MultiHeadAttention>(seq_hidden, heads_number, "attention" + sfx),
                                                    {current});

        const Index attention_norm_index =
            add_residual_and_norm(net, seq_hidden, "attention_layer_norm" + sfx, current, attention_index);

        const Index feed_forward_index = add_feed_forward(net, seq_hidden, intermediate_size,
            "intermediate" + sfx, "feed_forward_output" + sfx, "GELU");

        current = add_residual_and_norm(net, seq_hidden, "output_layer_norm" + sfx,
                                        attention_norm_index, feed_forward_index);
    }

    return current;
}

Bert::Bert()
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
}

Bert::Bert(Index sequence_length,
           Index vocabulary_size,
           Index hidden_size,
           Index heads_number,
           Index intermediate_size,
           Index layers_number,
           Index type_vocabulary_size)
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
    add_bert_encoder(*this, sequence_length, vocabulary_size, hidden_size, heads_number,
                     intermediate_size, layers_number, type_vocabulary_size);
    finalize_build(*this);
}

Qwen3::Qwen3()
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
}

Qwen3::Qwen3(Index sequence_length,
             Index vocabulary_size,
             Index hidden_size,
             Index layers_number,
             Index query_heads,
             Index key_value_heads,
             Index head_dimension,
             Index intermediate_size,
             float rope_theta,
             float rms_epsilon)
    : NeuralNetwork(NetworkTask::LanguageModeling)
{
    throw_if(sequence_length == 0 || vocabulary_size == 0 || hidden_size == 0 ||
             layers_number == 0 || query_heads == 0 || key_value_heads == 0 ||
             head_dimension == 0 || intermediate_size == 0,
             "Qwen3: all dimensions must be > 0.");
    throw_if(query_heads % key_value_heads != 0,
             "Qwen3: query_heads must be divisible by key_value_heads.");

    auto embedding = make_unique<Embedding>(Shape{vocabulary_size + 1, sequence_length}, hidden_size, "embed_tokens");
    embedding->set_scale_embedding(false);
    embedding->set_weights_follow_compute_dtype(true);
    Index current = add_layer(std::move(embedding), {-1});

    const Shape block{sequence_length, hidden_size};

    auto add_norm = [&](const string& name, Index source)
    {
        auto norm = make_unique<Normalization3d>(block, name);
        norm->set_method(NormalizationMethod::RMS);
        norm->set_epsilon(rms_epsilon);
        return add_layer(std::move(norm), {source});
    };

    auto add_linear = [&](const Shape& in_shape, Index out_features, const string& name, Index source)
    {
        auto dense = make_unique<Dense>(in_shape, Shape{out_features}, "Identity", BatchNormalization::No, name);
        dense->set_use_bias(false);
        return add_layer(std::move(dense), {source});
    };

    for (Index i = 0; i < layers_number; ++i)
    {
        const string suffix = "_" + to_string(i);

        const Index input_norm = add_norm("input_norm" + suffix, current);
        const Index attention = add_layer(make_unique<GroupedQueryAttention>(block, query_heads, key_value_heads, head_dimension,
                                                                             rope_theta, rms_epsilon,   true,
                                                                             "attn" + suffix), {input_norm});
        const Index residual = add_layer(make_unique<Addition>(block, "attn_add" + suffix), {current, attention});

        const Index post_norm = add_norm("post_norm" + suffix, residual);

        auto gate_up = make_unique<Dense>(block, Shape{intermediate_size}, "Identity", BatchNormalization::No, "gate_up" + suffix);
        gate_up->set_use_bias(false);
        gate_up->set_gated(true);
        const Index ffn = add_layer(std::move(gate_up), {post_norm});
        const Index down = add_linear(Shape{sequence_length, intermediate_size}, hidden_size, "down" + suffix, ffn);
        static_cast<Dense*>(layers[size_t(down)].get())->set_transposed_inference(true);
        current = add_layer(make_unique<Addition>(block, "ffn_add" + suffix), {residual, down});
    }

    current = add_norm("final_norm", current);

    const Index lm_head = add_linear(block, vocabulary_size + 1, "lm_head", current);
    static_cast<Dense*>(layers[size_t(lm_head)].get())->set_tied_weight_source(layers.front().get());

    compile();
    set_parameters_random();
}

void TextGenerationNetwork::set_dropout_rate(const float new_dropout_rate)
{
    set_attention_and_dense_dropout(*this, new_dropout_rate,
                                    {"internal_dense", "external_dense"});
}

void TextGenerationNetwork::set_attention_sdpa_auto(bool new_sdpa_auto)
{
    apply_and_recompile(*this, [&](Layer& layer)
    {
        if (auto* mha = dynamic_cast<MultiHeadAttention*>(&layer))
            mha->set_sdpa_auto(new_sdpa_auto);
    });
}

namespace
{

template <typename Network>
auto& get_tokenizer_layer(Network& network, const char* label, const char* method)
{
    using TokenizerType = conditional_t<is_const_v<Network>, const Tokenizer, Tokenizer>;
    TokenizerType* tokenizer_layer = nullptr;

    try
    {
        tokenizer_layer =
            dynamic_cast<TokenizerType*>(network.get_layer(label).get());
    }
    catch (const exception&)
    {
    }

    throw_if(!tokenizer_layer,
             format("{}: network has no '{}' layer. Rebuild the network or "
                    "re-save the model with a tokenizer.",
                    method, label));

    return *tokenizer_layer;
}

}

Transformer::Transformer(const filesystem::path& path)
    : NeuralNetwork(path, NetworkTask::LanguageModeling)
{
}

void Transformer::set_input_vocabulary(const vector<string>& new_vocabulary)
{
    get_tokenizer_layer(*this, "encoder_tokenizer", "Transformer::set_input_vocabulary")
        .set_vocabulary(new_vocabulary);
}

void Transformer::set_target_vocabulary(const vector<string>& new_vocabulary)
{
    get_tokenizer_layer(*this, "decoder_tokenizer", "Transformer::set_target_vocabulary")
        .set_vocabulary(new_vocabulary);
}

const TokenizerOperator* Transformer::get_input_tokenizer() const
{
    return get_tokenizer_layer(*this, "encoder_tokenizer", "Transformer::get_input_tokenizer").get_tokenizer();
}

const TokenizerOperator* Transformer::get_target_tokenizer() const
{
    return get_tokenizer_layer(*this, "decoder_tokenizer", "Transformer::get_target_tokenizer").get_tokenizer();
}

const vector<string>& Transformer::get_input_vocabulary() const
{
    return get_tokenizer_layer(*this, "encoder_tokenizer", "Transformer::get_input_vocabulary").get_vocabulary();
}

const vector<string>& Transformer::get_target_vocabulary() const
{
    return get_tokenizer_layer(*this, "decoder_tokenizer", "Transformer::get_target_vocabulary").get_vocabulary();
}

TextGenerationNetwork::TextGenerationNetwork(const filesystem::path& path)
    : NeuralNetwork(path, NetworkTask::LanguageModeling)
{
}

void TextGenerationNetwork::set_tokenizer(unique_ptr<TokenizerOperator> new_tokenizer)
{
    get_tokenizer_layer(*this, "tokenizer", "TextGenerationNetwork::set_tokenizer")
        .set_tokenizer(std::move(new_tokenizer));
}

void TextGenerationNetwork::set_vocabulary(const vector<string>& new_vocabulary)
{
    get_tokenizer_layer(*this, "tokenizer", "TextGenerationNetwork::set_vocabulary")
        .set_vocabulary(new_vocabulary);
}

const TokenizerOperator* TextGenerationNetwork::get_tokenizer() const
{
    return get_tokenizer_layer(*this, "tokenizer", "TextGenerationNetwork::get_tokenizer").get_tokenizer();
}

void TextClassificationNetwork::set_tokenizer(unique_ptr<TokenizerOperator> new_tokenizer)
{
    get_tokenizer_layer(*this, "tokenizer", "TextClassificationNetwork::set_tokenizer")
        .set_tokenizer(std::move(new_tokenizer));
}

const TokenizerOperator* TextClassificationNetwork::get_tokenizer() const
{
    return get_tokenizer_layer(*this, "tokenizer", "TextClassificationNetwork::get_tokenizer").get_tokenizer();
}

MatrixR TextClassificationNetwork::calculate_text_outputs(
    const Tensor<string, 1>& input_documents)
{
    const Tokenizer& tokenizer_layer = get_tokenizer_layer(
        *this, "tokenizer", "TextClassificationNetwork::calculate_text_outputs");
    const TokenizerOperator* tokenizer = tokenizer_layer.get_tokenizer();

    throw_if(!tokenizer || tokenizer->get_vocabulary_size() == 0,
             "TextClassificationNetwork::calculate_text_outputs: the tokenizer "
             "has no vocabulary; call set_tokenizer() first.");

    const Index sequence_length = tokenizer_layer.get_output_shape()[0];
    const Index batch_size = input_documents.size();
    MatrixR inputs = MatrixR::Zero(batch_size, sequence_length);

    for (Index i = 0; i < batch_size; ++i)
    {
        const vector<Index> ids =
            tokenizer->encode_sequence(input_documents.data()[i], sequence_length);

        for (Index j = 0; j < min(ssize(ids), sequence_length); ++j)
            inputs(i, j) = float(ids[size_t(j)]);
    }

    return calculate_outputs(inputs);
}

BertForSequenceClassification::BertForSequenceClassification()
    : NeuralNetwork(NetworkTask::TextClassification)
{
}

BertForSequenceClassification::BertForSequenceClassification(Index sequence_length,
                                                             Index vocabulary_size,
                                                             Index hidden_size,
                                                             Index heads_number,
                                                             Index intermediate_size,
                                                             Index layers_number,
                                                             Index labels_number,
                                                             Index type_vocabulary_size)
    : NeuralNetwork(NetworkTask::TextClassification)
{
    throw_if(labels_number == 0, "BertForSequenceClassification: labels_number must be > 0.");

    const Index encoder_index = add_bert_encoder(*this, sequence_length, vocabulary_size, hidden_size,
                                                 heads_number, intermediate_size, layers_number,
                                                 type_vocabulary_size);

    add_layer(make_unique<Pooling3d>(Shape{sequence_length, hidden_size},
                                     PoolingMethod::FirstToken, "cls_pooling"),
              {encoder_index});

    add_layer(make_unique<Dense>(Shape{hidden_size}, Shape{hidden_size}, "Tanh", BatchNormalization::No, "pooler"));

    add_layer(make_unique<Dense>(Shape{hidden_size}, Shape{labels_number},
                                 labels_number == 1 ? "Sigmoid" : "Softmax", BatchNormalization::No, "classifier"));

    finalize_build(*this);
}

void BertForSequenceClassification::set_dropout_rate(const float new_dropout_rate)
{
    set_attention_and_dense_dropout(*this, new_dropout_rate,
                                    {"feed_forward_output", "pooler"});
}

#endif
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
