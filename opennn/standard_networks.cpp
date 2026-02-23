//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A N D A R D   N E T W O R K S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <stdexcept>

#include "pch.h"
#include "standard_networks.h"
#include "multihead_attention_layer.h"
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
#include "strings_utilities.h"

using namespace std;

namespace opennn
{

ApproximationNetwork::ApproximationNetwork(const Shape& input_shape,
                                           const Shape& complexity_dimensions,
                                           const Shape& output_shape) : NeuralNetwork()
{
    const Index complexity_size = complexity_dimensions.size();

    add_layer(make_unique<Scaling<2>>(input_shape));

    for(Index i = 0; i < complexity_size; i++)
        add_layer(make_unique<Dense<2>>(get_output_shape(),
                                       Shape{ complexity_dimensions[i] },
                                       "HyperbolicTangent",
                                       false,
                                       "dense2d_layer_" + to_string(i + 1)));

    add_layer(make_unique<Dense<2>>(get_output_shape(),
                                   output_shape,
                                   "Linear",
                                   false,
                                   "approximation_layer"));

    add_layer(make_unique<Unscaling>(output_shape));

    add_layer(make_unique<Bounding>(output_shape));

    this->compile();
    this->set_parameters_glorot();

    const Index inputs_number = get_inputs_number();
    input_names.resize(inputs_number);

    const Index outputs_number = get_outputs_number();
    output_names.resize(outputs_number);
}


ClassificationNetwork::ClassificationNetwork(const Shape& input_shape,
                                             const Shape& complexity_dimensions,
                                             const Shape& output_shape) : NeuralNetwork()
{
    const Index complexity_size = complexity_dimensions.size();

    add_layer(make_unique<Scaling<2>>(input_shape));

    for(Index i = 0; i < complexity_size; i++)
        add_layer(make_unique<Dense<2>>(get_output_shape(),
                                       Shape{complexity_dimensions[i]},
                                       "HyperbolicTangent",
                                       false,
                                       "dense2d_layer_" + to_string(i + 1)));

    add_layer(make_unique<Dense<2>>(get_output_shape(),
                                   output_shape,
                                   output_shape[0] == 1 ? "Sigmoid" : "Softmax",
                                   false,
                                   "classification_layer"));

    this->compile();
    this->set_parameters_random();

    const Index inputs_number = get_inputs_number();
    input_names.resize(inputs_number);

    const Index outputs_number = get_outputs_number();
    output_names.resize(outputs_number);
}


ForecastingNetwork::ForecastingNetwork(const Shape& input_shape,
                                       const Shape& complexity_dimensions,
                                       const Shape& output_shape) : NeuralNetwork()
{
    add_layer(make_unique<Scaling<3>>(input_shape));

    add_layer(make_unique<Recurrent>(input_shape,
                                     complexity_dimensions));

    add_layer(make_unique<Dense<2>>(complexity_dimensions,
                                   output_shape,
                                   "Linear",
                                   "dense_layer"));

    add_layer(make_unique<Unscaling>(output_shape));

    add_layer(make_unique<Bounding>(output_shape));

    this->compile();
    this->set_parameters_random();

    const Index inputs_number = get_inputs_number();
    input_names.resize(inputs_number);

    const Index outputs_number = get_outputs_number();
    output_names.resize(outputs_number);
}


AutoAssociationNetwork::AutoAssociationNetwork(const Shape& input_shape,
                                               const Shape& complexity_dimensions,
                                               const Shape& output_shape) : NeuralNetwork()
{
    add_layer(make_unique<Scaling<2>>(input_shape));

    const Index mapping_neurons_number = 10;
    const Index bottle_neck_neurons_number = complexity_dimensions[0];

    add_layer(make_unique<Dense<2>>(input_shape,
                                   Shape{mapping_neurons_number},
                                   "HyperbolicTangent",
                                   false,
                                   "mapping_layer"));

    add_layer(make_unique<Dense<2>>(Shape{ mapping_neurons_number },
                                   Shape{ bottle_neck_neurons_number },
                                   "Linear",
                                   false,
                                   "bottleneck_layer"));

    add_layer(make_unique<Dense<2>>(Shape{ bottle_neck_neurons_number },
                                   Shape{ mapping_neurons_number },
                                   "HyperbolicTangent",
                                   false,
                                   "demapping_layer"));

    add_layer(make_unique<Dense<2>>(Shape{ mapping_neurons_number },
                                   Shape{ output_shape },
                                   "Linear",
                                   false,
                                   "output_layer"));

    add_layer(make_unique<Unscaling>(output_shape));

    this->compile();
    this->set_parameters_random();
}


ImageClassificationNetwork::ImageClassificationNetwork(const Shape& input_shape,
                                                       const Shape& complexity_dimensions,
                                                       const Shape& output_shape) : NeuralNetwork()
{
    if (input_shape.size() != 3)
        throw runtime_error("Input shape size is not 3.");

    reference_all_layers();

    auto scaling_layer = make_unique<Scaling<4>>(input_shape);
    scaling_layer->set_scalers("ImageMinMax");
    add_layer(move(scaling_layer));

    const Index complexity_size = complexity_dimensions.size();
    
    for(Index i = 0; i < complexity_size; i++)
    {
        const Shape kernel_shape = { 3, 3, get_output_shape()[2], complexity_dimensions[i] };
        const Shape stride_shape = { 1, 1 };
        
        add_layer(make_unique<Convolutional>(get_output_shape(),
                                             kernel_shape,
                                             "RectifiedLinear",
                                             stride_shape,
                                             "Same",
                                             false, // Batch normalization
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
    
    add_layer(make_unique<Flatten<4>>(get_output_shape()));

    add_layer(make_unique<Dense<2>>(get_output_shape(),
                                   output_shape,
                                   "Softmax",
                                   false, // Batch normalization
                                   "dense_2d_layer"));

    this->compile();
    this->set_parameters_random();

    const Index inputs_number = get_inputs_number();
    input_names.resize(inputs_number);

    const Index outputs_number = get_outputs_number();
    output_names.resize(outputs_number);
}


SimpleResNet::SimpleResNet(const Shape& input_shape,
                           const vector<Index>& blocks_per_stage,
                           const Shape& initial_filters,
                           const Shape& output_shape) : NeuralNetwork()
{
    if (input_shape.size() != 3)
        throw runtime_error("Input shape size must be 3.");
    if (blocks_per_stage.size() != initial_filters.size())
        throw runtime_error("blocks_per_stage and initial_filters must have the same size.");

    reference_all_layers();

    add_layer(make_unique<Scaling<4>>(input_shape));

    Index last_layer_index = 0;

    auto stem_conv = make_unique<Convolutional>(get_layer(last_layer_index)->get_output_shape(),
                                                Shape{ 7, 7, input_shape[2], initial_filters[0] },
                                                "RectifiedLinear",
                                                Shape{ 2, 2 },
                                                "Same",
                                                false,
                                                "stem_conv_1");

    add_layer(std::move(stem_conv), { last_layer_index });

    last_layer_index = get_layers_number() - 1;

    auto stem_pool = make_unique<Pooling>(get_layer(last_layer_index)->get_output_shape(),
                                          Shape{ 3, 3 },
                                          Shape{ 2, 2 },
                                          Shape{ 1, 1 },
                                          "MaxPooling",
                                          "stem_pool");

    add_layer(std::move(stem_pool), { last_layer_index });

    last_layer_index = get_layers_number() - 1;

    for(size_t stage = 0; stage < blocks_per_stage.size(); ++stage)
    {
        for(Index block = 0; block < blocks_per_stage[stage]; ++block)
        {
            const Index block_input_index = last_layer_index;

            Shape current_input_shape = get_layer(block_input_index)->get_output_shape();

            const Index filters = initial_filters[stage];

            const Index stride = (stage > 0 && block == 0) ? 2 : 1;

            // Main
            auto conv1 = make_unique<Convolutional>(current_input_shape,
                                                    Shape{ 3, 3, current_input_shape[2], filters },
                                                    "RectifiedLinear",
                                                    Shape{ stride, stride },
                                                    "Same",
                                                    false,
                                                    "s" + to_string(stage) + "b" + to_string(block) + "_conv1");

            add_layer(std::move(conv1), { block_input_index });

            Index main_path_index = get_layers_number() - 1;

            auto conv2 = make_unique<Convolutional>(get_layer(main_path_index)->get_output_shape(),
                                                    Shape{ 3, 3, filters, filters },
                                                    "Linear",
                                                    Shape{ 1, 1 },
                                                    "Same",
                                                    false,
                                                    "s" + to_string(stage) + "b" + to_string(block) + "_conv2");

            add_layer(std::move(conv2), { main_path_index });

            main_path_index = get_layers_number() - 1;

            // Skip Connection
            Index skip_path_index = block_input_index;

            if (stride != 1 || current_input_shape[2] != filters)
            {
                auto skip_conv = make_unique<Convolutional>(current_input_shape,
                                                            Shape{ 1, 1, current_input_shape[2], filters },
                                                            "Linear",
                                                            Shape{ stride, stride },
                                                            "Same",
                                                            false,
                                                            "s" + to_string(stage) + "b" + to_string(block) + "_skip");

                add_layer(std::move(skip_conv), { block_input_index });

                skip_path_index = get_layers_number() - 1;
            }

            const Shape main_out_dims = get_layer(main_path_index)->get_output_shape();

            auto addition_layer = make_unique<Addition<4>>(main_out_dims, "s" + to_string(stage) + "b" + to_string(block) + "_add");

            add_layer(std::move(addition_layer), { main_path_index, skip_path_index });

            last_layer_index = get_layers_number() - 1;

            auto activation_layer = make_unique<Convolutional>(get_layer(last_layer_index)->get_output_shape(),
                                                               Shape{ 1, 1, filters, filters },
                                                               "RectifiedLinear",
                                                               Shape{ 1, 1 },
                                                               "Same",
                                                               false,
                                                               "s" + to_string(stage) + "b" + to_string(block) + "_relu");

            add_layer(std::move(activation_layer), { last_layer_index });

            last_layer_index = get_layers_number() - 1;
        }
    }

    const Shape pre_pool_dims = get_layer(last_layer_index)->get_output_shape();

    auto global_pool = make_unique<Pooling>(pre_pool_dims,
                                            Shape{ pre_pool_dims[0], pre_pool_dims[1] },
                                            Shape{ 1, 1 },
                                            Shape{ 0, 0 },
                                            "AveragePooling",
                                            "global_avg_pool");

    add_layer(std::move(global_pool), { last_layer_index });

    last_layer_index = get_layers_number() - 1;

    auto flatten_layer = make_unique<Flatten<2>>(get_layer(last_layer_index)->get_output_shape());

    add_layer(std::move(flatten_layer), { last_layer_index });

    last_layer_index = get_layers_number() - 1;

    auto dense_layer = make_unique<Dense<2>>(get_layer(last_layer_index)->get_output_shape(),
                                            output_shape,
                                            "Softmax",
                                            false,
                                            "dense_classifier");

    add_layer(std::move(dense_layer), { last_layer_index });

    this->compile();
    this->set_parameters_random();
}


VGG16::VGG16(const Shape& new_input_shape, const Shape& new_target_shape)
    : NeuralNetwork()
{
    set(new_input_shape, new_target_shape);
}


void VGG16::set(const Shape& new_input_shape, const Shape& new_target_shape)
{
    reference_all_layers();

    // Scaling 4D
    add_layer(make_unique<Scaling<4>>(new_input_shape));

    // --- Conv 3×3, 64 kernels, ReLU x2 -> Pooling 2×2 stride 2 ---
    {
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, new_input_shape[2], 64 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_1"));
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 64, 64 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_2"));
        add_layer(make_unique<Pooling>(
            get_output_shape(),
            Shape{ 2, 2 },
            Shape{ 2, 2 },
            Shape{ 0, 0 },
            "MaxPooling",
            "pool1"));
    }

    // --- Conv 3×3, 128 kernels, ReLU x2 -> Pooling 2×2 stride 2 ---
    {
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 64, 128 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_3"));
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 128, 128 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_4"));
        add_layer(make_unique<Pooling>(
            get_output_shape(),
            Shape{ 2, 2 },
            Shape{ 2, 2 },
            Shape{ 0, 0 },
            "MaxPooling",
            "pool2"));
    }

    // --- Conv 3×3, 256 kernels, ReLU x3 -> Pooling 2×2 stride 2 ---
    {
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 128, 256 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_5"));
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 256, 256 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_6"));
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 256, 256 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_7"));
        add_layer(make_unique<Pooling>(
            get_output_shape(),
            Shape{ 2, 2 },
            Shape{ 2, 2 },
            Shape{ 0, 0 },
            "MaxPooling", "pool3"));
    }

    // --- Conv 3×3, 512 kernels, ReLU x3 -> Pooling 2×2 stride 2 ---
    {
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 256, 512 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_8"));
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 512, 512 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_9"));
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 512, 512 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_10"));
        add_layer(make_unique<Pooling>(
            get_output_shape(),
            Shape{ 2, 2 },
            Shape{ 2, 2 },
            Shape{ 0, 0 },
            "MaxPooling", "pool4"));
    }

    // --- Conv 3×3, 512 kernels, ReLU x3 -> Pooling 2×2 stride 2 ---
    {
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 512, 512 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_11"));
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 512, 512 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_12"));
        add_layer(make_unique<Convolutional>(
            get_output_shape(),
            Shape{ 3, 3, 512, 512 },
            "RectifiedLinear",
            Shape{ 1, 1 },
            "Same",
            "conv_13"));
        add_layer(make_unique<Pooling>(
            get_output_shape(),
            Shape{ 2, 2 },
            Shape{ 2, 2 },
            Shape{ 0, 0 },
            "MaxPooling", "pool5"));
    }

    const Shape pre_pool_dims = get_output_shape();

    add_layer(make_unique<Pooling>(
        pre_pool_dims,
        Shape{ pre_pool_dims[0], pre_pool_dims[1] },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        "AveragePooling",
        "global_avg_pool"));

    // Flatten
    add_layer(make_unique<Flatten<2>>(get_output_shape()));

    //Classifier
    add_layer(make_unique<Dense<2>>(get_output_shape(),
                                   new_target_shape,
                                   "Softmax",
                                   false,
                                   "dense_classifier"));

    this->compile();
    this->set_parameters_random();
}


VGG16::VGG16(const filesystem::path& file_name)
    : NeuralNetwork(file_name)
{

}


TextClassificationNetwork::TextClassificationNetwork(const Shape& input_shape,
                                                     const Shape& complexity_dimensions,
                                                     const Shape& output_shape,
                                                     const vector<string>& new_input_vocabulary) : NeuralNetwork()
{
    layers.clear();

    reference_all_layers();

    const Index vocabulary_size = input_shape[0];
    const Index sequence_length = input_shape[1];
    const Index embedding_dimension = input_shape[2];

    const Index heads_number = complexity_dimensions[0];
    //const bool use_causal_mask = false;

    const string classification_layer_activation = output_shape[0] == 1 ? "Sigmoid" : "Softmax";

    add_layer(make_unique<Embedding>(Shape({vocabulary_size, sequence_length}),
                                     embedding_dimension,
                                     "embedding_layer"));

    add_layer(make_unique<MultiHeadAttention>(
         Shape({sequence_length, embedding_dimension}),
         heads_number,
         "multihead_attention_layer"));

    add_layer(make_unique<Pooling3d>(
        get_output_shape()));

    // add_layer(make_unique<Flatten<3>>(
    //     get_output_shape()
    //     ));

    add_layer(make_unique<Dense<2>>(
        get_output_shape(),
        output_shape,
        classification_layer_activation,
        "classification_layer"));

    input_vocabulary = new_input_vocabulary;

    this->compile();
    this->set_parameters_random();
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

    if (input_sequence_length == 0 || decoder_sequence_length == 0)
        return;

    input_names.resize(input_sequence_length + decoder_sequence_length);

    // Embedding Layers: vocabulary size, sequence length, embedding dimension

    auto decoder_embedding = make_unique<Embedding>(Shape{output_vocabulary_size, decoder_sequence_length},
                                                    embedding_dimension,
                                                    "decoder_embedding");

    decoder_embedding->set_scale_embedding(true);
    decoder_embedding->set_add_positional_encoding(true);

    add_layer(std::move(decoder_embedding), {-1});

    Index current_dec_idx = 0;

    auto input_embedding = make_unique<Embedding>(Shape{input_vocabulary_size, input_sequence_length},
                                                  embedding_dimension,
                                                  "input_embedding");

    input_embedding->set_scale_embedding(true);
    input_embedding->set_add_positional_encoding(true);

    add_layer(std::move(input_embedding), {-2});

    Index current_enc_idx = 1;

    // Encoder

    for(Index i = 0; i < layers_number; i++)
    {
        add_layer(make_unique<MultiHeadAttention>(Shape{input_sequence_length, embedding_dimension},
                                                  heads_number,
                                                  "input_self_attention_" + to_string(i+1)),
                  {current_enc_idx});

        const Index attn_idx = get_layers_number() - 1;

        add_layer(make_unique<Addition<3>>(Shape{input_sequence_length, embedding_dimension},
                                           "input_self_attention_addition_" + to_string(i+1)),
                  {current_enc_idx, attn_idx});

        add_layer(make_unique<Normalization3d>(Shape{input_sequence_length, embedding_dimension},
                                               "input_self_attention_normalization_" + to_string(i+1)));

        const Index norm_1_idx = get_layers_number() - 1;

        add_layer(make_unique<Dense<3>>(input_sequence_length,
                                       embedding_dimension,
                                       feed_forward_dimension,
                                       "RectifiedLinear",
                                       "encoder_internal_dense_" + to_string(i+1)));

        add_layer(make_unique<Dense<3>>(input_sequence_length,
                                       feed_forward_dimension,
                                       embedding_dimension,
                                       "HyperbolicTangent",
                                       "encoder_external_dense_" + to_string(i+1)));

        const Index ff_idx = get_layers_number() - 1;

        add_layer(make_unique<Addition<3>>(Shape{input_sequence_length, embedding_dimension},
                                           "encoder_dense_addition_" + to_string(i+1)),
                  {norm_1_idx, ff_idx});

        add_layer(make_unique<Normalization3d>(Shape{input_sequence_length, embedding_dimension},
                                               "encoder_dense_normalization_" + to_string(i+1)));

        current_enc_idx = get_layers_number() - 1;
    }

    // Decoder

    const Index encoder_final_output_idx = current_enc_idx;

    for(Index i = 0; i < layers_number; i++)
    {
        // chatgpt says that here uses causal mask???

        add_layer(make_unique<MultiHeadAttention>(Shape{decoder_sequence_length, embedding_dimension},
                                                  heads_number,
                                                  "decoder_self_attention_" + to_string(i+1)),
                  {current_dec_idx});

        const Index self_attn_idx = get_layers_number() - 1;

        add_layer(make_unique<Addition<3>>(Shape{decoder_sequence_length, embedding_dimension},
                                           "decoder_self_attention_addition_" + to_string(i+1)),
                  {current_dec_idx, self_attn_idx});

        add_layer(make_unique<Normalization3d>(Shape{decoder_sequence_length, embedding_dimension},
                                               "decoder_self_attention_normalization_" + to_string(i+1)));

        const Index norm_1_idx = get_layers_number() - 1;

        add_layer(make_unique<MultiHeadAttention>(Shape{decoder_sequence_length, embedding_dimension},
                                                  Shape{input_sequence_length, embedding_dimension},
                                                  heads_number,
                                                  "cross_attention_" + to_string(i+1)),
                  {norm_1_idx, encoder_final_output_idx});

        const Index cross_attn_idx = get_layers_number() - 1;

        add_layer(make_unique<Addition<3>>(Shape{decoder_sequence_length, embedding_dimension},
                                           "cross_attention_addition_" + to_string(i+1)),
                  {norm_1_idx, cross_attn_idx});

        add_layer(make_unique<Normalization3d>(Shape{decoder_sequence_length, embedding_dimension},
                                               "cross_attention_normalization_" + to_string(i+1)));

        const Index norm_2_idx = get_layers_number() - 1;

        add_layer(make_unique<Dense<3>>(decoder_sequence_length,
                                       embedding_dimension,
                                       feed_forward_dimension,
                                       "RectifiedLinear",
                                       "decoder_internal_dense_" + to_string(i+1)));

        add_layer(make_unique<Dense<3>>(decoder_sequence_length,
                                       feed_forward_dimension,
                                       embedding_dimension,
                                       "HyperbolicTangent",
                                       "decoder_external_dense_" + to_string(i+1)));

        const Index ff_idx = get_layers_number() - 1;

        add_layer(make_unique<Addition<3>>(Shape{decoder_sequence_length, embedding_dimension},
                                           "decoder_dense_addition_" + to_string(i+1)),
                  {norm_2_idx, ff_idx});

        add_layer(make_unique<Normalization3d>(Shape{decoder_sequence_length, embedding_dimension},
                                               "decoder_dense_normalization_" + to_string(i+1)));

        current_dec_idx = get_layers_number() - 1;
    }

    add_layer(make_unique<Dense<3>>(decoder_sequence_length,
                                   embedding_dimension,
                                   output_vocabulary_size,
                                   "Softmax", // Change from "Linear" to "Softmax"
                                   "output_projection"));

    this->compile();
    this->set_parameters_random();
}


void Transformer::set_dropout_rate(const type new_dropout_rate)
{
}


void Transformer::set_input_vocabulary(const vector<string>& new_input_vocabulary)
{
    input_vocabulary = new_input_vocabulary;
}


void Transformer::set_output_vocabulary(const vector<string>& new_output_vocabulary)
{
    output_vocabulary = new_output_vocabulary;
}


Index Transformer::get_input_sequence_length() const
{
    return get_layer("enc_embed")->get_input_shape()[0];
}


Index Transformer::get_decoder_sequence_length() const
{
    return get_layer("dec_embed")->get_input_shape()[0];
}


Index Transformer::get_embedding_dimension() const
{
    return get_layer(0)->get_output_shape().back();
}

Index Transformer::get_heads_number() const
{
    for(const auto& layer : layers)
        if(layer->get_name() == "MultiHeadAttention")
            return static_cast<MultiHeadAttention*>(layer.get())->get_heads_number();

    return 0;
}


string Transformer::calculate_outputs(const string& source)
{
    if (input_vocabulary_map.empty() || output_inverse_vocabulary_map.empty())
        throw runtime_error("Transformer::calculate_outputs Error: Vocabularies not initialized.");

    constexpr type PAD   = 0.0f;
    constexpr type UNK   = 1.0f;
    constexpr type START = 2.0f;
    constexpr type END   = 3.0f;

    const Index input_sequence_length = get_input_sequence_length();
    const Index decoder_sequence_length = get_decoder_sequence_length();
    const Index batch_size = 1;

    const vector<string> source_tokens = tokenize(source);

    Tensor2 source_ids(batch_size, input_sequence_length);
    source_ids.setConstant(PAD);

    for(size_t i = 0; i < source_tokens.size() && i < static_cast<size_t>(input_sequence_length); i++)
    {
        const auto it = input_vocabulary_map.find(source_tokens[i]);

        source_ids(0, i) = (it != input_vocabulary_map.end())
                               ? static_cast<type>(it->second)
                               : UNK;
    }

    Tensor2 target_ids(batch_size, decoder_sequence_length);
    target_ids.setConstant(PAD);
    target_ids(0, 0) = START;

    ForwardPropagation forward_propagation(batch_size, this);

    for(Index i = 1; i < decoder_sequence_length; i++)
    {
        const vector<TensorView> inputs = {TensorView(target_ids.data(), {batch_size, decoder_sequence_length}),
                                           TensorView(source_ids.data(), {batch_size, input_sequence_length})};

        forward_propagate(inputs, forward_propagation, false);

        const TensorView output_view = forward_propagation.get_outputs();

        const Index vocabulary_size = output_view.shape[2];

        const TensorMap3 probabilities(output_view.data, batch_size, decoder_sequence_length, vocabulary_size);

        // GREEDY SELECTION:
        // The prediction for the "next" word is found at the output of the current word's index.
        // We look at the distribution for position i-1 to pick word for position i.
        //const VectorR current_distribution = probabilities.chip(0, 0).chip(i-1, 0);

        type* distribution_ptr = output_view.data + (i - 1) * vocabulary_size;

        const Map<const VectorR> current_distribution(distribution_ptr, vocabulary_size);

        const Index best_id = maximal_index(current_distribution);

        target_ids(0, i) = static_cast<type>(best_id);

        if(best_id == 3) break; // [END]
    }

    string result;

    for(Index i = 1; i < decoder_sequence_length; i++)
    {
        const Index id = static_cast<Index>(target_ids(0, i));

        if(id == END || id == PAD) break;

        const auto it = output_inverse_vocabulary_map.find(id);

        if (it == output_inverse_vocabulary_map.end())
            continue;

        if(!result.empty())
            result += " ";

        result += it->second;
    }

    return result;
}

} 

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
