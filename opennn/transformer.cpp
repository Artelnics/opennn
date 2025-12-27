//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "transformer.h"
#include "tensors.h"
#include "strings_utilities.h"
#include "embedding_layer.h"
#include "normalization_layer_3d.h"
#include "multihead_attention_layer.h"
#include "addition_layer.h"
#include "dense_layer_3d.h"

namespace opennn
{

Transformer::Transformer(const Index& decoder_length,
                         const Index& input_length,
                         const Index& decoder_dimensions_xxx,
                         const Index& input_dimension_xxx,
                         const Index& embedding_dimension,
                         const Index& perceptron_depth,
                         const Index& heads_number,
                         const Index& layers_number)
{
    set(decoder_length, //Bruno's input is the decoder
        input_length,   //Bruno's context is the input
        decoder_dimensions_xxx,
        input_dimension_xxx,
        embedding_dimension,
        perceptron_depth,
        heads_number,
        layers_number);
}


void Transformer::set(const Index& new_decoder_length,
                      const Index& new_input_length,
                      const Index& new_decoder_dimensions,
                      const Index& new_input_dimension,
                      const Index& new_embedding_dimension,
                      const Index& new_perceptron_depth,
                      const Index& new_heads_number,
                      const Index& new_blocks_number)
{
    throw runtime_error("Transformer is not yet implemented. Please check back in a future version.");

    name = "transformer";

    layers.clear();
    layer_input_indices.clear();

    if (new_input_length == 0 || new_decoder_length == 0)
        return;

    feature_names.resize(new_input_length + new_decoder_length);

    // Embedding Layers

    add_layer(make_unique<Embedding>(dimensions({new_decoder_dimensions, new_decoder_length}),
                                     new_embedding_dimension,
                                     "decoder_embedding"));

    set_layer_inputs_indices("decoder_embedding", "decoder");

    //decoder_embedding_layer->set_dropout_rate(dropout_rate);

    add_layer(make_unique<Embedding>(dimensions({new_input_dimension, new_input_length}),
                                     new_embedding_dimension,
                                     "input_embedding"));

    set_layer_inputs_indices("input_embedding", "input");
    //input_embedding_layer->set_dropout_rate(dropout_rate);

    // Encoder

    for(Index i = 0; i < new_blocks_number; i++)
    {
        add_layer(make_unique<MultiHeadAttention>(dimensions({new_input_length, new_embedding_dimension}),
                                                  new_heads_number,
                                                  "input_self_attention_" + to_string(i+1)));

        i == 0
            ? set_layer_inputs_indices("input_self_attention_1",
                                      {"input_embedding", "input_embedding"})
            : set_layer_inputs_indices("input_self_attention_" + to_string(i+1),
                                      {"encoder_perceptron_normalization_" + to_string(i), "encoder_perceptron_normalization_" + to_string(i)});

        //input_self_attention_layer->set_dropout_rate(dropout_rate);

        // Addition

        add_layer(make_unique<Addition<3>>(dimensions({new_input_length, new_embedding_dimension}),
                                          "input_self_attention_addition_" + to_string(i+1)));

        i == 0
            ? set_layer_inputs_indices("input_self_attention_addition_" + to_string(i+1),
                                       { "input_embedding", "input_self_attention_" + to_string(i+1) })
            : set_layer_inputs_indices("input_self_attention_addition_" + to_string(i+1),
                                       { "encoder_perceptron_normalization_" + to_string(i), "input_self_attention_" + to_string(i+1) });

        // Normalization
        
        add_layer(make_unique<Normalization3d>(dimensions({new_input_length, new_embedding_dimension}),
                                               "input_self_attention_normalization_" + to_string(i+1)));
        
        set_layer_inputs_indices("input_self_attention_normalization_" + to_string(i+1), "input_self_attention_addition_" + to_string(i+1));
        
        // Dense2d

        add_layer(make_unique<Dense3d>(new_input_length,
                                       new_embedding_dimension,
                                       new_perceptron_depth,
                                       "RectifiedLinear",
                                       "encoder_internal_perceptron_" + to_string(i+1)));
        
        set_layer_inputs_indices("encoder_internal_perceptron_" + to_string(i+1), "input_self_attention_normalization_" + to_string(i+1));

        // Dense2d

        add_layer(make_unique<Dense3d>(new_input_length,
                                       new_perceptron_depth,
                                       new_embedding_dimension,
                                       "HyperbolicTangent",
                                       "encoder_external_perceptron_" + to_string(i+1)));

        set_layer_inputs_indices("encoder_external_perceptron_" + to_string(i+1), "encoder_internal_perceptron_" + to_string(i+1));

//        encoder_external_perceptron_layer->set_dropout_rate(dropout_rate);

        add_layer(make_unique<Addition<3>>(dimensions({new_input_length, new_embedding_dimension}),
                                          "encoder_perceptron_addition_" + to_string(i+1)));
        
        set_layer_inputs_indices("encoder_perceptron_addition_" + to_string(i+1),
                                 { "input_self_attention_normalization_" + to_string(i+1), "encoder_external_perceptron_" + to_string(i+1)});
        
        add_layer(make_unique<Normalization3d>(dimensions({new_input_length, new_embedding_dimension}),
                                               "encoder_perceptron_normalization_" + to_string(i+1)));
        
        set_layer_inputs_indices("encoder_perceptron_normalization_" + to_string(i+1), "encoder_perceptron_addition_" + to_string(i+1));
    }
    
    // Decoder

    for(Index i = 0; i < new_blocks_number; i++)
    {
        // chatgpt says that here uses causal mask???

        add_layer(make_unique<MultiHeadAttention>(dimensions({new_decoder_length, new_embedding_dimension}),
                                                  new_heads_number,
                                                  "decoder_self_attention_" + to_string(i+1)));


        i == 0
            ? set_layer_inputs_indices("decoder_self_attention_1",
                                       {"decoder_embedding", "decoder_embedding"})
            : set_layer_inputs_indices("decoder_self_attention_" + to_string(i+1),
                                       {"decoder_perceptron_normalization_" + to_string(i), "decoder_perceptron_normalization_" + to_string(i)});

        //decoder_self_attention_layer->set_dropout_rate(dropout_rate);

        add_layer(make_unique<Addition<3>>(dimensions({new_decoder_length, new_embedding_dimension}),
                                          "decoder_self_attention_addition_" + to_string(i+1)));
        i == 0
            ? set_layer_inputs_indices("decoder_self_attention_addition_" + to_string(i+1),
                                       { "decoder_embedding", "decoder_self_attention_" + to_string(i+1) })
            : set_layer_inputs_indices("decoder_self_attention_addition_" + to_string(i+1),
                                       { "decoder_perceptron_normalization_" + to_string(i), "decoder_self_attention_" + to_string(i+1) });

        add_layer(make_unique<Normalization3d>(dimensions({new_decoder_length, new_embedding_dimension}),
                                               "decoder_self_attention_normalization_" + to_string(i+1)));

        set_layer_inputs_indices("decoder_self_attention_normalization_" + to_string(i+1), "decoder_self_attention_addition_" + to_string(i+1));

        add_layer(make_unique<MultiHeadAttention>(dimensions({new_decoder_length, new_embedding_dimension}),
                                                  dimensions({new_input_length, new_embedding_dimension}),
                                                  new_heads_number,
                                                  //false,
                                                  "cross_attention_" + to_string(i+1)));

        set_layer_inputs_indices("cross_attention_" + to_string(i+1), {"decoder_self_attention_normalization_" + to_string(i+1), "encoder_perceptron_normalization_" + to_string(new_blocks_number)});

        //cross_attention_layer->set_dropout_rate(dropout_rate);

        add_layer(make_unique<Addition<3>>(dimensions({new_decoder_length, new_embedding_dimension}),
                                          "cross_attention_addition_" + to_string(i+1)));
        
        set_layer_inputs_indices("cross_attention_addition_" + to_string(i+1), { "decoder_self_attention_normalization_" + to_string(i+1), "cross_attention_" + to_string(i+1) });

        add_layer(make_unique<Normalization3d>(dimensions({new_decoder_length, new_embedding_dimension}),
                                               "cross_attention_normalization_" + to_string(i+1)));

        set_layer_inputs_indices("cross_attention_normalization_" + to_string(i+1), "cross_attention_addition_" + to_string(i+1));

        add_layer(make_unique<Dense3d>(new_decoder_length,
                                       new_embedding_dimension,
                                       new_perceptron_depth,
                                       "RectifiedLinear",
                                       "decoder_internal_perceptron_" + to_string(i+1)));
        
        set_layer_inputs_indices("decoder_internal_perceptron_" + to_string(i+1), "cross_attention_normalization_" + to_string(i+1));

        add_layer(make_unique<Dense3d>(new_decoder_length,
                                       new_perceptron_depth,
                                       new_embedding_dimension,
                                       "HyperbolicTangent",
                                       "decoder_external_perceptron_" + to_string(i+1)));

        set_layer_inputs_indices("decoder_external_perceptron_" + to_string(i+1), "decoder_internal_perceptron_" + to_string(i+1));

        add_layer(make_unique<Addition<3>>(dimensions({new_decoder_length, new_embedding_dimension}),
                                               "decoder_perceptron_addition_" + to_string(i+1)));

        set_layer_inputs_indices("decoder_perceptron_addition_" + to_string(i+1), { "cross_attention_normalization_" + to_string(i+1), "decoder_external_perceptron_" + to_string(i+1) });

        add_layer(make_unique<Normalization3d>(dimensions({new_decoder_length, new_embedding_dimension}),
                                               "decoder_perceptron_normalization_" + to_string(i+1)));

        set_layer_inputs_indices("decoder_perceptron_normalization_" + to_string(i+1), "decoder_perceptron_addition_" + to_string(i+1));
    }
    
    add_layer(make_unique<Dense3d>(new_decoder_length,
                                   new_embedding_dimension,
                                   new_decoder_dimensions,
                                   "output"));

    set_layer_inputs_indices("output", "decoder_perceptron_normalization_" + to_string(new_blocks_number));
}


void Transformer::set_dropout_rate(const type& new_dropout_rate)
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


Index Transformer::get_input_sequence_length() const {
    // We know our architecture: encoder embedding is usually Layer 1
    // or we can find it by name/type
    return get_layer("enc_embed")->get_input_dimensions()[0];
}

Index Transformer::get_decoder_sequence_length() const {
    return get_layer("dec_embed")->get_input_dimensions()[0];
}

Index Transformer::get_embedding_dimension() const {
    // Every layer in a transformer uses the same embedding dimension
    return get_layer(0)->get_output_dimensions().back();
}

Index Transformer::get_heads_number() const {
    // Find the first MultiHeadAttention layer and ask it
    for(const auto& l : layers) {
        if(l->get_name() == "MultiHeadAttention") {
            return static_cast<MultiHeadAttention*>(l.get())->get_heads_number();
        }
    }
    return 0;
}


string Transformer::calculate_outputs(const vector<string>& input_string)
{
    const Index input_sequence_length = get_input_sequence_length();
    const Index decoder_sequence_length = get_decoder_sequence_length();

    type start_indicator = 2;
    type end_indicator = 3;

    //@todo
    vector<vector<string>> input_tokens(input_string.size());
    for(size_t i = 0; i <input_tokens.size(); i++)
        input_tokens[i] = preprocess_language_document(input_string[i], true);

    const Index samples_number = 1;

    Tensor<type, 2> input(samples_number, input_sequence_length);
    input.setZero();

    tokenize_wordpiece(input_tokens[0], input);

    cout << "Input codification:\n" << input << endl;

    Tensor<type, 2> decoder(samples_number, decoder_sequence_length);

    decoder.setZero();
    decoder(0) = start_indicator;

    ForwardPropagation forward_propagation(samples_number, this);

    const TensorView input_pair(input.data(), { samples_number, input_sequence_length });
    const TensorView decoder_pair(decoder.data(), { samples_number, decoder_sequence_length });

    const vector<TensorView> input_views = {decoder_pair, input_pair};

    const Index layers_number = get_layers_number();

    const TensorView outputs_view 
        = forward_propagation.layers[layers_number - 1]->get_output_view();

    TensorMap <Tensor<type, 2>> outputs(outputs_view.data, outputs_view.dims[1], outputs_view.dims[2]);
    outputs.setZero();

    Tensor<type, 1> current_outputs(outputs_view.dims[2]);
    current_outputs.setZero();

    Tensor<Index, 0> prediction;

    cout << "Output dimensions: " << outputs.dimensions() << endl;

    for(Index i = 1; i < decoder_sequence_length; i++)
    {
        forward_propagate(input_views, forward_propagation, false);

        current_outputs = outputs.chip(i - 1, 0);

        prediction = current_outputs.argmax();

        decoder(i) = type(prediction(0));

        if(prediction(0) == end_indicator)
            break;
    }

    ostringstream output_buffer;

    cout << "Output codification:\n" << decoder << endl;

    detokenize_wordpiece(decoder, output_buffer);

    return output_buffer.str();   

    return string();
}


Tensor<type, 3> Transformer::calculate_outputs(const Tensor<type, 2>& input, const Tensor<type, 2>& context)
{
    const Index samples_number = input.dimension(0);

    const TensorView input_pair((type*)input.data(), { samples_number, input.dimension(1) });
    const TensorView context_pair((type*)context.data(), { samples_number, context.dimension(1) });

    const vector<TensorView> input_views = { input_pair, context_pair };

    ForwardPropagation forward_propagation(samples_number, this);

    forward_propagate(input_views, forward_propagation, false);

    const TensorView output_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    return tensor_map<3>(output_pair);
}

};

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
