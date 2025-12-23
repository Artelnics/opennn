//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "transformer.h"
#include "tensors.h"
#include "embedding_layer.h"
#include "normalization_layer_3d.h"
#include "multihead_attention_layer.h"
#include "addition_layer.h"
#include "dense_layer_3d.h"
#include "probabilistic_layer_3d.h"

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
    set(decoder_length,                           //Bruno's input is the decoder
        input_length,                             //Bruno's context is the input
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

    input_length = new_input_length;
    decoder_length = new_decoder_length;

    layers.clear();
    
    if (input_length == 0 || decoder_length == 0)
        return;

    feature_names.resize(input_length + decoder_length);

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
    
    add_layer(make_unique<Probabilistic3d>(new_decoder_length,
                                           new_embedding_dimension,
                                           new_decoder_dimensions,
                                           "probabilistic"));

    set_layer_inputs_indices("probabilistic", "decoder_perceptron_normalization_" + to_string(new_blocks_number));
}


void Transformer::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void Transformer::set_input_vocabulary(const vector<string>& new_input_vocabulary)
{
    input_vocabulary = new_input_vocabulary;
}


void Transformer::set_output_vocabulary(const vector<string>& new_output_vocabulary)
{
    output_vocabulary = new_output_vocabulary;
}


void Transformer::set_input_length(const Index& new_input_length)
{
    input_length = new_input_length;
}


void Transformer::set_decoder_length(const Index& new_decoder_length)
{
    decoder_length = new_decoder_length;
}


Index Transformer::get_maximum_input_sequence_length() const
{
    return input_length;
}


Index Transformer::get_decoder_length() const
{
    return decoder_length;
}

string Transformer::calculate_outputs(const vector<string>& input_string)
{

    type start_indicator = 2;
    type end_indicator = 3;

    //@todo
    vector<vector<string>> input_tokens(input_string.size());
    for(size_t i = 0; i <input_tokens.size(); i++)
        input_tokens[i] = preprocess_language_document(input_string[i], true);

    const Index samples_number = 1;

    Tensor<type, 2> input(samples_number, input_length);
    input.setZero();

    tokenize_wordpiece(input_tokens[0], input);

    cout << "Input codification:\n" << input << endl;

    Tensor<type, 2> decoder(samples_number, decoder_length);

    decoder.setZero();
    decoder(0) = start_indicator;

    ForwardPropagation forward_propagation(samples_number, this);

    const TensorView input_pair(input.data(), { samples_number, input_length });
    const TensorView decoder_pair(decoder.data(), { samples_number, decoder_length });

    const vector<TensorView> input_views = {decoder_pair, input_pair};

    const Index layers_number = get_layers_number();

    const TensorView outputs_view 
        = forward_propagation.layers[layers_number - 1]->get_output_pair();

    TensorMap <Tensor<type, 2>> outputs(outputs_view.data, outputs_view.dims[1], outputs_view.dims[2]);
    outputs.setZero();

    Tensor<type, 1> current_outputs(outputs_view.dims[2]);
    current_outputs.setZero();

    Tensor<Index, 0> prediction;

    cout << "Output dimensions: " << outputs.dimensions() << endl;

    for(Index i = 1; i < decoder_length; i++)
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


vector<string> Transformer::preprocess_language_document(const string &document, const bool &input)
{
    vector<string> tokens;

    // if(!input)
    tokens.push_back("[START]");

    string currentToken;

    for (char c : document)
    {
        if (isalnum(c))
        {
            // Add alphanumeric characters to the current token
            currentToken += tolower(c);
        }
        else
        {
            // If the current token is not empty, add it to the tokens list
            if (!currentToken.empty())
            {
                tokens.push_back(currentToken);
                currentToken.clear();
            }
            // Treat punctuation as a separate token

            if (ispunct(c))
            {
                tokens.push_back(string(1, c));
            }
            else if (isspace(c))
            {
                // Ignore spaces, they just delimit tokens
            }
        }
    }

    // Add the last token if it's not empty
    if (!currentToken.empty())
        tokens.push_back(currentToken);

    // Add [END] token
    // if(!input)
    tokens.push_back("[END]");

    return tokens;
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


void Transformer::tokenize_whitespace(const vector<string>& context_tokens, Tensor<type, 2>& context)
{

    bool line_ended = false;

    for(Index j = 0; j < input_length - 1; j++)
    {
        if(j < Index(context_tokens.size()))
        {
/*
            auto it = input_vocabulary.find(context_tokens[j]);

            const Index word_index = (it != input_vocabulary.end()) ? it->second : 0;

            context(j + 1) = type(word_index);
*/
        }
        else
        {
            if(j == Index(context_tokens.size()) || (j == input_length - 2 && !line_ended))
            {
                context(j + 1) = 3; // end indicator
                line_ended = true;
            }
            else
            {
                break;
            }
        }
    }

}


void Transformer::tokenize_wordpiece(const vector<string>& context_tokens, Tensor<type, 2>& context)
{
/*
    // unordered_map<string, type> context_vocabulary_map;

    // for(Index i = 0; i < input_vocabulary.size(); i++)
    //     context_vocabulary_map[input_vocabulary[i]] = type(i);

    Index token_counter = 0;
    //bool line_ended = false;

    string word;
    string wordpiece;
    string rest;

    auto wordpiece_entry = input_vocabulary.find("");
    bool tokenized;

    for(Index j = 0; j < input_length - 1; j++)
    {
        if(j < Index(context_tokens.size()) && token_counter < input_length - 1)
        {
            word = context_tokens[j];

            wordpiece_entry = input_vocabulary.find(word);

            if(wordpiece_entry != input_vocabulary.end())
            {
                context(token_counter++) = wordpiece_entry->second;
                continue;
            }

            tokenized = false;

            for(Index wordpiece_length = word.length(); wordpiece_length > 0; wordpiece_length--)
            {
                if(token_counter == input_length - 1)
                {
                    tokenized = true;
                    break;
                }

                wordpiece = word.substr(0, wordpiece_length);
                wordpiece_entry = input_vocabulary.find(wordpiece);

                if(wordpiece_entry != input_vocabulary.end())
                {
                    context(token_counter++) = wordpiece_entry->second;

                    rest = word.substr(wordpiece_length);

                    if(rest.empty())
                    {
                        tokenized = true;
                        break;
                    }

                    word = "##" + rest;
                    wordpiece_length = word.length() + 1;
                }
            }

            if(!tokenized)
                context(token_counter++) = 1; // unknown indicator
        }
        else
        {
            // if(j == Index(context_tokens.size())
            // || (token_counter == input_length - 1 && !line_ended))
            // {
            //     context(token_counter++) = 3; // end indicator
            //     line_ended = true;
            // }
            // else
            // {
                break;
            // }
        }
    }
*/
}

void Transformer::detokenize_whitespace(Tensor<type, 2>& predictions, ostringstream& output_string)
{
/*
    for(Index i = 1; i < decoder_length; i++)
    {
        if(predictions(i) == 2) break;

        for (const auto& pair : output_vocabulary)
        {
            if (pair.second == Index(predictions(i)))
            {
                output_string << pair.first << " ";
                break;
            }
        }
    }
*/
}


void Transformer::detokenize_wordpiece(Tensor<type, 2>& predictions, ostringstream& buffer)
{
/*
    for (const auto& pair : output_vocabulary) {
        if (pair.second == Index(predictions(1))) {
            buffer << pair.first;
            break;
        }
    }

    string current_prediction;

    for(Index i = 2; i < decoder_length; i++)
    {
        if(predictions(i) == 3) // [END] token
            break;

        for (const auto& pair : output_vocabulary) {
            if (pair.second == Index(predictions(i))) {
                current_prediction = pair.first;
                break;
            }
        }

        current_prediction.substr(0, 2) == "##"
            ? buffer << current_prediction.substr(2)
            : buffer << " " << current_prediction;
    }
*/
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
