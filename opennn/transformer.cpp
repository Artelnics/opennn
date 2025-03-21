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
#include "addition_layer_3d.h"
#include "perceptron_layer_3d.h"
#include "probabilistic_layer_3d.h"
#include "forward_propagation.h"

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
    set(decoder_length,                           //Bruno's input was the decoder
        input_length,                             //Bruno's context was the input
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
    name = "transformer";

    model_type = ModelType::TextClassification;

    input_length = new_input_length;
    decoder_length = new_decoder_length;

    layers.clear();
    
    if (input_length == 0 || decoder_length == 0)
        return;

    input_names.resize(input_length + decoder_length);

    // Embedding Layers

    add_layer(make_unique<EmbeddingLayer>(new_decoder_dimensions,
                                          new_decoder_length,
                                          new_embedding_dimension,
                                          "decoder_embedding"));

    set_layer_inputs_indices("decoder_embedding", "decoder");
    //decoder_embedding_layer->set_dropout_rate(dropout_rate);

    add_layer(make_unique<EmbeddingLayer>(new_input_dimension,
                                          new_input_length,
                                          new_embedding_dimension,
                                          "input_embedding"));

    set_layer_inputs_indices("input_embedding", "input");
    //input_embedding_layer->set_dropout_rate(dropout_rate);

    // Encoder

    for(Index i = 0; i < new_blocks_number; i++)
    {
        add_layer(make_unique<MultiHeadAttention>(new_input_length,
                                                       new_input_length,
                                                       new_embedding_dimension,
                                                       new_heads_number,
                                                       false,
                                                       "input_self_attention_" + to_string(i+1)));

        i == 0
            ? set_layer_inputs_indices("input_self_attention_1",
                                    {"input_embedding", "input_embedding"})
            : set_layer_inputs_indices("input_self_attention_" + to_string(i+1),
                { "encoder_perceptron_normalization_" + to_string(i), "encoder_perceptron_normalization_" + to_string(i) });

        //input_self_attention_layer->set_dropout_rate(dropout_rate);

        // Addition

        add_layer(make_unique<AdditionLayer3D>(new_input_length,
                                               new_embedding_dimension,
                                               "input_self_attention_addition_" + to_string(i+1)));

        i == 0
            ? set_layer_inputs_indices("input_self_attention_addition_" + to_string(i+1), { "input_embedding", "input_self_attention_" + to_string(i+1) })
            : set_layer_inputs_indices("input_self_attention_addition_" + to_string(i+1), { "encoder_perceptron_normalization_" + to_string(i), "input_self_attention_" + to_string(i+1) });

        // Normalization
        
        add_layer(make_unique<NormalizationLayer3D>(new_input_length,
                                                    new_embedding_dimension,
                                                    "input_self_attention_normalization_" + to_string(i+1)));
        
        set_layer_inputs_indices("input_self_attention_normalization_" + to_string(i+1), "input_self_attention_addition_" + to_string(i+1));
        
        // Perceptron

        add_layer(make_unique<PerceptronLayer3D>(new_input_length,
                                                 new_embedding_dimension,
                                                 new_perceptron_depth,
                                                 PerceptronLayer3D::ActivationFunction::RectifiedLinear,
                                                 "encoder_internal_perceptron_" + to_string(i+1)));
        
        set_layer_inputs_indices("encoder_internal_perceptron_" + to_string(i+1), "input_self_attention_normalization_" + to_string(i+1));

        // Perceptron

        add_layer(make_unique<PerceptronLayer3D>(new_input_length,
                                                 new_perceptron_depth,
                                                 new_embedding_dimension,
                                                 PerceptronLayer3D::ActivationFunction::HyperbolicTangent,
                                                 "encoder_external_perceptron_" + to_string(i+1)));

        set_layer_inputs_indices("encoder_external_perceptron_" + to_string(i+1), "encoder_internal_perceptron_" + to_string(i+1));

//        encoder_external_perceptron_layer->set_dropout_rate(dropout_rate);
                
        add_layer(make_unique<AdditionLayer3D>(new_input_length,
                                               new_embedding_dimension,
                                               "encoder_perceptron_addition_" + to_string(i+1)));
        
        set_layer_inputs_indices("encoder_perceptron_addition_" + to_string(i+1), { "input_self_attention_normalization_" + to_string(i+1), "encoder_external_perceptron_" + to_string(i+1)});
        
        add_layer(make_unique<NormalizationLayer3D>(new_input_length,
                                                    new_embedding_dimension,
                                                    "encoder_perceptron_normalization_" + to_string(i+1)));
        
        set_layer_inputs_indices("encoder_perceptron_normalization_" + to_string(i+1), "encoder_perceptron_addition_" + to_string(i+1));
    }
    
    // Decoder

    for(Index i = 0; i < new_blocks_number; i++)
    {
        add_layer(make_unique<MultiHeadAttention>(new_decoder_length,
                                                       new_decoder_length,
                                                       new_embedding_dimension,
                                                       new_heads_number,
                                                       false, // chatgpt says that here uses causal mask???
                                                       "decoder_self_attention_" + to_string(i+1)));

        i == 0
            ? set_layer_inputs_indices("decoder_self_attention_1", {"decoder_embedding", "decoder_embedding"})
            : set_layer_inputs_indices("decoder_self_attention_" + to_string(i+1), {"decoder_perceptron_normalization_" + to_string(i), "decoder_perceptron_normalization_" + to_string(i)});

        //decoder_self_attention_layer->set_dropout_rate(dropout_rate);

        add_layer(make_unique<AdditionLayer3D>(new_decoder_length,
                                               new_embedding_dimension,
                                               "decoder_self_attention_addition_" + to_string(i+1)));
        i == 0
            ? set_layer_inputs_indices("decoder_self_attention_addition_" + to_string(i+1), { "decoder_embedding", "decoder_self_attention_" + to_string(i+1) })
            : set_layer_inputs_indices("decoder_self_attention_addition_" + to_string(i+1), { "decoder_perceptron_normalization_" + to_string(i), "decoder_self_attention_" + to_string(i+1) });

        add_layer(make_unique<NormalizationLayer3D>(new_decoder_length,
                                                    new_embedding_dimension,
                                                    "decoder_self_attention_normalization_" + to_string(i+1)));

        set_layer_inputs_indices("decoder_self_attention_normalization_" + to_string(i+1), "decoder_self_attention_addition_" + to_string(i+1));

        add_layer(make_unique<MultiHeadAttention>(new_decoder_length,
                                                       new_input_length,        //previously called context length
                                                       new_embedding_dimension,
                                                       new_heads_number,
                                                       false, 
                                                       "cross_attention_" + to_string(i+1)));

        set_layer_inputs_indices("cross_attention_" + to_string(i+1), {"decoder_self_attention_normalization_" + to_string(i+1), "encoder_perceptron_normalization_" + to_string(new_blocks_number)});

        //cross_attention_layer->set_dropout_rate(dropout_rate);

        add_layer(make_unique<AdditionLayer3D>(new_decoder_length,
                                               new_embedding_dimension,
                                               "cross_attention_addition_" + to_string(i+1)));
        
        set_layer_inputs_indices("cross_attention_addition_" + to_string(i+1), { "decoder_self_attention_normalization_" + to_string(i+1), "cross_attention_" + to_string(i+1) });
       
        add_layer(make_unique<NormalizationLayer3D>(new_decoder_length,
                                                    new_embedding_dimension,
                                                    "cross_attention_normalization_" + to_string(i+1)));

        set_layer_inputs_indices("cross_attention_normalization_" + to_string(i+1), "cross_attention_addition_" + to_string(i+1));

        add_layer(make_unique<PerceptronLayer3D>(new_decoder_length,
                                                 new_embedding_dimension,
                                                 new_perceptron_depth,
                                                 PerceptronLayer3D::ActivationFunction::RectifiedLinear,
                                                 "decoder_internal_perceptron_" + to_string(i+1)));
        
        set_layer_inputs_indices("decoder_internal_perceptron_" + to_string(i+1), "cross_attention_normalization_" + to_string(i+1));

        add_layer(make_unique<PerceptronLayer3D>(new_decoder_length,
                                                 new_perceptron_depth,
                                                 new_embedding_dimension,
                                                 PerceptronLayer3D::ActivationFunction::HyperbolicTangent,
                                                 "decoder_external_perceptron_" + to_string(i+1)));

        set_layer_inputs_indices("decoder_external_perceptron_" + to_string(i+1), "decoder_internal_perceptron_" + to_string(i+1));

        add_layer(make_unique<AdditionLayer3D>(new_decoder_length,
                                               new_embedding_dimension,
                                               "decoder_perceptron_addition_" + to_string(i+1)));

        set_layer_inputs_indices("decoder_perceptron_addition_" + to_string(i+1), { "cross_attention_normalization_" + to_string(i+1), "decoder_external_perceptron_" + to_string(i+1) });

        add_layer(make_unique<NormalizationLayer3D>(new_decoder_length,
                                                    new_embedding_dimension,
                                                    "decoder_perceptron_normalization_" + to_string(i+1)));

        set_layer_inputs_indices("decoder_perceptron_normalization_" + to_string(i+1), "decoder_perceptron_addition_" + to_string(i+1));
    }
    
    add_layer(make_unique<ProbabilisticLayer3D>(new_decoder_length,
                                                new_embedding_dimension,
                                                new_decoder_dimensions,
                                                "probabilistic"));

    set_layer_inputs_indices("probabilistic", "decoder_perceptron_normalization_" + to_string(new_blocks_number));
}


void Transformer::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


void Transformer::set_input_vocabulary(const unordered_map<string, Index>& new_input_vocabulary)
{
    input_vocabulary = new_input_vocabulary;
}


void Transformer::set_output_vocabulary(const unordered_map<string, Index>& new_output_vocabulary)
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

Index Transformer::get_input_length() const
{
    return input_length;
}

Index Transformer::get_decoder_length() const
{
    return decoder_length;
}
/*
// string Transformer::calculate_outputs(const string& context_string, const bool& imported_vocabulary)
// {
//     type start_indicator = 1;
//     type end_indicator = 2;

//     if(imported_vocabulary)
//     {
//     type start_indicator = 2;
//     type end_indicator = 3;
//     }

//     const Index batch_samples_number = 1;

//     Tensor<type, 2> context(batch_samples_number, decoder_length);
//     context.setZero();
//     context(0) = start_indicator;

//     if(!imported_vocabulary)    tokenize_whitespace(context_tokens[0], context);
//     else
//     tokenize_wordpiece(context_tokens[0], context);

//     Tensor<type, 2> input(batch_samples_number, input_length);
//     input.setZero();
//     input(0) = start_indicator;

//     ForwardPropagation forward_propagation(batch_samples_number, this);

//     const pair<type*, dimensions> context_pair(context.data(), { 1, decoder_length });
//     const pair<type*, dimensions> input_pair(input.data(), { 1, input_length });

//     const vector<pair<type*, dimensions>> input_pairs = {input_pair, context_pair};

//     const Index layers_number = get_layers_number();

//     const pair<type*, dimensions> outputs_pair
//         = forward_propagation.layers[layers_number - 1]->get_outputs_pair();

//     TensorMap<Tensor<type, 2>> outputs = tensor_map_2(outputs_pair);

//     Tensor<type, 1> current_outputs(outputs_pair.second[2]);
//     Tensor<Index, 0> prediction;

//     for(Index i = 1; i < input_length; i++)
//     {
//         //forward_propagate(input_pairs, forward_propagation);

//         current_outputs.device(*thread_pool_device) = outputs.chip(i - 1, 0);

//         prediction.device(*thread_pool_device) = current_outputs.argmax();

//         input(i) = type(prediction(0));

//         if(prediction(0) == end_indicator)
//             break;
//     }

//     ostringstream output_string;

//     //if(!imported_vocabulary)
//     // detokenize_whitespace(input, output_string);
//     //else
//     detokenize_wordpiece(input, output_string);

//     return output_string.str();

// }
*/

string Transformer::calculate_outputs(const vector<string>& input_string)
{

    //if(imported_vocabulary)
    //{
    type start_indicator = 2;
    type end_indicator = 3;
    //}

    //@todo
    vector<vector<string>> input_tokens(input_string.size());
    for(size_t i = 0; i <input_tokens.size(); i++)
        input_tokens[i] = preprocess_language_document(input_string[i], true);

    const Index samples_number = 1;

    Tensor<type, 2> input(samples_number, input_length);
    input.setZero();

    //if(!imported_vocabulary)    tokenize_whitespace(context_tokens[0], context);
    //else
    // tokenize_wordpiece(input_tokens[0], input);

    tokenize_wordpiece(input_tokens[0], input);
    // tokenize_whitespace(input_tokens[0], input);

    Tensor<type, 2> decoder(samples_number, decoder_length);

    decoder.setZero();
    decoder(0) = start_indicator;

    ForwardPropagation forward_propagation(samples_number, this);

    const pair<type*, dimensions> context_pair(input.data(), { samples_number, input_length });
    const pair<type*, dimensions> decoder_pair(decoder.data(), { samples_number, decoder_length });

    const vector<pair<type*, dimensions>> input_pairs = {decoder_pair, context_pair};

    const Index layers_number = get_layers_number();

    const pair<type*, dimensions> outputs_pair 
        = forward_propagation.layers[layers_number - 1]->get_outputs_pair();

    TensorMap <Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[1], outputs_pair.second[2]);
    outputs.setZero();

    Tensor<type, 1> current_outputs(outputs_pair.second[2]);
    current_outputs.setZero();

    Tensor<Index, 0> prediction;

    for(Index i = 1; i < input_length; i++)
    {
        forward_propagate(input_pairs, forward_propagation, false);

        current_outputs = outputs.chip(i - 1, 0);

        prediction = current_outputs.argmax();

        decoder(i) = type(prediction(0));

        if(prediction(0) == end_indicator)
            break;
    }

    ostringstream output_buffer;

    //if(!imported_vocabulary)    
    // detokenize_whitespace(input, output_string);
    //else

    detokenize_wordpiece(decoder, output_buffer);
    // detokenize_whitespace(decoder, output_buffer);

    return output_buffer.str();   

    return string();
}


Tensor<type, 3> Transformer::calculate_outputs(const Tensor<type, 2>& input, const Tensor<type, 2>& context)
{
    const Index samples_number = input.dimension(0);

    const pair<type*, dimensions> input_pair((type*)input.data(), { samples_number, input.dimension(1) });
    const pair<type*, dimensions> context_pair((type*)context.data(), { samples_number, context.dimension(1) });

    const vector<pair<type*, dimensions>> input_pairs = { input_pair, context_pair };

    ForwardPropagation forward_propagation(samples_number, this);

    forward_propagate(input_pairs, forward_propagation, false);

    const pair<type*, dimensions> output_pair = forward_propagation.get_last_trainable_layer_outputs_pair();

    return tensor_map_3(output_pair);
}

/*
// void Transformer::tokenize_whitespace(const vector<string>& context_tokens, Tensor<type, 2>& context)
// {
//     const Index context_vocabulary_size = input_vocabulary.size();

//     bool line_ended = false;

//     for(Index j = 0; j < input_length - 1; j++)
//     {
//         if(j < context_tokens.size())
//         {
//             auto it = find(input_vocabulary.data(), input_vocabulary.data() + context_vocabulary_size, context_tokens[j]);

//             const Index word_index = it - input_vocabulary.data();

//             context(j + 1) = type(word_index);
//         }
//         else
//         {
//             if(j == context_tokens.size() || (j == input_length - 2 && !line_ended))
//             {
//                 context(j + 1) = 3; // end indicator
//                 line_ended = true;
//             }
//             else
//             {
//                 break;
//             }
//         }
//     }
// }
*/

void Transformer::tokenize_whitespace(const vector<string>& context_tokens, Tensor<type, 2>& context)
{
    bool line_ended = false;

    for(Index j = 0; j < input_length - 1; j++)
    {
        if(j < Index(context_tokens.size()))
        {
            auto it = input_vocabulary.find(context_tokens[j]);

            const Index word_index = (it != input_vocabulary.end()) ? it->second : 0;

            context(j + 1) = type(word_index);
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
    // unordered_map<string, type> context_vocabulary_map;

    // for(Index i = 0; i < input_vocabulary.size(); i++)
    //     context_vocabulary_map[input_vocabulary[i]] = type(i);

    Index token_counter = 1;
    bool line_ended = false;

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
            if(j == Index(context_tokens.size())
            || (token_counter == input_length - 1 && !line_ended))
            {
                context(token_counter++) = 3; // end indicator
                line_ended = true;
            }
            else
            {
                break;
            }
        }
    }
}

/*
//void Transformer::detokenize_whitespace(Tensor<type, 2>& predictions, ostringstream& output_string)
//{
    // @todo prediction is of rank 2 but only one loop. Why?

//    for(Index i = 1; i < input_length; i++)
//    {
//        if(predictions(i) == 2) break;

//        output_string << input_vocabulary[Index(predictions(i))] << " ";
//    }
//}
*/
void Transformer::detokenize_whitespace(Tensor<type, 2>& predictions, ostringstream& output_string)
{
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
}


/*
void Transformer::detokenize_wordpiece(Tensor<type, 2>& predictions, ostringstream& buffer)
{
    buffer << output_vocabulary[Index(predictions(1))];

    string current_prediction;

    for(Index i = 2; i < output_length; i++)
    {
        if(predictions(i) == 3)
            break;

        current_prediction = output_vocabulary[Index(predictions(i))];

        current_prediction.substr(0, 2) == "##"
           ? buffer << current_prediction.substr(2)
           : buffer << " " << current_prediction;
    }
}
*/

void Transformer::detokenize_wordpiece(Tensor<type, 2>& predictions, ostringstream& buffer)
{
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

        (current_prediction.substr(0, 2) == "##")
            ? buffer << current_prediction.substr(2)
            : buffer << " " << current_prediction;
    }
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
