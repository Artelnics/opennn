#include "transformer.h"

namespace opennn
{

Transformer::Transformer() : NeuralNetwork()
{
    NeuralNetwork::set();
}


Transformer::Transformer(const Tensor<Index, 1>& architecture)
{
    Index inputs_length = architecture(0);

    Index context_length = architecture(1);

    Index inputs_dimensions = architecture(2);

    Index context_dim = architecture(3);

    Index embedding_depth = architecture(4);

    Index perceptron_depth = architecture(5);

    Index heads_number = architecture(6);

    Index number_of_layers = architecture(7);

    set(inputs_length, context_length, inputs_dimensions, context_dim, embedding_depth, perceptron_depth, heads_number, number_of_layers);
}


Transformer::Transformer(const initializer_list<Index>& architecture_list)
{
    Tensor<Index, 1> architecture(architecture_list.size());
    architecture.setValues(architecture_list);

    Index inputs_length = architecture(0);

    Index context_length = architecture(1);

    Index inputs_dimensions = architecture(2);

    Index context_dim = architecture(3);

    Index embedding_depth = architecture(4);

    Index perceptron_depth = architecture(5);

    Index heads_number = architecture(6);

    Index number_of_layers = architecture(7);

    set(inputs_length, context_length, inputs_dimensions, context_dim, embedding_depth, perceptron_depth, heads_number, number_of_layers);
}


/// @todo set_layer_inputs_indices(name, {names}) for each layer
/// maybe add if names = Dataset, something
void Transformer::set(const Index& inputs_length, const Index& context_length, const Index& inputs_dimensions, const Index& context_dim,
                      const Index& embedding_depth, const Index& perceptron_depth, const Index& heads_number, const Index& number_of_layers)
{
    delete_layers();

    inputs_names.resize(inputs_length + context_length);


    EmbeddingLayer* context_embedding_layer = new EmbeddingLayer(context_dim, context_length, embedding_depth, true);

    context_embedding_layer->set_name("context_embedding");
    add_layer(context_embedding_layer);
    set_layer_inputs_indices("context_embedding", "dataset");


    EmbeddingLayer* input_embedding_layer = new EmbeddingLayer(inputs_dimensions, inputs_length, embedding_depth, true);

    input_embedding_layer->set_name("input_embedding");
    add_layer(input_embedding_layer);
    set_layer_inputs_indices("input_embedding", "dataset");


    for(Index i = 0; i < number_of_layers; i++)
    {
        MultiheadAttentionLayer* context_self_attention_layer =
                new MultiheadAttentionLayer(context_length, context_length, embedding_depth, heads_number);

        context_self_attention_layer->set_name("context_self_attention_" + to_string(i+1));
        add_layer(context_self_attention_layer);
        if(i == 0)
        {
            set_layer_inputs_indices("context_self_attention_1", {"context_embedding", "context_embedding"});
        }
        else
        {
            set_layer_inputs_indices("context_self_attention_" + to_string(i+1), {"encoder_external_perceptron_" + to_string(i), "encoder_external_perceptron_" + to_string(i)});
        }
/*
        PerceptronLayer* encoder_internal_perceptron_layer =
                new PerceptronLayer(embedding_depth, perceptron_depth, PerceptronLayer::ActivationFunction::RectifiedLinear);

        encoder_internal_perceptron_layer->set_name("encoder_internal_perceptron_" + to_string(i+1));
        add_layer(encoder_internal_perceptron_layer);
        set_layer_inputs_indices("encoder_internal_perceptron_" + to_string(i+1), "context_self_attention_" + to_string(i+1));


        PerceptronLayer* encoder_external_perceptron_layer = new PerceptronLayer(perceptron_depth, embedding_depth);

        encoder_external_perceptron_layer->set_name("encoder_external_perceptron_" + to_string(i+1));
        add_layer(encoder_external_perceptron_layer);
        set_layer_inputs_indices("encoder_external_perceptron_" + to_string(i+1), "encoder_internal_perceptron_" + to_string(i+1));
*/
    }


    for(Index i = 0; i < number_of_layers; i++)
    {
        MultiheadAttentionLayer* input_self_attention_layer =
                new MultiheadAttentionLayer(inputs_length, inputs_length, embedding_depth, heads_number, true);

        input_self_attention_layer->set_name("input_self_attention_" + to_string(i+1));
        add_layer(input_self_attention_layer);
        if(i == 0)
        {
            set_layer_inputs_indices("input_self_attention_1", {"input_embedding", "input_embedding"});
        }
        else
        {
            set_layer_inputs_indices("input_self_attention_" + to_string(i+1), {"decoder_external_perceptron_" + to_string(i), "decoder_external_perceptron_" + to_string(i)});
        }


        MultiheadAttentionLayer* cross_attention_layer =
                new MultiheadAttentionLayer(inputs_length, context_length, embedding_depth, heads_number);

        cross_attention_layer->set_name("cross_attention_" + to_string(i+1));
        add_layer(cross_attention_layer);
        set_layer_inputs_indices("cross_attention_" + to_string(i+1), {"input_self_attention_" + to_string(i+1), "context_self_attention_1"/*"encoder_external_perceptron_" + to_string(number_of_layers)*/});
/*
        PerceptronLayer* decoder_internal_perceptron_layer =
                new PerceptronLayer(embedding_depth, perceptron_depth, PerceptronLayer::ActivationFunction::RectifiedLinear);

        decoder_internal_perceptron_layer->set_name("decoder_internal_perceptron_" + to_string(i+1));
        add_layer(decoder_internal_perceptron_layer);
        set_layer_inputs_indices("decoder_internal_perceptron_" + to_string(i+1), "cross_attention_" + to_string(i+1));


        PerceptronLayer* decoder_external_perceptron_layer = new PerceptronLayer(perceptron_depth, embedding_depth);

        decoder_external_perceptron_layer->set_name("decoder_external_perceptron_" + to_string(i+1));
        add_layer(decoder_external_perceptron_layer);
        set_layer_inputs_indices("decoder_external_perceptron_" + to_string(i+1), "decoder_internal_perceptron_" + to_string(i+1));
*/
    }

/*
    ProbabilisticLayer* final_layer = new ProbabilisticLayer(embedding_depth, inputs_dimensions);
    final_layer->set_name("probabilistic");
    add_layer(final_layer);
    set_layer_inputs_indices("probabilistic", "decoder_external_perceptron_" + to_string(number_of_layers));
*/
}

};
