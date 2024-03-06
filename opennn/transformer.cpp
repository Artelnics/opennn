#include "transformer.h"

namespace opennn
{

Transformer::Transformer() : NeuralNetwork()
{
    NeuralNetwork::set();
}


Transformer::Transformer(const Tensor<Index, 1>& architecture)
{
    set(architecture);
}


Transformer::Transformer(const initializer_list<Index>& architecture_list)
{
    set(architecture_list);
}

void Transformer::set(const Tensor<Index, 1>& architecture)
{
    inputs_length = architecture(0);

    context_length = architecture(1);

    inputs_dimension = architecture(2);

    context_dimension = architecture(3);

    embedding_depth = architecture(4);

    perceptron_depth = architecture(5);

    heads_number = architecture(6);

    number_of_layers = architecture(7);

    set(inputs_length, context_length, inputs_dimension, context_dimension, embedding_depth, perceptron_depth, heads_number, number_of_layers);
}

void Transformer::set(const initializer_list<Index>& architecture_list)
{
    Tensor<Index, 1> architecture(architecture_list.size());
    architecture.setValues(architecture_list);

    set(architecture);
}


void Transformer::set(const Index& inputs_length, const Index& context_length, const Index& inputs_dimension, const Index& context_dimension,
                      const Index& embedding_depth, const Index& perceptron_depth, const Index& heads_number, const Index& number_of_layers)
{
    delete_layers();

    inputs_names.resize(inputs_length + context_length);


    // Embedding Layers

    EmbeddingLayer* context_embedding_layer = new EmbeddingLayer(context_dimension, context_length, embedding_depth, true);

    context_embedding_layer->set_name("context_embedding");
    add_layer(context_embedding_layer);
    set_layer_inputs_indices("context_embedding", "dataset");


    EmbeddingLayer* input_embedding_layer = new EmbeddingLayer(inputs_dimension, inputs_length, embedding_depth, true);

    input_embedding_layer->set_name("input_embedding");
    add_layer(input_embedding_layer);
    set_layer_inputs_indices("input_embedding", "dataset");


    // Encoder

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

        PerceptronLayer3D* encoder_internal_perceptron_layer =
                new PerceptronLayer3D(context_length, embedding_depth, perceptron_depth, PerceptronLayer3D::ActivationFunction::RectifiedLinear);

        encoder_internal_perceptron_layer->set_name("encoder_internal_perceptron_" + to_string(i+1));
        add_layer(encoder_internal_perceptron_layer);
        set_layer_inputs_indices("encoder_internal_perceptron_" + to_string(i+1), "context_self_attention_" + to_string(i+1));


        PerceptronLayer3D* encoder_external_perceptron_layer =
            new PerceptronLayer3D(context_length, perceptron_depth, embedding_depth, PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

        encoder_external_perceptron_layer->set_name("encoder_external_perceptron_" + to_string(i+1));
        add_layer(encoder_external_perceptron_layer);
        set_layer_inputs_indices("encoder_external_perceptron_" + to_string(i+1), "encoder_internal_perceptron_" + to_string(i+1));

    }


    // Decoder

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
        set_layer_inputs_indices("cross_attention_" + to_string(i+1), {"input_self_attention_" + to_string(i+1), "encoder_external_perceptron_" + to_string(number_of_layers)});

        PerceptronLayer3D* decoder_internal_perceptron_layer =
                new PerceptronLayer3D(inputs_length, embedding_depth, perceptron_depth, PerceptronLayer3D::ActivationFunction::RectifiedLinear);

        decoder_internal_perceptron_layer->set_name("decoder_internal_perceptron_" + to_string(i+1));
        add_layer(decoder_internal_perceptron_layer);
        set_layer_inputs_indices("decoder_internal_perceptron_" + to_string(i+1), "cross_attention_" + to_string(i+1));


        PerceptronLayer3D* decoder_external_perceptron_layer =
                new PerceptronLayer3D(inputs_length, perceptron_depth, embedding_depth, PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

        decoder_external_perceptron_layer->set_name("decoder_external_perceptron_" + to_string(i+1));
        add_layer(decoder_external_perceptron_layer);
        set_layer_inputs_indices("decoder_external_perceptron_" + to_string(i+1), "decoder_internal_perceptron_" + to_string(i+1));

    }


    // Final layer

    ProbabilisticLayer3D* final_layer = new ProbabilisticLayer3D(inputs_length, embedding_depth, inputs_dimension);
    final_layer->set_name("probabilistic");
    add_layer(final_layer);
    set_layer_inputs_indices("probabilistic", "decoder_external_perceptron_" + to_string(number_of_layers));

}


void Transformer::forward_propagate(const Batch& batch,
                                    ForwardPropagation& forward_propagation,
                                    const bool& is_training) const
{
    const Index layers_number = get_layers_number();

    const Tensor<Layer*, 1> layers = get_layers();

    const Tensor<Tensor<Index, 1>, 1> layers_inputs_indices = get_layers_inputs_indices();

    const Index first_trainable_layer_index = get_first_trainable_layer_index();
    const Index last_trainable_layer_index = get_last_trainable_layer_index();

    Index first_layer_index;
    Index last_layer_index;

    if (is_training)
    {
        first_layer_index = first_trainable_layer_index;
        last_layer_index = last_trainable_layer_index;
    }
    else
    {
        first_layer_index = 0;
        last_layer_index = layers_number - 1;
    }

    pair<type*, dimensions> layer_inputs_pair;
    Tensor<pair<type*, dimensions>, 1> layer_inputs_pairs_tensor;

    for (Index i = first_layer_index; i <= last_layer_index; i++)
    {
        if ( i == first_layer_index || is_input_layer(layers_inputs_indices(i)) )
        {
            layer_inputs_pair = batch.get_inputs_pair();
        }
        else if ( is_context_layer(layers_inputs_indices(i)) )
        {
            layer_inputs_pair = batch.get_context_pair();
        }
        else
        {
            layer_inputs_pairs_tensor.resize(layers_inputs_indices(i).size());

            for (Index j = 0; j < layers_inputs_indices(i).size(); j++)
            {
                layer_inputs_pairs_tensor(i) = forward_propagation.layers(layers_inputs_indices(i)(j))->get_outputs_pair();
            }

            layer_inputs_pair = join_pairs(layer_inputs_pairs_tensor);
        }

        layers(i)->forward_propagate(layer_inputs_pair,
                                     forward_propagation.layers(i),
                                     is_training);
    }
}

bool Transformer::is_input_layer(const Tensor<Index, 1>& layer_inputs_indices) const
{
    for (Index i = 0; i < layer_inputs_indices.size(); i++)
        if (layer_inputs_indices(i) == -1) return true;

    return false;
}

bool Transformer::is_context_layer(const Tensor<Index, 1>& layer_inputs_indices) const
{
    for (Index i = 0; i < layer_inputs_indices.size(); i++)
        if (layer_inputs_indices(i) == -2) return true;

    return false;
}


void TransformerForwardPropagation::set(const Index& new_batch_samples, NeuralNetwork* new_neural_network)
{
    Transformer* neural_network = static_cast<Transformer*>(new_neural_network);

    batch_samples_number = new_batch_samples;

    const Tensor<Layer*, 1> neural_network_layers = neural_network->get_layers();

    const Index layers_number = layers.size();

    layers.resize(layers_number);

    for (Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers(i)->get_type())
        {

        case Layer::Type::Embedding:
        {
            layers(i) = new EmbeddingLayerForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::MultiheadAttention:
        {
            layers(i) = new MultiheadAttentionLayerForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::Perceptron3D:
        {
            layers(i) = new PerceptronLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::Probabilistic3D:
        {
            layers(i) = new ProbabilisticLayer3DForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        default: break;
        }
    }
}


TransformerForwardPropagation::~TransformerForwardPropagation()
{
    const Index layers_number = layers.size();

    for (Index i = 0; i < layers_number; i++)
    {
        delete layers(i);
    }
}

void TransformerForwardPropagation::set(const Index& new_batch_samples, NeuralNetwork* new_neural_network)
{
    Transformer* neural_network = static_cast<Transformer*>(new_neural_network);

    batch_samples_number = new_batch_samples;

    const Tensor<Layer*, 1> neural_network_layers = neural_network->get_layers();

    const Index layers_number = layers.size();

    layers.resize(layers_number);

    for (Index i = 0; i < layers_number; i++)
    {
        switch (neural_network_layers(i)->get_type())
        {

        case Layer::Type::Embedding:
        {
            layers(i) = new EmbeddingLayerForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::MultiheadAttention:
        {
            layers(i) = new MultiheadAttentionLayerForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::Perceptron:
        {
            layers(i) = new PerceptronLayerForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        case Layer::Type::Probabilistic:
        {
            layers(i) = new ProbabilisticLayerForwardPropagation(batch_samples_number, neural_network_layers(i));

        }
        break;

        default: break;
        }
    }
}


void TransformerForwardPropagation::print() const
{
    cout << "Transformer forward propagation" << endl;

    const Index layers_number = layers.size();

    cout << "Layers number: " << layers_number << endl;

    for (Index i = 0; i < layers_number; i++)
    {
        cout << "Layer " << i + 1 << ": " << layers(i)->layer->get_name() << endl;

        layers(i)->print();
    }
}

};
