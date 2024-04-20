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
    
    EmbeddingLayer* input_embedding_layer = new EmbeddingLayer(inputs_dimension, inputs_length, embedding_depth, true);

    input_embedding_layer->set_name("input_embedding");
    add_layer(input_embedding_layer);
    set_layer_inputs_indices("input_embedding", "input");
    

    EmbeddingLayer* context_embedding_layer = new EmbeddingLayer(context_dimension, context_length, embedding_depth, true);

    context_embedding_layer->set_name("context_embedding");
    add_layer(context_embedding_layer);
    set_layer_inputs_indices("context_embedding", "context");


    // Encoder

    for(Index i = 0; i < number_of_layers; i++)
    {
        MultiheadAttentionLayer* context_self_attention_layer =
                new MultiheadAttentionLayer(context_length, context_length, embedding_depth, heads_number);

        context_self_attention_layer->set_name("context_self_attention_" + to_string(i+1));
        add_layer(context_self_attention_layer);
        if(i == 0)
            set_layer_inputs_indices("context_self_attention_1", {"context_embedding", "context_embedding"});
        else
            set_layer_inputs_indices("context_self_attention_" + to_string(i+1), { "encoder_perceptron_normalization_" + to_string(i), "encoder_perceptron_normalization_" + to_string(i) });


        AdditionLayer3D* context_self_attention_addition_layer = new AdditionLayer3D(context_length, embedding_depth);
        context_self_attention_addition_layer->set_name("context_self_attention_addition_" + to_string(i + 1));
        add_layer(context_self_attention_addition_layer);
        if(i == 0)
            set_layer_inputs_indices("context_self_attention_addition_" + to_string(i + 1), { "context_embedding", "context_self_attention_" + to_string(i + 1) });
        else
            set_layer_inputs_indices("context_self_attention_addition_" + to_string(i + 1), { "encoder_perceptron_normalization_" + to_string(i), "context_self_attention_" + to_string(i + 1) });


        NormalizationLayer3D* context_self_attention_normalization_layer = new NormalizationLayer3D(context_length, embedding_depth);
        context_self_attention_normalization_layer->set_name("context_self_attention_normalization_" + to_string(i + 1));
        add_layer(context_self_attention_normalization_layer);
        set_layer_inputs_indices("context_self_attention_normalization_" + to_string(i + 1), "context_self_attention_addition_" + to_string(i + 1));

        PerceptronLayer3D* encoder_internal_perceptron_layer =
                new PerceptronLayer3D(context_length, embedding_depth, perceptron_depth, PerceptronLayer3D::ActivationFunction::RectifiedLinear);

        encoder_internal_perceptron_layer->set_name("encoder_internal_perceptron_" + to_string(i+1));
        add_layer(encoder_internal_perceptron_layer);
        set_layer_inputs_indices("encoder_internal_perceptron_" + to_string(i+1), "context_self_attention_normalization_" + to_string(i + 1));


        PerceptronLayer3D* encoder_external_perceptron_layer =
            new PerceptronLayer3D(context_length, perceptron_depth, embedding_depth, PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

        encoder_external_perceptron_layer->set_name("encoder_external_perceptron_" + to_string(i+1));
        add_layer(encoder_external_perceptron_layer);
        set_layer_inputs_indices("encoder_external_perceptron_" + to_string(i+1), "encoder_internal_perceptron_" + to_string(i+1));

        AdditionLayer3D* encoder_perceptron_addition_layer = new AdditionLayer3D(context_length, embedding_depth);
        encoder_perceptron_addition_layer->set_name("encoder_perceptron_addition_" + to_string(i + 1));
        add_layer(encoder_perceptron_addition_layer);
        set_layer_inputs_indices("encoder_perceptron_addition_" + to_string(i + 1), { "context_self_attention_normalization_" + to_string(i + 1), "encoder_external_perceptron_" + to_string(i + 1) });

        NormalizationLayer3D* encoder_perceptron_normalization_layer = new NormalizationLayer3D(context_length, embedding_depth);
        encoder_perceptron_normalization_layer->set_name("encoder_perceptron_normalization_" + to_string(i + 1));
        add_layer(encoder_perceptron_normalization_layer);
        set_layer_inputs_indices("encoder_perceptron_normalization_" + to_string(i + 1), "encoder_perceptron_addition_" + to_string(i + 1));
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
            set_layer_inputs_indices("input_self_attention_" + to_string(i+1), {"decoder_perceptron_normalization_" + to_string(i), "decoder_perceptron_normalization_" + to_string(i)});
        }


        AdditionLayer3D* input_self_attention_addition_layer = new AdditionLayer3D(inputs_length, embedding_depth);
        input_self_attention_addition_layer->set_name("input_self_attention_addition_" + to_string(i + 1));
        add_layer(input_self_attention_addition_layer);
        if (i == 0)
            set_layer_inputs_indices("input_self_attention_addition_" + to_string(i + 1), { "input_embedding", "input_self_attention_" + to_string(i + 1) });
        else
            set_layer_inputs_indices("input_self_attention_addition_" + to_string(i + 1), { "decoder_perceptron_normalization_" + to_string(i), "input_self_attention_" + to_string(i + 1) });


        NormalizationLayer3D* input_self_attention_normalization_layer = new NormalizationLayer3D(inputs_length, embedding_depth);
        input_self_attention_normalization_layer->set_name("input_self_attention_normalization_" + to_string(i + 1));
        add_layer(input_self_attention_normalization_layer);
        set_layer_inputs_indices("input_self_attention_normalization_" + to_string(i + 1), "input_self_attention_addition_" + to_string(i + 1));


        MultiheadAttentionLayer* cross_attention_layer =
                new MultiheadAttentionLayer(inputs_length, context_length, embedding_depth, heads_number);

        cross_attention_layer->set_name("cross_attention_" + to_string(i+1));
        add_layer(cross_attention_layer);
        set_layer_inputs_indices("cross_attention_" + to_string(i+1), {"input_self_attention_normalization_" + to_string(i+1), "encoder_perceptron_normalization_" + to_string(number_of_layers)});


        AdditionLayer3D* cross_attention_addition_layer = new AdditionLayer3D(inputs_length, embedding_depth);
        cross_attention_addition_layer->set_name("cross_attention_addition_" + to_string(i + 1));
        add_layer(cross_attention_addition_layer);
        set_layer_inputs_indices("cross_attention_addition_" + to_string(i + 1), { "input_self_attention_normalization_" + to_string(i + 1), "cross_attention_" + to_string(i + 1) });
        

        NormalizationLayer3D* cross_attention_normalization_layer = new NormalizationLayer3D(inputs_length, embedding_depth);
        cross_attention_normalization_layer->set_name("cross_attention_normalization_" + to_string(i + 1));
        add_layer(cross_attention_normalization_layer);
        set_layer_inputs_indices("cross_attention_normalization_" + to_string(i + 1), "cross_attention_addition_" + to_string(i + 1));


        PerceptronLayer3D* decoder_internal_perceptron_layer =
                new PerceptronLayer3D(inputs_length, embedding_depth, perceptron_depth, PerceptronLayer3D::ActivationFunction::RectifiedLinear);

        decoder_internal_perceptron_layer->set_name("decoder_internal_perceptron_" + to_string(i+1));
        add_layer(decoder_internal_perceptron_layer);
        set_layer_inputs_indices("decoder_internal_perceptron_" + to_string(i+1), "cross_attention_normalization_" + to_string(i+1));


        PerceptronLayer3D* decoder_external_perceptron_layer =
                new PerceptronLayer3D(inputs_length, perceptron_depth, embedding_depth, PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

        decoder_external_perceptron_layer->set_name("decoder_external_perceptron_" + to_string(i+1));
        add_layer(decoder_external_perceptron_layer);
        set_layer_inputs_indices("decoder_external_perceptron_" + to_string(i+1), "decoder_internal_perceptron_" + to_string(i+1));


        AdditionLayer3D* decoder_perceptron_addition_layer = new AdditionLayer3D(inputs_length, embedding_depth);
        decoder_perceptron_addition_layer->set_name("decoder_perceptron_addition_" + to_string(i + 1));
        add_layer(decoder_perceptron_addition_layer);
        set_layer_inputs_indices("decoder_perceptron_addition_" + to_string(i + 1), { "cross_attention_normalization_" + to_string(i + 1), "decoder_external_perceptron_" + to_string(i + 1) });


        NormalizationLayer3D* decoder_perceptron_normalization_layer = new NormalizationLayer3D(inputs_length, embedding_depth);
        decoder_perceptron_normalization_layer->set_name("decoder_perceptron_normalization_" + to_string(i + 1));
        add_layer(decoder_perceptron_normalization_layer);
        set_layer_inputs_indices("decoder_perceptron_normalization_" + to_string(i + 1), "decoder_perceptron_addition_" + to_string(i + 1));
    }


    // Final layer
    
    ProbabilisticLayer3D* final_layer = new ProbabilisticLayer3D(inputs_length, embedding_depth, inputs_dimension + 1);

    final_layer->set_name("probabilistic");
    add_layer(final_layer);
    set_layer_inputs_indices("probabilistic", "decoder_perceptron_normalization_" + to_string(number_of_layers));
    
}


void Transformer::set_input_vocabulary(Tensor<string, 1>& new_input_vocabulary)
{
    input_vocabulary = new_input_vocabulary;
}


void Transformer::set_context_vocabulary(Tensor<string, 1>& new_context_vocabulary)
{
    context_vocabulary = new_context_vocabulary;
}

/*
string Transformer::calculate_outputs(const string& context_string)
{
    const Index context_vocabulary_size = context_vocabulary.size();

    const Tensor<Tensor<string, 1>, 1> context_tokens = preprocess_language_documents(tensor_wrapper(context_string));

    Tensor<type, 2> context(1, context_length);
    context.setZero();
    context(0) = 1;

    bool line_ended = false;

    for (Index j = 1; j < context_length; j++)
    {
        if (j <= context_tokens.size() && contains(context_vocabulary, context_tokens(j - 1)))
        {
            auto it = find(context_vocabulary.data(), context_vocabulary.data() + context_vocabulary_size, context_tokens(j - 1));

            const Index word_index = it - context_vocabulary.data();

            context(j) = type(word_index + 3); /// +3 because 0 (padding), 1 (start) and 2 (end) are reserved
        }
        else
        {
            if (j == context_tokens.size() + 1 || (j == context_length - 1 && !line_ended))
            {
                context(j) = 2; /// end indicator
                line_ended = true;
            }
            else
            {
                context(j) = type(0);
            }
        }
    }

    /*
    const Index batch_samples_number = inputs.dimension(0);
    const Index inputs_number = inputs.dimension(1);
    const Index outputs_number = get_outputs_number();

    ForwardPropagation neural_network_forward_propagation(batch_samples_number, this);

    const pair<type*, dimensions> inputs_pair((type*)inputs.data(), { {batch_samples_number, inputs_number} });

    forward_propagate(tensor_wrapper(inputs_pair), neural_network_forward_propagation);

    const Index layers_number = get_layers_number();

    if (layers_number == 0) return Tensor<type, 2>();

    const pair<type*, dimensions> outputs_pair = neural_network_forward_propagation.layers(layers_number - 1)->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1]);

    return outputs;
    
}
*/

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
