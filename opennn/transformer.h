//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <errno.h>

// OpenNN includes

#include "embedding_layer.h"
#include "multihead_attention_layer.h"
#include "neural_network.h"


namespace opennn
{

struct TransformerForwardPropagation;
struct TransformerBackPropagation;

/// @todo explain

class Transformer : public NeuralNetwork
{
public:

    // Constructors

    explicit Transformer();

    explicit Transformer(const Tensor<Index, 1>&);

    explicit Transformer(const initializer_list<Index>&);


    void set(const Index& input_length, const Index& context_length, const Index& input_dim, const Index& context_dim,
             const Index& embedding_depth, const Index& perceptron_depth, const Index& number_of_heads, const Index& number_of_layers);


protected:

    string name = "transformer";

    /// Length of input entries

    Index input_length;

    /// Length of context entries

    Index context_length;

    /// Maximum value in input

    Index input_dim;

    /// Maximum value in context

    Index context_dim;

    /// Embedding depth for each EmbeddingLayer

    Index embedding_depth;

    /// Depth of internal perceptron layers

    Index perceptron_depth;

    /// Number of attention heads per MultiheadAttentionLayer

    Index number_of_heads;

    /// Number of encoder and decoder layers

    Index number_of_layers;

};


struct TransformerForwardPropagation
{
    // Constructors

    TransformerForwardPropagation() {}

    TransformerForwardPropagation(const Index& new_batch_samples, NeuralNetwork* new_neural_network_pointer)
    {
        set(new_batch_samples, new_neural_network_pointer);
    }


    // Destructor

    virtual ~TransformerForwardPropagation()
    {
        const Index layers_number = layers.size();

        for(Index i = 0; i < layers_number; i++)
        {
            delete layers(i);
        }
    }


    void set(const Index& new_batch_samples, NeuralNetwork* new_neural_network_pointer)
    {
        Transformer* neural_network_pointer = static_cast<Transformer*>(new_neural_network_pointer);

        batch_samples_number = new_batch_samples;

        const Tensor<Layer*, 1> layers_pointers = neural_network_pointer->get_layers_pointers();

        const Index layers_number = layers_pointers.size();

        layers.resize(layers_number);

        for(Index i = 0; i < layers_number; i++)
        {
            switch (layers_pointers(i)->get_type())
            {

            case Layer::Type::Embedding:
            {
                layers(i) = new EmbeddingLayerForwardPropagation(batch_samples_number, layers_pointers(i));

            }
            break;

            case Layer::Type::MultiheadAttention:
            {
                layers(i) = new MultiheadAttentionLayerForwardPropagation(batch_samples_number, layers_pointers(i));

            }
            break;

            case Layer::Type::Perceptron:
            {
                layers(i) = new PerceptronLayerForwardPropagation(batch_samples_number, layers_pointers(i));

            }
            break;

            case Layer::Type::Probabilistic:
            {
                layers(i) = new ProbabilisticLayerForwardPropagation(batch_samples_number, layers_pointers(i));

            }
            break;

            default: break;
            }
        }
    }


    void print() const
    {
        cout << "Transformer forward propagation" << endl;

        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << ": " << layers(i)->layer_pointer->get_name() << endl;

            layers(i)->print();
        }
    }


    Index batch_samples_number = 0;

    Tensor<LayerForwardPropagation*, 1> layers;
};
};

#endif // TRANSFORMER_H
















