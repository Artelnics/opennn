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


    void set(const Index& inputs_length, const Index& context_length, const Index& inputs_dimensions, const Index& context_dim,
             const Index& embedding_depth, const Index& perceptron_depth, const Index& heads_number, const Index& number_of_layers);


protected:

    string name = "transformer";

    /// Length of input entries

    Index inputs_length;

    /// Length of context entries

    Index context_length;

    /// Maximum value in input

    Index inputs_dimensions;

    /// Maximum value in context

    Index context_dim;

    /// Embedding depth for each EmbeddingLayer

    Index embedding_depth;

    /// Depth of internal perceptron layers

    Index perceptron_depth;

    /// Number of attention heads per MultiheadAttentionLayer

    Index heads_number;

    /// Number of encoder and decoder layers

    Index number_of_layers;

};


struct TransformerForwardPropagation
{
    // Constructors

    TransformerForwardPropagation() {}

    TransformerForwardPropagation(const Index& new_batch_samples, NeuralNetwork* new_neural_network)
    {
        set(new_batch_samples, new_neural_network);
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


    void set(const Index& new_batch_samples, NeuralNetwork* new_neural_network)
    {
        Transformer* neural_network = static_cast<Transformer*>(new_neural_network);

        batch_samples_number = new_batch_samples;

        const Tensor<Layer*, 1> neural_network_layers = neural_network->get_layers();

        const Index layers_number = layers.size();

        layers.resize(layers_number);

        for(Index i = 0; i < layers_number; i++)
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


    void print() const
    {
        cout << "Transformer forward propagation" << endl;

        const Index layers_number = layers.size();

        cout << "Layers number: " << layers_number << endl;

        for(Index i = 0; i < layers_number; i++)
        {
            cout << "Layer " << i + 1 << ": " << layers(i)->layer->get_name() << endl;

            layers(i)->print();
        }
    }


    Index batch_samples_number = 0;

    Tensor<LayerForwardPropagation*, 1> layers;
};
};

#endif // TRANSFORMER_H
















