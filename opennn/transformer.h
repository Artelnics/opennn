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
#include "neural_network_forward_propagation.h"
#include "batch.h"


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

    void set(const Tensor<Index, 1>&);

    void set(const initializer_list<Index>&);

    void set(const Index& inputs_length, const Index& context_length, const Index& inputs_dimension, const Index& context_dimension,
             const Index& embedding_depth, const Index& perceptron_depth, const Index& heads_number, const Index& number_of_layers);

    /// @todo move to NeuralNetwork
    void forward_propagate(const Batch&, ForwardPropagation&, const bool&) const;
    bool is_input_layer(const Tensor<Index, 1>&) const;
    bool is_context_layer(const Tensor<Index, 1>&) const;

protected:

    string name = "transformer";

    /// Length of input entries

    Index inputs_length;

    /// Length of context entries

    Index context_length;

    /// Maximum value in input

    Index inputs_dimension;

    /// Maximum value in context

    Index context_dimension;

    /// Embedding depth for each EmbeddingLayer

    Index embedding_depth;

    /// Depth of internal perceptron layers

    Index perceptron_depth;

    /// Number of attention heads per MultiheadAttentionLayer

    Index heads_number;

    /// Number of encoder and decoder layers

    Index number_of_layers;

};


struct TransformerForwardPropagation : ForwardPropagation
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


    void set(const Index& new_batch_samples, NeuralNetwork* new_neural_network);


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
















