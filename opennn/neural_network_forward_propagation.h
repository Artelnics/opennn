#ifndef FORWARDPROPAGATION_H
#define FORWARDPROPAGATION_H

//#include <string>

#include "neural_network.h"
#include "batch.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct ForwardPropagation
{
    ForwardPropagation() {}

    ForwardPropagation(const Index& new_batch_samples_number, NeuralNetwork* new_neural_network)
    {
        set(new_batch_samples_number, new_neural_network);
    }

    virtual ~ForwardPropagation()
    {
        const Index layers_number = layers.size();

//        for(Index i = 0; i < layers_number; i++)
//            delete layers[i];
    }

    void set(const Index&, NeuralNetwork*);

    pair<type*, dimensions> get_last_trainable_layer_outputs_pair() const;

    vector<vector<pair<type*, dimensions>>> get_layer_input_pairs(const Batch& batch) const;

    void print() const;

    Index batch_samples_number = 0;

    NeuralNetwork* neural_network = nullptr;

    vector<unique_ptr<LayerForwardPropagation>> layers;
};

}
#endif
