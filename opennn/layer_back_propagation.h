#ifndef LAYERBACKPROPAGATION_H
#define LAYERBACKPROPAGATION_H

#include "pch.h"

namespace opennn
{

class Layer;

struct LayerBackPropagation
{
    LayerBackPropagation() {}

    virtual vector<pair<type*, dimensions>> get_input_derivative_pairs() const = 0;

    virtual void print() const {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;
};

#ifdef OPENNN_CUDA

struct LayerBackPropagationCuda
{
    LayerBackPropagationCuda() {}

    virtual vector<pair<type*, dimensions>> get_input_derivative_pairs_device() const
    {
        return vector<pair<type*, dimensions>>();
    } // @todo change it to = 0; when implemented in all layers

    virtual void free() {}

    virtual void print() const {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;

    float* input_derivatives = nullptr;
    cudnnTensorDescriptor_t input_derivatives_tensor_descriptor = nullptr;
};

#endif

}
#endif
