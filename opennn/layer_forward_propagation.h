#ifndef LAYERFORWARDPROPAGATION_H
#define LAYERFORWARDPROPAGATION_H

#include "pch.h"

namespace opennn
{

class Layer;

struct LayerForwardPropagation
{
    LayerForwardPropagation() {}

    virtual ~LayerForwardPropagation() {}

    virtual void print() const {}

    virtual pair<type*, dimensions> get_outputs_pair() const = 0;

    Index batch_size = type(0);

    Layer* layer = nullptr;
};

#ifdef OPENNN_CUDA

struct LayerForwardPropagationCuda
{
    explicit LayerForwardPropagationCuda() {}

    virtual ~LayerForwardPropagationCuda() {}

    virtual void print() const {}

    virtual void free() {}

    virtual pair<type*, dimensions> get_outputs_pair_device() const = 0;

    Index batch_size = type(0);

    Layer* layer = nullptr;

    float* outputs = nullptr;
    cudnnTensorDescriptor_t outputs_tensor_descriptor = nullptr;
};

#endif

}

#endif
