#ifndef LAYERFORWARDPROPAGATION_H
#define LAYERFORWARDPROPAGATION_H

#include "pch.h"

namespace opennn
{

class Layer;

struct LayerForwardPropagation
{
    explicit LayerForwardPropagation()
    {
    }

    virtual void set(const Index&, Layer*) = 0;

    virtual void print() const {}

    Index batch_samples_number = type(0);

    Layer* layer = nullptr;

    virtual pair<type*, dimensions> get_outputs_pair() const = 0;
};

}

#endif
