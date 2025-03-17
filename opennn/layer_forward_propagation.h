#ifndef LAYERFORWARDPROPAGATION_H
#define LAYERFORWARDPROPAGATION_H

#include "pch.h"

namespace opennn
{

class Layer;

struct LayerForwardPropagation
{
    LayerForwardPropagation()
    {
    }

    virtual ~LayerForwardPropagation() {}

    virtual void print() const {}

    Index samples_number = type(0);

    Layer* layer = nullptr;

    virtual pair<type*, dimensions> get_outputs_pair() const = 0;
};

}

#endif
