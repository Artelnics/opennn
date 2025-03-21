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

}
#endif
