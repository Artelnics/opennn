#ifndef LAYERBACKPROPAGATIONLM_H
#define LAYERBACKPROPAGATIONLM_H

#include "pch.h"

namespace opennn
{

class Layer;

struct LayerBackPropagationLM
{
    LayerBackPropagationLM() {}

    virtual vector<pair<type*, dimensions>> get_input_derivative_pairs() const = 0;

    virtual void print() const {}

    Index samples_number = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;
};

}
#endif
