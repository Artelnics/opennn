#ifndef LAYERBACKPROPAGATION_H
#define LAYERBACKPROPAGATION_H

#include <string>

#include "layer.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct LayerBackPropagation
{
    explicit LayerBackPropagation() {}

    virtual ~LayerBackPropagation() {}
    
    vector<pair<type*, dimensions>>& get_inputs_derivatives_pair()
    {
        return inputs_derivatives;
    }

    virtual void set(const Index&, Layer*) {}

    virtual void print() const {}

    Index batch_samples_number = 0;

    Layer* layer = nullptr;

    vector<pair<type*, dimensions>> inputs_derivatives;

    bool is_first_layer = false;
};

}
#endif
