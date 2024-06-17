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
    /// Default constructor.

    explicit LayerBackPropagation() {}

    virtual ~LayerBackPropagation() {}
    
    Tensor<pair<type*, dimensions>, 1>& get_inputs_derivatives_pair()
    {
        return inputs_derivatives;
    }

    virtual void set(const Index&, Layer*) {}

    virtual void print() const {}

    Index batch_samples_number = 0;

    Layer* layer = nullptr;

    Tensor<pair<type*, dimensions>, 1> inputs_derivatives;

    bool is_first_layer = false;
};

}
#endif
