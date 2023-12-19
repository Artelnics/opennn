#ifndef LAYERFORWARDPROPAGATION_H
#define LAYERFORWARDPROPAGATION_H

#include <string>

#include "dynamic_tensor.h"
#include "layer.h"


using namespace std;
using namespace Eigen;

namespace opennn
{

struct LayerForwardPropagation
{
    /// Default constructor.

    explicit LayerForwardPropagation()
    {
    }

    virtual ~LayerForwardPropagation()
    {
    }

    virtual void set(const Index&, Layer*) = 0;

    virtual void print() const {}

    Index batch_samples_number;

    Layer* layer_pointer = nullptr;

    Tensor<DynamicTensor<type>, 1> outputs;
};


}
#endif
