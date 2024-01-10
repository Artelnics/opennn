#ifndef LAYERFORWARDPROPAGATION_H
#define LAYERFORWARDPROPAGATION_H

#include <string>

#include "layer.h"
#include "config.h"

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

    Index batch_samples_number = type(0);

    Layer* layer_pointer = nullptr;

    type* outputs_data = nullptr;

    virtual pair<type*, dimensions> get_outputs() const = 0;
};


}
#endif
