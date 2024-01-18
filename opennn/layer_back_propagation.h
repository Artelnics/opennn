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
    
    virtual pair<type*, dimensions> get_deltas_pair() const
    {
        return pair<type*, dimensions>();
    }

    virtual void set(const Index&, Layer*) {}

    virtual void print() const {}

    Index batch_samples_number = 0;

    Layer* layer_pointer = nullptr;

    type* deltas_data = nullptr;
};

}
#endif
