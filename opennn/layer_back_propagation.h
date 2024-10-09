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
    
    virtual vector<pair<type*, dimensions>> get_input_derivative_pairs() const = 0;

    virtual void set(const Index&, Layer*) {}

    virtual void print() const {}

    Index batch_samples_number = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;
};

}
#endif
