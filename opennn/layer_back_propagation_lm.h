#ifndef LAYERBACKPROPAGATIONLM_H
#define LAYERBACKPROPAGATIONLM_H

#include "layer.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct LayerBackPropagationLM
{
    explicit LayerBackPropagationLM() {}

    virtual ~LayerBackPropagationLM() {}

    virtual vector<pair<type*, dimensions>> get_input_derivative_pairs() const = 0;

    virtual void set(const Index&, Layer*) {}

    virtual void print() const {}

    Index batch_samples_number = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;
};

}
#endif
