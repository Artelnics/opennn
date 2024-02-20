#ifndef LAYERBACKPROPAGATIONLM_H
#define LAYERBACKPROPAGATIONLM_H

#include <string>

#include "layer.h"


using namespace std;
using namespace Eigen;

namespace opennn
{

struct LayerBackPropagationLM
{
    /// Default constructor.

    explicit LayerBackPropagationLM() {}

    virtual ~LayerBackPropagationLM() {}

    virtual void set(const Index&, Layer*) {}

    virtual void print() const {}

    Index batch_samples_number;

    Layer* layer = nullptr;

    Tensor<type, 2> deltas;
};

}
#endif
