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

    virtual ~LayerBackPropagation()
    {
        free(deltas_data);
    }

    virtual void set(const Index&, Layer*) {}

    virtual void print() const {}

    virtual Tensor< TensorMap< Tensor<type, 1> >*, 1> get_layer_gradient()
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "virtual Tensor< TensorMap< Tensor<type, 1> >*, 1> get_layer_gradient() method.\n"
               << "This method is not implemented in the layer type (" << layer_pointer->get_type_string() << ").\n";

        throw invalid_argument(buffer.str());
    }

    Index batch_samples_number;

    Layer* layer_pointer = nullptr;

    type* deltas_data = nullptr;

    Tensor<Index, 1> deltas_dimensions;

    Tensor<type, 1> gradient;

};

}
#endif
