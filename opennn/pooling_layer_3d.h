//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S E Q U E N C E   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SEQUENCE_POOLING_LAYER_H
#define SEQUENCE_POOLING_LAYER_H

#include "layer.h"

namespace opennn
{

class Pooling3d : public Layer
{

public:

    enum class PoolingMethod{MaxPooling, AveragePooling};

    Pooling3d(const dimensions& = {0, 0}, // Input dimensions {sequence_length, features}
              const PoolingMethod& = PoolingMethod::MaxPooling,
              const string& = "sequence_pooling_layer");

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;
    PoolingMethod get_pooling_method() const;
    string write_pooling_method() const;

    void set(const dimensions&, const PoolingMethod&, const string&);
    void set_pooling_method(const PoolingMethod&);
    void set_pooling_method(const string&);

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    void print() const override;

private:
    dimensions input_dimensions;
    PoolingMethod pooling_method;
};


struct Pooling3dForwardPropagation : LayerForwardPropagation
{
    Pooling3dForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type*, dimensions> get_output_pair() const override;

    void set(const Index&, Layer*) override;

    Tensor<type, 2> outputs;

    Tensor<Index, 2> maximal_indices;
};


struct Pooling3dBackPropagation : LayerBackPropagation
{
    Pooling3dBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index&, Layer*) override;

    Tensor<type, 3> input_derivatives;
};

}

#endif // SEQUENCE_POOLING_LAYER_H
