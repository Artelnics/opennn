//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

#include <string>

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{

class ConvolutionalLayer;

struct PoolingLayerForwardPropagation;
struct PoolingLayerBackPropagation;
struct ConvolutionalLayerForwardPropagation;
struct ConvolutionalLayerBackPropagation;

#ifdef OPENNN_CUDA
struct PoolingLayerForwardPropagationCuda;
struct PoolingLayerBackPropagationCuda;
#endif


class PoolingLayer : public Layer
{

public:

    enum class PoolingMethod{NoPooling, MaxPooling, AveragePooling};

    // Constructors

    explicit PoolingLayer();

    explicit PoolingLayer(const dimensions&, const dimensions& = {1, 1});

    // Get

    dimensions get_input_dimensions() const;

    dimensions get_output_dimensions() const;

    Index get_inputs_number() const;

    Index get_input_height() const;
    Index get_input_width() const;
    Index get_channels_number() const;

    Index get_neurons_number() const;

    Index get_output_height() const;
    Index get_output_width() const;

    Index get_padding_heigth() const;
    Index get_padding_width() const;

    Index get_row_stride() const;
    Index get_column_stride() const;

    Index get_pool_height() const;
    Index get_pool_width() const;

    PoolingMethod get_pooling_method() const;

    string write_pooling_method() const;

    // Set

    void set(const dimensions&, const dimensions&);

    void set_inputs_number(const Index&) {}
    void set_neurons_number(const Index&) {}
    void set_name(const string&);

    void set_inputs_dimensions(const dimensions&);

    void set_padding_heigth(const Index&);
    void set_padding_width(const Index&);

    void set_row_stride(const Index&);
    void set_column_stride(const Index&);

    void set_pool_size(const Index&, const Index&);

    void set_pooling_method(const PoolingMethod&);
    void set_pooling_method(const string&);

    void set_default();

    // Outputs

    // First order activations

    void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                           LayerForwardPropagation*,
                           const bool&) final;

    void forward_propagate_no_pooling(const Tensor<type, 4>&,
                                      LayerForwardPropagation*,
                                      const bool&);

    void forward_propagate_max_pooling(const Tensor<type, 4>&,
                                       LayerForwardPropagation*,
                                       const bool&) const;

    void forward_propagate_average_pooling(const Tensor<type, 4>&,
                                           LayerForwardPropagation*,
                                           const bool&) const;

    // Back-propagation

    void back_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                        const Tensor<pair<type*, dimensions>, 1>&,
                        LayerForwardPropagation*,
                        LayerBackPropagation*) const final;

    void back_propagate_max_pooling(const Tensor<type, 4>&,
                                    const Tensor<type, 4>&,
                                    LayerForwardPropagation*,
                                    LayerBackPropagation*) const;

    void back_propagate_average_pooling(const Tensor<type, 4>&,
                                        const Tensor<type, 4>&,
                                        LayerBackPropagation*) const;

    // Serialization

    void from_XML(const tinyxml2::XMLDocument&) final;
    void to_XML(tinyxml2::XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/pooling_layer_cuda.h"
    #endif

protected:

    dimensions input_dimensions;

    Index pool_height = 1;

    Index pool_width = 1;

    Index padding_heigth = 0;

    Index padding_width = 0;

    Index row_stride = 1;

    Index column_stride = 1;

    PoolingMethod pooling_method = PoolingMethod::AveragePooling;

    const Eigen::array<ptrdiff_t, 4> average_pooling_dimensions = {0, 1, 2, 3}; // For average pooling
    const Eigen::array<ptrdiff_t, 2> max_pooling_dimensions = {1, 2};


};


struct PoolingLayerForwardPropagation : LayerForwardPropagation
{
    // Default constructor

    explicit PoolingLayerForwardPropagation();

    // Constructor

    explicit PoolingLayerForwardPropagation(const Index&, Layer*);
    
    pair<type*, dimensions> get_outputs_pair() const final;

    void set(const Index&, Layer*) final;

    void print() const;

    Tensor<type, 4> pool;

    Tensor<type, 4> outputs;

    Tensor<type, 5> image_patches;

    Tensor<Index, 4> maximal_indices;
};


struct PoolingLayerBackPropagation : LayerBackPropagation
{

    explicit PoolingLayerBackPropagation();

    explicit PoolingLayerBackPropagation(const Index&, Layer*);

    virtual ~PoolingLayerBackPropagation();

    void set(const Index&, Layer*) final;

    void print() const;

    Tensor<type, 4> input_derivatives;

};


#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/pooling_layer_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/pooling_layer_back_propagation_cuda.h"
#endif


}

#endif // POOLING_LAYER_H
