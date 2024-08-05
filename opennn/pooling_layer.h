//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

// System includes

//#include <cmath>
//#include <cstdlib>
//#include <ctype.h>
//#include <fstream>
//#include <iostream>
#include <string>
//#include <sstream>

// OpenNN includes

#include "config.h"
//#include "convolutional_layer.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

//#include "flatten_layer.h"

#include "statistics.h"

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


/// This class represents the Pooling Layer in Convolutional Neural Network(CNN).
/// Pooling: is the procross_entropy_errors of merging, ie, reducing the size of the data and remove some noise by different processes.

class PoolingLayer : public Layer
{

public:

    /// Enumeration of the available methods for pooling data.

    enum class PoolingMethod{NoPooling, MaxPooling, AveragePooling};

    // Constructors

    explicit PoolingLayer();

    explicit PoolingLayer(const dimensions&, const dimensions& = {1, 1});

    // Get methods

    dimensions get_inputs_dimensions() const;

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

    // Set methods

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

    // Serialization methods

    void from_XML(const tinyxml2::XMLDocument&) final;
    void write_XML(tinyxml2::XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/pooling_layer_cuda.h"
    #endif

protected:

    dimensions input_dimensions;

    Index pool_height = 2;

    Index pool_width = 2;

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

    Tensor<type, 4> outputs;

    Tensor<type, 5> image_patches;

    Tensor<Index, 4> inputs_max_indices;
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
