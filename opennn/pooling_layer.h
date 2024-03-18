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

#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
//#include "convolutional_layer.h"
#include "layer.h"
#include "flatten_layer.h"

#include "statistics.h"

namespace opennn
{

class ConvolutionalLayer;

struct PoolingLayerForwardPropagation;
struct PoolingLayerBackPropagation;
struct ConvolutionalLayerForwardPropagation;
struct ConvolutionalLayerBackPropagation;

/// This class represents the Pooling Layer in Convolutional Neural Network(CNN).
/// Pooling: is the procross_entropy_errors of merging, ie, reducing the size of the data and remove some noise by different processes.

class PoolingLayer : public Layer
{

public:

    /// Enumeration of the available methods for pooling data.

    enum class PoolingMethod{NoPooling, MaxPooling, AveragePooling};

    // Constructors

    explicit PoolingLayer();

    explicit PoolingLayer(const Tensor<Index, 1>&);

    explicit PoolingLayer(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    // Get methods

    Tensor<Index, 1> get_inputs_dimensions() const;
    Tensor<Index, 1> get_outputs_dimensions() const;

    Index get_inputs_number() const;

    Index get_channels_number() const;

    Index get_inputs_rows_number() const;

    Index get_inputs_columns_number() const;

    Index get_neurons_number() const;

    Index get_outputs_rows_number() const;

    Index get_outputs_columns_number() const;

    Index get_padding_width() const;

    Index get_row_stride() const;

    Index get_column_stride() const;

    Index get_pool_rows_number() const;

    Index get_pool_columns_number() const;

    Index get_parameters_number() const final;

    Tensor<type, 1> get_parameters() const final;

    PoolingMethod get_pooling_method() const;

    string write_pooling_method() const;

    // Set methods

    void set(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    void set_inputs_number(const Index&) {}
    void set_neurons_number(const Index&) {}
    void set_name(const string&);

    void set_inputs_dimensions(const Tensor<Index, 1>&);

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

    // Delta methods

    void calculate_hidden_delta(LayerForwardPropagation*,
                                LayerBackPropagation*,
                                LayerForwardPropagation*,
                                LayerBackPropagation*) const final;

    void calculate_hidden_delta(ConvolutionalLayerForwardPropagation*,
                                ConvolutionalLayerBackPropagation*,
                                LayerForwardPropagation*,
                                LayerBackPropagation*) const;

    void calculate_hidden_delta(PoolingLayerForwardPropagation*,
                                PoolingLayerBackPropagation*,
                                PoolingLayerForwardPropagation*,
                                PoolingLayerBackPropagation*) const;

    void calculate_hidden_delta(FlattenLayerForwardPropagation*,
                                FlattenLayerBackPropagation*,
                                PoolingLayerForwardPropagation*,
                                PoolingLayerBackPropagation*) const;

    // Serialization methods

    void from_XML(const tinyxml2::XMLDocument&) final;
    void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

    Tensor<Index, 1> inputs_dimensions;

    Index pool_rows_number = 2;

    Index pool_columns_number = 2;

    Index padding_width = 0;

    Index row_stride = 1;

    Index column_stride = 1;

    PoolingMethod pooling_method = PoolingMethod::AveragePooling;

    const Eigen::array<ptrdiff_t, 4> convolution_dimensions = {0, 1, 2, 3}; // For average pooling
    const Eigen::array<ptrdiff_t, 2> max_pooling_dimensions = {1, 2};

#ifdef OPENNN_CUDA
//    #include "../../opennn-cuda/opennn-cuda/pooling_layer_cuda.h"
#endif

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

    Tensor<Index, 1> inputs_max_indices;
};


struct PoolingLayerBackPropagation : LayerBackPropagation
{

    explicit PoolingLayerBackPropagation();

    explicit PoolingLayerBackPropagation(const Index&, Layer*);

    virtual ~PoolingLayerBackPropagation();    
    
    pair<type*, dimensions> get_deltas_pair() const final;

    void set(const Index&, Layer*) final;

    void print() const;

    Tensor<type, 4> deltas;

};

#ifdef OPENNN_CUDA
//    #include "../../opennn-cuda/opennn-cuda/struct_convolutional_layer_cuda.h"
#endif

}

#endif // POOLING_LAYER_H
