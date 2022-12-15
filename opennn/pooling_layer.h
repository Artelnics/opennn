//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

//// System includes

#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "layer.h"

#include "statistics.h"
#include "perceptron_layer.h"

namespace opennn
{

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

    Tensor<Index, 1> get_outputs_dimensions() const;

    Index get_inputs_number() const;

    Index get_inputs_channels_number() const;

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

    Index get_parameters_number() const;

    Tensor<type, 1> get_parameters() const;

    PoolingMethod get_pooling_method() const;

    Tensor<Index, 1> get_input_variables_dimensions() const;

    string write_pooling_method() const;

    // Set methods

    void set_inputs_number(const Index&) {}
    void set_neurons_number(const Index&) {}

    void set_input_variables_dimensions(const Tensor<Index, 1>&);

    void set_padding_width(const Index&);

    void set_row_stride(const Index&);

    void set_column_stride(const Index&);

    void set_pool_size(const Index&, const Index&);

    void set_pooling_method(const PoolingMethod&);
    void set_pooling_method(const string&);

    void set_default();

    // Outputs

//    Tensor<type, 4> calculate_outputs(const Tensor<type, 4>&) final;

    void calculate_outputs(type*, const Tensor<Index, 1>&,  type*, const Tensor<Index, 1>&) final;

    void calculate_activations(const Tensor<type, 4>&, Tensor<type, 4>&) {}

    Tensor<type, 4> calculate_no_pooling_outputs(const Tensor<type, 4>&) const;

    Tensor<type, 4> calculate_max_pooling_outputs(const Tensor<type, 4>&) const;

    Tensor<type, 4> calculate_average_pooling_outputs(const Tensor<type, 4>&) const;

    // Activations derivatives

    Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const;

    void calculate_activations_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const
    {

    }

    // First order activations

    void forward_propagate(const Tensor<type, 4>&, LayerForwardPropagation*)
    {
    }

    // Delta methods

    void calculate_hidden_delta(LayerForwardPropagation*,
                                LayerBackPropagation*,
                                LayerBackPropagation*) const;

//    Tensor<type, 4> calculate_hidden_delta_convolutional(ConvolutionalLayer*, const Tensor<type, 4>&, const Tensor<type, 4>&, const Tensor<type, 4>&) const;
    Tensor<type, 4> calculate_hidden_delta_pooling(PoolingLayer*, const Tensor<type, 4>&, const Tensor<type, 4>&, const Tensor<type, 4>&) const;
    Tensor<type, 4> calculate_hidden_delta_perceptron(PerceptronLayer*, const Tensor<type, 4>&, const Tensor<type, 4>&, const Tensor<type, 4>&) const;
    Tensor<type, 4> calculate_hidden_delta_probabilistic(ProbabilisticLayer*, const Tensor<type, 4>&, const Tensor<type, 4>&, const Tensor<type, 4>&) const;

    // Gradient methods

    void from_XML(const tinyxml2::XMLDocument&) final;
    void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

    Tensor<Index, 1> input_variables_dimensions;

    Index pool_rows_number = 2;

    Index pool_columns_number = 2;

    Index padding_width = 0;

    Index row_stride = 1;

    Index column_stride = 1;

    PoolingMethod pooling_method = PoolingMethod::AveragePooling;

#ifdef OPENNN_CUDA
#include "../../opennn-cuda/opennn-cuda/pooling_layer_cuda.h"
#endif

};
}

#endif // POOLING_LAYER_H

