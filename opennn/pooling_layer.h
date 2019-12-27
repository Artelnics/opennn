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

#include "layer.h"
#include "matrix.h"
#include "statistics.h"
#include "tinyxml2.h"
#include "vector.h"
#include "functions.h"
#include "perceptron_layer.h"
//#include "probabilistic_layer.h"
#include "convolutional_layer.h"

namespace OpenNN
{

/// This class is used to store information about the Pooling Layer in Convolutional Neural Network(CNN).

///
/// Pooling: is the procees of merging, ie, reducing the size of the data and remove some noise by different processes.

class PoolingLayer : public Layer
{

public:

    /// Enumeration of available methods for pooling data.

    enum PoolingMethod {NoPooling, MaxPooling, AveragePooling};

    // Constructors

    explicit PoolingLayer();

    explicit PoolingLayer(const Vector<size_t>&);

    explicit PoolingLayer(const Vector<size_t>&, const Vector<size_t>&);

    // Destructor

    virtual ~PoolingLayer();

     // Get methods

     Vector<size_t> get_input_variables_dimensions() const;
     Vector<size_t> get_outputs_dimensions() const;

     size_t get_inputs_number() const;

     size_t get_inputs_channels_number() const;

     size_t get_inputs_rows_number() const;

     size_t get_inputs_columns_number() const;

     size_t get_neurons_number() const;

     size_t get_outputs_rows_number() const;

     size_t get_outputs_columns_number() const;

     size_t get_padding_width() const;

     size_t get_row_stride() const;

     size_t get_column_stride() const;

     size_t get_pool_rows_number() const;

     size_t get_pool_columns_number() const;

     size_t get_parameters_number() const;

     Vector<double> get_parameters() const;

     Vector<size_t> get_inputs_indices(const size_t&) const;

     PoolingMethod get_pooling_method() const;

     // Set methods

     void set_inputs_number(const size_t&) {}
     void set_neurons_number(const size_t&) {}

     void set_inputs_dimensions(const Vector<size_t>&);

    void set_padding_width(const size_t&);

    void set_row_stride(const size_t&);

    void set_column_stride(const size_t&);

    void set_pool_size(const size_t&, const size_t&);

    void set_pooling_method(const PoolingMethod&);

    void set_default();

    // Outputs

    Tensor<double> calculate_outputs(const Tensor<double>&);

    Tensor<double> calculate_no_pooling_outputs(const Tensor<double>&) const;

    Tensor<double> calculate_max_pooling_outputs(const Tensor<double>&) const;

    Tensor<double> calculate_average_pooling_outputs(const Tensor<double>&) const;

    FirstOrderActivations calculate_first_order_activations(const Tensor<double>&);

    // Activations derivatives

    Tensor<double> calculate_activations_derivatives(const Tensor<double>&) const;

    Tensor<double> calculate_no_pooling_activations_derivatives(const Tensor<double>&) const;

    Tensor<double> calculate_average_pooling_activations_derivatives(const Tensor<double>&) const;

    Tensor<double> calculate_max_pooling_activations_derivatives(const Tensor<double>&) const;

    // Delta methods

    Tensor<double> calculate_output_delta(const Tensor<double>&, const Tensor<double>&) const;

    Tensor<double> calculate_hidden_delta(Layer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;

    Tensor<double> calculate_average_pooling_delta(Layer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;

    Tensor<double> calculate_max_pooling_delta(const Layer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;

    // Gradient methods

    Vector<double> calculate_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&);

protected:

    Vector<size_t> inputs_dimensions;

    size_t pool_rows_number = 2;

    size_t pool_columns_number = 2;

    size_t padding_width = 0;

    size_t row_stride = 1;

    size_t column_stride = 1;

    PoolingMethod pooling_method = AveragePooling;

};
}

#endif // POOLING_LAYER_H

