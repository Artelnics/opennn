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
#include "layer.h"

#include "statistics.h"
#include "tinyxml2.h"

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

    explicit PoolingLayer(const vector<int>&);

    explicit PoolingLayer(const vector<int>&, const vector<int>&);

    // Destructor

    virtual ~PoolingLayer();

     // Get methods

     vector<int> get_input_variables_dimensions() const;
     vector<int> get_outputs_dimensions() const;

     int get_inputs_number() const;

     int get_inputs_channels_number() const;

     int get_inputs_rows_number() const;

     int get_inputs_columns_number() const;

     int get_neurons_number() const;

     int get_outputs_rows_number() const;

     int get_outputs_columns_number() const;

     int get_padding_width() const;

     int get_row_stride() const;

     int get_column_stride() const;

     int get_pool_rows_number() const;

     int get_pool_columns_number() const;

     int get_parameters_number() const;

     Tensor<type, 1> get_parameters() const;

     vector<int> get_inputs_indices(const int&) const;

     PoolingMethod get_pooling_method() const;

     // Set methods

     void set_inputs_number(const int&) {}
     void set_neurons_number(const int&) {}

     void set_input_variables_dimensions(const vector<int>&);

    void set_padding_width(const int&);

    void set_row_stride(const int&);

    void set_column_stride(const int&);

    void set_pool_size(const int&, const int&);

    void set_pooling_method(const PoolingMethod&);

    void set_default();

    // Outputs

    Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

    void calculate_activations(const Tensor<type, 2>&,  Tensor<type, 2>&) {}

    Tensor<type, 2> calculate_no_pooling_outputs(const Tensor<type, 2>&) const;

    Tensor<type, 2> calculate_max_pooling_outputs(const Tensor<type, 2>&) const;

    Tensor<type, 2> calculate_average_pooling_outputs(const Tensor<type, 2>&) const;    

    // Activations derivatives

    Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const;

    void calculate_activations_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const
    {

    }

    Tensor<type, 2> calculate_no_pooling_activations_derivatives(const Tensor<type, 2>&) const;

    Tensor<type, 2> calculate_average_pooling_activations_derivatives(const Tensor<type, 2>&) const;

    Tensor<type, 2> calculate_max_pooling_activations_derivatives(const Tensor<type, 2>&) const;

    // First order activations

    ForwardPropagation calculate_forward_propagation(const Tensor<type, 2>&);

    void calculate_forward_propagation(const Tensor<type, 2>& inputs, ForwardPropagation& forward_propagation)
    {
/*
        calculate_activations(inputs, forward_propagation.activations);

        calculate_activations_derivatives(forward_propagation.activations, forward_propagation.activations_derivatives);
*/
    }

    // Delta methods

    Tensor<type, 2> calculate_output_delta(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

    Tensor<type, 2> calculate_hidden_delta(Layer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;

    Tensor<type, 2> calculate_hidden_delta_convolutional(ConvolutionalLayer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;
    Tensor<type, 2> calculate_hidden_delta_pooling(PoolingLayer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;
    Tensor<type, 2> calculate_hidden_delta_perceptron(PerceptronLayer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;
    Tensor<type, 2> calculate_hidden_delta_probabilistic(ProbabilisticLayer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;

    // Gradient methods

    Tensor<type, 1> calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);

protected:

    vector<int> input_variables_dimensions;

    int pool_rows_number = 2;

    int pool_columns_number = 2;

    int padding_width = 0;

    int row_stride = 1;

    int column_stride = 1;

    PoolingMethod pooling_method = AveragePooling;

};
}

#endif // POOLING_LAYER_H

