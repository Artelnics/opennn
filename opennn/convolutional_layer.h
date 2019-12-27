//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <ctype.h>
#include <stdexcept>

// OpenNN includes

#include "vector.h"
#include "matrix.h"
#include "layer.h"
#include "functions.h"
//#include "pooling_layer.h"
//#include "perceptron_layer.h"
//#include "probabilistic_layer.h"

#include "tinyxml2.h"

namespace OpenNN
{

class PoolingLayer;
class PerceptronLayer;
class ProbabilisticLayer;


class ConvolutionalLayer : public Layer
{

public:

    /// Enumeration of available activation functions for the convolutional layer.

    enum ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear, ExponentialLinear, ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

    enum PaddingOption{NoPadding, Same};

    // Constructors

    explicit ConvolutionalLayer();

    explicit ConvolutionalLayer(const Vector<size_t>&, const Vector<size_t>&);

    // Destructor

    // Get methods

    bool is_empty() const;

    Vector<double> get_biases() const;
    Vector<double> extract_biases(const Vector<double>&) const;

    Tensor<double> get_synaptic_weights() const;
    Tensor<double> extract_synaptic_weights(const Vector<double>&) const;

    ActivationFunction get_activation_function() const;

    Vector<size_t> get_outputs_dimensions() const;

    size_t get_outputs_rows_number() const;

    size_t get_outputs_columns_number() const;

    PaddingOption get_padding_option() const;

    size_t get_column_stride() const;

    size_t get_row_stride() const;

    size_t get_filters_number() const;

    size_t get_filters_channels_number() const;

    size_t get_filters_rows_number() const;

    size_t get_filters_columns_number() const;

    size_t get_padding_width() const;
    size_t get_padding_height() const;

    size_t get_inputs_channels_number() const;
    size_t get_inputs_rows_number() const;
    size_t get_inputs_columns_number() const;

    size_t get_inputs_number() const;
    size_t get_neurons_number() const;

    Vector<double> get_parameters() const;
    size_t get_parameters_number() const;

    // Set methods

    void set(const Vector<size_t>&, const Vector<size_t>&);

    void set_activation_function(const ActivationFunction&);

    void set_biases(const Vector<double>&);

    void set_synaptic_weights(const Tensor<double>&);

    void set_padding_option(const PaddingOption&);

    void set_parameters(const Vector<double>&);

    void set_row_stride(const size_t&);

    void set_column_stride(const size_t&);

    // Initialization

    void initialize_biases(const double&);

    void initialize_synaptic_weights(const double&);

    void initialize_parameters(const double&);

    // Combinations

    Matrix<double> calculate_image_convolution(const Tensor<double>&, const Tensor<double>&) const;

    Tensor<double> calculate_convolutions(const Tensor<double>&) const;
    Tensor<double> calculate_convolutions(const Tensor<double>&, const Vector<double>&) const;

    // Activation

    Tensor<double> calculate_activations(const Tensor<double>&) const;

    Tensor<double> calculate_activations_derivatives(const Tensor<double>&) const;

   // Outputs

   Tensor<double> calculate_outputs(const Tensor<double>&);
   Tensor<double> calculate_outputs(const Tensor<double>&, const Vector<double>&);

   FirstOrderActivations calculate_first_order_activations(const Tensor<double>&);

   // Delta methods

   Tensor<double> calculate_output_delta(const Tensor<double>&, const Tensor<double>&) const;

   Tensor<double> calculate_hidden_delta(Layer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;

   Tensor<double> calculate_hidden_delta_convolutional(ConvolutionalLayer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;
   Tensor<double> calculate_hidden_delta_pooling(PoolingLayer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;
   Tensor<double> calculate_hidden_delta_perceptron(PerceptronLayer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;
   Tensor<double> calculate_hidden_delta_probabilistic(ProbabilisticLayer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;

   // Gradient methods

   Vector<double> calculate_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&);

   // Padding methods

   Tensor<double> insert_padding(const Tensor<double>&) const;

protected:

   Tensor<double> synaptic_weights;

   Vector<double> biases;

   size_t row_stride = 1;

   size_t column_stride = 1;

   Vector<size_t> inputs_dimensions;

   PaddingOption padding_option = NoPadding;

   ActivationFunction activation_function = RectifiedLinear;
};
}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software

// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
