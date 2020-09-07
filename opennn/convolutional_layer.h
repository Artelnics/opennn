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

#include "layer.h"
#include "config.h"

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

    enum ConvolutionType{Valid, Same};

    // Constructors

    explicit ConvolutionalLayer();

    explicit ConvolutionalLayer(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    // Destructor

    // Get methods

    bool is_empty() const;

    const Tensor<type, 1>& get_biases() const;

    const Tensor<type, 4>& get_synaptic_weights() const;

    Index get_synaptic_weights_number() const;

    ActivationFunction get_activation_function() const;

    Tensor<Index, 1> get_outputs_dimensions() const;

    Tensor<Index, 1> get_input_variables_dimensions() const;

    Index get_outputs_rows_number() const;

    Index get_outputs_columns_number() const;

    ConvolutionType get_convolution_type() const;

    Index get_column_stride() const;

    Index get_row_stride() const;

    Index get_filters_number() const;
    Index get_filters_channels_number() const;
    Index get_filters_rows_number() const;
    Index get_filters_columns_number() const;

    Index get_padding_width() const;
    Index get_padding_height() const;

    Index get_inputs_channels_number() const;
    Index get_inputs_rows_number() const;
    Index get_inputs_columns_number() const;

    Index get_inputs_number() const;
    Index get_neurons_number() const;

    Tensor<type, 1> get_parameters() const;
    Index get_parameters_number() const;

    // Set methods

    void set(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    void set(const Tensor<type, 4>&, const Tensor<type, 4>&, const Tensor<type, 1>&);

    void set_activation_function(const ActivationFunction&);

    void set_biases(const Tensor<type, 1>&);

    void set_synaptic_weights(const Tensor<type, 4>&);

    void set_convolution_type(const ConvolutionType&);

    void set_parameters(const Tensor<type, 1>&, const Index& index);

    void set_row_stride(const Index&);

    void set_column_stride(const Index&);

    // Initialization

    void set_biases_constant(const type&);

    void set_synaptic_weights_constant(const type&);

    void set_parameters_constant(const type&);

    void set_parameters_random();

    // Padding

    void insert_padding(const Tensor<type, 4>&, Tensor<type, 4>&);

    // Combinations

    void calculate_combinations(const Tensor<type, 4>&, Tensor<type, 4>&) const;

    void calculate_combinations(const Tensor<type, 4>& inputs,
                                const Tensor<type, 2>& biases,
                                const Tensor<type, 4>& synaptic_weights,
                                Tensor<type, 4>& combinations_4d) const;

    // Activation

    void calculate_activations(const Tensor<type, 4>&, Tensor<type, 4>&) const;

    void calculate_activations_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&, Tensor<type, 4>&) const;

   // Outputs

   void calculate_outputs(const Tensor<type, 4>&, Tensor<type, 4>&);
//   void calculate_outputs(const Tensor<type, 2>&, Tensor<type, 2>&);
   void calculate_outputs(const Tensor<type, 4>&, Tensor<type, 2>&);
//   void calculate_outputs_2d(const Tensor<type, 2>&, Tensor<type, 2>&);

   void forward_propagate(const Tensor<type, 4>&, ForwardPropagation&) const;
   void forward_propagate(const Tensor<type, 2>&, ForwardPropagation&) const;

   void forward_propagate(const Tensor<type, 4>&, Tensor<type, 1>, ForwardPropagation&) const;
   void forward_propagate(const Tensor<type, 2>&, Tensor<type, 1>, ForwardPropagation&) const;


   // Delta methods

//   void calculate_hidden_delta(Layer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&, Tensor<type, 4>&) const;
   void calculate_hidden_delta(Layer* next_layer_pointer,
                               const Tensor<type, 2>&,
                               ForwardPropagation& forward_propagation,
                               const Tensor<type, 2>& next_layer_delta,
                               Tensor<type, 2>& hidden_delta) const;


   void calculate_hidden_delta_convolutional(ConvolutionalLayer*, const Tensor<type, 4>&, const Tensor<type, 4>&, const Tensor<type, 4>&, Tensor<type, 2>&) const;
   void calculate_hidden_delta_pooling(PoolingLayer*, const Tensor<type, 4>&, const Tensor<type, 4>&, const Tensor<type, 2>&, Tensor<type, 2>&) const;
   void calculate_hidden_delta_perceptron(const PerceptronLayer*, const Tensor<type, 4>&, const Tensor<type, 4>&, const Tensor<type, 2>&, Tensor<type, 2>&) const;
   void calculate_hidden_delta_probabilistic(ProbabilisticLayer*, const Tensor<type, 4>&, const Tensor<type, 4>&, const Tensor<type, 2>&, Tensor<type, 2>&) const;

   // Gradient methods

   void calculate_error_gradient(const Tensor<type, 4>&, const Layer::ForwardPropagation&, const Tensor<type, 4>&, Tensor<type, 1>&);
   void calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 4>&, Tensor<type, 1>&);

   void calculate_error_gradient(const Tensor<type, 4>&, const Layer::ForwardPropagation&, Layer::BackPropagation&) const;
   void calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, Layer::BackPropagation&) const;

   void insert_gradient(const BackPropagation&, const Index&, Tensor<type, 1>&) const;

   void to_2d(const Tensor<type, 4>&, Tensor<type, 2>) const;

protected:

   /// This tensor containing conection strengths from a layer's inputs to its neurons.

   Tensor<type, 4> synaptic_weights;

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Tensor<type, 1> biases;

   Index row_stride = 1;

   Index column_stride = 1;

   Tensor<Index, 1> input_variables_dimensions;

   ConvolutionType convolution_type = Valid;

   ActivationFunction activation_function = RectifiedLinear;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/convolutional_layer_cuda.h"
#endif


};
}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
