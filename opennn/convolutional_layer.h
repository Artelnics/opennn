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

    struct ConvolutionalLayerForwardPropagation : ForwardPropagation
    {
        /// Default constructor.

        explicit ConvolutionalLayerForwardPropagation() : ForwardPropagation(){}

        virtual ~ConvolutionalLayerForwardPropagation() {}

        void allocate()
        {
//            const ConvolutionalLayer* convolutional_layer = dynamic_cast<ConvolutionalLayer*>(trainable_layers_pointers[i]);

//            const Index outputs_channels_number = convolutional_layer->get_filters_number();
//            const Index outputs_rows_number = convolutional_layer->get_outputs_rows_number();
//            const Index outputs_columns_number = convolutional_layer->get_outputs_columns_number();

//            layers[i].combinations_2d.resize(Tensor<Index, 1>({batch_instances_number, outputs_channels_number, outputs_rows_number, outputs_columns_number}));
//            layers[i].activations_2d.resize(Tensor<Index, 1>({batch_instances_number, outputs_channels_number, outputs_rows_number, outputs_columns_number}));
//            layers[i].activations_derivatives.resize(Tensor<Index, 1>({batch_instances_number, outputs_channels_number, outputs_rows_number, outputs_columns_number}));
        }

    };

    /// Enumeration of available activation functions for the convolutional layer.

    enum ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear, ExponentialLinear, ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

    enum PaddingOption{NoPadding, Same};

    // Constructors

    explicit ConvolutionalLayer();

    explicit ConvolutionalLayer(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    // Destructor

    // Get methods

    bool is_empty() const;

    Tensor<type, 1> get_biases() const;

    Tensor<type, 2> get_synaptic_weights() const;

    ActivationFunction get_activation_function() const;

    Tensor<Index, 1> get_outputs_dimensions() const;

    Index get_outputs_rows_number() const;

    Index get_outputs_columns_number() const;

    PaddingOption get_padding_option() const;

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

    void set_activation_function(const ActivationFunction&);

    void set_biases(const Tensor<type, 1>&);

    void set_synaptic_weights(const Tensor<type, 2>&);

    void set_padding_option(const PaddingOption&);

    void set_parameters(const Tensor<type, 1>&, const Index& index);

    void set_row_stride(const Index&);

    void set_column_stride(const Index&);

    // Initialization

    void set_biases_constant(const type&);

    void set_synaptic_weights_constant(const type&);

    void set_parameters_constant(const type&);

    // Combinations

    void calculate_convolutions(const Tensor<type, 4>&, Tensor<type, 4>&) const
    {
    }

    // Activation

    void calculate_activations(const Tensor<type, 4>&, Tensor<type, 4>&) const
    {

    }

    void calculate_activations_derivatives(const Tensor<type, 4>&, Tensor<type, 4>&) const
    {

    }

   // Outputs

   Tensor<type, 4> calculate_outputs(const Tensor<type, 4>&);

   void forward_propagate(const Tensor<type, 4>& inputs, ForwardPropagation& forward_propagation) const
   {
       calculate_convolutions(inputs, forward_propagation.combinations_4d);

       calculate_activations(forward_propagation.combinations_4d, forward_propagation.activations_4d);

       calculate_activations_derivatives(forward_propagation.combinations_4d, forward_propagation.activations_derivatives_4d);
   }

   // Delta methods

   Tensor<type, 2> calculate_hidden_delta(Layer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<type, 2> calculate_hidden_delta_convolutional(ConvolutionalLayer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 4>&) const;
   Tensor<type, 2> calculate_hidden_delta_pooling(PoolingLayer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;
   Tensor<type, 2> calculate_hidden_delta_perceptron(PerceptronLayer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;
   Tensor<type, 2> calculate_hidden_delta_probabilistic(ProbabilisticLayer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   // Gradient methods

   Tensor<type, 1> calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);

protected:

   /// This tensor containing conection strengths from a layer's inputs to its neurons.

   Tensor<type, 2> synaptic_weights;

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Tensor<type, 1> biases;

   Index row_stride = 1;

   Index column_stride = 1;

   Tensor<Index, 1> input_variables_dimensions;

   PaddingOption padding_option = NoPadding;

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
