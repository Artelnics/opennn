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
#include "functions.h"
#include "config.h"

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

    explicit ConvolutionalLayer(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    // Destructor

    // Get methods

    bool is_empty() const;

    Tensor<type, 1> get_biases() const;
    Tensor<type, 1> extract_biases(const Tensor<type, 1>&) const;

    Tensor<type, 2> get_synaptic_weights() const;
    Tensor<type, 2> extract_synaptic_weights(const Tensor<type, 1>&) const;

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

    void set_parameters(const Tensor<type, 1>&);

    void set_row_stride(const Index&);

    void set_column_stride(const Index&);

    // Initialization

    void initialize_biases(const type&);

    void initialize_synaptic_weights(const type&);

    void set_parameters_constant(const type&);

    // Combinations

    Tensor<type, 2> calculate_image_convolution(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

    Tensor<type, 2> calculate_combinations(const Tensor<type, 2>&) const;
    Tensor<type, 2> calculate_combinations(const Tensor<type, 2>&, const Tensor<type, 1>&) const;

    void calculate_combinations(const Tensor<type, 2>& inputs, Tensor<type, 2>& convolutions) const
    {
/*
        // Inputs

        const Index images_number = inputs.dimension(0);
        const Index channels_number = get_inputs_channels_number();

        // Filters

        const Index filters_number = get_filters_number();
        const Index filters_rows_number = get_filters_rows_number();
        const Index filters_columns_number = get_filters_columns_number();

        // Outputs

        const Index outputs_rows_number = get_outputs_rows_number();
        const Index outputs_columns_number = get_outputs_columns_number();

        // Convolution loops

//        #pragma omp parallel for

        for(Index image_index = 0; image_index < images_number; image_index++)
        {
            for(Index filter_index = 0; filter_index < filters_number; filter_index++)
            {
                for(Index output_row_index = 0; output_row_index < outputs_rows_number; output_row_index++)
                {
                    for(Index output_column_index = 0; output_column_index < outputs_columns_number; output_column_index++)
                    {
                        type sum = 0.0;

                        for(Index channel_index = 0; channel_index < channels_number; channel_index++)
                        {
                            for(Index filter_row_index = 0; filter_row_index < filters_rows_number; filter_row_index++)
                            {
                                const Index row = output_row_index*row_stride + filter_row_index;

                                for(Index filter_column_index = 0; filter_column_index < filters_columns_number; filter_column_index++)
                                {
                                    const Index column = output_column_index*column_stride + filter_column_index;

                                    const type image_element = inputs(image_index, channel_index, row, column);
                                    const type filter_element = synaptic_weights(filter_index, channel_index, filter_row_index, filter_column_index);

                                    sum += image_element*filter_element;
                                }
                            }
                        }

                        convolutions(image_index, filter_index, output_row_index, output_column_index) = sum + biases[filter_index];
                    }
                }
            }
        }
*/
    }

    // Activation

    Tensor<type, 2> calculate_activations(const Tensor<type, 2>&) const;

    Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const;

    void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const
    {

    }

    void calculate_activations_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const
    {

    }

   // Outputs

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);
   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&);

   ForwardPropagation calculate_forward_propagation(const Tensor<type, 2>&);

   void calculate_forward_propagation(const Tensor<type, 2>& inputs, ForwardPropagation& forward_propagation)
   {
/*
       calculate_combinations(inputs, forward_propagation.combinations);

       calculate_activations(forward_propagation.combinations, forward_propagation.activations);

       calculate_activations_derivatives(forward_propagation.combinations, forward_propagation.activations_derivatives);
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

   // Padding methods

   Tensor<type, 2> insert_padding(const Tensor<type, 2>&) const;

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
