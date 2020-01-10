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

    Tensor<double> calculate_combinations(const Tensor<double>&) const;
    Tensor<double> calculate_combinations(const Tensor<double>&, const Vector<double>&) const;

    void calculate_combinations(const Tensor<double>& inputs, Tensor<double>& convolutions) const
    {
        // Inputs

        const size_t images_number = inputs.get_dimension(0);
        const size_t channels_number = get_inputs_channels_number();

        // Filters

        const size_t filters_number = get_filters_number();
        const size_t filters_rows_number = get_filters_rows_number();
        const size_t filters_columns_number = get_filters_columns_number();

        // Outputs

        const size_t outputs_rows_number = get_outputs_rows_number();
        const size_t outputs_columns_number = get_outputs_columns_number();

        // Convolution loops

//        #pragma omp parallel for

        for(size_t image_index = 0; image_index < images_number; image_index++)
        {
            for(size_t filter_index = 0; filter_index < filters_number; filter_index++)
            {
                for(size_t output_row_index = 0; output_row_index < outputs_rows_number; output_row_index++)
                {
                    for(size_t output_column_index = 0; output_column_index < outputs_columns_number; output_column_index++)
                    {
                        double sum = 0.0;

                        for(size_t channel_index = 0; channel_index < channels_number; channel_index++)
                        {
                            for(size_t filter_row_index = 0; filter_row_index < filters_rows_number; filter_row_index++)
                            {
                                const size_t row = output_row_index*row_stride + filter_row_index;

                                for(size_t filter_column_index = 0; filter_column_index < filters_columns_number; filter_column_index++)
                                {
                                    const size_t column = output_column_index*column_stride + filter_column_index;

                                    const double image_element = inputs(image_index, channel_index, row, column);
                                    const double filter_element = synaptic_weights(filter_index, channel_index, filter_row_index, filter_column_index);

                                    sum += image_element*filter_element;
                                }
                            }
                        }

                        convolutions(image_index, filter_index, output_row_index, output_column_index) = sum + biases[filter_index];
                    }
                }
            }
        }
    }

    // Activation

    Tensor<double> calculate_activations(const Tensor<double>&) const;

    Tensor<double> calculate_activations_derivatives(const Tensor<double>&) const;

    void calculate_activations(const Tensor<double>&, Tensor<double>&) const
    {

    }

    void calculate_activations_derivatives(const Tensor<double>&, Tensor<double>&) const
    {

    }

   // Outputs

   Tensor<double> calculate_outputs(const Tensor<double>&);
   Tensor<double> calculate_outputs(const Tensor<double>&, const Vector<double>&);

   ForwardPropagation calculate_forward_propagation(const Tensor<double>&);

   void calculate_forward_propagation(const Tensor<double>& inputs, ForwardPropagation& forward_propagation)
   {
       calculate_combinations(inputs, forward_propagation.combinations);

       calculate_activations(forward_propagation.combinations, forward_propagation.activations);

       calculate_activations_derivatives(forward_propagation.combinations, forward_propagation.activations_derivatives);
   }

   // Delta methods

   Tensor<double> calculate_output_delta(const Tensor<double>&, const Tensor<double>&) const;

   Tensor<double> calculate_hidden_delta(Layer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;

   Tensor<double> calculate_hidden_delta_convolutional(ConvolutionalLayer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;
   Tensor<double> calculate_hidden_delta_pooling(PoolingLayer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;
   Tensor<double> calculate_hidden_delta_perceptron(PerceptronLayer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;
   Tensor<double> calculate_hidden_delta_probabilistic(ProbabilisticLayer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;

   // Gradient methods

   Vector<double> calculate_error_gradient(const Tensor<double>&, const Layer::ForwardPropagation&, const Tensor<double>&);

   // Padding methods

   Tensor<double> insert_padding(const Tensor<double>&) const;

protected:

   /// This tensor containing conection strengths from a layer's inputs to its neurons.

   Tensor<double> synaptic_weights;

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Vector<double> biases;

   size_t row_stride = 1;

   size_t column_stride = 1;

   Vector<size_t> input_variables_dimensions;

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
