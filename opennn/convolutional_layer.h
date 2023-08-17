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
#include "flatten_layer.h"
#include "pooling_layer.h"

namespace opennn
{

struct ConvolutionalLayerForwardPropagation;
struct ConvolutionalLayerBackPropagation;

#ifdef OPENNN_CUDA
    struct ConvolutionalLayerForwardPropagationCuda;
#endif

class ConvolutionalLayer : public Layer
{

public:

    /// Enumeration of the available activation functions for the convolutional layer.

    enum class ActivationFunction{Threshold,
                                  SymmetricThreshold,
                                  Logistic,
                                  HyperbolicTangent,
                                  Linear,
                                  RectifiedLinear,
                                  ExponentialLinear,
                                  ScaledExponentialLinear,
                                  SoftPlus,
                                  SoftSign,
                                  HardSigmoid};

    enum class ConvolutionType{Valid, Same};

    // Constructors

    explicit ConvolutionalLayer();

    explicit ConvolutionalLayer(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    // Destructor

    // Get methods

    bool is_empty() const;

    const Tensor<type, 1>& get_biases() const;

    const Tensor<type, 4>& get_synaptic_weights() const;

    bool get_batch_normalization() const;

    Index get_biases_number() const;

    Index get_synaptic_weights_number() const;

    ActivationFunction get_activation_function() const;

    string write_activation_function() const;

    Tensor<Index, 1> get_inputs_dimensions() const;
    Tensor<Index, 1> get_outputs_dimensions() const;

    pair<Index, Index> get_padding() const;

    Eigen::array<pair<Index, Index>, 4> get_paddings() const;

    Eigen::array<ptrdiff_t, 4> get_strides() const;

    Index get_outputs_rows_number() const;

    Index get_outputs_columns_number() const;

    ConvolutionType get_convolution_type() const;
    string write_convolution_type() const;

    Index get_column_stride() const;

    Index get_row_stride() const;

    Index get_kernels_number() const;
    Index get_kernels_channels_number() const;
    Index get_kernels_rows_number() const;
    Index get_kernels_columns_number() const;

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

    void set_name(const string&);

    void set_activation_function(const ActivationFunction&);
    void set_activation_function(const string&);

    void set_biases(const Tensor<type, 1>&);

    void set_synaptic_weights(const Tensor<type, 4>&);

    void set_batch_normalization(const bool&);

    void set_convolution_type(const ConvolutionType&);
    void set_convolution_type(const string&);

    void set_parameters(const Tensor<type, 1>&, const Index& index = 0);

    void set_row_stride(const Index&);

    void set_column_stride(const Index&);

    void set_inputs_dimenisons(const Tensor<Index,1>&);

    // Initialization

    void set_biases_constant(const type&);

    void set_synaptic_weights_constant(const type&);

    void set_parameters_constant(const type&);

    void set_parameters_random();

    // Padding

    void insert_padding(const Tensor<type, 4>&, Tensor<type, 4>&) const;

    // Combinations

    void calculate_convolutions(type*, LayerForwardPropagation*) const;

    void normalize(LayerForwardPropagation*, const bool&);
    void shift(LayerForwardPropagation*);

    void calculate_activations(LayerForwardPropagation*) const;
    void calculate_activations_derivatives(LayerForwardPropagation*) const;

   // Outputs

    void forward_propagate(type*,
                           const Tensor<Index, 1>&,
                           LayerForwardPropagation*,
                           const bool&) final;

   // Outputs

   // Delta methods

   void calculate_hidden_delta(LayerForwardPropagation*,
                               LayerBackPropagation*,
                               LayerBackPropagation*) const final;

   void calculate_hidden_delta(ConvolutionalLayerForwardPropagation*,
                               ConvolutionalLayerBackPropagation*,
                               ConvolutionalLayerBackPropagation*) const;

   void calculate_hidden_delta(PoolingLayerForwardPropagation*,
                               PoolingLayerBackPropagation*,
                               ConvolutionalLayerBackPropagation*) const;

   void calculate_hidden_delta(FlattenLayerForwardPropagation*,
                               FlattenLayerBackPropagation*,
                               ConvolutionalLayerBackPropagation*) const;

   // Gradient methods

   void calculate_error_gradient(type*,
                                 LayerForwardPropagation*,
                                 LayerBackPropagation*) const; //change

   void insert_gradient(LayerBackPropagation*,
                        const Index&,
                        Tensor<type, 1>&) const; // change

   void from_XML(const tinyxml2::XMLDocument&) final;
   void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

   /// This tensor containing conection strengths from a layer's inputs to its neurons.

   Tensor<type, 4> synaptic_weights;

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's transfer function to generate the neuron's output.

   Tensor<type, 1> biases;

   Index row_stride = 1;

   Index column_stride = 1;

   Tensor<Index, 1> inputs_dimensions;

   ConvolutionType convolution_type = ConvolutionType::Valid;

   ActivationFunction activation_function = ActivationFunction::Linear;

   const Eigen::array<ptrdiff_t, 3> convolutions_dimensions = {1, 2, 3};
   const Eigen::array<ptrdiff_t, 3> means_dimensions = {0, 1, 2};

   // Batch normalization

   bool batch_normalization = false;

   Tensor<type, 1> moving_means;
   Tensor<type, 1> moving_standard_deviations;

   type momentum = type(0.9);
   const type epsilon = type(1.0e-5);

   Tensor<type, 1> scales;
   Tensor<type, 1> offsets;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/convolutional_layer_cuda.h"
#else
};
#endif


struct ConvolutionalLayerForwardPropagation : LayerForwardPropagation
{
    // Default constructor

    explicit ConvolutionalLayerForwardPropagation()
        : LayerForwardPropagation()
    {
    }

    // Constructor

    explicit ConvolutionalLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    type* get_activations_derivatives_data()
    {
        return activations_derivatives.data();
    }

    Eigen::array<ptrdiff_t, 4> get_inputs_dimensions_array() const
    {
        ConvolutionalLayer* convolutional_layer_pointer = static_cast<ConvolutionalLayer*>(layer_pointer);

        const Index inputs_rows_number = convolutional_layer_pointer->get_inputs_rows_number();
        const Index inputs_columns_number = convolutional_layer_pointer->get_inputs_columns_number();
        const Index inputs_channels_number = convolutional_layer_pointer->get_inputs_channels_number();

        return Eigen::array<ptrdiff_t, 4>({batch_samples_number,
                                           inputs_rows_number,
                                           inputs_columns_number,
                                           inputs_channels_number});
    }


    Eigen::array<ptrdiff_t, 4> get_outputs_dimensions_array() const
    {
        ConvolutionalLayer* convolutional_layer_pointer = static_cast<ConvolutionalLayer*>(layer_pointer);

        const Index outputs_rows_number = convolutional_layer_pointer->get_outputs_rows_number();
        const Index outputs_columns_number = convolutional_layer_pointer->get_outputs_columns_number();
        const Index kernels_number = convolutional_layer_pointer->get_kernels_number();

        return Eigen::array<ptrdiff_t, 4>({batch_samples_number,
                                           outputs_rows_number,
                                           outputs_columns_number,
                                           kernels_number});
    }


    TensorMap<Tensor<type, 4>> get_outputs() const
    {
        const Eigen::array<ptrdiff_t, 4> outputs_dimensions_array
                = get_outputs_dimensions_array();

        return TensorMap<Tensor<type, 4>>(outputs_data, outputs_dimensions_array);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        layer_pointer = new_layer_pointer;

        const ConvolutionalLayer* convolutional_layer_pointer = static_cast<ConvolutionalLayer*>(layer_pointer);

        const Index inputs_rows_number = convolutional_layer_pointer->get_inputs_rows_number();
        const Index inputs_columns_number = convolutional_layer_pointer->get_inputs_columns_number();

        const Index inputs_channels_number = convolutional_layer_pointer->get_inputs_channels_number();

        const Index kernels_number = convolutional_layer_pointer->get_kernels_number();
        const Index outputs_rows_number = convolutional_layer_pointer->get_outputs_rows_number();
        const Index outputs_columns_number = convolutional_layer_pointer->get_outputs_columns_number();

        preprocessed_inputs.resize(batch_samples_number,
                                   inputs_rows_number,
                                   inputs_columns_number,
                                   inputs_channels_number);

        outputs_data = (type*) malloc(static_cast<size_t>(batch_samples_number*kernels_number*outputs_rows_number*outputs_columns_number*sizeof(type)));

        outputs_dimensions.resize(4);
        outputs_dimensions.setValues({batch_samples_number,
                                      outputs_rows_number,
                                      outputs_columns_number,
                                      kernels_number});

        means.resize(kernels_number);
        standard_deviations.resize(kernels_number);

        activations_derivatives.resize(batch_samples_number,
                                       outputs_rows_number,
                                       outputs_columns_number,
                                       kernels_number);               
    }


    void print() const
    {
        cout << "Convolutional" << endl;

        cout << "Outputs:" << endl;

        cout << TensorMap<Tensor<type,4>>(outputs_data,
                                          outputs_dimensions(0),
                                          outputs_dimensions(1),
                                          outputs_dimensions(2),
                                          outputs_dimensions(3)) << endl;

        cout << "Outputs dimensions:" << endl;
        cout << outputs_dimensions << endl;

        cout << "Activations derivatives:" << endl;
        cout << activations_derivatives << endl;
    }

    Tensor<type, 4> preprocessed_inputs;

    Tensor<type, 1> means;
    Tensor<type, 1> standard_deviations;

    Tensor<type, 4> activations_derivatives;


};


struct ConvolutionalLayerBackPropagation : LayerBackPropagation
{

    explicit ConvolutionalLayerBackPropagation() : LayerBackPropagation()
    {
    }


    virtual ~ConvolutionalLayerBackPropagation()
    {
    }


    explicit ConvolutionalLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    Eigen::array<ptrdiff_t, 4> get_deltas_dimensions_array()
    {
        ConvolutionalLayer* convolutional_layer_pointer = static_cast<ConvolutionalLayer*>(layer_pointer);

        const Index deltas_rows_number = convolutional_layer_pointer->get_outputs_rows_number();
        const Index deltas_columns_number = convolutional_layer_pointer->get_outputs_columns_number();
        const Index kernels_number = convolutional_layer_pointer->get_kernels_number();

        return Eigen::array<ptrdiff_t, 4>({batch_samples_number,
                                           deltas_rows_number,
                                           deltas_columns_number,
                                           kernels_number});
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        layer_pointer = new_layer_pointer;

        const ConvolutionalLayer* convolutional_layer_pointer = static_cast<ConvolutionalLayer*>(layer_pointer);

        const Index kernesl_rows_number = convolutional_layer_pointer->get_kernels_rows_number();
        const Index kernels_columns_number = convolutional_layer_pointer->get_kernels_columns_number();
        const Index kernels_number = convolutional_layer_pointer->get_kernels_number();
        const Index kernels_channels_number = convolutional_layer_pointer->get_kernels_channels_number();

        const Index outputs_rows_number = convolutional_layer_pointer->get_outputs_rows_number();
        const Index outputs_columns_number = convolutional_layer_pointer->get_outputs_columns_number();
        const Index synaptic_weights_number = convolutional_layer_pointer->get_synaptic_weights_number();

        deltas_dimensions.resize(4);

        deltas_dimensions.setValues({batch_samples_number,
                                     outputs_rows_number,
                                     outputs_columns_number,
                                     kernels_number});

        deltas_data = (type*)malloc(static_cast<size_t>(batch_samples_number
                                                        *outputs_rows_number
                                                        *outputs_columns_number
                                                        *kernels_number*sizeof(type)));

        deltas_times_activations_derivatives.resize(batch_samples_number,
                                                    outputs_rows_number,
                                                    outputs_columns_number,
                                                    kernels_number);

        biases_derivatives.resize(kernels_number);

        synaptic_weights_derivatives.resize(kernesl_rows_number,
                                            kernels_columns_number,
                                            kernels_channels_number,
                                            kernels_number);
    }

    void print() const
    {
        cout << "Deltas:" << endl;
        //cout << deltas << endl;ca

        cout << "Biases derivatives:" << endl;
        cout << biases_derivatives << endl;

        cout << "Synaptic weights derivatives:" << endl;
        cout << synaptic_weights_derivatives << endl;

    }

    Tensor<type, 4> deltas_times_activations_derivatives;

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 4> synaptic_weights_derivatives;
};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/struct_convolutional_layer_cuda.h"
#endif

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
