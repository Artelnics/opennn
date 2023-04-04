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
#include "perceptron_layer.h"
#include "probabilistic_layer.h"
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

    enum class ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear, ExponentialLinear,
                                  ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

    enum class ConvolutionType{Valid, Same};

    // Constructors

    explicit ConvolutionalLayer();

    explicit ConvolutionalLayer(const Index&, const Index&, const ActivationFunction& = ConvolutionalLayer::ActivationFunction::Linear);

    explicit ConvolutionalLayer(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    // Destructor

    // Get methods

    bool is_empty() const;

    const Tensor<type, 1>& get_biases() const;

    const Tensor<type, 4>& get_synaptic_weights() const;

    Index get_biases_number() const;

    Index get_synaptic_weights_number() const;

    ActivationFunction get_activation_function() const;

    string write_activation_function() const;

    Tensor<Index, 1> get_outputs_dimensions() const;

    Tensor<Index, 1> get_input_variables_dimensions() const;

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

    Index get_inputs_images_number() const;
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

    void set_name(const string&);

    void set_activation_function(const ActivationFunction&);
    void set_activation_function(const string&);

    void set_biases(const Tensor<type, 1>&);

    void set_synaptic_weights(const Tensor<type, 4>&);

    void set_convolution_type(const ConvolutionType&);
    void set_convolution_type(const string&);

    void set_parameters(const Tensor<type, 1>&, const Index& index = 0);

    void set_row_stride(const Index&);

    void set_column_stride(const Index&);

    void set_input_variables_dimenisons(const Tensor<Index,1>&);

    // Initialization

    void set_biases_constant(const type&);

    void set_synaptic_weights_constant(const type&);

    void set_parameters_constant(const type&);

    void set_parameters_random();

    // Padding

    void insert_padding(const Tensor<type, 4>&, Tensor<type, 4>&);

    // Combinations

    void calculate_convolutions(const Tensor<type, 4>&, type*) const; //change

    void calculate_convolutions(const Tensor<type, 4>&,
                                const Tensor<type, 2>&,
                                const Tensor<type, 4>&,
                                Tensor<type, 4>&) const; //change
    // Activation

    void calculate_activations(type*, const Tensor<Index, 1>&,
                               type*, const Tensor<Index, 1>&) const;

    void calculate_activations_derivatives(type*, const Tensor<Index, 1>&,
                                           type*, const Tensor<Index, 1>&,
                                           type*, const Tensor<Index, 1>&) const;


   // Outputs

//   void forward_propagate(const Tensor<type, 4>&, LayerForwardPropagation*); //change
//   void forward_propagate(const Tensor<type, 4>&, Tensor<type,1>, LayerForwardPropagation*); //change
    void forward_propagate(type*, const Tensor<Index, 1>&, LayerForwardPropagation*, bool&) final; // --> New


//   void forward_propagate(const Tensor<type, 2>&, LayerForwardPropagation*);
//   void forward_propagate(const Tensor<type, 4>&, Tensor<type, 1>, LayerForwardPropagation*);
//   void forward_propagate(const Tensor<type, 2>&, Tensor<type, 1>, LayerForwardPropagation*);

   // Outputs


   // Delta methods

   void calculate_hidden_delta(LayerForwardPropagation*,
                               LayerBackPropagation*,
                               LayerBackPropagation*) const final;

   void calculate_hidden_delta(PerceptronLayerForwardPropagation*,
                               PerceptronLayerBackPropagation*,
                               ConvolutionalLayerBackPropagation*) const;

   // @todo probabilistic hidden delta

   // Gradient methods

//   void calculate_error_gradient(const Tensor<type, 2>&,
   void calculate_error_gradient(const Tensor<type, 4>&,
                                 LayerForwardPropagation*,
                                 LayerBackPropagation*) const; //change

   void calculate_error_gradient(type*,
                                 LayerForwardPropagation*,
                                 LayerBackPropagation*) const; //change

   void insert_gradient(LayerBackPropagation*,
                        const Index&,
                        Tensor<type, 1>&) const; // change

   void to_2d(const Tensor<type, 4>&, Tensor<type, 2>&) const;

   void from_XML(const tinyxml2::XMLDocument&) final;
   void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

   /// This tensor containing conection strengths from a layer's inputs to its neurons.

   Tensor<type, 4> synaptic_weights;

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Tensor<type, 1> biases;

   Index row_stride = 1;

   Index column_stride = 1;

   //0->rows, 1->cols, 2->channels, 3->batch

   Tensor<Index, 1> input_variables_dimensions;

   ConvolutionType convolution_type = ConvolutionType::Valid;

   ActivationFunction activation_function = ActivationFunction::Linear;

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


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        layer_pointer = new_layer_pointer;

        const Index kernels_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_kernels_number();
        const Index outputs_rows_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_outputs_rows_number();
        const Index outputs_columns_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_outputs_columns_number();

        batch_samples_number = new_batch_samples_number;

        combinations.resize(outputs_rows_number, outputs_columns_number, kernels_number, batch_samples_number);
        outputs.resize(outputs_rows_number, outputs_columns_number, kernels_number, batch_samples_number);
        activations_derivatives.resize(outputs_rows_number, outputs_columns_number, kernels_number, batch_samples_number);

        combinations.setZero();
        outputs.setZero();
        activations_derivatives.setZero();

        outputs_data = outputs.data();
        outputs_dimensions = get_dimensions(outputs);
    }

    void print() const
    {
        cout << "Combinations:" << endl;
        cout << combinations << endl;

        cout << "Outputs:" << endl;
        cout << outputs << endl;

        cout << "Activations derivatives:" << endl;
        cout << activations_derivatives << endl;
    }

    type* get_combinations_data()
    {
        return combinations.data();
    }

    type* get_activations_derivatives_data()
    {
        return activations_derivatives.data();
    }

    Tensor<type, 4> combinations;
    Tensor<type, 4> outputs;
    Tensor<type, 4> activations_derivatives;
};


struct ConvolutionalLayerBackPropagation : LayerBackPropagation
{

    explicit ConvolutionalLayerBackPropagation() : LayerBackPropagation()
    {
    }


    explicit ConvolutionalLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        layer_pointer = new_layer_pointer;

        batch_samples_number = new_batch_samples_number;

        const Index kernels_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_kernels_number();
        const Index outputs_rows_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_outputs_rows_number();
        const Index outputs_columns_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_outputs_columns_number();
        const Index synaptic_weights_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_synaptic_weights_number();

//        delta.resize(batch_samples_number, kernels_number*outputs_rows_number*outputs_columns_number); --> old
        deltas_dimensions.resize(2);
        deltas_dimensions.setValues({batch_samples_number, kernels_number*outputs_rows_number*outputs_columns_number});

        //delete deltas_data;
        deltas_data = (type*)malloc(static_cast<size_t>(batch_samples_number*kernels_number*outputs_rows_number*outputs_columns_number*sizeof(type)));

        convolutional_delta.resize(outputs_rows_number, outputs_columns_number, kernels_number, batch_samples_number);

        biases_derivatives.resize(kernels_number);

        synaptic_weights_derivatives.resize(kernels_number+synaptic_weights_number);
    }


    void print() const
    {
        cout << "Deltas:" << endl;
        //cout << deltas << endl;

//        cout << "Biases derivatives:" << endl;
//        cout << biases_derivatives << endl;

//        cout << "Synaptic weights derivatives:" << endl;
//        cout << synaptic_weights_derivatives << endl;

    }

    Tensor<type, 2> delta; // --> delete?
    Tensor<type, 4> convolutional_delta; // --> delete?

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 1> synaptic_weights_derivatives;
};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/struct_convolutional_layer_cuda.h"
#endif

}


#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
