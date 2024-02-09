//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef RECURRENTLAYER_H
#define RECURRENTLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "tensor_utilities.h"
#include "layer.h"

#include "probabilistic_layer.h"
#include "perceptron_layer.h"

namespace opennn
{

struct RecurrentLayerForwardPropagation;
struct RecurrentLayerBackPropagation;


#ifdef OPENNN_CUDA
        #include "../../opennn-cuda/opennn-cuda/struct_recurrent_layer_cuda.h"
#endif

/// This class represents a layer of neurons.
/// Layers of neurons will be used to construct multilayer neurons.

class RecurrentLayer : public Layer
{

public:

    /// Enumeration of the available activation functions for the recurrent layer.

    enum class ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent,
                            Linear, RectifiedLinear, ExponentialLinear,
                            ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

   // Constructors

   explicit RecurrentLayer();

   explicit RecurrentLayer(const Index&, const Index&);

   // Get methods

   bool is_empty() const;

   Index get_inputs_number() const final;
   Index get_neurons_number() const final;

   const Tensor<type, 1>& get_hidden_states() const;

   // Parameters

   Index get_timesteps() const;

   Tensor<type, 1> get_biases() const;
   const Tensor<type, 2>& get_input_weights() const;
   const Tensor<type, 2>& get_recurrent_weights() const;

   Index get_biases_number() const;
   Index get_input_weights_number() const;
   Index get_recurrent_weights_number() const;

   Index get_parameters_number() const final;
   Tensor<type, 1> get_parameters() const final;

   Tensor<type, 2> get_biases(const Tensor<type, 1>&) const;
   Tensor<type, 2> get_input_weights(const Tensor<type, 1>&) const;
   Tensor<type, 2> get_recurrent_weights(const Tensor<type, 1>&) const;

   // Activation functions

   const RecurrentLayer::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&);
   void set(const RecurrentLayer&);

   void set_default();

   // Architecture

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;
   void set_input_shape(const Tensor<Index, 1>&);

   // Parameters

   void set_timesteps(const Index&);

   void set_biases(const Tensor<type, 1>&);

   void set_input_weights(const Tensor<type, 2>&);

   void set_recurrent_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& = 0) final;

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods

   void set_hidden_states_constant(const type&);

   void set_biases_constant(const type&);

   void set_input_weights_constant(const type&);
   void set_recurrent_weights_constant(const type&);

   void set_input_weights_random();
   void set_recurrent_weights_random();

   void set_parameters_constant(const type&) final;

   void set_parameters_random() final;

   // Forward propagation

   void calculate_combinations(const Tensor<type, 1>&,
                               const Tensor<type, 2>&,
                               const Tensor<type, 2>&,
                               const Tensor<type, 1>&,
                               Tensor<type, 1>&) const;

   void calculate_activations(const Tensor<type, 1>&,
                              Tensor<type, 1>&) const;

   void calculate_activations_derivatives(const Tensor<type, 1>&,
                                          Tensor<type, 1>&,
                                          Tensor<type, 1>&);

   void forward_propagate(const pair<type*, dimensions>&,
                          LayerForwardPropagation*,
                          const bool&) final;

   void forward_propagate(const pair<type*, dimensions>&,
                          Tensor<type, 1>&,
                          LayerForwardPropagation*) final;

   // Back propagation

   void calculate_hidden_delta(LayerForwardPropagation*,
                               LayerBackPropagation*,
                               LayerBackPropagation*) const final;

   void calculate_hidden_delta(PerceptronLayerForwardPropagation*,
                               PerceptronLayerBackPropagation*,
                               RecurrentLayerBackPropagation*) const;

   void calculate_hidden_delta(ProbabilisticLayerForwardPropagation*,
                               ProbabilisticLayerBackPropagation*,
                               RecurrentLayerBackPropagation*) const;

   void insert_gradient(LayerBackPropagation*,
                        const Index& ,
                        Tensor<type, 1>&) const final;

   void calculate_error_gradient(const pair<type*, dimensions>&,
                                 LayerForwardPropagation*,
                                 LayerBackPropagation*) const final;

   void calculate_biases_error_gradient(const Tensor<type, 2>&,
                                        RecurrentLayerForwardPropagation*,
                                        RecurrentLayerBackPropagation*) const;

   void calculate_input_weights_error_gradient(const Tensor<type, 2>&,
                                               RecurrentLayerForwardPropagation*,
                                               RecurrentLayerBackPropagation*) const;

   void calculate_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                   RecurrentLayerForwardPropagation*,
                                                   RecurrentLayerBackPropagation*) const;

   // Expression methods

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

   string write_activation_function_expression() const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

   Index timesteps = 1;

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Tensor<type, 1> biases;

   Tensor<type, 2> input_weights;

   /// This matrix contains conection strengths from a recurrent layer inputs to its neurons.

   Tensor<type, 2> recurrent_weights;

   /// Activation function variable.

   ActivationFunction activation_function = ActivationFunction::HyperbolicTangent;

   Tensor<type, 1> hidden_states;

   /// Display messages to screen.

   bool display = true;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/recurrent_layer_cuda.h"
#else
};
#endif

struct RecurrentLayerForwardPropagation : LayerForwardPropagation
{
    explicit RecurrentLayerForwardPropagation() : LayerForwardPropagation()
    {
    }


    explicit RecurrentLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer) : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }
    
    
    pair<type*, dimensions> get_outputs_pair() const final
    {
        return pair<type*, dimensions>();
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer) final
    {
        layer_pointer = new_layer_pointer;

        const Index neurons_number = layer_pointer->get_neurons_number();
        const Index inputs_number = layer_pointer->get_inputs_number();

        batch_samples_number = new_batch_samples_number;

        outputs.resize(batch_samples_number, neurons_number);

        previous_activations.resize(neurons_number);

        current_inputs.resize(inputs_number);
        current_combinations.resize(neurons_number);
        current_activations_derivatives.resize(neurons_number);

        activations_derivatives.resize(batch_samples_number, neurons_number);
    }

    void print() const
    {
    }

    Tensor<type, 2> outputs;

    Tensor<type, 1> previous_activations;

    Tensor<type, 1> current_inputs;
    Tensor<type, 1> current_combinations;
    Tensor<type, 1> current_activations_derivatives;

    Tensor<type, 2> activations_derivatives;
};


struct RecurrentLayerBackPropagation : LayerBackPropagation
{
    explicit RecurrentLayerBackPropagation() : LayerBackPropagation()
    {
    }

    virtual ~RecurrentLayerBackPropagation()
    {
    }

    explicit RecurrentLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer) final
    {
        layer_pointer = new_layer_pointer;

        batch_samples_number = new_batch_samples_number;

        const Index neurons_number = layer_pointer->get_neurons_number();
        const Index inputs_number = layer_pointer->get_inputs_number();

        deltas.resize(batch_samples_number, neurons_number);

        current_deltas.resize(neurons_number);

        biases_derivatives.resize(neurons_number);

        input_weights_derivatives.resize(inputs_number*neurons_number);

        recurrent_weights_derivatives.resize(neurons_number*neurons_number);

        combinations_biases_derivatives.resize(neurons_number, neurons_number);

        combinations_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);

        combinations_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);

        error_current_combinations_derivatives.resize(neurons_number);
    }


    void print() const
    {

    }

    Tensor<type, 2> deltas;

    Tensor<type, 1> current_deltas;

    Tensor<type, 1> biases_derivatives;

    Tensor<type, 1> input_weights_derivatives;

    Tensor<type, 1> recurrent_weights_derivatives;

    Tensor<type, 2> combinations_biases_derivatives;
    Tensor<type, 2> combinations_weights_derivatives;
    Tensor<type, 2> combinations_recurrent_weights_derivatives;

    Tensor<type, 1> error_current_combinations_derivatives;
};




}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
