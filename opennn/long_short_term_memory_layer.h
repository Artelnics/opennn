//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LONGSHORTTERMMEMORYLAYER_H
#define LONGSHORTTERMMEMORYLAYER_H

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

namespace OpenNN
{

struct LongShortTermMemoryLayerForwardPropagation;
struct LongShortTermMemoryLayerBackPropagation;


/// This class represents a layer of neurons.
/// Layers of neurons will be used to construct multilayer neurons.

class LongShortTermMemoryLayer : public Layer
{

public:

    /// Enumeration of available activation functions for the long-short term memory layer.

    enum ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent,
                            Linear, RectifiedLinear, ExponentialLinear, ScaledExponentialLinear,
                            SoftPlus, SoftSign, HardSigmoid};

   // Constructors

   explicit LongShortTermMemoryLayer();

   explicit LongShortTermMemoryLayer(const Index&, const Index&);

   // Destructor

   virtual ~LongShortTermMemoryLayer();

   // Get methods

   bool is_empty() const;

   Index get_inputs_number() const;
   Index get_neurons_number() const;

   // Parameters

   Tensor<type, 1> get_input_biases() const;
   Tensor<type, 1> get_forget_biases() const;
   Tensor<type, 1> get_state_biases() const;
   Tensor<type, 1> get_output_biases() const;

   Tensor<type, 2> get_input_weights() const;
   Tensor<type, 2> get_forget_weights() const;
   Tensor<type, 2> get_state_weights() const;
   Tensor<type, 2> get_output_weights() const;

   Tensor<type, 2> get_input_recurrent_weights() const;
   Tensor<type, 2> get_forget_recurrent_weights() const;
   Tensor<type, 2> get_state_recurrent_weights() const;
   Tensor<type, 2> get_output_recurrent_weights() const;

   Index get_timesteps() const;

   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   // Activation functions

   const LongShortTermMemoryLayer::ActivationFunction& get_activation_function() const;
   const LongShortTermMemoryLayer::ActivationFunction& get_recurrent_activation_function() const;

   string write_activation_function() const;
   string write_recurrent_activation_function() const;
   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&);
   void set(const LongShortTermMemoryLayer&);

   void set_default();
   void set_name(const string&);

   // Architecture

   void set_inputs_number(const Index&);
   void set_neurons_number(const Index&);
   void set_input_shape(const Tensor<Index, 1>&);

   // Parameters

   void set_input_biases(const Tensor<type, 1>&);
   void set_forget_biases(const Tensor<type, 1>&);
   void set_state_biases(const Tensor<type, 1>&);
   void set_output_biases(const Tensor<type, 1>&);

   void set_input_weights(const Tensor<type, 2>&);
   void set_forget_weights(const Tensor<type, 2>&);
   void set_state_weights(const Tensor<type, 2>&);
   void set_output_weights(const Tensor<type, 2>&);

   void set_input_recurrent_weights(const Tensor<type, 2>&);
   void set_forget_recurrent_weights(const Tensor<type, 2>&);
   void set_state_recurrent_weights(const Tensor<type, 2>&);
   void set_output_recurrent_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& = 0);

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   void set_recurrent_activation_function(const ActivationFunction&);
   void set_recurrent_activation_function(const string&);

   void set_timesteps(const Index&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods

   void set_biases_constant(const type&);

   void set_forget_biases_constant(const type&);
   void set_input_biases_constant(const type&);
   void set_state_biases_constant(const type&);
   void set_output_biases_constant(const type&);

   void set_weights_constant(const type&);

   void set_forget_weights_constant(const type&);
   void set_input_weights_constant(const type&);
   void set_state_weights_constant(const type&);
   void set_output_weights_constant(const type&);

   void set_recurrent_weights_constant(const type&);

   void set_forget_recurrent_weights_constant(const type&);
   void set_input_recurrent_weights_constant(const type&);
   void set_state_recurrent_weights_constant(const type&);
   void set_output_recurrent_weights_constant(const type&);

   void set_hidden_states_constant(const type&);
   void set_cell_states_constant(const type&);

   void set_parameters_constant(const type&);

   void set_parameters_random();

   // Long short term memory layer combinations

   void calculate_combinations(const Tensor<type, 1>&,
                               const Tensor<type, 2>&,
                               const Tensor<type, 2>&,
                               const Tensor<type, 1>&,
                               Tensor<type, 1>&) const;

   // Long short term memory layer activations

   void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const;
   void calculate_activations(const Tensor<type, 1>&, Tensor<type, 1>&) const;
   Tensor<type, 1> calculate_activations(const Tensor<type, 1>&) const;
   void calculate_recurrent_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const;
   void calculate_recurrent_activations(const Tensor<type, 1>&, Tensor<type, 1>&) const;

   // Long short term memory layer derivatives

   void calculate_activations_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&, Tensor<type, 2>&) const;
   void calculate_activations_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
   void calculate_recurrent_activations_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;

   // Long short term memory layer outputs

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

   void calculate_hidden_delta(LayerForwardPropagation*,
                               LayerBackPropagation*,
                               LayerBackPropagation*) const;


   void calculate_hidden_delta_perceptron(PerceptronLayerForwardPropagation*,
                                          PerceptronLayerBackPropagation*,
                                          LongShortTermMemoryLayerBackPropagation*) const;


   void calculate_hidden_delta_probabilistic(ProbabilisticLayerForwardPropagation*,
                                             ProbabilisticLayerBackPropagation*,
                                             LongShortTermMemoryLayerBackPropagation*) const;

   // Forward propagate

   void forward_propagate(const Tensor<type, 2>&, LayerForwardPropagation*);

   void forward_propagate(const Tensor<type, 2>&, Tensor<type, 1>, LayerForwardPropagation*);

   // Eror gradient

   void insert_gradient(LayerBackPropagation*, const Index& , Tensor<type, 1>&) const;

   void calculate_error_gradient(const Tensor<type, 2>&, LayerForwardPropagation*, LayerBackPropagation*) const;

   void calculate_forget_weights_error_gradient(const Tensor<type, 2>&,
                                                LongShortTermMemoryLayerForwardPropagation*,
                                                LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_input_weights_error_gradient(const Tensor<type, 2>&,
                                               LongShortTermMemoryLayerForwardPropagation*,
                                               LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_state_weights_error_gradient(const Tensor<type, 2>&,
                                               LongShortTermMemoryLayerForwardPropagation*,
                                               LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_output_weights_error_gradient(const Tensor<type, 2>&,
                                                LongShortTermMemoryLayerForwardPropagation*,
                                                LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_forget_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                          LongShortTermMemoryLayerForwardPropagation*,
                                                          LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_input_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                         LongShortTermMemoryLayerForwardPropagation*,
                                                         LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_state_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                         LongShortTermMemoryLayerForwardPropagation*,
                                                         LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_output_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                          LongShortTermMemoryLayerForwardPropagation*,
                                                          LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_forget_biases_error_gradient(const Tensor<type, 2>&,
                                               LongShortTermMemoryLayerForwardPropagation*,
                                               LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_input_biases_error_gradient(const Tensor<type, 2>&,
                                              LongShortTermMemoryLayerForwardPropagation*,
                                              LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_state_biases_error_gradient(const Tensor<type, 2>&,
                                              LongShortTermMemoryLayerForwardPropagation*,
                                              LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_output_biases_error_gradient(const Tensor<type, 2>&,
                                               LongShortTermMemoryLayerForwardPropagation*,
                                               LongShortTermMemoryLayerBackPropagation*) const;

   // Expression methods

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_recurrent_activation_function_expression() const;
   string write_activation_function_expression() const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

protected:

   Index timesteps = 3;

   Tensor<type, 1> input_biases;
   Tensor<type, 1> forget_biases;
   Tensor<type, 1> state_biases;
   Tensor<type, 1> output_biases;

   Tensor<type, 2> input_weights;
   Tensor<type, 2> forget_weights;
   Tensor<type, 2> state_weights;
   Tensor<type, 2> output_weights;

   Tensor<type, 2> forget_recurrent_weights;
   Tensor<type, 2> input_recurrent_weights;
   Tensor<type, 2> state_recurrent_weights;
   Tensor<type, 2> output_recurrent_weights;

   /// Activation function variable.

   ActivationFunction activation_function = HyperbolicTangent;
   ActivationFunction recurrent_activation_function = Linear;

   Index batch;
   Index variables;

   Tensor<type, 1> hidden_states;
   Tensor<type, 1> cell_states;

   /// Display messages to screen.

   bool display = true;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/long_short_term_memory_layer_cuda.h"
#endif

};


struct LongShortTermMemoryLayerForwardPropagation : LayerForwardPropagation
{
    explicit LongShortTermMemoryLayerForwardPropagation() : LayerForwardPropagation()
    {
    }

    explicit LongShortTermMemoryLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        layer_pointer = new_layer_pointer;

        const Index inputs_number = layer_pointer->get_inputs_number();
        const Index neurons_number = layer_pointer->get_neurons_number();

        batch_samples_number = new_batch_samples_number;

        previous_hidden_state_activations.resize(neurons_number);
        previous_cell_state_activations.resize(neurons_number);

        current_inputs.resize(inputs_number);

        current_forget_combinations.resize(neurons_number);
        current_input_combinations.resize(neurons_number);
        current_state_combinations.resize(neurons_number);
        current_output_combinations.resize(neurons_number);

        current_forget_activations.resize(neurons_number);
        current_input_activations.resize(neurons_number);
        current_state_activations.resize(neurons_number);
        current_output_activations.resize(neurons_number);

        current_cell_state_activations.resize(neurons_number);

        current_forget_activations_derivatives.resize(neurons_number);
        current_input_activations_derivatives.resize(neurons_number);
        current_state_activations_derivatives.resize(neurons_number);
        current_output_activations_derivatives.resize(neurons_number);
        current_hidden_states_derivatives.resize(neurons_number);

        forget_activations.resize(batch_samples_number, neurons_number);
        input_activations.resize(batch_samples_number, neurons_number);
        state_activations.resize(batch_samples_number, neurons_number);
        output_activations.resize(batch_samples_number, neurons_number);
        cell_states_activations.resize(batch_samples_number, neurons_number);
        hidden_states_activations.resize(batch_samples_number, neurons_number);

        forget_activations_derivatives.resize(batch_samples_number, neurons_number);
        input_activations_derivatives.resize(batch_samples_number, neurons_number);
        state_activations_derivatives.resize(batch_samples_number, neurons_number);
        output_activations_derivatives.resize(batch_samples_number, neurons_number);
        cell_states_activations_derivatives.resize(batch_samples_number, neurons_number);
        hidden_states_activations_derivatives.resize(batch_samples_number, neurons_number);

        combinations.resize(batch_samples_number, neurons_number);
        activations.resize(batch_samples_number, neurons_number);
    }

    void print() const
    {

    }

    Tensor<type, 2> combinations;
    Tensor<type, 2> activations;

    Tensor<type, 1> previous_hidden_state_activations;
    Tensor<type, 1> previous_cell_state_activations;

    Tensor<type, 1> current_inputs;

    Tensor<type, 1> current_forget_combinations;
    Tensor<type, 1> current_input_combinations;
    Tensor<type, 1> current_state_combinations;
    Tensor<type, 1> current_output_combinations;

    Tensor<type, 1> current_forget_activations;
    Tensor<type, 1> current_input_activations;
    Tensor<type, 1> current_state_activations;
    Tensor<type, 1> current_output_activations;

    Tensor<type, 1> current_forget_activations_derivatives;
    Tensor<type, 1> current_input_activations_derivatives;
    Tensor<type, 1> current_state_activations_derivatives;
    Tensor<type, 1> current_output_activations_derivatives;

    Tensor<type, 1> current_hidden_states_derivatives;

    Tensor<type, 1> current_cell_state_activations;

    Tensor<type, 2, RowMajor> forget_activations;
    Tensor<type, 2, RowMajor> input_activations;
    Tensor<type, 2, RowMajor> state_activations;
    Tensor<type, 2, RowMajor> output_activations;
    Tensor<type, 2, RowMajor> cell_states_activations;
    Tensor<type, 2, RowMajor> hidden_states_activations;

    Tensor<type, 2, RowMajor> forget_activations_derivatives;
    Tensor<type, 2, RowMajor> input_activations_derivatives;
    Tensor<type, 2, RowMajor> state_activations_derivatives;
    Tensor<type, 2, RowMajor> output_activations_derivatives;
    Tensor<type, 2, RowMajor> cell_states_activations_derivatives;
    Tensor<type, 2, RowMajor> hidden_states_activations_derivatives;
};


struct LongShortTermMemoryLayerBackPropagation : LayerBackPropagation
{
    explicit LongShortTermMemoryLayerBackPropagation() : LayerBackPropagation()
    {
    }


    explicit LongShortTermMemoryLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        layer_pointer = new_layer_pointer;
        batch_samples_number = new_batch_samples_number;

        const Index neurons_number = layer_pointer->get_neurons_number();
        const Index inputs_number = layer_pointer->get_inputs_number();

        current_layer_deltas.resize(neurons_number);

        forget_weights_derivatives.resize(inputs_number*neurons_number);
        input_weights_derivatives.resize(inputs_number*neurons_number);
        state_weights_derivatives.resize(inputs_number*neurons_number);
        output_weights_derivatives.resize(inputs_number*neurons_number);

        forget_recurrent_weights_derivatives.resize(neurons_number*neurons_number);
        input_recurrent_weights_derivatives.resize(neurons_number*neurons_number);
        state_recurrent_weights_derivatives.resize(neurons_number*neurons_number);
        output_recurrent_weights_derivatives.resize(neurons_number*neurons_number);

        forget_biases_derivatives.resize(neurons_number);
        input_biases_derivatives.resize(neurons_number);
        state_biases_derivatives.resize(neurons_number);
        output_biases_derivatives.resize(neurons_number);

        delta.resize(batch_samples_number, neurons_number);

        input_combinations_biases_derivatives.resize(neurons_number, neurons_number);
        forget_combinations_biases_derivatives.resize(neurons_number, neurons_number);
        state_combinations_biases_derivatives.resize(neurons_number, neurons_number);
        output_combinations_biases_derivatives.resize(neurons_number, neurons_number);

        hidden_states_biases_derivatives.resize(neurons_number, neurons_number);
        cell_state_biases_derivatives.resize(neurons_number, neurons_number);

        input_combinations_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);
        forget_combinations_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);
        state_combinations_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);
        output_combinations_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);

        hidden_states_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);
        cell_state_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);

        input_combinations_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);
        forget_combinations_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);
        state_combinations_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);
        output_combinations_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);

        hidden_states_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);
        cell_state_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);
    }

    void print() const
    {

    }

    Tensor<type, 1> current_layer_deltas;

    Tensor<type, 2> delta;

    Tensor<type, 1> forget_weights_derivatives;
    Tensor<type, 1> input_weights_derivatives;
    Tensor<type, 1> state_weights_derivatives;
    Tensor<type, 1> output_weights_derivatives;

    Tensor<type, 1> forget_recurrent_weights_derivatives;
    Tensor<type, 1> input_recurrent_weights_derivatives;
    Tensor<type, 1> state_recurrent_weights_derivatives;
    Tensor<type, 1> output_recurrent_weights_derivatives;

    Tensor<type, 1> forget_biases_derivatives;
    Tensor<type, 1> input_biases_derivatives;
    Tensor<type, 1> state_biases_derivatives;
    Tensor<type, 1> output_biases_derivatives;

    Tensor<type, 2> input_combinations_biases_derivatives;
    Tensor<type, 2> forget_combinations_biases_derivatives;
    Tensor<type, 2> state_combinations_biases_derivatives;
    Tensor<type, 2> output_combinations_biases_derivatives;

    Tensor<type, 2> hidden_states_biases_derivatives;
    Tensor<type, 2> cell_state_biases_derivatives;

    Tensor<type, 2> input_combinations_weights_derivatives;
    Tensor<type, 2> forget_combinations_weights_derivatives;
    Tensor<type, 2> state_combinations_weights_derivatives;
    Tensor<type, 2> output_combinations_weights_derivatives;

    Tensor<type, 2> hidden_states_weights_derivatives;
    Tensor<type, 2> cell_state_weights_derivatives;

    Tensor<type, 2> input_combinations_recurrent_weights_derivatives;
    Tensor<type, 2> forget_combinations_recurrent_weights_derivatives;
    Tensor<type, 2> state_combinations_recurrent_weights_derivatives;
    Tensor<type, 2> output_combinations_recurrent_weights_derivatives;

    Tensor<type, 2> hidden_states_recurrent_weights_derivatives;
    Tensor<type, 2> cell_state_recurrent_weights_derivatives;
};


}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
