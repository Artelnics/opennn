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
#include "layer.h"

#include "perceptron_layer.h"

namespace OpenNN
{

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

   void set_parameters(const Tensor<type, 1>&);
   void set_parameters(const Tensor<type, 1>&, const Index&);

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

   void initialize_forget_biases(const type&);
   void initialize_input_biases(const type&);
   void initialize_state_biases(const type&);
   void initialize_output_biases(const type&);

   void initialize_weights(const type&);

   void initialize_forget_weights(const type&);
   void initialize_input_weights(const type&);
   void initialize_state_weights(const type&);
   void initialize_output_weights(const type&);

   void initialize_recurrent_weights(const type&);

   void initialize_forget_recurrent_weights(const type&);
   void initialize_input_recurrent_weights(const type&);
   void initialize_state_recurrent_weights(const type&);
   void initialize_output_recurrent_weights(const type&);

   void initialize_hidden_states(const type&);
   void initialize_cell_states(const type&);

   void set_synaptic_weights_glorot(const type&, const type&);

   void set_parameters_constant(const type&);

   void set_parameters_random();

   // Long short term memory layer combinations_2d

   void calculate_combinations(const Tensor<type, 1>& ,
                               const Tensor<type, 2>& ,
                               const Tensor<type, 2>& ,
                               const Tensor<type, 1>& ,
                               Tensor<type, 1>&) const;

   void calculate_forget_combinations(const Tensor<type, 1>& ,
                                      const Tensor<type, 2>& ,
                                      const Tensor<type, 2>& ,
                                      const Tensor<type, 1>& ,
                                      Tensor<type, 1>&) const;

   void calculate_input_combinations(const Tensor<type, 1>& ,
                                     const Tensor<type, 2>& ,
                                     const Tensor<type, 2>& ,
                                     const Tensor<type, 1>& ,
                                     Tensor<type, 1>&) const;

   void calculate_state_combinations(const Tensor<type, 1>& ,
                                     const Tensor<type, 2>& ,
                                     const Tensor<type, 2>& ,
                                     const Tensor<type, 1>& ,
                                     Tensor<type, 1>&) const;

   void calculate_output_combinations(const Tensor<type, 1>& ,
                                      const Tensor<type, 2>& ,
                                      const Tensor<type, 2>& ,
                                      const Tensor<type, 1>& ,
                                      Tensor<type, 1>&) const;

   Tensor<type, 3> calculate_activations_states(const Tensor<type, 2>&);

   // Long short term memory layer activations_2d

   void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const;
   void calculate_activations(const Tensor<type, 1>&, Tensor<type, 1>&) const;
   Tensor<type, 1> calculate_activations(const Tensor<type, 1>&) const;
   void calculate_recurrent_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const;
   void calculate_recurrent_activations(const Tensor<type, 1>&, Tensor<type, 1>&) const;

   // Long short term memory layer derivatives

   void calculate_activations_derivatives(const Tensor<type, 2>&, Tensor<type,2>&, Tensor<type, 2>&) const;
   void calculate_activations_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;
   void calculate_recurrent_activations_derivatives(const Tensor<type, 1>&, Tensor<type, 1>&, Tensor<type, 1>&) const;

   // Long short term memory layer outputs

   void update_cell_states(const Tensor<type, 1>&);
   void update_hidden_states(const Tensor<type, 1>&);

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

   Tensor<type, 2> calculate_hidden_delta(Layer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   // Long short term memory layer forward_propagate

   void forward_propagate(const Tensor<type, 2>& inputs, ForwardPropagation& forward_propagation);
/*   {
       const Index samples_number = inputs.dimension(0);
       const Index neurons_number = get_neurons_number();

//       Tensor<type, 2> activations_2d(samples_number,neurons_number);

       // forget, input, state, output and tanh(cell_states) derivatives
//       Tensor<type, 3> activations_derivatives(samples_number,neurons_number, 5);
//       activations_derivatives.setZero();

       Index forget_activations_index = 0;
       Index input_activations_index = samples_number*neurons_number;
       Index state_activations_index = 2*samples_number*neurons_number;
       Index output_activations_index = 3*samples_number*neurons_number;
       Index hidden_states_index = 4*samples_number*neurons_number;

       Tensor<type, 1> forget_combinations(neurons_number);
       Tensor<type, 1> forget_activations(neurons_number);
       Tensor<type, 1> forget_activations_derivatives(neurons_number);

       Tensor<type, 1> input_combinations(neurons_number);
       Tensor<type, 1> input_activations(neurons_number);
       Tensor<type, 1> input_activations_derivatives(neurons_number);

       Tensor<type, 1> state_combinations(neurons_number);
       Tensor<type, 1> state_activations(neurons_number);
       Tensor<type, 1> state_activations_derivatives(neurons_number);

       Tensor<type, 1> output_combinations(neurons_number);
       Tensor<type, 1> output_activations(neurons_number);
       Tensor<type, 1> output_activations_derivatives(neurons_number);

       Tensor<type, 1> hidden_states_derivatives(neurons_number);

       for(Index i = 0; i < samples_number; i++)
       {
           if(i%timesteps == 0)
           {
               hidden_states.setZero();
               cell_states.setZero();
           }

           const Tensor<type, 1> current_inputs = inputs.chip(i, 0);

    #pragma omp parallel
           {

               calculate_forget_combinations(current_inputs, forget_weights, forget_recurrent_weights, forget_biases, forget_combinations);
               calculate_recurrent_activations_derivatives(forget_combinations, forget_activations, forget_activations_derivatives);

               calculate_input_combinations(current_inputs, input_weights, input_recurrent_weights, input_biases, input_combinations);
               calculate_recurrent_activations_derivatives(input_combinations, input_activations, input_activations_derivatives);

               calculate_state_combinations(current_inputs, state_weights, state_recurrent_weights, state_biases, state_combinations);
//               calculate_activations_derivatives(state_combinations, state_activations, state_activations_derivatives);

               calculate_output_combinations(current_inputs, output_weights, output_recurrent_weights, output_biases, output_combinations);
               calculate_recurrent_activations_derivatives(output_combinations, output_activations, output_activations_derivatives);

           }

           cell_states = forget_activations * cell_states + input_activations * state_activations;
//           hidden_states = output_activations * calculate_activations(cell_states);
//           const Tensor<type, 1> hidden_states_derivatives = calculate_activations_derivatives(cell_states);
           calculate_activations_derivatives(cell_states, hidden_states, hidden_states_derivatives);
           hidden_states *= output_activations;

//           forward_propagation.activations_2d.set_row(i,hidden_states);
           for(Index j = 0; j < neurons_number; j++)
               forward_propagation.activations_2d(i,j) = hidden_states(j);

//           forward_propagation.activations_derivatives.embed(forget_activations_index, forget_activations_derivatives);
//           forward_propagation.activations_derivatives.embed(input_activations_index, input_activations_derivatives);
//           forward_propagation.activations_derivatives.embed(state_activations_index, state_activations_derivatives);
//           forward_propagation.activations_derivatives.embed(output_activations_index, output_activations_derivatives);
//           forward_propagation.activations_derivatives.embed(hidden_states_index, hidden_states_derivatives);

           memcpy(forward_propagation.activations_derivatives_3d.data() + forget_activations_index, forget_activations_derivatives.data(), static_cast<size_t>(forget_activations_derivatives.size())*sizeof(type));
           memcpy(forward_propagation.activations_derivatives_3d.data() + input_activations_index, input_activations_derivatives.data(), static_cast<size_t>(input_activations_derivatives.size())*sizeof(type));
           memcpy(forward_propagation.activations_derivatives_3d.data() + state_activations_index, state_activations_derivatives.data(), static_cast<size_t>(state_activations_derivatives.size())*sizeof(type));
           memcpy(forward_propagation.activations_derivatives_3d.data() + output_activations_index, output_activations_derivatives.data(), static_cast<size_t>(output_activations_derivatives.size())*sizeof(type));
           memcpy(forward_propagation.activations_derivatives_3d.data() + hidden_states_index, hidden_states_derivatives.data(), static_cast<size_t>(hidden_states_derivatives.size())*sizeof(type));

           forget_activations_index++;
           input_activations_index++;
           state_activations_index++;
           output_activations_index++;
           hidden_states_index++;
       }

//       Layer::ForwardPropagation layers;

//       layers.activations_2d = activations_2d;
//       layers.activations_derivatives = activations_derivatives;

//       return layers;

//       return Layer::ForwardPropagation();

   }*/

   // Long short term memory layer error gradient

   Tensor<type, 1> calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);

   Tensor<type, 1> calculate_forget_weights_error_gradient(const Tensor<type, 2>&,
                                                           const Layer::ForwardPropagation&,
                                                           const Tensor<type, 2>&,
                                                           const Tensor<type, 3>&);

   Tensor<type, 1> calculate_input_weights_error_gradient(const Tensor<type, 2>&,
                                                          const Layer::ForwardPropagation&,
                                                          const Tensor<type, 2>&,
                                                          const Tensor<type, 3>&);

   Tensor<type, 1> calculate_state_weights_error_gradient(const Tensor<type, 2>&,
                                                          const Layer::ForwardPropagation&,
                                                          const Tensor<type, 2>&,
                                                          const Tensor<type, 3>&);

   Tensor<type, 1> calculate_output_weights_error_gradient(const Tensor<type, 2>&,
                                                           const Layer::ForwardPropagation&,
                                                           const Tensor<type, 2>&,
                                                           const Tensor<type, 3>&);

   Tensor<type, 1> calculate_forget_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                                     const Layer::ForwardPropagation&,
                                                                     const Tensor<type, 2>&,
                                                                     const Tensor<type, 3>&);

   Tensor<type, 1> calculate_input_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                                    const Layer::ForwardPropagation&,
                                                                    const Tensor<type, 2>&,
                                                                    const Tensor<type, 3>&);

   Tensor<type, 1> calculate_state_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                                    const Layer::ForwardPropagation&,
                                                                    const Tensor<type, 2>&,
                                                                    const Tensor<type, 3>&);

   Tensor<type, 1> calculate_output_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                                     const Layer::ForwardPropagation&,
                                                                     const Tensor<type, 2>&,
                                                                     const Tensor<type, 3>&);

   Tensor<type, 1> calculate_forget_biases_error_gradient(const Tensor<type, 2>&,
                                                          const Layer::ForwardPropagation&,
                                                          const Tensor<type, 2>&,
                                                          const Tensor<type, 3>&);

   Tensor<type, 1> calculate_input_biases_error_gradient(const Tensor<type, 2>&,
                                                         const Layer::ForwardPropagation&,
                                                         const Tensor<type, 2>&,
                                                         const Tensor<type, 3>&);

   Tensor<type, 1> calculate_state_biases_error_gradient(const Tensor<type, 2>&,
                                                         const Layer::ForwardPropagation&,
                                                         const Tensor<type, 2>&,
                                                         const Tensor<type, 3>&);

   Tensor<type, 1> calculate_output_biases_error_gradient(const Tensor<type, 2>&,
                                                          const Layer::ForwardPropagation&,
                                                          const Tensor<type, 2>&,
                                                          const Tensor<type, 3>&);

   // Expression methods

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_recurrent_activation_function_expression() const;
   string write_activation_function_expression() const;

   // Utilities

   Tensor<type, 2> multiply_rows(const Tensor<type,2>&, const Tensor<type,1>&) const;



   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) {}

   void write_XML(tinyxml2::XMLPrinter&) const {}

protected:

   Index timesteps = 10;

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
   ActivationFunction recurrent_activation_function = HardSigmoid;

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
