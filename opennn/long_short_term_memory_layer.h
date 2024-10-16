//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LONGSHORTTERMMEMORYLAYER_H
#define LONGSHORTTERMMEMORYLAYER_H

#include <iostream>
#include <string>

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{

struct LongShortTermMemoryLayerForwardPropagation;
struct LongShortTermMemoryLayerBackPropagation;

class LongShortTermMemoryLayer : public Layer
{

public:

    enum class ActivationFunction{Logistic, 
                                  HyperbolicTangent,
                                  Linear, 
                                  RectifiedLinear, 
                                  ExponentialLinear, 
                                  ScaledExponentialLinear,
                                  SoftPlus, 
                                  SoftSign, 
                                  HardSigmoid};

   // Constructors

   explicit LongShortTermMemoryLayer();

   explicit LongShortTermMemoryLayer(const Index&, const Index&, const Index&);


   // Get

   bool is_empty() const;

   Index get_inputs_number() const final;
   Index get_neurons_number() const final;

   dimensions get_output_dimensions() const final;

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

   Index get_parameters_number() const final;
   Tensor<type, 1> get_parameters() const final;

   // Activation functions

   const LongShortTermMemoryLayer::ActivationFunction& get_activation_function() const;
   const LongShortTermMemoryLayer::ActivationFunction& get_recurrent_activation_function() const;

   string write_activation_function() const;
   string write_recurrent_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set

   void set();
   void set(const Index&, const Index&, const Index&);
   void set(const LongShortTermMemoryLayer&);

   void set_default();

   // Architecture

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;

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

   void set_parameters(const Tensor<type, 1>&, const Index& = 0) final;

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   void set_recurrent_activation_function(const ActivationFunction&);
   void set_recurrent_activation_function(const string&);

   void set_timesteps(const Index&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization

   void set_parameters_constant(const type&) final;

   void set_parameters_random() final;

   // Forward propagation

   void calculate_combinations(const Tensor<type, 1>&,
                               const Tensor<type, 2>&,
                               const Tensor<type, 1>&,
                               const Tensor<type, 2>&,
                               const Tensor<type, 1>&,
                               Tensor<type, 1>&) const;

   void calculate_activations(Tensor<type, 1>&,
                              Tensor<type, 1>&) const;

   void calculate_recurrent_activations(Tensor<type, 1>&,
                                        Tensor<type, 1>&) const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>&,
                          const bool&) final;

   // Back propagation

   void insert_gradient(unique_ptr<LayerBackPropagation>,
                        const Index& ,
                        Tensor<type, 1>&) const final;

   void back_propagate(const vector<pair<type*, dimensions>>&,
                       const vector<pair<type*, dimensions>>&,
                       unique_ptr<LayerForwardPropagation>&,
                       unique_ptr<LayerBackPropagation>&) const final;

   void calculate_forget_parameters_derivatives(const Tensor<type, 2>&,
                                                const Tensor<type, 2>&,
       unique_ptr<LongShortTermMemoryLayerForwardPropagation>,
       unique_ptr<LongShortTermMemoryLayerBackPropagation>) const;

   void calculate_input_parameters_derivatives(const Tensor<type, 2>&,
                                               const Tensor<type, 2>&,
       unique_ptr < LongShortTermMemoryLayerForwardPropagation>,
       unique_ptr < LongShortTermMemoryLayerBackPropagation>) const;

   void calculate_state_parameters_derivatives(const Tensor<type, 2>&,
                                               const Tensor<type, 2>&,
       unique_ptr < LongShortTermMemoryLayerForwardPropagation>,
       unique_ptr < LongShortTermMemoryLayerBackPropagation>) const;

   void calculate_output_parameters_derivatives(const Tensor<type, 2>&,
                                                const Tensor<type, 2>&,
       unique_ptr < LongShortTermMemoryLayerForwardPropagation>,
       unique_ptr < LongShortTermMemoryLayerBackPropagation>) const;

   // Expression

   string write_recurrent_activation_function_expression() const;

   string write_activation_function_expression() const;


   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

   // Serialization

   void from_XML(const tinyxml2::XMLDocument&) final;

   void to_XML(tinyxml2::XMLPrinter&) const final;

protected:

   Index time_steps = 3;

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

   ActivationFunction activation_function = ActivationFunction::HyperbolicTangent;
   ActivationFunction recurrent_activation_function = ActivationFunction::HardSigmoid;

   bool display = true;

   Tensor<type, 1> empty;

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/long_short_term_memory_layer_cuda.h"
#endif

};


struct LongShortTermMemoryLayerForwardPropagation : LayerForwardPropagation
{
    explicit LongShortTermMemoryLayerForwardPropagation() : LayerForwardPropagation()
    {
    }

    explicit LongShortTermMemoryLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }
    
    
    pair<type*, dimensions> get_outputs_pair() const final;


    void set(const Index& new_batch_samples_number, Layer* new_layer) final;


    void print() const
    {
        cout << "Current inputs: " << endl;
        cout << current_inputs << endl;

        cout << "Current input activations: " << endl;
        cout << current_input_activations << endl;

        cout << "Current input activations derivatives: " << endl;
        cout << current_input_activations_derivatives << endl;
     }


    Tensor<type, 1> current_inputs;

    Tensor<type, 1> current_forget_activations;
    Tensor<type, 1> current_forget_activations_derivatives;
    Tensor<type, 2, RowMajor> forget_activations;
    Tensor<type, 2, RowMajor> forget_activations_derivatives;

    Tensor<type, 1> current_input_activations;
    Tensor<type, 1> current_input_activations_derivatives;
    Tensor<type, 2, RowMajor> input_activations;
    Tensor<type, 2, RowMajor> input_activations_derivatives;

    Tensor<type, 1> current_state_activations;
    Tensor<type, 1> current_state_activations_derivatives;
    Tensor<type, 2, RowMajor> state_activations;
    Tensor<type, 2, RowMajor> state_activations_derivatives;

    Tensor<type, 1> current_output_activations;
    Tensor<type, 1> current_output_activations_derivatives;
    Tensor<type, 2, RowMajor> output_activations;
    Tensor<type, 2, RowMajor> output_activations_derivatives;

    Tensor<type, 1> previous_cell_states;
    Tensor<type, 1> current_cell_states;
    Tensor<type, 2, RowMajor> cell_states;

    Tensor<type, 1> previous_hidden_states;
    Tensor<type, 1> current_hidden_states;
    Tensor<type, 1> current_hidden_states_activations_derivatives;
    Tensor<type, 2, RowMajor> hidden_states;
    Tensor<type, 2, RowMajor> hidden_states_activations_derivatives;

    Tensor<type, 2> outputs;
};


struct LongShortTermMemoryLayerBackPropagation : LayerBackPropagation
{
    explicit LongShortTermMemoryLayerBackPropagation() : LayerBackPropagation()
    {
    }

    explicit LongShortTermMemoryLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& new_batch_samples_number, Layer* new_layer) final;

    void set_derivatives_zero();

    void print() const
    {
    }

    Tensor<type, 1> current_deltas;

    Tensor<type, 1> forget_weights_derivatives;
    Tensor<type, 1> input_weights_derivatives;
    Tensor<type, 1> state_weights_derivatives;
    Tensor<type, 1> output_weights_derivatives;

    Tensor<type, 2> hidden_states_weights_derivatives;
    Tensor<type, 2> cell_states_weights_derivatives;

    Tensor<type, 1> forget_recurrent_weights_derivatives;
    Tensor<type, 1> input_recurrent_weights_derivatives;
    Tensor<type, 1> state_recurrent_weights_derivatives;
    Tensor<type, 1> output_recurrent_weights_derivatives;

    Tensor<type, 2> hidden_states_recurrent_weights_derivatives;
    Tensor<type, 2> cell_states_recurrent_weights_derivatives;

    Tensor<type, 1> forget_biases_derivatives;
    Tensor<type, 1> input_biases_derivatives;
    Tensor<type, 1> state_biases_derivatives;
    Tensor<type, 1> output_biases_derivatives;

    Tensor<type, 2> hidden_states_biases_derivatives;
    Tensor<type, 2> cell_states_biases_derivatives;

    Tensor<type, 2> input_combinations_weights_derivatives;
    Tensor<type, 2> forget_combinations_weights_derivatives;
    Tensor<type, 2> state_combinations_weights_derivatives;
    Tensor<type, 2> output_combinations_weights_derivatives;

    Tensor<type, 2> input_combinations_recurrent_weights_derivatives;
    Tensor<type, 2> forget_combinations_recurrent_weights_derivatives;
    Tensor<type, 2> state_combinations_recurrent_weights_derivatives;
    Tensor<type, 2> output_combinations_recurrent_weights_derivatives;

    Tensor<type, 2> input_combinations_biases_derivatives;
    Tensor<type, 2> forget_combinations_biases_derivatives;
    Tensor<type, 2> state_combinations_biases_derivatives;
    Tensor<type, 2> output_combinations_biases_derivatives;

    Tensor<type, 2> input_derivatives;
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
