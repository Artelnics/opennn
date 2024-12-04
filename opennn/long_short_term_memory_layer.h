//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LONGSHORTTERMMEMORYLAYER_H
#define LONGSHORTTERMMEMORYLAYER_H

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

   LongShortTermMemoryLayer(const Index& = 0, const Index& = 0, const Index& = 0);

   dimensions get_input_dimensions() const override;
   dimensions get_output_dimensions() const override;

   Index get_timesteps() const;

   Index get_parameters_number() const override;
   Tensor<type, 1> get_parameters() const override;

   const LongShortTermMemoryLayer::ActivationFunction& get_activation_function() const;
   const LongShortTermMemoryLayer::ActivationFunction& get_recurrent_activation_function() const;

   string get_activation_function_string() const;
   string write_recurrent_activation_function() const;

   void set(const Index& = 0, const Index& = 0, const Index& = 0);

   void set_input_dimensions(const dimensions&) override;
   void set_output_dimensions(const dimensions&) override;

   void set_parameters(const Tensor<type, 1>&, const Index&) override;

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   void set_recurrent_activation_function(const ActivationFunction&);
   void set_recurrent_activation_function(const string&);

   void set_timesteps(const Index&);

   // Parameters initialization

   void set_parameters_constant(const type&) override;

   void set_parameters_random() override;

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
                          const bool&) override;

   // Back propagation

   void insert_gradient(unique_ptr<LayerBackPropagation>&,
                        const Index& ,
                        Tensor<type, 1>&) const override;

   void back_propagate(const vector<pair<type*, dimensions>>&,
                       const vector<pair<type*, dimensions>>&,
                       unique_ptr<LayerForwardPropagation>&,
                       unique_ptr<LayerBackPropagation>&) const override;

   void calculate_forget_parameter_derivatives(const Tensor<type, 2>&,
                                               const Tensor<type, 2>&,
                                               unique_ptr<LongShortTermMemoryLayerForwardPropagation>&,
                                               unique_ptr<LongShortTermMemoryLayerBackPropagation>&) const;

   void calculate_input_parameter_derivatives(const Tensor<type, 2>&,
                                              const Tensor<type, 2>&,
                                              unique_ptr<LongShortTermMemoryLayerForwardPropagation>&,
                                              unique_ptr<LongShortTermMemoryLayerBackPropagation>&) const;

   void calculate_state_parameter_derivatives(const Tensor<type, 2>&,
                                              const Tensor<type, 2>&,
                                              unique_ptr<LongShortTermMemoryLayerForwardPropagation>&,
                                              unique_ptr<LongShortTermMemoryLayerBackPropagation>&) const;

   void calculate_output_parameter_derivatives(const Tensor<type, 2>&,
                                               const Tensor<type, 2>&,
                                               unique_ptr<LongShortTermMemoryLayerForwardPropagation>&,
                                               unique_ptr<LongShortTermMemoryLayerBackPropagation>&) const;

   // Expression

   string write_recurrent_activation_function_expression() const;

   string get_activation_function_string_expression() const;

   string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

   // Serialization

   void from_XML(const XMLDocument&) override;

   void to_XML(XMLPrinter&) const override;

private:

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

   Tensor<type, 1> empty;

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/long_short_term_memory_layer_cuda.h"
#endif

};


struct LongShortTermMemoryLayerForwardPropagation : LayerForwardPropagation
{
    LongShortTermMemoryLayerForwardPropagation(const Index& = 0, Layer* = nullptr);
        
    pair<type*, dimensions> get_outputs_pair() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const;

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

    LongShortTermMemoryLayerBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& = 0, Layer* = nullptr) override;

    void set_derivatives_zero();

    void print() const;

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
