//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef RECURRENTLAYER_H
#define RECURRENTLAYER_H



#include <string>



#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{

struct RecurrentLayerForwardPropagation;
struct RecurrentLayerBackPropagation;

#ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/struct_recurrent_layer_cuda.h"
#endif

class RecurrentLayer : public Layer
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

   explicit RecurrentLayer();

   explicit RecurrentLayer(const Index&, const Index&, const Index&);

   // Get

   bool is_empty() const;

   Index get_inputs_number() const final;
   Index get_neurons_number() const final;

   dimensions get_output_dimensions() const final;

   const Tensor<type, 3>& get_hidden_states() const;

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

   // Set

   void set();
   void set(const Index&, const Index&, const Index&);
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

   // Parameters initialization

   void set_parameters_constant(const type&) final;

   void set_parameters_random() final;

   // Forward propagation

   void calculate_combinations(const Tensor<type, 2>&,
                               Tensor<type, 2>&) const;

   void calculate_activations(Tensor<type, 2>&,
                              Tensor<type, 2>&) const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>,
                          const bool&) final;

   // Back propagation

   void insert_gradient(unique_ptr<LayerBackPropagation>,
                        const Index& ,
                        Tensor<type, 1>&) const final;

   void back_propagate(const vector<pair<type*, dimensions>>&,
                       const vector<pair<type*, dimensions>>&,
                       unique_ptr<LayerForwardPropagation>,
                       unique_ptr<LayerBackPropagation>) const final;

   // Expression

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

   string write_activation_function_expression() const;

   // Serialization

   void from_XML(const tinyxml2::XMLDocument&) final;

   void to_XML(tinyxml2::XMLPrinter&) const final;

protected:

   Index time_steps = 1;

   Tensor<type, 1> biases;

   Tensor<type, 2> input_weights;

   Tensor<type, 2> recurrent_weights;

   ActivationFunction activation_function = ActivationFunction::HyperbolicTangent;

   Tensor<type, 3> hidden_states;

   bool display = true;

   Tensor<type, 2> empty;

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/recurrent_layer_cuda.h"
#endif

};


struct RecurrentLayerForwardPropagation : LayerForwardPropagation
{
    explicit RecurrentLayerForwardPropagation() : LayerForwardPropagation()
    {
    }


    explicit RecurrentLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer) : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }

    pair<type*, dimensions> get_outputs_pair() const final;

    void set(const Index& new_batch_samples_number, Layer* new_layer) final;

    void print() const
    {
    }

    Tensor<type, 2> current_inputs;
    Tensor<type, 2> current_activations_derivatives;

    Tensor<type, 3> activations_derivatives;
};


struct RecurrentLayerBackPropagation : LayerBackPropagation
{
    explicit RecurrentLayerBackPropagation() : LayerBackPropagation()
    {
    }

    virtual ~RecurrentLayerBackPropagation()
    {
    }

    explicit RecurrentLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& new_batch_samples_number, Layer* new_layer) final;

    void print() const
    {

    }

    //Tensor<type, 1> current_deltas;

    Tensor<type, 2> combinations_derivatives;
    Tensor<type, 1> current_combinations_derivatives;

    Tensor<type, 2> combinations_biases_derivatives;
    Tensor<type, 3> combinations_input_weights_derivatives;
    Tensor<type, 3> combinations_recurrent_weights_derivatives;

    Tensor<type, 1> biases_derivatives;

    Tensor<type, 2> input_weights_derivatives;

    Tensor<type, 2> recurrent_weights_derivatives;

    Tensor<type, 3> input_derivatives;
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
