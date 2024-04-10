//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S   H E A D E R             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PERCEPTRONLAYER_H
#define PERCEPTRONLAYER_H

// System includes

#include <algorithm>
#include <cstdlib>
#include <execution>
#include <iostream>
#include <string>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "probabilistic_layer.h"

namespace opennn
{

struct LayerForwardPropagation;
struct LayerBackPropagation;
struct LayerBackPropagationLM;

struct PerceptronLayerForwardPropagation;
struct PerceptronLayerBackPropagation;
struct PerceptronLayerBackPropagationLM;

struct ProbabilisticLayerForwardPropagation;
struct ProbabilisticLayerBackPropagation;
struct ProbabilisticLayerBackPropagationLM;

#ifdef OPENNN_CUDA
struct PerceptronLayerForwardPropagationCuda;
struct PerceptronLayerBackPropagationCuda;
#endif


/// This class represents a layer of perceptrons.

/// PerceptronLayer is a single-layer network with a hard-limit transfer function.
/// This network is often trained with the perceptron learning rule.
///
/// Layers of perceptrons will be used to construct multilayer perceptrons, such as an approximation problems .

class PerceptronLayer : public Layer
{

public:

    /// Enumeration of the available activation functions for the perceptron neuron model.

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

   explicit PerceptronLayer();

   explicit PerceptronLayer(const Index&,
                            const Index&,
                            const ActivationFunction& = PerceptronLayer::ActivationFunction::HyperbolicTangent);

   // Get methods

   Index get_inputs_number() const final;
   Index get_neurons_number() const final;

   // Parameters

   const Tensor<type, 1>& get_biases() const;
   const Tensor<type, 2>& get_synaptic_weights() const;
   Tensor<type, 1> get_parameters() const final;

   Index get_biases_number() const;
   Index get_synaptic_weights_number() const;
   Index get_parameters_number() const final;
   type get_dropout_rate() const;

   // Activation functions

   const PerceptronLayer::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();

   void set(const Index&,
            const Index&,
            const PerceptronLayer::ActivationFunction& = PerceptronLayer::ActivationFunction::HyperbolicTangent);

   void set_default();
   void set_name(const string&);

   // Architecture

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;

   // Parameters

   void set_biases(const Tensor<type, 1>&);
   void set_synaptic_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& index=0) final;

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);
   void set_dropout_rate(const type&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods

   void set_biases_constant(const type&);
   void set_synaptic_weights_constant(const type&);
   
   void set_parameters_constant(const type&) final;

   void set_parameters_random() final;

   // Forward propagation

   void calculate_combinations(const Tensor<type, 2>&,
                               Tensor<type, 2>&) const;

   void dropout(Tensor<type, 2>&) const;

   void calculate_activations(const Tensor<type, 2>&,
                              Tensor<type, 2>&) const;

   void calculate_activations_derivatives(const Tensor<type, 2>&,
                                          Tensor<type, 2>&,
                                          Tensor<type, 2>&) const;

   void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&layer,
                          LayerForwardPropagation*,
                          const bool&) final;

   // Delta methods

   void calculate_hidden_delta(LayerForwardPropagation*,
                               LayerBackPropagation*,
                               LayerForwardPropagation*,
                               LayerBackPropagation*) const final;

   void calculate_hidden_delta(PerceptronLayerForwardPropagation*,
                               PerceptronLayerBackPropagation*,
                               PerceptronLayerBackPropagation*) const;

   void calculate_hidden_delta(ProbabilisticLayerForwardPropagation*,
                               ProbabilisticLayerBackPropagation*,
                               PerceptronLayerBackPropagation*) const;

   // Delta LM

   void calculate_hidden_delta_lm(LayerForwardPropagation*,
                                  LayerBackPropagationLM*,
                                  LayerBackPropagationLM*) const final;

   void calculate_hidden_delta_lm(PerceptronLayerForwardPropagation*,
                                  PerceptronLayerBackPropagationLM*,
                                  PerceptronLayerBackPropagationLM*) const;

   void calculate_hidden_delta_lm(ProbabilisticLayerForwardPropagation*,
                                  ProbabilisticLayerBackPropagationLM*,
                                  PerceptronLayerBackPropagationLM*) const;

   // Squared errors methods

   void calculate_squared_errors_Jacobian_lm(const Tensor<type, 2>&,
                                             LayerForwardPropagation*,
                                             LayerBackPropagationLM*) final;

   void insert_squared_errors_Jacobian_lm(LayerBackPropagationLM*,
                                          const Index&,
                                          Tensor<type, 2>&) const final;

   // Gradient methods

   void calculate_error_combinations_derivatives(const Tensor<type, 2>&,
                                                 const Tensor<type, 2>&,
                                                 Tensor<type, 2>&) const;

   void calculate_error_gradient(const Tensor<pair<type*, dimensions>, 1>&,
                                 LayerForwardPropagation*,
                                 LayerBackPropagation*) const final;

   void insert_gradient(LayerBackPropagation*,
                        const Index&,
                        Tensor<type, 1>&) const final;

   // Expression methods   

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

   string write_activation_function_expression() const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) final;
   void write_XML(tinyxml2::XMLPrinter&) const final;


protected:

   // MEMBERS


   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's transfer function to generate the neuron's output.

   Tensor<type, 1> biases;

   /// This matrix contains conection strengths from a layer's inputs to its neurons.

   Tensor<type, 2> synaptic_weights;

   /// Activation function variable.

   ActivationFunction activation_function;

   type dropout_rate = type(0);

   /// Display messages to screen. 

   bool display = true;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/perceptron_layer_cuda.h"
    #endif

};


struct PerceptronLayerForwardPropagation : LayerForwardPropagation
{
    explicit PerceptronLayerForwardPropagation();

    explicit PerceptronLayerForwardPropagation(const Index&, Layer*);

    virtual ~PerceptronLayerForwardPropagation();

    pair<type *, dimensions> get_outputs_pair() const final;

    void set(const Index&, Layer*) final;

    void print() const;

    Tensor<type, 2> outputs;

    Tensor<type, 2> activations_derivatives;
};


struct PerceptronLayerBackPropagation : LayerBackPropagation
{
    // Default constructor

    explicit PerceptronLayerBackPropagation();

    explicit PerceptronLayerBackPropagation(const Index&, Layer*);

    virtual ~PerceptronLayerBackPropagation();

    pair<type *, dimensions> get_deltas_pair() const final;

    void set(const Index&, Layer*) final;

    void print() const;

    Tensor<type, 2> deltas;

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 2> synaptic_weights_derivatives;

    Tensor<type, 2> error_combinations_derivatives;
};


struct PerceptronLayerBackPropagationLM : LayerBackPropagationLM
{
    // Default constructor

    explicit PerceptronLayerBackPropagationLM();

    explicit PerceptronLayerBackPropagationLM(const Index&, Layer*);

    virtual ~PerceptronLayerBackPropagationLM();

    void set(const Index&, Layer*) final;

    void print() const;

    Tensor<type, 2> squared_errors_Jacobian;

    Tensor<type, 2> error_combinations_derivatives;

};


#ifdef OPENNN_CUDA
<<<<<<< HEAD
    #include "../../opennn_cuda/opennn_cuda/perceptron_layer_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/perceptron_layer_back_propagation_cuda.h"
#endif
=======
    #include "../../opennn_cuda/opennn_cuda/struct_perceptron_layer_cuda.h"
#endif

>>>>>>> 9aaea002d10155013a639c82f43f18234f5caf7a

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
