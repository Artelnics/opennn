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

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "probabilistic_layer.h"

#include "opennn_strings.h"

namespace opennn
{

struct PerceptronLayerForwardPropagation;
struct PerceptronLayerBackPropagation;
struct PerceptronLayerBackPropagationLM;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/struct_perceptron_layer_cuda.h"
#endif


/// This class represents a layer of perceptrons.

/// PerceptronLayer is a single-layer network with a hard-limit trabsfer function.
/// This network is often trained with the perceptron learning rule.
///
/// Layers of perceptrons will be used to construct multilayer perceptrons, such as an approximation problems .

class PerceptronLayer : public Layer

{

public:

    /// Enumeration of the available activation functions for the perceptron neuron model.

    enum class ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear,
                            ExponentialLinear, ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

   // Constructors

   explicit PerceptronLayer();

   explicit PerceptronLayer(const Index&, const Index&, const ActivationFunction& = PerceptronLayer::ActivationFunction::HyperbolicTangent);

   // Get methods

   bool is_empty() const;

   Index get_inputs_number() const override;
   Index get_neurons_number() const final;

   // Parameters

   const Tensor<type, 2>& get_biases() const;
   const Tensor<type, 2>& get_synaptic_weights() const;

   Tensor<type, 2> get_biases(const Tensor<type, 1>&) const;
   Tensor<type, 2> get_synaptic_weights(const Tensor<type, 1>&) const;

   Index get_biases_number() const;
   Index get_synaptic_weights_number() const;
   Index get_parameters_number() const final;
   Tensor<type, 1> get_parameters() const final;

   // Activation functions

   const PerceptronLayer::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&, const PerceptronLayer::ActivationFunction& = PerceptronLayer::ActivationFunction::HyperbolicTangent);

   void set_default();
   void set_name(const string&);

   // Architecture

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;

   // Parameters

   void set_biases(const Tensor<type, 2>&);
   void set_synaptic_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& index=0) final;

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods
   void set_biases_constant(const type&);
   void set_synaptic_weights_constant(const type&);
   
   void set_parameters_constant(const type&) final;

   void set_parameters_random() final;

   // Perceptron layer combinations

   void calculate_combinations(type*, Tensor<Index,1>&,
                               const Tensor<type, 2>&,
                               const Tensor<type, 2>&,
                               type*, Tensor<Index, 1>&) const;

   // Perceptron layer activations

   void calculate_activations(type*, Tensor<Index, 1>&,
                              type*, Tensor<Index, 1>&) const;

   void calculate_activations_derivatives(type*, Tensor<Index, 1>&,
                                          type*, Tensor<Index, 1>&,
                                          type*, Tensor<Index, 1>&) const;

   // Perceptron layer outputs


   void calculate_outputs(type*, Tensor<Index, 1>&, type*, Tensor<Index, 1>&) final;

   void forward_propagate(type*,
                          Tensor<Index, 1>&,
                          LayerForwardPropagation*) final;

   void forward_propagate(type*,
                          Tensor<Index, 1>&,
                          Tensor<type, 1>&,
                          LayerForwardPropagation*) final;

   // Delta methods

   void calculate_hidden_delta(LayerForwardPropagation*,
                               LayerBackPropagation*,
                               LayerBackPropagation*) const final;

   void calculate_hidden_delta_perceptron(PerceptronLayerForwardPropagation*,
                                          PerceptronLayerBackPropagation*,
                                          PerceptronLayerBackPropagation*) const;

   void calculate_hidden_delta_probabilistic(ProbabilisticLayerForwardPropagation*,
                                             ProbabilisticLayerBackPropagation*,
                                             PerceptronLayerBackPropagation*) const;

   // Delta LM

   void calculate_hidden_delta_lm(LayerForwardPropagation*,
                                  LayerBackPropagationLM*,
                                  LayerBackPropagationLM*) const final;

   void calculate_hidden_delta_perceptron_lm(PerceptronLayerForwardPropagation*,
                                             PerceptronLayerBackPropagationLM*,
                                             PerceptronLayerBackPropagationLM*) const;

   void calculate_hidden_delta_probabilistic_lm(ProbabilisticLayerForwardPropagation*,
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

   void calculate_error_gradient(const Tensor<type, 2>&,
                                 LayerForwardPropagation*,
                                 LayerBackPropagation*) const final;

   void insert_gradient(LayerBackPropagation*,
                        const Index&,
                        Tensor<type, 1>&) const final;

   // Expression methods   

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

   string write_activation_function_expression() const;

   string write_expression_c() const final;
   string write_combinations_c() const;
   string write_activations_c() const;

   string write_combinations_python() const;
   string write_activations_python() const;
   string write_expression_python() const final;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) final;
   void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

   // MEMBERS

   /// Inputs

   Tensor<type, 2> inputs;

   /// Outputs

   Tensor<type, 2> outputs;

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's transfer function to generate the neuron's output.

   Tensor<type, 2> biases;

   /// This matrix contains conection strengths from a layer's inputs to its neurons.

   Tensor<type, 2> synaptic_weights;

   /// Activation function variable.

   ActivationFunction activation_function;

   /// Display messages to screen. 

   bool display = true;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/perceptron_layer_cuda.h"
#else
};
#endif

struct PerceptronLayerForwardPropagation : LayerForwardPropagation
{
    // Default constructor

    explicit PerceptronLayerForwardPropagation() : LayerForwardPropagation()
    {
    }

    explicit PerceptronLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }

    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        layer_pointer = new_layer_pointer;

        batch_samples_number = new_batch_samples_number;

        const Index neurons_number = layer_pointer->get_neurons_number();

        combinations.resize(batch_samples_number, neurons_number);

        activations.resize(batch_samples_number, neurons_number);

        activations_derivatives.resize(batch_samples_number, neurons_number);

        outputs_ptr = activations.data();

        outputs_dims = get_dimensions(activations);
    }

    void print() const
    {
        cout << "Combinations:" << endl;
        cout << combinations << endl;

        cout << "Activations:" << endl;
        cout << activations << endl;

        cout << "Activations derivatives:" << endl;
        cout << activations_derivatives << endl;
    }

    Tensor<type, 2> combinations;
    Tensor<type, 2> activations;
    Tensor<type, 2> activations_derivatives;
};


struct PerceptronLayerBackPropagationLM : LayerBackPropagationLM
{
    // Default constructor

    explicit PerceptronLayerBackPropagationLM() : LayerBackPropagationLM()
    {

    }


    explicit PerceptronLayerBackPropagationLM(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagationLM()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        layer_pointer = new_layer_pointer;

        batch_samples_number = new_batch_samples_number;

        const Index neurons_number = layer_pointer->get_neurons_number();
        const Index parameters_number = layer_pointer->get_parameters_number();

        delta.resize(batch_samples_number, neurons_number);

        squared_errors_Jacobian.resize(batch_samples_number, parameters_number);
    }

    void print() const
    {
        cout << "Delta:" << endl;
        cout << delta << endl;

        cout << "Squared errors Jacobian: " << endl;
        cout << squared_errors_Jacobian << endl;

    }

    Tensor<type, 2> delta;

    Tensor<type, 2> squared_errors_Jacobian;
};



struct PerceptronLayerBackPropagation : LayerBackPropagation
{
    // Default constructor

    explicit PerceptronLayerBackPropagation() : LayerBackPropagation()
    {

    }


    explicit PerceptronLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
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

        delta.resize(batch_samples_number, neurons_number);

        biases_derivatives.resize(neurons_number);

        synaptic_weights_derivatives.resize(inputs_number, neurons_number);
    }

    void print() const
    {
        cout << "Delta:" << endl;
        cout << delta << endl;

        cout << "Biases derivatives:" << endl;
        cout << biases_derivatives << endl;

        cout << "Synaptic weights derivatives:" << endl;
        cout << synaptic_weights_derivatives << endl;
    }

    Tensor<type, 2> delta;

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 2> synaptic_weights_derivatives;
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
