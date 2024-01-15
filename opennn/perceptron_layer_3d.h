//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PERCEPTRONLAYER3D_H
#define PERCEPTRONLAYER3D_H

// System includes

#include <cstdlib>
#include <iostream>
#include <string>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "probabilistic_layer_3d.h"

#ifdef OPENNN_MKL
    #include "../mkl/mkl.h"
#endif

namespace opennn
{

struct PerceptronLayer3DForwardPropagation;
struct PerceptronLayer3DBackPropagation;
struct PerceptronLayer3DBackPropagationLM;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/struct_perceptron_layer_cuda.h"
#endif


/// This class represents a layer of perceptrons.

/// PerceptronLayer is a single-layer network with a hard-limit transfer function.
/// This network is often trained with the perceptron learning rule.
///
/// Layers of perceptrons will be used to construct multilayer perceptrons, such as an approximation problems .

class PerceptronLayer3D : public Layer
{

public:

    /// Enumeration of the available activation functions for the perceptron neuron model.

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

   // Constructors

   explicit PerceptronLayer3D();

   explicit PerceptronLayer3D(const Index&,
                              const Index&,
                              const Index&,
                              const ActivationFunction& = PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

   // Get methods

   bool is_empty() const;

   Index get_inputs_number() const override;
   Index get_inputs_size() const;
   Index get_neurons_number() const final;

   // Parameters

   const Tensor<type, 1>& get_biases() const;
   const Tensor<type, 2>& get_synaptic_weights() const;

   Tensor<type, 2> get_biases(const Tensor<type, 1>&) const;
   Tensor<type, 2> get_synaptic_weights(const Tensor<type, 1>&) const;

   Index get_biases_number() const;
   Index get_synaptic_weights_number() const;
   Index get_parameters_number() const final;
   type get_dropout_rate() const;
   Tensor<type, 1> get_parameters() const final;

   // Activation functions

   const PerceptronLayer3D::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&,
            const Index&,
            const Index&,
            const PerceptronLayer3D::ActivationFunction& = PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

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

   void calculate_combinations(const Tensor<type, 3>&,
                               const Tensor<type, 1>&,
                               const Tensor<type, 2>&,
                               Tensor<type, 3>&) const;

   void dropout(Tensor<type, 3>&);

   void calculate_activations(const Tensor<type, 3>&,
                              Tensor<type, 3>&) const;

   void calculate_activations_derivatives(const Tensor<type, 3>&,
                                          Tensor<type, 3>&,
                                          Tensor<type, 3>&) const;

   void forward_propagate(const pair<type*, dimensions>&,
                          LayerForwardPropagation*,
                          const bool&) final;

   void forward_propagate(const pair<type*, dimensions>&,
                          Tensor<type, 1>&,
                          LayerForwardPropagation*) final;

   // Delta methods

   void calculate_hidden_delta(LayerForwardPropagation*,
                               LayerBackPropagation*,
                               LayerBackPropagation*) const final;

   void calculate_hidden_delta(PerceptronLayer3DForwardPropagation*,
                               PerceptronLayer3DBackPropagation*,
                               PerceptronLayer3DBackPropagation*) const;

   void calculate_hidden_delta(ProbabilisticLayer3DForwardPropagation*,
                               ProbabilisticLayer3DBackPropagation*,
                               PerceptronLayer3DBackPropagation*) const;

   // Delta LM

   void calculate_hidden_delta_lm(LayerForwardPropagation*,
                                  LayerBackPropagationLM*,
                                  LayerBackPropagationLM*) const final;

   void calculate_hidden_delta_lm(PerceptronLayer3DForwardPropagation*,
                                  PerceptronLayer3DBackPropagationLM*,
                                  PerceptronLayer3DBackPropagationLM*) const;

   void calculate_hidden_delta_lm(ProbabilisticLayer3DForwardPropagation*,
                                  ProbabilisticLayer3DBackPropagationLM*,
                                  PerceptronLayer3DBackPropagationLM*) const;

   // Squared errors methods

//   void calculate_squared_errors_Jacobian_lm(const Tensor<type, 2>&,
//                                             LayerForwardPropagation*,
//                                             LayerBackPropagationLM*) final;

//   void insert_squared_errors_Jacobian_lm(LayerBackPropagationLM*,
//                                          const Index&,
//                                          Tensor<type, 2>&) const final;

   // Gradient methods

   void calculate_error_gradient(const pair<type*, dimensions>&,
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


   Index inputs_size;

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
    #include "../../opennn-cuda/opennn-cuda/perceptron_layer_cuda.h"
#else
};
#endif


struct PerceptronLayer3DForwardPropagation : LayerForwardPropagation
{
    // Default constructor

     explicit PerceptronLayer3DForwardPropagation() : LayerForwardPropagation()
     {
     }


     explicit PerceptronLayer3DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
         : LayerForwardPropagation()
     {        
        set(new_batch_samples_number, new_layer_pointer);
     }


    virtual ~PerceptronLayer3DForwardPropagation()
    {
    }


    pair<type*, dimensions> get_outputs() const final
    {
        PerceptronLayer3D* perceptron_layer_3d_pointer = static_cast<PerceptronLayer3D*>(layer_pointer);

        const Index neurons_number = perceptron_layer_3d_pointer->get_neurons_number();

        const Index inputs_size = perceptron_layer_3d_pointer->get_inputs_size();

        return pair<type*, dimensions>(outputs_data, {{batch_samples_number, inputs_size, neurons_number}});
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer) final
    {
        layer_pointer = new_layer_pointer;

        PerceptronLayer3D* perceptron_layer_3d_pointer = static_cast<PerceptronLayer3D*>(layer_pointer);

        batch_samples_number = new_batch_samples_number;

        const Index neurons_number = perceptron_layer_3d_pointer->get_neurons_number();

        const Index inputs_size = perceptron_layer_3d_pointer->get_inputs_size();

        outputs.resize(batch_samples_number, inputs_size, neurons_number);

        outputs_data = outputs.data();

        activations_derivatives.resize(batch_samples_number, inputs_size, neurons_number);
    }


     void print() const
     {
         cout << "Outputs:" << endl;
         cout << outputs << endl;

         cout << "Activations derivatives:" << endl;
         cout << activations_derivatives << endl;
     }

     Tensor<type, 3> outputs;

     Tensor<type, 3> activations_derivatives;
};


struct PerceptronLayer3DBackPropagation : LayerBackPropagation
{
    // Default constructor

    explicit PerceptronLayer3DBackPropagation() : LayerBackPropagation()
    {

    }


    explicit PerceptronLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    virtual ~PerceptronLayer3DBackPropagation()
    {
    }


    pair<type*, dimensions> get_deltas() const final
    {
        const Index neurons_number = layer_pointer->get_neurons_number();

        return pair<type*, dimensions>(deltas_data, {{batch_samples_number, neurons_number}});
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer) final
    {
        layer_pointer = new_layer_pointer;

        batch_samples_number = new_batch_samples_number;

        const Index neurons_number = layer_pointer->get_neurons_number();
        const Index inputs_number = layer_pointer->get_inputs_number();

        deltas.resize(batch_samples_number, neurons_number);

        deltas_data = deltas.data();

        biases_derivatives.resize(neurons_number);

        synaptic_weights_derivatives.resize(inputs_number, neurons_number);

        deltas_times_activations_derivatives.resize(batch_samples_number, neurons_number);
    }


    void print() const
    {
        cout << "Deltas:" << endl;
        cout << deltas << endl;

        cout << "Biases derivatives:" << endl;
        cout << biases_derivatives << endl;

        cout << "Synaptic weights derivatives:" << endl;
        cout << synaptic_weights_derivatives << endl;
    }

    Tensor<type, 2> deltas;

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 2> synaptic_weights_derivatives;

    Tensor<type, 2> deltas_times_activations_derivatives;
};


struct PerceptronLayer3DBackPropagationLM : LayerBackPropagationLM
{
    // Default constructor

    explicit PerceptronLayer3DBackPropagationLM() : LayerBackPropagationLM()
    {

    }


    explicit PerceptronLayer3DBackPropagationLM(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagationLM()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    virtual ~PerceptronLayer3DBackPropagationLM()
    {

    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer) final
    {
        layer_pointer = new_layer_pointer;

        batch_samples_number = new_batch_samples_number;

        const Index neurons_number = layer_pointer->get_neurons_number();
        const Index parameters_number = layer_pointer->get_parameters_number();

        deltas.resize(batch_samples_number, neurons_number);

        squared_errors_Jacobian.resize(batch_samples_number, parameters_number);
    }

    void print() const
    {
        cout << "Deltas:" << endl;
        cout << deltas << endl;

        cout << "Squared errors Jacobian: " << endl;
        cout << squared_errors_Jacobian << endl;
    }

    Tensor<type, 2> squared_errors_Jacobian;
};

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
