//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PROBABILISTICLAYER_H
#define PROBABILISTICLAYER_H

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
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"
#include "layer_back_propagation_lm.h"

namespace opennn
{

struct ProbabilisticLayerForwardPropagation;
struct ProbabilisticLayerBackPropagation;
struct ProbabilisticLayerBackPropagationLM;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/struct_probabilistic_layer_cuda.h"
#endif


/// This class represents a layer of probabilistic neurons.

///
/// The neural network defined in OpenNN includes a probabilistic layer for those problems
/// when the outptus are to be interpreted as probabilities.
/// It does not has Synaptic weights or Biases

class ProbabilisticLayer : public Layer
{

public:

   // Constructors

   explicit ProbabilisticLayer();

   explicit ProbabilisticLayer(const Index&, const Index&);

   // Enumerations

   /// Enumeration of the available methods for interpreting variables as probabilities.

   enum class ActivationFunction{Binary, Logistic, Competitive, Softmax};

   // Get methods

   Index get_inputs_number() const override;
   Index get_neurons_number() const final;

   Index get_biases_number() const;
   Index get_synaptic_weights_number() const;

   const type& get_decision_threshold() const;

   const ActivationFunction& get_activation_function() const;
   string write_activation_function() const;
   string write_activation_function_text() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&);
   void set(const ProbabilisticLayer&);

   void set_name(const string&);

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;

   void set_biases(const Tensor<type, 1>&);
   void set_synaptic_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& index=0) final;
   void set_decision_threshold(const type&);

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   void set_default();

   // Parameters

   const Tensor<type, 1>& get_biases() const;
   const Tensor<type, 2>& get_synaptic_weights() const;

//   Tensor<type, 1> get_biases(Tensor<type, 1>&) const;
//   Tensor<type, 2> get_synaptic_weights(Tensor<type, 1>&) const;   

   Index get_parameters_number() const final;
   Tensor<type, 1> get_parameters() const final;

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods

   void set_biases_constant(const type&);
   void set_synaptic_weights_constant(const type&);
   void set_synaptic_weights_constant_Glorot();

   void set_parameters_constant(const type&) final;

   void set_parameters_random() final;

   void insert_parameters(const Tensor<type, 1>&, const Index&);

   // Forward propagation

   void calculate_combinations(const Tensor<type, 2>&,
                               const Tensor<type, 1>&,
                               const Tensor<type, 2>&,
                               Tensor<type, 2>&) const;

   void calculate_activations(const Tensor<type, 2>&,
                              Tensor<type, 2>&) const;

   void calculate_activations_derivatives(const Tensor<type, 2>&,
                                          Tensor<type, 2>&,
                                          Tensor<type, 3>&) const;

   void logistic_derivatives(const Tensor<type, 2>&,
                            Tensor<type, 2>&,
                            Tensor<type, 3>&) const;

   void competitive(const Tensor<type, 2>&, Tensor<type, 2>&) const;

   // Outputs

   void forward_propagate(const pair<type*, dimensions>&,
                          LayerForwardPropagation*,
                          const bool&) final;

   void forward_propagate(const pair<type*, dimensions>&,
                          Tensor<type, 1>&,
                          LayerForwardPropagation*) final;

   // Gradient methods

   void calculate_error_combinations_derivatives(const Tensor<type, 2>&,
                                                 const Tensor<type, 3>&,
                                                 Tensor<type, 2>&) const;

   void calculate_error_gradient(const pair<type*, dimensions>&,
                                 LayerForwardPropagation*,
                                 LayerBackPropagation*) const final;

   void insert_gradient(LayerBackPropagation*, const Index&, Tensor<type, 1>&) const final;

   // Squared errors methods

   void calculate_squared_errors_Jacobian_lm(const Tensor<type, 2>&,
                                             LayerForwardPropagation*,
                                             LayerBackPropagationLM*) final;

   void insert_squared_errors_Jacobian_lm(LayerBackPropagationLM*,
                                          const Index&,
                                          Tensor<type, 2>&) const final;

   // Expression methods

   string write_binary_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_logistic_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_competitive_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_softmax_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_no_probabilistic_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;
   string write_combinations(const Tensor<string, 1>&) const;
   string write_activations(const Tensor<string, 1>&) const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;


protected:

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Tensor<type, 1> biases;

   /// This matrix contains conection strengths from a layer's inputs to its neurons.

   Tensor<type, 2> synaptic_weights;

   /// Activation function variable.

   ActivationFunction activation_function = ActivationFunction::Logistic;

   type decision_threshold;

   /// Display messages to screen.

   bool display = true;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/probabilistic_layer_cuda.h"
#else
};
#endif


struct ProbabilisticLayerForwardPropagation : LayerForwardPropagation
{
    // Constructor

    explicit ProbabilisticLayerForwardPropagation();

    explicit ProbabilisticLayerForwardPropagation(const Index&, Layer*);

    virtual ~ProbabilisticLayerForwardPropagation();

    pair<type *, dimensions> get_outputs_pair() const final;

    void set(const Index &new_batch_samples_number,
             Layer *new_layer_pointer) final;

    void print() const;

    Tensor<type, 2> outputs;
    Tensor<type, 3> activations_derivatives;
};


struct ProbabilisticLayerBackPropagation : LayerBackPropagation
{
    explicit ProbabilisticLayerBackPropagation();

    virtual ~ProbabilisticLayerBackPropagation();

    explicit ProbabilisticLayerBackPropagation(const Index&, Layer*);

    pair<type *, dimensions> get_deltas_pair() const final;

    void set(const Index&, Layer*) final;

    void print() const;

    Tensor<type, 2> deltas;

    Tensor<type, 1> deltas_row;
    Tensor<type, 2> activations_derivatives_matrix;

    Tensor<type, 2> error_combinations_derivatives;

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 2> synaptic_weights_derivatives;
};


struct ProbabilisticLayerBackPropagationLM : LayerBackPropagationLM
{
    explicit ProbabilisticLayerBackPropagationLM() : LayerBackPropagationLM()
    {

    }


    explicit ProbabilisticLayerBackPropagationLM(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagationLM()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    virtual ~ProbabilisticLayerBackPropagationLM()
    {

    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer) final
    {
        layer_pointer = new_layer_pointer;

        batch_samples_number = new_batch_samples_number;

        const Index neurons_number = layer_pointer->get_neurons_number();
        const Index parameters_number = layer_pointer->get_parameters_number();

        deltas.resize(batch_samples_number, neurons_number);
        deltas_row.resize(neurons_number);

        squared_errors_Jacobian.resize(batch_samples_number, parameters_number);

        error_combinations_derivatives.resize(batch_samples_number, neurons_number);
    }


    void print() const
    {
        cout << "Deltas:" << endl;
        cout << deltas << endl;

        cout << "Squared errors Jacobian: " << endl;
        cout << squared_errors_Jacobian << endl;
    }

    Tensor<type, 1> deltas_row;

    Tensor<type, 2> error_combinations_derivatives;

    Tensor<type, 2> squared_errors_Jacobian;
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
