//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ProbabilisticLayer3D_H
#define ProbabilisticLayer3D_H

// System includes

//#include <cmath>
//#include <cstdlib>
//#include <fstream>
#include <iostream>
#include <string>
//#include <sstream>

// OpenNN includes

#include "config.h"

#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{

struct ProbabilisticLayer3DForwardPropagation;
struct ProbabilisticLayer3DBackPropagation;

#ifdef OPENNN_CUDA
struct ProbabilisticLayer3DForwardPropagationCuda;
struct ProbabilisticLayer3DBackPropagationCuda;
#endif


/// This class represents a layer of probabilistic neurons.

///
/// The neural network defined in OpenNN includes a probabilistic layer for those problems
/// when the outptus are to be interpreted as probabilities.
/// It does not has Synaptic weights or Biases

class ProbabilisticLayer3D : public Layer
{

public:

   // Constructors

   explicit ProbabilisticLayer3D();

   explicit ProbabilisticLayer3D(const Index&, const Index&, const Index&);

   // Enumerations

   /// Enumeration of the available methods for interpreting variables as probabilities.

   enum class ActivationFunction{Softmax, Competitive};

   // Get methods

   Index get_inputs_number() const final;
   Index get_inputs_depth() const;
   Index get_neurons_number() const final;

   dimensions get_output_dimensions() const final;

   Index get_biases_number() const;
   Index get_synaptic_weights_number() const;

   const type& get_decision_threshold() const;

   const ActivationFunction& get_activation_function() const;
   string write_activation_function() const;
   string write_activation_function_text() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&, const Index&);
   void set(const ProbabilisticLayer3D&);

   void set_name(const string&);

   void set_inputs_number(const Index&) final;
   void set_inputs_depth(const Index&);
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
   void set_parameters_glorot();

   void insert_parameters(const Tensor<type, 1>&, const Index&);

   // Forward propagation

   void calculate_combinations(const Tensor<type, 3>&,
                               Tensor<type, 3>&) const;

   void calculate_activations(const Tensor<type, 3>&,
                              Tensor<type, 3>&) const;

   // Outputs

   void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                          LayerForwardPropagation*,
                          const bool&) final;

   // Gradient methods

   void back_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                                 const Tensor<pair<type*, dimensions>, 1>&,
                                 LayerForwardPropagation*,
                                 LayerBackPropagation*) const final;

   void calculate_error_combinations_derivatives(const Tensor<type, 3>&,
                                                 const Tensor<type, 2>&,
                                                 const Tensor<type, 2>&,
                                                 Tensor<type, 3>&) const;

   void insert_gradient(LayerBackPropagation*, 
                        const Index&, 
                        Tensor<type, 1>&) const final;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
       #include "../../opennn_cuda/opennn_cuda/probabilistic_layer_3d_cuda.h"
    #endif

protected:

   Index inputs_number;

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Tensor<type, 1> biases;

   /// This matrix contains conection strengths from a layer's inputs to its neurons.

   Tensor<type, 2> synaptic_weights;

   /// Activation function variable.

   ActivationFunction activation_function = ActivationFunction::Softmax;

   type decision_threshold;

   /// Display messages to screen.

   bool display = true;

   const Eigen::array<IndexPair<Index>, 1> contraction_indices = { IndexPair<Index>(2, 0) };

};


struct ProbabilisticLayer3DForwardPropagation : LayerForwardPropagation
{
    // Constructor

    explicit ProbabilisticLayer3DForwardPropagation() : LayerForwardPropagation()
    {
    }


    // Constructor

    explicit ProbabilisticLayer3DForwardPropagation(const Index new_batch_samples_number, Layer* new_layer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    virtual ~ProbabilisticLayer3DForwardPropagation()
    {
    }
    
    
    pair<type*, dimensions> get_outputs_pair() const final;


    void set(const Index& new_batch_samples_number, Layer* new_layer) final;


    void print() const
    {
        cout << "Outputs:" << endl;
        cout << outputs << endl;
    }

    Tensor<type, 3> outputs;
};


struct ProbabilisticLayer3DBackPropagation : LayerBackPropagation
{
    explicit ProbabilisticLayer3DBackPropagation() : LayerBackPropagation()
    {

    }

    virtual ~ProbabilisticLayer3DBackPropagation()
    {
    }


    explicit ProbabilisticLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }

    void set(const Index& new_batch_samples_number, Layer* new_layer) final;


    void print() const
    {
        cout << "Biases derivatives:" << endl;
        cout << biases_derivatives << endl;

        cout << "Synaptic weights derivatives:" << endl;
        cout << synaptic_weights_derivatives << endl;
    }

    Tensor<type, 2> targets;
    Tensor<type, 2> mask;
    bool built_mask = false;

    Tensor<type, 3> error_combinations_derivatives;
    Tensor<type, 3> input_derivatives;

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 2> synaptic_weights_derivatives;
};

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/probabilistic_layer_3d_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/probabilistic_layer_3d_back_propagation_cuda.h"
#endif

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
