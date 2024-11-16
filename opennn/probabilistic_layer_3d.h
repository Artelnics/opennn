//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ProbabilisticLayer3D_H
#define ProbabilisticLayer3D_H

#include <iostream>
#include <string>

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{

#ifdef OPENNN_CUDA
struct ProbabilisticLayer3DForwardPropagationCuda;
struct ProbabilisticLayer3DBackPropagationCuda;
#endif


class ProbabilisticLayer3D : public Layer
{

public:

   explicit ProbabilisticLayer3D(const Index& = 0, const Index& = 0, const Index& = 0);

   enum class ActivationFunction{Softmax, Competitive, Binary, Logistic};

   Index get_inputs_number() const;
   Index get_inputs_depth() const;
   Index get_neurons_number() const;

   dimensions get_output_dimensions() const final;

   const type& get_decision_threshold() const;

   const ActivationFunction& get_activation_function() const;
   string get_activation_function_string() const;
   string get_activation_function_string_text() const;

   void set(const Index& = 0, const Index& = 0, const Index& = 0);

   void set_input_dimensions(const dimensions&) final;
   void set_inputs_depth(const Index&);
   void set_output_dimensions(const dimensions&) final;

   void set_parameters(const Tensor<type, 1>&, const Index& index = 0) final;
   void set_decision_threshold(const type&);

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   // Parameters

   Index get_parameters_number() const final;
   Tensor<type, 1> get_parameters() const final;

   // Parameters initialization

   void set_parameters_constant(const type&) final;
   void set_parameters_random() final;
   void set_parameters_glorot();

   // Forward propagation

   void calculate_combinations(const Tensor<type, 3>&,
                               Tensor<type, 3>&) const;

   void calculate_activations(Tensor<type, 3>&) const;

   // Outputs

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>&,
                          const bool&) final;

   // Gradient

   void back_propagate(const vector<pair<type*, dimensions>>&,
                       const vector<pair<type*, dimensions>>&,
                       unique_ptr<LayerForwardPropagation>&,
                       unique_ptr<LayerBackPropagation>&) const final;

   void calculate_combinations_derivatives(const Tensor<type, 3>&,
                                                 const Tensor<type, 2>&,
                                                 const Tensor<type, 2>&,
                                                 Tensor<type, 3>&) const;

   void insert_gradient(unique_ptr<LayerBackPropagation>&,
                        const Index&, 
                        Tensor<type, 1>&) const final;

   // Serialization

   void from_XML(const XMLDocument&) final;
   void to_XML(XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
       #include "../../opennn_cuda/opennn_cuda/probabilistic_layer_3d_cuda.h"
    #endif

private:

   Index inputs_number;

   Tensor<type, 1> biases;

   Tensor<type, 2> synaptic_weights;

   ActivationFunction activation_function = ActivationFunction::Softmax;

   type decision_threshold;

   Tensor<type, 3> empty;

   const Eigen::array<IndexPair<Index>, 2> double_contraction_indices = { IndexPair<Index>(0, 0), IndexPair<Index>(1, 1) };
   const Eigen::array<IndexPair<Index>, 1> single_contraction_indices = { IndexPair<Index>(2, 1) };

   const Eigen::array<IndexPair<Index>, 1> contraction_indices = { IndexPair<Index>(2, 0) };
   const Eigen::array<Index, 2> sum_dimensions = { 0, 1 };
};


struct ProbabilisticLayer3DForwardPropagation : LayerForwardPropagation
{
    explicit ProbabilisticLayer3DForwardPropagation(const Index& = 0, Layer* = nullptr);
    
    pair<type*, dimensions> get_outputs_pair() const final;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 3> outputs;
};


struct ProbabilisticLayer3DBackPropagation : LayerBackPropagation
{
    explicit ProbabilisticLayer3DBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 2> targets;
    Tensor<type, 2> mask;
    bool built_mask = false;

    Tensor<type, 3> combinations_derivatives;
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
