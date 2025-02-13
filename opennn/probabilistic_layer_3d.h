//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ProbabilisticLayer3D_H
#define ProbabilisticLayer3D_H

#include "layer.h"

namespace opennn
{

#ifdef OPENNN_CUDA
struct ProbabilisticLayer3DForwardPropagationCuda;
struct ProbabilisticLayer3DBackPropagationCuda;
#endif


class ProbabilisticLayer3D : public Layer
{

public:

   enum class ActivationFunction { Softmax, Competitive };

   ProbabilisticLayer3D(const Index& = 0, 
                        const Index& = 0, 
                        const Index& = 0,
                        const string& = "probabilistic_layer_3d");

   Index get_inputs_number_xxx() const;
   Index get_inputs_depth() const;
   Index get_neurons_number() const;

   // @todo

   dimensions get_input_dimensions() const override
   {
       throw runtime_error("XXX");
   }


   dimensions get_output_dimensions() const override;

   const ActivationFunction& get_activation_function() const;
   string get_activation_function_string() const;
   string get_activation_function_text() const;

   void set(const Index& = 0, const Index& = 0, const Index& = 0, const string& = "probabilistic_layer_3d");

   void set_inputs_number(const Index);
   void set_input_dimensions(const dimensions&) override;

   void set_inputs_depth(const Index&);
   void set_output_dimensions(const dimensions&) override;

   void set_parameters(const Tensor<type, 1>&, const Index& index = 0) override;

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   // Parameters

   Index get_parameters_number() const override;
   Tensor<type, 1> get_parameters() const override;

   // Parameters initialization

   void set_parameters_constant(const type&) override;
   void set_parameters_random() override;
   void set_parameters_glorot();

   // Forward propagation

   void calculate_combinations(const Tensor<type, 3>&,
                               Tensor<type, 3>&) const;

   void calculate_activations(Tensor<type, 3>&) const;

   // Outputs

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>&,
                          const bool&) override;

   // Gradient

   void back_propagate(const vector<pair<type*, dimensions>>&,
                       const vector<pair<type*, dimensions>>&,
                       unique_ptr<LayerForwardPropagation>&,
                       unique_ptr<LayerBackPropagation>&) const override;

   void calculate_combinations_derivatives(const Tensor<type, 3>&,
                                                 const Tensor<type, 2>&,
                                                 const Tensor<type, 2>&,
                                                 Tensor<type, 3>&) const;

   void insert_gradient(unique_ptr<LayerBackPropagation>&,
                        const Index&, 
                        Tensor<type, 1>&) const override;

   // Serialization

   void from_XML(const XMLDocument&) override;
   void to_XML(XMLPrinter&) const override;

    #ifdef OPENNN_CUDA
       #include "../../opennn_cuda/opennn_cuda/probabilistic_layer_3d_cuda.h"
    #endif

private:

   Index inputs_number_xxx;

   Tensor<type, 1> biases;

   Tensor<type, 2> synaptic_weights;

   ActivationFunction activation_function = ActivationFunction::Softmax;

   Tensor<type, 3> empty;

   const Eigen::array<IndexPair<Index>, 2> double_contraction_indices = { IndexPair<Index>(0, 0), IndexPair<Index>(1, 1) };
   const Eigen::array<IndexPair<Index>, 1> single_contraction_indices = { IndexPair<Index>(2, 1) };

   const Eigen::array<IndexPair<Index>, 1> contraction_indices = { IndexPair<Index>(2, 0) };
   const Eigen::array<Index, 2> sum_dimensions = { 0, 1 };
};


struct ProbabilisticLayer3DForwardPropagation : LayerForwardPropagation
{
    ProbabilisticLayer3DForwardPropagation(const Index& = 0, Layer* = nullptr);
    
    pair<type*, dimensions> get_outputs_pair() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 3> outputs;
};


struct ProbabilisticLayer3DBackPropagation : LayerBackPropagation
{
    ProbabilisticLayer3DBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 2> targets;
    Tensor<type, 2> mask;
    bool built_mask = false;

    Tensor<type, 3> combination_derivatives;
    Tensor<type, 3> input_derivatives;

    Tensor<type, 1> bias_derivatives;
    Tensor<type, 2> synaptic_weight_derivatives;
};

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/probabilistic_layer_3d_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/probabilistic_layer_3d_back_propagation_cuda.h"
#endif

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
