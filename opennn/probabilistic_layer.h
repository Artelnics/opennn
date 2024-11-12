//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PROBABILISTICLAYER_H
#define PROBABILISTICLAYER_H

#include <iostream>
#include <string>

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"
#include "layer_back_propagation_lm.h"

namespace opennn
{

#ifdef OPENNN_CUDA
    struct ProbabilisticLayerForwardPropagationCuda;
    struct ProbabilisticLayerBackPropagationCuda;
#endif


struct ProbabilisticLayerForwardPropagation : LayerForwardPropagation
{
    explicit ProbabilisticLayerForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type *, dimensions> get_outputs_pair() const final;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 2> outputs;
    Tensor<type, 2> activation_derivatives;
};


struct ProbabilisticLayerBackPropagation : LayerBackPropagation
{
    explicit ProbabilisticLayerBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 2> targets;

    Tensor<type, 1> deltas_row;
    Tensor<type, 2> activations_derivatives_matrix;

    Tensor<type, 2> combinations_derivatives;

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 2> synaptic_weights_derivatives;

    Tensor<type, 2> input_derivatives;
};


struct ProbabilisticLayerBackPropagationLM : LayerBackPropagationLM
{
    explicit ProbabilisticLayerBackPropagationLM(const Index& new_batch_samples_number = 0, 
                                                 Layer* new_layer = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 1> deltas_row;

    Tensor<type, 2> combinations_derivatives;

    Tensor<type, 2> squared_errors_Jacobian;

    Tensor<type, 2> targets;
};


#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/probabilistic_layer_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/probabilistic_layer_back_propagation_cuda.h"
#endif


class ProbabilisticLayer : public Layer
{

public:

    enum class ActivationFunction { Binary, Logistic, Competitive, Softmax };

    explicit ProbabilisticLayer(const dimensions& = {0},
                                const dimensions& = {0},
                                const string& = "probabilistic_layer");

    dimensions get_input_dimensions() const final;
    dimensions get_output_dimensions() const final;

    const type& get_decision_threshold() const;

    const ActivationFunction& get_activation_function() const;
    string get_activation_function_string() const;

    Index get_parameters_number() const final;
    Tensor<type, 1> get_parameters() const final;

    void set(const dimensions& = {0},
             const dimensions & = {0},
             const string& = "probabilistic_layer");

    void set_input_dimensions(const dimensions&) final;
    void set_output_dimensions(const dimensions&) final;

    void set_parameters(const Tensor<type, 1>&, const Index& index = 0) final;
    void set_decision_threshold(const type&);

    void set_activation_function(const ActivationFunction&);
    void set_activation_function(const string&);

    void set_parameters_constant(const type&) final;
    void set_parameters_random() final;

    void calculate_combinations(const Tensor<type, 2>&,
                                Tensor<type, 2>&) const;

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) final;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const final;

    void insert_gradient(unique_ptr<LayerBackPropagation>&,
                         const Index&,
                         Tensor<type, 1>&) const final;

    void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>&,
                                           const Index&,
                                           Tensor<type, 2>&) const final;

    string write_binary_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
    string write_logistic_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
    string write_competitive_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
    string write_softmax_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

    string get_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;
    string write_combinations(const Tensor<string, 1>&) const;
    string write_activations(const Tensor<string, 1>&) const;

    void from_XML(const tinyxml2::XMLDocument&) final;
    void to_XML(tinyxml2::XMLPrinter&) const final;

    void print() const;

private:

    Tensor<type, 1> biases;

    Tensor<type, 2> synaptic_weights;

    ActivationFunction activation_function = ActivationFunction::Logistic;

    type decision_threshold;

    Tensor<type, 2> empty;

    const Eigen::array<Index, 1> sum_dimensions = {0};

#ifdef OPENNN_CUDA
#include "../../opennn_cuda/opennn_cuda/probabilistic_layer_cuda.h"
#endif

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
