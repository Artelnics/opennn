//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PROBABILISTICLAYER_H
#define PROBABILISTICLAYER_H

#include "layer.h"

namespace opennn
{

class ProbabilisticLayer : public Layer
{

public:

    enum class Activation { Binary, Logistic, Competitive, Softmax };

    ProbabilisticLayer(const dimensions& = {0},
                       const dimensions& = {0},
                       const string& = "probabilistic_layer");

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    const type& get_decision_threshold() const;

    const Activation& get_activation_function() const;
    string get_activation_function_string() const;

    Index get_parameters_number() const override;
    Tensor<type, 1> get_parameters() const override;

    void set(const dimensions& = {0},
             const dimensions & = {0},
             const string& = "probabilistic_layer");

    void set_input_dimensions(const dimensions&) override;
    void set_output_dimensions(const dimensions&) override;

    void set_parameters(const Tensor<type, 1>&, Index&) override;
    void set_decision_threshold(const type&);

    void set_activation_function(const Activation&);
    void set_activation_function(const string&);

    void set_parameters_constant(const type&) override;
    void set_parameters_random() override;

    void calculate_combinations(const Tensor<type, 2>&,
                                Tensor<type, 2>&) const;

    void calculate_activations(Tensor<type, 2>&,Tensor<type, 2>&) const;

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void insert_gradient(unique_ptr<LayerBackPropagation>&,
                         Index&,
                         Tensor<type, 1>&) const override;

    void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>&,
                                           const Index&,
                                           Tensor<type, 2>&) const override;

    string write_binary_expression(const vector<string>&, const vector<string>&) const;
    string write_logistic_expression(const vector<string>&, const vector<string>&) const;
    string write_competitive_expression(const vector<string>&, const vector<string>&) const;
    string write_softmax_expression(const vector<string>&, const vector<string>&) const;

    string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;
    string write_combinations(const vector<string>&) const;
    string write_activations(const vector<string>&) const;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    void print() const override;

private:

    Tensor<type, 1> biases;

    Tensor<type, 2> weights;

    Activation activation_function = Activation::Logistic;

    type decision_threshold;

#ifdef OPENNN_CUDA_test

public:

    void forward_propagate_cuda(const vector<pair<type*, dimensions>>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                const bool&) override;

    void back_propagate_cuda(const vector<pair<type*, dimensions>>&,
                             const vector<pair<type*, dimensions>>&,
                             unique_ptr<LayerForwardPropagationCuda>&,
                             unique_ptr<LayerBackPropagationCuda>&) const override;

    void insert_gradient_cuda(unique_ptr<LayerBackPropagationCuda>&,
                              Index&,
                              float*) const override;

    void set_parameters_cuda(const float*, Index&);

    void get_parameters_cuda(const Tensor<type, 1>&, const Index&);

    void copy_parameters_host();

    void copy_parameters_device();

    void allocate_parameters_device();

    void free_parameters_device();

    float* get_weights_device() const;
    float* get_biases_device() const;

protected:

    float* biases_device = nullptr;
    float* weights_device = nullptr;

#endif

};


struct ProbabilisticLayerForwardPropagation : LayerForwardPropagation
{
    ProbabilisticLayerForwardPropagation(const Index & = 0, Layer* = nullptr);

    pair<type*, dimensions> get_outputs_pair() const override;

    void set(const Index & = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 2> outputs;
    Tensor<type, 2> activation_derivatives;
};


struct ProbabilisticLayerBackPropagation : LayerBackPropagation
{
    ProbabilisticLayerBackPropagation(const Index & = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index & = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 1> bias_derivatives;
    Tensor<type, 2> weight_derivatives;

    Tensor<type, 2> input_derivatives;
};


struct ProbabilisticLayerBackPropagationLM : LayerBackPropagationLM
{
    ProbabilisticLayerBackPropagationLM(const Index & = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index & = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 2> input_derivatives;

    Tensor<type, 2> squared_errors_Jacobian;
};


#ifdef OPENNN_CUDA_test

struct ProbabilisticLayerForwardPropagationCuda : public LayerForwardPropagationCuda
{
    explicit ProbabilisticLayerForwardPropagationCuda(const Index& = 0, Layer* = nullptr);

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    void free() override;

    pair<type*, dimensions> get_outputs_pair() const override;

    cudnnTensorDescriptor_t outputs_softmax_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t outputs_batch_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t biases_batch_tensor_descriptor = nullptr;

    cudnnActivationDescriptor_t activation_descriptor = nullptr;

    float* combinations = nullptr;
};


struct ProbabilisticLayerBackPropagationCuda : public LayerBackPropagationCuda
{
    explicit ProbabilisticLayerBackPropagationCuda(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs_device() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    void free() override;

    float* targets = nullptr;
    float* error_combinations_derivatives_cuda = nullptr;
    float* ones = nullptr;
    float* biases_derivatives_cuda = nullptr;
    float* weights_derivatives_cuda = nullptr;
    float* inputs_derivatives = nullptr;
    float one = 1.0f;

    cudnnOpTensorDescriptor_t operator_sum_descriptor = nullptr;
    cudnnTensorDescriptor_t error_combinations_derivatives_tensor_descriptor = nullptr;
};

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
