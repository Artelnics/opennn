//   OpenNN: Open Neural Networks Library
//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef DENSE2D_H
#define DENSE2D_H

#include "layer.h"

namespace opennn
{

class Dense2d : public Layer
{

public:

    Dense2d(const dimensions& = {0},
            const dimensions& = {0},
            const string& = "HyperbolicTangent",
            const bool& = false,
            const string& = "dense2d_layer");

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    vector<pair<type*, Index>> get_parameter_pairs() const override;

    type get_dropout_rate() const;

    bool get_batch_normalization() const;
    Tensor<type, 1> get_scales() const;
    Tensor<type, 1> get_offsets() const;

    const string& get_activation_function() const;

    void set(const dimensions& = {0},
             const dimensions& = {0},
             const string& = "HyperbolicTangent",
             const bool& = false,
             const string& = "dense2d_layer");

    void set_input_dimensions(const dimensions&) override;
    void set_output_dimensions(const dimensions&) override;

    void set_activation_function(const string&);
    void set_dropout_rate(const type&);

    void calculate_combinations(const Tensor<type, 2>&, Tensor<type, 2>&) const;

    void normalization(Tensor<type, 1>&, Tensor<type, 1>&, const Tensor<type, 2>&, Tensor<type, 2>&) const;

    void set_batch_normalization(const bool&);
    void apply_batch_normalization(unique_ptr<LayerForwardPropagation>&, const bool&);
    void apply_batch_normalization_backward(TensorMap<Tensor<type, 2>>&, unique_ptr<LayerForwardPropagation>&, unique_ptr<LayerBackPropagation>&) const;

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void back_propagate_lm(const vector<pair<type*, dimensions>>&,
                           const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           unique_ptr<LayerBackPropagationLM>&) const override;

    void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>&,
                                           const Index&,
                                           Tensor<type, 2>&) const override;

    string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

    void print() const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA

public:

    void forward_propagate_cuda(const vector<float*>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                const bool&) override;

    void back_propagate_cuda(const vector<float*>&,
                             const vector<float*>&,
                             unique_ptr<LayerForwardPropagationCuda>&,
                             unique_ptr<LayerBackPropagationCuda>&) const override;

    vector<pair<float*, Index>> get_parameter_pair_device() const override;

    void copy_parameters_host();

    void copy_parameters_device();

    void allocate_parameters_device();

    void free_parameters_device();

    bool use_combinations = true;

private:

    float* biases_device = nullptr;
    float* weights_device = nullptr;
    cudnnActivationDescriptor_t activation_descriptor = nullptr;

    // Batch Normalization
    float* bn_scale_device = nullptr;
    float* bn_offset_device = nullptr;
    float* bn_running_mean_device = nullptr;
    float* bn_running_variance_device = nullptr;
    cudnnTensorDescriptor_t bn_tensor_descriptor = nullptr;

#endif

private:

    Tensor<type, 1> biases;

    Tensor<type, 2> weights;

    bool batch_normalization = false;

    Tensor<type, 1> scales;
    Tensor<type, 1> offsets;

    Tensor<type, 1> moving_means;
    Tensor<type, 1> moving_standard_deviations;

    type momentum = type(0.9);
    const type epsilon = type(1e-5);

    string activation_function = "HyperbolicTangent";

    type dropout_rate = type(0);
};


struct Dense2dForwardPropagation : LayerForwardPropagation
{
    Dense2dForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type*, dimensions> get_output_pair() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 1> means;
    Tensor<type, 1> standard_deviations;
    Tensor<type, 2> normalized_outputs;

    Tensor<type, 2> outputs;

    Tensor<type, 2> activation_derivatives;
};


struct Dense2dBackPropagation : LayerBackPropagation
{
    Dense2dBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    vector<pair<type*, Index>> get_parameter_delta_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 2> input_deltas;

    Tensor<type, 1> bias_deltas;
    Tensor<type, 2> weight_deltas;

    Tensor<type, 1> bn_scale_deltas;
    Tensor<type, 1> bn_offset_deltas;
};


struct Dense2dLayerBackPropagationLM : LayerBackPropagationLM
{
    Dense2dLayerBackPropagationLM(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 2> input_deltas;

    Tensor<type, 2> squared_errors_Jacobian;
};


#ifdef OPENNN_CUDA

struct Dense2dForwardPropagationCuda : public LayerForwardPropagationCuda
{
    Dense2dForwardPropagationCuda(const Index& = 0, Layer* = nullptr);

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    void free() override;

    float* combinations = nullptr;

    cudnnTensorDescriptor_t output_softmax_tensor_descriptor = nullptr;

    cudnnTensorDescriptor_t biases_tensor_descriptor = nullptr;

    cudnnDropoutDescriptor_t dropout_descriptor = nullptr;
    void* dropout_states = nullptr;
    size_t dropout_states_size = 0;
    unsigned long long dropout_seed;

    void* dropout_reserve_space = nullptr;
    size_t dropout_reserve_space_size = 0;

    float* bn_saved_mean = nullptr;
    float* bn_saved_inv_variance = nullptr;
};


struct Dense2dBackPropagationCuda : public LayerBackPropagationCuda
{
    Dense2dBackPropagationCuda(const Index& = 0, Layer* = nullptr);

    vector<pair<float*, Index>> get_parameter_delta_pair_device() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    void free() override;

    float* bias_deltas_device = nullptr;
    float* weight_deltas_device = nullptr;

    float* ones = nullptr;

    cudnnTensorDescriptor_t deltas_tensor_descriptor = nullptr;

    float* bn_scale_deltas_device = nullptr;
    float* bn_offset_deltas_device = nullptr;
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
