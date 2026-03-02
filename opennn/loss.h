//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "neural_network.h"

namespace opennn
{

class Dataset;

struct Batch;
struct ForwardPropagation;
struct BackPropagation;
struct BackPropagationLM;

#ifdef OPENNN_CUDA
struct BatchCuda;
struct BackPropagationCuda;
#endif


class Loss
{

public:

    Loss(const NeuralNetwork* = nullptr, const Dataset* = nullptr);
    virtual ~Loss() = default;

    enum class RegularizationMethod{L1, L2, ElasticNet, NoRegularization};

    inline NeuralNetwork* get_neural_network() const
    {
        return neural_network;
    }

    inline Dataset* get_dataset() const
    {
        return dataset;
    }

    type get_regularization_weight() const;

    bool get_display() const;

    bool has_neural_network() const;

    bool has_dataset() const;

    string get_regularization_method() const;

    void set(const NeuralNetwork* = nullptr, const Dataset* = nullptr);

    void set_neural_network(const NeuralNetwork*);

    virtual void set_dataset(const Dataset*);

    void set_regularization_method(const string&);
    void set_regularization_weight(const type);

    void set_display(bool);

    virtual void set_normalization_coefficient() {}

    virtual type get_Minkowski_parameter() const { return 1.5; }

    // Back propagation

    virtual void calculate_error(const Batch&,
                                 const ForwardPropagation&,
                                 BackPropagation&) const = 0;

    void add_regularization(BackPropagation&) const;
    void add_regularization_lm(BackPropagationLM&) const;

    virtual void calculate_output_gradients(const Batch&,
                                        ForwardPropagation&,
                                        BackPropagation&) const = 0;

    void calculate_layers_error_gradient(const Batch&,
                                         ForwardPropagation&,
                                         BackPropagation&) const;

    void add_regularization_gradient(VectorR&) const;

    void add_regularization_to_gradients(BackPropagation&) const;

    void back_propagate(const Batch&,
                        ForwardPropagation&,
                        BackPropagation&) const;

    // Back propagation LM

    void calculate_errors_lm(const Batch&,
                             const ForwardPropagation&,
                             BackPropagationLM&) const;

    virtual void calculate_squared_errors_lm(const Batch&,
                                             const ForwardPropagation&,
                                             BackPropagationLM&) const;

    virtual void calculate_error_lm(const Batch&,
                                    const ForwardPropagation&,
                                    BackPropagationLM&) const {}

    virtual void calculate_output_gradients_lm(const Batch&,
                                               ForwardPropagation&,
                                               BackPropagationLM&) const {}

    void calculate_layers_squared_errors_jacobian_lm(const Batch&,
                                                     ForwardPropagation&,
                                                     BackPropagationLM&) const;

    virtual void calculate_error_gradient_lm(const Batch&,
                                             BackPropagationLM&) const;

    virtual void calculate_error_hessian_lm(const Batch&,
                                            BackPropagationLM&) const {}

    void back_propagate_lm(const Batch&,
                           ForwardPropagation&,
                           BackPropagationLM&) const;

    // Regularization

    type calculate_regularization(const VectorR&) const;

    // Serialization

    virtual void from_XML(const XMLDocument&) = 0;

    virtual void to_XML(XMLPrinter&) const;

    void regularization_from_XML(const XMLDocument&);
    void write_regularization_XML(XMLPrinter&) const;

    string get_name() const;

    // Numerical differentiation

    static type calculate_h(const type);

    type calculate_numerical_error() const;

    VectorR calculate_gradient();

    VectorR calculate_numerical_gradient();
    VectorR calculate_numerical_gradient_lm();
    MatrixR calculate_numerical_jacobian();
    VectorR calculate_numerical_input_gradients();
    MatrixR calculate_numerical_hessian();
    MatrixR calculate_inverse_hessian();

#ifdef OPENNN_CUDA

public:

    virtual void calculate_error(const BatchCuda&,
                                      const ForwardPropagationCuda&,
                                      BackPropagationCuda&) const = 0;

    virtual void calculate_output_gradients(const BatchCuda&,
                                             ForwardPropagationCuda&,
                                             BackPropagationCuda&) const = 0;

    void calculate_layers_error_gradient_cuda(const BatchCuda&,
                                              ForwardPropagationCuda&,
                                              BackPropagationCuda&) const;

    void back_propagate(const BatchCuda&,
                             ForwardPropagationCuda&,
                             BackPropagationCuda&);

    void add_regularization_cuda(BackPropagationCuda&) const;

protected:

    const float alpha = 1.0f;
    const float beta = 0.0f;

#endif

    void print(){}

protected:

    NeuralNetwork* neural_network = nullptr;

    Dataset* dataset = nullptr;

    string regularization_method = "L2";

    type regularization_weight = type(0.01);

    bool display = true;

    string name;
};


struct BackPropagationLM
{
    BackPropagationLM(const Index = 0, Loss* = nullptr);
    virtual ~BackPropagationLM() = default;

    void set(const Index = 0, Loss* = nullptr);

    void print() const;

    TensorView get_output_gradients() const;

    vector<vector<TensorView>> get_layer_gradients() const;

    Index samples_number = 0;

    VectorR output_gradients;
    Shape output_gradient_dimensions;

    Loss* loss_index = nullptr;

    type error;
    type regularization = type(0);
    type loss = type(0);

    NeuralNetworkBackPropagationLM neural_network;

    VectorR errors;
    VectorR squared_errors;
    MatrixR squared_errors_jacobian;

    VectorR gradient;
    MatrixR hessian;

    VectorR regularization_gradient;
    MatrixR regularization_hessian;
};


struct BackPropagation
{
    BackPropagation(const Index = 0, const Loss* = nullptr);
    virtual ~BackPropagation() = default;

    void set(const Index = 0, const Loss* = nullptr);

    vector<vector<TensorView>> get_layer_gradients() const;

    TensorView get_output_gradients() const;

    void print() const;

    Index samples_number = 0;

    Loss* loss_index = nullptr;

    NeuralNetworkBackPropagation neural_network;

    type error;
    MatrixR errors;
    MatrixR errors_weights;
    VectorR output_gradients;
    Shape output_gradient_dimensions;

    Tensor0 accuracy;
    MatrixR predictions;

    MatrixB matches;
    MatrixB mask;

    bool built_mask = false;
    type loss = type(0);
};


#ifdef OPENNN_CUDA

struct BackPropagationCuda
{
    BackPropagationCuda(const Index = 0, Loss* = nullptr);

    ~BackPropagationCuda() { free(); }

    void set(const Index = 0, Loss* = nullptr);

    vector<vector<TensorViewCuda>> get_layer_delta_views_device() const;

    TensorViewCuda get_output_gradient_views_device() const;

    void print() const;

    void free();

    Index samples_number = 0;

    Loss* loss_index = nullptr;

    NeuralNetworkBackPropagationCuda neural_network;

    float* errors = nullptr;

    type error;
    float* error_device = nullptr;

    type regularization = type(0);
    type loss = type(0);

    cudnnReduceTensorDescriptor_t reduce_tensor_descriptor;

    void* workspace = nullptr;
    size_t workspace_size = 0;

    cudnnTensorDescriptor_t output_reduce_tensor_descriptor = nullptr;

    TensorCuda output_gradients;

    Tensor0 accuracy;
    float* predictions = nullptr;
    float* matches = nullptr;
    float* mask = nullptr;
    bool built_mask = false;
};

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
