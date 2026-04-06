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

#ifdef CUDA
struct BatchCuda;
struct BackPropagationCuda;
#endif

class Loss
{

public:

    enum class Error{MeanSquaredError,
                     NormalizedSquaredError,
                     WeightedSquaredError,
                     CrossEntropy,
                     MinkowskiError};

    enum class Regularization{L1,
                              L2,
                              ElasticNet,
                              NoRegularization};

    Loss(NeuralNetwork* = nullptr, Dataset* = nullptr);

    virtual ~Loss() = default;

    inline NeuralNetwork* get_neural_network() const
    {
        return neural_network;
    }

    inline Dataset* get_dataset() const
    {
        return dataset;
    }

    bool has_neural_network() const;

    bool has_dataset() const;

    const string& get_regularization_method() const;

    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void set_neural_network(NeuralNetwork*);

    virtual void set_dataset(Dataset*);

    void set_regularization(const string&);
    void set_regularization_weight(const type);

    virtual void set_normalization_coefficient() {}

    virtual type get_Minkowski_parameter() const { return 1.5; }

    // Back propagation

    void calculate_error(const Batch&,
                         const ForwardPropagation&,
                         BackPropagation&) const;

    void add_regularization(BackPropagation&) const;

    void set_error(const Error&);
    void set_error(const string&);

    void calculate_layers_error_gradient(const Batch&,
                                         ForwardPropagation&,
                                         BackPropagation&) const;

    void add_regularization_gradient(VectorR&) const;

    void add_regularization_to_gradients(BackPropagation&) const;

    void back_propagate(const Batch&,
                        ForwardPropagation&,
                        BackPropagation&) const;

    // Regularization

    type calculate_regularization(const VectorR&) const;

    // Serialization

    void from_XML(const XMLDocument&);

    void to_XML(XMLPrinter&) const;

    void regularization_from_XML(const XMLDocument&);
    void write_regularization_XML(XMLPrinter&) const;

    const string& get_name() const;

    // Numerical differentiation

    static type calculate_h(const type);

    type calculate_numerical_error() const;

    void calculate_output_gradients(const Batch&,
                                    const ForwardPropagation&,
                                    BackPropagation&) const;

    VectorR calculate_gradient();

    VectorR calculate_numerical_gradient();
    VectorR calculate_numerical_input_gradients();
    MatrixR calculate_numerical_hessian();
    MatrixR calculate_inverse_hessian();

    void print(){}

protected:

    Error error = Error::MeanSquaredError;

    // Method-specific parameters
    type normalization_coefficient = 1.0f;
    type positives_weight = 1.0f;
    type negatives_weight = 1.0f;
    type minkowski_parameter = 1.5f;

    // Regularization
    string regularization_method = "L2";
    type regularization_weight = 0.001f;

    NeuralNetwork* neural_network = nullptr;
    Dataset* dataset = nullptr;

    string name = "Loss";
};


struct BackPropagation
{
    BackPropagation(const Index = 0, Loss* = nullptr);

    virtual ~BackPropagation() = default;

    void set(const Index = 0, Loss* = nullptr);

    NeuralNetwork* get_neural_network() const;

    NeuralNetwork* neural_network = nullptr;

    VectorR gradient;
    vector<vector<TensorView>> gradient_views;

    VectorR backward;
    vector<vector<vector<TensorView>>> backward_views;

    vector<vector<TensorView>> get_layer_gradients() const;

    TensorView get_output_gradients() const;

    void print() const;

    Index batch_size = 0;

    Loss* loss = nullptr;

    type error;
    MatrixR errors;
    VectorR output_gradients;
    Shape output_gradient_dimensions;

    Tensor0 accuracy;
    MatrixR predictions;

    MatrixB matches;
    MatrixB mask;

    bool built_mask = false;
    type loss_value = type(0);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
