//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "neural_network.h"
#include "back_propagation.h"

namespace opennn
{

class Dataset;

struct Batch;
struct ForwardPropagation;

class Loss
{

public:

    enum class Error{MeanSquaredError,
                     NormalizedSquaredError,
                     WeightedSquaredError,
                     CrossEntropy,
                     CrossEntropy3d,
                     MinkowskiError};

    enum class Regularization{L1, L2, ElasticNet, NoRegularization};

    static const EnumMap<Regularization>& regularization_map()
    {
        static const vector<pair<Regularization, string>> entries = {
            {Regularization::NoRegularization, "None"},
            {Regularization::L1,               "L1"},
            {Regularization::L2,               "L2"},
            {Regularization::ElasticNet,       "ElasticNet"}
        };
        static const EnumMap<Regularization> map{entries};
        return map;
    }

    static const string& regularization_to_string(Regularization regularization)
    {
        return regularization_map().to_string(regularization);
    }

    static Regularization string_to_regularization(const string& name)
    {
        if (name == "NoRegularization") return Regularization::NoRegularization;
        return regularization_map().from_string(name);
    }

    Loss(NeuralNetwork* = nullptr, Dataset* = nullptr);

    virtual ~Loss() = default;

    const NeuralNetwork* get_neural_network() const
    {
        return neural_network;
    }

    NeuralNetwork* get_neural_network()
    {
        return neural_network;
    }

    const Dataset* get_dataset() const
    {
        return dataset;
    }

    Dataset* get_dataset()
    {
        return dataset;
    }

    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void set_neural_network(NeuralNetwork* new_neural_network) { neural_network = new_neural_network; }

    virtual void set_dataset(Dataset* new_dataset) { dataset = new_dataset; }

    void set_regularization(const string& new_regularization_method) { regularization_method = string_to_regularization(new_regularization_method); }
    void set_regularization(Regularization new_regularization) { regularization_method = new_regularization; }
    void set_regularization_weight(const float new_regularization_weight) { regularization_weight = new_regularization_weight; }

    void set_normalization_coefficient();

    // Back propagation

    void calculate_error(const Batch&,
                         const ForwardPropagation&,
                         BackPropagation&) const;

    void set_error(const Error&);
    void set_error(const string&);

    Error get_error() const { return error; }

    void back_propagate(const Batch&,
                        ForwardPropagation&,
                        BackPropagation&) const;

    // Regularization

    float calculate_regularization(const VectorR&) const;

    // Serialization

    void from_JSON(const JsonDocument&);

    void to_JSON(JsonWriter&) const;

    void regularization_from_JSON(const JsonDocument&);
    void regularization_to_JSON(JsonWriter&) const;

    const string& get_name() const { return name; }

    // Used by Levenberg–Marquardt for numerical jacobian/hessian. Other
    // numerical-differentiation helpers (calculate_numerical_gradient et al.)
    // live in tests/numerical_derivatives.{h,cpp} as free functions.
    static float calculate_h(const float);

    void print() const {}

private:

    void check_neural_network() const
    {
        if (!neural_network)
            throw runtime_error("Loss error: neural network is not set.");
    }

    void check_dataset() const
    {
        if (!dataset)
            throw runtime_error("Loss error: dataset is not set.");
    }

    void add_regularization(BackPropagation&) const;

    float get_weighted_coefficient(const Batch&) const;

    void calculate_layers_error_gradient(const Batch&,
                                         ForwardPropagation&,
                                         BackPropagation&) const;

    void add_regularization_gradient(BackPropagation&) const;

    void calculate_output_deltas(const Batch&,
                                    const ForwardPropagation&,
                                    BackPropagation&) const;

protected:

    Error error = Error::MeanSquaredError;

    // Method-specific parameters
    float normalization_coefficient = 1.0f;
    float positives_weight = 1.0f;
    float negatives_weight = 1.0f;
    float minkowski_parameter = 1.5f;

    // Regularization
    Regularization regularization_method = Regularization::L2;
    float regularization_weight = 0.001f;

    NeuralNetwork* neural_network = nullptr;
    Dataset* dataset = nullptr;

    string name = "Loss";
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
