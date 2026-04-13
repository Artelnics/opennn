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

    static const string& regularization_to_string(Regularization r)
    {
        return regularization_map().to_string(r);
    }

    static Regularization string_to_regularization(const string& name)
    {
        if(name == "NoRegularization") return Regularization::NoRegularization;
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

    const string& get_regularization_method() const { return regularization_to_string(regularization_method); }
    Regularization get_regularization_type() const { return regularization_method; }

    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void set_neural_network(NeuralNetwork* nn) { neural_network = nn; }

    virtual void set_dataset(Dataset* ds) { dataset = ds; }

    void set_regularization(const string& m) { regularization_method = string_to_regularization(m); }
    void set_regularization(Regularization r) { regularization_method = r; }
    void set_regularization_weight(const type w) { regularization_weight = w; }

    virtual void set_normalization_coefficient() {}

    virtual type get_Minkowski_parameter() const { return minkowski_parameter; }

    // Back propagation

    void calculate_error(const Batch&,
                         const ForwardPropagation&,
                         BackPropagation&) const;

    void set_error(const Error&);
    void set_error(const string&);

    void back_propagate(const Batch&,
                        ForwardPropagation&,
                        BackPropagation&) const;

    // Regularization

    type calculate_regularization(const VectorR&) const;

    // Serialization

    void from_XML(const XmlDocument&);

    void to_XML(XmlPrinter&) const;

    void regularization_from_XML(const XmlDocument&);
    void regularization_to_XML(XmlPrinter&) const;

    const string& get_name() const { return name; }

    // Numerical differentiation

    static type calculate_h(const type);

    type calculate_numerical_error() const;

    VectorR calculate_gradient();

    VectorR calculate_numerical_gradient();
    VectorR calculate_numerical_input_gradients();
    MatrixR calculate_numerical_hessian();
    MatrixR calculate_inverse_hessian();

    void print(){}

private:

    void check_neural_network() const
    {
        if(!neural_network)
            throw runtime_error("Loss error: neural network is not set.");
    }

    void check_dataset() const
    {
        if(!dataset)
            throw runtime_error("Loss error: dataset is not set.");
    }

    void add_regularization(BackPropagation&) const;

    void calculate_layers_error_gradient(const Batch&,
                                         ForwardPropagation&,
                                         BackPropagation&) const;

    void add_regularization_gradient(BackPropagation&) const;

    void calculate_output_gradients(const Batch&,
                                    const ForwardPropagation&,
                                    BackPropagation&) const;

protected:

    Error error = Error::MeanSquaredError;

    // Method-specific parameters
    type normalization_coefficient = 1.0f;
    type positives_weight = 1.0f;
    type negatives_weight = 1.0f;
    type minkowski_parameter = 1.5f;

    // Regularization
    Regularization regularization_method = Regularization::L2;
    type regularization_weight = 0.001f;

    NeuralNetwork* neural_network = nullptr;
    Dataset* dataset = nullptr;

    string name = "Loss";
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
