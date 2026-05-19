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

/// @brief Unified loss container supporting MSE, cross-entropy, Minkowski, weighted, and regularized variants.
class Loss
{

public:

    /// @brief Error function selector used to dispatch the loss kernel.
    enum class Error{MeanSquaredError,
                     NormalizedSquaredError,
                     WeightedSquaredError,
                     CrossEntropy,
                     CrossEntropy3d,
                     MinkowskiError};

    /// @brief Parameter regularization method applied on top of the base loss.
    enum class Regularization{L1, L2, ElasticNet, NoRegularization};

    /// @brief Returns the static string<->enum map used to (de)serialize regularization types.
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

    /// @brief Returns the canonical string name for a Regularization value.
    static const string& regularization_to_string(Regularization regularization)
    {
        return regularization_map().to_string(regularization);
    }

    /// @brief Parses a regularization name (accepts both "NoRegularization" and "None") back to the enum.
    static Regularization string_to_regularization(const string& name)
    {
        if (name == "NoRegularization") return Regularization::NoRegularization;
        return regularization_map().from_string(name);
    }

    /// @brief Constructs a Loss bound to an optional neural network and dataset.
    /// @param neural_network Network whose outputs are scored (may be null and set later).
    /// @param dataset Dataset providing batches for error computation (may be null and set later).
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

    /// @brief Resets the bound neural network and dataset pointers.
    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void set_neural_network(NeuralNetwork* new_neural_network) { neural_network = new_neural_network; }

    virtual void set_dataset(Dataset* new_dataset) { dataset = new_dataset; }

    void set_regularization(const string& new_regularization_method) { regularization_method = string_to_regularization(new_regularization_method); }
    void set_regularization(Regularization new_regularization) { regularization_method = new_regularization; }
    void set_regularization_weight(const float new_regularization_weight) { regularization_weight = new_regularization_weight; }

    /// @brief Recomputes the normalization coefficient (used by NormalizedSquaredError) from the dataset.
    void set_normalization_coefficient();

    /// @brief Result of calculate_error; accuracy and active_tokens_count are populated only by classification losses.
    struct EvaluationResult
    {
        float error = 0.0f;
        float accuracy = 0.0f;
        Index active_tokens_count = 0;
    };

    /// @brief Computes the loss for one batch using the cached forward-pass outputs.
    /// @param batch The minibatch containing inputs and targets.
    /// @param forward_propagation Cached forward-pass state for this batch.
    /// @return Loss value and, for classification losses, accuracy and active-token count.
    EvaluationResult calculate_error(const Batch&,
                                     const ForwardPropagation&) const;

    /// @brief Selects the error function variant.
    void set_error(const Error&);
    /// @brief Selects the error function variant from its string name.
    void set_error(const string&);

    Error get_error() const { return error; }

    /// @brief Performs the full backward pass: output deltas, layer gradients, and regularization gradient.
    /// @param batch The minibatch containing inputs and targets.
    /// @param forward_propagation Cached forward-pass state (may be updated for autograd-style layers).
    /// @param back_propagation Output structure receiving error and parameter gradients.
    void back_propagate(const Batch&,
                        ForwardPropagation&,
                        BackPropagation&) const;

#ifdef OPENNN_HAS_CUDA
    /// @brief Whether the loss supports device-side accumulation of per-epoch error and accuracy metrics.
    bool supports_device_epoch_metrics() const;

    /// @brief Backpropagates and accumulates per-batch error/accuracy directly on the GPU.
    bool back_propagate_device_metrics(const Batch&,
                                       ForwardPropagation&,
                                       BackPropagation&,
                                       float* error_sum_device,
                                       float* accuracy_sum_device) const;

    /// @brief Evaluates the loss and accumulates per-batch error/accuracy directly on the GPU.
    bool calculate_error_device_metrics(const Batch&,
                                        const ForwardPropagation&,
                                        float* error_sum_device,
                                        float* accuracy_sum_device) const;
#endif

    /// @brief Returns the regularization penalty (L1, L2, or ElasticNet) for the given parameter vector.
    float calculate_regularization(const VectorR&) const;
    /// @brief Restores loss configuration (error type, regularization, weights) from a JSON document.
    void from_JSON(const JsonDocument&);

    /// @brief Serializes the loss configuration (error type, regularization, weights) to JSON.
    void to_JSON(JsonWriter&) const;

    /// @brief Restores the regularization sub-configuration from JSON.
    void regularization_from_JSON(const JsonDocument&);
    /// @brief Serializes the regularization sub-configuration to JSON.
    void regularization_to_JSON(JsonWriter&) const;

    const string& get_name() const { return name; }
    /// @brief Returns the finite-difference step size h tuned for the given parameter value.
    static float calculate_h(const float);

    /// @brief Prints a human-readable description of the loss (no-op default).
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

    void back_propagate_layers(ForwardPropagation&,
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

#ifdef OPENNN_HAS_CUDA
    // Reduction workspace for cublasSasum/cudaMemcpy used by the loss kernels.
    // mutable because calculate_error grows it lazily on the first call; the
    // method's logical contract is still const (no observable state changes).
    mutable Buffer errors_device{Device::CUDA};
    mutable Buffer metric_results_device{Device::CUDA};
#endif

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
