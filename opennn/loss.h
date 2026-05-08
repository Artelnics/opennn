//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file loss.h
 * @brief Declares the unified Loss class plus its Error and Regularization
 *        enumerations.
 *
 * Loss couples a NeuralNetwork to a Dataset and implements forward
 * evaluation, gradient back-propagation and regularization for every
 * built-in loss function (MeanSquaredError, NormalizedSquaredError,
 * WeightedSquaredError, CrossEntropy, CrossEntropy3d, MinkowskiError).
 */

#pragma once

#include "neural_network.h"
#include "back_propagation.h"

namespace opennn
{

class Dataset;

struct Batch;
struct ForwardPropagation;

/**
 * @class Loss
 * @brief Trainable loss function attached to a NeuralNetwork and a Dataset.
 *
 * Holds non-owning pointers to a NeuralNetwork and a Dataset, the choice
 * of loss term (Loss::Error) and regularization term (Loss::Regularization).
 * Per-loss hyperparameters (e.g. minkowski_parameter, positives_weight)
 * are stored as protected fields and configured through the corresponding
 * setters.
 */
class Loss
{

public:

    /**
     * @enum Error
     * @brief Built-in loss functions.
     */
    enum class Error{MeanSquaredError,         ///< Mean of squared errors over outputs and samples.
                     NormalizedSquaredError,   ///< MSE divided by the variance of the targets.
                     WeightedSquaredError,     ///< MSE with per-class weights for imbalanced binary tasks.
                     CrossEntropy,             ///< Binary or multi-class cross entropy.
                     CrossEntropy3d,           ///< Sequence-level cross entropy used by language models.
                     MinkowskiError};          ///< Mean of |error|^p with configurable p.

    /**
     * @enum Regularization
     * @brief Parameter-norm regularization terms.
     */
    enum class Regularization{L1,                 ///< L1 norm of the parameters (sparsity).
                              L2,                 ///< L2 norm of the parameters (weight decay).
                              ElasticNet,         ///< Mix of L1 and L2 regularization.
                              NoRegularization};  ///< No regularization term.

    /**
     * @brief Returns the singleton string<->enum mapping for Regularization values.
     * @return Reference to a process-wide EnumMap initialized on first call.
     */
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

    /**
     * @brief Converts a Regularization to its canonical string name.
     * @param regularization Regularization value.
     * @return Reference to the canonical string.
     */
    static const string& regularization_to_string(Regularization regularization)
    {
        return regularization_map().to_string(regularization);
    }

    /**
     * @brief Parses a Regularization from string.
     *
     * Accepts canonical names plus the alias "NoRegularization" (also
     * encoded as "None" in the map).
     *
     * @param name String to parse.
     * @return Matching Regularization value.
     */
    static Regularization string_to_regularization(const string& name)
    {
        if (name == "NoRegularization") return Regularization::NoRegularization;
        return regularization_map().from_string(name);
    }

    /**
     * @brief Constructs a Loss bound to a network and dataset.
     * @param neural_network Network whose parameters are being trained.
     * @param dataset Dataset that provides training data.
     */
    Loss(NeuralNetwork* neural_network = nullptr, Dataset* dataset = nullptr);

    /** @brief Virtual destructor. */
    virtual ~Loss() = default;

    /** @brief Read-only access to the bound network. */
    const NeuralNetwork* get_neural_network() const
    {
        return neural_network;
    }

    /** @brief Mutable access to the bound network. */
    NeuralNetwork* get_neural_network()
    {
        return neural_network;
    }

    /** @brief Read-only access to the bound dataset. */
    const Dataset* get_dataset() const
    {
        return dataset;
    }

    /** @brief Mutable access to the bound dataset. */
    Dataset* get_dataset()
    {
        return dataset;
    }

    /**
     * @brief Re-initializes the Loss by binding network and dataset pointers.
     * @param neural_network Network whose parameters are being trained.
     * @param dataset Dataset that provides training data.
     */
    void set(NeuralNetwork* neural_network = nullptr, Dataset* dataset = nullptr);

    /**
     * @brief Sets the bound network.
     * @param new_neural_network Network whose parameters are being trained.
     */
    void set_neural_network(NeuralNetwork* new_neural_network) { neural_network = new_neural_network; }

    /**
     * @brief Sets the bound dataset; subclasses may override to refresh
     *        cached state derived from the dataset.
     * @param new_dataset Dataset that provides training data.
     */
    virtual void set_dataset(Dataset* new_dataset) { dataset = new_dataset; }

    /**
     * @brief Sets the regularization method by name.
     * @param new_regularization_method Canonical name (see Regularization).
     */
    void set_regularization(const string& new_regularization_method) { regularization_method = string_to_regularization(new_regularization_method); }
    /**
     * @brief Sets the regularization method directly.
     * @param new_regularization Regularization enum value.
     */
    void set_regularization(Regularization new_regularization) { regularization_method = new_regularization; }
    /**
     * @brief Sets the strength of the regularization term.
     * @param new_regularization_weight Multiplier applied to the
     *                                  regularization term.
     */
    void set_regularization_weight(const float new_regularization_weight) { regularization_weight = new_regularization_weight; }

    /**
     * @brief Recomputes the dataset-derived normalization coefficient.
     *
     * Called automatically after the dataset is bound; only relevant for
     * NormalizedSquaredError.
     */
    void set_normalization_coefficient();

    /**
     * @struct EvaluationResult
     * @brief Output of calculate_error().
     *
     * @ref accuracy and @ref active_tokens_count are only meaningful for
     * classification losses (CrossEntropy3d sets both; CrossEntropy with
     * binary / multi-class leaves them at zero).
     */
    struct EvaluationResult
    {
        /** @brief Error term for the batch. */
        float error = 0.0f;
        /** @brief Mean classification accuracy for the batch (0 if not applicable). */
        float accuracy = 0.0f;
        /** @brief Number of non-pad tokens (CrossEntropy3d only; 0 otherwise). */
        Index active_tokens_count = 0;
    };

    /**
     * @brief Computes the loss on a batch given its forward intermediates.
     * @param batch Current training batch.
     * @param forward_propagation Forward intermediates for the batch.
     * @return Error term, accuracy and active token count.
     */
    EvaluationResult calculate_error(const Batch& batch,
                                     const ForwardPropagation& forward_propagation) const;

    /**
     * @brief Sets the loss term directly.
     *
     * Receives the Loss::Error enum value to install.
     */
    void set_error(const Error&);
    /**
     * @brief Sets the loss term by name.
     *
     * Receives the canonical name of the loss term.
     */
    void set_error(const string&);

    /** @brief Currently selected loss term. */
    Error get_error() const { return error; }

    /**
     * @brief Computes the gradient of the loss with respect to the parameters.
     * @param batch Current training batch.
     * @param forward_propagation Forward intermediates for the batch.
     * @param back_propagation Output buffer in which to accumulate gradients.
     */
    void back_propagate(const Batch& batch,
                        ForwardPropagation& forward_propagation,
                        BackPropagation& back_propagation) const;
    /**
     * @brief Computes the regularization term given the parameter vector.
     * @param parameters Flat vector of all trainable parameters.
     * @return Scalar regularization term (already weighted).
     */
    float calculate_regularization(const VectorR& parameters) const;
    /**
     * @brief Loads loss hyperparameters from a parsed JSON document.
     */
    void from_JSON(const JsonDocument&);

    /**
     * @brief Writes loss hyperparameters to a streaming JSON writer.
     */
    void to_JSON(JsonWriter&) const;

    /**
     * @brief Reads only the regularization fields from a parsed JSON document.
     */
    void regularization_from_JSON(const JsonDocument&);
    /**
     * @brief Writes only the regularization fields to a streaming JSON writer.
     */
    void regularization_to_JSON(JsonWriter&) const;

    /** @brief Canonical name of the loss (e.g. "Loss"). */
    const string& get_name() const { return name; }
    /**
     * @brief Computes the finite-difference perturbation step for a parameter.
     * @param parameter Parameter value at which the gradient is evaluated.
     * @return Perturbation magnitude h used by numerical-gradient checks.
     */
    static float calculate_h(const float parameter);

    /** @brief Prints a human-readable summary of the loss to stdout. */
    void print() const {}

private:

    /** @brief Throws if no NeuralNetwork has been bound. */
    void check_neural_network() const
    {
        if (!neural_network)
            throw runtime_error("Loss error: neural network is not set.");
    }

    /** @brief Throws if no Dataset has been bound. */
    void check_dataset() const
    {
        if (!dataset)
            throw runtime_error("Loss error: dataset is not set.");
    }

    /**
     * @brief Adds the regularization term to the loss value in @p bp.
     * @param bp BackPropagation buffer being filled.
     */
    void add_regularization(BackPropagation& bp) const;

    /**
     * @brief Computes the per-class weighting coefficient for WeightedSquaredError.
     * @param batch Current training batch.
     * @return Scalar coefficient applied to per-sample errors.
     */
    float get_weighted_coefficient(const Batch& batch) const;

    /**
     * @brief Drives the per-layer backward pass that fills the gradient buffer.
     * @param batch Current training batch.
     * @param forward_propagation Forward intermediates for the batch.
     * @param back_propagation Output buffer in which to accumulate gradients.
     */
    void calculate_layers_error_gradient(const Batch& batch,
                                         ForwardPropagation& forward_propagation,
                                         BackPropagation& back_propagation) const;

    /**
     * @brief Adds the regularization-term gradient to the parameter gradients in @p bp.
     * @param bp BackPropagation buffer being filled.
     */
    void add_regularization_gradient(BackPropagation& bp) const;

    /**
     * @brief Computes the deltas at the network output layer for the chosen loss term.
     * @param batch Current training batch.
     * @param forward_propagation Forward intermediates for the batch.
     * @param back_propagation Output buffer in which to write the output delta.
     */
    void calculate_output_deltas(const Batch& batch,
                                    const ForwardPropagation& forward_propagation,
                                    BackPropagation& back_propagation) const;

protected:

    /** @brief Currently selected loss term. */
    Error error = Error::MeanSquaredError;

    /** @brief Variance of the targets, used by NormalizedSquaredError. */
    float normalization_coefficient = 1.0f;
    /** @brief Weight applied to positive-class samples (WeightedSquaredError). */
    float positives_weight = 1.0f;
    /** @brief Weight applied to negative-class samples (WeightedSquaredError). */
    float negatives_weight = 1.0f;
    /** @brief Exponent p used by MinkowskiError. */
    float minkowski_parameter = 1.5f;

#ifdef OPENNN_HAS_CUDA
    /**
     * @brief Reduction workspace for cublasSasum / cudaMemcpy used by the
     *        loss kernels.
     *
     * Mutable because calculate_error grows it lazily on the first call;
     * the method's logical contract remains const (no observable state
     * changes).
     */
    mutable Buffer errors_device{Device::CUDA};
#endif

    /** @brief Currently selected regularization term. */
    Regularization regularization_method = Regularization::L2;
    /** @brief Multiplier applied to the regularization term in the total loss. */
    float regularization_weight = 0.001f;

    /** @brief Network whose parameters are being trained; not owned. */
    NeuralNetwork* neural_network = nullptr;
    /** @brief Dataset that provides training data; not owned. */
    Dataset* dataset = nullptr;

    /** @brief Canonical name of the loss instance. */
    string name = "Loss";
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
