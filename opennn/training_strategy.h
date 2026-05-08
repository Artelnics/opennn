//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file training_strategy.h
 * @brief Declares the TrainingStrategy class.
 *
 * TrainingStrategy combines a Loss term and an Optimizer and runs the training
 * loop on a NeuralNetwork against a Dataset.
 */

#pragma once

#include "loss.h"
#include "optimizer.h"

namespace opennn
{

class Loss;
class Optimizer;

struct TrainingResults;

/**
 * @class TrainingStrategy
 * @brief Coordinates the training of a NeuralNetwork on a Dataset.
 *
 * Aggregates a Loss (error term) and an Optimizer (parameter-update rule),
 * keeps non-owning pointers to the network and dataset they operate on, and
 * exposes a single train() entry point that delegates to the optimizer.
 *
 * Sensible defaults are picked automatically based on the network's layer
 * composition: AdaptiveMomentEstimation + appropriate loss for recurrent,
 * convolutional, transformer-like and text-classification networks; QuasiNewtonMethod
 * + CrossEntropy / WeightedSquaredError / MeanSquaredError for the
 * classification and approximation cases.
 */
class TrainingStrategy
{

public:

    /**
     * @brief Constructs a training strategy for a given network and dataset.
     *
     * Both pointers may be null and supplied later via set(). When both are
     * provided the constructor calls set_default() to pick a loss and optimizer
     * appropriate for the network's architecture.
     *
     * @param new_neural_network Non-owning pointer to the network to be trained.
     * @param new_dataset Non-owning pointer to the dataset providing samples.
     */
    TrainingStrategy(NeuralNetwork* new_neural_network = nullptr, Dataset* new_dataset = nullptr);

    /**
     * @brief Returns the dataset associated with the training strategy.
     * @return Const pointer to the dataset (may be nullptr).
     */
    const Dataset* get_dataset() const { return dataset; }

    /**
     * @brief Returns the dataset associated with the training strategy.
     * @return Mutable pointer to the dataset (may be nullptr).
     */
    Dataset* get_dataset() { return dataset; }

    /**
     * @brief Returns the neural network being trained.
     * @return Const pointer to the network (may be nullptr).
     */
    const NeuralNetwork* get_neural_network() const { return neural_network; }

    /**
     * @brief Returns the neural network being trained.
     * @return Mutable pointer to the network (may be nullptr).
     */
    NeuralNetwork* get_neural_network() { return neural_network; }

    /**
     * @brief Returns the loss term used during training.
     * @return Const pointer to the Loss instance owned by this strategy.
     */
    const Loss* get_loss() const { return loss.get(); }

    /**
     * @brief Returns the loss term used during training.
     * @return Mutable pointer to the Loss instance owned by this strategy.
     */
    Loss* get_loss() { return loss.get(); }

    /**
     * @brief Returns the optimizer that performs parameter updates.
     * @return Const pointer to the Optimizer instance owned by this strategy.
     */
    const Optimizer* get_optimization_algorithm() const { return optimizer.get(); }

    /**
     * @brief Returns the optimizer that performs parameter updates.
     * @return Mutable pointer to the Optimizer instance owned by this strategy.
     */
    Optimizer* get_optimization_algorithm() { return optimizer.get(); }

    /**
     * @brief Resets the strategy to point at a new network and dataset.
     *
     * Stores both pointers and calls set_default() to rebuild the loss and
     * optimizer to match the network's architecture.
     *
     * @param new_neural_network Non-owning pointer to the network.
     * @param new_dataset Non-owning pointer to the dataset.
     */
    void set(NeuralNetwork* new_neural_network = nullptr, Dataset* new_dataset = nullptr);

    /**
     * @brief Picks a default loss and optimizer based on the network architecture.
     *
     * Inspects the layers of the configured neural network and selects:
     *   - Recurrent networks: MeanSquaredError + AdaptiveMomentEstimation.
     *   - Convolutional networks: CrossEntropy + AdaptiveMomentEstimation.
     *   - Transformer-style networks (Dense over rank-2 inputs): CrossEntropy + Adam.
     *   - Text classification (Embedding or MultiHeadAttention): CrossEntropy + Adam.
     *   - Multi-class classification (Softmax output): CrossEntropy + QuasiNewtonMethod.
     *   - Binary classification (Sigmoid output): WeightedSquaredError + QuasiNewtonMethod.
     *   - Regression (default): MeanSquaredError + QuasiNewtonMethod.
     */
    void set_default();

    /**
     * @brief Replaces the dataset pointer without rebuilding loss/optimizer.
     * @param new_dataset Non-owning pointer to the new dataset.
     */
    void set_dataset(Dataset* new_dataset) { dataset = new_dataset; }

    /**
     * @brief Replaces the neural network pointer without rebuilding loss/optimizer.
     * @param new_neural_network Non-owning pointer to the new network.
     */
    void set_neural_network(NeuralNetwork* new_neural_network) { neural_network = new_neural_network; }

    /**
     * @brief Selects the loss term by name.
     *
     * Constructs a fresh Loss instance bound to the current network and dataset,
     * sets the requested error type and re-binds the optimizer (if any).
     *
     * @param new_loss One of: "MeanSquaredError", "NormalizedSquaredError",
     *                 "WeightedSquaredError", "CrossEntropy", "CrossEntropyError3d",
     *                 "MinkowskiError".
     */
    void set_loss(const string& new_loss);

    /**
     * @brief Selects the optimization algorithm by name.
     *
     * Looks up the optimizer in the registry, instantiates it and binds it to
     * the current loss.
     *
     * @param new_optimization_algorithm One of: "AdaptiveMomentEstimation",
     *                                   "QuasiNewtonMethod", "StochasticGradientDescent",
     *                                   "LevenbergMarquardtAlgorithm".
     */
    void set_optimization_algorithm(const string& new_optimization_algorithm);

    /**
     * @brief Runs the training loop.
     *
     * Validates that network, dataset, loss and optimizer are all set, applies
     * the forecasting batch-size adjustment when relevant, then delegates to
     * the optimizer's train() method.
     *
     * @return TrainingResults with the per-epoch loss/selection-error history
     *         and the final stopping condition.
     * @throws runtime_error if any required component is not configured.
     */
    TrainingResults train();

    /**
     * @brief Restores the strategy state from a JSON document.
     * @param document Parsed JSON document produced by to_JSON().
     */
    void from_JSON(const JsonDocument& document);

    /**
     * @brief Serializes the strategy state to JSON.
     * @param writer JSON writer that receives the strategy's element tree.
     */
    void to_JSON(JsonWriter& writer) const;

    /**
     * @brief Saves the strategy state to a JSON file on disk.
     * @param file_name Destination path.
     * @throws runtime_error if the file cannot be opened for writing.
     */
    void save(const filesystem::path& file_name) const;

    /**
     * @brief Loads the strategy state from a JSON file on disk.
     *
     * Calls set_default() before reading, so any fields missing from the file
     * keep architecture-appropriate defaults.
     *
     * @param file_name Source path.
     */
    void load(const filesystem::path& file_name);

private:

    /**
     * @brief Adjusts optimizer batch size for forecasting/recurrent networks.
     *
     * Ensures the batch size is a multiple of the recurrent layer's lookback
     * window so each batch is a clean sequence boundary.
     */
    void fix_forecasting();

    /// Non-owning pointer to the dataset providing training samples.
    Dataset* dataset = nullptr;

    /// Non-owning pointer to the network being trained.
    NeuralNetwork* neural_network = nullptr;

    /// Loss term computed during forward/backward propagation. Owned.
    unique_ptr<Loss> loss;

    /// Parameter-update rule that consumes gradients from @p loss. Owned.
    unique_ptr<Optimizer> optimizer;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
