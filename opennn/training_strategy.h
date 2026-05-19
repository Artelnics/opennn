//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "loss.h"
#include "optimizer.h"

namespace opennn
{

class Loss;
class Optimizer;

struct TrainingResults;

/// @brief High-level orchestrator pairing a Loss with an Optimizer for a network/dataset.
class TrainingStrategy
{

public:

    /// @brief Constructs the strategy with default loss (MSE) and optimizer (Adam) bound to the given network and dataset.
    TrainingStrategy(NeuralNetwork* = nullptr, Dataset* = nullptr);

    const Dataset* get_dataset() const { return dataset; }
    Dataset* get_dataset() { return dataset; }

    const NeuralNetwork* get_neural_network() const { return neural_network; }
    NeuralNetwork* get_neural_network() { return neural_network; }

    const Loss* get_loss() const { return loss.get(); }
    Loss* get_loss() { return loss.get(); }

    const Optimizer* get_optimization_algorithm() const { return optimizer.get(); }
    Optimizer* get_optimization_algorithm() { return optimizer.get(); }
    /// @brief Rebinds the strategy to a new network/dataset, resetting loss and optimizer to defaults.
    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);
    /// @brief Resets the loss and optimizer to their default types and hyperparameters.
    void set_default();

    void set_dataset(Dataset* new_dataset) { dataset = new_dataset; }
    void set_neural_network(NeuralNetwork* new_neural_network) { neural_network = new_neural_network; }

    /// @brief Replaces the current loss with one selected by name (e.g. "MeanSquaredError", "CrossEntropy").
    void set_loss(const string&);
    /// @brief Replaces the current optimizer with one selected by name (e.g. "Adam", "SGD", "QuasiNewton", "LM").
    void set_optimization_algorithm(const string&);

    /// @brief Runs the configured optimizer against the configured loss and returns the training history.
    TrainingResults train();
    /// @brief Restores the full strategy (loss + optimizer configurations) from a JSON document.
    void from_JSON(const JsonDocument&);
    /// @brief Serializes the full strategy (loss + optimizer configurations) to JSON.
    void to_JSON(JsonWriter&) const;

    /// @brief Writes the strategy configuration to a JSON file at the given path.
    void save(const filesystem::path&) const;
    /// @brief Loads the strategy configuration from a JSON file at the given path.
    void load(const filesystem::path&);

private:

    void fix_forecasting();

    Dataset* dataset = nullptr;

    NeuralNetwork* neural_network = nullptr;

    unique_ptr<Loss> loss;

    unique_ptr<Optimizer> optimizer;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
