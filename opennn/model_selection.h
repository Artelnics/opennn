//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "inputs_selection.h"
#include "neuron_selection.h"

namespace opennn
{

class TrainingStrategy;

/// @brief Orchestrates model selection by combining inputs selection and neurons selection over a TrainingStrategy.
class ModelSelection
{

public:

    // Constructors

    /// @brief Constructs a model selection bound to an optional training strategy.
    ModelSelection(TrainingStrategy* = nullptr);
    const TrainingStrategy* get_training_strategy() const { return training_strategy; }
    bool has_training_strategy() const { return training_strategy; }
    void set(TrainingStrategy* new_training_strategy) { training_strategy = new_training_strategy; }

    /// @brief Restores default algorithms and parameters for inputs and neurons selection.
    void set_default();

    /// @brief Checks that the training strategy and its dependencies are consistent before running selection.
    void check() const;

    /// @brief Runs the configured neurons selection algorithm.
    /// @return Results including the optimal neuron count and error histories.
    NeuronsSelectionResults perform_neurons_selection();

    /// @brief Runs the configured inputs selection algorithm.
    /// @return Results including the optimal input variables and error histories.
    InputsSelectionResults perform_input_selection();

    /// @brief Loads model selection configuration from a JSON document.
    void from_JSON(const JsonDocument&);

    /// @brief Writes the current model selection configuration to a JSON writer.
    void to_JSON(JsonWriter&) const;

    /// @brief Saves the model selection configuration to disk.
    void save(const filesystem::path&) const;

    /// @brief Loads the model selection configuration from disk.
    void load(const filesystem::path&);

private:

    NeuronSelection* get_neurons_selection() const { return neurons_selection.get(); }
    InputsSelection* get_inputs_selection() const { return inputs_selection.get(); }
    void set_neurons_selection(const string&);
    void set_inputs_selection(const string&);

    TrainingStrategy* training_strategy = nullptr;

    unique_ptr<NeuronSelection> neurons_selection;

    unique_ptr<InputsSelection> inputs_selection;
};

}
