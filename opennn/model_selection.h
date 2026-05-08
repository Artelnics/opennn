//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file model_selection.h
 * @brief Declares the ModelSelection class.
 *
 * ModelSelection orchestrates two architecture-search algorithms (input
 * selection and neuron selection) on top of a TrainingStrategy.
 */

#pragma once

#include "inputs_selection.h"
#include "neuron_selection.h"

namespace opennn
{

class TrainingStrategy;

/**
 * @class ModelSelection
 * @brief Searches for the best generalizing architecture for a model.
 *
 * Wraps two pluggable algorithms:
 *   - an InputsSelection strategy that decides which input features to keep,
 *   - a NeuronSelection strategy that decides how many hidden neurons to use.
 *
 * Both algorithms run on top of a user-supplied TrainingStrategy, which
 * provides the underlying NeuralNetwork, Dataset and Loss they consume to
 * score candidate architectures.
 */
class ModelSelection
{

public:

    // Constructors

    /**
     * @brief Constructs a model selection bound to a training strategy.
     *
     * Picks default selection algorithms (GrowingInputs and GrowingNeurons).
     *
     * @param new_training_strategy Non-owning pointer to the strategy used to
     *                              evaluate candidate architectures. May be null
     *                              and supplied later via set().
     */
    ModelSelection(TrainingStrategy* new_training_strategy = nullptr);

    /**
     * @brief Returns the underlying training strategy.
     * @return Const pointer to the strategy (may be nullptr).
     */
    const TrainingStrategy* get_training_strategy() const { return training_strategy; }

    /**
     * @brief Reports whether a training strategy is configured.
     * @return true if a non-null TrainingStrategy is set.
     */
    bool has_training_strategy() const { return training_strategy; }

    /**
     * @brief Replaces the underlying training strategy.
     * @param new_training_strategy Non-owning pointer to the new strategy.
     */
    void set(TrainingStrategy* new_training_strategy) { training_strategy = new_training_strategy; }

    /**
     * @brief Picks default selection algorithms.
     *
     * Sets the neurons-selection algorithm to "GrowingNeurons" and the
     * inputs-selection algorithm to "GrowingInputs".
     */
    void set_default();

    /**
     * @brief Validates that all components needed for selection are set.
     *
     * Verifies the training strategy, its loss, the network behind the loss,
     * the dataset, and that the dataset has at least one validation sample.
     *
     * @throws runtime_error if any required component is missing.
     */
    void check() const;

    /**
     * @brief Runs the configured neuron-selection algorithm.
     * @return Results of the search (best size found, error history, etc.).
     */
    NeuronsSelectionResults perform_neurons_selection();

    /**
     * @brief Runs the configured input-selection algorithm.
     * @return Results of the search (selected feature subset, error history, etc.).
     */
    InputsSelectionResults perform_input_selection();

    /**
     * @brief Restores the model-selection state from a JSON document.
     * @param document Parsed JSON produced by to_JSON().
     */
    void from_JSON(const JsonDocument& document);

    /**
     * @brief Serializes the model-selection state to JSON.
     * @param writer JSON writer that receives the configuration tree.
     */
    void to_JSON(JsonWriter& writer) const;

    /**
     * @brief Saves the model-selection state to a JSON file on disk.
     * @param file_name Destination path.
     * @throws runtime_error if the file cannot be opened for writing.
     */
    void save(const filesystem::path& file_name) const;

    /**
     * @brief Loads the model-selection state from a JSON file on disk.
     * @param file_name Source path.
     */
    void load(const filesystem::path& file_name);

private:

    /**
     * @brief Returns the neuron-selection algorithm.
     * @return Pointer to the algorithm owned by this ModelSelection.
     */
    NeuronSelection* get_neurons_selection() const { return neurons_selection.get(); }

    /**
     * @brief Returns the input-selection algorithm.
     * @return Pointer to the algorithm owned by this ModelSelection.
     */
    InputsSelection* get_inputs_selection() const { return inputs_selection.get(); }

    /**
     * @brief Selects the neuron-selection algorithm by name.
     *
     * Instantiates the algorithm via the registry and binds it to the current
     * training strategy.
     *
     * @param new_neurons_selection Algorithm name, e.g. "GrowingNeurons".
     */
    void set_neurons_selection(const string& new_neurons_selection);

    /**
     * @brief Selects the input-selection algorithm by name.
     *
     * Instantiates the algorithm via the registry and binds it to the current
     * training strategy.
     *
     * @param new_inputs_selection Algorithm name, e.g. "GrowingInputs",
     *                             "GeneticAlgorithm".
     */
    void set_inputs_selection(const string& new_inputs_selection);

    /// Non-owning pointer to the training strategy used to evaluate architectures.
    TrainingStrategy* training_strategy = nullptr;

    /// Algorithm in charge of choosing how many hidden neurons to use. Owned.
    unique_ptr<NeuronSelection> neurons_selection;

    /// Algorithm in charge of choosing which input features to keep. Owned.
    unique_ptr<InputsSelection> inputs_selection;
};

}
