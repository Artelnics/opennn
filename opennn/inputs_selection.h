//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

class TrainingStrategy;

struct TrainingResults;
struct InputsSelectionResults;

/// @brief Abstract base class for algorithms that search the optimal subset of input variables.
class InputsSelection
{
public:

    /// @brief Reasons the inputs selection loop may terminate.
    enum class StoppingCondition {
        MaximumTime,
        SelectionErrorGoal,
        MaximumInputs,
        MaximumEpochs,
        MaximumSelectionFailures
    };

    /// @brief Constructs the algorithm bound to an optional training strategy.
    InputsSelection(TrainingStrategy* = nullptr);
    virtual ~InputsSelection() = default;

    const TrainingStrategy* get_training_strategy() const { return training_strategy; }

    bool has_training_strategy() const { return training_strategy; }

    bool get_display() const { return display; }

    /// @brief Returns the minimum number of input variables that the algorithm may select.
    virtual Index get_minimum_inputs_number() const = 0;

    /// @brief Returns the maximum number of input variables that the algorithm may select.
    virtual Index get_maximum_inputs_number() const = 0;

    void set(TrainingStrategy* new_training_strategy) { training_strategy = new_training_strategy; }

    void set_trials_number(const Index new_trials_number) { trials_number = new_trials_number; }

    void set_display(bool new_display) { display = new_display; }

    void set_validation_error_goal(const float new_validation_error_goal) { validation_error_goal = new_validation_error_goal; }
    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

    /// @brief Verifies that the training strategy and its dependencies are valid for inputs selection.
    void check() const;

    /// @brief Runs the inputs selection algorithm until a stopping criterion is met.
    /// @return Results including the chosen inputs, optimal parameters and error histories.
    virtual InputsSelectionResults perform_input_selection() = 0;

    string get_name() const
    {
        return name;
    }

    /// @brief Loads algorithm configuration from a JSON document.
    virtual void from_JSON(const JsonDocument&) = 0;

    /// @brief Writes algorithm configuration to a JSON writer.
    virtual void to_JSON(JsonWriter&) const = 0;

    /// @brief Saves the algorithm configuration to disk.
    void save(const filesystem::path&) const;

    /// @brief Loads the algorithm configuration from disk.
    void load(const filesystem::path&);

    /// @brief Prints a human-readable description of the algorithm to stdout.
    virtual void print() const {}

protected:

    TrainingStrategy* training_strategy = nullptr;

    Index trials_number = 1;

    bool display = true;

    // Stopping criteria

    float validation_error_goal;

    Index maximum_epochs;

    Index maximum_validation_failures = 100;

    float maximum_time;

    string name;
};

/// @brief Aggregated results of an inputs selection run including optimal inputs and error histories.
struct InputsSelectionResults
{
    /// @brief Builds an empty results structure able to hold up to the given number of epochs.
    InputsSelectionResults(const Index = 0);

    /// @brief Returns the number of epochs actually recorded in the histories.
    Index get_epochs_number() const;

    /// @brief Resets the structure and reserves storage for the given number of epochs.
    void set(const Index = 0);

    /// @brief Returns a human-readable string describing the stopping condition that ended the run.
    string write_stopping_condition() const;

    /// @brief Resizes the recorded error histories.
    void resize_history(const Index new_size);

    /// @brief Prints a summary of the selection results to stdout.
    void print() const;

    // Neural network

    VectorR optimal_parameters;

    // Loss index

    VectorR training_error_history;

    VectorR validation_error_history;

    VectorR  mean_validation_error_history;

    VectorR mean_training_error_history;

    float optimum_training_error = MAX;

    float optimum_validation_error = MAX;

    vector<string> optimal_input_variable_names;

    vector<Index> optimal_input_variables_indices;

    VectorB optimal_inputs;

    // Model selection

    InputsSelection::StoppingCondition stopping_condition = InputsSelection::StoppingCondition::MaximumTime;

    string elapsed_time;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
