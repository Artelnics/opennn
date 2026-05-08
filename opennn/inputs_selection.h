//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file inputs_selection.h
 * @brief Declares the InputsSelection abstract base and the
 *        InputsSelectionResults summary structure.
 *
 * Concrete subclasses (GeneticAlgorithm, GrowingInputs, ...) implement
 * perform_input_selection() to search the space of input feature subsets
 * and return the one that minimizes the validation error.
 */

#pragma once

#include "pch.h"

namespace opennn
{

class TrainingStrategy;

struct TrainingResults;
struct InputsSelectionResults;

/**
 * @class InputsSelection
 * @brief Abstract base class for input feature selection methods.
 *
 * Holds a pointer to a TrainingStrategy used to evaluate candidate input
 * subsets, the common stopping criteria (epoch budget, time budget,
 * validation goal, validation failure budget) and trial averaging.
 * Subclasses implement perform_input_selection() to drive the search.
 */
class InputsSelection
{
public:

    /**
     * @enum StoppingCondition
     * @brief Reasons that can terminate an input-selection run.
     */
    enum class StoppingCondition {
        MaximumTime,                ///< Configured time budget exhausted.
        SelectionErrorGoal,         ///< Validation error reached the configured goal.
        MaximumInputs,              ///< Maximum number of input variables reached.
        MaximumEpochs,              ///< Configured epoch budget exhausted.
        MaximumSelectionFailures    ///< Validation error increased too many times.
    };

    /**
     * @brief Constructs the selector.
     * @param training_strategy Training strategy used to evaluate candidate subsets.
     */
    InputsSelection(TrainingStrategy* training_strategy = nullptr);
    /** @brief Virtual destructor. */
    virtual ~InputsSelection() = default;

    /** @brief Read-only access to the bound training strategy. */
    const TrainingStrategy* get_training_strategy() const { return training_strategy; }

    /** @brief Whether a training strategy has been bound. */
    bool has_training_strategy() const { return training_strategy; }

    /** @brief Whether progress should be printed to stdout. */
    bool get_display() const { return display; }

    /** @brief Lower bound on the number of selected inputs in any candidate. */
    virtual Index get_minimum_inputs_number() const = 0;
    /** @brief Upper bound on the number of selected inputs in any candidate. */
    virtual Index get_maximum_inputs_number() const = 0;

    /**
     * @brief Re-initializes the selector by setting its training strategy.
     * @param new_training_strategy Strategy used to evaluate candidates.
     */
    void set(TrainingStrategy* new_training_strategy) { training_strategy = new_training_strategy; }

    /**
     * @brief Sets the number of training trials per candidate (mean is used).
     * @param new_trials_number Number of independent training runs averaged.
     */
    void set_trials_number(const Index new_trials_number) { trials_number = new_trials_number; }

    /**
     * @brief Toggles per-iteration progress printing.
     * @param new_display True to print progress to stdout.
     */
    void set_display(bool new_display) { display = new_display; }

    /**
     * @brief Sets the validation error goal.
     * @param new_validation_error_goal Selection stops when validation error reaches this value.
     */
    void set_validation_error_goal(const float new_validation_error_goal) { validation_error_goal = new_validation_error_goal; }
    /**
     * @brief Sets the maximum training epochs per evaluation.
     * @param new_maximum_epochs Epoch budget for each candidate's training.
     */
    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    /**
     * @brief Sets the maximum number of consecutive validation-error increases tolerated.
     * @param new_maximum_validation_failures Failure budget for early stopping.
     */
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    /**
     * @brief Sets the maximum wall-clock selection time.
     * @param new_maximum_time Time budget in seconds.
     */
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

    /**
     * @brief Validates the bound configuration; throws if anything is missing.
     */
    void check() const;

    /**
     * @brief Runs the selection algorithm.
     * @return Best-of-run input subset and supporting statistics.
     */
    virtual InputsSelectionResults perform_input_selection() = 0;

    /** @brief Canonical name of the selector (set by subclasses). */
    string get_name() const
    {
        return name;
    }

    /**
     * @brief Loads selector hyperparameters from a parsed JSON document.
     */
    virtual void from_JSON(const JsonDocument&) = 0;

    /**
     * @brief Writes selector hyperparameters to a streaming JSON writer.
     */
    virtual void to_JSON(JsonWriter&) const = 0;

    /**
     * @brief Saves the selector configuration to a file.
     *
     * Receives the destination path.
     */
    void save(const filesystem::path&) const;
    /**
     * @brief Loads the selector configuration from a file.
     *
     * Receives the source path.
     */
    void load(const filesystem::path&);

    /** @brief Prints a human-readable summary of the selector to stdout. */
    virtual void print() const {}

protected:

    /** @brief Training strategy used to evaluate candidate subsets; not owned. */
    TrainingStrategy* training_strategy = nullptr;

    /** @brief Number of independent training runs averaged per candidate. */
    Index trials_number = 1;

    /** @brief Whether progress should be printed to stdout during selection. */
    bool display = true;

    /** @brief Validation error goal; selection stops when reached. */
    float validation_error_goal;

    /** @brief Maximum training epochs per evaluation. */
    Index maximum_epochs;

    /** @brief Maximum consecutive validation-error increases tolerated. */
    Index maximum_validation_failures = 100;

    /** @brief Maximum wall-clock selection time in seconds. */
    float maximum_time;

    /** @brief Canonical name of the selector (set by subclasses). */
    string name;
};

/**
 * @struct InputsSelectionResults
 * @brief Outcome of an InputsSelection run.
 *
 * Contains the best (lowest validation error) input subset, the optimal
 * network parameters at that subset, and per-iteration error histories
 * suitable for plotting.
 */
struct InputsSelectionResults
{
    /**
     * @brief Constructs the result pre-sized for an expected iteration count.
     * @param expected_epochs Initial capacity for the error history vectors.
     */
    InputsSelectionResults(const Index expected_epochs = 0);

    /** @brief Number of selection iterations effectively run. */
    Index get_epochs_number() const;

    /**
     * @brief (Re)allocates buffers for a given iteration count.
     * @param expected_epochs Initial capacity for the error history vectors.
     */
    void set(const Index expected_epochs = 0);

    /**
     * @brief Returns the canonical string name of the stopping condition.
     * @return Name (e.g. "MaximumEpochs").
     */
    string write_stopping_condition() const;

    /**
     * @brief Resizes the error history vectors.
     * @param new_size New length for every history vector.
     */
    void resize_history(const Index new_size);

    /** @brief Prints a human-readable summary of the result to stdout. */
    void print() const;

    /** @brief Network parameters at the optimal input subset. */
    VectorR optimal_parameters;

    /** @brief Per-iteration training error of the best individual / step. */
    VectorR training_error_history;

    /** @brief Per-iteration validation error of the best individual / step. */
    VectorR validation_error_history;

    /** @brief Per-iteration mean validation error across trials. */
    VectorR  mean_validation_error_history;

    /** @brief Per-iteration mean training error across trials. */
    VectorR mean_training_error_history;

    /** @brief Training error at the optimal input subset. */
    float optimum_training_error = MAX;

    /** @brief Validation error at the optimal input subset. */
    float optimum_validation_error = MAX;

    /** @brief Names of the input variables in the optimal subset. */
    vector<string> optimal_input_variable_names;

    /** @brief Indices of the input variables in the optimal subset. */
    vector<Index> optimal_input_variables_indices;

    /** @brief Boolean mask of the optimal input subset (one entry per original input). */
    VectorB optimal_inputs;

    /** @brief Stopping condition that ended the selection run. */
    InputsSelection::StoppingCondition stopping_condition = InputsSelection::StoppingCondition::MaximumTime;

    /** @brief Total elapsed wall-clock time, formatted as "hh:mm:ss". */
    string elapsed_time;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
