//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file neuron_selection.h
 * @brief Declares the NeuronSelection abstract base and the
 *        NeuronsSelectionResults summary structure.
 *
 * Concrete subclasses (GrowingNeurons, ...) implement
 * perform_neurons_selection() to search the space of hidden-layer sizes
 * and return the architecture that minimizes the validation error.
 */

#pragma once

namespace opennn
{

class TrainingStrategy;
struct TrainingResults;
struct NeuronsSelectionResults;

/**
 * @class NeuronSelection
 * @brief Abstract base class for hidden-layer-size selection methods.
 *
 * Holds a pointer to a TrainingStrategy used to evaluate candidate
 * architectures, the common stopping criteria and trial averaging.
 * Subclasses implement perform_neurons_selection() to drive the search.
 */
class NeuronSelection
{
public:

    /**
     * @enum StoppingCondition
     * @brief Reasons that can terminate a neurons-selection run.
     */
    enum class StoppingCondition {
        MaximumTime,                ///< Configured time budget exhausted.
        SelectionErrorGoal,         ///< Validation error reached the configured goal.
        MaximumEpochs,              ///< Configured epoch budget exhausted.
        MaximumSelectionFailures,   ///< Validation error increased too many times.
        MaximumNeurons              ///< Maximum hidden-layer size reached.
    };

    /**
     * @brief Constructs the selector.
     * @param training_strategy Training strategy used to evaluate candidate sizes.
     */
    NeuronSelection(TrainingStrategy* training_strategy = nullptr);
    /** @brief Virtual destructor. */
    virtual ~NeuronSelection() = default;

    /** @brief Read-only access to the bound training strategy. */
    const TrainingStrategy* get_training_strategy() const { return training_strategy; }

    /** @brief Whether a training strategy has been bound. */
    bool has_training_strategy() const { return training_strategy; }

    /** @brief Whether progress should be printed to stdout. */
    bool get_display() const { return display; }

    /**
     * @brief Re-initializes the selector by setting its training strategy.
     *
     * Receives the training strategy used to evaluate candidates.
     */
    void set(TrainingStrategy*);

    /**
     * @brief Sets the training strategy directly.
     * @param new_training_strategy Strategy used to evaluate candidates.
     */
    void set_training_strategy(TrainingStrategy* new_training_strategy) { training_strategy = new_training_strategy; }

    /** @brief Resets all hyperparameters to their default values. */
    void set_default();

    /**
     * @brief Sets the maximum hidden-layer size considered.
     * @param new_maximum_neurons Upper bound on the number of neurons.
     */
    void set_maximum_neurons(const Index new_maximum_neurons) { maximum_neurons = new_maximum_neurons; }
    /**
     * @brief Sets the minimum hidden-layer size considered.
     * @param new_minimum_neurons Lower bound on the number of neurons.
     */
    void set_minimum_neurons(const Index new_minimum_neurons) { minimum_neurons = new_minimum_neurons; }
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
     * @return Best-of-run hidden-layer size and supporting statistics.
     */
    virtual NeuronsSelectionResults perform_neurons_selection() = 0;

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

    /** @brief Training strategy used to evaluate candidates; not owned. */
    TrainingStrategy* training_strategy = nullptr;

    /** @brief Per-iteration validation error of the best candidate so far. */
    VectorR validation_error_history;

    /** @brief Per-iteration training error of the best candidate so far. */
    VectorR training_error_history;

    /** @brief Lower bound on the hidden-layer size. */
    Index minimum_neurons = 0;

    /** @brief Upper bound on the hidden-layer size. */
    Index maximum_neurons = 0;

    /** @brief Number of independent training runs averaged per candidate. */
    Index trials_number = 1;

    /** @brief Validation error goal; selection stops when reached. */
    float validation_error_goal = 0;

    /** @brief Maximum training epochs per evaluation. */
    Index maximum_epochs = 10;

    /** @brief Maximum consecutive validation-error increases tolerated. */
    Index maximum_validation_failures = 100;

    /** @brief Maximum wall-clock selection time in seconds. */
    float maximum_time = 0;

    /** @brief Whether progress should be printed to stdout during selection. */
    bool display = true;

    /** @brief Canonical name of the selector (set by subclasses). */
    string name;
};

/**
 * @struct NeuronsSelectionResults
 * @brief Outcome of a NeuronSelection run.
 *
 * Contains the best (lowest validation error) hidden-layer size, the
 * optimal network parameters at that size, and per-iteration error
 * histories suitable for plotting.
 */
struct NeuronsSelectionResults
{
   /**
    * @brief Constructs the result pre-sized for an expected iteration count.
    * @param maximum_epochs Initial capacity for the error history vectors.
    */
   NeuronsSelectionResults(const Index maximum_epochs = 0);

   /**
    * @brief Resizes the error history vectors.
    * @param new_size New length for every history vector.
    */
   void resize_history(const Index new_size);

   /**
    * @brief Returns the canonical string name of the stopping condition.
    * @return Name (e.g. "MaximumNeurons").
    */
   string write_stopping_condition() const;

   /** @brief Prints a human-readable summary of the result to stdout. */
   void print() const;

   /** @brief Per-iteration hidden-layer size considered. */
   VectorI neurons_number_history;

   /** @brief Hidden-layer size at the optimal architecture. */
   Index optimal_neurons_number = 1;

   /** @brief Network parameters at the optimal architecture. */
   VectorR optimal_parameters;

   /** @brief Per-iteration training error of the best candidate. */
   VectorR training_error_history;

   /** @brief Per-iteration validation error of the best candidate. */
   VectorR validation_error_history;

   /** @brief Training error at the optimal architecture. */
   float optimum_training_error = 10.0f;

   /** @brief Validation error at the optimal architecture. */
   float optimum_validation_error = 10.0f;

   /** @brief Stopping condition that ended the selection run. */
   NeuronSelection::StoppingCondition stopping_condition = NeuronSelection::StoppingCondition::MaximumTime;

   /** @brief Total elapsed wall-clock time, formatted as "hh:mm:ss". */
   string elapsed_time;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
