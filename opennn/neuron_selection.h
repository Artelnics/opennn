//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

namespace opennn
{

class TrainingStrategy;
struct TrainingResults;
struct NeuronsSelectionResults;

/// @brief Abstract base class for algorithms that select the optimal number of hidden neurons.
class NeuronSelection
{
public:

    /// @brief Reasons the neurons selection loop may terminate.
    enum class StoppingCondition { MaximumTime, SelectionErrorGoal, MaximumEpochs, MaximumSelectionFailures, MaximumNeurons };

    /// @brief Constructs the algorithm bound to an optional training strategy.
    NeuronSelection(TrainingStrategy* = nullptr);
    virtual ~NeuronSelection() = default;

    const TrainingStrategy* get_training_strategy() const { return training_strategy; }

    bool has_training_strategy() const { return training_strategy; }

    bool get_display() const { return display; }

    /// @brief Binds the algorithm to the given training strategy.
    void set(TrainingStrategy*);

    void set_training_strategy(TrainingStrategy* new_training_strategy) { training_strategy = new_training_strategy; }

    /// @brief Restores default search bounds and stopping criteria.
    void set_default();

    void set_maximum_neurons(const Index new_maximum_neurons) { maximum_neurons = new_maximum_neurons; }
    void set_minimum_neurons(const Index new_minimum_neurons) { minimum_neurons = new_minimum_neurons; }
    void set_trials_number(const Index new_trials_number) { trials_number = new_trials_number; }

    void set_display(bool new_display) { display = new_display; }

    void set_validation_error_goal(const float new_validation_error_goal) { validation_error_goal = new_validation_error_goal; }
    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

/// @brief Verifies that the training strategy and its dependencies are valid for neurons selection.
void check() const;

    /// @brief Runs the neurons selection algorithm until a stopping criterion is met.
    /// @return Results including the optimal neuron count, optimal parameters and error histories.
    virtual NeuronsSelectionResults perform_neurons_selection() = 0;

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

    VectorR validation_error_history;

    VectorR training_error_history;

    Index minimum_neurons = 0;

    Index maximum_neurons = 0;

    Index trials_number = 1;

    float validation_error_goal = 0;

    Index maximum_epochs = 10;

    Index maximum_validation_failures = 100;

    float maximum_time = 0;

    bool display = true;

    string name;
};

/// @brief Aggregated results of a neurons selection run including the optimal neuron count and error histories.
struct NeuronsSelectionResults
{
   /// @brief Builds an empty results structure able to hold up to the given number of epochs.
   NeuronsSelectionResults(const Index maximum_epochs = 0);

   /// @brief Resizes the recorded error and neuron-count histories.
   void resize_history(const Index new_size);

   /// @brief Returns a human-readable string describing the stopping condition that ended the run.
   string write_stopping_condition() const;

   /// @brief Prints a summary of the selection results to stdout.
   void print() const;

   // Neural network

   VectorI neurons_number_history;

   Index optimal_neurons_number = 1;

   VectorR optimal_parameters;

   // Loss index

   VectorR training_error_history;

   VectorR validation_error_history;

   float optimum_training_error = 10.0f;

   float optimum_validation_error = 10.0f;

   // Model selection

   NeuronSelection::StoppingCondition stopping_condition = NeuronSelection::StoppingCondition::MaximumTime;

   string elapsed_time;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
