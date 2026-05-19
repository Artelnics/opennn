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

class NeuronSelection
{
public:

    enum class StoppingCondition { MaximumTime, SelectionErrorGoal, MaximumEpochs, MaximumSelectionFailures, MaximumNeurons };

    NeuronSelection(TrainingStrategy* = nullptr);
    virtual ~NeuronSelection() = default;

    const TrainingStrategy* get_training_strategy() const { return training_strategy; }

    bool has_training_strategy() const { return training_strategy; }

    bool get_display() const { return display; }

    void set(TrainingStrategy*);

    void set_training_strategy(TrainingStrategy* new_training_strategy) { training_strategy = new_training_strategy; }

    void set_default();

    void set_maximum_neurons(const Index new_maximum_neurons) { maximum_neurons = new_maximum_neurons; }
    void set_minimum_neurons(const Index new_minimum_neurons) { minimum_neurons = new_minimum_neurons; }
    void set_trials_number(const Index new_trials_number) { trials_number = new_trials_number; }

    void set_display(bool new_display) { display = new_display; }

    void set_validation_error_goal(const float new_validation_error_goal) { validation_error_goal = new_validation_error_goal; }
    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

void check() const;

    virtual NeuronsSelectionResults perform_neurons_selection() = 0;

    string get_name() const
    {
        return name;
    }

    virtual void from_JSON(const JsonDocument&) = 0;

    virtual void to_JSON(JsonWriter&) const = 0;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

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

struct NeuronsSelectionResults
{
   NeuronsSelectionResults(const Index maximum_epochs = 0);

   void resize_history(const Index new_size);

   string write_stopping_condition() const;

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
