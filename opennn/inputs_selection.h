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

class InputsSelection
{
public:

    enum class StoppingCondition {
        MaximumTime,
        SelectionErrorGoal,
        MaximumInputs,
        MaximumEpochs,
        MaximumSelectionFailures
    };

    InputsSelection(TrainingStrategy* = nullptr);
    virtual ~InputsSelection() = default;

    const TrainingStrategy* get_training_strategy() const { return training_strategy; }

    bool has_training_strategy() const { return training_strategy; }

    bool get_display() const { return display; }

    virtual Index get_minimum_inputs_number() const = 0;
    virtual Index get_maximum_inputs_number() const = 0;

    void set(TrainingStrategy* new_training_strategy) { training_strategy = new_training_strategy; }

    void set_trials_number(const Index new_trials_number) { trials_number = new_trials_number; }

    void set_display(bool new_display) { display = new_display; }

    void set_validation_error_goal(const float new_validation_error_goal) { validation_error_goal = new_validation_error_goal; }
    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

    void check() const;

    virtual InputsSelectionResults perform_input_selection() = 0;

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

    Index trials_number = 1;

    bool display = true;

    // Stopping criteria

    float validation_error_goal;

    Index maximum_epochs;

    Index maximum_validation_failures = 100;

    float maximum_time;

    string name;
};

struct InputsSelectionResults
{
    InputsSelectionResults(const Index = 0);

    Index get_epochs_number() const;

    void set(const Index = 0);

    string write_stopping_condition() const;

    void resize_history(const Index new_size);

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
