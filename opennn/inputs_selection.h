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

    void set(TrainingStrategy* new_ts) { training_strategy = new_ts; }

    void set_trials_number(const Index n) { trials_number = n; }

    void set_display(bool d) { display = d; }

    void set_validation_error_goal(const type v) { validation_error_goal = v; }
    void set_maximum_epochs(const Index n) { maximum_epochs = n; }
    void set_maximum_validation_failures(const Index n) { maximum_validation_failures = n; }
    void set_maximum_time(const type t) { maximum_time = t; }

    void check() const;

    virtual InputsSelectionResults perform_input_selection() = 0;

    string get_name() const
    {
        return name;
    }

    virtual void from_XML(const XmlDocument&) = 0;

    virtual void to_XML(XmlPrinter&) const = 0;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    virtual void print() const {}

protected:

    TrainingStrategy* training_strategy = nullptr;

    Index trials_number = 1;

    bool display = true;
   
    // Stopping criteria

    type validation_error_goal;

    Index maximum_epochs;

    Index maximum_validation_failures = 100;

    type maximum_time;

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

    type optimum_training_error = MAX;

    type optimum_validation_error = MAX;

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
