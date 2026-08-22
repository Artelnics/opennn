//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   N E U R O N S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"

namespace opennn
{

class TrainingStrategy;
struct NeuronsSelectionResult;

class GrowingNeurons
{

public:

    enum class StoppingCondition {
        MaximumTime,
        ValidationErrorGoal,
        MaximumEpochs,
        MaximumValidationFailures,
        MaximumNeurons
    };

    explicit GrowingNeurons(TrainingStrategy* = nullptr);

    const TrainingStrategy* get_training_strategy() const noexcept { return training_strategy; }

    void set(TrainingStrategy*);
    void set_training_strategy(TrainingStrategy* new_training_strategy) noexcept
    {
        training_strategy = new_training_strategy;
    }

    void set_default();

    void set_maximum_neurons(const Index new_maximum_neurons) { maximum_neurons = new_maximum_neurons; }
    void set_minimum_neurons(const Index new_minimum_neurons) { minimum_neurons = new_minimum_neurons; }
    void set_trials_number(const Index new_trials_number) { trials_number = new_trials_number; }

    void set_display(bool new_display) { display = new_display; }

    void set_validation_error_goal(const float new_validation_error_goal) { validation_error_goal = new_validation_error_goal; }
    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

    void set_folds_number(const Index new_folds_number) { folds_number = max<Index>(new_folds_number, Index(1)); }

    void set_neurons_increment(const Index new_neurons_increment) { neurons_increment = new_neurons_increment; }

    void set_warm_start(bool new_warm_start) { warm_start = new_warm_start; }

    NeuronsSelectionResult perform_neurons_selection();

    string get_name() const { return "GrowingNeurons"; }

    void from_JSON(const JsonDocument&);

    void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private:

    TrainingStrategy* training_strategy = nullptr;

    Index minimum_neurons = 0;

    Index maximum_neurons = 0;

    Index trials_number = 1;

    Index folds_number = 1;

    float validation_error_goal = 0;

    Index maximum_epochs = 10;

    Index maximum_validation_failures = 100;

    float maximum_time = 0;

    bool display = true;

    Index neurons_increment = 0;

    bool warm_start = true;
};

// Kept for Neural Designer, which spells the type NeuronSelection::StoppingCondition.
using NeuronSelection = GrowingNeurons;

struct NeuronsSelectionResult
{
   NeuronsSelectionResult(const Index maximum_epochs = 0);

   void resize_history(const Index);

   void print() const;

   VectorI neurons_number_history;

   Index optimal_neurons_number = 1;

   VectorR optimal_parameters;

   VectorR training_error_history;

   VectorR validation_error_history;

   float optimum_training_error = 10.0f;

   float optimum_validation_error = 10.0f;

   optional<GrowingNeurons::StoppingCondition> stopping_condition;

   string elapsed_time;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
