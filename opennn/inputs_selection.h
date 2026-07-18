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
class NeuralNetwork;
class Dataset;

struct TrainingResult;
struct InputsSelectionResult;

class InputsSelection
{
public:

    enum class StoppingCondition {
        MaximumTime,
        ValidationErrorGoal,
        MaximumInputs,
        MaximumEpochs,
        MaximumValidationFailures
    };

    explicit InputsSelection(TrainingStrategy* = nullptr);
    virtual ~InputsSelection() = default;

    const TrainingStrategy* get_training_strategy() const noexcept { return training_strategy; }

    bool has_training_strategy() const noexcept { return training_strategy; }

    bool get_display() const noexcept { return display; }

    virtual Index get_minimum_inputs_number() const = 0;
    virtual Index get_maximum_inputs_number() const = 0;

    void set(TrainingStrategy* new_training_strategy) { training_strategy = new_training_strategy; }

    void set_trials_number(const Index new_trials_number) { trials_number = new_trials_number; }

    void set_display(bool new_display) { display = new_display; }

    void set_validation_error_goal(const float new_validation_error_goal) { validation_error_goal = new_validation_error_goal; }
    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

    // Inner k-fold cross-validation for scoring feature subsets during selection. folds_number == 1
    // keeps the legacy single Training/Validation-split behaviour. >1 scores each subset by the mean
    // validation error over a stratified (or, for time series, sequential) partition of the
    // Training+Validation pool -- a robust, less overfittable selection criterion. Testing/None
    // samples are never touched and the persistent sample roles are never mutated (see FoldScope).
    void set_folds_number(const Index new_folds_number) { folds_number = max<Index>(new_folds_number, Index(1)); }
    void set_folds_seed(const Index new_folds_seed) { folds_seed = new_folds_seed; }

    virtual InputsSelectionResult perform_input_selection() = 0;

    string get_name() const { return name; }

    virtual void from_JSON(const JsonDocument&) = 0;

    virtual void to_JSON(JsonWriter&) const = 0;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    virtual void print() const {}

protected:

    void configure_neural_network_inputs(NeuralNetwork*, Dataset*, Index);

    // Build the k-fold partition of the Training+Validation pool ONCE per selection run. Returns k
    // index lists (the validation set of each fold). Stratified by target class for classification,
    // sequential contiguous blocks for time series. Deterministic given folds_seed.
    vector<vector<Index>> build_fold_partition() const;

    // Score the currently-configured input subset by k-fold CV over the given partition, opening a
    // FoldScope per fold so the dataset's persistent roles are never mutated. Returns the mean
    // validation error over folds; writes the mean training error to the out-param.
    float evaluate_folds(const vector<vector<Index>>& fold_partition, float& mean_training_error) const;

    TrainingStrategy* training_strategy = nullptr;

    Index trials_number = 1;

    Index folds_number = 1;

    Index folds_seed = 0;

    bool display = true;


    float validation_error_goal = 0;

    Index maximum_epochs = 10;

    Index maximum_validation_failures = 100;

    float maximum_time = 0;

    string name;
};

struct InputsSelectionResult
{
    InputsSelectionResult(const Index = 0);

    Index get_epochs_number() const;

    void set(const Index = 0);

    void resize_history(const Index);

    void print() const;


    VectorR optimal_parameters;


    VectorR training_error_history;

    VectorR validation_error_history;

    VectorR mean_validation_error_history;

    VectorR mean_training_error_history;

    float optimum_training_error = MAX;

    float optimum_validation_error = MAX;

    vector<string> optimal_input_variable_names;

    vector<Index> optimal_input_variables_indices;

    VectorB optimal_inputs;


    optional<InputsSelection::StoppingCondition> stopping_condition;

    string elapsed_time;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
