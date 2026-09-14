// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/core/opennn_types.h"
#include "opennn/model_selection/selection_algorithm.h"

namespace opennn
{

class Training;
class Network;
class Dataset;

struct TrainingResult;
struct InputSelectionResult;
struct Descriptives;

class InputSelection : public SelectionAlgorithm
{
public:

    enum class StoppingCondition {
        MaximumTime,
        ValidationErrorGoal,
        MaximumInputs,
        MaximumEpochs,
        MaximumValidationFailures
    };

    explicit InputSelection(Training* = nullptr);
    virtual ~InputSelection() = default;

    virtual Index get_minimum_inputs_number() const = 0;
    virtual Index get_maximum_inputs_number() const = 0;

    virtual InputSelectionResult perform_input_selection() = 0;

    string get_name() const { return name; }

    virtual void from_JSON(const JsonDocument&) = 0;

    virtual void to_JSON(JsonWriter&) const = 0;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

protected:

    void configure_network_inputs(Network*, Dataset*, Index) const;

    void install_optimal_inputs(Network*,
                                Dataset*,
                                const vector<Index>& optimal_input_indices,
                                const vector<Index>& target_indices,
                                const vector<Index>& time_indices) const;

    string name;
};

struct InputSelectionResult
{
    InputSelectionResult(const Index = 0);

    Index get_epochs_number() const { return training_error_history.size(); }

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

    optional<InputSelection::StoppingCondition> stopping_condition;

    string elapsed_time;
};

}
