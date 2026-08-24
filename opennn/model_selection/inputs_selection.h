//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"
#include "opennn/model_selection/selection_algorithm.h"

namespace opennn
{

class TrainingStrategy;
class NeuralNetwork;
class Dataset;

struct TrainingResult;
struct InputsSelectionResult;
struct Descriptives;

class InputsSelection : public SelectionAlgorithm
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

    virtual Index get_minimum_inputs_number() const = 0;
    virtual Index get_maximum_inputs_number() const = 0;

    virtual InputsSelectionResult perform_input_selection() = 0;

    string get_name() const { return name; }

    virtual void from_JSON(const JsonDocument&) = 0;

    virtual void to_JSON(JsonWriter&) const = 0;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

protected:

    void configure_neural_network_inputs(NeuralNetwork*, Dataset*, Index) const;

    void install_optimal_inputs(NeuralNetwork*,
                                Dataset*,
                                const vector<Index>& optimal_input_indices,
                                const vector<Index>& target_indices,
                                const vector<Index>& time_indices) const;

    string name;
};

struct InputsSelectionResult
{
    InputsSelectionResult(const Index = 0);

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

    optional<InputsSelection::StoppingCondition> stopping_condition;

    string elapsed_time;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
