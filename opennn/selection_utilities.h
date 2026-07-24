//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S E L E C T I O N   U T I L I T I E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn_types.h"
#include "tensor_types.h"

namespace opennn
{

class TrainingStrategy;
class NeuralNetwork;

// Scaffolding shared by the model-selection algorithms (growing inputs, genetic algorithm and
// growing neurons). InputsSelection and NeuronSelection are unrelated bases, so the common
// machinery lives here as free functions, next to the k-fold helpers in cross_validation.h.

struct CandidateEvaluation
{
    float training_error = MAX;
    float validation_error = MAX;
};

// Score the currently-configured candidate model: k-fold CV over fold_partition when
// folds_number > 1, otherwise trials_number random-restart trainings keeping the lowest
// validation error. After every training on_trial receives (trial, training_error,
// validation_error, improved) so each caller keeps its exact display and optimum bookkeeping.
// initialize_trial, when given, replaces the default set_parameters_random() before each trial
// (warm starts); the folds path never uses it.
CandidateEvaluation evaluate_candidate(TrainingStrategy*,
                                       NeuralNetwork*,
                                       Index folds_number,
                                       const vector<vector<Index>>& fold_partition,
                                       Index trials_number,
                                       bool use_validation_history_minimum,
                                       const function<void(Index, float, float, bool)>& on_trial,
                                       const function<void(Index)>& initialize_trial = {});

// Host copy of every parameter tensor, per layer and per view (tied slots skipped), taken
// before compile() rebuilds the parameter buffer for the next candidate architecture.
struct ParameterSnapshot
{
    struct Block { Shape shape; vector<float> values; };
    vector<vector<Block>> layers;
    bool empty() const noexcept { return layers.empty(); }
};

ParameterSnapshot capture_parameter_snapshot(NeuralNetwork*);

// Warm start: overwrite the region of each parameter tensor that overlaps its snapshot block
// (identical shapes fully, rank-1 prefix, rank-2 top-left block in row-major layout); the rest
// keeps its fresh random initialization. input_row_map (new row -> old row, -1 = keep random)
// remaps the first trainable layer's weight rows when input features were inserted mid-order.
void seed_parameters_from_snapshot(NeuralNetwork*,
                                   const ParameterSnapshot&,
                                   const vector<Index>& input_row_map = {});

// Install the winning parameters when they still fit the final architecture; otherwise refit on
// all development samples (k-fold mode) or retrain from random parameters.
void finalize_selected_model(TrainingStrategy*,
                             NeuralNetwork*,
                             const VectorR& optimal_parameters,
                             Index folds_number,
                             bool display,
                             const char* selected_label);

// JSON readers that fall back to a legacy key name.
float read_json_float_alias(const Json*, string_view primary, string_view legacy);
long long read_json_index_alias(const Json*, string_view primary, string_view legacy);

// ostringstream keeps the exact default float formatting of cout.
template <typename... Args>
string compose_message(const Args&... args)
{
    ostringstream stream;
    (stream << ... << args);
    return stream.str();
}

template <typename Condition>
struct StoppingCheck
{
    bool fired;
    Condition condition;
    string message;
};

// First-match stopping ladder: prints the fired check's message and returns its condition.
template <typename Condition>
optional<Condition> first_stopping_condition(const bool display,
                                             initializer_list<StoppingCheck<Condition>> checks)
{
    for (const auto& check : checks)
        if (check.fired)
        {
            if (display) cout << check.message;
            return check.condition;
        }

    return nullopt;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
