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

struct CandidateEvaluation
{
    float training_error = MAX;
    float validation_error = MAX;
};

CandidateEvaluation evaluate_candidate(TrainingStrategy*,
                                       NeuralNetwork*,
                                       Index folds_number,
                                       const vector<vector<Index>>& fold_partition,
                                       Index trials_number,
                                       bool use_validation_history_minimum,
                                       const function<void(Index, float, float, bool)>& on_trial,
                                       const function<void(Index)>& initialize_trial = {});

struct ParameterSnapshot
{
    struct Block { Shape shape; vector<float> values; };
    vector<vector<Block>> layers;
    bool empty() const noexcept { return layers.empty(); }
};

ParameterSnapshot capture_parameter_snapshot(NeuralNetwork*);

void seed_parameters_from_snapshot(NeuralNetwork*,
                                   const ParameterSnapshot&,
                                   const vector<Index>& input_row_map = {});

void finalize_selected_model(TrainingStrategy*,
                             NeuralNetwork*,
                             const VectorR& optimal_parameters,
                             Index folds_number,
                             bool display,
                             const char* selected_label);

float read_json_float_alias(const Json*, string_view primary, string_view legacy);
long long read_json_index_alias(const Json*, string_view primary, string_view legacy);

template <typename Condition>
struct StoppingCheck
{
    bool fired;
    Condition condition;
    string message;
};

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
