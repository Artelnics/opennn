// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/core/opennn_types.h"
#include "opennn/core/log.h"
#include "opennn/core/scaling.h"
#include "opennn/core/tensor_types.h"

namespace opennn
{

class Training;
class Network;
class Dataset;

struct CandidateEvaluation
{
    float training_error = MAX;
    float validation_error = MAX;
};

CandidateEvaluation evaluate_candidate(Training*,
                                       Network*,
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

FeatureScaling capture_input_scaling(Dataset*);

void apply_input_scaling(Network*, FeatureScaling);

ParameterSnapshot capture_parameter_snapshot(Network*);

void seed_parameters_from_snapshot(Network*,
                                   const ParameterSnapshot&,
                                   const vector<Index>& input_row_map = {});

void finalize_selected_model(Training*,
                             Network*,
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
    const auto check = ranges::find(checks, true, &StoppingCheck<Condition>::fired);
    if (check != checks.end())
    {
        if (display) logging::info() << check->message;
        return check->condition;
    }

    return nullopt;
}

}
