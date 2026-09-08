//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   R E S U L T   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"
#include "opennn/core/tensor_types.h"

namespace opennn
{

enum class StoppingCondition {MinimumLossDecrease,
                              LossGoal,
                              MaximumValidationErrorIncreases,
                              MaximumEpochsNumber,
                              MaximumTime};

struct TrainingResult
{
    TrainingResult(const Index = 0);
    virtual ~TrainingResult() = default;

    string write_stopping_condition() const;

    float get_training_error() const;

    float get_validation_error() const;

    Index get_epochs_number() const { return training_error_history.size(); }

    void save(const filesystem::path&) const;

    void print(const string& message = {}) const;

    optional<StoppingCondition> stopping_condition;

    Tensor<string, 2> write_override_results(const Index = 3) const;

    void resize_training_error_history(const Index new_size) { training_error_history.conservativeResize(new_size); }

    void resize_validation_error_history(const Index new_size) { validation_error_history.conservativeResize(new_size); }

    VectorR training_error_history;

    VectorR validation_error_history;

    string elapsed_time;

    double training_seconds = 0.0;

    float loss = QUIET_NAN;

    optional<Index> restored_epoch;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
