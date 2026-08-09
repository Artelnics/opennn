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

struct OptimizerData
{
    void set(const vector<Shape>&, Device device = Device::CPU);

    Buffer data;
    vector<TensorView> views;

    VectorR potential_parameters;
    VectorR training_direction;
    float initial_learning_rate = 0.0f;
    Index iteration = 0;

    Buffer gradient_accumulator;
    Index accumulated_batches = 0;
    float current_learning_rate = 0.0f;
    float training_slope = 0.0f;
    float learning_rate = 0.0f;
    float old_learning_rate = 0.0f;
    float damping_parameter = 0.0f;
};

struct TrainingResult
{
    TrainingResult(const Index = 0);
    virtual ~TrainingResult() = default;

    string write_stopping_condition() const;

    float get_training_error() const;

    float get_validation_error() const;

    Index get_epochs_number() const;

    void save(const filesystem::path&) const;

    void print(const string& message = {}) const;

    optional<StoppingCondition> stopping_condition;

    Tensor<string, 2> write_override_results(const Index = 3) const;

    void resize_training_error_history(const Index);

    void resize_validation_error_history(const Index);

    VectorR training_error_history;

    VectorR validation_error_history;

    string elapsed_time;

    float loss = QUIET_NAN;

    optional<Index> restored_epoch;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
