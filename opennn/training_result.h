//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   R E S U L T   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn_types.h"
#include "tensor_types.h"

namespace opennn
{

enum class StoppingCondition {MinimumLossDecrease,
                              LossGoal,
                              MaximumValidationErrorIncreases,
                              MaximumEpochsNumber,
                              MaximumTime};

struct OptimizerData
{
    OptimizerData() = default;
    virtual ~OptimizerData() = default;

    virtual void print() const;

    void set(const vector<Shape>&, Device device = Device::CPU);

    Buffer data;
    vector<TensorView> views;

    VectorR potential_parameters;
    VectorR training_direction;
    float initial_learning_rate = 0.0f;
    Index iteration = 0;

    Buffer graph_step{Device::CUDA};
    Buffer graph_effective_lr{Device::CUDA};
    Buffer graph_effective_eps{Device::CUDA};
};

struct TrainingResult
{
    TrainingResult(const Index = 0);
    virtual ~TrainingResult() = default;

    string write_stopping_condition() const;

    float get_training_error() const;

    float get_validation_error() const;

    // Number of completed epochs, i.e. the number of recorded history entries.
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

    float loss = NAN;

    Index validation_failures = 0;

    bool restored_best_parameters = false;

    Index restored_epoch = -1;

    float loss_decrease = 0.0f;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
