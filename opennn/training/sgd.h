// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/training/optimizer.h"

namespace opennn
{

struct BackPropagation;

class SGD final : public Optimizer
{

public:

    enum DataSlot { Velocity, GraphLearningRate };

    explicit SGD(Loss* = nullptr);

    void set_initial_learning_rate(const float new_learning_rate) { initial_learning_rate = new_learning_rate; }
    float get_initial_learning_rate() const { return initial_learning_rate; }
    void set_initial_decay(const float new_decay) { initial_decay = new_decay; }
    void set_momentum(const float new_momentum) { momentum = new_momentum; }
    void set_nesterov(bool new_nesterov_momentum) { nesterov = new_nesterov_momentum; }

    void update_parameters(BackPropagation&, OptimizerData&,
                           UpdateMode = UpdateMode::Standard) override;

    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    string get_display_name() const override { return "stochastic gradient descent (SGD)"; }
    bool supports_cuda_graph() const noexcept override { return true; }
    void setup_optimizer_data(OptimizerData&, Index, Device) override;
    void on_epoch_begin(Index, OptimizerData&) override;

    float initial_learning_rate = 0.001f;

    float initial_decay = 0.001f;

    float momentum = 0.0f;

    float current_learning_rate = 0.0f;

    bool nesterov = false;
};

}
