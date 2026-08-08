//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "optimizer.h"

namespace opennn
{

struct BackPropagation;

class StochasticGradientDescent final : public Optimizer
{

public:

    enum DataSlot { Velocity, GraphLearningRate };

    explicit StochasticGradientDescent(Loss* = nullptr);

    void set_default();

    void set_batch_size(const Index new_batch_size) { batch_size = new_batch_size; }

    void set_initial_learning_rate(const float new_learning_rate) { initial_learning_rate = new_learning_rate; }
    float get_initial_learning_rate() const { return initial_learning_rate; }
    void set_initial_decay(const float new_decay) { initial_decay = new_decay; }
    void set_momentum(const float new_momentum) { momentum = new_momentum; }
    void set_nesterov(bool new_nesterov_momentum) { nesterov = new_nesterov_momentum; }

    void update_parameters(BackPropagation&, OptimizerData&) override;
    void update_parameters_capturable(BackPropagation&, OptimizerData&) const override;

    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    string get_display_name() const override { return "stochastic gradient descent (SGD)"; }
    bool supports_cuda_graph() const noexcept override { return true; }
    void setup_optimizer_data(OptimizerData&, Index, Device) override;
    void on_epoch_begin(Index, OptimizerData&) override;

    float initial_learning_rate;

    float initial_decay;

    float momentum = 0.0f;

    bool nesterov = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
