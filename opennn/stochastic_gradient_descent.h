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

    enum DataSlot { Velocity };

    explicit StochasticGradientDescent(Loss* = nullptr);

    void set_default();

    void set_batch_size(const Index);

    void set_initial_learning_rate(const float);
    float get_initial_learning_rate() const { return initial_learning_rate; }
    void set_initial_decay(const float);
    void set_momentum(const float);
    void set_nesterov(bool);

    void update_parameters(BackPropagation&, OptimizerData&) override;
    void update_parameters_capturable(BackPropagation&, OptimizerData&) const;

    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    string get_display_name() const override { return "stochastic gradient descent (SGD)"; }
    void setup_optimizer_data(OptimizerData&, Index, Device, bool) override;
    void on_epoch_begin(Index, OptimizerData&) override;

    float initial_learning_rate;

    float initial_decay;

    float momentum = 0.0f;

    bool nesterov = false;

    float current_learning_rate = 0.0f;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
