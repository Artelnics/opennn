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

    StochasticGradientDescent(Loss* = nullptr);

    void set_default();

    void set_batch_size(const Index);

    void set_initial_learning_rate(const float);
    void set_initial_decay(const float);
    void set_momentum(const float);
    void set_nesterov(bool);

    void update_parameters(BackPropagation&, OptimizerData&, float) const;
    void update_parameters_capturable(BackPropagation&, OptimizerData&) const;

    TrainingResult train() override;

    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    float initial_learning_rate;

    float initial_decay;

    float momentum = 0.0f;

    bool nesterov = false;

    Index batch_size = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
