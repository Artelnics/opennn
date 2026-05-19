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

/// @brief Stochastic gradient descent with optional momentum, Nesterov, and learning-rate decay.
class StochasticGradientDescent final : public Optimizer
{

public:

    /// @brief Slot index into the optimizer scratch buffer (momentum velocity).
    enum DataSlot { Velocity };

    /// @brief Constructs SGD optionally bound to a Loss instance.
    StochasticGradientDescent(Loss* = nullptr);

    /// @brief Resets all hyperparameters (learning rate, decay, momentum, Nesterov) to library defaults.
    void set_default();

    /// @brief Sets the minibatch size used by train().
    void set_batch_size(const Index);

    /// @brief Returns the number of training samples seen by the bound dataset.
    Index get_samples_number() const;

    /// @brief Sets the initial learning rate eta_0.
    void set_initial_learning_rate(const float);
    /// @brief Sets the learning-rate decay applied each epoch.
    void set_initial_decay(const float);
    /// @brief Sets the momentum coefficient (0 disables momentum).
    void set_momentum(const float);
    /// @brief Enables or disables Nesterov-accelerated momentum.
    void set_nesterov(bool);

    /// @brief Applies one SGD update to the network parameters using the gradient and current learning rate.
    void update_parameters(BackPropagation&, OptimizerData&, float) const;

    /// @brief Runs the SGD training loop and returns the recorded error history.
    TrainingResults train() override;

    /// @brief Restores hyperparameters from a JSON document.
    void from_JSON(const JsonDocument&) override;

    /// @brief Serializes hyperparameters to JSON.
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
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
