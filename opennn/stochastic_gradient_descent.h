//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file stochastic_gradient_descent.h
 * @brief Declares the StochasticGradientDescent (SGD) optimizer with
 *        optional momentum and Nesterov acceleration.
 */

#pragma once

#include "optimizer.h"

namespace opennn
{

struct BackPropagation;

/**
 * @class StochasticGradientDescent
 * @brief Mini-batch SGD with optional momentum, Nesterov acceleration and
 *        learning-rate decay.
 *
 * Updates parameters as theta -= lr * grad. With momentum, accumulates a
 * velocity vector v = momentum * v + grad and steps along v; the Nesterov
 * variant evaluates the gradient at theta - lr * momentum * v to obtain a
 * lookahead update.
 */
class StochasticGradientDescent final : public Optimizer
{

public:

    /**
     * @enum DataSlot
     * @brief Slot indices into OptimizerData::views used by SGD.
     */
    enum DataSlot { ParameterUpdate,      ///< Current parameter increment.
                    LastParameterUpdate   ///< Previous increment (used by momentum).
                  };

    /**
     * @brief Constructs the optimizer.
     * @param loss Loss to optimize; may be nullptr if set later.
     */
    StochasticGradientDescent(Loss* loss = nullptr);

    /** @brief Resets all hyperparameters to their default values. */
    void set_default();

    /**
     * @brief Sets the mini-batch size used during training.
     *
     * Receives the number of samples per gradient update.
     */
    void set_batch_size(const Index);

    /**
     * @brief Returns the number of training samples in the bound dataset.
     * @return Sample count, or 0 if the optimizer is not bound to a loss.
     */
    Index get_samples_number() const;

    /**
     * @brief Sets the base learning rate (before any decay).
     *
     * Receives the learning rate used at the first epoch.
     */
    void set_initial_learning_rate(const float);
    /**
     * @brief Sets the per-epoch learning-rate decay factor.
     *
     * Receives the decay rate; the effective learning rate at epoch t is
     * lr / (1 + decay * t).
     */
    void set_initial_decay(const float);
    /**
     * @brief Sets the momentum coefficient.
     *
     * Receives the momentum value (0 disables momentum).
     */
    void set_momentum(const float);
    /**
     * @brief Toggles Nesterov accelerated gradient.
     *
     * Receives true to enable Nesterov, false to use vanilla momentum.
     */
    void set_nesterov(bool);

    /**
     * @brief Applies one SGD parameter update.
     * @param back_propagation Gradient buffer for the current batch.
     * @param data Mutable optimizer state (last update buffer).
     * @param learning_rate Effective learning rate for this epoch.
     */
    void update_parameters(BackPropagation& back_propagation,
                           OptimizerData& data,
                           float learning_rate) const;

    /**
     * @brief Runs SGD to completion.
     * @return Per-epoch error history and the stopping condition that fired.
     */
    TrainingResults train() override;

    /**
     * @brief Loads optimizer hyperparameters from a parsed JSON document.
     */
    void from_JSON(const JsonDocument&) override;

    /**
     * @brief Writes optimizer hyperparameters to a streaming JSON writer.
     */
    void to_JSON(JsonWriter&) const override;

private:

    /** @brief Initial learning rate (before decay). */
    float initial_learning_rate;

    /** @brief Per-epoch decay factor applied to the learning rate. */
    float initial_decay;

    /** @brief Momentum coefficient (0 disables momentum). */
    float momentum = 0.0f;

    /** @brief True to use Nesterov accelerated gradient instead of vanilla momentum. */
    bool nesterov = false;

    /** @brief Mini-batch size used during training. */
    Index batch_size = 1000;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
