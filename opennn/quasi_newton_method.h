//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D    C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file quasi_newton_method.h
 * @brief Declares the QuasiNewtonMethod optimizer (BFGS).
 */

#pragma once

#include "loss.h"
#include "optimizer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

/**
 * @class QuasiNewtonMethod
 * @brief BFGS quasi-Newton optimizer with line search.
 *
 * Maintains an approximate inverse Hessian H_k that is updated at each
 * iteration using the BFGS formula based on the parameter and gradient
 * differences between consecutive iterations. The search direction is
 * d = -H * grad and a line search selects the step length.
 *
 * Best suited to small / medium networks where evaluating the full
 * gradient over the dataset is cheap; not suited to very large models.
 */
class QuasiNewtonMethod final : public Optimizer
{

public:

    /**
     * @enum DataSlot
     * @brief Slot indices into OptimizerData::views used by BFGS.
     */
    enum DataSlot {
        OldParameters,                            ///< Parameters at the previous iteration.
        ParameterDifferences,                     ///< theta_k - theta_{k-1}.
        ParameterUpdates,                         ///< Step computed at the current iteration.
        OldGradient,                              ///< Gradient at the previous iteration.
        GradientDifference,                       ///< grad_k - grad_{k-1}.
        OldInverseHessianDotGradientDifference,   ///< H_{k-1} * y, scratch term used in the BFGS update.
        BFGS,                                     ///< BFGS update term.
        InverseHessian,                           ///< Current approximate inverse Hessian.
        OldInverseHessian                         ///< Previous approximate inverse Hessian.
    };

    /**
     * @brief Constructs the optimizer.
     * @param loss Loss to optimize; may be nullptr if set later.
     */
    QuasiNewtonMethod(Loss* loss = nullptr);
    /** @brief Resets all hyperparameters to their default values. */
    void set_default();
    /**
     * @brief Sets the minimum acceptable loss decrease between iterations.
     * @param new_minimum_loss_decrease Threshold below which training stops.
     */
    void set_minimum_loss_decrease(const float new_minimum_loss_decrease) { minimum_loss_decrease = new_minimum_loss_decrease; }
    /**
     * @brief Applies one BFGS parameter update.
     * @param batch Current training batch.
     * @param forward_propagation Forward intermediates for the batch.
     * @param back_propagation Gradient buffer for the batch.
     * @param data Mutable optimizer state (inverse Hessian, history).
     */
    void update_parameters(const Batch& batch,
                           ForwardPropagation& forward_propagation,
                           BackPropagation& back_propagation,
                           OptimizerData& data);

    /**
     * @brief Runs BFGS to completion.
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

    /**
     * @brief Updates the approximate inverse Hessian using the BFGS formula.
     * @param data Mutable optimizer state holding the previous and current
     *             parameter / gradient differences.
     */
    void calculate_inverse_hessian(OptimizerData& data) const;

    /**
     * @brief Performs a line search along the current search direction.
     * @param batch Current training batch.
     * @param forward_propagation Forward intermediates for the batch.
     * @param back_propagation Gradient buffer for the batch.
     * @param data Mutable optimizer state.
     * @param initial_step Initial step length used to bracket the minimum.
     * @return Pair (step_length, loss) at the chosen point.
     */
    pair<float, float> calculate_directional_point(const Batch& batch,
                                                 ForwardPropagation& forward_propagation,
                                                 BackPropagation& back_propagation,
                                                 OptimizerData& data,
                                                 float initial_step);

    /** @brief Initial step length for the line search. */
    float first_learning_rate = 0.01f;

    /** @brief Stopping threshold on the per-iteration loss decrease. */
    float minimum_loss_decrease = EPSILON;

    /** @brief Directional derivative of the loss along the current search direction. */
    float training_slope = 0.0f;
    /** @brief Step length chosen at the current iteration. */
    float learning_rate = 0.0f;
    /** @brief Step length chosen at the previous iteration. */
    float old_learning_rate = 0.0f;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
