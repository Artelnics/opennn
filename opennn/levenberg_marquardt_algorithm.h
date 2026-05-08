//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file levenberg_marquardt_algorithm.h
 * @brief Declares the LevenbergMarquardtAlgorithm optimizer and the
 *        BackPropagationLM helper structure that holds its per-iteration
 *        scratch state.
 */

#pragma once

#include "layer.h"
#include "batch.h"
#include "dense_layer.h"
#include "optimizer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class NeuralNetwork;
struct ForwardPropagation;

/**
 * @struct BackPropagationLM
 * @brief Scratch state used by LevenbergMarquardtAlgorithm.
 *
 * Holds per-sample errors and squared errors, the Jacobian of the squared
 * errors with respect to parameters, and the resulting approximate
 * gradient and Hessian used by the LM update.
 */
struct BackPropagationLM
{
    /**
     * @brief Constructs the scratch state.
     * @param samples_number Number of samples that will be processed.
     * @param loss Loss to which the state is bound.
     */
    BackPropagationLM(const Index samples_number = 0, Loss* loss = nullptr);
    /** @brief Virtual destructor. */
    virtual ~BackPropagationLM() = default;

    /**
     * @brief (Re)allocates buffers for the given dataset.
     * @param samples_number Number of samples to allocate for.
     * @param loss Loss to bind.
     */
    void set(const Index samples_number = 0, Loss* loss = nullptr);

    /** @brief Number of samples this scratch is sized for. */
    Index samples_number = 0;

    /** @brief Per-sample output deltas. */
    VectorR output_deltas;
    /** @brief Shape of @ref output_deltas as a multi-dimensional view. */
    Shape output_delta_dimensions;

    /** @brief Loss to which this scratch is bound; not owned. */
    Loss* loss = nullptr;

    /** @brief Mean error over the batch. */
    float error;
    /** @brief Regularization contribution to the loss. */
    float regularization = 0.0f;
    /** @brief Total loss = error + regularization. */
    float loss_value = 0.0f;

    /** @brief Per-sample residual errors. */
    VectorR errors;
    /** @brief Per-sample squared errors. */
    VectorR squared_errors;
    /** @brief Jacobian of squared_errors with respect to the parameters. */
    MatrixR squared_errors_jacobian;

    /** @brief Approximate gradient (J^T * e). */
    VectorR gradient;
    /** @brief Approximate Hessian (J^T * J + lambda * I). */
    MatrixR hessian;
};

/**
 * @class LevenbergMarquardtAlgorithm
 * @brief Levenberg-Marquardt optimizer with adaptive damping.
 *
 * Trust-region method that interpolates between gradient descent (large
 * damping) and Gauss-Newton (small damping). At each iteration it solves
 * (J^T J + lambda * I) d = -J^T r, accepting the step when the loss
 * decreases (and decreasing lambda) or rejecting it (and increasing
 * lambda) otherwise.
 *
 * Best suited to small / medium dense networks trained on regression
 * losses; not suited to very large models or non-twice-differentiable losses.
 */
class LevenbergMarquardtAlgorithm final : public Optimizer
{

public:

   /**
    * @brief Constructs the optimizer.
    * @param loss Loss to optimize; may be nullptr if set later.
    */
   LevenbergMarquardtAlgorithm(Loss* loss = nullptr);
   /** @brief Resets all hyperparameters to their default values. */
   void set_default();

   /**
    * @brief Sets the initial damping parameter (lambda).
    *
    * Receives the new damping value.
    */
   void set_damping_parameter(const float);

   /**
    * @brief Sets the multiplicative factor used to grow / shrink lambda.
    *
    * Receives the factor (>1); lambda is multiplied by it on rejected
    * steps and divided by it on accepted steps.
    */
   void set_damping_parameter_factor(const float);

   /**
    * @brief Sets the lower bound for the damping parameter.
    *
    * Receives the minimum lambda allowed.
    */
   void set_minimum_damping_parameter(const float);
   /**
    * @brief Sets the upper bound for the damping parameter.
    *
    * Receives the maximum lambda allowed.
    */
   void set_maximum_damping_parameter(const float);
   /**
    * @brief Sets the minimum acceptable loss decrease between iterations.
    *
    * Receives the threshold below which training stops.
    */
   void set_minimum_loss_decrease(const float);
   /**
    * @brief Runs the LM algorithm to completion.
    * @return Per-epoch error history and the stopping condition that fired.
    */
   TrainingResults train() override;

   /**
    * @enum DataSlot
    * @brief Slot indices into OptimizerData::views used by LM.
    */
   enum DataSlot { ParameterUpdate };

   /**
    * @brief Applies one LM parameter update.
    * @param batch Current training batch.
    * @param forward_propagation Forward intermediates for the batch.
    * @param back_propagation_lm Scratch holding the Jacobian and approximate Hessian.
    * @param data Mutable optimizer state.
    */
   void update_parameters(
           const Batch& batch,
           ForwardPropagation& forward_propagation,
           BackPropagationLM& back_propagation_lm,
           OptimizerData& data);
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
     * @brief Computes the gradient by finite differences (debug helper).
     * @return Numerical gradient with respect to the parameters.
     */
    VectorR calculate_numerical_gradient();
    /**
     * @brief Computes the Jacobian by finite differences (debug helper).
     * @return Numerical Jacobian of the residuals with respect to parameters.
     */
    MatrixR calculate_numerical_jacobian();
    /**
     * @brief Computes the Hessian by finite differences (debug helper).
     * @return Numerical Hessian of the loss with respect to parameters.
     */
    MatrixR calculate_numerical_hessian();

   /**
    * @brief Performs a complete LM-flavored backward pass over the batch,
    *        filling residuals, Jacobian, gradient and Hessian in @p bp.
    * @param batch Current training batch.
    * @param forward_propagation Forward intermediates.
    * @param bp LM scratch to fill.
    */
   void back_propagate(const Batch& batch,
                       const ForwardPropagation& forward_propagation,
                       BackPropagationLM& bp);

   /**
    * @brief Computes per-sample residual errors into @p bp.errors.
    */
   void calculate_errors(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;
   /**
    * @brief Computes per-sample squared errors into @p bp.squared_errors.
    */
   void calculate_squared_errors(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;
   /**
    * @brief Computes the mean error into @p bp.error.
    */
   void calculate_error(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;

   /**
    * @brief Assembles the squared-errors Jacobian over all layers.
    * @param batch Current training batch.
    * @param forward_propagation Forward intermediates.
    * @param back_propagation_lm Scratch receiving the Jacobian.
    */
   void compute_jacobian(const Batch& batch,
                         const ForwardPropagation& forward_propagation,
                         BackPropagationLM& back_propagation_lm);

   /**
    * @brief Fills the columns of @p jacobian corresponding to a Dense layer.
    * @param layer Pointer to the Dense layer.
    * @param forward_propagation Forward intermediates.
    * @param layer_index Index of the layer in the network.
    * @param parameter_offset Offset of this layer's parameters in the full Jacobian.
    * @param jacobian Output Jacobian being assembled.
    */
   void insert_dense_jacobian(const Dense* layer,
                              const ForwardPropagation& forward_propagation,
                              Index layer_index,
                              Index parameter_offset,
                              MatrixR& jacobian);
   /** @brief Current damping parameter (lambda). */
   float damping_parameter = 0.0f;

   /** @brief Lower bound for @ref damping_parameter. */
   float minimum_damping_parameter = 0.0f;

   /** @brief Upper bound for @ref damping_parameter. */
   float maximum_damping_parameter = 0.0f;

   /** @brief Multiplicative factor applied to lambda on accepted / rejected steps. */
   float damping_parameter_factor = 0.0f;

   /** @brief Stopping threshold on the per-iteration loss decrease. */
   float minimum_loss_decrease = 0.0f;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
