//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file adaptive_moment_estimation.h
 * @brief Declares the AdaptiveMomentEstimation (Adam) optimizer.
 */

#pragma once

#include "optimizer.h"

namespace opennn
{

struct BackPropagation;

/**
 * @class AdaptiveMomentEstimation
 * @brief Adam optimizer (Kingma & Ba, 2014).
 *
 * Maintains per-parameter exponential moving averages of the gradient
 * (m_t) and the squared gradient (v_t) and updates each parameter as
 * theta -= lr * m_hat / (sqrt(v_hat) + eps), with bias-corrected moments.
 * The optimizer is well suited to training deep networks with sparse or
 * noisy gradients.
 */
class AdaptiveMomentEstimation final : public Optimizer
{

public:

   /**
    * @enum DataSlot
    * @brief Slot indices into OptimizerData::views used by Adam.
    */
   enum DataSlot { GradientMoment,        ///< First moment estimate (running mean of gradients).
                   SquareGradientMoment   ///< Second moment estimate (running mean of squared gradients).
                 };

   /**
    * @brief Constructs the optimizer.
    * @param loss Loss to optimize; may be nullptr if set later.
    */
   AdaptiveMomentEstimation(Loss* loss = nullptr);
   /**
    * @brief Sets the mini-batch size used during training.
    * @param new_batch_size Number of samples per gradient update.
    */
   void set_batch_size(const Index new_batch_size);

   /** @brief Resets all hyperparameters to their default Adam values. */
   void set_default();
   /**
    * @brief Returns the number of training samples in the bound dataset.
    * @return Sample count, or 0 if the optimizer is not bound to a loss.
    */
   Index get_samples_number() const;
   /**
    * @brief Sets the base learning rate.
    *
    * Receives the learning rate (alpha) used to scale parameter updates.
    */
   void set_learning_rate(const float);
   /**
    * @brief Sets the first-moment decay rate.
    *
    * Receives beta_1, typically close to 1 (default 0.9).
    */
   void set_beta_1(const float);
   /**
    * @brief Sets the second-moment decay rate.
    *
    * Receives beta_2, typically close to 1 (default 0.999).
    */
   void set_beta_2(const float);
   /**
    * @brief Runs Adam to completion.
    * @return Per-epoch error history and the stopping condition that fired.
    */
   TrainingResults train() override;

   /**
    * @brief Applies one Adam parameter update.
    * @param back_propagation Gradient buffer for the current batch.
    * @param data Mutable optimizer state (moments, iteration counter).
    */
   void update_parameters(BackPropagation& back_propagation, OptimizerData& data) const;
   /**
    * @brief Loads optimizer hyperparameters from a parsed JSON document.
    */
   void from_JSON(const JsonDocument&) override;

   /**
    * @brief Writes optimizer hyperparameters to a streaming JSON writer.
    */
   void to_JSON(JsonWriter&) const override;

private:

   /** @brief Base learning rate (alpha). */
   float learning_rate = 0.001f;

   /** @brief First-moment exponential decay rate. */
   float beta_1 = 0.9f;

   /** @brief Second-moment exponential decay rate. */
   float beta_2 = 0.999f;

   /** @brief Mini-batch size used during training. */
   Index batch_size = 1000;
};

}
