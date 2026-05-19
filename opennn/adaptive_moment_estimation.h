//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "optimizer.h"

namespace opennn
{

struct BackPropagation;

/// @brief Adam optimizer with first/second gradient moments for stochastic minibatch training.
class AdaptiveMomentEstimation final : public Optimizer
{

public:

   /// @brief Slot indices into the optimizer scratch buffer (m_t and v_t moments).
   enum DataSlot { GradientMoment, SquareGradientMoment };

   /// @brief Constructs Adam optionally bound to a Loss instance.
   AdaptiveMomentEstimation(Loss* = nullptr);

   /// @brief Returns the number of training samples seen by the bound dataset.
   Index get_samples_number() const;

   /// @brief Sets the minibatch size used by train().
   void set_batch_size(const Index);

   /// @brief Resets all hyperparameters (learning rate, betas, stopping criteria) to library defaults.
   void set_default();

   /// @brief Sets the base learning rate alpha.
   void set_learning_rate(const float);
   /// @brief Sets the first-moment decay rate beta_1.
   void set_beta_1(const float);
   /// @brief Sets the second-moment decay rate beta_2.
   void set_beta_2(const float);

   /// @brief Runs the Adam training loop and returns the recorded error history.
   TrainingResults train() override;

   /// @brief Applies one Adam update to the network parameters using the gradient in back_propagation.
   void update_parameters(BackPropagation&, OptimizerData&) const;

   /// @brief Restores hyperparameters from a JSON document.
   void from_JSON(const JsonDocument&) override;

   /// @brief Serializes hyperparameters to JSON.
   void to_JSON(JsonWriter&) const override;

private:

   float learning_rate = 0.001f;

   float beta_1 = 0.9f;

   float beta_2 = 0.999f;

   Index batch_size = 0;
};

}
