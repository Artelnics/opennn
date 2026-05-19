//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

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

/// @brief Backprop scratch state specific to Levenberg-Marquardt (per-sample errors, Jacobian, Hessian approx).
struct BackPropagationLM
{
    /// @brief Constructs LM backprop state sized for the given sample count and loss.
    BackPropagationLM(const Index = 0, Loss* = nullptr);
    virtual ~BackPropagationLM() = default;

    /// @brief Resizes the LM backprop state for the given sample count and loss.
    void set(const Index = 0, Loss* = nullptr);

    Index samples_number = 0;

    VectorR output_deltas;
    Shape output_delta_dimensions;

    Loss* loss = nullptr;

    float error;
    float regularization = 0.0f;
    float loss_value = 0.0f;

    VectorR errors;
    VectorR squared_errors;
    MatrixR squared_errors_jacobian;

    VectorR gradient;
    MatrixR hessian;
};

/// @brief Levenberg-Marquardt optimizer combining Gauss-Newton and gradient descent through an adaptive damping factor.
class LevenbergMarquardtAlgorithm final : public Optimizer
{

public:

   /// @brief Constructs the LM optimizer optionally bound to a Loss instance.
   LevenbergMarquardtAlgorithm(Loss* = nullptr);
   /// @brief Resets all hyperparameters and damping bounds to library defaults.
   void set_default();

   /// @brief Sets the initial damping parameter (mu) controlling step interpolation.
   void set_damping_parameter(const float);

   /// @brief Sets the multiplicative factor used to grow/shrink the damping parameter between iterations.
   void set_damping_parameter_factor(const float);

   /// @brief Sets the lower clamp on the damping parameter.
   void set_minimum_damping_parameter(const float);
   /// @brief Sets the upper clamp on the damping parameter.
   void set_maximum_damping_parameter(const float);
   /// @brief Sets the minimum loss decrease that counts as progress between iterations.
   void set_minimum_loss_decrease(const float);
   /// @brief Runs the Levenberg-Marquardt training loop and returns the recorded error history.
   TrainingResults train() override;

   /// @brief Slot index into the optimizer scratch buffer (parameter update vector).
   enum DataSlot { ParameterUpdate };

   /// @brief Solves the LM linear system for the parameter update and adapts the damping parameter.
   void update_parameters(
           const Batch&,
           ForwardPropagation&,
           BackPropagationLM&,
           OptimizerData&);
   /// @brief Restores hyperparameters from a JSON document.
   void from_JSON(const JsonDocument&) override;

   /// @brief Serializes hyperparameters to JSON.
   void to_JSON(JsonWriter&) const override;

private:

   void back_propagate(const Batch&, const ForwardPropagation&, BackPropagationLM&);

   void calculate_errors(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;
   void calculate_squared_errors(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;
   void calculate_error(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;

   void compute_jacobian(const Batch& batch,
                         const ForwardPropagation& forward_propagation,
                         BackPropagationLM& back_propagation_lm);

   void insert_dense_jacobian(const Dense* layer,
                              const ForwardPropagation& forward_propagation,
                              Index layer_index,
                              Index parameter_offset,
                              MatrixR& jacobian);
   float damping_parameter = 0.0f;

   float minimum_damping_parameter = 0.0f;

   float maximum_damping_parameter = 0.0f;

   float damping_parameter_factor = 0.0f;

   // Stopping criteria

   float minimum_loss_decrease = 0.0f;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
