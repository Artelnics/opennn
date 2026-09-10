// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/training/optimizer.h"

namespace opennn
{

class Network;
class Dense;
struct ForwardPropagation;

struct BackPropagationLM
{
    explicit BackPropagationLM(const Index = 0, Loss* = nullptr);

    float error = 0.0f;
    float regularization = 0.0f;
    float loss = 0.0f;

    VectorR errors;
    MatrixR squared_errors_jacobian;

    VectorR gradient;
    MatrixR hessian;

    vector<Index> dense_indices;
    vector<Index> parameter_offsets;
    vector<MatrixR> deltas;
    vector<MatrixR> activation_derivatives;
};

class LevenbergMarquardt final : public Optimizer
{

public:

   enum DataSlot { ParameterUpdate };

   explicit LevenbergMarquardt(Loss* = nullptr);

   void set_damping_parameter_factor(const float new_damping_parameter_factor) { damping_parameter_factor = new_damping_parameter_factor; }

   void set_minimum_loss_decrease(const float new_minimum_loss_decrease) { minimum_loss_decrease = new_minimum_loss_decrease; }
   TrainingResult train() override;
   void from_JSON(const JsonDocument&) override;

   void to_JSON(JsonWriter&) const override;

private:

   void back_propagate(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;

   void update_full_batch_parameters(const Batch&,
                                     ForwardPropagation&,
                                     BackPropagationLM&,
                                     OptimizerData&);

   void calculate_errors(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;

   void compute_jacobian(const Batch&,
                         const ForwardPropagation&,
                         BackPropagationLM&) const;

   VectorR potential_parameters;

   float damping_parameter = 0.0f;

   float initial_damping_parameter = 1.0e-3f;

   float minimum_damping_parameter = 1.0e-6f;

   float maximum_damping_parameter = 1.0e6f;

   float damping_parameter_factor = 10.0f;

   float minimum_loss_decrease = 0.0f;

};

}
