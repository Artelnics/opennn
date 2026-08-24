//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/training_strategy/optimizer.h"

namespace opennn
{

class NeuralNetwork;
class Dense;
struct ForwardPropagation;

struct BackPropagationLM
{
    explicit BackPropagationLM(const Index = 0, Loss* = nullptr);
    virtual ~BackPropagationLM() = default;

    void set(const Index = 0, Loss* = nullptr);

    Index samples_number = 0;

    float error;
    float regularization = 0.0f;
    float loss = 0.0f;

    VectorR errors;
    VectorR squared_errors;
    MatrixR squared_errors_jacobian;

    VectorR gradient;
    MatrixR hessian;

    vector<Index> dense_indices;
    vector<Index> parameter_offsets;
    vector<MatrixR> deltas;
    vector<MatrixR> activation_derivatives;
};

class LevenbergMarquardtAlgorithm final : public Optimizer
{

public:

   enum DataSlot { ParameterUpdate };

   explicit LevenbergMarquardtAlgorithm(Loss* = nullptr);
   void set_default();

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
   void calculate_squared_errors(const Batch&,
                                                           const ForwardPropagation&,
                                                           BackPropagationLM& back_propagation_lm) const { back_propagation_lm.squared_errors = back_propagation_lm.errors.array().square(); }
   void calculate_error(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;

   void compute_jacobian(const Batch&,
                         const ForwardPropagation&,
                         BackPropagationLM&) const;

   // The real values. These used to read 0.0f here and be assigned properly in
   // set_default(), so the declaration said the damping started at nothing.
   float initial_damping_parameter = 1.0e-3f;

   float minimum_damping_parameter = 1.0e-6f;

   float maximum_damping_parameter = 1.0e6f;

   float damping_parameter_factor = 10.0f;

   float minimum_loss_decrease = 0.0f;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
