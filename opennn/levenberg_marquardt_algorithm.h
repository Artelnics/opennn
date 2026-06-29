//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "optimizer.h"

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

    Loss* loss_pointer = nullptr;

    float error;
    float regularization = 0.0f;
    float loss = 0.0f;

    VectorR errors;
    VectorR squared_errors;
    MatrixR squared_errors_jacobian;

    VectorR gradient;
    MatrixR hessian;
};

class LevenbergMarquardtAlgorithm final : public Optimizer
{

public:

   explicit LevenbergMarquardtAlgorithm(Loss* = nullptr);
   void set_default();

   void set_damping_parameter(const float);

   void set_damping_parameter_factor(const float);

   void set_minimum_damping_parameter(const float);
   void set_maximum_damping_parameter(const float);
   void set_minimum_loss_decrease(const float);
   TrainingResult train() override;

   enum DataSlot { ParameterUpdate };

   void update_parameters(
           const Batch&,
           ForwardPropagation&,
           BackPropagationLM&,
           OptimizerData&);
   void from_JSON(const JsonDocument&) override;

   void to_JSON(JsonWriter&) const override;

private:

   void back_propagate(const Batch&, const ForwardPropagation&, BackPropagationLM&);

   void calculate_errors(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;
   void calculate_squared_errors(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;
   void calculate_error(const Batch&, const ForwardPropagation&, BackPropagationLM&) const;

   void compute_jacobian(const Batch&,
                         const ForwardPropagation&,
                         BackPropagationLM&);

   float initial_damping_parameter = 0.0f;

   float damping_parameter = 0.0f;

   float minimum_damping_parameter = 0.0f;

   float maximum_damping_parameter = 0.0f;

   float damping_parameter_factor = 0.0f;


   float minimum_loss_decrease = 0.0f;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
