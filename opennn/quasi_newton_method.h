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

class QuasiNewtonMethod final : public Optimizer
{

public:

    enum DataSlot {
        OldParameters,
        ParameterDifferences,
        ParameterUpdates,
        OldGradient,
        GradientDifference,
        OldInverseHessianDotGradientDifference,
        BFGS,
        InverseHessian,
        OldInverseHessian
    };

    QuasiNewtonMethod(Loss* = nullptr);
    void set_default();
    void set_minimum_loss_decrease(const float new_minimum_loss_decrease) { minimum_loss_decrease = new_minimum_loss_decrease; }
    void update_parameters(const Batch& , ForwardPropagation& , BackPropagation& , OptimizerData&);

    TrainingResult train() override;
    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    void calculate_inverse_hessian(OptimizerData&) const;

    pair<float, float> calculate_directional_point(const Batch&,
                                                 ForwardPropagation&,
                                                 BackPropagation&,
                                                 OptimizerData&,
                                                 float);

    float first_learning_rate = 0.01f;


    float minimum_loss_decrease = EPSILON;


    float training_slope = 0.0f;
    float learning_rate = 0.0f;
    float old_learning_rate = 0.0f;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
