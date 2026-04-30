//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D    C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "loss.h"
#include "optimizer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

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

    // Set

    void set_default();

    // Stopping criteria

    void set_minimum_loss_decrease(const float new_minimum_loss_decrease) { minimum_loss_decrease = new_minimum_loss_decrease; }

    // Training

    void update_parameters(const Batch& , ForwardPropagation& , BackPropagation& , OptimizerData&);

    TrainingResults train() override;

    // Serialization

    void from_XML(const XmlDocument&) override;

    void to_XML(XmlPrinter&) const override;

private:

    void calculate_inverse_hessian(OptimizerData&) const;

    pair<float, float> calculate_directional_point(const Batch&,
                                                 ForwardPropagation&,
                                                 BackPropagation&,
                                                 OptimizerData&,
                                                 float);

    float first_learning_rate = float(0.01);

    // Stopping criteria

    float minimum_loss_decrease = EPSILON;

    // Optimizer-specific state (not shared across optimizers, so not in OptimizerData)

    float training_slope = float(0);
    float learning_rate = float(0);
    float old_learning_rate = float(0);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
