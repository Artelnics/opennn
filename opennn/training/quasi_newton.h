// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

#pragma once

#include "opennn/training/optimizer.h"

namespace opennn
{

class QuasiNewton final : public Optimizer
{

public:

    enum DataSlot {
        OldParameters,
        ParameterDifferences,
        ParameterUpdates,
        OldGradient,
        GradientDifference,
        BfgsWork,
        InverseHessian,
        OldInverseHessian
    };

    explicit QuasiNewton(Loss* = nullptr);
    void set_minimum_loss_decrease(const float new_minimum_loss_decrease) { minimum_loss_decrease = new_minimum_loss_decrease; }
    TrainingResult train() override;
    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    void update_full_batch_parameters(const Batch&,
                                      ForwardPropagation&,
                                      BackPropagation&,
                                      OptimizerData&);

    void calculate_inverse_hessian(OptimizerData&) const;

    struct LineSearchState
    {
        void reset(Index parameters_number)
        {
            direction.resize(parameters_number);
            potential.resize(parameters_number);
            initial = 0.0f;
            learning_rate = 0.0f;
            old_learning_rate = 0.0f;
            slope = 0.0f;
        }

        VectorR direction;
        VectorR potential;
        float initial = 0.0f;
        float learning_rate = 0.0f;
        float old_learning_rate = 0.0f;
        float slope = 0.0f;
    };

    pair<float, float> backtrack_along_direction(const Batch&,
                                                 ForwardPropagation&,
                                                 BackPropagation&,
                                                 float);

    LineSearchState line_search;

    float first_learning_rate = 0.01f;

    float minimum_loss_decrease = 1.0e-6f;
};

}
