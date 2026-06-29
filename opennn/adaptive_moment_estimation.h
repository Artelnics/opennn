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

class AdaptiveMomentEstimation final : public Optimizer
{

public:

    enum DataSlot { GradientMoment, SquareGradientMoment };

    explicit AdaptiveMomentEstimation(Loss* = nullptr);

    void set_batch_size(const Index);

    void set_default();

    void set_learning_rate(const float);
    void set_beta_1(const float);
    void set_beta_2(const float);

    void set_gradient_clip_norm(const float new_clip) { gradient_clip_norm = new_clip; }
    float get_gradient_clip_norm() const noexcept { return gradient_clip_norm; }

    TrainingResult train() override;

    void update_parameters(BackPropagation&, OptimizerData&) const;

    void update_parameters_capturable(BackPropagation&, OptimizerData&) const;

    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    float learning_rate = 0.001f;

    float beta_1 = 0.9f;

    float beta_2 = 0.999f;

    float gradient_clip_norm = 1.0f;

    Index batch_size = 0;
};

}
