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
    float get_learning_rate() const { return learning_rate; }
    void set_beta_1(const float);
    void set_beta_2(const float);

    // Gradient accumulation: with period K > 1 the parameter update runs once
    // every K mini-batches on the mean of their gradients -- one optimizer
    // step over a virtual batch of K * batch_size samples whose activation
    // memory stays O(batch_size). Exact for equal-sized mini-batches (the
    // standard convention); Adam's bias correction advances per update.
    void set_update_period(const Index new_period)
    {
        throw_if(new_period < 1, "update period must be >= 1.");
        update_period = new_period;
    }
    Index get_update_period() const noexcept { return update_period; }

    void update_parameters(BackPropagation&, OptimizerData&) override;

    void update_parameters_capturable(BackPropagation&, OptimizerData&) const;

    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    string get_display_name() const override { return "adaptive moment estimation \"Adam\""; }
    void setup_optimizer_data(OptimizerData&, Index, Device, bool) override;

    float learning_rate = 0.001f;

    float beta_1 = 0.9f;

    float beta_2 = 0.999f;

    Index update_period = 1;

    Buffer gradient_accumulator;
    Index accumulated_batches = 0;
};

}
