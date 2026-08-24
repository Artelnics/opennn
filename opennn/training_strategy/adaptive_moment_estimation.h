//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/training_strategy/optimizer.h"

namespace opennn
{

struct BackPropagation;

class AdaptiveMomentEstimation final : public Optimizer
{

public:

    enum DataSlot { GradientMoment, SquareGradientMoment, GraphScalars };

    explicit AdaptiveMomentEstimation(Loss* = nullptr);

    void set_default();

    void configure_for_task(NetworkTask) override;

    void set_learning_rate(const float new_learning_rate) { learning_rate = new_learning_rate; }
    float get_learning_rate() const { return learning_rate; }
    void set_beta_1(const float);
    void set_beta_2(const float);

    void update_parameters(BackPropagation&, OptimizerData&,
                           UpdateMode = UpdateMode::Standard) override;

    void on_epoch_begin(Index, OptimizerData&) override;

    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    string get_display_name() const override { return "adaptive moment estimation \"Adam\""; }
    bool supports_cuda_graph() const noexcept override { return true; }
    void setup_optimizer_data(OptimizerData&, Index, Device) override;

    float learning_rate = 0.001f;

    float beta_1 = 0.9f;

    float beta_2 = 0.999f;
};

}
