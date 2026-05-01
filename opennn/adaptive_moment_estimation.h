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

   AdaptiveMomentEstimation(Loss* = nullptr);

   // Set

   void set_batch_size(const Index new_batch_size);

   void set_default();

   // Get

   Index get_samples_number() const;

   // Training operators

   void set_learning_rate(const float);
   void set_beta_1(const float);
   void set_beta_2(const float);

   // Training

   TrainingResults train() override;

   void update_parameters(BackPropagation&, OptimizerData&) const;

   // Serialization

   void from_JSON(const JsonDocument&) override;

   void to_JSON(JsonWriter&) const override;

private:

   float learning_rate = float(0.001);

   float beta_1 = float(0.9);

   float beta_2 = float(0.98);

   Index batch_size = 1000;
};

}
