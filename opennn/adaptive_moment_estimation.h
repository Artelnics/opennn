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
struct AdaptiveMomentEstimationData;

class AdaptiveMomentEstimation final : public Optimizer
{

public:

   AdaptiveMomentEstimation(Loss* = nullptr);

   // Set

   void set_batch_size(const Index new_batch_size);

   void set_default();

   // Get

   Index get_samples_number() const;

   // Training operators

   void set_learning_rate(const type);
   void set_beta_1(const type);
   void set_beta_2(const type);

   // Training

   TrainingResults train() override;

   void update_parameters(BackPropagation&, AdaptiveMomentEstimationData&) const;

   // Serialization

   void from_XML(const XmlDocument&) override;

   void to_XML(XmlPrinter&) const override;

#ifdef OPENNN_WITH_CUDA

   TrainingResults train_cuda() override;

#endif

private:

   type learning_rate = type(0.001);

   type beta_1 = type(0.9);

   type beta_2 = type(0.98);

   Index batch_size = 1000;
};

struct AdaptiveMomentEstimationData final : public OptimizerData
{
    AdaptiveMomentEstimationData(AdaptiveMomentEstimation* = nullptr);

    void set(AdaptiveMomentEstimation* = nullptr);

    void print() const override;

    AdaptiveMomentEstimation* adaptive_moment_estimation = nullptr;

    Memory gradient_exponential_decay;
    Memory square_gradient_exponential_decay;

    Index iteration = 0;

    type step = 0;

    Index learning_rate_iteration = 0;
};

}
