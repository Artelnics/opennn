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

#ifdef CUDA
struct BackPropagationCuda;
struct ADAMOptimizationDataCuda;
#endif

class AdaptiveMomentEstimation final : public Optimizer
{
    
public:

   AdaptiveMomentEstimation(const Loss* = nullptr);
   
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

   // Serialization


   void from_XML(const XMLDocument&) override;

   void to_XML(XMLPrinter&) const override;

   void update_parameters(BackPropagation&, AdaptiveMomentEstimationData&) const;

private:

   // TRAINING OPERATORS

   type learning_rate = type(0.001);

   type beta_1 = type(0.9);

   type beta_2 = type(0.999);

   Index batch_size = 1000;

#ifdef CUDA

    public:

    TrainingResults train_cuda() override;

    void update_parameters(BackPropagationCuda&, ADAMOptimizationDataCuda&) const;

#endif

};


struct AdaptiveMomentEstimationData final : public OptimizerData
{
    AdaptiveMomentEstimationData(AdaptiveMomentEstimation* = nullptr);

    void set(AdaptiveMomentEstimation* = nullptr);

    void print() const override;

    AdaptiveMomentEstimation* adaptive_moment_estimation = nullptr;

    VectorR gradient_exponential_decay;
    VectorR square_gradient_exponential_decay;

    Index iteration = 0;

    type step = 0;

    Index learning_rate_iteration = 0;
};


#ifdef CUDA

    struct ADAMOptimizationDataCuda final : public OptimizerData
    {
        ADAMOptimizationDataCuda(AdaptiveMomentEstimation* = nullptr);

        //~ADAMOptimizationDataCuda() { free(); }

        void set(AdaptiveMomentEstimation* = nullptr);

        void print() const override;

        AdaptiveMomentEstimation* adaptive_moment_estimation = nullptr;

        TensorCuda gradient_exponential_decay;
        TensorCuda square_gradient_exponential_decay;

        Index iteration = 0;

        type step = 0;

        //Index learning_rate_iteration = 0;
    };

#endif

}
