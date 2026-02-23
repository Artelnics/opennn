//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "optimization_algorithm.h"
#include "tensors.h"

namespace opennn
{

struct BackPropagation;
struct AdaptiveMomentEstimationData;

#ifdef OPENNN_CUDA
struct BackPropagationCuda;
struct ADAMOptimizationDataCuda;
#endif

class AdaptiveMomentEstimation final : public Optimizer
{
    
public:

   AdaptiveMomentEstimation(const Loss* = nullptr);
   
   type get_learning_rate() const;
   type get_beta_1() const;
   type get_beta_2() const;

   // Stopping criteria

   type get_loss_goal() const;

   // Set

   void set_batch_size(const Index new_batch_size);

   void set_default();

   void set_display(bool) override;

   // Get

   Index get_samples_number() const;

   // Training operators

   void set_learning_rate(const type);
   void set_beta_1(const type);
   void set_beta_2(const type);

   // Training parameters

   void set_maximum_epochs(const Index);

   // Stopping criteria

   void set_loss_goal(const type);
   void set_accuracy_goal(const type);
   void set_maximum_time(const type);

   // Training

   TrainingResults train() override;

   // Serialization

   Tensor<string, 2> to_string_matrix() const override;

   void from_XML(const XMLDocument&) override;

   void to_XML(XMLPrinter&) const override;

   void update_parameters(BackPropagation&, AdaptiveMomentEstimationData&) const;

private:

   // TRAINING OPERATORS

   type learning_rate = type(0.001);

   type beta_1 = type(0.9);

   type beta_2 = type(0.999);

    // Stopping criteria

   type training_loss_goal = type(-10);
   
   type training_accuracy_goal = type(1);

   Index maximum_validation_failures = numeric_limits<Index>::max();

   Index batch_size = 1000;

#ifdef OPENNN_CUDA

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


#ifdef OPENNN_CUDA

    struct ADAMOptimizationDataCuda final : public OptimizerData
    {
        ADAMOptimizationDataCuda(AdaptiveMomentEstimation* = nullptr);

        //~ADAMOptimizationDataCuda() { free(); }

        void set(AdaptiveMomentEstimation* = nullptr);

        void print() const;

        AdaptiveMomentEstimation* adaptive_moment_estimation = nullptr;

        TensorCuda gradient_exponential_decay;
        TensorCuda square_gradient_exponential_decay;

        Index iteration = 0;

        type step = 0;

        //Index learning_rate_iteration = 0;
    };

#endif

}
