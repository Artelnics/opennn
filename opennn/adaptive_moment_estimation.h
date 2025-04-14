//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ADAPTIVEMOMENTESTIMATION_H
#define ADAPTIVEMOMENTESTIMATION_H

#include "optimization_algorithm.h"

namespace opennn
{

struct AdaptiveMomentEstimationData;

#ifdef OPENNN_CUDA_test
struct ADAMOptimizationDataCuda;
#endif

class AdaptiveMomentEstimation : public OptimizationAlgorithm
{
    
public:

   AdaptiveMomentEstimation(LossIndex* = nullptr);
   
   const type& get_learning_rate() const;
   const type& get_beta_1() const;
   const type& get_beta_2() const;
   const type& get_epsilon() const;

   // Stopping criteria

   const type& get_loss_goal() const;
   const type& get_maximum_time() const;

   // Set

   void set_batch_samples_number(const Index& new_batch_size);

   void set_default();

   // Get

   Index get_samples_number() const;

   // Training operators

   void set_learning_rate(const type&);
   void set_custom_learning_rate(const type&);
   void set_beta_1(const type&);
   void set_beta_2(const type&);
   void set_epsilon(const type&);

   // Training parameters

   void set_maximum_epochs_number(const Index&);

   // Stopping criteria

   void set_loss_goal(const type&);
   void set_accuracy_goal(const type&);
   void set_maximum_time(const type&);

   // Training

   TrainingResults perform_training() override;

   string write_optimization_algorithm_type() const override;

   // Serialization

   Tensor<string, 2> to_string_matrix() const override;

   void from_XML(const XMLDocument&) override;

   void to_XML(XMLPrinter&) const override;

   void update_parameters(BackPropagation&, AdaptiveMomentEstimationData&) const;

private:

   // TRAINING OPERATORS

   type learning_rate = type(0.001);

   bool use_custom_learning_rate = false;

   type initial_decay = type(0);

   type beta_1 = type(0.9);

   type beta_2 = type(0.999);

   type epsilon = type(1.e-6);

    // Stopping criteria

   type training_loss_goal = type(0);
   
   type training_accuracy_goal = type(1);

   Index maximum_epochs_number = 10000;

   Index maximum_selection_failures = numeric_limits<Index>::max();

   type maximum_time = type(360000);

   Index batch_size = 1000;

#ifdef OPENNN_CUDA_test

    public:

        TrainingResults perform_training_cuda();

        void update_parameteres_cuda(BackPropagationCuda&, ADAMOptimizationDataCuda&) const;

#endif

};


struct AdaptiveMomentEstimationData : public OptimizationAlgorithmData
{
    AdaptiveMomentEstimationData(AdaptiveMomentEstimation* = nullptr);

    void set(AdaptiveMomentEstimation* = nullptr);

    virtual void print() const;

    AdaptiveMomentEstimation* adaptive_moment_estimation = nullptr;

    Tensor<type, 1> gradient_exponential_decay;
    Tensor<type, 1> square_gradient_exponential_decay;

    Index iteration = 0;

    type step = 0;

    Index learning_rate_iteration = 0;
};

#ifdef OPENNN_CUDA_test

    struct ADAMOptimizationDataCuda : public OptimizationAlgorithmData
    {
        ADAMOptimizationDataCuda(AdaptiveMomentEstimation* = nullptr);

        void set(AdaptiveMomentEstimation* = nullptr);

        void free();

        void print() const;

        AdaptiveMomentEstimation* adaptive_moment_estimation = nullptr;

        float* square_gradient = nullptr;

        float* gradient_exponential_decay = nullptr;

        float* square_gradient_exponential_decay = nullptr;

        float* last_gradient_exponential_decay = nullptr;

        float* last_square_gradient_exponential_decay = nullptr;

        float* numerator = nullptr;

        float* denominator = nullptr;

        float* ones = nullptr;
        float one = 1.0f;

        Index iteration;

        type step;
    };

#endif

}

#endif
