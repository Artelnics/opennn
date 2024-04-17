//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ADAPTIVEMOMENTESTIMATION_H
#define ADAPTIVEMOMENTESTIMATION_H

// System includes

#include <string>
#include <limits>

// OpenNN includes

#include "config.h"
#include "loss_index.h"
#include "optimization_algorithm.h"

namespace opennn
{

struct AdaptiveMomentEstimationData;

#ifdef OPENNN_CUDA
struct ADAMOptimizationDataCuda;
#endif

/// This concrete class represents the adaptive moment estimation(Adam) optimization algorithm.
/// This algorithm is based on adaptive estimates of lower-order moments.
///
/// For more information visit:
///
/// \cite 1 C. Barranquero "High performance optimization algorithms for neural networks."
/// \ref https://www.opennn.net/files/high_performance_optimization_algorithms_for_neural_networks.pdf .
///
/// \cite 2 D. P. Kingma and J. L. Ba, "Adam: A Method for Stochastic Optimization." arXiv preprint arXiv:1412.6980v8 (2014).

class AdaptiveMomentEstimation : public OptimizationAlgorithm
{
    
public:

   // Constructors

   explicit AdaptiveMomentEstimation();

   explicit AdaptiveMomentEstimation(LossIndex*);   

   //virtual ~AdaptiveMomentEstimation();
   
   // Training operators

   const type& get_learning_rate() const;
   const type& get_beta_1() const;
   const type& get_beta_2() const;
   const type& get_epsilon() const;

   // Stopping criteria

   const type& get_loss_goal() const;
   const type& get_maximum_time() const;

   // Set methods

   void set_loss_index(LossIndex*) final;

   void set_batch_samples_number(const Index& new_batch_samples_number);

   void set_default() final;

   // Get methods

   Index get_batch_samples_number() const;

   // Training operators

   void set_learning_rate(const type&);
   void set_beta_1(const type&);
   void set_beta_2(const type&);
   void set_epsilon(const type&);

   // Training parameters

   void set_maximum_epochs_number(const Index&);

   // Stopping criteria

   void set_loss_goal(const type&);
   void set_maximum_time(const type&);

   // Training methods

   TrainingResults perform_training() final;

   /// Return the algorithm optimum for your model.

   string write_optimization_algorithm_type() const final;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const final;

   void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;

   void update_parameters(BackPropagation&, AdaptiveMomentEstimationData&) const;

private:

   // TRAINING OPERATORS

   /// Initial learning rate

   type learning_rate = type(0.001);

   /// Learning rate decay over each update.

   type initial_decay = type(0);

   /// Exponential decay over gradient estimates.

   type beta_1 = type(0.9);

   /// Exponential decay over square gradient estimates.

   type beta_2 = type(0.999);

   /// Small number to prevent any division by zero

   type epsilon =type(1.e-8);

    // Stopping criteria

   /// Goal value for the loss. It a stopping criterion.

   type training_loss_goal = type(0);

   /// Maximum epochs number.

   Index maximum_epochs_number = 10000;

   /// Maximum number of times when selection error increases.

   Index maximum_selection_failures = numeric_limits<Index>::max();

   /// Maximum training time. It is a stopping criterion.

   type maximum_time = type(3600);

   /// Training and selection batch size.

   Index batch_samples_number = 1000;


#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/adaptive_moment_estimation_cuda.h"
#endif

};


struct AdaptiveMomentEstimationData : public OptimizationAlgorithmData
{
    /// Default constructor.

    explicit AdaptiveMomentEstimationData();

    explicit AdaptiveMomentEstimationData(AdaptiveMomentEstimation*);

    void set(AdaptiveMomentEstimation*);

    virtual void print() const;

    AdaptiveMomentEstimation* adaptive_moment_estimation = nullptr;

    Tensor<type, 1> gradient_exponential_decay;
    Tensor<type, 1> square_gradient_exponential_decay;

    Index iteration = 0;

    Index learning_rate_iteration = 0;
};

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/struct_adaptive_moment_estimation_cuda.h"
#endif

}

#endif
