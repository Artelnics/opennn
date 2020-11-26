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
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>
#include <chrono>
#include <time.h>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

// OpenNN includes

#include "loss_index.h"
#include "optimization_algorithm.h"
#include "config.h"

namespace OpenNN
{

/// This concrete class represents the adaptive moment estimation(Adam) training algorithm,
/// based on adaptive estimates of lower-order moments.

///
/// For more information visit:
///
/// \cite 1 C. Barranquero "High performance optimization algorithms for neural networks."
/// \ref https://www.opennn.net/files/high_performance_optimization_algorithms_for_neural_networks.pdf .
///
/// \cite 2 D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization." arXiv preprint arXiv:1412.6980v8 (2014).

class AdaptiveMomentEstimation : public OptimizationAlgorithm
{

public:

    struct OptimizationData
    {
        /// Default constructor.

        explicit OptimizationData();

        explicit OptimizationData(AdaptiveMomentEstimation* new_stochastic_gradient_descent_pointer);

        virtual ~OptimizationData();

        void set(AdaptiveMomentEstimation* new_adaptive_moment_estimation_pointer);

        void print() const;

        AdaptiveMomentEstimation* adaptive_moment_estimation_pointer = nullptr;

        Index learning_rate_iteration = 0;

        Tensor<type, 1> parameters;
        Tensor<type, 1> minimal_selection_parameters;

        Tensor<type, 1> gradient_exponential_decay;
        Tensor<type, 1> square_gradient_exponential_decay;

        Tensor<type, 1> aux;

        Index iteration;
    };


   // Constructors

   explicit AdaptiveMomentEstimation();

   explicit AdaptiveMomentEstimation(LossIndex*);   

   virtual ~AdaptiveMomentEstimation();
   
   // Training operators

   const type& get_initial_learning_rate() const;
   const type& get_beta_1() const;
   const type& get_beta_2() const;
   const type& get_epsilon() const;

   // Stopping criteria

   const type& get_loss_goal() const;
   const type& get_maximum_time() const;
   const bool& get_choose_best_selection() const;

   // Reserve training history

   const bool& get_reserve_training_error_history() const;
   const bool& get_reserve_selection_error_history() const;

   // Set methods

   void set_loss_index_pointer(LossIndex*);

   void set_reserve_all_training_history(const bool&);

   void set_batch_samples_number(const Index& new_batch_samples_number);

   // Training operators

   void set_initial_learning_rate(const type&);
   void set_beta_1(const type&);
   void set_beta_2(const type&);
   void set_epsilon(const type&);

   // Training parameters

   void set_maximum_epochs_number(const Index&);

   // Stopping criteria

   void set_loss_goal(const type&);
   void set_maximum_time(const type&);
   void set_choose_best_selection(const bool&);

   // Reserve training history

   void set_reserve_training_error_history(const bool&);
   void set_reserve_selection_error_history(const bool&);

   // Training methods

   Results perform_training();

   /// Perform Neural Network training.

   void perform_training_void();

   /// Return the algorithm optimum for your model.

   string write_optimization_algorithm_type() const;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const;

   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   void update_iteration(const LossIndex::BackPropagation& back_propagation,
                                 OptimizationData& optimization_data);

private:

   // TRAINING OPERATORS

   /// Initial learning rate

   type initial_learning_rate = static_cast<type>(0.001);

   /// Learning rate decay over each update.

   type initial_decay = 0;

   /// Exponential decay over gradient estimates.

   type beta_1 = static_cast<type>(0.9);

   /// Exponential decay over square gradient estimates.

   type beta_2 = static_cast<type>(0.999);

   /// Small number to prevent any division by zero

   type epsilon =static_cast<type>(1.e-7);

    // Stopping criteria

   /// Goal value for the loss. It is used as a stopping criterion.

   type training_loss_goal = 0;

   /// gradient norm goal. It is used as a stopping criterion.

   type gradient_norm_goal = 0;

   /// Maximum epochs number

   Index maximum_epochs_number = 10000;

   /// Maximum selection error allowed

   Index maximum_selection_error_increases = 1000;

   /// Maximum training time. It is used as a stopping criterion.

   type maximum_time = 3600;

   /// True if the final model will be the neural network with the minimum selection error, false otherwise.

   bool choose_best_selection = false;

   // TRAINING HISTORY

   /// True if the error history vector is to be reserved, false otherwise.

   bool reserve_training_error_history = true;

   /// True if the selection error history vector is to be reserved, false otherwise.

   bool reserve_selection_error_history = true;

   /// Training and selection batch size.

   Index batch_samples_number = 1000;

   /// Hardware use.

   string hardware_use = "Multi-core";

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/adaptive_moment_estimation_cuda.h"
#endif

#ifdef OPENNN_MKL
    #include "../../opennn-mkl/opennn_mkl/adaptive_moment_estimation_mkl.h"
#endif

};

}

#endif
