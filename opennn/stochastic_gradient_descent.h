//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef STOCHASTICGRADIENTDESCENT_H
#define STOCHASTICGRADIENTDESCENT_H

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

// OpenNN includes

#include "config.h"

#include "loss_index.h"
//#include "mean_squared_error.h"
#include "optimization_algorithm.h"

namespace OpenNN
{

/// This concrete class represents the stochastic gradient descent optimization algorithm[1] for a loss index of a neural network.

/// It supports momentum, learning rate decay, and Nesterov momentum.
///
/// \cite 1  Neural Designer "5 Algorithms to Train a Neural Network."
/// \ref https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network

class StochasticGradientDescent : public OptimizationAlgorithm
{

public:

    struct SGDOptimizationData : public OptimizationData
    {
        /// Default constructor.

        explicit SGDOptimizationData()
        {
        }

        explicit SGDOptimizationData(StochasticGradientDescent* new_stochastic_gradient_descent_pointer)
        {
            set(new_stochastic_gradient_descent_pointer);
        }

        virtual ~SGDOptimizationData() {}

        void set(StochasticGradientDescent* new_stochastic_gradient_descent_pointer)
        {
            stochastic_gradient_descent_pointer = new_stochastic_gradient_descent_pointer;

            LossIndex* loss_index_pointer = stochastic_gradient_descent_pointer->get_loss_index_pointer();

            NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

            const Index parameters_number = neural_network_pointer->get_parameters_number();

            parameters.resize(parameters_number);

            parameters = neural_network_pointer->get_parameters();

            parameters_increment.resize(parameters_number);
            nesterov_increment.resize(parameters_number);
            last_parameters_increment.resize(parameters_number);

            parameters_increment.setZero();
            nesterov_increment.setZero();
            last_parameters_increment.setZero();
        }

        void print() const
        {
        }

        StochasticGradientDescent* stochastic_gradient_descent_pointer = nullptr;

        Index iteration = 0;

        Tensor<type, 1> parameters;
        Tensor<type, 1> parameters_increment;
        Tensor<type, 1> nesterov_increment;
        Tensor<type, 1> last_parameters_increment;

        Tensor<type, 1> minimal_selection_parameters;
    };

    static vector<Index> tensor_to_vector(const Tensor<Index, 1>& tensor)
    {
        const size_t size = static_cast<size_t>(tensor.dimension(0));

        vector<Index> new_vector(static_cast<size_t>(size));

        for(size_t i = 0; i < size; i++)
        {
            new_vector[i] = tensor(static_cast<Index>(i));
        }

        return new_vector;
    }

   // Constructors

   explicit StochasticGradientDescent(); 

   explicit StochasticGradientDescent(LossIndex*);

   // Destructor

   virtual ~StochasticGradientDescent();
   
   //Training operators

   const type& get_initial_learning_rate() const;
   const type& get_initial_decay() const;
   const type& get_momentum() const;
   const bool& get_nesterov() const;

   // Stopping criteria

   const type& get_loss_goal() const;
   const type& get_maximum_time() const;
   const bool& get_choose_best_selection() const;

   // Reserve training history

   const bool& get_reserve_training_error_history() const;
   const bool& get_reserve_selection_error_history() const;

   // Set methods

   void set_loss_index_pointer(LossIndex*);

   void set_default();

   void set_reserve_all_training_history(const bool&);


   void set_batch_samples_number(const Index& new_batch_samples_number)
   {
       batch_samples_number = new_batch_samples_number;
   }

   //Training operators

   void set_initial_learning_rate(const type&);
   void set_initial_decay(const type&);
   void set_momentum(const type&);
   void set_nesterov(const bool&);


   void set_maximum_epochs_number(const Index&);

   // Stopping criteria

   void set_loss_goal(const type&);
   void set_maximum_time(const type&);
   void set_choose_best_selection(const bool&);

   // Reserve training history

   void set_reserve_training_error_history(const bool&);
   void set_reserve_selection_error_history(const bool&);

   // Training methods

   void update_iteration(const LossIndex::BackPropagation& back_propagation,
                         SGDOptimizationData& optimization_data);

   Results perform_training();

   void perform_training_void();

   string write_optimization_algorithm_type() const;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const;

   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;   

private:

   // Training operators

   /// Initial learning rate

   type initial_learning_rate;

   /// Learning rate decay over each update.

   type initial_decay;

   /// Parameter that accelerates SGD in the relevant direction and dampens oscillations.

   type momentum;

   /// Boolean. Whether to apply Nesterov momentum.

   bool nesterov;

   // Stopping criteria

   /// Goal value for the loss. It is used as a stopping criterion.

   type training_loss_goal = 0;

   /// gradient norm goal. It is used as a stopping criterion.

   type gradient_norm_goal = 0;

   /// Maximum epochs number

   Index maximum_epochs_number;

   /// Maximum training time. It is used as a stopping criterion.

   type maximum_time;

   /// Maximum selection error allowed

   Index maximum_selection_error_increases = 1000;

   /// True if the final model will be the neural network with the minimum selection error, false otherwise.

   bool choose_best_selection = false;

   // Training history

   /// True if the loss history vector is to be reserved, false otherwise.

   bool reserve_training_error_history;

   /// True if the selection error history vector is to be reserved, false otherwise.

   bool reserve_selection_error_history;

   /// Number of samples per training batch.

   Index batch_samples_number = 1000;


#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/stochastic_gradient_descent_cuda.h"
#endif

#ifdef OPENNN_MKL
    #include "../../opennn-mkl/opennn_mkl/stochastic_gradient_descent_mkl.h"
#endif
};

}

#endif
