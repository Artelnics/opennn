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
#include "device.h"
#include "loss_index.h"
#include "mean_squared_error.h"
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

    struct OptimizationData
    {
        /// Default constructor.

        explicit OptimizationData()
        {
        }

        explicit OptimizationData(StochasticGradientDescent* new_stochastic_gradient_descent_pointer)
        {
            set(new_stochastic_gradient_descent_pointer);
        }

        virtual ~OptimizationData() {}

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

   explicit StochasticGradientDescent(const tinyxml2::XMLDocument&); 

   // Destructor

   virtual ~StochasticGradientDescent();
   
   //Training operators

   const type& get_initial_learning_rate() const;
   const type& get_initial_decay() const;
   const type& get_momentum() const;
   const bool& get_nesterov() const;

   // Training parameters

   const type& get_warning_parameters_norm() const;
   const type& get_warning_gradient_norm() const;
   const type& get_error_parameters_norm() const;
   const type& get_error_gradient_norm() const;

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


   void set_batch_instances_number(const Index& new_batch_instances_number)
   {
       batch_instances_number = new_batch_instances_number;
   }


   void set_batch_size(const Index& new_batch_instances_number)
   {
       batch_instances_number = new_batch_instances_number;
   }

   //Training operators

   void set_initial_learning_rate(const type&);
   void set_initial_decay(const type&);
   void set_momentum(const type&);
   void set_nesterov(const bool&);

   // Training parameters

   void set_warning_parameters_norm(const type&);
   void set_warning_gradient_norm(const type&);
   void set_error_parameters_norm(const type&);
   void set_error_gradient_norm(const type&);
   void set_maximum_epochs_number(const Index&);

   // Stopping criteria

   void set_loss_goal(const type&);
   void set_maximum_time(const type&);
   void set_choose_best_selection(const bool&);

   // Reserve training history

   void set_reserve_training_error_history(const bool&);
   void set_reserve_selection_error_history(const bool&);

   // Utilities

   void set_display_period(const Index&);

   // Training methods

   Results perform_training();

   void perform_training_void();

   string write_optimization_algorithm_type() const;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const;

   tinyxml2::XMLDocument* to_XML() const;

   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   void update_iteration(const LossIndex::BackPropagation& back_propagation,
                         OptimizationData& optimization_data)
   {
       const type learning_rate = initial_learning_rate/(1 + optimization_data.iteration*initial_decay);

       optimization_data.parameters_increment = back_propagation.gradient*(-learning_rate);

       if(momentum > 0 && !nesterov)
       {
           optimization_data.parameters_increment += momentum*optimization_data.last_parameters_increment;

           optimization_data.parameters += optimization_data.parameters_increment;
       }
       else if(momentum > 0 && nesterov)
       {
           optimization_data.parameters_increment += momentum*optimization_data.last_parameters_increment;

           optimization_data.nesterov_increment
                   = optimization_data.parameters_increment*momentum - back_propagation.gradient*learning_rate;

           optimization_data.parameters += optimization_data.nesterov_increment;
       }
       else
       {
           optimization_data.parameters += optimization_data.parameters_increment;
       }

       optimization_data.last_parameters_increment = optimization_data.parameters_increment;

       optimization_data.iteration++;
   }


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

   // Training parameters

   /// Value for the parameters norm at which a warning message is written to the screen. 

   type warning_parameters_norm;

   /// Value for the gradient norm at which a warning message is written to the screen. 

   type warning_gradient_norm;

   /// Value for the parameters norm at which the training process is assumed to fail. 
   
   type error_parameters_norm;

   /// Value for the gradient norm at which the training process is assumed to fail. 

   type error_gradient_norm;

   // Stopping criteria

   /// Goal value for the loss. It is used as a stopping criterion.

   type training_loss_goal = 0;

   /// Maximum epochs number

   Index maximum_epochs_number;

   /// Maximum training time. It is used as a stopping criterion.

   type maximum_time;

   /// True if the final model will be the neural network with the minimum selection error, false otherwise.

   bool choose_best_selection;

   // Training history

   /// True if the loss history vector is to be reserved, false otherwise.

   bool reserve_training_error_history;

   /// True if the selection error history vector is to be reserved, false otherwise.

   bool reserve_selection_error_history;

   Index batch_instances_number = 1000;

#ifdef OPENNN_CUDA
    #include "../../artelnics/opennn_cuda/opennn_cuda/stochastic_gradient_descent_cuda.h"
#endif

};

}

#endif
