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
#include "optimization_algorithm.h"

namespace opennn
{

struct StochasticGradientDescentData;

/// This concrete class represents the stochastic gradient descent optimization algorithm[1] for a loss index of a neural network.

/// It supports momentum, learning rate decay, and Nesterov momentum.
///
/// \cite 1  Neural Designer "5 Algorithms to Train a Neural Network."
/// \ref https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network

class StochasticGradientDescent : public OptimizationAlgorithm
{

public:

   // Constructors

   explicit StochasticGradientDescent(); 

   explicit StochasticGradientDescent(LossIndex*);

   //Training operators

   const type& get_initial_learning_rate() const;
   const type& get_initial_decay() const;
   const type& get_momentum() const;
   const bool& get_nesterov() const;

   // Stopping criteria

   const type& get_loss_goal() const;
   const type& get_maximum_time() const;

   // Set methods

   void set_loss_index_pointer(LossIndex*) override;

   void set_default() final;

   void set_batch_samples_number(const Index& new_batch_samples_number)
   {
       batch_samples_number = new_batch_samples_number;
   }

   // Get methods

   Index get_batch_samples_number() const;

   //Training operators

   void set_initial_learning_rate(const type&);
   void set_initial_decay(const type&);
   void set_momentum(const type&);
   void set_nesterov(const bool&);

   void set_maximum_epochs_number(const Index&);

   // Stopping criteria

   void set_loss_goal(const type&);
   void set_maximum_time(const type&);

   // Training methods

   void update_parameters(LossIndexBackPropagation& , StochasticGradientDescentData&) const;

   TrainingResults perform_training() final;

   string write_optimization_algorithm_type() const final;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const final;

   void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;

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

   /// Number of samples per training batch.

   Index batch_samples_number = 1000;

   // Stopping criteria

   /// Goal value for the loss. It is a stopping criterion.

   type training_loss_goal = type(0);

   /// Maximum selection error allowed

   Index maximum_selection_failures = numeric_limits<Index>::max();

   /// Maximum epochs number

   Index maximum_epochs_number = 10000;

   /// Maximum training time. It is a stopping criterion.

   type maximum_time = type(3600);

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/stochastic_gradient_descent_cuda.h"
#endif

};


struct StochasticGradientDescentData : public OptimizationAlgorithmData
{
    /// Default constructor.

    explicit StochasticGradientDescentData()
    {
    }

    explicit StochasticGradientDescentData(StochasticGradientDescent* new_stochastic_gradient_descent_pointer)
    {
        set(new_stochastic_gradient_descent_pointer);
    }

    virtual ~StochasticGradientDescentData() {}

    void set(StochasticGradientDescent* new_stochastic_gradient_descent_pointer)
    {
        stochastic_gradient_descent_pointer = new_stochastic_gradient_descent_pointer;

        const LossIndex* loss_index_pointer = stochastic_gradient_descent_pointer->get_loss_index_pointer();

        const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

        const Index parameters_number = neural_network_pointer->get_parameters_number();

        parameters_increment.resize(parameters_number);
        nesterov_increment.resize(parameters_number);
        last_parameters_increment.resize(parameters_number);

        parameters_increment.setZero();
        nesterov_increment.setZero();
        last_parameters_increment.setZero();
    }

    StochasticGradientDescent* stochastic_gradient_descent_pointer = nullptr;

    Index iteration = 0;

    Tensor<type, 1> parameters_increment;
    Tensor<type, 1> nesterov_increment;
    Tensor<type, 1> last_parameters_increment;
};

}

#endif
