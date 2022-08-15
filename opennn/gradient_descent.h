//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <limits.h>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "loss_index.h"

#include "optimization_algorithm.h"
#include "learning_rate_algorithm.h"
#include "config.h"

namespace opennn
{

struct GradientDescentData;

/// The process of making changes to weights and biases,
/// where the changes are propotyional to derivatives of network error with respect to those weights and biases.
/// This is done to minimize network error.

/// This concrete class represents the gradient descent optimization algorithm[1], used to minimize loss function.
///
/// \cite 1  Neural Designer "5 Algorithms to Train a Neural Network."
/// \ref https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network

class GradientDescent : public OptimizationAlgorithm
{

public:

   // Constructors

   explicit GradientDescent(); 

   explicit GradientDescent(LossIndex*);

   const LearningRateAlgorithm& get_learning_rate_algorithm() const;
   LearningRateAlgorithm* get_learning_rate_algorithm_pointer();

   virtual string get_hardware_use() const;

   // Stopping criteria   

   const type& get_minimum_loss_decrease() const;
   const type& get_loss_goal() const;

   const Index& get_maximum_selection_failures() const;

   const Index& get_maximum_epochs_number() const;
   const type& get_maximum_time() const;

   // Set methods

   void set_loss_index_pointer(LossIndex*) final;

   void set_learning_rate_algorithm(const LearningRateAlgorithm&);

   void set_default() override;

   // Stopping criteria

   void set_maximum_epochs_number(const Index&);

   void set_minimum_loss_decrease(const type&);
   void set_loss_goal(const type&);

   void set_maximum_selection_failures(const Index&);

   void set_maximum_time(const type&);

   // Training methods

   void calculate_training_direction(const Tensor<type, 1>&, Tensor<type, 1>&) const;

   void update_parameters(
           const DataSetBatch&,
           NeuralNetworkForwardPropagation&,
           LossIndexBackPropagation&,
           GradientDescentData&) const;

   TrainingResults perform_training() final;

   string write_optimization_algorithm_type() const final;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const final;

   void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;

private:

   // TRAINING OPERATORS

   /// Learning rate algorithm object for one-dimensional minimization. 

   LearningRateAlgorithm learning_rate_algorithm;

   const type first_learning_rate = static_cast<type>(0.01);

   // Stopping criteria 

   /// Minimum loss improvement between two successive iterations. It is a stopping criterion.

   type minimum_loss_decrease;

   /// Goal value for the loss. It is a stopping criterion.

   type training_loss_goal;

   /// Maximum number of epochs at which the selection error increases.
   /// This is an early stopping method for improving selection.

   Index maximum_selection_failures;

   /// Maximum epochs number

   Index maximum_epochs_number;

   /// Maximum training time. It is a stopping criterion.

   type maximum_time;

};


struct GradientDescentData : public OptimizationAlgorithmData
{
    /// Default constructor.

    explicit GradientDescentData()
    {
    }


    explicit GradientDescentData(GradientDescent* new_gradient_descent_pointer)
    {
        set(new_gradient_descent_pointer);
    }

    /// Destructor

    virtual ~GradientDescentData() {}

    void set(GradientDescent* new_gradient_descent_pointer)
    {
        gradient_descent_pointer = new_gradient_descent_pointer;

        const LossIndex* loss_index_pointer = gradient_descent_pointer->get_loss_index_pointer();

        const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

        const Index parameters_number = neural_network_pointer->get_parameters_number();

        // Neural network data

        potential_parameters.resize(parameters_number);

        parameters_increment.resize(parameters_number);

        // Optimization algorithm data

        training_direction.resize(parameters_number);
    }


    virtual void print() const
    {
        cout << "Training direction:" << endl;
        cout << training_direction << endl;

        cout << "Learning rate:" << endl;
        cout << learning_rate << endl;
    }

    GradientDescent* gradient_descent_pointer = nullptr;

    // Neural network data

//    Tensor<type, 1> potential_parameters;
//    Tensor<type, 1> training_direction;
//    type initial_learning_rate = type(0);

    Tensor<type, 1> parameters_increment;

    // Optimization algorithm data

    Index epoch = 0;

    Tensor<type, 0> training_slope;

    type learning_rate = type(0);
    type old_learning_rate = type(0);
};

}

#endif
