//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D    C L A S S   H E A D E R      
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef QUASINEWTONMETHOD_H
#define QUASINEWTONMETHOD_H

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
#include "learning_rate_algorithm.h"

// Eigen Includes

#include "../eigen/unsupported/Eigen/KroneckerProduct"

using Eigen::MatrixXd;

namespace OpenNN
{

/// Class of optimization algorithm based on Newton's method.
/// An approximate Hessian matrix is computed at each iteration of the algorithm based on the gradients.

/// This concrete class represents a quasi-Newton training algorithm[1], used to minimize loss function.
///
/// \cite 1  Neural Designer "5 Algorithms to Train a Neural Network."
/// \ref https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network

class QuasiNewtonMethod : public OptimizationAlgorithm
{

public:

    struct QNMOptimizationData : public OptimizationData
    {
        /// Default constructor.

        explicit QNMOptimizationData()
        {
        }

        explicit QNMOptimizationData(QuasiNewtonMethod* new_quasi_newton_method_pointer)
        {
            set(new_quasi_newton_method_pointer);
        }

        virtual ~QNMOptimizationData() {}

        void set(QuasiNewtonMethod* new_quasi_newton_method_pointer)
        {
            quasi_newton_method_pointer = new_quasi_newton_method_pointer;

            LossIndex* loss_index_pointer = quasi_newton_method_pointer->get_loss_index_pointer();

            NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

            const Index parameters_number = neural_network_pointer->get_parameters_number();

            // Neural network data

            parameters.resize(parameters_number);
            parameters = neural_network_pointer->get_parameters();

            old_parameters.resize(parameters_number);

            parameters_difference.resize(parameters_number);

            potential_parameters.resize(parameters_number);
            parameters_increment.resize(parameters_number);

            // Loss index data

            old_gradient.resize(parameters_number);
            old_gradient.setZero();

            gradient_difference.resize(parameters_number);

            inverse_hessian.resize(parameters_number, parameters_number);
            inverse_hessian.setZero();

            old_inverse_hessian.resize(parameters_number, parameters_number);
            old_inverse_hessian.setZero();

            // Optimization algorithm data

            training_direction.resize(parameters_number);

            old_inverse_hessian_dot_gradient_difference.resize(parameters_number);

        }

        void print() const
        {
            cout << "Training Direction:" << endl;
            cout << training_direction << endl;

            cout << "Learning rate:" << endl;
            cout << learning_rate << endl;

            cout << "Parameters:" << endl;
            cout << parameters << endl;
        }

        QuasiNewtonMethod* quasi_newton_method_pointer = nullptr;

        // Neural network data

        Tensor<type, 1> old_parameters;
        Tensor<type, 1> parameters_difference;

        Tensor<type, 1> parameters_increment;

        type parameters_increment_norm = 0;

        // Loss index data

        type old_training_loss = 0;

        Tensor<type, 1> old_gradient;
        Tensor<type, 1> gradient_difference;

        Tensor<type, 2> inverse_hessian;
        Tensor<type, 2> old_inverse_hessian;

        Tensor<type, 1> old_inverse_hessian_dot_gradient_difference;

        // Optimization algorithm data

        Index epoch = 0;

        Tensor<type, 0> training_slope;

        type learning_rate = 0;
        type old_learning_rate = 0;
    };


   // Enumerations

   /// Enumeration of the available training operators for obtaining the approximation to the inverse hessian.

   enum InverseHessianApproximationMethod{DFP, BFGS};

   // Constructors

   explicit QuasiNewtonMethod();

   explicit QuasiNewtonMethod(LossIndex*);

   // Destructor

   virtual ~QuasiNewtonMethod();

   // Get methods

   const LearningRateAlgorithm& get_learning_rate_algorithm() const;
   LearningRateAlgorithm* get_learning_rate_algorithm_pointer();

   const InverseHessianApproximationMethod& get_inverse_hessian_approximation_method() const;
   string write_inverse_hessian_approximation_method() const;

   const Index& get_epochs_number() const;

   // Stopping criteria

   const type& get_minimum_parameters_increment_norm() const;

   const type& get_minimum_loss_decrease() const;
   const type& get_loss_goal() const;
   const type& get_gradient_norm_goal() const;
   const Index& get_maximum_selection_error_increases() const;

   const Index& get_maximum_epochs_number() const;
   const type& get_maximum_time() const;

   const bool& get_choose_best_selection() const;

   // Reserve training history

   const bool& get_reserve_training_error_history() const;
   const bool& get_reserve_selection_error_history() const;

   // Set methods

   void set_loss_index_pointer(LossIndex*);

   void set_inverse_hessian_approximation_method(const InverseHessianApproximationMethod&);
   void set_inverse_hessian_approximation_method(const string&);

   void set_display(const bool&);

   void set_default();

   // Stopping criteria

   void set_minimum_parameters_increment_norm(const type&);

   void set_minimum_loss_decrease(const type&);
   void set_loss_goal(const type&);
   void set_gradient_norm_goal(const type&);
   void set_maximum_selection_error_increases(const Index&);

   void set_maximum_epochs_number(const Index&);
   void set_maximum_time(const type&);

   void set_choose_best_selection(const bool&);

   // Reserve training history

   void set_reserve_training_error_history(const bool&);
   void set_reserve_selection_error_history(const bool&);

   // Training methods

   void calculate_DFP_inverse_hessian(const LossIndex::BackPropagation&, QNMOptimizationData&) const;

   void calculate_BFGS_inverse_hessian(const LossIndex::BackPropagation&, QNMOptimizationData&) const;

   void initialize_inverse_hessian_approximation(QNMOptimizationData&) const;
   void calculate_inverse_hessian_approximation(const LossIndex::BackPropagation&, QNMOptimizationData&) const;

   const Tensor<type, 2> kronecker_product(Tensor<type, 2>&, Tensor<type, 2>&) const;
   const Tensor<type, 2> kronecker_product(Tensor<type, 1>&, Tensor<type, 1>&) const;

   void update_epoch(
           const DataSet::Batch& batch,
           NeuralNetwork::ForwardPropagation& forward_propagation,
           LossIndex::BackPropagation& back_propagation,
           QNMOptimizationData& optimization_data);

   Results perform_training();

   void perform_training_void();

   // Training history methods

   void set_reserve_all_training_history(const bool&);

   string write_optimization_algorithm_type() const;

   // Serialization methods

   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   
   Tensor<string, 2> to_string_matrix() const;


private: 

   /// Learning rate algorithm object.
   /// It is used to calculate the step for the quasi-Newton training direction.

   LearningRateAlgorithm learning_rate_algorithm;

   /// Variable containing the actual method used to obtain a suitable learning rate.

   InverseHessianApproximationMethod inverse_hessian_approximation_method;

   type first_learning_rate = static_cast<type>(0.01);

   // Stopping criteria

   /// Norm of the parameters increment vector at which training stops.

   type minimum_parameters_increment_norm;

   /// Minimum loss improvement between two successive epochs. It is used as a stopping criterion.

   type minimum_loss_decrease;

   /// Goal value for the loss. It is used as a stopping criterion.

   type training_loss_goal;

   /// Goal value for the norm of the error function gradient. It is used as a stopping criterion.

   type gradient_norm_goal;

   /// Maximum number of epochs at which the selection error increases.
   /// This is an early stopping method for improving selection.

   Index maximum_selection_error_increases;

   /// Maximum number of epochs to perform_training. It is used as a stopping criterion.

   Index maximum_epochs_number;

   /// Maximum training time. It is used as a stopping criterion.

   type maximum_time;

   /// True if the final model will be the neural network with the minimum selection error, false otherwise.

   bool choose_best_selection = false;

   // TRAINING HISTORY

   /// True if the training error history vector is to be reserved, false otherwise.

   bool reserve_training_error_history = true;

   /// True if the selection error history vector is to be reserved, false otherwise.

   bool reserve_selection_error_history = true;

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
