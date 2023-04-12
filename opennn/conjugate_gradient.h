//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N J U G A T E   G R A D I E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CONJUGATEGRADIENT_H
#define CONJUGATEGRADIENT_H

// System inlcludes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <climits>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "config.h"
#include "loss_index.h"
#include "optimization_algorithm.h"
#include "learning_rate_algorithm.h"

namespace opennn
{

struct ConjugateGradientData;

/// In the conjugate gradient algorithms a search is performed along conjugate directions,
/// which produces generally faster convergence than a search along the steepest descent directions.

/// This concrete class represents a conjugate gradient optimization algorithm.
///
/// \cite 1 \ref https://www.neuraldesigner.com/blog/5_algorithms_to_train_a_neural_network
///
/// \cite 2 D.P. O'Leary "The Block Conjugate Gradient Algorithm and Related Methods."

class ConjugateGradient : public OptimizationAlgorithm
{

public:

   // Enumerations

   /// Enumeration of the available training operators for obtaining the training direction.

   enum class TrainingDirectionMethod{PR, FR};

   // DEFAULT CONSTRUCTOR

   explicit ConjugateGradient(); 

   explicit ConjugateGradient(LossIndex*);   

   // Get methods

   const LearningRateAlgorithm& get_learning_rate_algorithm() const;
   LearningRateAlgorithm* get_learning_rate_algorithm_pointer();

   // Training operators

   const TrainingDirectionMethod& get_training_direction_method() const;
   string write_training_direction_method() const;

   // Stopping criteria

   

   const type& get_minimum_loss_decrease() const;
   const type& get_loss_goal() const;
   const Index& get_maximum_selection_failures() const;


   const Index& get_maximum_epochs_number() const;
   const type& get_maximum_time() const;

   // Set methods

   void set_default() final;

   void set_loss_index_pointer(LossIndex*) final;

   // Training operators

   void set_training_direction_method(const TrainingDirectionMethod&);
   void set_training_direction_method(const string&);

   // Stopping criteria

   

   void set_loss_goal(const type&);
   void set_minimum_loss_decrease(const type&);
   void set_maximum_selection_failures(const Index&);


   void set_maximum_epochs_number(const Index&);
   void set_maximum_time(const type&);

   // Utilities

   virtual void set_save_period(const Index&);

   // Training direction methods

   type calculate_PR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const;
   type calculate_FR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const;

   void calculate_PR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, Tensor<type, 1>&) const;
   void calculate_FR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, Tensor<type, 1>&) const;

   void calculate_gradient_descent_training_direction(const Tensor<type, 1>&, Tensor<type, 1>&) const;

   void calculate_conjugate_gradient_training_direction(const Tensor<type, 1>&,
                                                        const Tensor<type, 1>&,
                                                        const Tensor<type, 1>&,
                                                        Tensor<type, 1>&) const;

   // Training methods

   TrainingResults perform_training() final;

   string write_optimization_algorithm_type() const final;

   // Serialization methods

   Tensor<string, 2> to_string_matrix() const final;

   void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;

   void update_parameters(
           const DataSetBatch&,
           NeuralNetworkForwardPropagation&,
           LossIndexBackPropagation&,
           ConjugateGradientData&) const;

private:

   type first_learning_rate = static_cast<type>(0.01);

   /// Applied method for calculating the conjugate gradient direction.

   TrainingDirectionMethod training_direction_method = ConjugateGradient::TrainingDirectionMethod::FR;

   /// Learning rate algorithm object for one-dimensional minimization. 

   LearningRateAlgorithm learning_rate_algorithm;

   // Stopping criteria

   /// Minimum loss improvement between two successive iterations. It is a stopping criterion.

   type minimum_loss_decrease = type(0);

   /// Goal value for the loss. It is a stopping criterion.

   type training_loss_goal = type(0);

   /// Maximum number of epochs at which the selection error increases.
   /// This is an early stopping method for improving selection.

   Index maximum_selection_failures;

   /// Maximum number of epochs to perform_training. It is a stopping criterion.

   Index maximum_epochs_number;

   /// Maximum training time. It is a stopping criterion.

   type maximum_time;
};


struct ConjugateGradientData : public OptimizationAlgorithmData
{
    /// Default constructor.

    explicit ConjugateGradientData();

    explicit ConjugateGradientData(ConjugateGradient*);

    void set(ConjugateGradient*);

    virtual void print() const;

    ConjugateGradient* conjugate_gradient_pointer = nullptr;  

    Tensor<type, 1> parameters_increment;

    Tensor<type, 1> old_gradient;

    Tensor<type, 1> old_training_direction;

    Index epoch = 0;

    type learning_rate = type(0);
    type old_learning_rate = type(0);

    Tensor<type, 0> training_slope;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
