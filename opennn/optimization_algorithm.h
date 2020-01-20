//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S   H E A D E R 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef OPTIMIZATIONALGORITHM_H
#define OPTIMIZATIONALGORITHM_H

// System includes

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

#include "tinyxml2.h"

using namespace std;
using namespace Eigen;


namespace OpenNN
{

/// This abstract class represents the concept of optimization algorithm for a neural network in OpenNN library.

///
/// Any derived class must implement the perform_training() method.

class OptimizationAlgorithm
{

public:

   explicit OptimizationAlgorithm();

   explicit OptimizationAlgorithm(LossIndex*);

   explicit OptimizationAlgorithm(const tinyxml2::XMLDocument&);

   virtual ~OptimizationAlgorithm();

    /// Enumeration of all possibles condition of stop for the algorithms.

    enum StoppingCondition{MinimumParametersIncrementNorm, MinimumLossDecrease, LossGoal, GradientNormGoal,
                           MaximumSelectionErrorIncreases, MaximumEpochsNumber, MaximumTime};

   /// This structure contains the optimization algorithm results.    

   struct Results
   {
       explicit Results()
       {
       }

       virtual ~Results()
       {
       }

       string write_stopping_condition() const;

       /// Stopping condition of the algorithm.

       StoppingCondition stopping_condition;

       /// Returns a string representation of the results structure.

       string object_to_string() const;

       void save(const string&) const;

       /// Returns a default(empty) string matrix with the final results from training.

       Tensor<string, 2> write_final_results(const int&) const;

       /// Resizes training history variables.

       void resize_training_history(const int&);

       /// Writes final results of the training.
/*
       Tensor<string, 2> write_final_results(const int& precision = 3) const;
*/
       // Training history

       /// History of the loss function loss over the training iterations.

       Tensor<type, 1> training_error_history;

       /// History of the selection error over the training iterations.

       Tensor<type, 1> selection_error_history;

       // Final values

       /// Final neural network parameters vector.

       Tensor<type, 1> final_parameters;

       /// Final neural network parameters norm.

       double final_parameters_norm;

       /// Final loss function evaluation.

       double final_training_error;

       /// Final selection error.

       double final_selection_error;

       /// Final gradient norm.

       double final_gradient_norm;

       /// Elapsed time of the training process.

       double elapsed_time;

       /// Maximum number of training iterations.

       int epochs_number;

       /// Stopping criterion.

       string stopping_criterion;
   };


   // Get methods

   LossIndex* get_loss_index_pointer() const;

   bool has_loss_index() const;

   // Utilities

   const bool& get_display() const;

   const int& get_display_period() const;

   const int& get_save_period() const;

   const string& get_neural_network_file_name() const;

   // Set methods

   void set();
   void set(LossIndex*);

   virtual void set_default();

   virtual void set_loss_index_pointer(LossIndex*);

   virtual void set_display(const bool&);

   void set_display_period(const int&);

   void set_save_period(const int&);
   void set_neural_network_file_name(const string&);

   // Training methods

   virtual void check() const;

   /// Trains a neural network which has a loss index associated. 

   virtual Results perform_training() = 0;

   virtual string write_optimization_algorithm_type() const {return string();}

   // Serialization methods

   virtual string object_to_string() const;
   void print() const;

   virtual Tensor<string, 2> to_string_matrix() const;

   virtual tinyxml2::XMLDocument* to_XML() const;
   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;

   void save(const string&) const;
   void load(const string&);

protected:

   /// Pointer to a loss index for a neural network object.

   LossIndex* loss_index_pointer = nullptr;

   /// Number of training epochs in the neural network.

   int epochs_number = 10000;

   // UTILITIES

   /// Number of iterations between the training showing progress.

   int display_period;

   /// Number of iterations between the training saving progress.

   int save_period;

   /// Path where the neural network is saved.

   string neural_network_file_name;

   /// Display messages to screen.

   bool display;
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
