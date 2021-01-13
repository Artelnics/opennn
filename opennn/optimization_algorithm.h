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
#include <iomanip>

// OpenNN includes

#include "config.h"
#include "loss_index.h"

using namespace std;
using namespace Eigen;


namespace OpenNN
{

/// This abstract class represents the concept of optimization algorithm for a neural network in OpenNN library.
/// Any derived class must implement the perform_training() method.

class OptimizationAlgorithm
{

public:

   explicit OptimizationAlgorithm();

   explicit OptimizationAlgorithm(LossIndex*);

   virtual ~OptimizationAlgorithm();

    /// Enumeration of all possibles condition of stop for the algorithms.

    enum StoppingCondition{MinimumParametersIncrementNorm, MinimumLossDecrease, LossGoal, GradientNormGoal,
                           MaximumSelectionErrorIncreases, MaximumEpochsNumber, MaximumTime};

    struct OptimizationData
    {
        explicit OptimizationData()
        {
        }

        virtual ~OptimizationData()
        {
        }

        void print() const
        {
            cout << "Parameters:" << endl;
            cout << parameters << endl;

            cout << "Potential parameters:" << endl;
            cout << potential_parameters << endl;

            cout << "Training direction:" << endl;
            cout << training_direction << endl;

            cout << "Initial learning rate:" << endl;
            cout << initial_learning_rate << endl;
        }

        Tensor<type, 1> parameters;
        Tensor<type, 1> potential_parameters;
        Tensor<type, 1> training_direction;
        type initial_learning_rate = 0;



    };

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

       

       void save(const string&) const;

       /// Returns a default(empty) string matrix with the final results from training.

       Tensor<string, 2> write_final_results(const Index& = 3) const;

       /// Resizes training history variables.

       void resize_training_history(const Index&);

       /// Resizes selection history variables.

       void resize_selection_history(const Index&);

       /// Resizes the training error history keeping the values.

       void resize_training_error_history(const Index&);

       /// Resizes the selection error history keeping the values.

       void resize_selection_error_history(const Index&);

       /// Writes final results of the training.
/*
       Tensor<string, 2> write_final_results(const Index& precision = 3) const;
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

       type final_parameters_norm;

       /// Final loss function evaluation.

       type final_training_error;

       /// Final selection error.

       type final_selection_error;

       /// Final gradient norm.

       type final_gradient_norm;

       /// Elapsed time of the training process.

       string elapsed_time;

       /// Maximum number of training iterations.

       Index epochs_number;

       /// Stopping criterion.

       string stopping_criterion;
   };


   // Get methods

   LossIndex* get_loss_index_pointer() const;

   /// Hardware use.
   string get_hardware_use() const;
   void set_hardware_use(const string&);

   bool has_loss_index() const;

   // Utilities

   const bool& get_display() const;

   const Index& get_display_period() const;

   const Index& get_save_period() const;

   const string& get_neural_network_file_name() const;

   /// Writes the time from seconds in format HH:mm:ss.

   const string write_elapsed_time(const type&) const;

   // Set methods

   void set();
   void set(LossIndex*);

   virtual void set_default();

   virtual void set_threads_number(const int&);

   virtual void set_loss_index_pointer(LossIndex*);

   virtual void set_display(const bool&);

   void set_display_period(const Index&);

   void set_save_period(const Index&);
   void set_neural_network_file_name(const string&);

   virtual void set_reserve_selection_error_history(const bool&) = 0;

   // Training methods

   virtual void check() const;

   /// Trains a neural network which has a loss index associated. 

   virtual Results perform_training() = 0;

   virtual string write_optimization_algorithm_type() const {return string();}

   // Serialization methods

   virtual 
   void print() const;

   virtual Tensor<string, 2> to_string_matrix() const;

   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;

   void save(const string&) const;
   void load(const string&);

protected:

   NonBlockingThreadPool* non_blocking_thread_pool = nullptr;
   ThreadPoolDevice* thread_pool_device;

   /// Pointer to a loss index for a neural network object.


   LossIndex* loss_index_pointer = nullptr;

   /// Number of training epochs in the neural network.

   Index epochs_number = 10000;

   // UTILITIES

   ///Hardware use
   string hardware_use = "Multi-core";


   /// Number of iterations between the training showing progress.

   Index display_period = 5;

   /// Number of iterations between the training saving progress.

   Index save_period = 100;

   /// Path where the neural network is saved.

   string neural_network_file_name = "neural_network.xml";

   /// Display messages to screen.

   bool display = true;

   const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
   const Eigen::array<IndexPair<Index>, 1> product_vector_matrix = {IndexPair<Index>(0, 1)}; // Normal product vector times matrix
   const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

   void normalized(Tensor<type, 1>& tensor) const
   {      
       const type norm = l2_norm(tensor);

       tensor.device(*thread_pool_device) = tensor/norm;
   }


   type l2_norm(const Tensor<type, 1>& tensor) const
   {
       Tensor<type, 0> norm;

       norm.device(*thread_pool_device) = tensor.square().sum().sqrt();

       return norm(0);
   }

   bool is_zero(const Tensor<type, 1>& tensor) const
   {
       const Index size = tensor.size();

       for(Index i = 0; i < size; i++)
       {
           if(abs(tensor[i]) > numeric_limits<type>::min()) return false;
       }

       return true;
   }

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/optimization_algorithm_cuda.h"
#endif

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
