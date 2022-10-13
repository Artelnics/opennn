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
#include "tensor_utilities.h"
#include "loss_index.h"

namespace opennn
{

struct TrainingResults;

/// This abstract class represents the concept of optimization algorithm for a neural network in the OpenNN library.
/// Any derived class must implement the perform_training() method.

class OptimizationAlgorithm
{

public:

   explicit OptimizationAlgorithm();

   explicit OptimizationAlgorithm(LossIndex*);

   virtual ~OptimizationAlgorithm();

    /// Enumeration of all possible conditions of stop for the algorithms.

    enum class StoppingCondition{MinimumLossDecrease, LossGoal,
                           MaximumSelectionErrorIncreases, MaximumEpochsNumber, MaximumTime};

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

   string write_time(const type&) const;

   // Set methods

   void set();

   virtual void set_default();

   virtual void set_threads_number(const int&);

   virtual void set_loss_index_pointer(LossIndex*);

   virtual void set_display(const bool&);

   void set_display_period(const Index&);

   void set_save_period(const Index&);
   void set_neural_network_file_name(const string&);

   // Training methods

   virtual void check() const;

   /// Trains a neural network which has a loss index associated. 

   virtual TrainingResults perform_training() = 0;

   virtual string write_optimization_algorithm_type() const {return string();}

   // Serialization methods

   virtual void print() const;

   virtual Tensor<string, 2> to_string_matrix() const;

   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;

   void save(const string&) const;
   void load(const string&);

protected:

   ThreadPool* thread_pool = nullptr;
   ThreadPoolDevice* thread_pool_device;

   /// Pointer to a loss index for a neural network object.

   LossIndex* loss_index_pointer = nullptr;

   /// Number of training epochs in the neural network.

   Index epochs_number = 10000;

   // UTILITIES

   ///Hardware use

   string hardware_use = "Multi-core";

   /// Number of iterations between the training showing progress.

   Index display_period = 10;

   /// Number of iterations between the training saving progress.

   Index save_period = numeric_limits<Index>::max();

   /// Path where the neural network is saved.

   string neural_network_file_name = "neural_network.xml";

   /// Display messages to screen.

   bool display = true;

   const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
   const Eigen::array<IndexPair<Index>, 1> product_vector_matrix = {IndexPair<Index>(0, 1)}; // Normal product vector times matrix
   const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/optimization_algorithm_cuda.h"
#endif

};


struct OptimizationAlgorithmData
{
    explicit OptimizationAlgorithmData()
    {
    }

    virtual ~OptimizationAlgorithmData()
    {
    }

    void print() const
    {
//        cout << "Potential parameters:" << endl;
//        cout << potential_parameters << endl;

//        cout << "Training direction:" << endl;
//        cout << training_direction << endl;

//        cout << "Initial learning rate:" << endl;
//        cout << initial_learning_rate << endl;
    }

    Tensor<type, 1> potential_parameters;
    Tensor<type, 1> training_direction;
    type initial_learning_rate = type(0);

};


/// This structure contains the optimization algorithm results.

struct TrainingResults
{
    /// Default constructor.

    explicit TrainingResults()
    {
    }

    explicit TrainingResults(const Index& epochs_number)
    {
        training_error_history.resize(1+epochs_number);
        training_error_history.setConstant(type(-1.0));

        selection_error_history.resize(1+epochs_number);
        selection_error_history.setConstant(type(-1.0));
    }

    /// Destructor.

    virtual ~TrainingResults() {}

    string write_stopping_condition() const;

    type get_training_error()
    {
        const Index size = training_error_history.size();

        return training_error_history(size-1);
    }

    type get_selection_error()
    {
        const Index size = selection_error_history.size();

        return selection_error_history(size-1);
    }

    type get_loss()
    {
        return loss;
    }

    type get_loss_decrease()
    {
        return loss_decrease;
    }

    Index get_selection_failures()
    {
        return selection_failures;
    }

    Index get_epochs_number()
    {
        return training_error_history.size() - 1;
    }


    /// Returns a string representation of the results structure.

    void save(const string&) const;

    void print(const string& message = string())
    {
        cout << message << endl;

        const Index epochs_number = training_error_history.size();

        cout << "Training results" << endl;
        cout << "Epochs number: " << epochs_number-1 << endl;

        cout << "Training error: " << training_error_history(epochs_number-1) << endl;

        if(abs(training_error_history(epochs_number-1) + type(1))  < type(NUMERIC_LIMITS_MIN))
            cout << "Selection error: " << selection_error_history(epochs_number-1) << endl;

        cout << "Stopping condition: " << write_stopping_condition() << endl;
    }

    /// Stopping condition of the algorithm.

    OptimizationAlgorithm::StoppingCondition stopping_condition = OptimizationAlgorithm::StoppingCondition::MaximumTime;

    /// Writes final results of the training.

    Tensor<string, 2> write_final_results(const Index& = 3) const;

    /// Resizes the training error history keeping the values.

    void resize_training_error_history(const Index&);

    /// Resizes the selection error history keeping the values.

    void resize_selection_error_history(const Index&);

    // Training history

    /// History of the loss function over the training iterations.

    Tensor<type, 1> training_error_history;

    /// History of the selection error over the training iterations.

    Tensor<type, 1> selection_error_history;

    /// Elapsed time of the training process.

    string elapsed_time;

    type loss;

    Index selection_failures;

    type loss_decrease;
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
