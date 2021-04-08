//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N S   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NEURONSSELECTION_H
#define NEURONSSELECTION_H

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "config.h"
#include "training_strategy.h"

namespace OpenNN
{

struct NeuronsSelectionResults;

/// This abstract class represents the concept of neurons selection algorithm for a ModelSelection[1].

///
/// Any derived class must implement the perform_neurons_selection() method.
///
/// [1] Neural Designer "Model Selection Algorithms in Predictive Analytics." \ref https://www.neuraldesigner.com/blog/model-selection

class NeuronsSelection
{
public:

    // Constructors

    explicit NeuronsSelection();

    explicit NeuronsSelection(TrainingStrategy*);

    // Destructor

    virtual ~NeuronsSelection();

    // Enumerators

    /// Enumeration of all possibles condition of stop for the algorithms.

    enum StoppingCondition{MaximumTime, SelectionErrorGoal, MaximumEpochs, MaximumSelectionFailures, MaximumNeurons};

    // Get methods

    TrainingStrategy* get_training_strategy_pointer() const;

    bool has_training_strategy() const;

    const Index& get_maximum_neurons() const;
    const Index& get_minimum_neurons() const;
    const Index& get_trials_number() const;

    const bool& get_reserve_training_errors() const;
    const bool& get_reserve_selection_errors() const;

    const bool& get_display() const;

    const type& get_selection_error_goal() const;
    const Index& get_maximum_epochs_number() const;
    const type& get_maximum_time() const;

    // Set methods

    void set_training_strategy_pointer(TrainingStrategy*);

    void set_default();

    void set_maximum_neurons(const Index&);
    void set_minimum_neurons(const Index&);
    void set_trials_number(const Index&);

    void set_reserve_training_error_data(const bool&);
    void set_reserve_selection_error_data(const bool&);

    void set_display(const bool&);

    void set_selection_error_goal(const type&);
    void set_maximum_epochs_number(const Index&);
    void set_maximum_time(const type&);

    // Loss calculation methods

    string write_stopping_condition(const TrainingResults&) const;

    // Neuron selection methods

    void delete_selection_history();
    void delete_training_error_history();
    void check() const;

    // Utilities

    /// Performs the neurons selection for a neural network.

    virtual NeuronsSelectionResults perform_neurons_selection() = 0;

    /// Writes the time from seconds in format HH:mm:ss.

    const string write_elapsed_time(const type&) const;

protected:

    /// Pointer to a training strategy object.

    TrainingStrategy* training_strategy_pointer = nullptr;

    /// Neurons of all the neural networks trained.

    Tensor<Index, 1> neurons_history;

    /// Selection loss of all the neural networks trained.

    Tensor<type, 1> selection_error_history;

    /// Error of all the neural networks trained.

    Tensor<type, 1> training_error_history;

    /// Parameters of all the neural networks trained.

    Tensor<Tensor<type, 1>, 1> parameters_history;

    /// Minimum number of hidden neurons.

    Index minimum_neurons;

    /// Maximum number of hidden neurons.

    Index maximum_neurons;

    /// Number of trials for each neural network.

    Index trials_number = 1;

    // Neurons selection results

    /// True if the loss of all neural networks are to be reserved.

    bool reserve_training_errors;

    /// True if the selection error of all neural networks are to be reserved.

    bool reserve_selection_errors;

    /// Display messages to screen.

    bool display = true;

    /// Goal value for the selection error. It is used as a stopping criterion.

    type selection_error_goal;

    /// Maximum number of epochs to perform neurons selection. It is used as a stopping criterion.

    Index maximum_epochs_number;

    /// Maximum selection algorithm time. It is used as a stopping criterion.

    type maximum_time;
};


/// This structure contains the results from the neurons selection.

struct NeuronsSelectionResults
{
    // Default constructor

    explicit NeuronsSelectionResults() {}

    // Epochs constructor

   explicit NeuronsSelectionResults(const Index& maximum_epochs_number)
   {
        neurons_numbers.resize(maximum_epochs_number);

        training_errors.resize(maximum_epochs_number);
        selection_errors.resize(maximum_epochs_number);

        optimum_training_error = numeric_limits<type>::max();
        optimum_selection_error = numeric_limits<type>::max();
    }

   virtual ~NeuronsSelectionResults() {}

   string write_stopping_condition() const;

   // Neural network

   /// Neurons of the diferent neural networks.

   Tensor<Index, 1> neurons_numbers;

   /// Neurons of the neural network with minimum selection error.

   Index optimal_neurons_number;

   /// Vector of parameters for the neural network with minimum selection error.

   Tensor<type, 1> optimal_parameters;

   // Loss index

   /// Performance of the different neural networks.

   Tensor<type, 1> training_errors;

   /// Selection loss of the different neural networks.

   Tensor<type, 1> selection_errors;

   /// Value of loss for the neural network with minimum selection error.

   type optimum_training_error;

   /// Value of minimum selection error.

   type optimum_selection_error;

   // Model selection

   /// Number of iterations to perform the neurons selection.

   Index epochs_number;

   /// Stopping condition of the algorithm.

   NeuronsSelection::StoppingCondition stopping_condition;

   /// Elapsed time during the loss of the algortihm.

   string elapsed_time;
};


}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
