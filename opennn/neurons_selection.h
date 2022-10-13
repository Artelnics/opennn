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

namespace opennn
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

    // Enumerators

    /// Enumeration of all possible conditions of stop for the algorithms.

    enum class StoppingCondition{MaximumTime, SelectionErrorGoal, MaximumEpochs, MaximumSelectionFailures, MaximumNeurons};

    // Get methods

    TrainingStrategy* get_training_strategy_pointer() const;

    bool has_training_strategy() const;

    const Index& get_maximum_neurons() const;
    const Index& get_minimum_neurons() const;
    const Index& get_trials_number() const;

    const bool& get_display() const;

    const type& get_selection_error_goal() const;
    const Index& get_maximum_epochs_number() const;
    const type& get_maximum_time() const;

    // Set methods

    void set_training_strategy_pointer(TrainingStrategy*);

    void set_default();

    void set_maximum_neurons_number(const Index&);
    void set_minimum_neurons(const Index&);
    void set_trials_number(const Index&);

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

    string write_time(const type&) const;

protected:

    /// Pointer to a training strategy object.

    TrainingStrategy* training_strategy_pointer = nullptr;

    /// Neurons of all the neural networks trained.

    Tensor<Index, 1> neurons_history;

    /// Selection loss of all the neural networks trained.

    Tensor<type, 1> selection_error_history;

    /// Error of all the neural networks trained.

    Tensor<type, 1> training_error_history;

    /// Minimum number of hidden neurons.

    Index minimum_neurons;

    /// Maximum number of hidden neurons.

    Index maximum_neurons;

    /// Number of trials for each neural network.

    Index trials_number = 1;

    /// Display messages to screen.

    bool display = true;

    /// Goal value for the selection error. It is a stopping criterion.

    type selection_error_goal;

    /// Maximum number of epochs to perform neurons selection. It is a stopping criterion.

    Index maximum_epochs_number;

    /// Maximum selection algorithm time. It is a stopping criterion.

    type maximum_time;
};


/// This structure contains the results from the neurons selection.

struct NeuronsSelectionResults
{
    // Default constructor

    explicit NeuronsSelectionResults()
    {
    }

    // Epochs constructor

   explicit NeuronsSelectionResults(const Index& maximum_epochs_number)
   {
        neurons_number_history.resize(maximum_epochs_number);
        neurons_number_history.setConstant(0);

        training_error_history.resize(maximum_epochs_number);
        training_error_history.setConstant(type(-1));

        selection_error_history.resize(maximum_epochs_number);
        selection_error_history.setConstant(type(-1));

        optimum_training_error = numeric_limits<type>::max();
        optimum_selection_error = numeric_limits<type>::max();
    }

   virtual ~NeuronsSelectionResults() {}

   void resize_history(const Index& new_size)
   {
       const Tensor<Index, 1> old_neurons_number_history = neurons_number_history;
       const Tensor<type, 1> old_training_error_history = training_error_history;
       const Tensor<type, 1> old_selection_error_history = selection_error_history;

       neurons_number_history.resize(new_size);
       training_error_history.resize(new_size);
       selection_error_history.resize(new_size);

       for(Index i = 0; i < new_size; i++)
       {
           neurons_number_history(i) = old_neurons_number_history(i);
           training_error_history(i) = old_training_error_history(i);
           selection_error_history(i) = old_selection_error_history(i);
       }
   }

   string write_stopping_condition() const;

   void print() const
   {
       cout << endl;
       cout << "Neurons Selection Results" << endl;

       cout << "Optimal neurons number: " << optimal_neurons_number << endl;

       cout << "Optimum training error: " << optimum_training_error << endl;
       cout << "Optimum selection error: " << optimum_selection_error << endl;
   }

   // Neural network

   /// Neurons of the diferent neural networks.

   Tensor<Index, 1> neurons_number_history;

   /// Neurons of the neural network with minimum selection error.

   Index optimal_neurons_number = 1;

   /// Vector of parameters for the neural network with minimum selection error.

   Tensor<type, 1> optimal_parameters;

   // Loss index

   /// Performance of the different neural networks.

   Tensor<type, 1> training_error_history;

   /// Selection loss of the different neural networks.

   Tensor<type, 1> selection_error_history;

   /// Value of loss for the neural network with minimum selection error.

   type optimum_training_error /*- static_cast<type>(FLT_MAX)*/;

   /// Value of minimum selection error.

   type optimum_selection_error /*- static_cast<type>(FLT_MAX)*/;

   // Model selection

   /// Stopping condition of the algorithm.

   NeuronsSelection::StoppingCondition stopping_condition = NeuronsSelection::StoppingCondition::MaximumTime;

   /// Elapsed time during the loss of the algortihm.

   string elapsed_time;
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
