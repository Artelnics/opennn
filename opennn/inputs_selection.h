//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef INPUTSSELECTIONALGORITHM_H
#define INPUTSSELECTIONALGORITHM_H

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>
#include <limits>

// OpenNN includes

#include "training_strategy.h"
#include "config.h"

namespace opennn
{

struct InputsSelectionResults;

/// This abstract class represents the concept of inputs selection algorithm for a ModelSelection[1].

///
/// Any derived class must implement the perform_inputs_selection() method.
///
/// [1] Neural Designer "Model Selection Algorithms in Predictive Analytics."
/// \ref https://www.neuraldesigner.com/blog/model-selection

class InputsSelection
{
public:

    // Constructors

    explicit InputsSelection();

    explicit InputsSelection(TrainingStrategy*);

    // Enumerations

    /// Enumeration of all possible conditions of stop for the algorithms.

    enum class StoppingCondition{MaximumTime, SelectionErrorGoal, MaximumInputs, MinimumInputs, MaximumEpochs,
                           MaximumSelectionFailures, CorrelationGoal};

    // Get methods

    TrainingStrategy* get_training_strategy_pointer() const;

    bool has_training_strategy() const;

    const Index& get_trials_number() const;

    const bool& get_display() const;

    const type& get_selection_error_goal() const;
    const Index& get_maximum_iterations_number() const;
    const type& get_maximum_time() const;
    const type& get_maximum_correlation() const;
    const type& get_minimum_correlation() const;
    const type& get_tolerance() const;

    // Set methods

    void set(TrainingStrategy*);

    virtual void set_default();

    void set_trials_number(const Index&);

    void set_display(const bool&);

    void set_selection_error_goal(const type&);
    void set_maximum_epochs_number(const Index&);
    void set_maximum_time(const type&);
    void set_maximum_correlation(const type&);
    void set_minimum_correlation(const type&);

    // Performances calculation methods

    string write_stopping_condition(const TrainingResults&) const;

    // inputs selection methods

    void check() const;

    // Utilities

    Index get_input_index(const Tensor<DataSet::VariableUse, 1>&, const Index&) const;

    /// Performs the inputs selection for a neural network.

    virtual InputsSelectionResults perform_inputs_selection() = 0;

    /// Writes the time from seconds in format HH:mm:ss.

    string write_time(const type&) const;

protected:

    /// Pointer to a training strategy object.

    TrainingStrategy* training_strategy_pointer = nullptr;

    Tensor<Index, 1> original_input_columns_indices;
    Tensor<Index, 1> original_target_columns_indices;

    /// Number of trials for each neural network.

    Index trials_number = 1;

    /// Display messages to screen.

    bool display = true;

    // Stopping criteria

    /// Goal value for the selection error. It is a stopping criterion.

    type selection_error_goal;

    /// Maximum number of epochs to perform_inputs_selection. It is a stopping criterion.

    Index maximum_epochs_number;

    /// Maximum value for the correlations.

    type maximum_correlation;

    /// Minimum value for the correlations.

    type minimum_correlation;

    /// Maximum selection algorithm time. It is a stopping criterion.

    type maximum_time;

    const Eigen::array<int, 1> rows_sum = {Eigen::array<int, 1>({1})};
};


/// This structure contains the results from the inputs selection.

struct InputsSelectionResults
{
    // Default constructor

    explicit InputsSelectionResults()
    {
    }

    // Maximum epochs number constructor

    explicit InputsSelectionResults(const Index& maximum_epochs_number)
    {
        set(maximum_epochs_number);
    }

    Index get_epochs_number() const
    {
        return training_error_history.size();
    }

    void set(const Index& maximum_epochs_number)
    {
        training_error_history.resize(maximum_epochs_number);
        training_error_history.setConstant(type(-1));

        selection_error_history.resize(maximum_epochs_number);
        selection_error_history.setConstant(type(-1));
    }

   virtual ~InputsSelectionResults() {}

   string write_stopping_condition() const;

   void resize_history(const Index& new_size)
   {
       const Tensor<type, 1> old_training_error_history = training_error_history;
       const Tensor<type, 1> old_selection_error_history = selection_error_history;

       training_error_history.resize(new_size);
       selection_error_history.resize(new_size);

       for(Index i = 0; i < new_size; i++)
       {
           training_error_history(i) = old_training_error_history(i);
           selection_error_history(i) = old_selection_error_history(i);
       }
   }


   void print() const
   {
       cout << endl;
       cout << "Inputs Selection Results" << endl;

       cout << "Optimal inputs number: " << optimal_input_columns_names.size() << endl;

       cout << "Inputs: " << endl;

       for(Index i = 0; i < optimal_input_columns_names.size(); i++) cout << "   " << optimal_input_columns_names(i) << endl;

       cout << "Optimum training error: " << optimum_training_error << endl;
       cout << "Optimum selection error: " << optimum_selection_error << endl;
   }


   // Neural network

   /// Vector of parameters for the neural network with minimum selection error.

   Tensor<type, 1> optimal_parameters;

   // Loss index

   /// Final training errors of the different neural networks.

   Tensor<type, 1> training_error_history;

   /// Final selection errors of the different neural networks.

   Tensor<type, 1> selection_error_history;

   /// Value of training for the neural network with minimum selection error.

   type optimum_training_error = numeric_limits<type>::max();

   /// Value of minimum selection error.

   type optimum_selection_error = numeric_limits<type>::max();

   /// Inputs of the neural network with minimum selection error.

   Tensor<string, 1> optimal_input_columns_names;

   Tensor<Index, 1> optimal_input_columns_indices;

   Tensor<bool, 1> optimal_inputs;

   // Model selection

   /// Stopping condition of the algorithm.

   InputsSelection::StoppingCondition stopping_condition = InputsSelection::StoppingCondition::MaximumTime;

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
