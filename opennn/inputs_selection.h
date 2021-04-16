//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   A L G O R I T H M   C L A S S   H E A D E R
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

namespace OpenNN
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

    // Destructor

    virtual ~InputsSelection();

    // Enumerations

    /// Enumeration of all possibles condition of stop for the algorithms.

    enum StoppingCondition{MaximumTime, SelectionErrorGoal, MaximumInputs, MinimumInputs, MaximumEpochs,
                           MaximumSelectionFailures, CorrelationGoal};

    // Get methods

    TrainingStrategy* get_training_strategy_pointer() const;

    bool has_training_strategy() const;

    const Index& get_trials_number() const;

    const bool& get_reserve_training_errors() const;
    const bool& get_reserve_selection_errors() const;

    const bool& get_display() const;

    const type& get_selection_error_goal() const;
    const Index& get_maximum_iterations_number() const;
    const type& get_maximum_time() const;
    const type& get_maximum_correlation() const;
    const type& get_minimum_correlation() const;
    const type& get_tolerance() const;

    // Set methods

    void set_training_strategy_pointer(TrainingStrategy*);

    void set_default();

    void set_trials_number(const Index&);

    void set_reserve_training_error_data(const bool&);
    void set_reserve_selection_error_data(const bool&);

    void set_display(const bool&);

    void set_selection_error_goal(const type&);
    void set_maximum_iterations_number(const Index&);
    void set_maximum_time(const type&);
    void set_maximum_correlation(const type&);
    void set_minimum_correlation(const type&);

    // Performances calculation methods

    string write_stopping_condition(const TrainingResults&) const;

    // inputs selection methods

    void check() const;

    // Utilities

    Index get_input_index(const Tensor<DataSet::VariableUse, 1>&, const Index&);

    /// Performs the inputs selection for a neural network.

    virtual InputsSelectionResults perform_inputs_selection() = 0;

    /// Writes the time from seconds in format HH:mm:ss.

    const string write_elapsed_time(const type&) const;

protected:

    /// Pointer to a training strategy object.

    TrainingStrategy* training_strategy_pointer = nullptr;

    /// Inputs of all the neural networks trained.

    Tensor<bool, 2> inputs_history;

    /// Selection loss of all the neural networks trained.

    Tensor<type, 1> selection_error_history;

    /// Performance of all the neural networks trained.

    Tensor<type, 1> training_error_history;

    /// Parameters of all the neural network trained.

    Tensor<Tensor<type, 1>, 1> parameters_history;

    /// Number of trials for each neural network.

    Index trials_number = 1;

    // Inputs selection results

    /// True if the loss of all neural networks are to be reserved.

    bool reserve_training_errors;

    /// True if the selection error of all neural networks are to be reserved.

    bool reserve_selection_errors;

    /// Display messages to screen.

    bool display = true;

    // Stopping criteria

    /// Goal value for the selection error. It is used as a stopping criterion.

    type selection_error_goal;

    /// Maximum number of epochs to perform_inputs_selection. It is used as a stopping criterion.

    Index maximum_epochs_number;

    /// Maximum value for the correlations.

    type maximum_correlation;

    /// Minimum value for the correlations.

    type minimum_correlation;

    /// Maximum selection algorithm time. It is used as a stopping criterion.

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

    void set(const Index& maximum_epochs_number)
    {
        training_errors.resize(maximum_epochs_number);

        selection_errors.resize(maximum_epochs_number);
    }

   virtual ~InputsSelectionResults() {}

   string write_stopping_condition() const;

   void print()
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

   /// Performance of the different neural networks.

   Tensor<type, 1> training_errors;

   /// Selection loss of the different neural networks.

   Tensor<type, 1> selection_errors;

   /// Value of training for the neural network with minimum selection error.

   type optimum_training_error = numeric_limits<type>::max();

   /// Value of minimum selection error.

   type optimum_selection_error = numeric_limits<type>::max();

   /// Inputs of the neural network with minimum selection error.

   Tensor<string, 1> optimal_input_columns_names;

   Tensor<Index, 1> optimal_input_columns_indices;

   Tensor<bool, 1> optimal_inputs;

   // Model selection

   /// Number of iterations to perform the inputs selection.

   Index epochs_number;

   /// Stopping condition of the algorithm.

   InputsSelection::StoppingCondition stopping_condition;

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
