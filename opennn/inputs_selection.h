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

    enum StoppingCondition{MaximumTime,SelectionErrorGoal,MaximumInputs,MinimumInputs,MaximumEpochs,
                           MaximumSelectionFailures,CorrelationGoal,AlgorithmFinished};

    // STRUCTURES

    /// This structure contains the results from the inputs selection.

    struct Results
    {
       explicit Results() {}

       virtual ~Results() {}

       string write_stopping_condition() const;

       /// Inputs of the different neural networks.

       Tensor<bool, 2> inputs_data;
       
       /// Performance of the different neural networks.

       Tensor<type, 1> training_error_data;

       /// Selection loss of the different neural networks.

       Tensor<type, 1> selection_error_data;

       /// Vector of parameters for the neural network with minimum selection error.

       Tensor<type, 1> minimal_parameters;

       /// Value of minimum selection error.

       type final_selection_error;

       /// Value of loss for the neural network with minimum selection error.

       type final_training_error;

       /// Inputs of the neural network with minimum selection error.

       Tensor<Index, 1> optimal_inputs_indices;

       /// Inputs of the neural network with minimum selection error.

       Tensor<bool, 1> optimal_inputs;

       /// Number of iterations to perform the inputs selection.

       Index iterations_number;

       /// Stopping condition of the algorithm.

       StoppingCondition stopping_condition;

       /// Elapsed time during the loss of the algortihm.

       string elapsed_time;
    };

    // Get methods

    const bool& get_approximation() const;

    TrainingStrategy* get_training_strategy_pointer() const;

    bool has_training_strategy() const;

    const Index& get_trials_number() const;

    const bool& get_reserve_training_error_data() const;
    const bool& get_reserve_selection_error_data() const;
    const bool& get_reserve_minimal_parameters() const;

    const bool& get_display() const;

    const type& get_selection_error_goal() const;
    const Index& get_maximum_iterations_number() const;
    const type& get_maximum_time() const;
    const type& get_maximum_correlation() const;
    const type& get_minimum_correlation() const;
    const type& get_tolerance() const;

    // Set methods

    void set_approximation(const bool&);

    void set_training_strategy_pointer(TrainingStrategy*);

    void set_default();

    void set_trials_number(const Index&);

    void set_reserve_training_error_data(const bool&);
    void set_reserve_selection_error_data(const bool&);
    void set_reserve_minimal_parameters(const bool&);

    void set_display(const bool&);

    void set_selection_error_goal(const type&);
    void set_maximum_iterations_number(const Index&);
    void set_maximum_time(const type&);
    void set_maximum_correlation(const type&);
    void set_minimum_correlation(const type&);
    void set_tolerance(const type&);

    // Performances calculation methods

    Tensor<type, 1> calculate_losses(const Tensor<bool, 1>&);

    Tensor<type, 1> get_parameters_inputs(const Tensor<bool, 1>&) const;

    string write_stopping_condition(const OptimizationAlgorithm::Results&) const;

    // inputs selection methods

    void delete_selection_history();
    void delete_loss_history();
    void delete_parameters_history();
    void check() const;

    // Utilities

    Tensor<type, 1> insert_result(const type&, const Tensor<type, 1>&) const;
    Tensor<Index, 1> insert_result(const Index&, const Tensor<Index, 1>&) const;
    Tensor< Tensor<type, 1>, 1> insert_result(const Tensor<type, 1>&, const Tensor< Tensor<type, 1>, 1>&) const;

    Tensor<Index, 1> delete_result(const Index&, const Tensor<Index, 1>&) const;

    Index get_input_index(const Tensor<DataSet::VariableUse, 1>, const Index);

    /// Performs the inputs selection for a neural network.

    virtual Results* perform_inputs_selection() = 0;

    /// Writes the time from seconds in format HH:mm:ss.

    const string write_elapsed_time(const type&) const;

protected:

    /// Pointer to a training strategy object.

    TrainingStrategy* training_strategy_pointer = nullptr;

    /// True if this is a function regression problem.

    bool approximation;

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

    bool reserve_training_error_data;

    /// True if the selection error of all neural networks are to be reserved.

    bool reserve_selection_error_data;

    /// True if the vector parameters of the neural network presenting minimum selection error is to be reserved.

    bool reserve_minimal_parameters;

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

    /// Tolerance for the error in the trainings of the algorithm.

    type tolerance;
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
