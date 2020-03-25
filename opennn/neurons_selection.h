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

#include "vector.h"
#include "matrix.h"

#include "training_strategy.h"

#include "tinyxml2.h"

namespace OpenNN
{

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

    explicit NeuronsSelection(const string&);

    explicit NeuronsSelection(const tinyxml2::XMLDocument&);

    // Destructor

    virtual ~NeuronsSelection();

    // Enumerators

    /// Enumeration of all possibles condition of stop for the algorithms.

    enum StoppingCondition{MaximumTime, SelectionErrorGoal, MaximumIterations, MaximumSelectionFailures, AlgorithmFinished};

    // Structures

    /// This structure contains the results from the order selection.

    struct Results
    {
       explicit Results() {}

       virtual ~Results() {}

       string write_stopping_condition() const;

       string object_to_string() const;

       /// Order of the diferent neural networks.

       Vector<size_t> neurons_data;

       /// Performance of the different neural networks.

       Vector<double> training_loss_data;

       /// Selection loss of the different neural networks.

       Vector<double> selection_error_data;

       /// Vector of parameters for the neural network with minimum selection error.

       Vector<double> minimal_parameters;

       /// Value of minimum selection error.

       double final_selection_error;

       /// Value of loss for the neural network with minimum selection error.

       double final_training_loss;

       /// Order of the neural network with minimum selection error.

       size_t optimal_neurons_number;

       /// Number of iterations to perform the order selection.

       size_t iterations_number;

       /// Stopping condition of the algorithm.

       StoppingCondition stopping_condition;

       /// Elapsed time during the loss of the algortihm.

       double elapsed_time;
    };

    // Get methods

    TrainingStrategy* get_training_strategy_pointer() const;

    bool has_training_strategy() const;

    const size_t& get_maximum_order() const;
    const size_t& get_minimum_order() const;
    const size_t& get_trials_number() const;

    const bool& get_reserve_error_data() const;
    const bool& get_reserve_selection_error_data() const;
    const bool& get_reserve_minimal_parameters() const;

    const bool& get_display() const;

    const double& get_selection_error_goal() const;
    const size_t& get_maximum_iterations_number() const;
    const double& get_maximum_time() const;
    const double& get_tolerance() const;

    // Set methods

    void set_training_strategy_pointer(TrainingStrategy*);

    void set_default();

    void set_maximum_order(const size_t&);
    void set_minimum_order(const size_t&);
    void set_trials_number(const size_t&);

    void set_reserve_error_data(const bool&);
    void set_reserve_selection_error_data(const bool&);
    void set_reserve_minimal_parameters(const bool&);

    void set_display(const bool&);

    void set_selection_error_goal(const double&);
    void set_maximum_iterations_number(const size_t&);
    void set_maximum_time(const double&);
    void set_tolerance(const double&);

    // Loss calculation methods

    Vector<double> calculate_losses(const size_t&, NeuralNetwork&);

    string write_stopping_condition(const OptimizationAlgorithm::Results&) const;

    // order order selection methods

    void delete_selection_history();
    void delete_training_loss_history();
    void check() const;

    /// Performs the order selection for a neural network.

    virtual Results* perform_neurons_selection() = 0;

protected:

    /// Pointer to a training strategy object.

    TrainingStrategy* training_strategy_pointer = nullptr;

    /// Order of all the neural networks trained.

    Vector<size_t> order_history;

    /// Selection loss of all the neural networks trained.

    Vector<double> selection_error_history;

    /// Performance of all the neural networks trained.

    Vector<double> training_loss_history;

    Vector<Vector<double>> parameters_history;

    /// Minimum number of hidden neurons.

    size_t minimum_order;

    /// Maximum number of hidden neurons.

    size_t maximum_order;

    /// Number of trials for each neural network.

    size_t trials_number;

    // Order selection results

    /// True if the loss of all neural networks are to be reserved.

    bool reserve_error_data;

    /// True if the selection error of all neural networks are to be reserved.

    bool reserve_selection_error_data;

    /// True if the vector parameters of the neural network presenting minimum selection error is to be reserved.

    bool reserve_minimal_parameters;

    /// Display messages to screen.

    bool display;

    /// Goal value for the selection error. It is used as a stopping criterion.

    double selection_error_goal;

    /// Maximum number of iterations to perform_neurons_selection. It is used as a stopping criterion.

    size_t maximum_iterations_number;

    /// Maximum selection algorithm time. It is used as a stopping criterion.

    double maximum_time;

    /// Tolerance for the error in the trainings of the algorithm.

    double tolerance;
};
}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
