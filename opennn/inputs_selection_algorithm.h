/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N P U T S   S E L E C T I O N   A L G O R I T H M   C L A S S   H E A D E R                              */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __INPUTSSELECTIONALGORITHM_H__
#define __INPUTSSELECTIONALGORITHM_H__

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>
#include <limits>

// OpenNN includes

#include "vector.h"
#include "matrix.h"

#include "training_strategy.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

/// This abstract class represents the concept of inputs selection algorithm for a neural network.
/// Any derived class must implement the perform_inputs_selection() method.

class InputsSelectionAlgorithm
{
public:

    // DEFAULT CONSTRUCTOR

    explicit InputsSelectionAlgorithm();

    // TRAINING STRATEGY CONSTRUCTOR

    explicit InputsSelectionAlgorithm(TrainingStrategy*);

    // FILE CONSTRUCTOR

    explicit InputsSelectionAlgorithm(const string&);

    // XML CONSTRUCTOR

    explicit InputsSelectionAlgorithm(const tinyxml2::XMLDocument&);


    // DESTRUCTOR

    virtual ~InputsSelectionAlgorithm();

    // ENUMERATIONS

    /// Enumeration of available methods for the calculus of the losses.

    enum LossCalculationMethod{Minimum, Maximum, Mean};

    /// Enumeration of all possibles condition of stop for the algorithms.

    enum StoppingCondition{
        MaximumTime,
        SelectionErrorGoal,
        MaximumInputs,
        MinimumInputs,
        MaximumIterations,
        MaximumSelectionFailures,
        CorrelationGoal,
        AlgorithmFinished};

    // STRUCTURES

    ///
    /// This structure contains the results from the inputs selection.
    ///

    struct InputsSelectionResults
    {
       explicit InputsSelectionResults()
       {

       }

       virtual ~InputsSelectionResults()
       {

       }

       string write_stopping_condition() const;

       string object_to_string() const;

       /// Inputs of the different neural networks.

       Vector< Vector<bool> > inputs_data;

       /// Parameters of the different neural networks.

       Vector< Vector<double> > parameters_data;

       /// Performance of the different neural networks.

       Vector<double> loss_data;

       /// Selection loss of the different neural networks.

       Vector<double> selection_error_data;

       /// Vector of parameters for the neural network with minimum selection error.

       Vector<double> minimal_parameters;

       /// Value of minimum selection error.

       double final_selection_error;

       /// Value of loss for the neural network with minimum selection error.

       double final_loss;

       /// Inputs of the neural network with minimum selection error.

       Vector<bool> optimal_inputs;

       /// Number of iterations to perform the inputs selection.

       size_t iterations_number;

       /// Stopping condition of the algorithm.

       StoppingCondition stopping_condition;

       /// Elapsed time during the loss of the algortihm.

       double elapsed_time;
    };

    // METHODS

    // Get methods

    const bool& get_approximation() const;

    TrainingStrategy* get_training_strategy_pointer() const;

    bool has_training_strategy() const;

    const size_t& get_trials_number() const;

    const bool& get_reserve_parameters_data() const;
    const bool& get_reserve_error_data() const;
    const bool& get_reserve_selection_error_data() const;
    const bool& get_reserve_minimal_parameters() const;

    const LossCalculationMethod& get_loss_calculation_method() const;

    const bool& get_display() const;

    const double& get_selection_error_goal() const;
    const size_t& get_maximum_iterations_number() const;
    const double& get_maximum_time() const;
    const double& get_maximum_correlation() const;
    const double& get_minimum_correlation() const;
    const double& get_tolerance() const;

    string write_loss_calculation_method() const;

    // Set methods

    void set_approximation(const bool&);

    void set_training_strategy_pointer(TrainingStrategy*);

    void set_default();

    void set_trials_number(const size_t&);

    void set_reserve_parameters_data(const bool&);
    void set_reserve_error_data(const bool&);
    void set_reserve_selection_error_data(const bool&);
    void set_reserve_minimal_parameters(const bool&);

    void set_loss_calculation_method(const LossCalculationMethod&);
    void set_loss_calculation_method(const string&);

    void set_display(const bool&);

    void set_selection_error_goal(const double&);
    void set_maximum_iterations_number(const size_t&);
    void set_maximum_time(const double&);
    void set_maximum_correlation(const double&);
    void set_minimum_correlation(const double&);
    void set_tolerance(const double&);

    // Performances calculation methods

    Vector<double> perform_minimum_model_evaluation(const Vector<bool>&);
    Vector<double> perform_maximum_model_evaluation(const Vector<bool>&);
    Vector<double> perform_mean_model_evaluation(const Vector<bool>&);

    Vector<double> get_final_losses(const TrainingStrategy::Results&) const;

    Vector<double> perform_model_evaluation(const Vector<bool>&);

    Vector<double> get_parameters_inputs(const Vector<bool>&) const;

    string write_stopping_condition(const TrainingStrategy::Results&) const;

    // inputs selection methods

    void delete_selection_history();
    void delete_loss_history();
    void delete_parameters_history();
    void check() const;

    size_t get_input_index(const Vector<Variables::Use>, const size_t);

    /// Performs the inputs selection for a neural network.

    virtual InputsSelectionResults* perform_inputs_selection() = 0;

protected:

    // MEMBERS

    /// True if this is a function regression problem.

    bool approximation;

    /// Pointer to a training strategy object.

    TrainingStrategy* training_strategy_pointer;

    /// Inputs of all the neural networks trained.

    Vector< Vector<bool> > inputs_history;

    /// Selection loss of all the neural networks trained.

    Vector<double> selection_error_history;

    /// Performance of all the neural networks trained.

    Vector<double> loss_history;

    /// Parameters of all the neural network trained.

    Vector< Vector<double> > parameters_history;

    /// Number of trials for each neural network.

    size_t trials_number;

    /// Method used for the calculation of the loss and the generalizaton loss.

    LossCalculationMethod loss_calculation_method;

    // Inputs selection results

    /// True if the parameters of all neural networks are to be reserved.

    bool reserve_parameters_data;

    /// True if the loss of all neural networks are to be reserved.

    bool reserve_error_data;

    /// True if the selection error of all neural networks are to be reserved.

    bool reserve_selection_error_data;

    /// True if the vector parameters of the neural network presenting minimum selection error is to be reserved.

    bool reserve_minimal_parameters;

    /// Display messages to screen.

    bool display;

    // STOPPING CRITERIA

    /// Goal value for the selection error. It is used as a stopping criterion.

    double selection_error_goal;

    /// Maximum number of iterations to perform_inputs_selection. It is used as a stopping criterion.

    size_t maximum_iterations_number;

    /// Maximum value for the correlations.

    double maximum_correlation;

    /// Minimum value for the correlations.

    double minimum_correlation;

    /// Maximum selection algorithm time. It is used as a stopping criterion.

    double maximum_time;

    /// Tolerance for the error in the trainings of the algorithm.

    double tolerance;
};
}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
