/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   I N P U T S   S E L E C T I O N   A L G O R I T H M   C L A S S   H E A D E R                              */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
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

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

/// This abstract class represents the concept of inputs selection algorithm for a neural network.
/// Any derived class must implement the perform_inputs_selection(void) method.

class InputsSelectionAlgorithm
{
public:

    // DEFAULT CONSTRUCTOR

    explicit InputsSelectionAlgorithm(void);

    // TRAINING STRATEGY CONSTRUCTOR

    explicit InputsSelectionAlgorithm(TrainingStrategy*);

    // FILE CONSTRUCTOR

    explicit InputsSelectionAlgorithm(const std::string&);

    // XML CONSTRUCTOR

    explicit InputsSelectionAlgorithm(const tinyxml2::XMLDocument&);


    // DESTRUCTOR

    virtual ~InputsSelectionAlgorithm(void);

    // ENUMERATIONS

    /// Enumeration of available methods for the calculus of the performances.

    enum PerformanceCalculationMethod{Minimum, Maximum, Mean};

    /// Enumeration of all possibles condition of stop for the algorithms.

    enum StoppingCondition{MaximumTime, GeneralizationPerformanceGoal, MaximumIterations, MaximumGeneralizationFailures, CorrelationGoal, AlgorithmFinished};

    // STRUCTURES

    ///
    /// This structure contains the results from the inputs selection.
    ///

    struct InputsSelectionResults
    {
       explicit InputsSelectionResults(void)
       {

       }

       virtual ~InputsSelectionResults(void)
       {

       }

       std::string write_stopping_condition(void) const;

       std::string to_string(void) const;

       /// Parameters of the different neural networks.

       Vector< Vector<double> > parameters_data;

       /// Performance of the different neural networks.

       Vector< Vector<double> > performance_data;

       /// Generalization performance of the different neural networks.

       Vector< Vector<double> > generalization_performance_data;

       /// Vector of parameters for the neural network with minimum generalization performance.

       Vector<double> minimal_parameters;

       /// Value of minimum generalization performance.

       double final_generalization_performance;

       /// Value of performance for the neural network with minimum generalization performance.

       double final_performance;

       /// Inputs of the neural network with minimum generalization performance.

       Vector<double> optimal_inputs;

       /// Number of iterations to perform the inputs selection.

       size_t iterations_number;

       /// Stopping condition of the algorithm.

       StoppingCondition stopping_condition;

       /// Elapsed time during the performance of the algortihm.

       double elapsed_time;
    };

    // METHODS

    // Get methods

    const bool& get_regression(void) const;

    TrainingStrategy* get_training_strategy_pointer(void) const;

    bool has_training_strategy(void) const;

    const size_t& get_trials_number(void) const;

    const bool& get_reserve_parameters_data(void) const;
    const bool& get_reserve_performance_data(void) const;
    const bool& get_reserve_generalization_performance_data(void) const;
    const bool& get_reserve_minimal_parameters(void) const;

    const PerformanceCalculationMethod& get_performance_calculation_method(void) const;

    const bool& get_display(void) const;

    const double& get_generalization_performance_goal(void) const;
    const size_t& get_maximum_iterations_number(void) const;
    const double& get_maximum_time(void) const;
    const double& get_maximum_correlation(void) const;
    const double& get_minimum_correlation(void) const;
    const double& get_tolerance(void) const;

    std::string write_performance_calculation_method(void) const;

    // Set methods

    void set_regression(const bool&);

    void set_training_strategy_pointer(TrainingStrategy*);

    void set_default(void);

    void set_trials_number(const size_t&);

    void set_reserve_parameters_data(const bool&);
    void set_reserve_performance_data(const bool&);
    void set_reserve_generalization_performance_data(const bool&);
    void set_reserve_minimal_parameters(const bool&);

    void set_performance_calculation_method(const PerformanceCalculationMethod&);
    void set_performance_calculation_method(const std::string&);

    void set_display(const bool&);

    void set_generalization_performance_goal(const double&);
    void set_maximum_iterations_number(const size_t&);
    void set_maximum_time(const double&);
    void set_maximum_correlation(const double&);
    void set_minimum_correlation(const double&);
    void set_tolerance(const double&);

    // Correlation methods

    Matrix<double> calculate_logistic_correlations(void) const;

    Vector<double> calculate_final_correlations(void) const;

    // Performances calculation methods

    void set_neural_inputs(const Vector<double>&);

    Vector<double> calculate_minimum_final_performances(const Vector<double>&);
    Vector<double> calculate_maximum_final_performances(const Vector<double>&);
    Vector<double> calculate_mean_final_performances(const Vector<double>&);

    Vector<double> get_final_performances(const TrainingStrategy::Results&);

    Vector<double> calculate_performances(const Vector<double>&);

    Vector<double> get_parameters_inputs(const Vector<double>&);

    // inputs selection methods

    void delete_generalization_history(void);
    void delete_performance_history(void);
    void delete_parameters_history(void);
    void check(void) const;

    size_t get_input_index(const Vector<Variables::Use>, const size_t);

    /// Performs the inputs selection for a neural network.

    virtual InputsSelectionResults* perform_inputs_selection(void) = 0;


protected:

    // MEMBERS

    /// True if this is a regression problem.

    bool regression;

    /// Pointer to a training strategy object.

    TrainingStrategy* training_strategy_pointer;

    /// Generalization performance of all the neural networks trained.

    Vector< Vector<double> > generalization_performance_history;

    /// Performance of all the neural networks trained.

    Vector< Vector<double> > performance_history;

    /// Parameters of all the neural network trained.

    Vector< Vector<double> > parameters_history;

    /// Number of trials for each neural network.

    size_t trials_number;

    /// Method used for the calculation of the performance and the generalizaton performance.

    PerformanceCalculationMethod performance_calculation_method;

    // Inputs selection results

    /// True if the parameters of all neural networks are to be reserved.

    bool reserve_parameters_data;

    /// True if the performance of all neural networks are to be reserved.

    bool reserve_performance_data;

    /// True if the generalization performance of all neural networks are to be reserved.

    bool reserve_generalization_performance_data;

    /// True if the vector parameters of the neural network presenting minimum generalization performance is to be reserved.

    bool reserve_minimal_parameters;

    /// Display messages to screen.

    bool display;

    // STOPPING CRITERIA

    /// Goal value for the generalization performance. It is used as a stopping criterion.

    double generalization_performance_goal;

    /// Maximum number of iterations to perform_inputs_selection. It is used as a stopping criterion.

    size_t maximum_iterations_number;

    /// Maximum value for the correlations.

    double maximum_correlation;

    /// Minimum value for the correlations.

    double minimum_correlation;

    /// Maximum training time. It is used as a stopping criterion.

    double maximum_time;

    /// Tolerance for the error in the trainings of the algorithm.

    double tolerance;
};
}

#endif
