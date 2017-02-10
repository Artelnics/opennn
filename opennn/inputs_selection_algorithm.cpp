/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N P U T S   S E L E C T I O N   A L G O R I T H M   C L A S S                                            */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// OpenNN includes

#include "inputs_selection_algorithm.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

InputsSelectionAlgorithm::InputsSelectionAlgorithm(void)
    : training_strategy_pointer(NULL)
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a trainig strategy object.

InputsSelectionAlgorithm::InputsSelectionAlgorithm(TrainingStrategy* new_training_strategy_pointer)
    : training_strategy_pointer(new_training_strategy_pointer)
{
    set_default();
}


// FILE CONSTRUCTOR

/// File constructor.
/*/// @param file_name Name of XML inputs selection file.*/

InputsSelectionAlgorithm::InputsSelectionAlgorithm(const std::string&)
    : training_strategy_pointer(NULL)
{
}


// XML CONSTRUCTOR

/// XML constructor.
/*/// @param inputs_selection_document Pointer to a TinyXML document containing the inputs selection algorithm data.*/

InputsSelectionAlgorithm::InputsSelectionAlgorithm(const tinyxml2::XMLDocument&)
    : training_strategy_pointer(NULL)
{
}


// DESTRUCTOR

/// Destructor.

InputsSelectionAlgorithm::~InputsSelectionAlgorithm(void)
{
}


// METHODS

// const bool& get_approximation(void) const method

/// Returns whether the problem is of function regression type.

const bool& InputsSelectionAlgorithm::get_approximation(void) const
{
    return approximation;
}

// TrainingStrategy* get_training_strategy_pointer(void) const method

/// Returns a pointer to the training strategy object.

TrainingStrategy* InputsSelectionAlgorithm::get_training_strategy_pointer(void) const
{
#ifdef __OPENNN_DEBUG__

    if(!training_strategy_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "DataSet* get_training_strategy_pointer(void) const method.\n"
               << "Training strategy pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    return(training_strategy_pointer);
}

// bool has_training_strategy(void) const method

/// Returns true if this inputs selection algorithm has a training strategy associated, and false otherwise.

bool InputsSelectionAlgorithm::has_training_strategy(void) const
{
    if(training_strategy_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}

// const size_t& get_trials_number(void) const method

/// Returns the number of trials for each network architecture.

const size_t& InputsSelectionAlgorithm::get_trials_number(void) const
{
    return(trials_number);
}

// const bool& get_reserve_parameters_data(void) const method

/// Returns true if the neural network parameters are to be reserved, and false otherwise.

const bool& InputsSelectionAlgorithm::get_reserve_parameters_data(void) const
{
    return(reserve_parameters_data);
}


// const bool& get_reserve_loss_data(void) const method

/// Returns true if the loss functional losss are to be reserved, and false otherwise.

const bool& InputsSelectionAlgorithm::get_reserve_loss_data(void) const
{
    return(reserve_loss_data);
}


// const bool& get_reserve_selection_loss_data(void) const method

/// Returns true if the selection losss are to be reserved, and false otherwise.

const bool& InputsSelectionAlgorithm::get_reserve_selection_loss_data(void) const
{
    return(reserve_selection_loss_data);
}


// const bool& get_reserve_minimal_parameters(void) const method

/// Returns true if the parameters vector of the neural network with minimum selection loss is to be reserved, and false otherwise.

const bool& InputsSelectionAlgorithm::get_reserve_minimal_parameters(void) const
{
    return(reserve_minimal_parameters);
}

// const PerformanceCalculationMethod& get_loss_calculation_method(void) const method

/// Returns the method for the calculation of the loss and the selection loss.

const InputsSelectionAlgorithm::PerformanceCalculationMethod& InputsSelectionAlgorithm::get_loss_calculation_method(void) const
{
    return(loss_calculation_method);
}

// const bool& get_display(void) const method

/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& InputsSelectionAlgorithm::get_display(void) const
{
    return(display);
}

// const double& get_selection_loss_goal(void) const method

/// Returns the goal for the selection loss in the inputs selection algorithm.

const double& InputsSelectionAlgorithm::get_selection_loss_goal(void) const
{
    return(selection_loss_goal);
}


// const size_t& get_maximum_iterations_number(void) const method

/// Returns the maximum number of iterations in the inputs selection algorithm.

const size_t& InputsSelectionAlgorithm::get_maximum_iterations_number(void) const
{
    return(maximum_iterations_number);
}


// const double& get_maximum_time(void) const method

/// Returns the maximum time in the inputs selection algorithm.

const double& InputsSelectionAlgorithm::get_maximum_time(void) const
{
    return(maximum_time);
}

// const double& get_maximum_correlation(void) const method

/// Return the maximum correlation for the algorithm.

const double& InputsSelectionAlgorithm::get_maximum_correlation(void) const
{
    return(maximum_correlation);
}

// const double& get_minimum_correlation(void) const method

/// Return the minimum correlation for the algorithm.

const double& InputsSelectionAlgorithm::get_minimum_correlation(void) const
{
    return(minimum_correlation);
}

// const double& get_tolerance(void) const method

/// Return the tolerance of error for the algorithm.

const double& InputsSelectionAlgorithm::get_tolerance(void) const
{
    return(tolerance);
}

// std::string write_loss_calculation_method(void) const method

/// Return a string with the loss calculation method of this inputs selection algorithm.

std::string InputsSelectionAlgorithm::write_loss_calculation_method(void) const
{
    switch (loss_calculation_method)
    {
    case Maximum:
    {
        return ("Maximum");
    }
    case Minimum:
    {
        return ("Minimum");
    }
    case Mean:
    {
        return ("Mean");
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "std::string write_loss_calculation_method(void) const method.\n"
               << "Unknown loss calculation method.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }
}

// void set_approximation(const bool&) method

/// Sets a new regression value.
/// If it is set to true the problem will be taken as a function regression;
/// if it is set to false the problem will be taken as a classification.
/// @param new_approximation Regression value.

void InputsSelectionAlgorithm::set_approximation(const bool& new_approximation)
{
    approximation = new_approximation;
}


// void set_training_strategy_pointer(TrainingStrategy*) method

/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void InputsSelectionAlgorithm::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;
}


// void set_default(void) method

/// Sets the members of the inputs selection object to their default values.

void InputsSelectionAlgorithm::set_default(void)
{

    // MEMBERS

    trials_number = 1;

    // inputs selection results

    reserve_parameters_data = true;
    reserve_loss_data = true;
    reserve_selection_loss_data = true;
    reserve_minimal_parameters = true;

    loss_calculation_method = Minimum;

    display = true;

    // STOPPING CRITERIA

    selection_loss_goal = 0.0;

    maximum_iterations_number = 1000;

    maximum_correlation = 1e20;
    minimum_correlation = 0;

    maximum_time = 10000.0;

    tolerance = 1.0e-3;

}


// void set_trials_number(const size_t&) method

/// Sets the number of times that each different neural network is to be trained.
/// @param new_trials_number Number of trials for each set of parameters.

void InputsSelectionAlgorithm::set_trials_number(const size_t& new_trials_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_trials_number <= 0)
    {
        std::ostringstream buffer;
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_trials_number(const size_t&) method.\n"
               << "Number of assays must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    trials_number = new_trials_number;
}

// void set_reserve_parameters_data(const bool&) method

/// Sets the reserve flag for the parameters data.
/// @param new_reserve_parameters_data Flag value.

void InputsSelectionAlgorithm::set_reserve_parameters_data(const bool& new_reserve_parameters_data)
{
    reserve_parameters_data = new_reserve_parameters_data;
}


// void set_reserve_loss_data(const bool&) method

/// Sets the reserve flag for the loss data.
/// @param new_reserve_loss_data Flag value.

void InputsSelectionAlgorithm::set_reserve_loss_data(const bool& new_reserve_loss_data)
{
    reserve_loss_data = new_reserve_loss_data;
}


// void set_reserve_selection_loss_data(const bool&) method

/// Sets the reserve flag for the selection loss data.
/// @param new_reserve_selection_loss_data Flag value.

void InputsSelectionAlgorithm::set_reserve_selection_loss_data(const bool& new_reserve_selection_loss_data)
{
    reserve_selection_loss_data = new_reserve_selection_loss_data;
}


// void set_reserve_minimal_parameters(const bool&) method

/// Sets the reserve flag for the minimal parameters.
/// @param new_reserve_minimal_parameters Flag value.

void InputsSelectionAlgorithm::set_reserve_minimal_parameters(const bool& new_reserve_minimal_parameters)
{
    reserve_minimal_parameters = new_reserve_minimal_parameters;
}

// void set_loss_calculation_method(const PerformanceCalculationMethod&) method

/// Sets a new method to calculate the loss and the selection loss.
/// @param new_loss_calculation_method Method to calculate the loss (Minimum, Maximum or Mean).

void InputsSelectionAlgorithm::set_loss_calculation_method(const InputsSelectionAlgorithm::PerformanceCalculationMethod& new_loss_calculation_method)
{
    loss_calculation_method = new_loss_calculation_method;
}

// void set_loss_calculation_method(const std::string&) method

/// Sets a new loss calculation method from a string.
/// @param new_loss_calculation_method String with the loss calculation method.

void InputsSelectionAlgorithm::set_loss_calculation_method(const std::string& new_loss_calculation_method)
{
    if(new_loss_calculation_method == "Maximum")
    {
        loss_calculation_method = Maximum;

    }
    else if(new_loss_calculation_method == "Minimum")
    {
        loss_calculation_method = Minimum;

    }
    else if(new_loss_calculation_method == "Mean")
    {
        loss_calculation_method = Mean;

    }
    else{
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_loss_calculation_method(const std::string&) method.\n"
               << "Unknown loss calculation method.\n";

        throw std::logic_error(buffer.str());

    }
}


// void set_display(const bool&) method

/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void InputsSelectionAlgorithm::set_display(const bool& new_display)
{
    display = new_display;
}

// void set_selection_loss_goal(const double&) method

/// Sets the selection loss goal for the inputs selection algorithm.
/// @param new_selection_loss_goal Goal of the selection loss.

void InputsSelectionAlgorithm::set_selection_loss_goal(const double& new_selection_loss_goal)
{
#ifdef __OPENNN_DEBUG__

    if(new_selection_loss_goal < 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_selection_loss_goal(const double&) method.\n"
               << "Selection loss goal must be greater or equal than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    selection_loss_goal = new_selection_loss_goal;
}


// void set_maximum_iterations_number(const size_t&) method

/// Sets the maximum iterations number for the inputs selection algorithm.
/// @param new_maximum_iterations_number Maximum number of iterations.

void InputsSelectionAlgorithm::set_maximum_iterations_number(const size_t& new_maximum_iterations_number)
{
    maximum_iterations_number = new_maximum_iterations_number;
}


// void set_maximum_time(const double&) method

/// Sets the maximum time for the inputs selection algorithm.
/// @param new_maximum_time Maximum time for the algorithm.

void InputsSelectionAlgorithm::set_maximum_time(const double& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_maximum_time(const double&) method.\n"
               << "Maximum time must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    maximum_time = new_maximum_time;
}

// void set_maximum_correlation(const double&) method

/// Sets the maximum value for the correlations in the inputs selection algorithm.
/// @param new_maximum_correlation Maximum value of the correlations.

void InputsSelectionAlgorithm::set_maximum_correlation(const double& new_maximum_correlation)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_correlation < 0 || new_maximum_correlation > 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_maximum_correlation(const double&) method.\n"
               << "Maximum correlation must be comprised between 0 and 1.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    maximum_correlation = new_maximum_correlation;
}

// void set_minimum_correlation(const double&) method

/// Sets the minimum value for the correlations in the inputs selection algorithm.
/// @param new_minimum_correlation Minimum value of the correlations.

void InputsSelectionAlgorithm::set_minimum_correlation(const double& new_minimum_correlation)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_correlation < 0 || new_minimum_correlation > 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_minimum_correlation(const double&) method.\n"
               << "Minimum correaltion must be comprised between 0 and 1.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    minimum_correlation = new_minimum_correlation;
}

// void set_tolerance(const double&) method

/// Set the tolerance for the errors in the trainings of the algorithm.
/// @param new_tolerance Value of the tolerance.

void InputsSelectionAlgorithm::set_tolerance(const double& new_tolerance)
{
#ifdef __OPENNN_DEBUG__

    if(new_tolerance < 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_tolerance(const double&) method.\n"
               << "Tolerance must be equal or greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    tolerance = new_tolerance;
}

// Correlation methods

// Matrix<double> calculate_logistic_correlations(void) const method

/// Returns a matrix with the logistic correlations between all inputs and target variables.
/// The number of rows is the number of inputs variables.
/// The number of columns is the number of target variables.

Matrix<double> InputsSelectionAlgorithm::calculate_logistic_correlations(void) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    std::ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to training strategy is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Loss index stuff


    if(!training_strategy_pointer->has_loss_index())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to loss functional is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    if(!training_strategy_pointer->get_loss_index_pointer()->has_data_set())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to data set is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    #endif

    // Problem stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Variables& variables = data_set_pointer->get_variables();

    const Instances& instances = data_set_pointer->get_instances();

    const size_t instances_number = instances.get_instances_number();

    const size_t inputs_number = variables.count_inputs_number();
    const size_t targets_number = variables.count_targets_number();

    const Vector<size_t> input_indices = variables.arrange_inputs_indices();
    const Vector<size_t> target_indices = variables.arrange_targets_indices();

    Matrix<double> correlations(inputs_number, targets_number, 0.0);

    srand(0);

    for(size_t i = 0; i < inputs_number; i++)
    {
        const Vector<double> inputs = data_set_pointer->get_variable(input_indices[i]);

        for(size_t j = 0; j < targets_number; j++)
        {
            const Vector<double> targets = data_set_pointer->get_variable(target_indices[j]);

            if (inputs.is_binary())
            {

                correlations(i,j) = targets.calculate_linear_correlation(inputs);

                continue;
            }

            Matrix<double> data(instances_number, 2);

            data.set_column(0, inputs);
            data.set_column(1, targets);

            DataSet data_set(data);

            const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();

            Instances* instances_pointer = data_set.get_instances_pointer();

            instances_pointer->set_training();

            NeuralNetwork neural_network(1, 1);

            neural_network.construct_scaling_layer();

            ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

            scaling_layer_pointer->set_statistics(inputs_statistics);

            scaling_layer_pointer->set_scaling_method(ScalingLayer::NoScaling);

            MultilayerPerceptron* multilayer_perceptron_pointer = neural_network.get_multilayer_perceptron_pointer();

            multilayer_perceptron_pointer->set_layer_activation_function(0, Perceptron::Logistic);

            LossIndex loss_index(&neural_network, &data_set);

            loss_index.set_error_type(LossIndex::WEIGHTED_SQUARED_ERROR);

#ifdef __OPENNN_MPI__
            neural_network.set_MPI(&neural_network);

            TrainingStrategy training_strategy(&loss_index);

            training_strategy.set_main_type(TrainingStrategy::QUASI_NEWTON_METHOD);

            QuasiNewtonMethod* quasi_newton_method = training_strategy.get_quasi_Newton_method_pointer();

            quasi_newton_method->set_minimum_parameters_increment_norm(1.0e-3);

            quasi_newton_method->set_gradient_norm_goal(1.0e-4);

            quasi_newton_method->set_minimum_loss_increase(1.0e-12);

            quasi_newton_method->set_maximum_iterations_number(100);

            training_strategy.set_display(false);

            training_strategy.perform_training();

#else
            TrainingStrategy training_strategy(&loss_index);

            training_strategy.set_main_type(TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM);

            LevenbergMarquardtAlgorithm* levenberg_marquardt_algorithm = training_strategy.get_Levenberg_Marquardt_algorithm_pointer();

            levenberg_marquardt_algorithm->set_minimum_parameters_increment_norm(1.0e-3);

            levenberg_marquardt_algorithm->set_gradient_norm_goal(1.0e-4);

            levenberg_marquardt_algorithm->set_minimum_loss_increase(1.0e-12);

            levenberg_marquardt_algorithm->set_maximum_iterations_number(100);

            training_strategy.set_display(false);

            training_strategy.perform_training();

#endif
            scaling_layer_pointer->set_scaling_method(ScalingLayer::MinimumMaximum);

            const Vector<double> outputs = neural_network.calculate_output_data(inputs.to_column_matrix()).to_vector();

            correlations(i,j) = targets.calculate_linear_correlation(outputs);

            if (display)
            {
                std::cout << "Calculating correlation: Input " << i+1 << "; Target " << j+1 << std::endl;
                std::cout << "Correlation value = " << std::abs(correlations(i,j)) << std::endl;
            }

        }
    }

    srand((unsigned)time(NULL));

    return(correlations);
}

// Vector<double> calculate_final_correlations(void) const method

/// Calculate the correlation depending on whether the problem is a linear regression or a classification.
/// Return the absolute value of the correlation.
/// If there are many targets in the data set it returns the sum of the absolute values.

Vector<double> InputsSelectionAlgorithm::calculate_final_correlations(void) const
{
    Vector<double> final_correlations;
    Matrix<double> correlations;

    DataSet* data_set = training_strategy_pointer->get_loss_index_pointer()->get_data_set_pointer();

    if(approximation)
    {
        correlations = data_set->calculate_linear_correlations();
    }
    else
    {
        correlations = calculate_logistic_correlations();
    }

    correlations = correlations.calculate_absolute_value();

    final_correlations.resize(correlations.get_rows_number());

    for(size_t i = 0; i < final_correlations.size(); i++)
    {
        for(size_t j = 0; j < correlations.get_columns_number(); j ++)
        {
            final_correlations[i] += correlations(i,j);
        }
    }

    return final_correlations;
}

// Performances calculation methods


// void set_neural_inputs(const Vector<bool>&) method

/// Sets the neural network with the number of inputs encoded in the vector.
/// This method used the grow and prune inputs methods.
/// @param inputs Vector with the inputs to be set.

void InputsSelectionAlgorithm::set_neural_inputs(const Vector<bool>& inputs)
{
    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();
    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const size_t current_inputs_number = inputs.count_occurrences(true);
    const size_t neural_network_inputs_number = neural_network_pointer->get_inputs_number();

    if(current_inputs_number < neural_network_inputs_number)
    {
        for (size_t j = current_inputs_number; j < neural_network_inputs_number; j++)
        {
            neural_network_pointer->prune_input(0);
        }
    }
    else
    {
        for (size_t j = neural_network_inputs_number; j < current_inputs_number; j++)
        {
            neural_network_pointer->grow_input();
        }
    }

    neural_network_pointer->perturbate_parameters(0.001);
#ifdef __OPENNN_MPI__
    neural_network_pointer->set_MPI(neural_network_pointer);
#endif
}


// Vector<double> perform_minimum_model_evaluation(const Vector<bool>&) const method

/// Returns the minimum of the loss and selection loss in trials_number trainings.
/// @param inputs Vector of the inputs to be trained with.

Vector<double> InputsSelectionAlgorithm::perform_minimum_model_evaluation(const Vector<bool>& inputs)
{
#ifdef __OPENNN_DEBUG__

    if(inputs.count_occurrences(true) <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of inputs must be greater or equal than 1.\n";

        throw std::logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    NeuralNetwork* neural_network = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();

    TrainingStrategy::Results training_strategy_results;

    Vector<double> final(2);
    final[0] = 10;
    final[1] = 10;

    Vector<double> current_loss(2);

    Vector<double> final_parameters;

    bool flag_loss = false;
    bool flag_selection = false;

    for (size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            final[0] = loss_history[i];
            flag_loss = true;
        }
    }

    for (size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            final[1] = selection_loss_history[i];
            flag_selection = true;
        }
    }

    if(flag_loss && flag_selection)
    {
        return(final);
    }

    neural_network->perturbate_parameters(0.001);
#ifdef __OPENNN_MPI__

        neural_network->set_MPI(neural_network);

#endif
    training_strategy_results = training_strategy_pointer->perform_training();

    current_loss = get_final_losss(training_strategy_results);

    final[0] = current_loss[0];
    final[1] = current_loss[1];

    final_parameters.set(neural_network->arrange_parameters());

    for (size_t i = 1; i < trials_number; i++)
    {
        if(display)
        {
            std::cout << "Trial number: " << i << std::endl;
            std::cout << "Training loss: " << final[0] << std::endl;
            std::cout << "Selection loss: " << final[1] << std::endl;
        }

        neural_network->randomize_parameters_normal();

#ifdef __OPENNN_MPI__
        neural_network->set_MPI(neural_network);
#endif

        training_strategy_results = training_strategy_pointer->perform_training();

        current_loss = get_final_losss(training_strategy_results);

        if(!flag_loss && final[0] > current_loss[0])
        {
            final[0] = current_loss[0];

            final_parameters.set(neural_network->arrange_parameters());
        }

        if(!flag_selection && final[1] > current_loss[1])
        {
            final[1] = current_loss[1];

            final_parameters.set(neural_network->arrange_parameters());
        }

        if(i == trials_number - 1 && display)
        {
            std::cout << "Trial number: " << trials_number << std::endl;
            std::cout << "Training loss: " << final[0] << std::endl;
            std::cout << "Selection loss: " << final[1] << std::endl;
        }
    }

    inputs_history.push_back(inputs);

    loss_history.push_back(final[0]);

    selection_loss_history.push_back(final[1]);

    parameters_history.push_back(final_parameters);

    return final;
}


// Vector<double> perform_maximum_model_evaluation(const Vector<bool>&) method

/// Returns the maximum of the loss and selection loss in trials_number trainings.
/// @param inputs Vector of the inputs to be trained with.

Vector<double> InputsSelectionAlgorithm::perform_maximum_model_evaluation(const Vector<bool>& inputs)
{
#ifdef __OPENNN_DEBUG__

    if(inputs.count_occurrences(true) <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of inputs must be greater or equal than 1.\n";

        throw std::logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    NeuralNetwork* neural_network = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();

    TrainingStrategy::Results training_strategy_results;

    Vector<double> final(2);
    final[0] = 0;
    final[1] = 0;

    Vector<double> current_loss(2);

    Vector<double> final_parameters;

    bool flag_loss = false;
    bool flag_selection = false;

    for (size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            final[0] = loss_history[i];
            flag_loss = true;
        }
    }

    for (size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            final[1] = selection_loss_history[i];
            flag_selection = true;
        }
    }

    if(flag_loss && flag_selection)
    {
        return(final);
    }

    neural_network->perturbate_parameters(0.001);
#ifdef __OPENNN_MPI__

        neural_network->set_MPI(neural_network);

#endif
    training_strategy_results = training_strategy_pointer->perform_training();

    current_loss = get_final_losss(training_strategy_results);

    final[0] = current_loss[0];
    final[1] = current_loss[1];

    final_parameters.set(neural_network->arrange_parameters());

    for (size_t i = 1; i < trials_number; i++)
    {
        if(display)
        {
            std::cout << "Trial number: " << i << std::endl;
            std::cout << "Training loss: " << final[0] << std::endl;
            std::cout << "Selection loss: " << final[1] << std::endl;
        }

        neural_network->randomize_parameters_normal();

        training_strategy_results = training_strategy_pointer->perform_training();

        current_loss = get_final_losss(training_strategy_results);

        if(!flag_loss && final[0] < current_loss[0])
        {
            final[0] = current_loss[0];

            final_parameters.set(neural_network->arrange_parameters());
        }

        if(!flag_selection && final[1] < current_loss[1])
        {
            final[1] = current_loss[1];

            final_parameters.set(neural_network->arrange_parameters());
        }

        if(i == trials_number - 1 && display)
        {
            std::cout << "Trial number: " << trials_number << std::endl;
            std::cout << "Training loss: " << final[0] << std::endl;
            std::cout << "Selection loss: " << final[1] << std::endl;
        }
    }

    inputs_history.push_back(inputs);

    loss_history.push_back(final[0]);

    selection_loss_history.push_back(final[1]);

    parameters_history.push_back(final_parameters);

    return final;
}


// Vector<double> perform_mean_model_evaluation(const Vector<bool>&) method

/// Returns the mean of the loss and selection loss in trials_number trainings.
/// @param inputs Vector of the inputs to be trained with.

Vector<double> InputsSelectionAlgorithm::perform_mean_model_evaluation(const Vector<bool> &inputs)
{
#ifdef __OPENNN_DEBUG__

    if(inputs.count_occurrences(true) <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of inputs must be greater or equal than 1.\n";

        throw std::logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    NeuralNetwork* neural_network = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();

    TrainingStrategy::Results training_strategy_results;

    Vector<double> mean_final(2);
    mean_final[0] = 0;
    mean_final[1] = 0;

    Vector<double> current_loss(2);

    Vector<double> final_parameters;

    bool flag_loss = false;
    bool flag_selection = false;

    for (size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            mean_final[0] = loss_history[i];
            flag_loss = true;
        }
    }

    for (size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            mean_final[1] = selection_loss_history[i];
            flag_selection = true;
        }
    }

    if(flag_loss && flag_selection)
    {
        return(mean_final);
    }

    neural_network->perturbate_parameters(0.001);
#ifdef __OPENNN_MPI__

        neural_network->set_MPI(neural_network);

#endif
    training_strategy_results = training_strategy_pointer->perform_training();

    current_loss = get_final_losss(training_strategy_results);

    mean_final[0] = current_loss[0];
    mean_final[1] = current_loss[1];

    final_parameters.set(neural_network->arrange_parameters());

    for (size_t i = 1; i < trials_number; i++)
    {
        if(display)
        {
            std::cout << "Trial number: " << i << std::endl;
            std::cout << "Training loss: " << mean_final[0] << std::endl;
            std::cout << "Selection loss: " << mean_final[1] << std::endl;
        }

        neural_network->randomize_parameters_normal();

        training_strategy_results = training_strategy_pointer->perform_training();

        current_loss = get_final_losss(training_strategy_results);

        if(!flag_loss)
        {
            mean_final[0] += current_loss[0]/trials_number;
        }

        if(!flag_selection)
        {
            mean_final[1] += current_loss[1]/trials_number;
        }

        if(i == trials_number - 1 && display)
        {
            std::cout << "Trial number: " << trials_number << std::endl;
            std::cout << "Training loss: " << mean_final[0] << std::endl;
            std::cout << "Selection loss: " << mean_final[1] << std::endl;
        }
    }

    inputs_history.push_back(inputs);

    loss_history.push_back(mean_final[0]);

    selection_loss_history.push_back(mean_final[1]);

    parameters_history.push_back(final_parameters);

    return mean_final;
}

// Vector<double> get_final_losss(const TrainingStrategy::Results&) const method

/// Return final training loss and final selection loss depending on the training method.
/// @param results Results of the perform_training method.

Vector<double> InputsSelectionAlgorithm::get_final_losss(const TrainingStrategy::Results& results) const
{
    Vector<double> losss(2);
    switch(training_strategy_pointer->get_main_type())
    {
    case TrainingStrategy::NO_MAIN:
    {
        losss[0] = 0;
        losss[1] = 0;
        break;
    }
    case TrainingStrategy::GRADIENT_DESCENT:
    {
        losss[0] = results.gradient_descent_results_pointer->final_loss;
        losss[1] = results.gradient_descent_results_pointer->final_selection_loss;
        break;
    }
    case TrainingStrategy::CONJUGATE_GRADIENT:
    {
        losss[0] = results.conjugate_gradient_results_pointer->final_loss;
        losss[1] = results.conjugate_gradient_results_pointer->final_selection_loss;
        break;
    }
    case TrainingStrategy::QUASI_NEWTON_METHOD:
    {
        losss[0] = results.quasi_Newton_method_results_pointer->final_loss;
        losss[1] = results.quasi_Newton_method_results_pointer->final_selection_loss;
        break;
    }
    case TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM:
    {
        losss[0] = results.Levenberg_Marquardt_algorithm_results_pointer->final_loss;
        losss[1] = results.Levenberg_Marquardt_algorithm_results_pointer->final_selection_loss;
        break;
    }
    case TrainingStrategy::USER_MAIN:
    {
        losss[0] = 0;
        losss[1] = 0;
        break;
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> get_final_losss(const TrainingStrategy::Results) method.\n"
               << "Unknown main type method.\n";

        throw std::logic_error(buffer.str());
    }
    }

    return(losss);
}

// Vector<double> perform_model_evaluation(const Vector<bool>&) method

/// Return loss and selection depending on the loss calculation method.
/// @param inputs Vector of inputs to be trained with.

Vector<double> InputsSelectionAlgorithm::perform_model_evaluation(const Vector<bool>& inputs)
{
    set_neural_inputs(inputs);

    switch (loss_calculation_method)
    {
    case Maximum:
    {
        return(perform_maximum_model_evaluation(inputs));
    }
    case Minimum:
    {
        return(perform_minimum_model_evaluation(inputs));
    }
    case Mean:
    {
        return(perform_mean_model_evaluation(inputs));
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_model_evaluation(const size_t) method.\n"
               << "Unknown loss calculation method.\n";

        throw std::logic_error(buffer.str());
    }
    }
}


// Vector<double> get_parameters_inputs(const Vector<bool>&) method

/// Returns the parameters of the neural network if the inputs is in the history.
/// @param inputs Vector of inputs to be trained with.

Vector<double> InputsSelectionAlgorithm::get_parameters_inputs(const Vector<bool>& inputs) const
{
#ifdef __OPENNN_DEBUG__

    if(inputs.count_occurrences(true) <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> get_parameters_inputs(const Vector<bool>&) method.\n"
               << "Inputs must be greater than 1.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    size_t i;

    Vector<double> parameters;

    for (i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            parameters = parameters_history[i];

            break;
        }
    }

    return(parameters);

}


// void delete_selection_history(void) method

/// Delete the history of the selection loss values.

void InputsSelectionAlgorithm::delete_selection_history(void)
{
    selection_loss_history.set();
}


// void delete_loss_history(void) method

/// Delete the history of the loss values.

void InputsSelectionAlgorithm::delete_loss_history(void)
{
    loss_history.set();
}

// void delete_parameters_history(void) method

/// Delete the history of the parameters of the trained neural networks.

void InputsSelectionAlgorithm::delete_parameters_history(void)
{
    parameters_history.set();
}

// void check(void) const method

/// Checks that the different pointers needed for performing the inputs selection are not NULL.

void InputsSelectionAlgorithm::check(void) const
{
    // Training algorithm stuff

    std::ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to training strategy is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Loss index stuff


    if(!training_strategy_pointer->has_loss_index())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to loss functional is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Neural network stuff

    if(!loss_index_pointer->has_neural_network())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to neural network is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer->has_multilayer_perceptron())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to multilayer perceptron is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();


    if(multilayer_perceptron_pointer->is_empty())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Multilayer Perceptron is empty.\n";

        throw std::logic_error(buffer.str());
    }

    /*
   if(multilayer_perceptron_pointer->get_layers_number() != 2)
   {
      buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
             << "void check(void) const method.\n"
             << "Number of layers in multilayer perceptron (" << multilayer_perceptron_pointer->get_layers_number() << ") must be 2.\n";

      throw std::logic_error(buffer.str());
   }*/


    // Data set stuff


    if(!loss_index_pointer->has_data_set())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to data set is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Instances& instances = data_set_pointer->get_instances();

    const size_t selection_instances_number = instances.count_selection_instances_number();

    if(selection_instances_number == 0)
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Number of selection instances is zero.\n";

        throw std::logic_error(buffer.str());
    }

}
/*
// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes the input selection object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document.

tinyxml2::XMLDocument* InputsSelectionAlgorithm::to_XML(void) const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    std::ostringstream buffer;

    // Input input selection

    tinyxml2::XMLElement* inputs_selection_element = document->NewElement("InputsSelectionAlgorithm");

    document->InsertFirstChild(inputs_selection_element);

    // Hidden layer sizes

    tinyxml2::XMLElement* input_numbers_element = document->NewElement("InputsNumbers");
    inputs_selection_element->LinkEndChild(input_numbers_element);

    tinyxml2::XMLElement* maximum_input_element = document->NewElement("MaximumInputNumber");
    input_numbers_element->LinkEndChild(maximum_input_element);

    buffer.str("");
    buffer << maximum_input;

    tinyxml2::XMLText* maximum_input_text = document->NewText(buffer.str().c_str());
    maximum_input_element->LinkEndChild(maximum_input_text);

    tinyxml2::XMLElement* minimum_input_element = document->NewElement("MinimumInputNumber");
    input_numbers_element->LinkEndChild(minimum_input_element);

    buffer.str("");
    buffer << minimum_input;

    tinyxml2::XMLText* minimum_input_text = document->NewText(buffer.str().c_str());
    minimum_input_element->LinkEndChild(minimum_input_text);


    // ParametersAssaysNumber

    tinyxml2::XMLElement* trials_number_element = document->NewElement("ParametersAssaysNumber");
    inputs_selection_element->LinkEndChild(trials_number_element);

    buffer.str("");
    buffer << trials_number;

    tinyxml2::XMLText* trials_number_text = document->NewText(buffer.str().c_str());
    trials_number_element->LinkEndChild(trials_number_text);


    // Display

    tinyxml2::XMLElement* display_element = document->NewElement("Display");
    inputs_selection_element->LinkEndChild(display_element);

    buffer.str("");
    buffer << display;

    tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
    display_element->LinkEndChild(display_text);


    return(document);
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// @todo

void InputsSelectionAlgorithm::from_XML(const tinyxml2::XMLDocument&)
{
}


// void print(void) method

/// Prints to the screen the XML representation of this input selection object.

void InputsSelectionAlgorithm::print(void) const
{
    std::cout << to_XML();
}


// void save(const std::string&) const method

/// Saves the input selection members to a XML file.
/// @param file_name Name of input selection XML file.

void InputsSelectionAlgorithm::save(const std::string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
}


// void load(const std::string&) method

/// Loads the input selection members from a XML file.
/// @param file_name Name of input selection XML file.

void InputsSelectionAlgorithm::load(const std::string& file_name)
{
    std::ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    if(document->LoadFile(file_name.c_str()))
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void load(const std::string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw std::logic_error(buffer.str());
    }

    // Root

    tinyxml2::XMLElement* inputs_selection_element = document->FirstChildElement("InputsSelectionAlgorithm");

    if(!inputs_selection_element)
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void load(const std::string&) method.\n"
               << "Invalid input input selection XML root element.\n";

        throw std::logic_error(buffer.str());
    }

    // Hidden layer sizes


    // Parameters assays number

    tinyxml2::XMLElement* trials_number_element = inputs_selection_element->FirstChildElement("ParametersAssaysNumber");

    if(trials_number_element)
    {
        trials_number = atoi(trials_number_element->GetText());
    }
}
*/

// std::string write_stopping_condition(void) const method

/// Return a string with the stopping condition of the InputsSelectionResults

std::string InputsSelectionAlgorithm::InputsSelectionResults::write_stopping_condition(void) const
{
    switch (stopping_condition)
    {
    case MaximumTime:
    {
        return ("MaximumTime");
    }
    case SelectionLossGoal:
    {
        return("SelectionLossGoal");
    }
    case MaximumInputs:
    {
        return("MaximumInputs");
    }
    case MinimumInputs:
    {
        return("MinimumInputs");
    }
    case MaximumIterations:
    {
        return("MaximumIterations");
    }
    case MaximumSelectionFailures:
    {
        return("MaximumSelectionFailures");
    }
    case CorrelationGoal:
    {
        return("CorrelationGoal");
    }
    case AlgorithmFinished:
    {
        return("AlgorithmFinished");
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionResults struct.\n"
               << "std::string write_stopping_condition(void) const method.\n"
               << "Unknown stopping condition type.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }

}


// std::string to_string(void) const method

/// Returns a string representation of the current inputs selection results structure.

std::string InputsSelectionAlgorithm::InputsSelectionResults::to_string(void) const
{
   std::ostringstream buffer;

   // Inputs history

   if(!inputs_data.empty())
   {
     buffer << "% Inputs history:\n"
            << inputs_data.to_row_matrix() << "\n";
   }


   // Parameters history

   if(!parameters_data.empty())
   {
     buffer << "% Parameters history:\n"
            << parameters_data.to_row_matrix() << "\n";
   }

   // Performance history

   if(!loss_data.empty())
   {
       buffer << "% Performance history:\n"
              << loss_data.to_row_matrix() << "\n";
   }

   // Selection loss history

   if(!selection_loss_data.empty())
   {
       buffer << "% Selection loss history:\n"
              << selection_loss_data.to_row_matrix() << "\n";
   }

   // Minimal parameters

   if(!minimal_parameters.empty())
   {
       buffer << "% Minimal parameters:\n"
              << minimal_parameters << "\n";
   }

   // Stopping condition

   buffer << "% Stopping condition\n"
          << write_stopping_condition() << "\n";

   // Optimum selection loss

   if(final_selection_loss != 0)
   {
       buffer << "% Optimum selection loss:\n"
              << final_selection_loss << "\n";
   }

   // Final loss

   if(final_loss != 0)
   {
       buffer << "% Final loss:\n"
              << final_loss << "\n";
   }

   // Optimal input

   if(!optimal_inputs.empty())
   {
       buffer << "% Optimal input:\n"
              << optimal_inputs << "\n";
   }

   // Iterations number


   buffer << "% Number of iterations:\n"
          << iterations_number << "\n";


   // Elapsed time

   buffer << "% Elapsed time:\n"
          << elapsed_time << "\n";



   return(buffer.str());
}

// size_t get_input_index(const Vector<Variables::Use>, const size_t) method

/// Return the index of uses where is the (input_number)-th input.
/// @param uses vector of the uses of the variables.
/// @param input_number index of the input to find.

size_t InputsSelectionAlgorithm::get_input_index(const Vector<Variables::Use> uses, const size_t input_number)
{
#ifdef __OPENNN_DEBUG__

    if(uses.size() < input_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "const size_t get_input_index(const Vector<Variables::Use>, const size_t) method.\n"
               << "Size of uses vector must be greater than " <<  input_number << ".\n";

        throw std::logic_error(buffer.str());
    }
#endif

    size_t i = 0;
    size_t j = 0;
    while(i < uses.size())
    {
        if(uses[i] == Variables::Input &&
            input_number == j)
        {
            break;
        }
        else if(uses[i] == Variables::Input)
        {
            i++;
            j++;
        }
        else
        {
            i++;
        }
    }
    return i;
}

}



