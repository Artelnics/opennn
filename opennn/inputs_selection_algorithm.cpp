/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N P U T S   S E L E C T I O N   A L G O R I T H M   C L A S S                                            */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// OpenNN includes

#include "inputs_selection_algorithm.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

InputsSelectionAlgorithm::InputsSelectionAlgorithm()
    : training_strategy_pointer(nullptr)
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

InputsSelectionAlgorithm::InputsSelectionAlgorithm(const string&)
    : training_strategy_pointer(nullptr)
{
}


// XML CONSTRUCTOR

/// XML constructor.
/*/// @param inputs_selection_document Pointer to a TinyXML document containing the inputs selection algorithm data.*/

InputsSelectionAlgorithm::InputsSelectionAlgorithm(const tinyxml2::XMLDocument&)
    : training_strategy_pointer(nullptr)
{
}


// DESTRUCTOR

/// Destructor.

InputsSelectionAlgorithm::~InputsSelectionAlgorithm()
{
}


// METHODS


/// Returns whether the problem is of function regression type.

const bool& InputsSelectionAlgorithm::get_approximation() const
{
    return approximation;
}


/// Returns a pointer to the training strategy object.

TrainingStrategy* InputsSelectionAlgorithm::get_training_strategy_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!training_strategy_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "DataSet* get_training_strategy_pointer() const method.\n"
               << "Training strategy pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(training_strategy_pointer);
}


/// Returns true if this inputs selection algorithm has a training strategy associated, and false otherwise.

bool InputsSelectionAlgorithm::has_training_strategy() const
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


/// Returns the number of trials for each network architecture.

const size_t& InputsSelectionAlgorithm::get_trials_number() const
{
    return(trials_number);
}


/// Returns true if the neural network parameters are to be reserved, and false otherwise.

const bool& InputsSelectionAlgorithm::get_reserve_parameters_data() const
{
    return(reserve_parameters_data);
}


/// Returns true if the loss index losses are to be reserved, and false otherwise.

const bool& InputsSelectionAlgorithm::get_reserve_error_data() const
{
    return(reserve_error_data);
}


/// Returns true if the selection losses are to be reserved, and false otherwise.

const bool& InputsSelectionAlgorithm::get_reserve_selection_error_data() const
{
    return(reserve_selection_error_data);
}


/// Returns true if the parameters vector of the neural network with minimum selection error is to be reserved, and false otherwise.

const bool& InputsSelectionAlgorithm::get_reserve_minimal_parameters() const
{
    return(reserve_minimal_parameters);
}


/// Returns the method for the calculation of the loss and the selection error.

const InputsSelectionAlgorithm::LossCalculationMethod& InputsSelectionAlgorithm::get_loss_calculation_method() const
{
    return(loss_calculation_method);
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& InputsSelectionAlgorithm::get_display() const
{
    return(display);
}


/// Returns the goal for the selection error in the inputs selection algorithm.

const double& InputsSelectionAlgorithm::get_selection_error_goal() const
{
    return(selection_error_goal);
}


/// Returns the maximum number of iterations in the inputs selection algorithm.

const size_t& InputsSelectionAlgorithm::get_maximum_iterations_number() const
{
    return(maximum_iterations_number);
}


/// Returns the maximum time in the inputs selection algorithm.

const double& InputsSelectionAlgorithm::get_maximum_time() const
{
    return(maximum_time);
}


/// Return the maximum correlation for the algorithm.

const double& InputsSelectionAlgorithm::get_maximum_correlation() const
{
    return(maximum_correlation);
}


/// Return the minimum correlation for the algorithm.

const double& InputsSelectionAlgorithm::get_minimum_correlation() const
{
    return(minimum_correlation);
}


/// Return the tolerance of error for the algorithm.

const double& InputsSelectionAlgorithm::get_tolerance() const
{
    return(tolerance);
}


/// Return a string with the loss calculation method of this inputs selection algorithm.

string InputsSelectionAlgorithm::write_loss_calculation_method() const
{
    switch(loss_calculation_method)
    {
    case Maximum:
    {
        return("Maximum");
    }
    case Minimum:
    {
        return("Minimum");
    }
    case Mean:
    {
        return("Mean");
    }
    }

    return string();
}


/// Sets a new regression value.
/// If it is set to true the problem will be taken as a function regression;
/// if it is set to false the problem will be taken as a classification.
/// @param new_approximation Regression value.

void InputsSelectionAlgorithm::set_approximation(const bool& new_approximation)
{
    approximation = new_approximation;
}


/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void InputsSelectionAlgorithm::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;
}


/// Sets the members of the inputs selection object to their default values.

void InputsSelectionAlgorithm::set_default()
{
    // MEMBERS

    trials_number = 1;

    // inputs selection results

    reserve_parameters_data = true;
    reserve_error_data = true;
    reserve_selection_error_data = true;
    reserve_minimal_parameters = true;

    loss_calculation_method = Minimum;

    display = true;

    // STOPPING CRITERIA

    selection_error_goal = 0.0;

    maximum_iterations_number = 1000;

    maximum_correlation = 1.0;
    minimum_correlation = 0.0;

    maximum_time = 10000.0;

    tolerance = 0.0;
}


/// Sets the number of times that each different neural network is to be trained.
/// @param new_trials_number Number of trials for each set of parameters.

void InputsSelectionAlgorithm::set_trials_number(const size_t& new_trials_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_trials_number <= 0)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_trials_number(const size_t&) method.\n"
               << "Number of assays must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    trials_number = new_trials_number;
}


/// Sets the reserve flag for the parameters data.
/// @param new_reserve_parameters_data Flag value.

void InputsSelectionAlgorithm::set_reserve_parameters_data(const bool& new_reserve_parameters_data)
{
    reserve_parameters_data = new_reserve_parameters_data;
}


/// Sets the reserve flag for the loss data.
/// @param new_reserve_error_data Flag value.

void InputsSelectionAlgorithm::set_reserve_error_data(const bool& new_reserve_error_data)
{
    reserve_error_data = new_reserve_error_data;
}


/// Sets the reserve flag for the selection error data.
/// @param new_reserve_selection_error_data Flag value.

void InputsSelectionAlgorithm::set_reserve_selection_error_data(const bool& new_reserve_selection_error_data)
{
    reserve_selection_error_data = new_reserve_selection_error_data;
}


/// Sets the reserve flag for the minimal parameters.
/// @param new_reserve_minimal_parameters Flag value.

void InputsSelectionAlgorithm::set_reserve_minimal_parameters(const bool& new_reserve_minimal_parameters)
{
    reserve_minimal_parameters = new_reserve_minimal_parameters;
}


/// Sets a new method to calculate the loss and the selection error.
/// @param new_loss_calculation_method Method to calculate the loss(Minimum, Maximum or Mean).

void InputsSelectionAlgorithm::set_loss_calculation_method(const InputsSelectionAlgorithm::LossCalculationMethod& new_loss_calculation_method)
{
    loss_calculation_method = new_loss_calculation_method;
}


/// Sets a new loss calculation method from a string.
/// @param new_loss_calculation_method String with the loss calculation method.

void InputsSelectionAlgorithm::set_loss_calculation_method(const string& new_loss_calculation_method)
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
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_loss_calculation_method(const string&) method.\n"
               << "Unknown loss calculation method.\n";

        throw logic_error(buffer.str());

    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void InputsSelectionAlgorithm::set_display(const bool& new_display)
{
    display = new_display;
}


/// Sets the selection error goal for the inputs selection algorithm.
/// @param new_selection_error_goal Goal of the selection error.

void InputsSelectionAlgorithm::set_selection_error_goal(const double& new_selection_error_goal)
{
#ifdef __OPENNN_DEBUG__

    if(new_selection_error_goal < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_selection_error_goal(const double&) method.\n"
               << "Selection loss goal must be greater or equal than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    selection_error_goal = new_selection_error_goal;
}


/// Sets the maximum iterations number for the inputs selection algorithm.
/// @param new_maximum_iterations_number Maximum number of iterations.

void InputsSelectionAlgorithm::set_maximum_iterations_number(const size_t& new_maximum_iterations_number)
{
    maximum_iterations_number = new_maximum_iterations_number;
}


/// Sets the maximum time for the inputs selection algorithm.
/// @param new_maximum_time Maximum time for the algorithm.

void InputsSelectionAlgorithm::set_maximum_time(const double& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_maximum_time(const double&) method.\n"
               << "Maximum time must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_time = new_maximum_time;
}


/// Sets the maximum value for the correlations in the inputs selection algorithm.
/// @param new_maximum_correlation Maximum value of the correlations.

void InputsSelectionAlgorithm::set_maximum_correlation(const double& new_maximum_correlation)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_correlation < 0 || new_maximum_correlation > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_maximum_correlation(const double&) method.\n"
               << "Maximum correlation must be comprised between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_correlation = new_maximum_correlation;
}


/// Sets the minimum value for the correlations in the inputs selection algorithm.
/// @param new_minimum_correlation Minimum value of the correlations.

void InputsSelectionAlgorithm::set_minimum_correlation(const double& new_minimum_correlation)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_correlation < 0 || new_minimum_correlation > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_minimum_correlation(const double&) method.\n"
               << "Minimum correaltion must be comprised between 0 and 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_correlation = new_minimum_correlation;
}


/// Set the tolerance for the errors in the trainings of the algorithm.
/// @param new_tolerance Value of the tolerance.

void InputsSelectionAlgorithm::set_tolerance(const double& new_tolerance)
{
#ifdef __OPENNN_DEBUG__

    if(new_tolerance < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void set_tolerance(const double&) method.\n"
               << "Tolerance must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    tolerance = new_tolerance;
}


/// Returns the minimum of the loss and selection error in trials_number trainings.
/// @param inputs Vector of the inputs to be trained with.

Vector<double> InputsSelectionAlgorithm::perform_minimum_model_evaluation(const Vector<bool>& inputs)
{
#ifdef __OPENNN_DEBUG__

    if(inputs.count_equal_to(true) <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of inputs must be greater or equal than 1.\n";

        throw logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw logic_error(buffer.str());
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

    for(size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            final[0] = loss_history[i];
            flag_loss = true;
        }
    }

    for(size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            final[1] = selection_error_history[i];
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

    current_loss = get_final_losses(training_strategy_results);

    final[0] = current_loss[0];
    final[1] = current_loss[1];

    final_parameters.set(neural_network->get_parameters());

    for(size_t i = 1; i < trials_number; i++)
    {
        if(display)
        {
            cout << "Trial number: " << i << endl;
            if(i == 1)
            {
                cout << "Training loss: " << final[0] << endl;
                cout << "Selection error: " << final[1] << endl;
                cout << "Stopping condition: " << write_stopping_condition(training_strategy_results) << endl << endl;
            }
            else
            {
                cout << "Training loss: " << current_loss[0] << endl;
                cout << "Selection error: " << current_loss[1] << endl;
                cout << "Stopping condition: " << write_stopping_condition(training_strategy_results) << endl << endl;
            }
        }

        neural_network->randomize_parameters_normal();

#ifdef __OPENNN_MPI__
        neural_network->set_MPI(neural_network);
#endif

        training_strategy_results = training_strategy_pointer->perform_training();

        current_loss = get_final_losses(training_strategy_results);

        if(!flag_loss && final[0] > current_loss[0])
        {
            final[0] = current_loss[0];

            final_parameters.set(neural_network->get_parameters());
        }

        if(!flag_selection && final[1] > current_loss[1])
        {
            final[1] = current_loss[1];

            final_parameters.set(neural_network->get_parameters());
        }
    }

    if(display)
    {
        cout << "Trial number: " << trials_number << endl;
        cout << "Training loss: " << final[0] << endl;
        cout << "Selection error: " << final[1] << endl;
        cout << "Stopping condition: " << write_stopping_condition(training_strategy_results) << endl << endl;
    }

    inputs_history.push_back(inputs);

    loss_history.push_back(final[0]);

    selection_error_history.push_back(final[1]);

    parameters_history.push_back(final_parameters);

    return final;
}


/// Returns the maximum of the loss and selection error in trials_number trainings.
/// @param inputs Vector of the inputs to be trained with.

Vector<double> InputsSelectionAlgorithm::perform_maximum_model_evaluation(const Vector<bool>& inputs)
{
#ifdef __OPENNN_DEBUG__

    if(inputs.count_equal_to(true) <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of inputs must be greater or equal than 1.\n";

        throw logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw logic_error(buffer.str());
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

    for(size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            final[0] = loss_history[i];
            flag_loss = true;
        }
    }

    for(size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            final[1] = selection_error_history[i];
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

    current_loss = get_final_losses(training_strategy_results);

    final[0] = current_loss[0];
    final[1] = current_loss[1];

    final_parameters.set(neural_network->get_parameters());

    for(size_t i = 1; i < trials_number; i++)
    {
        if(display)
        {
            cout << "Trial number: " << i << endl;
            if(i == 1)
            {
                cout << "Training loss: " << final[0] << endl;
                cout << "Selection error: " << final[1] << endl;
                cout << "Stopping condition: " << write_stopping_condition(training_strategy_results) << endl << endl;
            }
            else
            {
                cout << "Training loss: " << current_loss[0] << endl;
                cout << "Selection error: " << current_loss[1] << endl;
                cout << "Stopping condition: " << write_stopping_condition(training_strategy_results) << endl << endl;
            }
        }

        neural_network->randomize_parameters_normal();

#ifdef __OPENNN_MPI__

        neural_network->set_MPI(neural_network);

#endif
        training_strategy_results = training_strategy_pointer->perform_training();

        current_loss = get_final_losses(training_strategy_results);

        if(!flag_loss && final[0] < current_loss[0])
        {
            final[0] = current_loss[0];

            final_parameters.set(neural_network->get_parameters());
        }

        if(!flag_selection && final[1] < current_loss[1])
        {
            final[1] = current_loss[1];

            final_parameters.set(neural_network->get_parameters());
        }
    }

    if(display)
    {
        cout << "Trial number: " << trials_number << endl;
        cout << "Training loss: " << final[0] << endl;
        cout << "Selection error: " << final[1] << endl;
        cout << "Stopping condition: " << write_stopping_condition(training_strategy_results) << endl << endl;
    }

    inputs_history.push_back(inputs);

    loss_history.push_back(final[0]);

    selection_error_history.push_back(final[1]);

    parameters_history.push_back(final_parameters);

    return final;
}


/// Returns the mean of the loss and selection error in trials_number trainings.
/// @param inputs Vector of the inputs to be trained with.

Vector<double> InputsSelectionAlgorithm::perform_mean_model_evaluation(const Vector<bool> &inputs)
{
#ifdef __OPENNN_DEBUG__

    if(inputs.count_equal_to(true) <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of inputs must be greater or equal than 1.\n";

        throw logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw logic_error(buffer.str());
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

    for(size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            mean_final[0] = loss_history[i];
            flag_loss = true;
        }
    }

    for(size_t i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            mean_final[1] = selection_error_history[i];
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

    current_loss = get_final_losses(training_strategy_results);

    mean_final[0] = current_loss[0];
    mean_final[1] = current_loss[1];

    final_parameters.set(neural_network->get_parameters());

    for(size_t i = 1; i < trials_number; i++)
    {
        if(display)
        {
            cout << "Trial number: " << i << endl;
            if(i == 1)
            {
                cout << "Training loss: " << mean_final[0] << endl;
                cout << "Selection error: " << mean_final[1] << endl;
                cout << "Stopping condition: " << write_stopping_condition(training_strategy_results) << endl << endl;
            }
            else
            {
                cout << "Training loss: " << current_loss[0] << endl;
                cout << "Selection error: " << current_loss[1] << endl;
                cout << "Stopping condition: " << write_stopping_condition(training_strategy_results) << endl << endl;
            }
        }

        neural_network->randomize_parameters_normal();

        training_strategy_results = training_strategy_pointer->perform_training();

        current_loss = get_final_losses(training_strategy_results);

        if(!flag_loss)
        {
            mean_final[0] += current_loss[0]/trials_number;
        }

        if(!flag_selection)
        {
            mean_final[1] += current_loss[1]/trials_number;
        }
    }

    if(display)
    {
        cout << "Trial number: " << trials_number << endl;
        cout << "Training loss: " << mean_final[0] << endl;
        cout << "Selection error: " << mean_final[1] << endl;
        cout << "Stopping condition: " << write_stopping_condition(training_strategy_results) << endl << endl;
    }

    inputs_history.push_back(inputs);

    loss_history.push_back(mean_final[0]);

    selection_error_history.push_back(mean_final[1]);

    parameters_history.push_back(final_parameters);

    return mean_final;
}


/// Return final training loss and final selection error depending on the training method.
/// @param results Results of the perform_training method.

Vector<double> InputsSelectionAlgorithm::get_final_losses(const TrainingStrategy::Results& results) const
{
    Vector<double> losses(2);

    switch(training_strategy_pointer->get_training_method())
    {
        case TrainingStrategy::GRADIENT_DESCENT:
        {
            losses[0] = results.gradient_descent_results_pointer->final_loss;
            losses[1] = results.gradient_descent_results_pointer->final_selection_error;

            return(losses);
        }
        case TrainingStrategy::CONJUGATE_GRADIENT:
        {
            losses[0] = results.conjugate_gradient_results_pointer->final_loss;
            losses[1] = results.conjugate_gradient_results_pointer->final_selection_error;

            return(losses);
        }
        case TrainingStrategy::QUASI_NEWTON_METHOD:
        {
            losses[0] = results.quasi_Newton_method_results_pointer->final_loss;
            losses[1] = results.quasi_Newton_method_results_pointer->final_selection_error;

            return(losses);
        }
        case TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM:
        {
            losses[0] = results.Levenberg_Marquardt_algorithm_results_pointer->final_loss;
            losses[1] = results.Levenberg_Marquardt_algorithm_results_pointer->final_selection_error;

            return(losses);
        }
//        default:
//        {
//            ostringstream buffer;

//            buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
//                   << "Vector<double> get_final_losses(const TrainingStrategy::Results) method.\n"
//                   << "Unknown main type method.\n";

//            throw logic_error(buffer.str());
//        }
    }

    // Default

    ostringstream buffer;

    buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
           << "Vector<double> get_final_losses(const TrainingStrategy::Results) method.\n"
           << "Unknown main type method.\n";

    throw logic_error(buffer.str());

//    return(losses);
}


/// Return loss and selection depending on the loss calculation method.
/// @param inputs Vector of inputs to be trained with.

Vector<double> InputsSelectionAlgorithm::perform_model_evaluation(const Vector<bool>& inputs)
{
    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();

    neural_network_pointer->set_inputs(inputs);

    switch(loss_calculation_method)
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
    }

    return Vector<double>();
}


/// Returns the parameters of the neural network if the inputs is in the history.
/// @param inputs Vector of inputs to be trained with.

Vector<double> InputsSelectionAlgorithm::get_parameters_inputs(const Vector<bool>& inputs) const
{
#ifdef __OPENNN_DEBUG__

    if(inputs.count_equal_to(true) <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "Vector<double> get_parameters_inputs(const Vector<bool>&) method.\n"
               << "Inputs must be greater than 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    size_t i;

    Vector<double> parameters;

    for(i = 0; i < inputs_history.size(); i++)
    {
        if(inputs_history[i] == inputs)
        {
            parameters = parameters_history[i];

            break;
        }
    }

    return(parameters);

}


/// Return a string with the stopping condition of the training depending on the training method.
/// @param results Results of the perform_training method.

string InputsSelectionAlgorithm::write_stopping_condition(const TrainingStrategy::Results& results) const
{
    switch(training_strategy_pointer->get_training_method())
    {
        case TrainingStrategy::GRADIENT_DESCENT:
        {
            return results.gradient_descent_results_pointer->write_stopping_condition();
        }
        case TrainingStrategy::CONJUGATE_GRADIENT:
        {
            return results.conjugate_gradient_results_pointer->write_stopping_condition();
        }
        case TrainingStrategy::QUASI_NEWTON_METHOD:
        {
            return results.quasi_Newton_method_results_pointer->write_stopping_condition();
        }
        case TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM:
        {
            return results.Levenberg_Marquardt_algorithm_results_pointer->write_stopping_condition();
        }
//        default:
//        {
//            ostringstream buffer;

//            buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
//                   << "Vector<double> get_final_losses(const TrainingStrategy::Results) method.\n"
//                   << "Unknown main type method.\n";

//            throw logic_error(buffer.str());
//        }
    }

    // Default

    ostringstream buffer;

    buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
           << "Vector<double> get_final_losses(const TrainingStrategy::Results) method.\n"
           << "Unknown main type method.\n";

    throw logic_error(buffer.str());
}


/// Delete the history of the selection error values.

void InputsSelectionAlgorithm::delete_selection_history()
{
    selection_error_history.set();
}


/// Delete the history of the loss values.

void InputsSelectionAlgorithm::delete_loss_history()
{
    loss_history.set();
}


/// Delete the history of the parameters of the trained neural networks.

void InputsSelectionAlgorithm::delete_parameters_history()
{
    parameters_history.set();
}


/// Checks that the different pointers needed for performing the inputs selection are not nullptr.

void InputsSelectionAlgorithm::check() const
{
    // Optimization algorithm stuff

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index stuff


    if(!training_strategy_pointer->has_loss_index())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Neural network stuff

    if(!loss_index_pointer->has_neural_network())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer->has_multilayer_perceptron())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to multilayer perceptron is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();


    if(multilayer_perceptron_pointer->is_empty())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Multilayer Perceptron is empty.\n";

        throw logic_error(buffer.str());
    }

    /*
   if(multilayer_perceptron_pointer->get_layers_number() != 2)
   {
      buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
             << "void check() const method.\n"
             << "Number of layers in multilayer perceptron(" << multilayer_perceptron_pointer->get_layers_number() << ") must be 2.\n";

      throw logic_error(buffer.str());
   }*/


    // Data set stuff


    if(!loss_index_pointer->has_data_set())
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Instances& instances = data_set_pointer->get_instances();

    const size_t selection_instances_number = instances.get_selection_instances_number();

    if(selection_instances_number == 0)
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Number of selection instances is zero.\n";

        throw logic_error(buffer.str());
    }

}
/*
// tinyxml2::XMLDocument* to_XML() const method

/// Serializes the input selection object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document.

tinyxml2::XMLDocument* InputsSelectionAlgorithm::to_XML() const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    ostringstream buffer;

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


// void print() method

/// Prints to the screen the XML representation of this input selection object.

void InputsSelectionAlgorithm::print() const
{
    cout << to_XML();
}


// void save(const string&) const method

/// Saves the input selection members to a XML file.
/// @param file_name Name of input selection XML file.

void InputsSelectionAlgorithm::save(const string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
}


// void load(const string&) method

/// Loads the input selection members from a XML file.
/// @param file_name Name of input selection XML file.

void InputsSelectionAlgorithm::load(const string& file_name)
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    if(document->LoadFile(file_name.c_str()))
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw logic_error(buffer.str());
    }

    // Root

    tinyxml2::XMLElement* inputs_selection_element = document->FirstChildElement("InputsSelectionAlgorithm");

    if(!inputs_selection_element)
    {
        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "void load(const string&) method.\n"
               << "Invalid input input selection XML root element.\n";

        throw logic_error(buffer.str());
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


/// Return a string with the stopping condition of the InputsSelectionResults

string InputsSelectionAlgorithm::InputsSelectionResults::write_stopping_condition() const
{
    switch(stopping_condition)
    {
        case MaximumTime:
        {
            return("MaximumTime");
        }
        case SelectionErrorGoal:
        {
            return("SelectionErrorGoal");
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
    }

    return string();
}


/// Returns a string representation of the current inputs selection results structure.

string InputsSelectionAlgorithm::InputsSelectionResults::object_to_string() const
{
   ostringstream buffer;

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

   // Loss history

   if(!loss_data.empty())
   {
       buffer << "% Loss history:\n"
              << loss_data.to_row_matrix() << "\n";
   }

   // Selection loss history

   if(!selection_error_data.empty())
   {
       buffer << "% Selection loss history:\n"
              << selection_error_data.to_row_matrix() << "\n";
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

   // Optimum selection error

   if(fabs(final_selection_error - 0) > numeric_limits<double>::epsilon())
   {
       buffer << "% Optimum selection error:\n"
              << final_selection_error << "\n";
   }

   // Final loss

   if(fabs(final_loss - 0) > numeric_limits<double>::epsilon())
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
          << write_elapsed_time(elapsed_time) << "\n";



   return(buffer.str());
}


/// Return the index of uses where is the(input_number)-th input.
/// @param uses vector of the uses of the variables.
/// @param input_number index of the input to find.

size_t InputsSelectionAlgorithm::get_input_index(const Vector<Variables::Use> uses, const size_t input_number)
{
#ifdef __OPENNN_DEBUG__

    if(uses.size() < input_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelectionAlgorithm class.\n"
               << "const size_t get_input_index(const Vector<Variables::Use>, const size_t) method.\n"
               << "Size of uses vector must be greater than " <<  input_number << ".\n";

        throw logic_error(buffer.str());
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



