/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O R D E R   S E L E C T I O N   A L G O R I T H M   C L A S S                                              */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// OpenNN includes

#include "order_selection_algorithm.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

OrderSelectionAlgorithm::OrderSelectionAlgorithm(void)
    : training_strategy_pointer(NULL)
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

OrderSelectionAlgorithm::OrderSelectionAlgorithm(TrainingStrategy* new_training_strategy_pointer)
    : training_strategy_pointer(new_training_strategy_pointer)
{
    set_default();
}


// FILE CONSTRUCTOR

/// File constructor.
/*/// @param file_name Name of XML order selection file.*/

OrderSelectionAlgorithm::OrderSelectionAlgorithm(const std::string&)
    : training_strategy_pointer(NULL)
{
    //load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor.
/*/// @param order_selection_document Pointer to a TinyXML document containing the order selection algorithm data.*/

OrderSelectionAlgorithm::OrderSelectionAlgorithm(const tinyxml2::XMLDocument& )
    : training_strategy_pointer(NULL)
{
    //from_XML(order_selection_document);
}


// DESTRUCTOR

/// Destructor.

OrderSelectionAlgorithm::~OrderSelectionAlgorithm(void)
{
}


// METHODS

// TrainingStrategy* get_training_strategy_pointer(void) const method

/// Returns a pointer to the training strategy object.

TrainingStrategy* OrderSelectionAlgorithm::get_training_strategy_pointer(void) const
{
#ifdef __OPENNN_DEBUG__

    if(!training_strategy_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "DataSet* get_training_strategy_pointer(void) const method.\n"
               << "Training strategy pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    return(training_strategy_pointer);
}

// bool has_training_strategy(void) const method

/// Returns true if this order selection algorithm has a training strategy associated, and false otherwise.

bool OrderSelectionAlgorithm::has_training_strategy(void) const
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

// const size_t& get_maximum_order(void) const method

/// Returns the maximum of the hidden perceptrons number used in the order order selection.

const size_t& OrderSelectionAlgorithm::get_maximum_order(void) const
{
    return(maximum_order);
}


// const size_t& get_minimum_order(void) const method

/// Returns the minimum of the hidden perceptrons number used in the order selection.

const size_t& OrderSelectionAlgorithm::get_minimum_order(void) const
{
    return(minimum_order);
}

// const size_t& get_trials_number(void) const method

/// Returns the number of trials for each network architecture.

const size_t& OrderSelectionAlgorithm::get_trials_number(void) const
{
    return(trials_number);
}

// const bool& get_reserve_parameters_data(void) const method

/// Returns true if the neural network parameters are to be reserved, and false otherwise.

const bool& OrderSelectionAlgorithm::get_reserve_parameters_data(void) const
{
    return(reserve_parameters_data);
}


// const bool& get_reserve_loss_data(void) const method

/// Returns true if the loss functional losss are to be reserved, and false otherwise.

const bool& OrderSelectionAlgorithm::get_reserve_loss_data(void) const
{
    return(reserve_loss_data);
}


// const bool& get_reserve_selection_loss_data(void) const method

/// Returns true if the loss functional selection losss are to be reserved, and false otherwise.

const bool& OrderSelectionAlgorithm::get_reserve_selection_loss_data(void) const
{
    return(reserve_selection_loss_data);
}


// const bool& get_reserve_minimal_parameters(void) const method

/// Returns true if the parameters vector of the neural network with minimum selection loss is to be reserved, and false otherwise.

const bool& OrderSelectionAlgorithm::get_reserve_minimal_parameters(void) const
{
    return(reserve_minimal_parameters);
}

// const PerformanceCalculationMethod& get_loss_calculation_method(void) const method

/// Returns the method for the calculation of the loss and the selection loss.

const OrderSelectionAlgorithm::PerformanceCalculationMethod& OrderSelectionAlgorithm::get_loss_calculation_method(void) const
{
    return(loss_calculation_method);
}

// const bool& get_display(void) const method

/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& OrderSelectionAlgorithm::get_display(void) const
{
    return(display);
}

// const double& get_selection_loss_goal(void) const method

/// Returns the goal for the selection loss in the order selection algorithm.

const double& OrderSelectionAlgorithm::get_selection_loss_goal(void) const
{
    return(selection_loss_goal);
}


// const size_t& get_maximum_iterations_number(void) const method

/// Returns the maximum number of iterations in the order selection algorithm.

const size_t& OrderSelectionAlgorithm::get_maximum_iterations_number(void) const
{
    return(maximum_iterations_number);
}


// const double& get_maximum_time(void) const method

/// Returns the maximum time in the order selection algorithm.

const double& OrderSelectionAlgorithm::get_maximum_time(void) const
{
    return(maximum_time);
}

// const double& get_tolerance(void) const method

/// Return the tolerance of error for the order selection algorithm.

const double& OrderSelectionAlgorithm::get_tolerance(void) const
{
    return(tolerance);
}

// std::string write_loss_calculation_method(void) const method

/// Return a string with the loss calculation method of this order selection algorithm.

std::string OrderSelectionAlgorithm::write_loss_calculation_method(void) const
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

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "std::string write_loss_calculation_method(void) const method.\n"
               << "Unknown loss calculation method.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }
}


// void set_training_strategy_pointer(TrainingStrategy*) method

/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void OrderSelectionAlgorithm::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;
}


// void set_default(void) method 

/// Sets the members of the order selection object to their default values.

void OrderSelectionAlgorithm::set_default(void)
{

    size_t inputs_number ;
    size_t outputs_number;

    if(training_strategy_pointer == NULL
            || !training_strategy_pointer->has_loss_index())
    {
        inputs_number = 0;
        outputs_number = 0;
    }
    else
    {
        inputs_number = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer()->get_inputs_number();
        outputs_number = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer()->get_outputs_number();
    }
    // MEMBERS

    minimum_order = 1;
    // Heuristic value for the maximum_order
    maximum_order = 2*(inputs_number + outputs_number);
    trials_number = 1;

    // order selection results

    reserve_parameters_data = true;
    reserve_loss_data = true;
    reserve_selection_loss_data = true;
    reserve_minimal_parameters = true;

    loss_calculation_method = Minimum;

    display = true;

    // STOPPING CRITERIA

    selection_loss_goal = 0.0;

    maximum_iterations_number = 1000;
    maximum_time = 10000.0;

    tolerance = 1.0e-3;
}

// void set_maximum_order(const size_t&) method

/// Sets the number of the maximum hidden perceptrons for the order selection algorithm.
/// @param new_maximum_order Number of maximum hidden perceptrons.

void OrderSelectionAlgorithm::set_maximum_order(const size_t& new_maximum_order)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_order <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_maximum_order(const size_t&) method.\n"
               << "maximum_order (" << new_maximum_order << ") must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

    if(new_maximum_order <= minimum_order)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_maximum_order(const size_t&) method.\n"
               << "maximum_order (" << new_maximum_order << ") must be greater than minimum_order (" << minimum_order << ").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    maximum_order = new_maximum_order;
}


// void set_minimum_order(const size_t&) method

/// Sets the number of the minimum hidden perceptrons for the order selection algorithm.
/// @param new_minimum_order Number of minimum hidden perceptrons.

void OrderSelectionAlgorithm::set_minimum_order(const size_t& new_minimum_order)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_order <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_minimum_order(const size_t&) method.\n"
               << "minimum_order (" << new_minimum_order << ") must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

    if(new_minimum_order >= maximum_order)
    {
        std::ostringstream buffer;
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_minimum_order(const size_t&) method.\n"
               << "minimum_order (" << new_minimum_order << ") must be less than maximum_order (" << maximum_order << ").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    minimum_order = new_minimum_order;
}

// void set_trials_number(const size_t&) method

/// Sets the number of times that each different neural network is to be trained.
/// @param new_trials_number Number of assays for each set of parameters.

void OrderSelectionAlgorithm::set_trials_number(const size_t& new_trials_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_trials_number <= 0)
    {
        std::ostringstream buffer;
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
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

void OrderSelectionAlgorithm::set_reserve_parameters_data(const bool& new_reserve_parameters_data)
{
    reserve_parameters_data = new_reserve_parameters_data;
}


// void set_reserve_loss_data(const bool&) method

/// Sets the reserve flag for the loss data.
/// @param new_reserve_loss_data Flag value.

void OrderSelectionAlgorithm::set_reserve_loss_data(const bool& new_reserve_loss_data)
{
    reserve_loss_data = new_reserve_loss_data;
}


// void set_reserve_selection_loss_data(const bool&) method

/// Sets the reserve flag for the selection loss data.
/// @param new_reserve_selection_loss_data Flag value.

void OrderSelectionAlgorithm::set_reserve_selection_loss_data(const bool& new_reserve_selection_loss_data)
{
    reserve_selection_loss_data = new_reserve_selection_loss_data;
}


// void set_reserve_minimal_parameters(const bool&) method

/// Sets the reserve flag for the minimal parameters.
/// @param new_reserve_minimal_parameters Flag value.

void OrderSelectionAlgorithm::set_reserve_minimal_parameters(const bool& new_reserve_minimal_parameters)
{
    reserve_minimal_parameters = new_reserve_minimal_parameters;
}

// void set_loss_calculation_method(const PerformanceCalculationMethod&) method

/// Sets a new method to calculate the loss and the selection loss.
/// @param new_loss_calculation_method Method to calculate the loss (Minimum, Maximum or Mean).

void OrderSelectionAlgorithm::set_loss_calculation_method(const OrderSelectionAlgorithm::PerformanceCalculationMethod& new_loss_calculation_method)
{
    loss_calculation_method = new_loss_calculation_method;
}

// void set_loss_calculation_method(const std::string&) method

/// Sets a new loss calculation method from a string.
/// @param new_loss_calculation_method String with the loss calculation method.

void OrderSelectionAlgorithm::set_loss_calculation_method(const std::string& new_loss_calculation_method)
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

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
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

void OrderSelectionAlgorithm::set_display(const bool& new_display)
{
    display = new_display;
}

// void set_selection_loss_goal(const double&) method

/// Sets the selection loss goal for the order selection algorithm.
/// @param new_selection_loss_goal Goal of the selection loss.

void OrderSelectionAlgorithm::set_selection_loss_goal(const double& new_selection_loss_goal)
{
#ifdef __OPENNN_DEBUG__

    if(new_selection_loss_goal < 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_selection_loss_goal(const double&) method.\n"
               << "Selection loss goal must be greater or equal than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    selection_loss_goal = new_selection_loss_goal;
}


// void set_maximum_iterations_number(const size_t&) method

/// Sets the maximum iterations number for the order selection algorithm.
/// @param new_maximum_iterations_number Maximum number of iterations.

void OrderSelectionAlgorithm::set_maximum_iterations_number(const size_t& new_maximum_iterations_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_iterations_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_maximum_iterations_number(const size_t&) method.\n"
               << "Maximum iterations number must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    maximum_iterations_number = new_maximum_iterations_number;
}


// void set_maximum_time(const double&) method

/// Sets the maximum time for the order selection algorithm.
/// @param new_maximum_time Maximum time for the algorithm.

void OrderSelectionAlgorithm::set_maximum_time(const double& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_maximum_time(const double&) method.\n"
               << "Maximum time must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    maximum_time = new_maximum_time;
}

// void set_tolerance(const double&) method

/// Set the tolerance for the errors in the trainings of the algorithm.
/// @param new_tolerance Value of the tolerance.

void OrderSelectionAlgorithm::set_tolerance(const double& new_tolerance)
{
#ifdef __OPENNN_DEBUG__

    if(new_tolerance < 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_tolerance(const double&) method.\n"
               << "Tolerance must be equal or greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    tolerance = new_tolerance;
}


// Vector<double> perform_minimum_model_evaluation(const size_t&) method

/// Returns the minimum of the loss and selection loss in trials_number trainings
/// @param order_number Number of perceptrons in the hidden layer to be trained with.

Vector<double> OrderSelectionAlgorithm::perform_minimum_model_evaluation(const size_t& order_number)
{
#ifdef __OPENNN_DEBUG__

    if(order_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of hidden perceptrons must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
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

    for (size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            final[0] = loss_history[i];
            flag_loss = true;
        }
    }

    for (size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            final[1] = selection_loss_history[i];
            flag_selection = true;
        }
    }

    if(flag_loss && flag_selection)
    {
        return(final);
    }

    MultilayerPerceptron* multilayer_perceptron = neural_network->get_multilayer_perceptron_pointer();
    const size_t last_hidden_layer = multilayer_perceptron->get_layers_number()-2;
    const size_t perceptrons_number = multilayer_perceptron->get_layer_pointer(last_hidden_layer)->get_perceptrons_number();

    if(order_number > perceptrons_number)
    {
        multilayer_perceptron->grow_layer_perceptron(last_hidden_layer,order_number-perceptrons_number);
        neural_network->perturbate_parameters(0.001);
#ifdef __OPENNN_MPI__

        neural_network->set_MPI(neural_network);

#endif
        training_strategy_results = training_strategy_pointer->perform_training();

        final_parameters.set(neural_network->arrange_parameters());
        final = get_final_losss(training_strategy_results);
    }
    else
    {
        for (size_t i = 0; i < (perceptrons_number-order_number); i++)
        {
            multilayer_perceptron->prune_layer_perceptron(last_hidden_layer,0);
        }

        neural_network->perturbate_parameters(0.001);

#ifdef __OPENNN_MPI__

        neural_network->set_MPI(neural_network);

#endif
        training_strategy_results = training_strategy_pointer->perform_training();

        final_parameters.set(neural_network->arrange_parameters());
        final = get_final_losss(training_strategy_results);
    }

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

    order_history.push_back(order_number);

    loss_history.push_back(final[0]);

    selection_loss_history.push_back(final[1]);

    parameters_history.push_back(final_parameters);

    return final;
}


// Vector<double> perform_maximum_model_evaluation(const size_t&) const method

/// Returns the maximum of the loss and selection loss in trials_number trainings
/// @param order_number Number of perceptrons in the hidden layer to be trained with.

Vector<double> OrderSelectionAlgorithm::perform_maximum_model_evaluation(const size_t& order_number)
{
#ifdef __OPENNN_DEBUG__

    if(order_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_maximum_model_evaluation(size_t) method.\n"
               << "Number of hidden perceptrons must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_maximum_model_evaluation(size_t) method.\n"
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

    for (size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            final[0] = loss_history[i];
            flag_loss = true;
        }
    }



    for (size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            final[1] = selection_loss_history[i];
            flag_selection = true;
        }
    }


    if(flag_loss && flag_selection)
    {
        return(final);
    }

    MultilayerPerceptron* multilayer_perceptron = neural_network->get_multilayer_perceptron_pointer();
    const size_t last_hidden_layer = multilayer_perceptron->get_layers_number()-2;
    const size_t perceptrons_number = multilayer_perceptron->get_layer_pointer(last_hidden_layer)->get_perceptrons_number();

    if(order_number > perceptrons_number)
    {
        multilayer_perceptron->grow_layer_perceptron(last_hidden_layer,order_number-perceptrons_number);
        neural_network->perturbate_parameters(0.001);
        training_strategy_results = training_strategy_pointer->perform_training();

        final_parameters.set(neural_network->arrange_parameters());
        final = get_final_losss(training_strategy_results);
    }
    else
    {
        training_strategy_pointer->get_quasi_Newton_method_pointer()->set_maximum_selection_loss_decreases(training_strategy_pointer->get_quasi_Newton_method_pointer()->get_maximum_iterations_number());

        for (size_t i = 0; i < (perceptrons_number-order_number); i++)
        {
            multilayer_perceptron->prune_layer_perceptron(last_hidden_layer,0);
        }

        neural_network->perturbate_parameters(0.001);
        training_strategy_results = training_strategy_pointer->perform_training();

        final_parameters.set(neural_network->arrange_parameters());
        final = get_final_losss(training_strategy_results);
    }

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

    order_history.push_back(order_number);

    loss_history.push_back(final[0]);

    selection_loss_history.push_back(final[1]);

    parameters_history.push_back(final_parameters);

    return final;
}


// Vector<double> perform_mean_model_evaluation(const size_t&) method

/// Returns the mean of the loss and selection loss in trials_number trainings
/// @param order_number Number of perceptrons in the hidden layer to be trained with.

Vector<double> OrderSelectionAlgorithm::perform_mean_model_evaluation(const size_t& order_number)
{
#ifdef __OPENNN_DEBUG__

    if(order_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_mean_model_evaluation(size_t) method.\n"
               << "Number of hidden perceptrons must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_mean_model_evaluation(size_t) method.\n"
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


    for (size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            mean_final[0] = loss_history[i];
            flag_loss = true;
        }
    }



    for (size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            mean_final[1] = selection_loss_history[i];
            flag_selection = true;
        }
    }


    if(flag_loss && flag_selection)
    {
        return(mean_final);
    }

    MultilayerPerceptron* multilayer_perceptron = neural_network->get_multilayer_perceptron_pointer();
    const size_t last_hidden_layer = multilayer_perceptron->get_layers_number()-2;
    const size_t perceptrons_number = multilayer_perceptron->get_layer_pointer(last_hidden_layer)->get_perceptrons_number();

    if(order_number > perceptrons_number)
    {
        multilayer_perceptron->grow_layer_perceptron(last_hidden_layer,order_number-perceptrons_number);
        neural_network->perturbate_parameters(0.001);
        training_strategy_results = training_strategy_pointer->perform_training();

        final_parameters.set(neural_network->arrange_parameters());
        mean_final = get_final_losss(training_strategy_results);
    }
    else
    {
        for (size_t i = 0; i < (perceptrons_number-order_number); i++)
        {
            multilayer_perceptron->prune_layer_perceptron(last_hidden_layer,0);
        }

        neural_network->perturbate_parameters(0.001);
        training_strategy_results = training_strategy_pointer->perform_training();

        final_parameters.set(neural_network->arrange_parameters());
        mean_final = get_final_losss(training_strategy_results);
    }

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

    order_history.push_back(order_number);

    loss_history.push_back(mean_final[0]);

    selection_loss_history.push_back(mean_final[1]);

    parameters_history.push_back(final_parameters);

    return mean_final;
}

// Vector<double> get_final_losss(const TrainingStrategy::Results&) const method

/// Return final training loss and final selection loss depending on the training method.
/// @param results Results of the perform_training method.

Vector<double> OrderSelectionAlgorithm::get_final_losss(const TrainingStrategy::Results& results) const
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

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> get_final_losss(const TrainingStrategy::Results) method.\n"
               << "Unknown main type method.\n";

        throw std::logic_error(buffer.str());
    }
    }

    return(losss);
}

// Vector<double> perform_model_evaluation(const size_t&) method

/// Return loss and selection depending on the loss calculation method.
/// @param order_number Number of perceptrons in the hidden layer to be trained with.

Vector<double> OrderSelectionAlgorithm::perform_model_evaluation(const size_t& order_number)
{
    switch (loss_calculation_method)
    {
    case Maximum:
    {
        return(perform_maximum_model_evaluation(order_number));
    }
    case Minimum:
    {
        return(perform_minimum_model_evaluation(order_number));
    }
    case Mean:
    {
        return(perform_mean_model_evaluation(order_number));
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_model_evaluation(const size_t) method.\n"
               << "Unknown loss calculation method.\n";

        throw std::logic_error(buffer.str());
    }
    }
}


// Vector<double> get_parameters_order(const size_t&) const method

/// Returns the parameters of the neural network if the order is in the history.
/// @param order Order of the neural network.

Vector<double> OrderSelectionAlgorithm::get_parameters_order(const size_t& order) const
{
#ifdef __OPENNN_DEBUG__

    if(order <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> get_parameters_order(const size_t&) method.\n"
               << "Order must be greater than 0.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    size_t i;
    Vector<double> parameters;

    for (i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order)
        {
            parameters = parameters_history[i];
            break;
        }
    }

    return(parameters);
}

// void delete_selection_history(void) method 

/// Delete the history of the selection loss values.

void OrderSelectionAlgorithm::delete_selection_history(void)
{
    selection_loss_history.set();
}

// void delete_loss_history(void) method 

/// Delete the history of the loss values.

void OrderSelectionAlgorithm::delete_loss_history(void)
{
    loss_history.set();
}

// void delete_parameters_history(void) method 

/// Delete the history of the parameters of the trained neural networks.

void OrderSelectionAlgorithm::delete_parameters_history(void)
{
    parameters_history.set();
}

// void check(void) const method

/// Checks that the different pointers needed for performing the order selection are not NULL.

void OrderSelectionAlgorithm::check(void) const
{
    // Training algorithm stuff

    std::ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to training strategy is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to loss functional is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Neural network stuff

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to neural network is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to multilayer perceptron is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    if(multilayer_perceptron_pointer->is_empty())
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Multilayer Perceptron is empty.\n";

        throw std::logic_error(buffer.str());
    }


   if(multilayer_perceptron_pointer->get_layers_number() == 1)
   {
      buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
             << "void check(void) const method.\n"
             << "Number of layers in multilayer perceptron must be greater than 1.\n";

      throw std::logic_error(buffer.str());
   }


    // Data set stuff

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Pointer to data set is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const Instances& instances = data_set_pointer->get_instances();

    const size_t selection_instances_number = instances.count_selection_instances_number();

    if(selection_instances_number == 0)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check(void) const method.\n"
               << "Number of selection instances is zero.\n";

        throw std::logic_error(buffer.str());
    }

}


// std::string write_stopping_condition(void) const method

/// Return a string with the stopping condition of the OrderSelectionResults

std::string OrderSelectionAlgorithm::OrderSelectionResults::write_stopping_condition(void) const
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
    case MaximumIterations:
    {
        return("MaximumIterations");
    }
    case MaximumSelectionFailures:
    {
        return("MaximumSelectionFailures");
    }
    case MinimumTemperature:
    {
        return("MinimumTemperature");
    }
    case AlgorithmFinished:
    {
        return("AlgorithmFinished");
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionResults struct.\n"
               << "std::string write_stopping_condition(void) const method.\n"
               << "Unknown stopping condition type.\n";

        throw std::logic_error(buffer.str());

        break;
    }
    }

}


// std::string to_string(void) const method

/// Returns a string representation of the current order selection results structure.

std::string OrderSelectionAlgorithm::OrderSelectionResults::to_string(void) const
{
   std::ostringstream buffer;

   // Order history

   if(!order_data.empty())
   {
     buffer << "% Order history:\n"
            << order_data.to_row_matrix() << "\n";
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

   // Optimal order

   if(optimal_order != 0)
   {
       buffer << "% Optimal order:\n"
              << optimal_order << "\n";
   }

   // Iterations number


   buffer << "% Number of iterations:\n"
          << iterations_number << "\n";


   // Elapsed time

   buffer << "% Elapsed time:\n"
          << elapsed_time << "\n";



   return(buffer.str());
}
}

// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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
