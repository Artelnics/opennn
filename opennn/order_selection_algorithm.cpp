/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O R D E R   S E L E C T I O N   A L G O R I T H M   C L A S S                                              */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// OpenNN includes

#include "order_selection_algorithm.h"

namespace OpenNN {

// DEFAULT CONSTRUCTOR

/// Default constructor.

OrderSelectionAlgorithm::OrderSelectionAlgorithm()
    : training_strategy_pointer(nullptr)
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

OrderSelectionAlgorithm::OrderSelectionAlgorithm(const string&)
    : training_strategy_pointer(nullptr)
{
    //load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor.
/*/// @param order_selection_document Pointer to a TinyXML document containing the order selection algorithm data.*/

OrderSelectionAlgorithm::OrderSelectionAlgorithm(const tinyxml2::XMLDocument& )
    : training_strategy_pointer(nullptr)
{
    //from_XML(order_selection_document);
}


// DESTRUCTOR

/// Destructor.

OrderSelectionAlgorithm::~OrderSelectionAlgorithm()
{
}


// METHODS


/// Returns a pointer to the training strategy object.

TrainingStrategy* OrderSelectionAlgorithm::get_training_strategy_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!training_strategy_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "DataSet* get_training_strategy_pointer() const method.\n"
               << "Training strategy pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(training_strategy_pointer);
}


/// Returns true if this order selection algorithm has a training strategy associated, and false otherwise.

bool OrderSelectionAlgorithm::has_training_strategy() const
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


/// Returns the maximum of the hidden perceptrons number used in the order order selection.

const size_t& OrderSelectionAlgorithm::get_maximum_order() const
{
    return(maximum_order);
}


/// Returns the minimum of the hidden perceptrons number used in the order selection.

const size_t& OrderSelectionAlgorithm::get_minimum_order() const
{
    return(minimum_order);
}


/// Returns the number of trials for each network architecture.

const size_t& OrderSelectionAlgorithm::get_trials_number() const
{
    return(trials_number);
}


/// Returns true if the neural network parameters are to be reserved, and false otherwise.

const bool& OrderSelectionAlgorithm::get_reserve_parameters_data() const
{
    return(reserve_parameters_data);
}


/// Returns true if the loss index losses are to be reserved, and false otherwise.

const bool& OrderSelectionAlgorithm::get_reserve_error_data() const
{
    return(reserve_error_data);
}


/// Returns true if the loss index selection losses are to be reserved, and false otherwise.

const bool& OrderSelectionAlgorithm::get_reserve_selection_error_data() const
{
    return(reserve_selection_error_data);
}


/// Returns true if the parameters vector of the neural network with minimum selection error is to be reserved, and false otherwise.

const bool& OrderSelectionAlgorithm::get_reserve_minimal_parameters() const
{
    return(reserve_minimal_parameters);
}


/// Returns the method for the calculation of the loss and the selection error.

const OrderSelectionAlgorithm::LossCalculationMethod& OrderSelectionAlgorithm::get_loss_calculation_method() const
{
    return(loss_calculation_method);
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& OrderSelectionAlgorithm::get_display() const
{
    return(display);
}


/// Returns the goal for the selection error in the order selection algorithm.

const double& OrderSelectionAlgorithm::get_selection_error_goal() const
{
    return(selection_error_goal);
}


/// Returns the maximum number of iterations in the order selection algorithm.

const size_t& OrderSelectionAlgorithm::get_maximum_iterations_number() const
{
    return(maximum_iterations_number);
}


/// Returns the maximum time in the order selection algorithm.

const double& OrderSelectionAlgorithm::get_maximum_time() const
{
    return(maximum_time);
}


/// Return the tolerance of error for the order selection algorithm.

const double& OrderSelectionAlgorithm::get_tolerance() const
{
    return(tolerance);
}


/// Return a string with the loss calculation method of this order selection algorithm.

string OrderSelectionAlgorithm::write_loss_calculation_method() const
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


/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void OrderSelectionAlgorithm::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;
}


/// Sets the members of the order selection object to their default values.

void OrderSelectionAlgorithm::set_default()
{
    size_t inputs_number;
    size_t outputs_number;

    if(training_strategy_pointer == nullptr
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
    reserve_error_data = true;
    reserve_selection_error_data = true;
    reserve_minimal_parameters = true;

    loss_calculation_method = Minimum;

    display = true;

    // STOPPING CRITERIA

    selection_error_goal = 0.0;

    maximum_iterations_number = 1000;
    maximum_time = 10000.0;

    tolerance = 0.0;
}


/// Sets the number of the maximum hidden perceptrons for the order selection algorithm.
/// @param new_maximum_order Number of maximum hidden perceptrons.

void OrderSelectionAlgorithm::set_maximum_order(const size_t& new_maximum_order)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_order <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_maximum_order(const size_t&) method.\n"
               << "maximum_order(" << new_maximum_order << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(new_maximum_order < minimum_order)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_maximum_order(const size_t&) method.\n"
               << "maximum_order(" << new_maximum_order << ") must be equal or greater than minimum_order(" << minimum_order << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_order = new_maximum_order;
}


/// Sets the number of the minimum hidden perceptrons for the order selection algorithm.
/// @param new_minimum_order Number of minimum hidden perceptrons.

void OrderSelectionAlgorithm::set_minimum_order(const size_t& new_minimum_order)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_order <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_minimum_order(const size_t&) method.\n"
               << "minimum_order(" << new_minimum_order << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(new_minimum_order >= maximum_order)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_minimum_order(const size_t&) method.\n"
               << "minimum_order(" << new_minimum_order << ") must be less than maximum_order(" << maximum_order << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_order = new_minimum_order;
}


/// Sets the number of times that each different neural network is to be trained.
/// @param new_trials_number Number of assays for each set of parameters.

void OrderSelectionAlgorithm::set_trials_number(const size_t& new_trials_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_trials_number <= 0)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_trials_number(const size_t&) method.\n"
               << "Number of assays must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    trials_number = new_trials_number;
}


/// Sets the reserve flag for the parameters data.
/// @param new_reserve_parameters_data Flag value.

void OrderSelectionAlgorithm::set_reserve_parameters_data(const bool& new_reserve_parameters_data)
{
    reserve_parameters_data = new_reserve_parameters_data;
}


/// Sets the reserve flag for the loss data.
/// @param new_reserve_error_data Flag value.

void OrderSelectionAlgorithm::set_reserve_error_data(const bool& new_reserve_error_data)
{
    reserve_error_data = new_reserve_error_data;
}


/// Sets the reserve flag for the selection error data.
/// @param new_reserve_selection_error_data Flag value.

void OrderSelectionAlgorithm::set_reserve_selection_error_data(const bool& new_reserve_selection_error_data)
{
    reserve_selection_error_data = new_reserve_selection_error_data;
}


/// Sets the reserve flag for the minimal parameters.
/// @param new_reserve_minimal_parameters Flag value.

void OrderSelectionAlgorithm::set_reserve_minimal_parameters(const bool& new_reserve_minimal_parameters)
{
    reserve_minimal_parameters = new_reserve_minimal_parameters;
}


/// Sets a new method to calculate the loss and the selection error.
/// @param new_loss_calculation_method Method to calculate the loss(Minimum, Maximum or Mean).

void OrderSelectionAlgorithm::set_loss_calculation_method(const OrderSelectionAlgorithm::LossCalculationMethod& new_loss_calculation_method)
{
    loss_calculation_method = new_loss_calculation_method;
}


/// Sets a new loss calculation method from a string.
/// @param new_loss_calculation_method String with the loss calculation method.

void OrderSelectionAlgorithm::set_loss_calculation_method(const string& new_loss_calculation_method)
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

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_loss_calculation_method(const string&) method.\n"
               << "Unknown loss calculation method.\n";

        throw logic_error(buffer.str());

    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void OrderSelectionAlgorithm::set_display(const bool& new_display)
{
    display = new_display;
}


/// Sets the selection error goal for the order selection algorithm.
/// @param new_selection_error_goal Goal of the selection error.

void OrderSelectionAlgorithm::set_selection_error_goal(const double& new_selection_error_goal)
{
#ifdef __OPENNN_DEBUG__

    if(new_selection_error_goal < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_selection_error_goal(const double&) method.\n"
               << "Selection loss goal must be greater or equal than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    selection_error_goal = new_selection_error_goal;
}


/// Sets the maximum iterations number for the order selection algorithm.
/// @param new_maximum_iterations_number Maximum number of iterations.

void OrderSelectionAlgorithm::set_maximum_iterations_number(const size_t& new_maximum_iterations_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_iterations_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_maximum_iterations_number(const size_t&) method.\n"
               << "Maximum iterations number must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_iterations_number = new_maximum_iterations_number;
}


/// Sets the maximum time for the order selection algorithm.
/// @param new_maximum_time Maximum time for the algorithm.

void OrderSelectionAlgorithm::set_maximum_time(const double& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_maximum_time(const double&) method.\n"
               << "Maximum time must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_time = new_maximum_time;
}


/// Set the tolerance for the errors in the trainings of the algorithm.
/// @param new_tolerance Value of the tolerance.

void OrderSelectionAlgorithm::set_tolerance(const double& new_tolerance)
{
#ifdef __OPENNN_DEBUG__

    if(new_tolerance < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void set_tolerance(const double&) method.\n"
               << "Tolerance must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    tolerance = new_tolerance;
}


/// Returns the minimum of the loss and selection error in trials_number trainings
/// @param order_number Number of perceptrons in the hidden layer to be trained with.

Vector<double> OrderSelectionAlgorithm::perform_minimum_model_evaluation(const size_t& order_number)
{
#ifdef __OPENNN_DEBUG__

    if(order_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of hidden perceptrons must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_minimum_model_evaluation(size_t) method.\n"
               << "Number of parameters assay must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    NeuralNetwork* neural_network = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();

    TrainingStrategy::Results training_strategy_results;

    Vector<double> final(2);
    final[0] = std::numeric_limits<double>::max();
    final[1] = std::numeric_limits<double>::max();

    Vector<double> current_loss(2);

    Vector<double> final_parameters;

    bool flag_loss = false;
    bool flag_selection = false;

    for(size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            final[0] = loss_history[i];
            flag_loss = true;
        }
    }

    for(size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            final[1] = selection_error_history[i];
            flag_selection = true;
        }
    }

    if(flag_loss && flag_selection)
    {
        return(final);
    }

    MultilayerPerceptron* multilayer_perceptron = neural_network->get_multilayer_perceptron_pointer();
    const size_t last_hidden_layer = multilayer_perceptron->get_layers_number()-2;

    multilayer_perceptron->set_layer_perceptrons_number(last_hidden_layer, order_number);
    neural_network->randomize_parameters_normal();

#ifdef __OPENNN_MPI__

        neural_network->set_MPI(neural_network);

#endif

    training_strategy_results = training_strategy_pointer->perform_training();

    final_parameters.set(neural_network->get_parameters());
    final = get_final_losses(training_strategy_results);

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

    order_history.push_back(order_number);

    loss_history.push_back(final[0]);

    selection_error_history.push_back(final[1]);

    parameters_history.push_back(final_parameters);
//    cout << "final " << final << endl; cout.flush();
//    cout << "trials " << trials_number << endl; cout.flush();
//    cout << neural_network->get_multilayer_perceptron_pointer()->get_architecture() << endl; cout.flush();
    return final;
}


/// Returns the maximum of the loss and selection error in trials_number trainings
/// @param order_number Number of perceptrons in the hidden layer to be trained with.

Vector<double> OrderSelectionAlgorithm::perform_maximum_model_evaluation(const size_t& order_number)
{
#ifdef __OPENNN_DEBUG__

    if(order_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_maximum_model_evaluation(size_t) method.\n"
               << "Number of hidden perceptrons must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_maximum_model_evaluation(size_t) method.\n"
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

    for(size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            final[0] = loss_history[i];
            flag_loss = true;
        }
    }



    for(size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            final[1] = selection_error_history[i];
            flag_selection = true;
        }
    }


    if(flag_loss && flag_selection)
    {
        return(final);
    }

    MultilayerPerceptron* multilayer_perceptron = neural_network->get_multilayer_perceptron_pointer();
    const size_t last_hidden_layer = multilayer_perceptron->get_layers_number()-2;

    multilayer_perceptron->set_layer_perceptrons_number(last_hidden_layer, order_number);
    neural_network->randomize_parameters_normal();

#ifdef __OPENNN_MPI__

        neural_network->set_MPI(neural_network);

#endif

    training_strategy_results = training_strategy_pointer->perform_training();

    final_parameters.set(neural_network->get_parameters());
    final = get_final_losses(training_strategy_results);

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

    order_history.push_back(order_number);

    loss_history.push_back(final[0]);

    selection_error_history.push_back(final[1]);

    parameters_history.push_back(final_parameters);

    return final;
}


/// Returns the mean of the loss and selection error in trials_number trainings
/// @param order_number Number of perceptrons in the hidden layer to be trained with.

Vector<double> OrderSelectionAlgorithm::perform_mean_model_evaluation(const size_t& order_number)
{
#ifdef __OPENNN_DEBUG__

    if(order_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_mean_model_evaluation(size_t) method.\n"
               << "Number of hidden perceptrons must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(trials_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> perform_mean_model_evaluation(size_t) method.\n"
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


    for(size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            mean_final[0] = loss_history[i];
            flag_loss = true;
        }
    }



    for(size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order_number)
        {
            mean_final[1] = selection_error_history[i];
            flag_selection = true;
        }
    }


    if(flag_loss && flag_selection)
    {
        return(mean_final);
    }

    MultilayerPerceptron* multilayer_perceptron = neural_network->get_multilayer_perceptron_pointer();
    const size_t last_hidden_layer = multilayer_perceptron->get_layers_number()-2;

    multilayer_perceptron->set_layer_perceptrons_number(last_hidden_layer, order_number);
    neural_network->randomize_parameters_normal();

#ifdef __OPENNN_MPI__

        neural_network->set_MPI(neural_network);

#endif

    training_strategy_results = training_strategy_pointer->perform_training();

    final_parameters.set(neural_network->get_parameters());
    mean_final = get_final_losses(training_strategy_results);

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

#ifdef __OPENNN_MPI__

        neural_network->set_MPI(neural_network);

#endif
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

    order_history.push_back(order_number);

    loss_history.push_back(mean_final[0]);

    selection_error_history.push_back(mean_final[1]);

    parameters_history.push_back(final_parameters);

    return mean_final;
}


/// Return final training loss and final selection error depending on the training method.
/// @param results Results of the perform_training method.

Vector<double> OrderSelectionAlgorithm::get_final_losses(const TrainingStrategy::Results& results) const
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
        case TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT:
        {
            losses[0] = results.stochastic_gradient_descent_results_pointer->final_loss;
            losses[1] = results.stochastic_gradient_descent_results_pointer->final_selection_error;
            return(losses);
        }
        case TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION:
        {
            losses[0] = results.adaptive_moment_estimation_results_pointer->final_loss;
            losses[1] = results.adaptive_moment_estimation_results_pointer->final_selection_error;
            return(losses);
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


/// Return loss and selection depending on the loss calculation method.
/// @param order_number Number of perceptrons in the hidden layer to be trained with.

Vector<double> OrderSelectionAlgorithm::perform_model_evaluation(const size_t& order_number)
{
    switch(loss_calculation_method)
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
    }

    return Vector<double>();
}


/// Returns the parameters of the neural network if the order is in the history.
/// @param order Order of the neural network.

Vector<double> OrderSelectionAlgorithm::get_parameters_order(const size_t& order) const
{
#ifdef __OPENNN_DEBUG__

    if(order <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "Vector<double> get_parameters_order(const size_t&) method.\n"
               << "Order must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    size_t i;
    Vector<double> parameters;

    for(i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == order)
        {
            parameters = parameters_history[i];
            break;
        }
    }

    return(parameters);
}


/// Return a string with the stopping condition of the training depending on the training method.
/// @param results Results of the perform_training method.

string OrderSelectionAlgorithm::write_stopping_condition(const TrainingStrategy::Results& results) const
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
        case TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT:
        {
            return results.stochastic_gradient_descent_results_pointer->write_stopping_condition();
        }
        case TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION:
        {
            return results.adaptive_moment_estimation_results_pointer->write_stopping_condition();
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

void OrderSelectionAlgorithm::delete_selection_history()
{
    selection_error_history.set();
}


/// Delete the history of the loss values.

void OrderSelectionAlgorithm::delete_loss_history()
{
    loss_history.set();
}


/// Delete the history of the parameters of the trained neural networks.

void OrderSelectionAlgorithm::delete_parameters_history()
{
    parameters_history.set();
}


/// Checks that the different pointers needed for performing the order selection are not nullptr.

void OrderSelectionAlgorithm::check() const
{
    // Optimization algorithm stuff

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Neural network stuff

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to multilayer perceptron is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(multilayer_perceptron_pointer->is_empty())
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Multilayer Perceptron is empty.\n";

        throw logic_error(buffer.str());
    }


   if(multilayer_perceptron_pointer->get_layers_number() == 1)
   {
      buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
             << "void check() const method.\n"
             << "Number of layers in multilayer perceptron must be greater than 1.\n";

      throw logic_error(buffer.str());
   }


    // Data set stuff

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const Instances& instances = data_set_pointer->get_instances();

    const size_t selection_instances_number = instances.get_selection_instances_number();

    if(selection_instances_number == 0)
    {
        buffer << "OpenNN Exception: OrderSelectionAlgorithm class.\n"
               << "void check() const method.\n"
               << "Number of selection instances is zero.\n";

        throw logic_error(buffer.str());
    }

}


/// Return a string with the stopping condition of the OrderSelectionResults

string OrderSelectionAlgorithm::OrderSelectionResults::write_stopping_condition() const
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
    }

    return string();
}


/// Returns a string representation of the current order selection results structure.

string OrderSelectionAlgorithm::OrderSelectionResults::object_to_string() const
{
   ostringstream buffer;

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
          << write_elapsed_time(elapsed_time) << "\n";



   return(buffer.str());
}
}

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
