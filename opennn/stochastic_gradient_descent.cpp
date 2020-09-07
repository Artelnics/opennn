//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "stochastic_gradient_descent.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a stochastic gradient descent optimization algorithm not associated to any loss index object.
/// It also initializes the class members to their default values.

StochasticGradientDescent::StochasticGradientDescent()
    :OptimizationAlgorithm()
{
    set_default();
}


/// Loss index constructor.
/// It creates a stochastic gradient descent optimization algorithm associated to a loss index.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

StochasticGradientDescent::StochasticGradientDescent(LossIndex* new_loss_index_pointer)
    : OptimizationAlgorithm(new_loss_index_pointer)
{
    set_default();
}


/// Destructor.

StochasticGradientDescent::~StochasticGradientDescent()
{
}


/// Returns the initial learning rate.

const type& StochasticGradientDescent::get_initial_learning_rate() const
{
    return initial_learning_rate;
}


/// Returns the initial decay.

const type& StochasticGradientDescent::get_initial_decay() const
{
    return initial_decay;
}


/// Returns the momentum.

const type& StochasticGradientDescent::get_momentum() const
{
    return momentum;
}


///Returns true if nesterov is active, and false otherwise.

const bool& StochasticGradientDescent::get_nesterov() const
{
    return nesterov;
}


/// Returns the goal value for the loss.
/// This is used as a stopping criterion when training a neural network

const type& StochasticGradientDescent::get_loss_goal() const
{
    return training_loss_goal;
}


/// Returns the maximum training time.

const type& StochasticGradientDescent::get_maximum_time() const
{
    return maximum_time;
}


/// Returns true if the final model will be the neural network with the minimum selection error, false otherwise.

const bool& StochasticGradientDescent::get_choose_best_selection() const
{
    return choose_best_selection;
}


/// Returns true if the loss history vector is to be reserved, and false otherwise.

const bool& StochasticGradientDescent::get_reserve_training_error_history() const
{
    return reserve_training_error_history;
}


/// Returns true if the selection error history vector is to be reserved, and false otherwise.

const bool& StochasticGradientDescent::get_reserve_selection_error_history() const
{
    return reserve_selection_error_history;
}


const string& StochasticGradientDescent::get_hardware_use() const
{
    return hardware_use;
}


/// Sets a pointer to a loss index object to be associated to the gradient descent object.
/// It also sets that loss index to the learning rate algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void StochasticGradientDescent::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
    loss_index_pointer = new_loss_index_pointer;
}


void StochasticGradientDescent::set_default()
{
    // TRAINING OPERATORS

    initial_learning_rate = static_cast<type>(0.01);
    initial_decay = 0;
    momentum = 0;
    nesterov = false;

    // Stopping criteria

    training_loss_goal = 0;
    maximum_time = 3600.0;
    maximum_epochs_number = 1000;
    choose_best_selection = false;

    // TRAINING HISTORY

    reserve_training_error_history = true;
    reserve_selection_error_history = true;

    // UTILITIES

    display_period = 5;
}


/// Set the initial value for the learning rate. If dacay is not active learning rate will be constant
/// otherwise learning rate will decay over each update.
/// @param new_initial_learning_rate initial learning rate value.

void StochasticGradientDescent::set_initial_learning_rate(const type& new_learning_rate)
{
#ifdef __OPENNN_DEBUG__

    if(new_learning_rate <= static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_initial_learning_rate(const type&) method.\n"
               << "initial_learning_rate must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set learning rate

    initial_learning_rate = new_learning_rate;
}


/// Set the initial value for the decay.
/// @param new_initial_learning_rate initial value for the decay.

void StochasticGradientDescent::set_initial_decay(const type& new_dacay)
{
#ifdef __OPENNN_DEBUG__

    if(new_dacay < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_initial_decay(const type&) method.\n"
               << "new_dacay must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set  initial decay

    initial_decay = new_dacay;
}


/// Set a new value for momentum, this parameter accelerates SGD in the relevant direction
/// and dampens oscillations.
/// @param new_momentum initial value for the mometum.

void StochasticGradientDescent::set_momentum(const type& new_momentum)
{
#ifdef __OPENNN_DEBUG__

    if(new_momentum < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_momentum(const type&) method.\n"
               << "new_momentum must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set momentum

    momentum = new_momentum;
}


/// Set nesterov, boolean. Whether to apply Nesterov momentum.
/// @param new_momentum initial value for the mometum.

void StochasticGradientDescent::set_nesterov(const bool& new_nesterov_momentum)
{
    nesterov = new_nesterov_momentum;
}


/// Makes the training history of all variables to reseved or not in memory:
/// <ul>
/// <li> Parameters.
/// <li> Parameters norm.
/// <li> Loss.
/// <li> Gradient.
/// <li> Gradient norm.
/// <li> Selection loss.
/// <li> Learning rate.
/// <li> Elapsed_time.
/// </ul>
/// @param new_reserve_all_training_history True if the training history of all variables is to be reserved, false otherwise.

void StochasticGradientDescent::set_reserve_all_training_history(const bool& new_reserve_all_training_history)
{
    reserve_training_error_history = new_reserve_all_training_history;

    reserve_selection_error_history = new_reserve_all_training_history;
}


/// Set the a new maximum for the epochs number.
/// @param new_maximum_epochs number New maximum epochs number.

void StochasticGradientDescent:: set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{


#ifdef __OPENNN_DEBUG__

    if(new_maximum_epochs_number < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_maximum_epochs_number(const type&) method.\n"
               << "Maximum epochs number must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set maximum_epochs number

    maximum_epochs_number = new_maximum_epochs_number;
}


/// Sets a new goal value for the loss.
/// This is used as a stopping criterion when training a neural network
/// @param new_loss_goal Goal value for the loss.

void StochasticGradientDescent::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


/// Sets a new maximum training time.
/// @param new_maximum_time Maximum training time.

void StochasticGradientDescent::set_maximum_time(const type& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set maximum time

    maximum_time = new_maximum_time;
}


/// Makes the minimum selection error neural network of all the iterations to be returned or not.
/// @param new_choose_best_selection True if the final model will be the neural network with the minimum selection error, false otherwise.

void StochasticGradientDescent::set_choose_best_selection(const bool& new_choose_best_selection)
{
    choose_best_selection = new_choose_best_selection;
}


/// Makes the error history vector to be reseved or not in memory.
/// @param new_reserve_training_error_history True if the error history vector is to be reserved, false otherwise.

void StochasticGradientDescent::set_reserve_training_error_history(const bool& new_reserve_training_error_history)
{
    reserve_training_error_history = new_reserve_training_error_history;
}


/// Makes the selection error history to be reserved or not in memory.
/// This is a vector.
/// @param new_reserve_selection_error_history True if the selection error history is to be reserved, false otherwise.

void StochasticGradientDescent::set_reserve_selection_error_history(const bool& new_reserve_selection_error_history)
{
    reserve_selection_error_history = new_reserve_selection_error_history;
}


void StochasticGradientDescent::set_hardware_use(const string& new_hardware_use)
{
    hardware_use = new_hardware_use;
}


/// Sets a new number of iterations between the training showing progress.
/// @param new_display_period
/// Number of iterations between the training showing progress.

void StochasticGradientDescent::set_display_period(const Index& new_display_period)
{
#ifdef __OPENNN_DEBUG__

    if(new_display_period <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_display_period(const type&) method.\n"
               << "First learning rate must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    display_period = new_display_period;
}


void StochasticGradientDescent::update_iteration(const LossIndex::BackPropagation& back_propagation,
                      SGDOptimizationData& optimization_data)
{


    const type learning_rate = initial_learning_rate/(1 + optimization_data.iteration*initial_decay);

    optimization_data.parameters_increment.device(*thread_pool_device) = back_propagation.gradient*(-learning_rate);

    if(momentum > 0)
    {
        optimization_data.parameters_increment.device(*thread_pool_device) += momentum*optimization_data.last_parameters_increment;

        if(!nesterov)
        {
            optimization_data.parameters.device(*thread_pool_device) += optimization_data.parameters_increment;
        }
        else
        {
            optimization_data.nesterov_increment.device(*thread_pool_device)
                    = optimization_data.parameters_increment*momentum - back_propagation.gradient*learning_rate;

            optimization_data.parameters.device(*thread_pool_device) += optimization_data.nesterov_increment;
        }
    }
    else
    {
        optimization_data.parameters.device(*thread_pool_device) += optimization_data.parameters_increment;
    }

    optimization_data.last_parameters_increment = optimization_data.parameters_increment;

    optimization_data.iteration++;

}


/// Trains a neural network with an associated loss index,
/// according to the stochastic gradient descent method.
/// Training occurs according to the training parameters and stopping criteria.
/// It returns a results structure with the history and the final values of the reserved variables.

OptimizationAlgorithm::Results StochasticGradientDescent::perform_training()
{
    Results results;
#ifdef __OPENNN_DEBUG__

    check();

#endif
    // Start training

    if(display) cout << "Training with stochastic gradient descent...\n";

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const bool has_selection = data_set_pointer->has_selection();

    const Tensor<Index, 1> input_variables_indices = data_set_pointer->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set_pointer->get_target_variables_indices();

    Tensor<Index, 1> training_samples_indices = data_set_pointer->get_training_samples_indices();
    Tensor<Index, 1> selection_samples_indices = data_set_pointer->get_selection_samples_indices();

    const Index training_samples_number = data_set_pointer->get_training_samples_number();
    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();

    if(training_samples_number < batch_samples_number) batch_samples_number = training_samples_number;

    Tensor<Index, 1> batch_samples_indices(batch_samples_number);
    batch_samples_indices.setConstant(1);

    DataSet::Batch batch(batch_samples_number, data_set_pointer);

    const Index training_batches_number = training_samples_number/batch_samples_number;

    Tensor<Index, 2> training_batches(training_batches_number, batch_samples_number);

    DataSet::Batch selection_batch(selection_samples_number, data_set_pointer);
    selection_batch.fill(selection_samples_indices, input_variables_indices, target_variables_indices);

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    NeuralNetwork::ForwardPropagation forward_propagation(batch_samples_number, neural_network_pointer);

    NeuralNetwork::ForwardPropagation selection_forward_propagation(selection_samples_number, neural_network_pointer);

    // Loss index

    LossIndex::BackPropagation back_propagation(batch_samples_number, loss_index_pointer);

    LossIndex::BackPropagation selection_back_propagation(selection_samples_number, loss_index_pointer);

    type training_error = numeric_limits<type>::max();
    type training_loss = numeric_limits<type>::max();

    type old_selection_error = 0;
    Index selection_error_increases = 0;

    // Optimization algorithm

    SGDOptimizationData optimization_data(this);

    Tensor<type, 1> minimal_selection_parameters;
    type minimum_selection_error = numeric_limits<type>::max();

    bool stop_training = false;

    time_t beginning_time, current_time;
    time(&beginning_time);
    type elapsed_time = 0;

    results.resize_training_history(maximum_epochs_number + 1);
    if(has_selection) results.resize_selection_history(maximum_epochs_number + 1);

    bool shuffle = false;

    // Main loop

    for(Index epoch = 0; epoch <= epochs_number; epoch++)
    {
        training_batches = data_set_pointer->get_batches(training_samples_indices,
                                                         batch_samples_number,
                                                         shuffle);

        training_error = 0;
        training_loss = 0;

        optimization_data.iteration = 0;

        for(Index iteration = 0; iteration < training_batches_number; iteration++)
        {
            // Data set

            batch.fill(training_batches.chip(iteration,0),
                       input_variables_indices, target_variables_indices);

            // Neural network

            neural_network_pointer->forward_propagate(batch, forward_propagation);

            // Loss

            loss_index_pointer->back_propagate(batch, forward_propagation, back_propagation);

            training_error += back_propagation.error;
            training_loss += back_propagation.loss;

            // Optimization algorithm

            update_iteration(back_propagation, optimization_data);

            neural_network_pointer->set_parameters(optimization_data.parameters);
        }

        // Loss

        training_error /= static_cast<type>(training_batches_number);
        training_loss /= static_cast<type>(training_batches_number);

        if(has_selection)
        {
            neural_network_pointer->forward_propagate(selection_batch, selection_forward_propagation);

            loss_index_pointer->calculate_error(selection_batch, selection_forward_propagation, selection_back_propagation);

            if(epoch == 0)
            {
                minimum_selection_error = selection_back_propagation.error;
            }
            else if(selection_back_propagation.error > old_selection_error)
            {
                selection_error_increases++;
            }
            else if(selection_back_propagation.error <= minimum_selection_error)
            {
                minimum_selection_error = selection_back_propagation.error;
                minimal_selection_parameters = optimization_data.parameters;
            }
        }

        // Training history loss index

        if(reserve_training_error_history) results.training_error_history(epoch) = training_error;

        if(has_selection && reserve_selection_error_history) results.selection_error_history(epoch) = selection_back_propagation.error;

        // Stopping criteria

        time(&current_time);

        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        if(training_loss <= training_loss_goal)
        {
            if(display) cout << "Epoch " << epoch+1 << ": Training loss goal reached.\n";

            stop_training = true;

            results.stopping_condition = LossGoal;
        }

        if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch+1 << ": Maximum number of epochs reached.\n";

            stop_training = true;

            results.stopping_condition = MaximumEpochsNumber;
        }

        else if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch+1 << ": Maximum training time reached.\n";

            stop_training = true;

            results.stopping_condition = MaximumTime;
        }

        if(epoch != 0 && epoch % save_period == 0)
        {
            neural_network_pointer->save(neural_network_file_name);
        }

        if(stop_training)
        {
            if(display)
            {
                cout << "Training error: " << training_error << "\n"
                     << "Batch size: " << batch_samples_number << "\n"
                     << "Elapsed time: " << write_elapsed_time(elapsed_time)<<"\n";

                if(has_selection) cout << "Selection error: " << selection_back_propagation.error << endl << endl;
            }

            results.resize_training_error_history(epoch+1);
            if(has_selection) results.resize_selection_error_history(epoch+1);

            results.final_parameters = optimization_data.parameters;
            results.final_training_error = training_error;
            results.final_selection_error = selection_back_propagation.error;
            results.elapsed_time = elapsed_time;
            results.epochs_number = epoch;

            break;
        }
        else if((display && epoch == 0) || (display && (epoch+1) % display_period == 0))
        {
            cout << "Epoch " << epoch+1 << "/"<<maximum_epochs_number << ":\n"
                 << "Training error: " << training_error << "\n"
                 << "Batch size: " << batch_samples_number << "\n"
                 << "Elapsed time: " << write_elapsed_time(elapsed_time)<<"\n";

            if(has_selection) cout << "Selection error: " << selection_back_propagation.error << endl << endl;
        }

        old_selection_error = selection_back_propagation.error;

        if(stop_training) break;
    }

    if(has_selection && choose_best_selection)
    {
        optimization_data.parameters = minimal_selection_parameters;

        neural_network_pointer->set_parameters(optimization_data.parameters);

        results.final_selection_error = selection_back_propagation.error;
    }

    return results;
}


void StochasticGradientDescent::perform_training_void()
{
    perform_training();
}


string StochasticGradientDescent::write_optimization_algorithm_type() const
{
    return "STOCHASTIC_GRADIENT_DESCENT";
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> StochasticGradientDescent::to_string_matrix() const
{
    Tensor<string, 2> labels_values(9, 2);

    // Initial learning rate

    labels_values(0,0) = "Inital learning rate";

    labels_values(0,1) = std::to_string(initial_learning_rate);

    // Initial decay

    labels_values(1,0) = "Inital decay";

    labels_values(1,1) = std::to_string(initial_decay);

    // Momentum

    labels_values(2,0) = "Apply momentum";

    if(momentum > 0)
    {
        labels_values(2,1) = "true";
    }
    else
    {
        labels_values(2,1) = "false";
    }

    // Training loss goal

    labels_values(3,0) = "Training loss goal";

    labels_values(3,1) = std::to_string(training_loss_goal);

    // Maximum epochs number

    labels_values(4,0) = "Maximum epochs number";

    labels_values(4,1) = std::to_string(maximum_epochs_number);

    // Maximum time

    labels_values(5,0) = "Maximum time";

    labels_values(5,1) = std::to_string(maximum_time);

    // Batch samples number

    labels_values(6,0) = "Batch samples number";

    labels_values(6,1) = std::to_string(batch_samples_number);

    // Reserve training error history

    labels_values(7,0) = "Reserve training error history";

    if(reserve_training_error_history)
    {
        labels_values(7,1) = "true";
    }
    else
    {
        labels_values(7,1) = "false";
    }

    // Reserve selection error history

    labels_values(8,0) = "Reserve selection error history";

    if(reserve_training_error_history)
    {
        labels_values(8,1) = "true";
    }
    else
    {
        labels_values(8,1) = "false";
    }

    return labels_values;
}


/// Serializes the gradient descent object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void StochasticGradientDescent::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("StochasticGradientDescent");

    // Batch size

    file_stream.OpenElement("BatchSize");

    buffer.str("");
    buffer << batch_samples_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Apply momentum

    file_stream.OpenElement("ApplyMomentum");

    buffer.str("");
    buffer << (momentum > static_cast<type>(0.0));

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Return minimum selection error neural network

    file_stream.OpenElement("ReturnMinimumSelectionErrorNN");

    buffer.str("");
    buffer << choose_best_selection;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Loss goal

    file_stream.OpenElement("LossGoal");

    buffer.str("");
    buffer << training_loss_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum iterations number

    file_stream.OpenElement("MaximumEpochsNumber");

    buffer.str("");
    buffer << maximum_epochs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve training error history

    file_stream.OpenElement("ReserveTrainingErrorHistory");

    buffer.str("");
    buffer << reserve_training_error_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection error history

    file_stream.OpenElement("ReserveSelectionErrorHistory");

    buffer.str("");
    buffer << reserve_selection_error_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Hardware use

    file_stream.OpenElement("HardwareUse");

    buffer.str("");
    buffer << hardware_use;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // End element

    file_stream.CloseElement();
}


void StochasticGradientDescent::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("StochasticGradientDescent");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Stochastic gradient descent element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Batch size

    const tinyxml2::XMLElement* batch_size_element = root_element->FirstChildElement("BatchSize");

    if(batch_size_element)
    {
        const Index new_batch_size = static_cast<Index>(atoi(batch_size_element->GetText()));

        try
        {
            set_batch_samples_number(new_batch_size);

//            const Index training_samples_number = loss_index_pointer->get_data_set_pointer()->get_training_samples_number();

//            if(new_batch_size > training_samples_number || new_batch_size == 0)
//            {
//                set_batch_samples_number(training_samples_number);
//            }
//            else
//            {
//                set_batch_samples_number(new_batch_size);
//            }
        }
        catch(const logic_error& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Momentum

    const tinyxml2::XMLElement* apply_momentum_element = root_element->FirstChildElement("ApplyMomentum");

    if(batch_size_element)
    {
        string new_apply_momentum_state = apply_momentum_element->GetText();

        try
        {
            if(new_apply_momentum_state != "0")
            {
                set_momentum(static_cast<type>(0.9));
            }
            else
            {
                set_momentum(static_cast<type>(0.0));
            }
        }
        catch(const logic_error& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Return minimum selection error neural network

    const tinyxml2::XMLElement* choose_best_selection_element = root_element->FirstChildElement("ReturnMinimumSelectionErrorNN");

    if(choose_best_selection_element)
    {
        string new_choose_best_selection = choose_best_selection_element->GetText();

        try
        {
            set_choose_best_selection(new_choose_best_selection != "0");
        }
        catch(const logic_error& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Loss goal
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("LossGoal");

        if(element)
        {
            const type new_loss_goal = static_cast<type>(atof(element->GetText()));

            try
            {
                set_loss_goal(new_loss_goal);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum epochs number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumEpochsNumber");

        if(element)
        {
            const Index new_maximum_epochs_number = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_maximum_epochs_number(new_maximum_epochs_number);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum time
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumTime");

        if(element)
        {
            const type new_maximum_time = static_cast<type>(atof(element->GetText()));

            try
            {
                set_maximum_time(new_maximum_time);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve training error history
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveTrainingErrorHistory");

        if(element)
        {
            const string new_reserve_training_error_history = element->GetText();

            try
            {
                set_reserve_training_error_history(new_reserve_training_error_history != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve selection error history
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

        if(element)
        {
            const string new_reserve_selection_error_history = element->GetText();

            try
            {
                set_reserve_selection_error_history(new_reserve_selection_error_history != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Hardware use
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("HardwareUse");

        if(element)
        {
            const string new_hardware_use = element->GetText();

            try
            {
                set_hardware_use(new_hardware_use);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }
}

}


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
