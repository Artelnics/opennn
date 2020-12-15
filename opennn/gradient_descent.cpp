//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R A D I E N T   D E S C E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "gradient_descent.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a gradient descent optimization algorithm not associated to any loss index object.
/// It also initializes the class members to their default values.

GradientDescent::GradientDescent()
    : OptimizationAlgorithm()
{
    set_default();
}


/// Loss index constructor.
/// It creates a gradient descent optimization algorithm associated to a loss index.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

GradientDescent::GradientDescent(LossIndex* new_loss_index_pointer)
    : OptimizationAlgorithm(new_loss_index_pointer)
{
    learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);

    set_default();
}


/// Destructor.

GradientDescent::~GradientDescent()
{
}


/// Returns a constant reference to the learning rate algorithm object inside the gradient descent object.

const LearningRateAlgorithm& GradientDescent::get_learning_rate_algorithm() const
{
    return learning_rate_algorithm;
}


/// Returns a pointer to the learning rate algorithm object inside the gradient descent object.

LearningRateAlgorithm* GradientDescent::get_learning_rate_algorithm_pointer()
{
    return &learning_rate_algorithm;
}


/// Returns the hardware used. Default: Multi-core

string GradientDescent::get_hardware_use() const
{
    return hardware_use;
}


/// Returns the minimum norm of the parameter increment vector used as a stopping criteria when training.

const type& GradientDescent::get_minimum_parameters_increment_norm() const
{
    return minimum_parameters_increment_norm;
}


/// Returns the minimum loss improvement during training.

const type& GradientDescent::get_minimum_loss_decrease() const
{
    return minimum_loss_decrease;
}


/// Returns the goal value for the loss.
/// This is used as a stopping criterion when training a neural network.

const type& GradientDescent::get_loss_goal() const
{
    return training_loss_goal;
}


/// Returns the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a neural network.

const type& GradientDescent::get_gradient_norm_goal() const
{
    return gradient_norm_goal;
}


/// Returns the maximum number of selection error increases during the training process.

const Index& GradientDescent::get_maximum_selection_error_increases() const
{
    return maximum_selection_error_increases;
}


/// Returns the maximum number of iterations for training.

const Index& GradientDescent::get_maximum_epochs_number() const
{
    return maximum_epochs_number;
}


/// Returns the maximum training time.

const type& GradientDescent::get_maximum_time() const
{
    return maximum_time;
}


/// Returns true if the final model will be the neural network with the minimum selection error, false otherwise.

const bool& GradientDescent::get_choose_best_selection() const
{
    return choose_best_selection;
}


/// Returns true if the loss history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_training_error_history() const
{
    return reserve_training_error_history;
}


/// Returns true if the selection error history vector is to be reserved, and false otherwise.

const bool& GradientDescent::get_reserve_selection_error_history() const
{
    return reserve_selection_error_history;
}


/// Sets a pointer to a loss index object to be associated to the gradient descent object.
/// It also sets that loss index to the learning rate algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void GradientDescent::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
    loss_index_pointer = new_loss_index_pointer;

    learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);
}


void GradientDescent::set_default()
{
    // Stopping criteria

    minimum_parameters_increment_norm = static_cast<type>(1.0e-3);

    minimum_loss_decrease = static_cast<type>(0.0);

    training_loss_goal = 0;
    gradient_norm_goal = 0;
    maximum_selection_error_increases = 100;

    maximum_epochs_number = 1000;
    maximum_time = 3600;

    choose_best_selection = false;

    // TRAINING HISTORY

    reserve_training_error_history = true;
    reserve_selection_error_history = true;

    // UTILITIES

    display_period = 5;
}


/// Makes the training history of all variables to reseved or not in memory:
/// <ul>
/// <li> Parameters.
/// <li> Parameters norm.
/// <li> Loss.
/// <li> Gradient.
/// <li> Gradient norm.
/// <li> Selection loss.
/// <li> Training direction.
/// <li> Training direction norm.
/// <li> Learning rate.
/// </ul>
/// @param new_reserve_all_training_history True if the training history of all variables is to be reserved,
/// false otherwise.

void GradientDescent::set_reserve_all_training_history(const bool& new_reserve_all_training_history)
{
    reserve_training_error_history = new_reserve_all_training_history;

    reserve_selection_error_history = new_reserve_all_training_history;
}

/// Set the a new maximum for the epochs number.
/// @param new_maximum_epochs number New maximum epochs number.

void GradientDescent::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{


#ifdef __OPENNN_DEBUG__

    if(new_maximum_epochs_number < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void set_maximum_epochs_number(const type&) method.\n"
               << "Maximum epochs number must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set maximum_epochs number

    maximum_epochs_number = new_maximum_epochs_number;
}


/// Sets a new value for the minimum parameters increment norm stopping criterion.
/// @param new_minimum_parameters_increment_norm Value of norm of parameters increment norm used to stop training.

void GradientDescent::set_minimum_parameters_increment_norm(const type& new_minimum_parameters_increment_norm)
{


#ifdef __OPENNN_DEBUG__

    if(new_minimum_parameters_increment_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void new_minimum_parameters_increment_norm(const type&) method.\n"
               << "Minimum parameters increment norm must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set error learning rate

    minimum_parameters_increment_norm = new_minimum_parameters_increment_norm;
}


/// Sets a new minimum loss improvement during training.
/// @param new_minimum_loss_decrease Minimum improvement in the loss between two iterations.

void GradientDescent::set_minimum_loss_decrease(const type& new_minimum_loss_decrease)
{


#ifdef __OPENNN_DEBUG__

    if(new_minimum_loss_decrease < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void set_minimum_loss_decrease(const type&) method.\n"
               << "Minimum loss improvement must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set minimum loss improvement

    minimum_loss_decrease = new_minimum_loss_decrease;
}


/// Sets a new goal value for the loss.
/// This is used as a stopping criterion when training a neural network.
/// @param new_loss_goal Goal value for the loss.

void GradientDescent::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


/// Sets a new the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a neural network.
/// @param new_gradient_norm_goal Goal value for the norm of the error function gradient.

void GradientDescent::set_gradient_norm_goal(const type& new_gradient_norm_goal)
{


#ifdef __OPENNN_DEBUG__

    if(new_gradient_norm_goal < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void set_gradient_norm_goal(const type&) method.\n"
               << "Gradient norm goal must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set gradient norm goal

    gradient_norm_goal = new_gradient_norm_goal;
}


/// Sets a new maximum number of selection error increases.
/// @param new_maximum_selection_error_increases Maximum number of epochs in which the selection evalutation
/// increases.

void GradientDescent::set_maximum_selection_error_increases(const Index& new_maximum_selection_error_increases)
{
    maximum_selection_error_increases = new_maximum_selection_error_increases;
}


/// Sets a new maximum training time.
/// @param new_maximum_time Maximum training time.

void GradientDescent::set_maximum_time(const type& new_maximum_time)
{


#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set maximum time

    maximum_time = new_maximum_time;
}


/// Makes the minimum selection error neural network of all the iterations to be returned or not.
/// @param new_choose_best_selection True if the final model will be the neural network
///  with the minimum selection error, false otherwise.

void GradientDescent::set_choose_best_selection(const bool& new_choose_best_selection)
{
    choose_best_selection = new_choose_best_selection;
}


/// Makes the error history vector to be reseved or not in memory.
/// @param new_reserve_training_error_history True if the loss history vector is to be reserved, false otherwise.

void GradientDescent::set_reserve_training_error_history(const bool& new_reserve_training_error_history)
{
    reserve_training_error_history = new_reserve_training_error_history;
}


/// Makes the selection error history to be reserved or not in memory.
/// This is a vector.
/// @param new_reserve_selection_error_history True if the selection error history is to be reserved, false otherwise.

void GradientDescent::set_reserve_selection_error_history(const bool& new_reserve_selection_error_history)
{
    reserve_selection_error_history = new_reserve_selection_error_history;
}


/// Returns the gradient descent training direction,
/// which is the negative of the normalized gradient.
/// @param gradient Loss index gradient.

void GradientDescent::calculate_training_direction(const Tensor<type, 1>& gradient, Tensor<type, 1>& training_direction) const
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "Tensor<type, 1> calculate_training_direction(const Tensor<type, 1>&) const method.\n"
               << "Loss index pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    const Index gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "Tensor<type, 1> calculate_training_direction(const Tensor<type, 1>&) const method.\n"
               << "Size of gradient(" << gradient_size
               << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    training_direction.device(*thread_pool_device) = -gradient;
}



////// \brief GradientDescent::update_epoch
////// \param batch
////// \param forward_propagation
////// \param back_propagation
////// \param optimization_data
void GradientDescent::update_epoch(
        const DataSet::Batch& batch,
        NeuralNetwork::ForwardPropagation& forward_propagation,
        LossIndex::BackPropagation& back_propagation,
        GDOptimizationData& optimization_data)
{

    calculate_training_direction(back_propagation.gradient, optimization_data.training_direction);

    if(is_zero(optimization_data.training_direction))
        throw logic_error("Training direction is zero");

    // Training slope    

    optimization_data.training_slope.device(*thread_pool_device)
            = back_propagation.gradient.contract(optimization_data.training_direction, AT_B);

    if(optimization_data.training_slope(0) >= static_cast<type>(0.0))
        throw logic_error("Training slope is equal or greater than zero");

    // Get initial learning_rate

    optimization_data.initial_learning_rate = 0;

    optimization_data.epoch == 0
            ? optimization_data.initial_learning_rate = first_learning_rate
            : optimization_data.initial_learning_rate = optimization_data.old_learning_rate;


    pair<type,type> directional_point = learning_rate_algorithm.calculate_directional_point(
                            batch,
                            forward_propagation,
                            back_propagation,
                            optimization_data);

    optimization_data.learning_rate = directional_point.first;

//    if(abs(optimization_data.learning_rate) < numeric_limits<type>::min())
//        throw logic_error("Learning rate is zero");

    optimization_data.parameters_increment.device(*thread_pool_device)
            = optimization_data.training_direction*optimization_data.learning_rate;

    optimization_data.old_parameters = optimization_data.parameters;

    optimization_data.parameters += optimization_data.parameters_increment;

    optimization_data.old_gradient = back_propagation.gradient;

    optimization_data.parameters_increment_norm = l2_norm(optimization_data.parameters_increment);

    optimization_data.old_learning_rate = optimization_data.learning_rate;

    optimization_data.old_training_loss = back_propagation.loss;
}


/// Trains a neural network with an associated loss index,
/// according to the gradient descent method.
/// Training occurs according to the training parameters and stopping criteria.
/// It returns a results structure with the history and the final values of the reserved variables.

OptimizationAlgorithm::Results GradientDescent::perform_training()
{
    Results results;

#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Start training

    if(display) cout << "Training with gradient descent...\n";

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Index training_samples_number = data_set_pointer->get_training_samples_number();
    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();

    const bool has_selection = data_set_pointer->has_selection();

    Tensor<Index, 1> training_samples_indices = data_set_pointer->get_training_samples_indices();
    Tensor<Index, 1> selection_samples_indices = data_set_pointer->get_selection_samples_indices();
    Tensor<Index, 1> inputs_indices = data_set_pointer->get_input_variables_indices();
    Tensor<Index, 1> target_indices = data_set_pointer->get_target_variables_indices();

    DataSet::Batch training_batch(training_samples_number, data_set_pointer);
    DataSet::Batch selection_batch(selection_samples_number, data_set_pointer);

    training_batch.fill(training_samples_indices, inputs_indices, target_indices);
    selection_batch.fill(selection_samples_indices, inputs_indices, target_indices);

    training_samples_indices.resize(0);
    selection_samples_indices.resize(0);
    inputs_indices.resize(0);
    target_indices.resize(0);

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    type parameters_norm = 0;

    NeuralNetwork::ForwardPropagation training_forward_propagation(training_samples_number, neural_network_pointer);
    NeuralNetwork::ForwardPropagation selection_forward_propagation(selection_samples_number, neural_network_pointer);

    // Loss index

    type old_selection_error = 0;

    type training_loss_decrease = -numeric_limits<type>::max();

    type gradient_norm = 0;

    LossIndex::BackPropagation training_back_propagation(training_samples_number, loss_index_pointer);
    LossIndex::BackPropagation selection_back_propagation(selection_samples_number, loss_index_pointer);

    // Learning rate

    bool stop_training = false;

    // Optimization algorithm

    GDOptimizationData optimization_data(this);

    Index selection_error_increases = 0;

    type parameters_increment_norm = 0;

    type minimum_selection_error = numeric_limits<type>::max();

    Tensor<type, 1> minimal_selection_parameters;

    results.resize_training_history(maximum_epochs_number+1);

    if(has_selection) results.resize_selection_history(maximum_epochs_number+1);

    // Main loop

    time_t beginning_time, current_time;
    time(&beginning_time);
    type elapsed_time = 0;

    for(Index epoch = 0; epoch < maximum_epochs_number; epoch++)
    {
        optimization_data.epoch = epoch;

        // Neural network

        parameters_norm = l2_norm(optimization_data.parameters);

        neural_network_pointer->forward_propagate(training_batch, training_forward_propagation);

        // Loss index

        loss_index_pointer->back_propagate(training_batch, training_forward_propagation, training_back_propagation);

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

        if(epoch != 0) training_loss_decrease = training_back_propagation.loss - optimization_data.old_training_loss;

        gradient_norm = l2_norm(training_back_propagation.gradient);

        if(gradient_norm < numeric_limits<type>::min()) throw logic_error("Gradient is zero");

        // Optimization algorithm

        update_epoch(training_batch, training_forward_propagation, training_back_propagation, optimization_data);

        neural_network_pointer->set_parameters(optimization_data.parameters);

        // Elapsed time

        time(&current_time);
        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        // Training history loss index

        if(reserve_training_error_history) results.training_error_history(epoch) = training_back_propagation.error;

        if(has_selection && reserve_selection_error_history) results.selection_error_history(epoch) = selection_back_propagation.error;

        // Stopping Criteria

        if(optimization_data.parameters_increment_norm <= minimum_parameters_increment_norm)
        {
            if(display)
            {
                cout << "Epoch " << epoch+1 << ": Minimum parameters increment norm reached.\n";
                cout << "Parameters increment norm: " << parameters_increment_norm << endl;
            }

            stop_training = true;

            results.stopping_condition = MinimumParametersIncrementNorm;
        }

        else if(epoch != 0 && abs(training_loss_decrease) <= minimum_loss_decrease)
        {
            if(display)
            {
                cout << "Epoch " << epoch+1 << ": Minimum loss decrease (" << minimum_loss_decrease << ") reached.\n"
                     << "Loss decrease: " << training_loss_decrease << endl;
            }

            stop_training = true;

            results.stopping_condition = MinimumLossDecrease;
        }

        else if(training_back_propagation.loss <= training_loss_goal)
        {
            if(display) cout << "Epoch " << epoch+1 << ": Loss goal reached.\n";

            stop_training = true;

            results.stopping_condition = LossGoal;
        }

        else if(selection_error_increases >= maximum_selection_error_increases)
        {
            if(display)
            {
                cout << "Epoch " << epoch+1 << ": Maximum selection error increases reached.\n"
                     << "Selection error increases: " << selection_error_increases << endl;
            }

            stop_training = true;

            results.stopping_condition = MaximumSelectionErrorIncreases;
        }

        else if(gradient_norm <= gradient_norm_goal)
        {
            if(display) cout << "Epoch " << epoch+1 << ": Gradient norm goal reached.\n";

            stop_training = true;

            results.stopping_condition = GradientNormGoal;
        }

        else if(epoch == maximum_epochs_number)
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
                cout << "Parameters norm: " << parameters_norm << "\n"
                     << "Training error: " << training_back_propagation.error << "\n"
                     << "Gradient norm: " << gradient_norm << "\n"
                     << "Learning rate: " << optimization_data.learning_rate << "\n"
                     << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

                if(has_selection) cout << "Selection error: " << selection_back_propagation.error << endl;
            }

            results.resize_training_error_history(epoch+1);
            if(has_selection) results.resize_selection_error_history(epoch+1);

            results.final_parameters = optimization_data.parameters;

            results.final_parameters_norm = parameters_norm;

            results.final_training_error = training_back_propagation.error;

            results.final_selection_error = selection_back_propagation.error;

            results.final_gradient_norm = gradient_norm;

            results.elapsed_time = write_elapsed_time(elapsed_time);

            results.epochs_number = epoch;

            break;
        }
        else if((display && epoch == 0) || (display && (epoch+1) % display_period == 0))
        {
            cout << "Epoch " << epoch+1 << ";\n"
                 << "Parameters norm: " << parameters_norm << "\n"
                 << "Training error: " << training_back_propagation.error << "\n"
                 << "Gradient norm: " << gradient_norm << "\n"
                 << "Learning rate: " << optimization_data.learning_rate << "\n"
                 << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

            if(has_selection) cout << "Selection error: " << selection_back_propagation.error << endl;
        }

        // Update stuff

        old_selection_error = selection_back_propagation.error;

        if(stop_training) break;
    }

    if(choose_best_selection)
    {
        neural_network_pointer->set_parameters(minimal_selection_parameters);
    }

    return results;
}


void GradientDescent::perform_training_void()
{
    perform_training();
}


string GradientDescent::write_optimization_algorithm_type() const
{
    return "GRADIENT_DESCENT";
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> GradientDescent::to_string_matrix() const
{
    Tensor<string, 2> labels_values(11, 2);

    // Learning rate method

    labels_values(0,0) = "Learning rate method";

    const string learning_rate_method = learning_rate_algorithm.write_learning_rate_method();

    labels_values(0,1) = learning_rate_method;

    // Loss tolerance

    labels_values(1,0) = "Learning rate tolerance";

    labels_values(1,1) = std::to_string(learning_rate_algorithm.get_learning_rate_tolerance());

    // Minimum parameters increment norm

    labels_values(2,0) = "Minimum parameters increment norm";

    labels_values(2,1) = std::to_string(minimum_parameters_increment_norm);

    // Minimum loss decrease

    labels_values(3,0) = "Minimum loss decrease";

    labels_values(3,1) = std::to_string(minimum_loss_decrease);

    // Loss goal

    labels_values(4,0) = "Loss goal";

    labels_values(4,1) = std::to_string(training_loss_goal);

    // Gradient norm goal

    labels_values(5,0) = "Gradient norm goal";

    labels_values(5,1) = std::to_string(gradient_norm_goal);

    // Maximum selection error increases

    labels_values(6,0) = "Maximum selection error increases";

    labels_values(6,1) = std::to_string(maximum_selection_error_increases);

    // Maximum epochs number

    labels_values(7,0) = "Maximum epochs number";

    labels_values(7,1) = std::to_string(maximum_epochs_number);

    // Maximum time

    labels_values(8,0) = "Maximum time";

    labels_values(8,1) = std::to_string(maximum_time);

    // Reserve training error history

    labels_values(9,0) = "Reserve training error history";

    if(reserve_training_error_history)
    {
        labels_values(9,1) = "true";
    }
    else
    {
        labels_values(9,1) = "false";
    }

    // Reserve selection error history

    labels_values(10,0) = "Reserve selection error history";

    if(reserve_selection_error_history)
    {
        labels_values(10,1) = "true";
    }
    else
    {
        labels_values(10,1) = "false";
    }

    return labels_values;
}


/// Serializes the gradient descent object into a XML document of the TinyXML library
/// without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GradientDescent::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Learning rate algorithm

    file_stream.OpenElement("GradientDescent");

    learning_rate_algorithm.write_XML(file_stream);

    // Return minimum selection error neural network

    file_stream.OpenElement("ReturnMinimumSelectionErrorNN");

    buffer.str("");
    buffer << choose_best_selection;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum parameters increment norm

    file_stream.OpenElement("MinimumParametersIncrementNorm");

    buffer.str("");
    buffer << minimum_parameters_increment_norm;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum loss decrease

    file_stream.OpenElement("MinimumLossDecrease");

    buffer.str("");
    buffer << minimum_loss_decrease;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Loss goal

    file_stream.OpenElement("LossGoal");

    buffer.str("");
    buffer << training_loss_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Gradient norm goal

    file_stream.OpenElement("GradientNormGoal");

    buffer.str("");
    buffer << gradient_norm_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum selection error increases

    file_stream.OpenElement("MaximumSelectionErrorIncreases");

    buffer.str("");
    buffer << maximum_selection_error_increases;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum epochs number

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

    file_stream.CloseElement();
}


void GradientDescent::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GradientDescent");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Gradient descent element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Learning rate algorithm
    {
        const tinyxml2::XMLElement* learning_rate_algorithm_element
                = root_element->FirstChildElement("LearningRateAlgorithm");

        if(learning_rate_algorithm_element)
        {
            tinyxml2::XMLDocument learning_rate_algorithm_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = learning_rate_algorithm_element->DeepClone(&learning_rate_algorithm_document);

            learning_rate_algorithm_document.InsertFirstChild(element_clone);

            learning_rate_algorithm.from_XML(learning_rate_algorithm_document);
        }
    }

    // Return minimum selection error neural network

    const tinyxml2::XMLElement* choose_best_selection_element
            = root_element->FirstChildElement("ReturnMinimumSelectionErrorNN");

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

    // Minimum parameters increment norm
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumParametersIncrementNorm");

        if(element)
        {
            const type new_minimum_parameters_increment_norm = static_cast<type>(atof(element->GetText()));

            try
            {
                set_minimum_parameters_increment_norm(new_minimum_parameters_increment_norm);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Minimum loss decrease
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumLossDecrease");

        if(element)
        {
            cout << "MinimumLossDecrease" << endl;
            const type new_minimum_loss_decrease = static_cast<type>(atof(element->GetText()));

            try
            {
                set_minimum_loss_decrease(new_minimum_loss_decrease);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
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

    // Gradient norm goal
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("GradientNormGoal");

        if(element)
        {
            const type new_gradient_norm_goal = static_cast<type>(atof(element->GetText()));

            try
            {
                set_gradient_norm_goal(new_gradient_norm_goal);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum selection error increases
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumSelectionErrorIncreases");

        if(element)
        {
            const Index new_maximum_selection_error_increases = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_maximum_selection_error_increases(new_maximum_selection_error_increases);
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
            cout << "ReserveTrainingErrorHistory" << endl;
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

