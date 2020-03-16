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


/// XML constructor.
/// It creates a gradient descent optimization algorithm not associated to any loss index object.
/// It also loads the class members from a XML document.
/// @param document TinyXML document with the members of a gradient descent object.

StochasticGradientDescent::StochasticGradientDescent(const tinyxml2::XMLDocument& document)
    : OptimizationAlgorithm(document)
{
    set_default();

    from_XML(document);
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


/// Returns the minimum value for the norm of the parameters vector at wich a warning message is
/// written to the screen.

const type& StochasticGradientDescent::get_warning_parameters_norm() const
{
    return warning_parameters_norm;
}


/// Returns the minimum value for the norm of the gradient vector at wich a warning message is written
/// to the screen.

const type& StochasticGradientDescent::get_warning_gradient_norm() const
{
    return warning_gradient_norm;
}


/// Returns the value for the norm of the parameters vector at wich an error message is
/// written to the screen and the program exits.

const type& StochasticGradientDescent::get_error_parameters_norm() const
{
    return error_parameters_norm;
}


/// Returns the value for the norm of the gradient vector at wich an error message is written
/// to the screen and the program exits.

const type& StochasticGradientDescent::get_error_gradient_norm() const
{
    return error_gradient_norm;
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

    // TRAINING PARAMETERS

    warning_parameters_norm = 1.0e6;
    warning_gradient_norm = 1.0e6;

    error_parameters_norm = 1.0e9;
    error_gradient_norm = 1.0e9;

    // Stopping criteria

    training_loss_goal = 0;
    maximum_time = 1000.0;
    maximum_epochs_number = 1000;
    choose_best_selection = false;

    // TRAINING HISTORY

    reserve_training_error_history = true;
    reserve_selection_error_history = false;

    // UTILITIES

    display = true;
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


/// Sets a new value for the parameters vector norm at which a warning message is written to the
/// screen.
/// @param new_warning_parameters_norm Warning norm of parameters vector value.

void StochasticGradientDescent::set_warning_parameters_norm(const type& new_warning_parameters_norm)
{


#ifdef __OPENNN_DEBUG__

    if(new_warning_parameters_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_warning_parameters_norm(const type&) method.\n"
               << "Warning parameters norm must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set warning parameters norm

    warning_parameters_norm = new_warning_parameters_norm;
}


/// Sets a new value for the gradient vector norm at which
/// a warning message is written to the screen.
/// @param new_warning_gradient_norm Warning norm of gradient vector value.

void StochasticGradientDescent::set_warning_gradient_norm(const type& new_warning_gradient_norm)
{
#ifdef __OPENNN_DEBUG__

    if(new_warning_gradient_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_warning_gradient_norm(const type&) method.\n"
               << "Warning gradient norm must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set warning gradient norm

    warning_gradient_norm = new_warning_gradient_norm;
}


/// Sets a new value for the parameters vector norm at which an error message is written to the
/// screen and the program exits.
/// @param new_error_parameters_norm Error norm of parameters vector value.

void StochasticGradientDescent::set_error_parameters_norm(const type& new_error_parameters_norm)
{
#ifdef __OPENNN_DEBUG__

    if(new_error_parameters_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_error_parameters_norm(const type&) method.\n"
               << "Error parameters norm must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set error parameters norm

    error_parameters_norm = new_error_parameters_norm;
}


/// Sets a new value for the gradient vector norm at which an error message is written to the screen
/// and the program exits.
/// @param new_error_gradient_norm Error norm of gradient vector value.

void StochasticGradientDescent::set_error_gradient_norm(const type& new_error_gradient_norm)
{


#ifdef __OPENNN_DEBUG__

    if(new_error_gradient_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_error_gradient_norm(const type&) method.\n"
               << "Error gradient norm must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set error gradient norm

    error_gradient_norm = new_error_gradient_norm;
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
               << "First training rate must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    display_period = new_display_period;
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

    Tensor<Index, 1> training_instances_indices = data_set_pointer->get_training_instances_indices();

    const Index training_instances_number = data_set_pointer->get_training_instances_number();
    const Index selection_instances_number = data_set_pointer->get_selection_instances_number();

    Tensor<Index, 1> batch_instances_indices(batch_instances_number);
    batch_instances_indices.setConstant(1);

    DataSet::Batch batch(batch_instances_number, data_set_pointer);

    const Index training_batches_number = training_instances_number/batch_instances_number;
    const Index selection_batches_number = selection_instances_number/batch_instances_number;

    Tensor<Index, 2> training_batches(training_batches_number, batch_instances_number);
    Tensor<Index, 2> selection_batches(selection_batches_number, batch_instances_number);

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    NeuralNetwork::ForwardPropagation forward_propagation(batch_instances_number, neural_network_pointer);

    // Loss index

    LossIndex::BackPropagation back_propagation(batch_instances_number, loss_index_pointer);

    type training_loss = numeric_limits<type>::max();

    type selection_error = numeric_limits<type>::max();

    // Optimization algorithm

    OptimizationData optimization_data(this);

    Tensor<type, 1> minimal_selection_parameters(parameters_number);
    type minimum_selection_error = numeric_limits<type>::max();

    bool stop_training = false;

    time_t beginning_time, current_time;
    time(&beginning_time);
    type elapsed_time = 0;

    results.resize_training_history(maximum_epochs_number + 1);

    bool shuffle = false;

//    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Main loop

    for(Index epoch = 1; epoch <= epochs_number; epoch++)
    {
//        random_shuffle(training_instances_indices.data(),
//                       training_instances_indices.data() + training_instances_indices.size());

//        training_batches = data_set_pointer->get_training_batches(batch_instances_number, shuffle);

        training_loss = 0;

        for(Index iteration = 0; iteration < training_batches_number; iteration++)
        {
            // Data set

//            batch_instances_indices = training_batches.chip(iteration, 0);
//            batch_instances_indices = training_batches.slice({0}, batch_instances_number);

            memcpy(batch_instances_indices.data(),
                   training_instances_indices.data(),
                   static_cast<size_t>(batch_instances_number)*sizeof(Index));

            sort(batch_instances_indices.data(), batch_instances_indices.data() + batch_instances_indices.size());



//            cout << batch_instances_indices << endl;system("pause");

            batch.fill(batch_instances_indices, input_variables_indices, target_variables_indices);

            // Neural network

            neural_network_pointer->forward_propagate(batch, forward_propagation);

            // Loss

            loss_index_pointer->back_propagate(batch, forward_propagation, back_propagation);

            training_loss += back_propagation.loss;

            // Optimization algorithm

            update_iteration(back_propagation, optimization_data);

            neural_network_pointer->set_parameters(optimization_data.parameters);
        }

        // Loss

        training_loss /= static_cast<type>(training_batches_number);

        selection_error = 0;

        if(has_selection)
        {
            selection_batches = data_set_pointer->get_selection_batches(batch_instances_number, false);

            selection_error = 0;

            for(Index iteration = 0; iteration < selection_batches_number; iteration++)
            {
                // Data set

                batch_instances_indices = training_batches.chip(iteration,0);

                batch.fill(batch_instances_indices, input_variables_indices, target_variables_indices);

                // Neural network

                neural_network_pointer->forward_propagate(batch, forward_propagation);

                // Loss index

                loss_index_pointer->calculate_error(batch, forward_propagation, back_propagation);

                selection_error += back_propagation.loss;
            }

            selection_error /= static_cast<type>(selection_batches_number);

            if(selection_error <= minimum_selection_error)
            {
                minimum_selection_error = selection_error;
                minimal_selection_parameters = optimization_data.parameters;
            }
        }

        // Training history loss index

        if(reserve_training_error_history) results.training_error_history(epoch) = training_loss;

        if(reserve_selection_error_history) results.selection_error_history(epoch) = selection_error;

        // Stopping criteria

        time(&current_time);

        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        if(training_loss <= training_loss_goal)
        {
            if(display) cout << "Epoch " << epoch << ": Training loss goal reached.\n";

            stop_training = true;

            results.stopping_condition = MaximumEpochsNumber;
        }

        if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << ": Maximum number of epochs reached.\n";

            stop_training = true;

            results.stopping_condition = MaximumEpochsNumber;
        }

        else if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << ": Maximum training time reached.\n";

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
                cout << "Training loss: " << training_loss << "\n"
                     << "Batch size: " << batch_instances_number << "\n"
                     << loss_index_pointer->write_information()
                     << "Elapsed time: " << write_elapsed_time(elapsed_time)<<"\n";

                if(has_selection) cout << "Selection error: " << selection_error << endl << endl;
            }

            results.resize_training_history(1 + epoch);
            results.final_parameters = optimization_data.parameters;
            results.final_training_error = training_loss;
            results.final_selection_error = selection_error;
            results.elapsed_time = elapsed_time;
            results.epochs_number = epoch;

            break;
        }
        else if(display && epoch % display_period == 0)
        {
            cout << "Epoch " << epoch << "/"<<maximum_epochs_number << ":\n"
                 << "Training loss: " << training_loss << "\n"
                 << "Batch size: " << batch_instances_number << "\n"
                 << loss_index_pointer->write_information()
                 << "Elapsed time: " << write_elapsed_time(elapsed_time)<<"\n";

            if(has_selection) cout << "Selection error: " << selection_error << endl << endl;
        }

        if(stop_training) break;

    }

    if(has_selection && choose_best_selection)
    {
        optimization_data.parameters = minimal_selection_parameters;

        neural_network_pointer->set_parameters(optimization_data.parameters);

        selection_error = minimum_selection_error;
    }

    results.final_parameters = optimization_data.parameters;

    results.final_training_error= training_loss;
    results.final_selection_error = selection_error;

    results.elapsed_time = elapsed_time;

    return results;
}


void StochasticGradientDescent::perform_training_void()
{
    perform_training();
}


string StochasticGradientDescent::write_optimization_algorithm_type() const
{
    return "GRADIENT_DESCENT";
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> StochasticGradientDescent::to_string_matrix() const
{
    /*
        ostringstream buffer;

        Tensor<string, 1> labels;
        Tensor<string, 1> values;

       // Minimum parameters increment norm

       labels.push_back("Minimum parameters increment norm");

       buffer.str("");
       buffer << minimum_parameters_increment_norm;

       values.push_back(buffer.str());

       // Minimum loss decrease

       labels.push_back("Minimum loss decrease");

       buffer.str("");
       buffer << minimum_loss_decrease;

       values.push_back(buffer.str());

       // Loss goal

       labels.push_back(" Loss goal");

       buffer.str("");
       buffer << training_loss_goal;

       values.push_back(buffer.str());

       // Gradient norm goal

       labels.push_back("Gradient norm goal");

       buffer.str("");
       buffer << gradient_norm_goal;

       values.push_back(buffer.str());

       // Maximum selection error increases

       labels.push_back("Maximum selection error increases");

       buffer.str("");
       buffer << maximum_selection_error_increases;

       values.push_back(buffer.str());

       // Maximum iterations number

       labels.push_back("Maximum iterations number");

       buffer.str("");
       buffer << maximum_epochs_number;

       values.push_back(buffer.str());

       // Maximum time

       labels.push_back("Maximum time");

       buffer.str("");
       buffer << maximum_time;

       values.push_back(buffer.str());

       // Reserve training error history

       labels.push_back("Reserve training error history");

       buffer.str("");

       if(reserve_training_error_history)
       {
           buffer << "true";
       }
       else
       {
           buffer << "false";
       }

       values.push_back(buffer.str());

       // Reserve selection error history

       labels.push_back("Reserve selection error history");

       buffer.str("");

       if(reserve_selection_error_history)
       {
           buffer << "true";
       }
       else
       {
           buffer << "false";
       }

       values.push_back(buffer.str());

       const Index rows_number = labels.size();
       const Index columns_number = 2;

       Tensor<string, 2> string_matrix(rows_number, columns_number);

       string_matrix.set_column(0, labels, "name");
       string_matrix.set_column(1, values, "value");

        return string_matrix;
    */
    return Tensor<string, 2>();
}


/// Serializes the training parameters, the stopping criteria and other user stuff
/// concerning the gradient descent object.

tinyxml2::XMLDocument* StochasticGradientDescent::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Optimization algorithm

    tinyxml2::XMLElement* root_element = document->NewElement("StochasticGradientDescent");

    document->InsertFirstChild(root_element);

    tinyxml2::XMLElement* element = nullptr;
    tinyxml2::XMLText* text = nullptr;

    // Return minimum selection error neural network

    element = document->NewElement("ReturnMinimumSelectionErrorNN");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << choose_best_selection;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Warning parameters norm

    element = document->NewElement("WarningParametersNorm");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << warning_parameters_norm;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Warning gradient norm

    element = document->NewElement("WarningGradientNorm");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << warning_gradient_norm;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Error parameters norm

    element = document->NewElement("ErrorParametersNorm");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << error_parameters_norm;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Error gradient norm

    element = document->NewElement("ErrorGradientNorm");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << error_gradient_norm;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Loss goal

    element = document->NewElement("LossGoal");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << training_loss_goal;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Maximum iterations number

    element = document->NewElement("MaximumEpochsNumber");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << maximum_epochs_number;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Maximum time

    element = document->NewElement("MaximumTime");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << maximum_time;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Reserve training error history

    element = document->NewElement("ReserveTrainingErrorHistory");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << reserve_training_error_history;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Reserve selection error history

    element = document->NewElement("ReserveSelectionErrorHistory");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << reserve_selection_error_history;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    //Reserve selection error history

    element = document->NewElement("ReserveSelectionErrorHistory");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << reserve_selection_error_history;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Display period

    element = document->NewElement("DisplayPeriod");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << display_period;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Save period

    element = document->NewElement("SavePeriod");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << save_period;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Neural network file name

    element = document->NewElement("NeuralNetworkFileName");
    root_element->LinkEndChild(element);

    text = document->NewText(neural_network_file_name.c_str());
    element->LinkEndChild(text);

    // Display warnings

    element = document->NewElement("Display");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << display;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    return document;
}


/// Serializes the gradient descent object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void StochasticGradientDescent::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Batch size

    file_stream.OpenElement("BatchSize");

    buffer.str("");
    buffer << batch_instances_number;

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
            const Index training_instances_number = loss_index_pointer->get_data_set_pointer()->get_training_instances_number();

            if(new_batch_size > training_instances_number || new_batch_size == 0)
            {
                set_batch_instances_number(training_instances_number);
            }
            else
            {
                set_batch_instances_number(new_batch_size);
            }
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
