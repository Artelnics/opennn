//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "adaptive_moment_estimation.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a adaptive moment estimation optimization algorithm not associated to any loss index object.
/// It also initializes the class members to their default values.

AdaptiveMomentEstimation::AdaptiveMomentEstimation()
    :OptimizationAlgorithm()
{
    set_default();
}


/// Loss index constructor.
/// It creates a adaptive moment estimation optimization algorithm associated to a loss index.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

AdaptiveMomentEstimation::AdaptiveMomentEstimation(LossIndex* new_loss_index_pointer)
    : OptimizationAlgorithm(new_loss_index_pointer)
{
    set_default();
}


/// XML constructor.
/// It creates a gradient descent optimization algorithm not associated to any loss index object.
/// It also loads the class members from a XML document.
/// @param document TinyXML document with the members of a gradient descent object.

AdaptiveMomentEstimation::AdaptiveMomentEstimation(const tinyxml2::XMLDocument& document)
    : OptimizationAlgorithm(document)
{
    set_default();

    from_XML(document);
}


/// Destructor.

AdaptiveMomentEstimation::~AdaptiveMomentEstimation()
{
}


/// Returns the initial learning rate.

const type& AdaptiveMomentEstimation::get_initial_learning_rate() const
{
    return initial_learning_rate;
}


/// Returns beta 1.

const type& AdaptiveMomentEstimation::get_beta_1() const
{
    return beta_1;
}


/// Returns beta 2.

const type& AdaptiveMomentEstimation::get_beta_2() const
{
    return beta_2;
}


/// Returns epsilon.

const type& AdaptiveMomentEstimation::get_epsilon() const
{
    return epsilon;
}


/// Returns the goal value for the loss.
/// This is used as a stopping criterion when training a neural network

const type& AdaptiveMomentEstimation::get_loss_goal() const
{
    return training_loss_goal;
}


/// Returns the maximum training time.

const type& AdaptiveMomentEstimation::get_maximum_time() const
{
    return maximum_time;
}


/// Returns true if the final model will be the neural network with the minimum selection error, false otherwise.

const bool& AdaptiveMomentEstimation::get_choose_best_selection() const
{
    return choose_best_selection;
}


/// Returns true if the error history vector is to be reserved, and false otherwise.

const bool& AdaptiveMomentEstimation::get_reserve_training_error_history() const
{
    return reserve_training_error_history;
}


/// Returns true if the selection error history vector is to be reserved, and false otherwise.

const bool& AdaptiveMomentEstimation::get_reserve_selection_error_history() const
{
    return reserve_selection_error_history;
}


/// Sets a pointer to a loss index object to be associated to the gradient descent object.
/// It also sets that loss index to the learning rate algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void AdaptiveMomentEstimation::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
    loss_index_pointer = new_loss_index_pointer;
}


void AdaptiveMomentEstimation::set_default()
{
    // TRAINING OPERATORS

    initial_learning_rate = static_cast<type>(0.001);
    initial_decay = 0;
    beta_1 = static_cast<type>(0.9);
    beta_2 = static_cast<type>(0.999);

    epsilon =static_cast<type>(1.e-7);

    // Stopping criteria

    training_loss_goal = 0;
    maximum_time = 1000.0;
    maximum_epochs_number = 10000;
    choose_best_selection = false;

    // TRAINING HISTORY

    reserve_training_error_history = true;
    reserve_selection_error_history = false;

    // UTILITIES

    display = true;
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
/// @param new_reserve_all_training_history True if the training history of all variables is to be reserved, false otherwise.

void AdaptiveMomentEstimation::set_reserve_all_training_history(const bool& new_reserve_all_training_history)
{
    reserve_training_error_history = new_reserve_all_training_history;

    reserve_selection_error_history = new_reserve_all_training_history;
}


/// Sets a new learning rate.
/// @param new_learning_rate.

void AdaptiveMomentEstimation::set_initial_learning_rate(const type& new_learning_rate)
{
    initial_learning_rate= new_learning_rate;
}


/// Sets beta 1 generally close to 1.
/// @param new_beta_1.

void AdaptiveMomentEstimation::set_beta_1(const type& new_beta_1)
{
    beta_1= new_beta_1;
}


/// Sets beta 2 generally close to 1.
/// @param new_beta_2.

void AdaptiveMomentEstimation::set_beta_2(const type& new_beta_2)
{
    beta_2= new_beta_2;
}


/// Sets epsilon.
/// @param epsilon.

void AdaptiveMomentEstimation::set_epsilon(const type& new_epsilon)
{
    epsilon= new_epsilon;
}


/// Set the a new maximum for the epochs number.
/// @param new_maximum_epochs number New maximum epochs number.

void AdaptiveMomentEstimation:: set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_epochs_number < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: AdaptiveMomentEstimation class.\n"
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

void AdaptiveMomentEstimation::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


/// Sets a new maximum training time.
/// @param new_maximum_time Maximum training time.

void AdaptiveMomentEstimation::set_maximum_time(const type& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: AdaptiveMomentEstimation class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set maximum time

    maximum_time = new_maximum_time;
}


/// Makes the minimum selection error neural network of all the iterations to be returned or not.
/// @param new_choose_best_selection True if the final model will be the neural network with the minimum selection error,
/// false otherwise.

void AdaptiveMomentEstimation::set_choose_best_selection(const bool& new_choose_best_selection)
{
    choose_best_selection = new_choose_best_selection;
}


/// Makes the error history vector to be reseved or not in memory.
/// @param new_reserve_training_error_history True if the error history vector is to be reserved, false otherwise.

void AdaptiveMomentEstimation::set_reserve_training_error_history(const bool& new_reserve_training_error_history)
{
    reserve_training_error_history = new_reserve_training_error_history;
}


/// Makes the selection error history to be reserved or not in memory.
/// This is a vector.
/// @param new_reserve_selection_error_history True if the selection error history is to be reserved, false otherwise.

void AdaptiveMomentEstimation::set_reserve_selection_error_history(const bool& new_reserve_selection_error_history)
{
    reserve_selection_error_history = new_reserve_selection_error_history;
}


/// Trains a neural network with an associated loss index,
/// according to the gradient descent method.
/// Training occurs according to the training parameters and stopping criteria.
/// It returns a results structure with the history and the final values of the reserved variables.

OptimizationAlgorithm::Results AdaptiveMomentEstimation::perform_training()
{
    Results results;

    check();

    // Start training

    if(display) cout << "Training with adaptive moment estimator \"Adam\" ...\n";

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Tensor<Index, 1> input_variables_indices = data_set_pointer->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set_pointer->get_target_variables_indices();

    const Tensor<Index, 1> training_instances_indices = data_set_pointer->get_training_instances_indices();
    const Tensor<Index, 1> selection_instances_indices = data_set_pointer->get_selection_instances_indices();

    const Index training_instances_number = data_set_pointer->get_training_instances_number();
    const Index selection_instances_number = data_set_pointer->get_selection_instances_number();

    if(training_instances_number < batch_instances_number) batch_instances_number = training_instances_number;

    const bool has_selection = data_set_pointer->has_selection();

    DataSet::Batch training_batch(batch_instances_number, data_set_pointer);

    DataSet::Batch selection_batch(selection_instances_number, data_set_pointer);
    selection_batch.fill(selection_instances_indices, input_variables_indices, target_variables_indices);

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    type parameters_norm = 0;
    Index parameters_number = neural_network_pointer->get_parameters_number();

    NeuralNetwork::ForwardPropagation training_forward_propagation(batch_instances_number, neural_network_pointer);
    NeuralNetwork::ForwardPropagation selection_forward_propagation(selection_instances_number, neural_network_pointer);

    // Loss index

    LossIndex::BackPropagation training_back_propagation(batch_instances_number, loss_index_pointer);
    LossIndex::BackPropagation selection_back_propagation(selection_instances_number, loss_index_pointer);

    type training_error = 0;
    type training_loss = 0;

    type old_selection_error = 0;
    Index selection_error_increases = 0;

    // Optimization algorithm

    type learning_rate = 0;

    Tensor<type, 1> minimal_selection_parameters(parameters_number);
    type minimum_selection_error = numeric_limits<type>::max();

    bool stop_training = false;

    time_t beginning_time, current_time;
    time(&beginning_time);
    type elapsed_time = 0;

    results.resize_training_history(maximum_epochs_number + 1);
    if(has_selection) results.resize_selection_history(maximum_epochs_number + 1);

    OptimizationData optimization_data(this);

    bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer()
    || neural_network_pointer->has_recurrent_layer())
        is_forecasting = true;

    // Main loop

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        const Tensor<Index, 2> training_batches = data_set_pointer->get_batches(training_instances_indices,
                                                                                         batch_instances_number,
                                                                                         is_forecasting);
        const Index batches_number = training_batches.dimension(0);

        parameters_norm = l2_norm(optimization_data.parameters);

        training_loss = 0;
        training_error = 0;

        optimization_data.iteration = 0;

        for(Index iteration = 0; iteration < batches_number; iteration++)
        {
            optimization_data.iteration++;

            // Data set

            training_batch.fill(training_batches.chip(iteration, 0), input_variables_indices, target_variables_indices);

            // Neural network

            neural_network_pointer->forward_propagate(training_batch, training_forward_propagation);

            // Loss index

            loss_index_pointer->back_propagate(training_batch, training_forward_propagation, training_back_propagation);

            training_error += training_back_propagation.error;
            training_loss += training_back_propagation.loss;

            // Gradient

            update_iteration(training_back_propagation, optimization_data);

            neural_network_pointer->set_parameters(optimization_data.parameters);
        }

        // Loss

        training_loss /= static_cast<type>(batches_number);
        training_error /= static_cast<type>(batches_number);

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

        // Elapsed time

        time(&current_time);
        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        // Training history

        if(reserve_training_error_history) results.training_error_history(epoch) = training_error;

        if(reserve_selection_error_history) results.selection_error_history(epoch) = selection_back_propagation.error;

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

        else if(training_loss <= training_loss_goal)
        {
            if(display) cout << "Epoch " << epoch+1 << ": Loss goal reached.\n";

            stop_training = true;

            results.stopping_condition  = LossGoal;
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
                     << "Training error: " << training_error << "\n"
                     << loss_index_pointer->write_information()
                     << "Learning rate: " << learning_rate << "\n"
                     << "Elapsed time: " << write_elapsed_time(elapsed_time)<<"\n";

                if(has_selection) cout << "Selection error: " << selection_back_propagation.error << endl<<endl;
            }

            results.resize_training_error_history(epoch+1);
            if(has_selection) results.resize_selection_error_history(epoch+1);

            results.final_parameters = optimization_data.parameters;

            results.final_parameters_norm = parameters_norm;

            results.final_training_error = training_error;

            results.final_selection_error = selection_back_propagation.error;

            results.elapsed_time = elapsed_time;

            results.epochs_number = epoch;

            break;
        }
        else if(epoch == 0 || (epoch+1)%display_period == 0)
        {
            if(display)
            {
                cout << "Epoch " << epoch+1 << ";\n"
                     << "Training error: " << training_error << "\n"
                     << "Batch size: " << batch_instances_number << "\n"
                     << loss_index_pointer->write_information()
                     << "Elapsed time: " << write_elapsed_time(elapsed_time)<<"\n";

                if(has_selection) cout << "Selection error: " << selection_back_propagation.error << endl<<endl;
            }
        }

        // Update stuff

        old_selection_error = selection_back_propagation.error;

        if(stop_training) break;
    }

    if(choose_best_selection)
    {
        optimization_data.parameters = optimization_data.minimal_selection_parameters;
        parameters_norm = l2_norm(optimization_data.parameters);

        neural_network_pointer->set_parameters(optimization_data.parameters);

        selection_back_propagation.error = minimum_selection_error;
    }

    return results;
}


void AdaptiveMomentEstimation::perform_training_void()
{
   perform_training();
}


string AdaptiveMomentEstimation::write_optimization_algorithm_type() const
{
    return "ADAPTIVE_MOMENT_ESTIMATION";
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> AdaptiveMomentEstimation::to_string_matrix() const
{
    Tensor<string, 2> labels_values(11, 2);

    Index row_index = 0;

    // Initial learning rate

    labels_values(row_index,0) = "Initial learning rate";
    labels_values(row_index,1) = std::to_string(initial_learning_rate);

    row_index++;

    // Initial decay

    labels_values(row_index,0) = "Initial decay";
    labels_values(row_index,1) = std::to_string(initial_decay);

    row_index++;

    // Beta 1

    labels_values(row_index,0) = "Beta 1";
    labels_values(row_index,1) = std::to_string(beta_1);

    row_index++;

    // Beta 2

    labels_values(row_index,0) = "Beta 2";
    labels_values(row_index,1) = std::to_string(beta_2);

    row_index++;

    // Epsilon

    labels_values(row_index,0) = "Epsilon";
    labels_values(row_index,1) = std::to_string(epsilon);

    row_index++;

    // Training loss goal

    labels_values(row_index,0) = "Training loss goal";
    labels_values(row_index,1) = std::to_string(training_loss_goal);

    row_index++;

    // Maximum epochs number

    labels_values(row_index,0) = "Maximum epochs number";
    labels_values(row_index,1) = std::to_string(maximum_epochs_number);

    row_index++;

    // Maximum time

    labels_values(row_index,0) = "Maximum time";
    labels_values(row_index,1) = std::to_string(maximum_time);

    row_index++;

    // Batch instances number

    labels_values(row_index,0) = "Batch instances number";
    labels_values(row_index,1) = std::to_string(batch_instances_number);

    row_index++;

    // Reserve training error history

    labels_values(row_index,0) = "Reserve training error history";

    if(reserve_training_error_history)
    {
        labels_values(row_index,1) = "true";
    }
    else
    {
        labels_values(row_index,1) = "false";
    }

    row_index++;

    // Reserve selection error history

    labels_values(row_index,0) = "Reserve selection error history";

    if(reserve_training_error_history)
    {
        labels_values(row_index,1) = "true";
    }
    else
    {
        labels_values(row_index,1) = "false";
    }

    return labels_values;
}


/// Serializes the training parameters, the stopping criteria and other user stuff
/// concerning the gradient descent object.

tinyxml2::XMLDocument* AdaptiveMomentEstimation::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Optimization algorithm

    tinyxml2::XMLElement* root_element = document->NewElement("AdaptiveMomentEstimation");

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

void AdaptiveMomentEstimation::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("AdaptiveMomentEstimation");

    // Batch size

    file_stream.OpenElement("BatchSize");

    buffer.str("");
    buffer << batch_instances_number;

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

    file_stream.CloseElement();
}


void AdaptiveMomentEstimation::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("AdaptiveMomentEstimation");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: AdaptiveMomentEstimation class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Adaptive moment estimation element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Batch size

    const tinyxml2::XMLElement* batch_size_element = root_element->FirstChildElement("BatchSize");

    if(batch_size_element)
    {
        const Index new_batch_size = static_cast<Index>(atoi(batch_size_element->GetText()));

        try
        {
            set_batch_instances_number(new_batch_size);
        }
        catch(const logic_error& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Return minimum selection error neural network

    const tinyxml2::XMLElement* choose_best_selection_element
        = root_element->FirstChildElement("ReturnMinimumSelectionErrorNN");

    if(choose_best_selection_element)
    {
        const string new_choose_best_selection
            = choose_best_selection_element->GetText();

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

    // Maximum eochs number
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


void AdaptiveMomentEstimation::set_batch_instances_number(const Index& new_batch_instances_number)
{
    batch_instances_number = new_batch_instances_number;
}


void AdaptiveMomentEstimation::update_iteration(const LossIndex::BackPropagation& back_propagation,
                              OptimizationData& optimization_data)
{
    const type learning_rate =
            initial_learning_rate*
            sqrt(1 - pow(beta_2, static_cast<type>(optimization_data.iteration)))/
            (1 - pow(beta_1, static_cast<type>(optimization_data.iteration)));

    optimization_data.gradient_exponential_decay.device(*thread_pool_device)
            = optimization_data.last_gradient_exponential_decay*beta_1
            + back_propagation.gradient*(1 - beta_1);

    optimization_data.last_gradient_exponential_decay = optimization_data.gradient_exponential_decay;

    optimization_data.square_gradient_exponential_decay.device(*thread_pool_device)
            = optimization_data.last_square_gradient_exponential_decay*beta_2
            + back_propagation.gradient*back_propagation.gradient*(1 - beta_2);

    optimization_data.last_square_gradient_exponential_decay = optimization_data.square_gradient_exponential_decay;

    // Update parameters

    optimization_data.parameters.device(*thread_pool_device) -=
            optimization_data.gradient_exponential_decay*learning_rate/(optimization_data.square_gradient_exponential_decay.sqrt() + epsilon);
}


AdaptiveMomentEstimation::OptimizationData::OptimizationData()
{
}


AdaptiveMomentEstimation::OptimizationData::OptimizationData(AdaptiveMomentEstimation* new_stochastic_gradient_descent_pointer)
{
    set(new_stochastic_gradient_descent_pointer);
}


AdaptiveMomentEstimation::OptimizationData::~OptimizationData()
{
}


void AdaptiveMomentEstimation::OptimizationData::set(AdaptiveMomentEstimation* new_adaptive_moment_estimation_pointer)
{
    adaptive_moment_estimation_pointer = new_adaptive_moment_estimation_pointer;

    LossIndex* loss_index_pointer = new_adaptive_moment_estimation_pointer->get_loss_index_pointer();

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    parameters.resize(parameters_number);
    parameters = neural_network_pointer->get_parameters();

    minimal_selection_parameters.resize(parameters_number);

    gradient_exponential_decay.resize(parameters_number);
    gradient_exponential_decay.setZero();

    square_gradient_exponential_decay.resize(parameters_number);
    square_gradient_exponential_decay.setZero();

    last_gradient_exponential_decay.resize(parameters_number);
    last_gradient_exponential_decay.setZero();

    last_square_gradient_exponential_decay.resize(parameters_number);
    last_square_gradient_exponential_decay.setZero();
}

void AdaptiveMomentEstimation::OptimizationData::print() const
{
    cout << "Gradien exponential decay:" << endl <<gradient_exponential_decay << endl;

    cout << "Square gradient exponential decay:" << endl << square_gradient_exponential_decay << endl;

    cout << "Parameters:" << endl << parameters << endl;
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
