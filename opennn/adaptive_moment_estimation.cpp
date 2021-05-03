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


/// Sets a pointer to a loss index object to be associated to the gradient descent object.
/// It also sets that loss index to the learning rate algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void AdaptiveMomentEstimation::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
    loss_index_pointer = new_loss_index_pointer;
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

void AdaptiveMomentEstimation::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
#ifdef OPENNN_DEBUG

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
#ifdef OPENNN_DEBUG

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


/// Trains a neural network with an associated loss index,
/// according to the gradient descent method.
/// Training occurs according to the training parameters and stopping criteria.
/// It returns a results structure with the history and the final values of the reserved variables.

TrainingResults AdaptiveMomentEstimation::perform_training()
{
    TrainingResults results(maximum_epochs_number+1);

    check();

    // Start training

    if(display) cout << "Training with adaptive moment estimation \"Adam\" ...\n";

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const bool has_selection = data_set_pointer->has_selection();

    const Tensor<Index, 1> input_variables_indices = data_set_pointer->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set_pointer->get_target_variables_indices();

    const Tensor<Index, 1> training_samples_indices = data_set_pointer->get_training_samples_indices();
    const Tensor<Index, 1> selection_samples_indices = data_set_pointer->get_selection_samples_indices();

    Index batch_size_training = 0;
    Index batch_size_selection = 0;

    const Index training_samples_number = data_set_pointer->get_training_samples_number();
    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();

    training_samples_number < batch_samples_number
            ? batch_size_training = training_samples_number
            : batch_size_training = batch_samples_number;

    selection_samples_number < batch_samples_number && selection_samples_number != 0
            ? batch_size_selection = selection_samples_number
            : batch_size_selection = batch_samples_number;

    DataSetBatch batch_training(batch_size_training, data_set_pointer);
    DataSetBatch batch_selection(batch_size_selection, data_set_pointer);

    const Index training_batches_number = training_samples_number/batch_size_training;
    const Index selection_batches_number = selection_samples_number/batch_size_selection;

    Tensor<Index, 2> training_batches(training_batches_number, batch_size_training);
    Tensor<Index, 2> selection_batches(selection_batches_number, batch_size_selection);

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    NeuralNetworkForwardPropagation training_forward_propagation(batch_size_training, neural_network_pointer);
    NeuralNetworkForwardPropagation selection_forward_propagation(batch_size_selection, neural_network_pointer);

    // Loss index

    LossIndexBackPropagation training_back_propagation(batch_size_training, loss_index_pointer);
    LossIndexBackPropagation selection_back_propagation(batch_size_selection, loss_index_pointer);

    type training_error = 0;
    type training_loss = 0;

    Index selection_failures = 0;
    type gradient_norm = 0;

    // Optimization algorithm

    AdaptiveMomentEstimationData optimization_data(this);

    type learning_rate = 0;

    bool stop_training = false;

    time_t beginning_time, current_time;
    time(&beginning_time);
    type elapsed_time = 0;

    bool shuffle = false;

    if(neural_network_pointer->has_long_short_term_memory_layer()
    || neural_network_pointer->has_recurrent_layer())
        shuffle = false;

    // Main loop

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = data_set_pointer->get_batches(training_samples_indices, batch_size_training, shuffle);

        const Index batches_number = training_batches.dimension(0);

        training_loss = 0;
        training_error = 0;

        optimization_data.iteration = 0;

        for(Index iteration = 0; iteration < batches_number; iteration++)
        {
            optimization_data.iteration++;

            // Data set

            batch_training.fill(training_batches.chip(iteration, 0), input_variables_indices, target_variables_indices);

            // Neural network

            neural_network_pointer->forward_propagate(batch_training, training_forward_propagation);

            // Loss index

            loss_index_pointer->back_propagate(batch_training, training_forward_propagation, training_back_propagation);

            training_error += training_back_propagation.error;
            training_loss += training_back_propagation.loss;

            update_parameters(training_back_propagation, optimization_data);
        }

        gradient_norm = l2_norm(training_back_propagation.gradient);

        // Loss

        training_loss /= static_cast<type>(batches_number);
        training_error /= static_cast<type>(batches_number);

        results.training_error_history(epoch) = training_error;

        if(has_selection)
        {
            selection_batches = data_set_pointer->get_batches(selection_samples_indices, batch_size_selection, shuffle);

            //selection_error = 0;

            for(Index iteration = 0; iteration < selection_batches_number; iteration++)
            {
                // Data set

                batch_selection.fill(selection_batches.chip(iteration,0), input_variables_indices, target_variables_indices);

                // Neural network

                neural_network_pointer->forward_propagate(batch_selection, selection_forward_propagation);

                // Loss

                loss_index_pointer->calculate_errors(batch_selection, selection_forward_propagation, selection_back_propagation);

                loss_index_pointer->calculate_error(batch_selection, selection_forward_propagation, selection_back_propagation);

                //selection_error += selection_back_propagation.error;
            }

            //selection_error /= static_cast<type>(selection_batches_number);

            //if(selection_error > old_selection_error) selection_failures++;

            //if(has_selection) results.selection_error_history(epoch) = selection_error;
        }


        // Elapsed time

        time(&current_time);
        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
//            if(has_selection) cout << "Selection error: " << selection_error << endl;
            cout << "Gradient norm: " << gradient_norm << endl;
            cout << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
        }

        // Training history


        if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Maximum number of epochs reached: " << epoch << endl;

            stop_training = true;

            results.stopping_condition = MaximumEpochsNumber;
        }

        if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Maximum training time reached: " << write_elapsed_time(elapsed_time) << endl;

            stop_training = true;

            results.stopping_condition = MaximumTime;
        }

        if(training_loss <= training_loss_goal)
        {
            if(display) cout << "Loss goal reached: " << training_loss << endl;

            stop_training = true;

            results.stopping_condition  = LossGoal;
        }

        if(gradient_norm <= gradient_norm_goal)
        {
            if(display) cout << "Gradient norm goal reached: " << gradient_norm << endl;

            stop_training = true;

            results.stopping_condition = GradientNormGoal;
        }

        if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Maximum selection failures reached: " << selection_failures << endl;

            stop_training = true;

            results.stopping_condition = MaximumSelectionErrorIncreases;
        }

        if(stop_training)
        {
            results.resize_training_error_history(epoch+1);

            if(has_selection) results.resize_selection_error_history(epoch+1);

            results.elapsed_time = write_elapsed_time(elapsed_time);

            break;
        }

        // Update stuff

        if(epoch != 0 && epoch % save_period == 0) neural_network_pointer->save(neural_network_file_name);

        if(stop_training) break;
    }

    if(display) results.print();

    return results;
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
    labels_values(row_index,1) = to_string(initial_learning_rate);

    row_index++;

    // Initial decay

    labels_values(row_index,0) = "Initial decay";
    labels_values(row_index,1) = to_string(initial_decay);

    row_index++;

    // Beta 1

    labels_values(row_index,0) = "Beta 1";
    labels_values(row_index,1) = to_string(beta_1);

    row_index++;

    // Beta 2

    labels_values(row_index,0) = "Beta 2";
    labels_values(row_index,1) = to_string(beta_2);

    row_index++;

    // Epsilon

    labels_values(row_index,0) = "Epsilon";
    labels_values(row_index,1) = to_string(epsilon);

    row_index++;

    // Training loss goal

    labels_values(row_index,0) = "Training loss goal";
    labels_values(row_index,1) = to_string(training_loss_goal);

    row_index++;

    // Maximum epochs number

    labels_values(row_index,0) = "Maximum epochs number";
    labels_values(row_index,1) = to_string(maximum_epochs_number);

    row_index++;

    // Maximum time

    labels_values(row_index,0) = "Maximum time";
    labels_values(row_index,1) = to_string(maximum_time);

    row_index++;

    // DataSetBatch samples number

    labels_values(row_index,0) = "DataSetBatch samples number";
    labels_values(row_index,1) = to_string(batch_samples_number);

    row_index++;

    return labels_values;
}


/// Serializes the gradient descent object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void AdaptiveMomentEstimation::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("AdaptiveMomentEstimation");

    // DataSetBatch size

    file_stream.OpenElement("BatchSize");

    buffer.str("");
    buffer << batch_samples_number;

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

    // Hardware use

    file_stream.OpenElement("HardwareUse");

    buffer.str("");
    buffer << this->get_hardware_use();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // End element

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

    // DataSetBatch size

    const tinyxml2::XMLElement* batch_size_element = root_element->FirstChildElement("BatchSize");

    if(batch_size_element)
    {
        const Index new_batch_size = static_cast<Index>(atoi(batch_size_element->GetText()));

        try
        {
            set_batch_samples_number(new_batch_size);
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


/// Set number of samples in each batch. Default 1000.

void AdaptiveMomentEstimation::set_batch_samples_number(const Index& new_batch_samples_number)
{
    batch_samples_number = new_batch_samples_number;
}


Index AdaptiveMomentEstimation::get_batch_samples_number() const
{
    return batch_samples_number;
}


/// Update iteration parameters

void AdaptiveMomentEstimation::update_parameters(LossIndexBackPropagation& back_propagation,
                              AdaptiveMomentEstimationData& optimization_data)
{  
    const type learning_rate =
            initial_learning_rate*
            sqrt(1 - pow(beta_2, static_cast<type>(optimization_data.iteration)))/
            (1 - pow(beta_1, static_cast<type>(optimization_data.iteration)));

    optimization_data.gradient_exponential_decay.device(*thread_pool_device)
            = optimization_data.gradient_exponential_decay*beta_1
            + back_propagation.gradient*(1 - beta_1);

    optimization_data.square_gradient_exponential_decay.device(*thread_pool_device)
            = optimization_data.square_gradient_exponential_decay*beta_2
            + back_propagation.gradient*back_propagation.gradient*(1 - beta_2);

    back_propagation.parameters.device(*thread_pool_device) -=
            optimization_data.gradient_exponential_decay*learning_rate/(optimization_data.square_gradient_exponential_decay.sqrt() + epsilon);          

    // Update parameters

    back_propagation.loss_index_pointer->get_neural_network_pointer()->set_parameters(back_propagation.parameters);
}


AdaptiveMomentEstimationData::AdaptiveMomentEstimationData()
{
}


AdaptiveMomentEstimationData::AdaptiveMomentEstimationData(AdaptiveMomentEstimation* new_stochastic_gradient_descent_pointer)
{
    set(new_stochastic_gradient_descent_pointer);
}


AdaptiveMomentEstimationData::~AdaptiveMomentEstimationData()
{
}


void AdaptiveMomentEstimationData::set(AdaptiveMomentEstimation* new_adaptive_moment_estimation_pointer)
{
    adaptive_moment_estimation_pointer = new_adaptive_moment_estimation_pointer;

    LossIndex* loss_index_pointer = new_adaptive_moment_estimation_pointer->get_loss_index_pointer();

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    gradient_exponential_decay.resize(parameters_number);
    gradient_exponential_decay.setZero();

    square_gradient_exponential_decay.resize(parameters_number);
    square_gradient_exponential_decay.setZero();
}


void AdaptiveMomentEstimationData::print() const
{
    cout << "Gradien exponential decay:" << endl
         <<gradient_exponential_decay << endl;

    cout << "Square gradient exponential decay:" << endl
         << square_gradient_exponential_decay << endl;

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
