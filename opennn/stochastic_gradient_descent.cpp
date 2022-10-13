//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "stochastic_gradient_descent.h"

namespace opennn
{

/// Default constructor.
/// It creates a stochastic gradient descent optimization algorithm not associated with any loss index object.
/// It also initializes the class members to their default values.

StochasticGradientDescent::StochasticGradientDescent()
    :OptimizationAlgorithm()
{
    set_default();
}


/// Loss index constructor.
/// It creates a stochastic gradient descent optimization algorithm associated with a loss index.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

StochasticGradientDescent::StochasticGradientDescent(LossIndex* new_loss_index_pointer)
    : OptimizationAlgorithm(new_loss_index_pointer)
{
    set_default();
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


/// Sets a pointer to a loss index object to be associated with the gradient descent object.
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
    initial_decay = type(0);
    momentum = type(0);
    nesterov = false;

    // Stopping criteria

    training_loss_goal = type(0);
    maximum_time = type(3600.0);
    maximum_epochs_number = 10000;

    // UTILITIES

    display_period = 100;
}


Index StochasticGradientDescent::get_batch_samples_number() const
{
    return batch_samples_number;
}


/// Set the initial value for the learning rate. If dacay is not active learning rate will be constant
/// otherwise learning rate will decay over each update.
/// @param new_initial_learning_rate initial learning rate value.

void StochasticGradientDescent::set_initial_learning_rate(const type& new_learning_rate)
{
#ifdef OPENNN_DEBUG

    if(new_learning_rate <= static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_initial_learning_rate(const type&) method.\n"
               << "initial_learning_rate must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    // Set learning rate

    initial_learning_rate = new_learning_rate;
}


/// Set the initial value for the decay.
/// @param new_initial_learning_rate initial value for the decay.

void StochasticGradientDescent::set_initial_decay(const type& new_dacay)
{
#ifdef OPENNN_DEBUG

    if(new_dacay < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_initial_decay(const type&) method.\n"
               << "new_dacay must be equal or greater than 0.\n";

        throw invalid_argument(buffer.str());
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
#ifdef OPENNN_DEBUG

    if(new_momentum < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_momentum(const type&) method.\n"
               << "new_momentum must be equal or greater than 0.\n";

        throw invalid_argument(buffer.str());
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


/// Set the a new maximum for the epochs number.
/// @param new_maximum_epochs number New maximum epochs number.

void StochasticGradientDescent::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_epochs_number < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_maximum_epochs_number(const type&) method.\n"
               << "Maximum epochs number must be equal or greater than 0.\n";

        throw invalid_argument(buffer.str());
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
#ifdef OPENNN_DEBUG

    if(new_maximum_time < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: StochasticGradientDescent class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be equal or greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    // Set maximum time

    maximum_time = new_maximum_time;
}


/// Set hardware to use. Default: Multi-core.

void StochasticGradientDescent::update_parameters(LossIndexBackPropagation& back_propagation,
                      StochasticGradientDescentData& optimization_data) const
{
    back_propagation.assemble = true;

    const type learning_rate = initial_learning_rate/(type(1) + type(optimization_data.iteration)*initial_decay);

    NeuralNetwork* neural_network_pointer = back_propagation.loss_index_pointer->get_neural_network_pointer();

    const Tensor< Tensor< TensorMap< Tensor<type, 1> >*, 1>, 1> layers_parameters = neural_network_pointer->get_layers_parameters();
    const Tensor< Tensor< TensorMap< Tensor<type, 1> >*, 1>, 1> layers_gradient = back_propagation.get_layers_gradient();

    for(Index i = 0; i < layers_parameters.size(); i++)
    {
        for(Index j = 0; j < layers_parameters(i).size(); j++)
        {
            (*layers_parameters(i)(j)).device(*thread_pool_device) += (*layers_gradient(i)(j))*(-learning_rate);
        }
    }
/* /// OLD UPDATE PARAMETERS
    const type learning_rate = initial_learning_rate/(type(1) + type(optimization_data.iteration)*initial_decay);

    optimization_data.parameters_increment.device(*thread_pool_device) = back_propagation.gradient*(-learning_rate);

    if(momentum > type(0))
    {
        optimization_data.parameters_increment.device(*thread_pool_device) += momentum*optimization_data.last_parameters_increment;

        if(!nesterov)
        {
            back_propagation.parameters.device(*thread_pool_device) += optimization_data.parameters_increment;
        }
        else
        {
            optimization_data.nesterov_increment.device(*thread_pool_device)
                    = optimization_data.parameters_increment*momentum - back_propagation.gradient*learning_rate;

            back_propagation.parameters.device(*thread_pool_device) += optimization_data.nesterov_increment;
        }
    }
    else
    {
        back_propagation.parameters.device(*thread_pool_device) += optimization_data.parameters_increment;
    }

    optimization_data.last_parameters_increment = optimization_data.parameters_increment;

    optimization_data.iteration++;

    // Update parameters

    back_propagation.loss_index_pointer->get_neural_network_pointer()->set_parameters(back_propagation.parameters);
*/
}


/// Trains a neural network with an associated loss index,
/// according to the stochastic gradient descent method.
/// Training occurs according to the training parameters and stopping criteria.
/// It returns a results structure with the history and the final values of the reserved variables.

TrainingResults StochasticGradientDescent::perform_training()
{
    TrainingResults results(maximum_epochs_number+1);

    check();

    // Start training

    if(display) cout << "Training with stochastic gradient descent (SGD)...\n";

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

    const Tensor<string, 1> inputs_names = data_set_pointer->get_input_variables_names();
    const Tensor<string, 1> targets_names = data_set_pointer->get_target_variables_names();

    const Tensor<Scaler, 1> input_variables_scalers = data_set_pointer->get_input_variables_scalers();
    const Tensor<Scaler, 1> target_variables_scalers = data_set_pointer->get_target_variables_scalers();

    const Tensor<Descriptives, 1> input_variables_descriptives = data_set_pointer->scale_input_variables();
    Tensor<Descriptives, 1> target_variables_descriptives;

    DataSetBatch batch_training(batch_size_training, data_set_pointer);
    DataSetBatch batch_selection(batch_size_selection, data_set_pointer);

    const Index training_batches_number = training_samples_number/batch_size_training;
    const Index selection_batches_number = selection_samples_number/batch_size_selection;

    Tensor<Index, 2> training_batches(training_batches_number, batch_size_training);
    Tensor<Index, 2> selection_batches(selection_batches_number, batch_size_selection);

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    neural_network_pointer->set_inputs_names(inputs_names);
    neural_network_pointer->set_outputs_names(targets_names);

    if(neural_network_pointer->has_scaling_layer())
    {
        ScalingLayer* scaling_layer_pointer = neural_network_pointer->get_scaling_layer_pointer();
        scaling_layer_pointer->set(input_variables_descriptives, input_variables_scalers);
    }

    if(neural_network_pointer->has_unscaling_layer())
    {
        target_variables_descriptives = data_set_pointer->scale_target_variables();

        UnscalingLayer* unscaling_layer_pointer = neural_network_pointer->get_unscaling_layer_pointer();
        unscaling_layer_pointer->set(target_variables_descriptives, target_variables_scalers);
    }

    NeuralNetworkForwardPropagation training_forward_propagation(batch_size_training, neural_network_pointer);
    NeuralNetworkForwardPropagation selection_forward_propagation(batch_size_selection, neural_network_pointer);

    // Loss index

    loss_index_pointer->set_normalization_coefficient();

    LossIndexBackPropagation training_back_propagation(batch_size_training, loss_index_pointer);
    LossIndexBackPropagation selection_back_propagation(batch_size_selection, loss_index_pointer);

    type training_error = type(0);
    type training_loss = type(0);

    type selection_error = type(0);

    Index selection_failures = 0;

    // Optimization algorithm

    StochasticGradientDescentData optimization_data(this);

    bool stop_training = false;

    time_t beginning_time;
    time_t current_time;
    time(&beginning_time);
    type elapsed_time = type(0);

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

        training_loss = type(0);
        training_error = type(0);

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
            results.training_error_history(epoch) = training_back_propagation.error;

            training_error += training_back_propagation.error;
            training_loss += training_back_propagation.loss;

            // Gradient

            update_parameters(training_back_propagation, optimization_data);
        }

        // Loss

        training_loss /= static_cast<type>(batches_number);
        training_error /= static_cast<type>(batches_number);

        results.training_error_history(epoch) = training_error;

        if(has_selection)
        {
            selection_batches = data_set_pointer->get_batches(selection_samples_indices, batch_size_selection, shuffle);

            selection_error = type(0);

            for(Index iteration = 0; iteration < selection_batches_number; iteration++)
            {
                // Data set

                batch_selection.fill(selection_batches.chip(iteration,0), input_variables_indices, target_variables_indices);

                // Neural network

                neural_network_pointer->forward_propagate(batch_selection, selection_forward_propagation);
                results.selection_error_history(epoch) = selection_error;

                // Loss

                loss_index_pointer->calculate_errors(batch_selection, selection_forward_propagation, selection_back_propagation);
                loss_index_pointer->calculate_error(batch_selection, selection_forward_propagation, selection_back_propagation);

                selection_error += selection_back_propagation.error;
            }

            selection_error /= static_cast<type>(selection_batches_number);

            results.selection_error_history(epoch) = selection_error;

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) selection_failures++;
        }

        // Elapsed time

        time(&current_time);
        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if(has_selection) cout << "Selection error: " << selection_error << endl<<endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        // Stopping criteria

        if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum number of epochs reached: " << epoch << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
        }

        if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum training time reached: " << write_time(elapsed_time) << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumTime;
        }

        if(training_loss <= training_loss_goal)
        {
            if(display) cout << "Epoch " << epoch << endl << "Loss goal reached: " << training_loss << endl;

            stop_training = true;

            results.stopping_condition  = StoppingCondition::LossGoal;
        }

        if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum selection failures reached: " << selection_failures << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }

        if(stop_training)
        {
            results.loss = training_back_propagation.loss;

            results.selection_failures = selection_failures;

            results.resize_training_error_history(epoch+1);

            if(has_selection) results.resize_selection_error_history(epoch+1);
            else results.resize_selection_error_history(0);

            results.elapsed_time = write_time(elapsed_time);

            break;
        }

        // Update stuff

        if(epoch != 0 && epoch%save_period == 0) neural_network_pointer->save(neural_network_file_name);
    }

    data_set_pointer->unscale_input_variables(input_variables_descriptives);

    if(neural_network_pointer->has_unscaling_layer())
        data_set_pointer->unscale_target_variables(target_variables_descriptives);

    if(display) results.print();

    return results;
}


string StochasticGradientDescent::write_optimization_algorithm_type() const
{
    return "STOCHASTIC_GRADIENT_DESCENT";
}


/// This method writes a matrix of strings the most representative atributes.

Tensor<string, 2> StochasticGradientDescent::to_string_matrix() const
{
    Tensor<string, 2> labels_values(7, 2);

    // Initial learning rate

    labels_values(0,0) = "Inital learning rate";
    labels_values(0,1) = to_string(double(initial_learning_rate));

    // Initial decay

    labels_values(1,0) = "Inital decay";
    labels_values(1,1) = to_string(double(initial_decay));

    // Momentum

    labels_values(2,0) = "Apply momentum";    
    momentum > type(0) ? labels_values(2,1) = "true" : labels_values(2,1) = "false";

    // Training loss goal

    labels_values(3,0) = "Training loss goal";
    labels_values(3,1) = to_string(double(training_loss_goal));

    // Maximum epochs number

    labels_values(4,0) = "Maximum epochs number";
    labels_values(4,1) = to_string(maximum_epochs_number);

    // Maximum time

    labels_values(5,0) = "Maximum time";
    labels_values(5,1) = write_time(maximum_time);

    // Batch samples number

    labels_values(6,0) = "Batch samples number";
    labels_values(6,1) = to_string(batch_samples_number);

    return labels_values;
}


/// Serializes the gradient descent object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void StochasticGradientDescent::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("StochasticGradientDescent");

    // DataSetBatch size

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

        throw invalid_argument(buffer.str());
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
        catch(const invalid_argument& e)
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
        catch(const invalid_argument& e)
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
            catch(const invalid_argument& e)
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
            catch(const invalid_argument& e)
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
            catch(const invalid_argument& e)
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
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
    }
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
