//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "stochastic_gradient_descent.h"
#include "neural_network_forward_propagation.h"
#include "back_propagation.h"
#include "language_data_set.h"

namespace opennn
{

StochasticGradientDescent::StochasticGradientDescent()
    :OptimizationAlgorithm()
{
    set_default();
}


StochasticGradientDescent::StochasticGradientDescent(LossIndex* new_loss_index)
    : OptimizationAlgorithm(new_loss_index)
{
    set_default();
}


const type& StochasticGradientDescent::get_initial_learning_rate() const
{
    return initial_learning_rate;
}


const type& StochasticGradientDescent::get_initial_decay() const
{
    return initial_decay;
}


const type& StochasticGradientDescent::get_momentum() const
{
    return momentum;
}


const bool& StochasticGradientDescent::get_nesterov() const
{
    return nesterov;
}


const type& StochasticGradientDescent::get_loss_goal() const
{
    return training_loss_goal;
}


const type& StochasticGradientDescent::get_maximum_time() const
{
    return maximum_time;
}


void StochasticGradientDescent::set_loss_index(LossIndex* new_loss_index)
{
    loss_index = new_loss_index;
}


void StochasticGradientDescent::set_default()
{
    // TRAINING OPERATORS

    initial_learning_rate = type(0.01);
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


void StochasticGradientDescent::set_initial_learning_rate(const type& new_learning_rate)
{
#ifdef OPENNN_DEBUG

    if(new_learning_rate <= type(0))
        throw runtime_error("initial_learning_rate must be greater than 0.\n");

#endif

    initial_learning_rate = new_learning_rate;
}


void StochasticGradientDescent::set_initial_decay(const type& new_decay)
{
    initial_decay = new_decay;
}


void StochasticGradientDescent::set_momentum(const type& new_momentum)
{
    momentum = new_momentum;
}


void StochasticGradientDescent::set_nesterov(const bool& new_nesterov_momentum)
{
    nesterov = new_nesterov_momentum;
}


void StochasticGradientDescent::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


void StochasticGradientDescent::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


void StochasticGradientDescent::set_maximum_time(const type& new_maximum_time)
{
    maximum_time = new_maximum_time;
}


void StochasticGradientDescent::update_parameters(BackPropagation& back_propagation,
                      StochasticGradientDescentData& optimization_data) const
{
    NeuralNetwork* neural_network = loss_index->get_neural_network();

    Tensor<type, 1>& parameters = back_propagation.parameters;
    const Tensor<type, 1>& gradient = back_propagation.gradient;

    Tensor<type, 1>& parameters_increment = optimization_data.parameters_increment;
    Tensor<type, 1>& last_parameters_increment = optimization_data.last_parameters_increment;
    
    const type learning_rate = initial_learning_rate/(type(1) + type(optimization_data.iteration)*initial_decay);
    
    if(momentum <= type(0))
    {
        parameters_increment.device(*thread_pool_device) = gradient * (-learning_rate);

        parameters.device(*thread_pool_device) += parameters_increment;
    }    
    else if(momentum > type(0) && !nesterov)
    {
        parameters_increment.device(*thread_pool_device) =
            gradient * (-learning_rate) + momentum * last_parameters_increment;

        last_parameters_increment.device(*thread_pool_device) = parameters_increment;

        parameters.device(*thread_pool_device) += parameters_increment;

    }
    else if(momentum > type(0) && nesterov)
    {
        parameters_increment.device(*thread_pool_device)
            = gradient * (-learning_rate) + momentum * last_parameters_increment;

        last_parameters_increment.device(*thread_pool_device) = parameters_increment;

        parameters.device(*thread_pool_device) += parameters_increment * momentum - gradient * learning_rate;
    }

    optimization_data.iteration++;

    // Update parameters

    neural_network->set_parameters(parameters);
}


TrainingResults StochasticGradientDescent::perform_training()
{
    TrainingResults results(maximum_epochs_number+1);
    
    check();

    // Start training

    if(display) cout << "Training with stochastic gradient descent (SGD)...\n";

    // Data set

    DataSet* data_set = loss_index->get_data_set();

    const bool has_selection = data_set->has_selection();
    
    const Tensor<Index, 1> input_variables_indices = data_set->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set->get_target_variables_indices();
    Tensor<Index, 1> context_variables_indices;

    if(is_instance_of<LanguageDataSet>(data_set))
    {
        LanguageDataSet* language_data_set = static_cast<LanguageDataSet*>(data_set);
        context_variables_indices = language_data_set->get_context_variables_indices();
    }
        
    const Tensor<Index, 1> training_samples_indices = data_set->get_training_samples_indices();
    const Tensor<Index, 1> selection_samples_indices = data_set->get_selection_samples_indices();

    Index training_batch_samples_number = 0;
    Index selection_batch_samples_number = 0;

    const Index training_samples_number = data_set->get_training_samples_number();
    const Index selection_samples_number = data_set->get_selection_samples_number();
    
    training_samples_number < batch_samples_number
            ? training_batch_samples_number = training_samples_number
            : training_batch_samples_number = batch_samples_number;

    selection_samples_number < batch_samples_number && selection_samples_number != 0
            ? selection_batch_samples_number = selection_samples_number
            : selection_batch_samples_number = batch_samples_number;

    const Tensor<string, 1> inputs_name = data_set->get_input_variables_names();
    const Tensor<string, 1> targets_names = data_set->get_target_variables_names();

    const Tensor<Scaler, 1> input_variables_scalers = data_set->get_input_variables_scalers();
    const Tensor<Scaler, 1> target_variables_scalers = data_set->get_target_variables_scalers();

    Tensor<Descriptives, 1> input_variables_descriptives;
    Tensor<Descriptives, 1> target_variables_descriptives;
    if(!is_instance_of<LanguageDataSet>(data_set))
        input_variables_descriptives = data_set->scale_input_variables();

    Batch training_batch(training_batch_samples_number, data_set);
    Batch selection_batch(selection_batch_samples_number, data_set);
    
    const Tensor<pair<type*, dimensions>, 1> training_inputs = training_batch.get_inputs_pair();

    const Index training_batches_number = training_samples_number/training_batch_samples_number;
    const Index selection_batches_number = selection_samples_number/selection_batch_samples_number;

    Tensor<Index, 2> training_batches(training_batches_number, training_batch_samples_number);
    Tensor<Index, 2> selection_batches(selection_batches_number, selection_batch_samples_number);

    Tensor<Index, 1> training_batch_indices(training_batch_samples_number);
    Tensor<Index, 1> selection_batch_indices(training_batch_samples_number);
    
    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    neural_network->set_inputs_names(inputs_name);
    neural_network->set_output_namess(targets_names);

    if(neural_network->has_scaling_layer())
    {
        if(neural_network->has_scaling_4d_layer())
        {
            ScalingLayer4D* scaling_layer_4d = neural_network->get_scaling_layer_4d();
            scaling_layer_4d->set(input_variables_descriptives, input_variables_scalers);
        }
        else
        {
            ScalingLayer2D* scaling_layer_2d = neural_network->get_scaling_layer_2d();
            scaling_layer_2d->set(input_variables_descriptives, input_variables_scalers);
        }
    }

    if(neural_network->has_unscaling_layer())
    {
        target_variables_descriptives = data_set->scale_target_variables();

        UnscalingLayer* unscaling_layer = neural_network->get_unscaling_layer();
        unscaling_layer->set(target_variables_descriptives, target_variables_scalers);
    }

    ForwardPropagation training_forward_propagation(training_batch_samples_number, neural_network);
    ForwardPropagation selection_forward_propagation(selection_batch_samples_number, neural_network);
    
    // Loss index

    loss_index->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_samples_number, loss_index);
    BackPropagation selection_back_propagation(selection_batch_samples_number, loss_index);

    //type training_loss = type(0);
    type training_error = type(0);
    type selection_error = type(0);

    Index selection_failures = 0;
    
    // Optimization algorithm

    StochasticGradientDescentData optimization_data(this);
    
    bool stop_training = false;
    bool is_training = true;

    time_t beginning_time;
    time_t current_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    bool shuffle = false;

    if(neural_network->has_long_short_term_memory_layer()
    || neural_network->has_recurrent_layer())
        shuffle = false;

    // Main loop
    
    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = data_set->get_batches(training_samples_indices, training_batch_samples_number, shuffle);               

        const Index batches_number = training_batches.dimension(0);

        //training_loss = type(0);
        training_error = type(0);

        optimization_data.iteration = 0;
        
        for(Index iteration = 0; iteration < batches_number; iteration++)
        {
            optimization_data.iteration++;
            
            // Data set

            training_batch_indices = training_batches.chip(iteration, 0);

            training_batch.fill(training_batch_indices,
                                input_variables_indices,
                                target_variables_indices,
                                context_variables_indices);

            // Neural network
            
            neural_network->forward_propagate(training_inputs,
                                              training_forward_propagation,
                                              is_training);
            
            // Loss index
            
            loss_index->back_propagate(training_batch,
                                       training_forward_propagation,
                                       training_back_propagation);

            results.training_error_history(epoch) = training_back_propagation.error;

            training_error += training_back_propagation.error;
            //training_loss += training_back_propagation.loss;
            
            // Gradient
            
            update_parameters(training_back_propagation, optimization_data);
            
            //if(display && epoch % display_period == 0)      display_progress_bar(iteration, batches_number - 1);
        }
        
        // Loss

        //training_loss /= type(batches_number);
        training_error /= type(batches_number);

        results.training_error_history(epoch) = training_error;
        
        if(has_selection)
        {
            selection_batches = data_set->get_batches(selection_samples_indices, selection_batch_samples_number, shuffle);

            selection_error = type(0);

            for(Index iteration = 0; iteration < selection_batches_number; iteration++)
            {
                // Data set

                selection_batch_indices = selection_batches.chip(iteration,0);

                selection_batch.fill(selection_batch_indices,
                                     input_variables_indices,
                                     target_variables_indices);

                // Neural network
                
                neural_network->forward_propagate(selection_batch.get_inputs_pair(),
                                                          selection_forward_propagation,
                                                          is_training);

                results.selection_error_history(epoch) = selection_error;

                // Loss

                loss_index->calculate_error(selection_batch, selection_forward_propagation, selection_back_propagation);

                selection_error += selection_back_propagation.error;
            }

            selection_error /= type(selection_batches_number);

            results.selection_error_history(epoch) = selection_error;

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) selection_failures++;
        }
        
        // Elapsed time

        time(&current_time);
        elapsed_time = type(difftime(current_time, beginning_time));

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

        if(results.training_error_history(epoch) < training_loss_goal)
        {
            stop_training = true;

            results.stopping_condition  = StoppingCondition::LossGoal;

            if(display) cout << "Epoch " << epoch << endl << "Loss goal reached: " << results.training_error_history(epoch) << endl;
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

            results.resize_selection_error_history(has_selection ? epoch + 1 : 0);

            results.elapsed_time = write_time(elapsed_time);

            break;
        }

        // Update stuff

        if(epoch != 0 && epoch%save_period == 0) neural_network->save(neural_network_file_name);
    }

    if(!is_instance_of<LanguageDataSet>(data_set))
        data_set->unscale_input_variables(input_variables_descriptives);

    if(neural_network->has_unscaling_layer())
        data_set->unscale_target_variables(target_variables_descriptives);

    if(display) results.print();
    
    return results;
}


string StochasticGradientDescent::write_optimization_algorithm_type() const
{
    return "STOCHASTIC_GRADIENT_DESCENT";
}


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


void StochasticGradientDescent::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("StochasticGradientDescent");

    // Batch size

    file_stream.OpenElement("BatchSize");
    file_stream.PushText(to_string(batch_samples_number).c_str());
    file_stream.CloseElement();

    // Apply momentum

    file_stream.OpenElement("ApplyMomentum");
    file_stream.PushText(to_string(momentum > type(0)).c_str());
    file_stream.CloseElement();

    // Loss goal

    file_stream.OpenElement("LossGoal");
    file_stream.PushText(to_string(training_loss_goal).c_str());
    file_stream.CloseElement();

    // Maximum epochs number

    file_stream.OpenElement("MaximumEpochsNumber");
    file_stream.PushText(to_string(maximum_epochs_number).c_str());
    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");
    file_stream.PushText(to_string(maximum_time).c_str());
    file_stream.CloseElement();

    // Hardware use

    file_stream.OpenElement("HardwareUse");
    file_stream.PushText(hardware_use.c_str());
    file_stream.CloseElement();

    // End element

    file_stream.CloseElement();
}


void StochasticGradientDescent::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("StochasticGradientDescent");

    if(!root_element)
        throw runtime_error("Stochastic gradient descent element is nullptr.\n");

    // Batch size

    const tinyxml2::XMLElement* batch_samples_number_element = root_element->FirstChildElement("BatchSize");

    if(batch_samples_number_element)
        set_batch_samples_number(Index(atoi(batch_samples_number_element->GetText())));

    // Momentum

    const tinyxml2::XMLElement* apply_momentum_element = root_element->FirstChildElement("ApplyMomentum");

    if(batch_samples_number_element)
    {
        string new_apply_momentum_state = apply_momentum_element->GetText();

        try
        {
            if(new_apply_momentum_state != "0")
            {
                set_momentum(type(0.9));
            }
            else
            {
                set_momentum(type(0));
            }
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Loss goal

    const tinyxml2::XMLElement* loss_goal_element = root_element->FirstChildElement("LossGoal");

    if(loss_goal_element)
        set_loss_goal(type(atof(loss_goal_element->GetText())));

    // Maximum epochs number

    const tinyxml2::XMLElement* maximum_epochs_number_element = root_element->FirstChildElement("MaximumEpochsNumber");

    if(maximum_epochs_number_element)
        set_maximum_epochs_number(Index(atoi(maximum_epochs_number_element->GetText())));

    // Maximum time

    const tinyxml2::XMLElement* maximum_time_element = root_element->FirstChildElement("MaximumTime");

    if(maximum_time_element)
        set_maximum_time(type(atof(maximum_time_element->GetText())));

    // Hardware use

    const tinyxml2::XMLElement* hardware_use_element = root_element->FirstChildElement("HardwareUse");

    if(hardware_use_element)
        set_hardware_use(hardware_use_element->GetText());
}


void StochasticGradientDescentData::set(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    stochastic_gradient_descent = new_stochastic_gradient_descent;

    const LossIndex* loss_index = stochastic_gradient_descent->get_loss_index();

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_number();

    parameters_increment.resize(parameters_number);
    last_parameters_increment.resize(parameters_number);

    parameters_increment.setZero();
    last_parameters_increment.setZero();
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
