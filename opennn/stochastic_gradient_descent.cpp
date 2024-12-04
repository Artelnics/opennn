//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "stochastic_gradient_descent.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "language_data_set.h"

namespace opennn
{

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


void StochasticGradientDescent::set_batch_samples_number(const Index& new_batch_samples_number)
{
    batch_samples_number = new_batch_samples_number;
}


Index StochasticGradientDescent::get_batch_samples_number() const
{
    return batch_samples_number;
}


void StochasticGradientDescent::set_initial_learning_rate(const type& new_learning_rate)
{
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
    
    const vector<Index> input_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Target);

    const vector<Index> context_variable_indices = is_instance_of<LanguageDataSet>(data_set)
        ? static_cast<LanguageDataSet*>(data_set)->get_variable_indices(DataSet::VariableUse::Context)
        : vector<Index>();

    const vector<Index> training_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Training);
    const vector<Index> selection_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Selection);

    const Index training_samples_number = data_set->get_samples_number(DataSet::SampleUse::Training);
    const Index selection_samples_number = data_set->get_samples_number(DataSet::SampleUse::Selection);
        
    const Index training_batch_samples_number = min(training_samples_number, batch_samples_number);

    const Index selection_batch_samples_number = (selection_samples_number > 0)
         ? min(selection_samples_number, batch_samples_number)
         : 0;

    const vector<Scaler> input_variable_scalers = data_set->get_variable_scalers(DataSet::VariableUse::Input);
    const vector<Scaler> target_variable_scalers = data_set->get_variable_scalers(DataSet::VariableUse::Target);

    vector<Descriptives> input_variable_descriptives;
    vector<Descriptives> target_variable_descriptives;
    if(!is_instance_of<LanguageDataSet>(data_set))
        input_variable_descriptives = data_set->scale_variables(DataSet::VariableUse::Input);

    Batch training_batch(training_batch_samples_number, data_set);
    Batch selection_batch(selection_batch_samples_number, data_set);

    const Index training_batches_number = training_samples_number/training_batch_samples_number;
    const Index selection_batches_number = selection_samples_number/selection_batch_samples_number;

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> selection_batches(selection_batches_number);

    vector<Index> training_batch_indices(training_batch_samples_number);
    vector<Index> selection_batch_indices(training_batch_samples_number);
    
    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    set_names();

    set_scaling();

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
    time(&beginning_time);
    type elapsed_time = type(0);

    bool shuffle = false;

    if(neural_network->has(Layer::Type::LongShortTermMemory)
    || neural_network->has(Layer::Type::Recurrent))
        shuffle = false;

    // Main loop
    
    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = data_set->get_batches(training_samples_indices, training_batch_samples_number, shuffle);               

        const Index batches_number = training_batches.size();

        //training_loss = type(0);
        training_error = type(0);

        optimization_data.iteration = 0;
        
        for(Index iteration = 0; iteration < batches_number; iteration++)
        {
            optimization_data.iteration++;
            
            // Data set

            training_batch.fill(training_batches[iteration],
                                input_variable_indices,
                                target_variable_indices,
                                context_variable_indices);

            // Neural network
            
            neural_network->forward_propagate(training_batch.get_input_pairs(),
                                              training_forward_propagation,
                                              is_training);
            
            // Loss index
            
            loss_index->back_propagate(training_batch,
                                       training_forward_propagation,
                                       training_back_propagation);

            results.training_error_history(epoch) = training_back_propagation.error();

            training_error += training_back_propagation.error();
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

                selection_batch.fill(selection_batches[iteration],
                                     input_variable_indices,
                                     target_variable_indices);

                // Neural network
                
                neural_network->forward_propagate(selection_batch.get_input_pairs(),
                                                  selection_forward_propagation,
                                                  is_training);

                results.selection_error_history(epoch) = selection_error;

                // Loss

                loss_index->calculate_error(selection_batch,
                                            selection_forward_propagation, 
                                            selection_back_propagation);

                selection_error += selection_back_propagation.error();
            }

            selection_error /= type(selection_batches_number);

            results.selection_error_history(epoch) = selection_error;

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) selection_failures++;
        }
        
        // Elapsed time

        elapsed_time = get_elapsed_time(beginning_time);

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if(has_selection) cout << "Selection error: " << selection_error << endl<<endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        // Stopping criteria

        if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch << endl;
            stop_training = true;
            results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
        }
        else if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum training time reached: " << write_time(elapsed_time) << endl;
            stop_training = true;
            results.stopping_condition = StoppingCondition::MaximumTime;
        }
        else if(results.training_error_history(epoch) < training_loss_goal)
        {
            stop_training = true;
            results.stopping_condition  = StoppingCondition::LossGoal;
            if(display) cout << "Epoch " << epoch << "\nLoss goal reached: " << results.training_error_history(epoch) << endl;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << selection_failures << endl;
            stop_training = true;
            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }

        if(stop_training)
        {
            results.loss = training_back_propagation.loss;
            results.selection_failures = selection_failures;
            results.elapsed_time = write_time(elapsed_time);

            results.resize_training_error_history(epoch+1);
            results.resize_selection_error_history(has_selection ? epoch + 1 : 0);

            break;
        }

        // Update stuff

        if(epoch != 0 && epoch%save_period == 0)
            neural_network->save(neural_network_file_name);
    }

    set_unscaling();

    if(display) results.print();
    
    return results;
}


string StochasticGradientDescent::write_optimization_algorithm_type() const
{
    return "STOCHASTIC_GRADIENT_DESCENT";
}


Tensor<string, 2> StochasticGradientDescent::to_string_matrix() const
{
    Tensor<string, 2> string_matrix(7, 2);

    const string apply_momentum = momentum > type(0)
        ? "true"
        : "false";

    string_matrix.setValues({
    {"Inital learning rate", to_string(double(initial_learning_rate))},
    {"Inital decay", to_string(double(initial_decay))},
    {"Apply momentum", apply_momentum},
    {"Training loss goal", to_string(double(training_loss_goal))},
    {"Maximum epochs number", to_string(maximum_epochs_number)},
    {"Maximum time", write_time(maximum_time)},
    {"Batch samples number", to_string(batch_samples_number)}});

    return string_matrix;
}


void StochasticGradientDescent::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("StochasticGradientDescent");

    add_xml_element(printer, "BatchSize", to_string(batch_samples_number));
    add_xml_element(printer, "ApplyMomentum", to_string(momentum > type(0)));
    add_xml_element(printer, "LossGoal", to_string(training_loss_goal));
    add_xml_element(printer, "MaximumEpochsNumber", to_string(maximum_epochs_number));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));
    add_xml_element(printer, "HardwareUse", hardware_use);

    printer.CloseElement();
}


void StochasticGradientDescent::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("StochasticGradientDescent");

    if(!root_element)
        throw runtime_error("Stochastic gradient descent element is nullptr.\n");

    set_batch_samples_number(read_xml_index(root_element, "BatchSize"));

    const bool apply_momentum = read_xml_bool(root_element, "ApplyMomentum");
    set_momentum(apply_momentum ? type(0.9) : type(0));

    set_loss_goal(read_xml_type(root_element, "LossGoal"));
    set_maximum_epochs_number(read_xml_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
    set_hardware_use(read_xml_string(root_element, "HardwareUse"));
}


StochasticGradientDescentData::StochasticGradientDescentData(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    set(new_stochastic_gradient_descent);
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
