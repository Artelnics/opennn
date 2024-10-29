//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R A D I E N T   D E S C E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "gradient_descent.h"
#include "neural_network_forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

GradientDescent::GradientDescent()
    : OptimizationAlgorithm()
{
    set_default();
}


GradientDescent::GradientDescent(LossIndex* new_loss_index)
    : OptimizationAlgorithm(new_loss_index)
{
    learning_rate_algorithm.set_loss_index(new_loss_index);

    set_default();
}


const LearningRateAlgorithm& GradientDescent::get_learning_rate_algorithm() const
{
    return learning_rate_algorithm;
}


LearningRateAlgorithm* GradientDescent::get_learning_rate_algorithm()
{
    return &learning_rate_algorithm;
}


const type& GradientDescent::get_minimum_loss_decrease() const
{
    return minimum_loss_decrease;
}


const type& GradientDescent::get_loss_goal() const
{
    return training_loss_goal;
}


const Index& GradientDescent::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


const Index& GradientDescent::get_maximum_epochs_number() const
{
    return maximum_epochs_number;
}


const type& GradientDescent::get_maximum_time() const
{
    return maximum_time;
}


void GradientDescent::set_loss_index(LossIndex* new_loss_index)
{
    loss_index = new_loss_index;

    learning_rate_algorithm.set_loss_index(new_loss_index);
}


void GradientDescent::set_default()
{
    // Stopping criteria

    minimum_loss_decrease = type(0);

    training_loss_goal = type(0);
    maximum_selection_failures = numeric_limits<Index>::max();

    maximum_epochs_number = 1000;
    maximum_time = type(3600);

    // UTILITIES

    display_period = 10;
}


void GradientDescent::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


void GradientDescent::set_minimum_loss_decrease(const type& new_minimum_loss_decrease)
{
    minimum_loss_decrease = new_minimum_loss_decrease;
}


void GradientDescent::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


void GradientDescent::set_maximum_selection_failures(const Index& new_maximum_selection_failures)
{
    maximum_selection_failures = new_maximum_selection_failures;
}


void GradientDescent::set_maximum_time(const type& new_maximum_time)
{
    maximum_time = new_maximum_time;
}


void GradientDescent::calculate_training_direction(const Tensor<type, 1>& gradient, Tensor<type, 1>& training_direction) const
{
    training_direction.device(*thread_pool_device) = -gradient;
}


void GradientDescent::update_parameters(
        const Batch& batch,
        ForwardPropagation& forward_propagation,
        BackPropagation& back_propagation,
        GradientDescentData& optimization_data) const
{
    NeuralNetwork* neural_network = back_propagation.loss_index->get_neural_network();

    calculate_training_direction(back_propagation.gradient, optimization_data.training_direction);

    // Get initial learning_rate

    optimization_data.epoch == 0
            ? optimization_data.initial_learning_rate = first_learning_rate
            : optimization_data.initial_learning_rate = optimization_data.old_learning_rate;

    const pair<type, type> directional_point = learning_rate_algorithm.calculate_directional_point(
                            batch,
                            forward_propagation,
                            back_propagation,
                            optimization_data);

    optimization_data.learning_rate = directional_point.first;
    back_propagation.loss = directional_point.second;

    if(abs(optimization_data.learning_rate) > type(0))
    {
        back_propagation.parameters.device(*thread_pool_device)
            -= back_propagation.gradient*optimization_data.learning_rate;
    }
    else
    {
        const type epsilon = std::numeric_limits<type>::epsilon();

        const Index parameters_number = neural_network->get_parameters_number();

        #pragma omp parallel for

        for (Index i = 0; i < parameters_number; i++)
            if (std::abs(back_propagation.gradient(i)) >= type(NUMERIC_LIMITS_MIN))
                back_propagation.parameters(i) += (back_propagation.gradient(i) > type(0)) 
                    ? -epsilon 
                    : epsilon;
        
        optimization_data.learning_rate = optimization_data.old_learning_rate;
    }

    // Update parameters

    optimization_data.old_learning_rate = optimization_data.learning_rate;

    forward_propagation.neural_network->set_parameters(back_propagation.parameters);
}


TrainingResults GradientDescent::perform_training()
{
    TrainingResults results(maximum_epochs_number+1);

    // Start training

    if(display) cout << "Training with gradient descent...\n";

    // Data set

    DataSet* data_set = loss_index->get_data_set();

    const Index training_samples_number = data_set->get_samples_number(DataSet::SampleUse::Training);
    const Index selection_samples_number = data_set->get_samples_number(DataSet::SampleUse::Selection);

    const bool has_selection = data_set->has_selection();

    const Tensor<Index, 1> training_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Training);
    const Tensor<Index, 1> selection_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Selection);

    const Tensor<Index, 1> input_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Input);
    const Tensor<Index, 1> target_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Target);

    const Tensor<string, 1> input_names = data_set->get_variable_names(DataSet::VariableUse::Input);
    const Tensor<string, 1> target_names = data_set->get_variable_names(DataSet::VariableUse::Target);

    const Tensor<Scaler, 1> input_variables_scalers = data_set->get_variable_scalers(DataSet::VariableUse::Input);
    const Tensor<Scaler, 1> target_variables_scalers = data_set->get_variable_scalers(DataSet::VariableUse::Target);

    const Tensor<Descriptives, 1> input_variables_descriptives = data_set->scale_variables(DataSet::VariableUse::Input);
    Tensor<Descriptives, 1> target_variables_descriptives;

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    neural_network->set_inputs_names(input_names);
    neural_network->set_output_namess(target_names);

    if(neural_network->has(Layer::Type::Scaling2D))
    {
        ScalingLayer2D* scaling_layer_2d = neural_network->get_scaling_layer_2d();

        scaling_layer_2d->set_descriptives(input_variables_descriptives);
        scaling_layer_2d->set_scalers(input_variables_scalers);
    }

    if(neural_network->has(Layer::Type::Unscaling))
    {
        target_variables_descriptives = data_set->scale_variables(DataSet::VariableUse::Target);

        UnscalingLayer* unscaling_layer = neural_network->get_unscaling_layer();
        unscaling_layer->set(target_variables_descriptives, target_variables_scalers);
    }

    ForwardPropagation training_forward_propagation(training_samples_number, neural_network);
    ForwardPropagation selection_forward_propagation(selection_samples_number, neural_network);

    Batch training_batch(training_samples_number, data_set);
    training_batch.fill(training_samples_indices, input_variable_indices, target_variable_indices);

    Batch selection_batch(selection_samples_number, data_set);
    selection_batch.fill(selection_samples_indices, input_variable_indices, target_variable_indices);

    // Loss index

    const string error_type = loss_index->get_loss_method();

    loss_index->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_samples_number, loss_index);
    BackPropagation selection_back_propagation(selection_samples_number, loss_index);

    // Optimization algorithm

    GradientDescentData optimization_data(this);

    Index selection_failures = 0;

    bool stop_training = false;
    bool is_training = true;

    type old_loss = type(0);
    type loss_decrease = numeric_limits<type>::max();

    // Main loop

    time_t beginning_time;
    time_t current_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        optimization_data.epoch = epoch;

        // Neural network
        
        neural_network->forward_propagate(training_batch.get_input_pairs(),
                                          training_forward_propagation, 
                                          is_training);

        // Loss index

        loss_index->back_propagate(training_batch, 
                                   training_forward_propagation, 
                                   training_back_propagation);

        results.training_error_history(epoch) = training_back_propagation.error();


        // Update parameters

        update_parameters(training_batch, training_forward_propagation, training_back_propagation, optimization_data);

        if(has_selection)
        {
            neural_network->forward_propagate(selection_batch.get_input_pairs(),
                                              selection_forward_propagation, 
                                              is_training);

            loss_index->calculate_error(selection_batch, 
                                        selection_forward_propagation, 
                                        selection_back_propagation);

            results.selection_error_history(epoch) = selection_back_propagation.error();

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) selection_failures++;
        }

        // Optimization algorithm

        time(&current_time);
        elapsed_time = type(difftime(current_time, beginning_time));

        // Print progress

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_back_propagation.error << endl;
            if(has_selection) cout << "Selection error: " << selection_back_propagation.error << endl;
            cout << "Learning rate: " << optimization_data.learning_rate << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        // Stopping Criteria

        if(results.training_error_history(epoch) < training_loss_goal)
        {
            stop_training = true;

            results.stopping_condition = StoppingCondition::LossGoal;

            if(display) cout << "Epoch " << epoch << endl << "Loss goal reached: " << results.training_error_history(epoch) << endl;
        }

        else if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum selection failures reached: " << selection_failures << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }

        else if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum epochs number reached: " << epoch << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
        }

        else if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum training time reached: " << elapsed_time;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumTime;
        }

        if(epoch != 0) loss_decrease = old_loss - training_back_propagation.loss;

        if(loss_decrease < minimum_loss_decrease)
        {
            if(display) cout << "Epoch " << epoch << endl << "Minimum loss decrease reached: " << loss_decrease << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MinimumLossDecrease;
        }

        old_loss = training_back_propagation.loss;

        if(stop_training)
        {
            results.loss = training_back_propagation.loss;

            results.loss_decrease = loss_decrease;

            results.selection_failures = selection_failures;

            results.resize_training_error_history(epoch+1);

            results.resize_selection_error_history(has_selection ? epoch + 1 : 0);

            results.elapsed_time = write_time(elapsed_time);

            break;
        }

        if(epoch != 0 && epoch%save_period == 0) neural_network->save(neural_network_file_name);
    }

    data_set->unscale_variables(DataSet::VariableUse::Input, input_variables_descriptives);

    if(neural_network->has(Layer::Type::Unscaling))
        data_set->unscale_variables(DataSet::VariableUse::Target, target_variables_descriptives);

    if(display) results.print();

    return results;
}


string GradientDescent::write_optimization_algorithm_type() const
{
    return "GRADIENT_DESCENT";
}


Tensor<string, 2> GradientDescent::to_string_matrix() const
{
    Tensor<string, 2> labels_values(7, 2);

    labels_values(0,0) = "Learning rate method";
    labels_values(0,1) = learning_rate_algorithm.write_learning_rate_method();

    labels_values(1,0) = "Learning rate tolerance";
    labels_values(1,1) = to_string(double(learning_rate_algorithm.get_learning_rate_tolerance()));

    labels_values(2,0) = "Minimum loss decrease";
    labels_values(2,1) = to_string(double(minimum_loss_decrease));

    labels_values(3,0) = "Loss goal";
    labels_values(3,1) = to_string(double(training_loss_goal));

    labels_values(4,0) = "Maximum selection error increases";
    labels_values(4,1) = to_string(maximum_selection_failures);

    labels_values(5,0) = "Maximum epochs number";
    labels_values(5,1) = to_string(maximum_epochs_number);

    labels_values(6,0) = "Maximum time";
    labels_values(6,1) = write_time(maximum_time);

    return labels_values;
}


void GradientDescent::to_XML(tinyxml2::XMLPrinter& printer) const
{
    printer.OpenElement("GradientDescent");

    learning_rate_algorithm.to_XML(printer);

    add_xml_element(printer, "MinimumLossDecrease", std::to_string(minimum_loss_decrease));
    add_xml_element(printer, "LossGoal", std::to_string(training_loss_goal));
    add_xml_element(printer, "MaximumSelectionFailures", std::to_string(maximum_selection_failures));
    add_xml_element(printer, "MaximumEpochsNumber", std::to_string(maximum_epochs_number));
    add_xml_element(printer, "MaximumTime", std::to_string(maximum_time));

    printer.CloseElement();  
}


void GradientDescent::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GradientDescent");

    if(!root_element)
        throw runtime_error("Gradient descent element is nullptr.\n");

    // Learning rate algorithm

    const tinyxml2::XMLElement* learning_rate_algorithm_element
            = root_element->FirstChildElement("LearningRateAlgorithm");

    if(learning_rate_algorithm_element)
    {
        tinyxml2::XMLDocument learning_rate_algorithm_document;
        learning_rate_algorithm_document.InsertFirstChild(learning_rate_algorithm_element->DeepClone(&learning_rate_algorithm_document));
        learning_rate_algorithm.from_XML(learning_rate_algorithm_document);
    }

    set_minimum_loss_decrease(read_xml_type(root_element, "MinimumLossDecrease"));
    set_loss_goal(read_xml_type(root_element, "LossGoal"));
    set_maximum_selection_failures(read_xml_index(root_element, "MaximumSelectionFailures"));
    set_maximum_epochs_number(read_xml_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
}


void GradientDescentData::set(GradientDescent* new_gradient_descent)
{
    gradient_descent = new_gradient_descent;

    const LossIndex* loss_index = gradient_descent->get_loss_index();

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_number();

    // Neural network data

    potential_parameters.resize(parameters_number);

    // Optimization algorithm data

    training_direction.resize(parameters_number);
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
