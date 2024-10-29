//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "levenberg_marquardt_algorithm.h"
#include "neural_network_forward_propagation.h"

namespace opennn
{

LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithm()
    : OptimizationAlgorithm()
{
    set_default();
}


LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithm(LossIndex* new_loss_index)
    : OptimizationAlgorithm(new_loss_index)
{
    set_default();
}


const type& LevenbergMarquardtAlgorithm::get_minimum_loss_decrease() const
{
    return minimum_loss_decrease;
}


const type& LevenbergMarquardtAlgorithm::get_loss_goal() const
{
    return training_loss_goal;
}


const Index& LevenbergMarquardtAlgorithm::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


const Index& LevenbergMarquardtAlgorithm::get_maximum_epochs_number() const
{
    return maximum_epochs_number;
}


const type& LevenbergMarquardtAlgorithm::get_maximum_time() const
{
    return maximum_time;
}


const type& LevenbergMarquardtAlgorithm::get_damping_parameter() const
{
    return damping_parameter;
}


const type& LevenbergMarquardtAlgorithm::get_damping_parameter_factor() const
{
    return damping_parameter_factor;
}


const type& LevenbergMarquardtAlgorithm::get_minimum_damping_parameter() const
{
    return minimum_damping_parameter;
}


const type& LevenbergMarquardtAlgorithm::get_maximum_damping_parameter() const
{
    return maximum_damping_parameter;
}


void LevenbergMarquardtAlgorithm::set_default()
{
    // Stopping criteria

    minimum_loss_decrease = type(0);
    training_loss_goal = type(0);
    maximum_selection_failures = 1000;

    maximum_epochs_number = 1000;
    maximum_time = type(3600.0);

    // UTILITIES

    display_period = 10;

    // Training parameters

    damping_parameter = type(1.0e-3);

    damping_parameter_factor = type(10.0);

    minimum_damping_parameter = type(1.0e-6);
    maximum_damping_parameter = type(1.0e6);
}


void LevenbergMarquardtAlgorithm::set_damping_parameter(const type& new_damping_parameter)
{
    damping_parameter = clamp(new_damping_parameter, minimum_damping_parameter, maximum_damping_parameter);
}


void LevenbergMarquardtAlgorithm::set_damping_parameter_factor(const type& new_damping_parameter_factor)
{
    damping_parameter_factor = new_damping_parameter_factor;
}


void LevenbergMarquardtAlgorithm::set_minimum_damping_parameter(const type& new_minimum_damping_parameter)
{
    minimum_damping_parameter = new_minimum_damping_parameter;
}


void LevenbergMarquardtAlgorithm::set_maximum_damping_parameter(const type& new_maximum_damping_parameter)
{
    maximum_damping_parameter = new_maximum_damping_parameter;
}


void LevenbergMarquardtAlgorithm::set_minimum_loss_decrease(const type& new_minimum_loss_decrease)
{
    minimum_loss_decrease = new_minimum_loss_decrease;
}


void LevenbergMarquardtAlgorithm::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


void LevenbergMarquardtAlgorithm::set_maximum_selection_failures(
        const Index& new_maximum_selection_failures)
{
    maximum_selection_failures = new_maximum_selection_failures;
}


void LevenbergMarquardtAlgorithm::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


void LevenbergMarquardtAlgorithm::set_maximum_time(const type& new_maximum_time)
{
    maximum_time = new_maximum_time;
}


void LevenbergMarquardtAlgorithm::check() const
{
    if(!loss_index)
        throw runtime_error("Pointer to loss index is nullptr.\n");

    const DataSet* data_set = loss_index->get_data_set();

    if(!data_set)
        throw runtime_error("The loss funcional has no data set.");

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    if(!neural_network)
        throw runtime_error("Pointer to neural network is nullptr.");
}


TrainingResults LevenbergMarquardtAlgorithm::perform_training()
{
    if(loss_index->get_loss_method() == "MINKOWSKI_ERROR")
        throw runtime_error("Levenberg-Marquard algorithm cannot work with Minkowski error.");
    else if(loss_index->get_loss_method() == "CROSS_ENTROPY_ERROR")
        throw runtime_error("Levenberg-Marquard algorithm cannot work with cross-entropy error.");
    else if(loss_index->get_loss_method() == "WEIGHTED_SQUARED_ERROR")
        throw runtime_error("Levenberg-Marquard algorithm is not implemented yet with weighted squared error.");

    // Start training

    if(display) cout << "Training with Levenberg-Marquardt algorithm...\n";

    TrainingResults results(maximum_epochs_number+1);

    // Data set

    DataSet* data_set = loss_index->get_data_set();
    
    const bool has_selection = data_set->has_selection();

    const Index training_samples_number = data_set->get_samples_number(DataSet::SampleUse::Training);
    const Index selection_samples_number = data_set->get_samples_number(DataSet::SampleUse::Selection);

    const Tensor<Index, 1> training_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Training);
    const Tensor<Index, 1> selection_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Selection);

    const Tensor<Index, 1> input_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Input);
    const Tensor<Index, 1> target_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Target);

    const Tensor<string, 1> input_names = data_set->get_variable_names(DataSet::VariableUse::Input);
    const Tensor<string, 1> target_names = data_set->get_variable_names(DataSet::VariableUse::Target);

    const Tensor<Scaler, 1> input_variables_scalers = data_set->get_variable_scalers(DataSet::VariableUse::Input);
    const Tensor<Scaler, 1> target_variables_scalers = data_set->get_variable_scalers(DataSet::VariableUse::Target);

    Tensor<Descriptives, 1> input_variables_descriptives;
    Tensor<Descriptives, 1> target_variables_descriptives;

    // Neural network
    
    NeuralNetwork* neural_network = loss_index->get_neural_network();
    
    neural_network->set_inputs_names(input_names);
    neural_network->set_output_namess(target_names);

    if(neural_network->has(Layer::Type::Scaling2D))
    {
        input_variables_descriptives = data_set->scale_variables(DataSet::VariableUse::Input);

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
    
    Batch training_batch(training_samples_number, data_set);
    training_batch.fill(training_samples_indices, input_variable_indices, target_variable_indices);

    Batch selection_batch(selection_samples_number, data_set);
    selection_batch.fill(selection_samples_indices, input_variable_indices, target_variable_indices);

    ForwardPropagation training_forward_propagation(training_samples_number, neural_network);
    ForwardPropagation selection_forward_propagation(selection_samples_number, neural_network);
    
    // Loss index

    loss_index->set_normalization_coefficient();

    type old_loss = type(0);
    type loss_decrease = numeric_limits<type>::max();

    Index selection_failures = 0;
    
    BackPropagationLM training_back_propagation_lm(training_samples_number, loss_index);
    BackPropagationLM selection_back_propagation_lm(selection_samples_number, loss_index);

    // Training strategy stuff

    bool stop_training = false;
    bool is_training = true;

    time_t beginning_time;
    time_t current_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    LevenbergMarquardtAlgorithmData optimization_data(this);

    // Main loop

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        optimization_data.epoch = epoch;

        // Neural network
        
        neural_network->forward_propagate(training_batch,
                                          training_forward_propagation,
                                          is_training);
        
        // Loss index
        
        loss_index->back_propagate_lm(training_batch,
                                      training_forward_propagation,
                                      training_back_propagation_lm);
        
        results.training_error_history(epoch) = training_back_propagation_lm.error();
        
        if(has_selection)
        {           
            neural_network->forward_propagate(selection_batch,
                                              selection_forward_propagation,
                                              is_training);

            loss_index->calculate_errors_lm(selection_batch,
                                            selection_forward_propagation,
                                            selection_back_propagation_lm);

            loss_index->calculate_squared_errors_lm(selection_batch,
                                                    selection_forward_propagation,
                                                    selection_back_propagation_lm);

            loss_index->calculate_error_lm(selection_batch,
                                           selection_forward_propagation,
                                           selection_back_propagation_lm);

            results.selection_error_history(epoch) = selection_back_propagation_lm.error();

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) 
                selection_failures++;
        }
        
        // Elapsed time

        time(&current_time);
        elapsed_time = type(difftime(current_time, beginning_time));

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_back_propagation_lm.error << endl;
            if(has_selection) cout << "Selection error: " << selection_back_propagation_lm.error << endl;
            cout << "Damping parameter: " << damping_parameter << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        if(results.training_error_history(epoch) < training_loss_goal)
        {
            stop_training = true;

            results.stopping_condition = StoppingCondition::LossGoal;

            if(display) cout << "Epoch " << epoch << "\nLoss goal reached: " << results.training_error_history(epoch) << endl;
        }

        if(epoch != 0) loss_decrease = old_loss - training_back_propagation_lm.loss;

        if(loss_decrease < minimum_loss_decrease)
        {
            if(display) cout << "Epoch " << epoch << endl << "Minimum loss decrease reached: " << loss_decrease << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MinimumLossDecrease;
        }

        old_loss = training_back_propagation_lm.loss;

        if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << "Maximum selection failures reached: " << selection_failures << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }

        if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum epochs number reached: " << epoch << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
        }

        if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << "Maximum training time reached: " << elapsed_time << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumTime;
        }

        if(stop_training)
        {
            results.loss = training_back_propagation_lm.loss;
            results.loss_decrease = loss_decrease;
            results.selection_failures = selection_failures;
            results.resize_training_error_history(epoch+1);
            results.resize_selection_error_history(has_selection ? epoch + 1 : 0);
            results.elapsed_time = write_time(elapsed_time);

            break;
        }

        if(epoch != 0 && epoch%save_period == 0) neural_network->save(neural_network_file_name);
        
        update_parameters(training_batch,
                          training_forward_propagation,
                          training_back_propagation_lm,
                          optimization_data);
        
    }

    if(neural_network->has(Layer::Type::Scaling2D))
        data_set->unscale_variables(DataSet::VariableUse::Input, input_variables_descriptives);

    if(neural_network->has(Layer::Type::Unscaling))
        data_set->unscale_variables(DataSet::VariableUse::Target, target_variables_descriptives);

    if(display) results.print();

    return results;
}


void LevenbergMarquardtAlgorithm::update_parameters(const Batch& batch,
                                                    ForwardPropagation& forward_propagation,
                                                    BackPropagationLM& back_propagation_lm,
                                                    LevenbergMarquardtAlgorithmData& optimization_data)
{
    const type regularization_weight = loss_index->get_regularization_weight();
    
    NeuralNetwork* neural_network = loss_index->get_neural_network();
    
    Tensor<type, 1>& parameters = back_propagation_lm.parameters;
    
    type& error = back_propagation_lm.error();
    type& loss = back_propagation_lm.loss;

    const Tensor<type, 1>& gradient = back_propagation_lm.gradient;
    Tensor<type, 2>& hessian = back_propagation_lm.hessian;

    Tensor<type, 1>& potential_parameters = optimization_data.potential_parameters;
    Tensor<type, 1>& parameters_increment = optimization_data.parameters_increment;

    const Index parameters_number = parameters.size();

    bool success = false;
    
    do
    {
        sum_diagonal(hessian, damping_parameter);

        parameters_increment = perform_Householder_QR_decomposition(hessian, type(-1)*gradient);

        potential_parameters.device(*thread_pool_device) = parameters + parameters_increment;
        
        neural_network->forward_propagate(batch,
                                          potential_parameters,
                                          forward_propagation);

        loss_index->calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        loss_index->calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        loss_index->calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        type new_loss;

        try
        {
            new_loss = error + regularization_weight*loss_index->calculate_regularization(potential_parameters);

        }catch(exception)
        {
            new_loss = loss;
        }

        if(new_loss < loss) // succesfull step
        {
            set_damping_parameter(damping_parameter/damping_parameter_factor);

            parameters = potential_parameters;

            loss = new_loss;

            success = true;

            break;
        }
        else
        {
            sum_diagonal(hessian, -damping_parameter);

            set_damping_parameter(damping_parameter*damping_parameter_factor);
        }

    }while(damping_parameter < maximum_damping_parameter);

    if(!success)
    {
        const type epsilon = numeric_limits<type>::epsilon();

        #pragma omp parallel for

        for(Index i = 0; i < parameters_number; i++)
        {
            if (abs(gradient(i)) < type(NUMERIC_LIMITS_MIN))
            {
                parameters_increment(i) = type(0);
            }
            else
            {
                parameters_increment(i) = (gradient(i) > type(0)) ? -epsilon : epsilon;
                parameters(i) += parameters_increment(i);
            }
        }
    }

    // Set parameters

    neural_network->set_parameters(parameters);
}


string LevenbergMarquardtAlgorithm::write_optimization_algorithm_type() const
{
    return "LEVENBERG_MARQUARDT_ALGORITHM";
}


Tensor<string, 2> LevenbergMarquardtAlgorithm::to_string_matrix() const
{
    Tensor<string, 2> labels_values(7, 2);

    labels_values(0,0) = "Damping parameter factor";
    labels_values(0,1) = to_string(double(damping_parameter_factor));

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


void LevenbergMarquardtAlgorithm::to_XML(tinyxml2::XMLPrinter& printer) const
{
    printer.OpenElement("LevenbergMarquardt");

    add_xml_element(printer, "DampingParameterFactor", to_string(damping_parameter_factor));
    add_xml_element(printer, "MinimumLossDecrease", to_string(minimum_loss_decrease));
    add_xml_element(printer, "LossGoal", to_string(training_loss_goal));
    add_xml_element(printer, "MaximumSelectionFailures", to_string(maximum_selection_failures));
    add_xml_element(printer, "MaximumEpochsNumber", to_string(maximum_epochs_number));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));

    printer.CloseElement();
}


void LevenbergMarquardtAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("LevenbergMarquardt");

    if(!root_element)
        throw runtime_error("Levenberg-Marquardt algorithm element is nullptr.\n");

    set_damping_parameter_factor(read_xml_type(root_element, "DampingParameterFactor"));
    set_minimum_loss_decrease(read_xml_type(root_element, "MinimumLossDecrease"));
    set_loss_goal(read_xml_type(root_element, "LossGoal"));
    set_maximum_selection_failures(read_xml_index(root_element, "MaximumSelectionFailures"));
    set_maximum_epochs_number(read_xml_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
}


void LevenbergMarquardtAlgorithmData::set(LevenbergMarquardtAlgorithm* new_Levenberg_Marquardt_method)
{
    Levenberg_Marquardt_algorithm = new_Levenberg_Marquardt_method;

    const LossIndex* loss_index = Levenberg_Marquardt_algorithm->get_loss_index();

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_number();

    // Neural network data

    old_parameters.resize(parameters_number);

    parameters_difference.resize(parameters_number);

    potential_parameters.resize(parameters_number);
    parameters_increment.resize(parameters_number);
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
