//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N J U G A T E   G R A D I E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "conjugate_gradient.h"
#include "scaling_layer_2d.h"
#include "unscaling_layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

ConjugateGradient::ConjugateGradient(LossIndex* new_loss_index)
    : OptimizationAlgorithm(new_loss_index)
{
    learning_rate_algorithm.set_loss_index(new_loss_index);

    set_default();
}


void ConjugateGradient::calculate_conjugate_gradient_training_direction(const Tensor<type, 1>& old_gradient,
                                                                        const Tensor<type, 1>& gradient,
                                                                        const Tensor<type, 1>& old_training_direction,
                                                                        Tensor<type, 1>& training_direction) const
{
    switch(training_direction_method)
    {
    case TrainingDirectionMethod::FR:
        calculate_FR_training_direction(old_gradient, gradient, old_training_direction, training_direction);
        return;

    case TrainingDirectionMethod::PR:
        calculate_PR_training_direction(old_gradient, gradient, old_training_direction, training_direction);
        return;

    default:
        return;
    }
}


type ConjugateGradient::calculate_FR_parameter(const Tensor<type, 1>& old_gradient, const Tensor<type, 1>& gradient) const
{
    type FR_parameter = type(0);

    Tensor<type, 0> numerator;
    Tensor<type, 0> denominator;

    numerator.device(*thread_pool_device) = gradient.contract(gradient, AT_B);
    denominator.device(*thread_pool_device) = old_gradient.contract(old_gradient, AT_B);

    FR_parameter = (abs(denominator(0)) < NUMERIC_LIMITS_MIN)
        ? type(0)
        : numerator(0) / denominator(0);

    return clamp(FR_parameter, type(0), type(1));
}


void ConjugateGradient::calculate_FR_training_direction(const Tensor<type, 1>& old_gradient,
                                                        const Tensor<type, 1>& gradient,
                                                        const Tensor<type, 1>& old_training_direction,
                                                        Tensor<type, 1>& training_direction) const
{
    const type FR_parameter = calculate_FR_parameter(old_gradient, gradient);

    training_direction.device(*thread_pool_device) = -gradient + old_training_direction*FR_parameter;
}


void ConjugateGradient::calculate_gradient_descent_training_direction(const Tensor<type, 1>& gradient,
                                                                      Tensor<type, 1>& training_direction) const
{
    training_direction.device(*thread_pool_device) = -gradient;
}


type ConjugateGradient::calculate_PR_parameter(const Tensor<type, 1>& old_gradient, const Tensor<type, 1>& gradient) const
{
    type PR_parameter = type(0);

    Tensor<type, 0> numerator;
    Tensor<type, 0> denominator;

    numerator.device(*thread_pool_device) = (gradient-old_gradient).contract(gradient, AT_B);

    denominator.device(*thread_pool_device) = old_gradient.contract(old_gradient, AT_B);

    PR_parameter = (abs(denominator(0)) < NUMERIC_LIMITS_MIN)
        ? type(0)
        : numerator(0) / denominator(0);

    return clamp(PR_parameter, type(0), type(1));
}


void ConjugateGradient::calculate_PR_training_direction(const Tensor<type, 1>& old_gradient,
                                                        const Tensor<type, 1>& gradient,
                                                        const Tensor<type, 1>& old_training_direction,
                                                        Tensor<type, 1>& training_direction) const
{
    const type PR_parameter = calculate_PR_parameter(old_gradient, gradient);

    training_direction.device(*thread_pool_device) = -gradient + old_training_direction*PR_parameter;
}


const LearningRateAlgorithm& ConjugateGradient::get_learning_rate_algorithm() const
{
    return learning_rate_algorithm;
}


LearningRateAlgorithm* ConjugateGradient::get_learning_rate_algorithm()
{
    return &learning_rate_algorithm;
}


const type& ConjugateGradient::get_loss_goal() const
{
    return training_loss_goal;
}


const Index& ConjugateGradient::get_maximum_epochs_number() const
{
    return maximum_epochs_number;
}


const Index& ConjugateGradient::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


const type& ConjugateGradient::get_maximum_time() const
{
    return maximum_time;
}


const type& ConjugateGradient::get_minimum_loss_decrease() const
{
    return minimum_loss_decrease;
}


const ConjugateGradient::TrainingDirectionMethod& ConjugateGradient::get_training_direction_method() const
{
    return training_direction_method;
}


void ConjugateGradient::set_default()
{
    // Stopping criteria

    minimum_loss_decrease = type(0);
    training_loss_goal = type(0);
    maximum_selection_failures = 1000000;

    maximum_epochs_number = 1000;
    maximum_time = type(3600.0);

    // UTILITIES

    display_period = 10;

    training_direction_method = TrainingDirectionMethod::FR;
}


void ConjugateGradient::set_loss_index(LossIndex* new_loss_index)
{
    loss_index = new_loss_index;

    learning_rate_algorithm.set_loss_index(new_loss_index);
}


void ConjugateGradient::set_training_direction_method
                        (const ConjugateGradient::TrainingDirectionMethod& new_training_direction_method)
{
    training_direction_method = new_training_direction_method;
}


void ConjugateGradient::set_training_direction_method(const string& new_training_direction_method_name)
{
    if(new_training_direction_method_name == "PR")
        training_direction_method = TrainingDirectionMethod::PR;
    else if(new_training_direction_method_name == "FR")
        training_direction_method = TrainingDirectionMethod::FR;
    else
        throw runtime_error("Unknown training direction method: " + new_training_direction_method_name + ".\n");
}


void ConjugateGradient::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


void ConjugateGradient::set_maximum_selection_failures(const Index& new_maximum_selection_failures)
{
    maximum_selection_failures = new_maximum_selection_failures;
}


void ConjugateGradient::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


void ConjugateGradient::set_maximum_time(const type& new_maximum_time)
{
    maximum_time = new_maximum_time;
}


void ConjugateGradient::set_minimum_loss_decrease(const type& new_minimum_loss_decrease)
{
    minimum_loss_decrease = new_minimum_loss_decrease;
}


TrainingResults ConjugateGradient::perform_training()
{
    if (!loss_index || !loss_index->has_neural_network() || !loss_index->has_data_set())
        return TrainingResults();

    // Start training

    if(display) cout << "Training with conjugate gradient...\n";

    TrainingResults results(maximum_epochs_number+1);

    // Elapsed time

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    // Data set

    DataSet* data_set = loss_index->get_data_set();

    const Index training_samples_number = data_set->get_samples_number(DataSet::SampleUse::Training);
    const Index selection_samples_number = data_set->get_samples_number(DataSet::SampleUse::Selection);
    const bool has_selection = data_set->has_selection();

    const vector<Index> training_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Training);
    const vector<Index> selection_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Selection);

    const vector<Index> input_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Target);

    const vector<string> input_names = data_set->get_variable_names(DataSet::VariableUse::Input);
    const vector<string> target_names = data_set->get_variable_names(DataSet::VariableUse::Target);

    const vector<Scaler> input_variable_scalers = data_set->get_variable_scalers(DataSet::VariableUse::Input);
    const vector<Scaler> target_variable_scalers = data_set->get_variable_scalers(DataSet::VariableUse::Target);

    print_vector(input_variable_indices);

    // throw runtime_error("Dimensions check");

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    set_scaling();


    Batch training_batch(training_samples_number, data_set);
    training_batch.fill(training_samples_indices, input_variable_indices, {}, target_variable_indices);

    Batch selection_batch(selection_samples_number, data_set);
    selection_batch.fill(selection_samples_indices, input_variable_indices, {}, target_variable_indices);

    ForwardPropagation training_forward_propagation(training_samples_number, neural_network);
    ForwardPropagation selection_forward_propagation(selection_samples_number, neural_network);

    // Loss index

    string information;

    loss_index->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_samples_number, loss_index);
    BackPropagation selection_back_propagation(selection_samples_number, loss_index);

    // Optimization algorithm

    type old_loss = type(0);
    type loss_decrease = numeric_limits<type>::max();

    bool stop_training = false;
    bool is_training = true;
    Index selection_failures = 0;

    ConjugateGradientData optimization_data(this);

    // Main loop

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        optimization_data.epoch = epoch;

        // Neural network

        neural_network->forward_propagate(training_batch.get_input_pairs(), training_forward_propagation, is_training);

        // Loss index

        loss_index->back_propagate(training_batch, training_forward_propagation, training_back_propagation);
        results.training_error_history(epoch) = training_back_propagation.error();

        // Update parameters

        update_parameters(training_batch, training_forward_propagation, training_back_propagation, optimization_data);

        if(has_selection)
        {
            neural_network->forward_propagate(selection_batch.get_input_pairs(), selection_forward_propagation, is_training);

            loss_index->calculate_error(selection_batch, selection_forward_propagation, selection_back_propagation);

            results.selection_error_history(epoch) = selection_back_propagation.error();

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) selection_failures++;
        }

        // Optimization algorithm

        elapsed_time = get_elapsed_time(beginning_time);

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

            if(display) cout << "Epoch " << epoch << "\nLoss goal reached: " << results.training_error_history(epoch) << endl;
        }

        if(has_selection && selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << selection_failures << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }

        if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
        }

        if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum training time reached: " << write_time(elapsed_time) << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumTime;
        }

        if(epoch != 0) loss_decrease = old_loss - training_back_propagation.loss;

        if(loss_decrease <= minimum_loss_decrease)
        {
            if(display) cout << "Epoch " << epoch << "\nMinimum loss decrease reached: " << minimum_loss_decrease << endl;

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

        // Update stuff

        if(epoch != 0 && epoch%save_period == 0) 
            neural_network->save(neural_network_file_name);
    }

    set_unscaling();

    // cout << "Unscaled data:\n" << not_scaled_data - data_set->get_data() << endl;

    // throw runtime_error("Checking scaled data");
    // data_set->unscale_variables(DataSet::VariableUse::Input, input_variable_descriptives);

    // if(neural_network->has(Layer::Type::Unscaling))
    //     data_set->unscale_variables(DataSet::VariableUse::Target, target_variable_descriptives);

    if(display) results.print();

    return results;
}


Tensor<string, 2> ConjugateGradient::to_string_matrix() const
{
    Tensor<string, 2> string_matrix(8, 2);

    string_matrix.setValues({
    {"Training direction method", write_training_direction_method()},
    {"Learning rate method", learning_rate_algorithm.write_learning_rate_method()},
    {"Learning rate tolerance", to_string(double(learning_rate_algorithm.get_learning_rate_tolerance()))},
    {"Minimum loss decrease", to_string(double(minimum_loss_decrease))},
    {"Loss goal", to_string(double(training_loss_goal))},
    {"Maximum selection error increases", to_string(maximum_selection_failures)},
    {"Maximum epochs number", to_string(maximum_epochs_number)},
    {"Maximum time", write_time(maximum_time)}});

    return string_matrix;
}


void ConjugateGradient::update_parameters(
        const Batch& batch,
        ForwardPropagation& forward_propagation,
        BackPropagation& back_propagation,
        ConjugateGradientData& optimization_data) const
{
    const Index parameters_number = back_propagation.parameters.dimension(0);

    optimization_data.epoch == 0 || optimization_data.epoch % parameters_number == 0
        ? calculate_gradient_descent_training_direction(
                    back_propagation.gradient,
                    optimization_data.training_direction)
        : calculate_conjugate_gradient_training_direction(
                    optimization_data.old_gradient,
                    back_propagation.gradient,
                    optimization_data.old_training_direction,
                    optimization_data.training_direction);

    optimization_data.training_slope.device(*thread_pool_device)
            = back_propagation.gradient.contract(optimization_data.training_direction, AT_B);

    if(optimization_data.training_slope(0) >= type(0))
    {
        calculate_gradient_descent_training_direction(
                    back_propagation.gradient,
                    optimization_data.training_direction);
    }

    // Get initial learning rate

    optimization_data.epoch == 0
            ? optimization_data.initial_learning_rate = first_learning_rate
            : optimization_data.initial_learning_rate = optimization_data.old_learning_rate;

    pair<type,type> directional_point = learning_rate_algorithm.calculate_directional_point(
         batch,
         forward_propagation,
         back_propagation,
         optimization_data);

    optimization_data.learning_rate = directional_point.first;
    back_propagation.loss = directional_point.second;

    if(abs(optimization_data.learning_rate) > type(0))
    {
        optimization_data.parameters_increment.device(*thread_pool_device)
                = optimization_data.training_direction*optimization_data.learning_rate;

        back_propagation.parameters.device(*thread_pool_device) += optimization_data.parameters_increment;
    }
    else
    {
        constexpr type epsilon = numeric_limits<type>::epsilon();

        //const Index parameters_number = back_propagation.parameters.size();

        #pragma omp parallel for

        for(Index i = 0; i < parameters_number; i++)
        {
            if (abs(back_propagation.gradient(i)) < NUMERIC_LIMITS_MIN)
            {
                optimization_data.parameters_increment(i) = type(0);
            }
            else
            {
                optimization_data.parameters_increment(i) = (back_propagation.gradient(i) > type(0)) 
                    ? -epsilon 
                    : epsilon;

                back_propagation.parameters(i) += optimization_data.parameters_increment(i);
            }
        }

        optimization_data.learning_rate = optimization_data.initial_learning_rate;
    }

    // Update stuff

    optimization_data.old_gradient = back_propagation.gradient;

    optimization_data.old_training_direction = optimization_data.training_direction;

    optimization_data.old_learning_rate = optimization_data.learning_rate;

    // Update parameters

    forward_propagation.neural_network->set_parameters(back_propagation.parameters);
}


string ConjugateGradient::write_optimization_algorithm_type() const
{
    return "CONJUGATE_GRADIENT";
}


string ConjugateGradient::write_training_direction_method() const
{
    switch(training_direction_method)
    {
    case TrainingDirectionMethod::PR:
        return "PR";

    case TrainingDirectionMethod::FR:
        return "FR";
    default:
        return string();
    }
}


void ConjugateGradient::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("ConjugateGradient");

    add_xml_element(printer, "TrainingDirectionMethod", write_training_direction_method());

    learning_rate_algorithm.to_XML(printer);

    add_xml_element(printer, "MinimumLossDecrease", to_string(minimum_loss_decrease));
    add_xml_element(printer, "LossGoal", to_string(training_loss_goal));
    add_xml_element(printer, "MaximumSelectionFailures", to_string(maximum_selection_failures));
    add_xml_element(printer, "MaximumEpochsNumber", to_string(maximum_epochs_number));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));

    printer.CloseElement();
}


void ConjugateGradient::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("ConjugateGradient");

    if (!root_element) 
        throw runtime_error("Conjugate gradient element is nullptr.\n");

    set_training_direction_method(read_xml_string(root_element, "TrainingDirectionMethod"));

    const XMLElement* learning_rate_algorithm_element = root_element->FirstChildElement("LearningRateAlgorithm");

    if (learning_rate_algorithm_element) 
    {
        XMLDocument learning_rate_algorithm_document;
        learning_rate_algorithm_document.InsertFirstChild(learning_rate_algorithm_element->DeepClone(&learning_rate_algorithm_document));
        learning_rate_algorithm.from_XML(learning_rate_algorithm_document);
    }

    set_minimum_loss_decrease(read_xml_type(root_element, "MinimumLossDecrease"));
    set_loss_goal(read_xml_type(root_element, "LossGoal"));
    set_maximum_selection_failures(read_xml_index(root_element, "MaximumSelectionFailures"));
    set_maximum_epochs_number(read_xml_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
}


ConjugateGradientData::ConjugateGradientData(ConjugateGradient* new_conjugate_gradient) 
    : OptimizationAlgorithmData()
{
    set(new_conjugate_gradient);
}


void ConjugateGradientData::set(ConjugateGradient* new_conjugate_gradient)
{
    conjugate_gradient = new_conjugate_gradient;

    const LossIndex* loss_index = conjugate_gradient->get_loss_index();

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_number();

    potential_parameters.resize(parameters_number);

    parameters_increment.resize(parameters_number);

    old_gradient.resize(parameters_number);

    training_direction.resize(parameters_number);
    old_training_direction.resize(parameters_number);
}


void ConjugateGradientData::print() const
{
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
