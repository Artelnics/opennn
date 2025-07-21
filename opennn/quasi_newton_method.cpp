//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "dataset.h"
#include "loss_index.h"
#include "quasi_newton_method.h"

namespace opennn
{

QuasiNewtonMethod::QuasiNewtonMethod(const LossIndex* new_loss_index)
    : OptimizationAlgorithm(new_loss_index)
{
    set_default();
}


const type& QuasiNewtonMethod::get_minimum_loss_decrease() const
{
    return minimum_loss_decrease;
}


const type& QuasiNewtonMethod::get_loss_goal() const
{
    return training_loss_goal;
}


const Index& QuasiNewtonMethod::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


void QuasiNewtonMethod::set_loss_index(LossIndex* new_loss_index)
{
    loss_index = new_loss_index;
}


void QuasiNewtonMethod::set_display(const bool& new_display)
{
    display = new_display;
}


void QuasiNewtonMethod::set_default()
{
    learning_rate_tolerance = numeric_limits<type>::epsilon();
    loss_tolerance = numeric_limits<type>::epsilon();

    // Stopping criteria

    minimum_loss_decrease = type(0);
    training_loss_goal = type(0);
    maximum_selection_failures = 1000;

    maximum_epochs_number = 1000;
    maximum_time = type(3600);

    // UTILITIES

    display = true;
    display_period = 10;
}


void QuasiNewtonMethod::set_minimum_loss_decrease(const type& new_minimum_loss_decrease)
{
    minimum_loss_decrease = new_minimum_loss_decrease;
}


void QuasiNewtonMethod::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


void QuasiNewtonMethod::set_maximum_selection_failures(const Index& new_maximum_selection_failures)
{
    maximum_selection_failures = new_maximum_selection_failures;
}


void QuasiNewtonMethod::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


void QuasiNewtonMethod::set_maximum_time(const type& new_maximum_time)
{
    maximum_time = new_maximum_time;
}


void QuasiNewtonMethod::calculate_inverse_hessian(QuasiNewtonMethodData& optimization_data) const
{
    const Tensor<type, 1>& parameters_difference = optimization_data.parameters_difference;
    const Tensor<type, 1>& gradient_difference = optimization_data.gradient_difference;

    Tensor<type, 1>& old_inverse_hessian_dot_gradient_difference = optimization_data.old_inverse_hessian_dot_gradient_difference;

    const Tensor<type, 2>& old_inverse_hessian = optimization_data.old_inverse_hessian;
    Tensor<type, 2>& inverse_hessian = optimization_data.inverse_hessian;

    Tensor<type, 1>& BFGS = optimization_data.BFGS;

    Tensor<type, 0> parameters_difference_dot_gradient_difference;
    Tensor<type, 0> gradient_dot_hessian_dot_gradient;

    parameters_difference_dot_gradient_difference.device(*thread_pool_device)
            = parameters_difference.contract(gradient_difference, axes(0,0));

    old_inverse_hessian_dot_gradient_difference.device(*thread_pool_device)
            = old_inverse_hessian.contract(gradient_difference, axes(0,0));

    gradient_dot_hessian_dot_gradient.device(*thread_pool_device)
            = gradient_difference.contract(old_inverse_hessian_dot_gradient_difference, axes(0,0));

    BFGS.device(*thread_pool_device)
            = parameters_difference/parameters_difference_dot_gradient_difference(0)
            - old_inverse_hessian_dot_gradient_difference/gradient_dot_hessian_dot_gradient(0);

    // Calculate approximation

    inverse_hessian.device(*thread_pool_device) = old_inverse_hessian;

    inverse_hessian.device(*thread_pool_device)
        += self_kronecker_product(thread_pool_device.get(), parameters_difference)
        / parameters_difference_dot_gradient_difference(0);

    inverse_hessian.device(*thread_pool_device)
        -= self_kronecker_product(thread_pool_device.get(), old_inverse_hessian_dot_gradient_difference)
        / gradient_dot_hessian_dot_gradient(0);

    inverse_hessian.device(*thread_pool_device)
        += self_kronecker_product(thread_pool_device.get(), BFGS)*(gradient_dot_hessian_dot_gradient(0));
}


void QuasiNewtonMethod::update_parameters(const Batch& batch,
                                          ForwardPropagation& forward_propagation,
                                          BackPropagation& back_propagation,
                                          QuasiNewtonMethodData& optimization_data)
{
    NeuralNetwork* neural_network = forward_propagation.neural_network;

    Tensor<type, 1>& parameters = optimization_data.parameters;
    neural_network->get_parameters(parameters);

    const Tensor<type, 1>& gradient = optimization_data.gradient;

    Tensor<type, 1>& old_parameters = optimization_data.old_parameters;
    Tensor<type, 1>& parameters_difference = optimization_data.parameters_difference;
    Tensor<type, 1>& parameters_increment = optimization_data.parameters_increment;

    Tensor<type, 1>& old_gradient = optimization_data.old_gradient;
    Tensor<type, 1>& gradient_difference = optimization_data.gradient_difference;

    Tensor<type, 1>& training_direction = optimization_data.training_direction;

    Tensor<type, 2>& inverse_hessian = optimization_data.inverse_hessian;

    Tensor<type, 0>& training_slope = optimization_data.training_slope;

    parameters_difference.device(*thread_pool_device) = parameters - old_parameters;

    gradient_difference.device(*thread_pool_device) = gradient - old_gradient;

    old_parameters.device(*thread_pool_device) = parameters; // do not move above

    // Get training direction

    optimization_data.epoch == 0 || is_equal(parameters_difference, type(0)) || is_equal(gradient_difference, type(0))
        ? set_identity(inverse_hessian)
        : calculate_inverse_hessian(optimization_data);

    training_direction.device(*thread_pool_device) = -inverse_hessian.contract(gradient, axes(1,0));

    training_slope.device(*thread_pool_device) = gradient.contract(training_direction, axes(0,0));

    if(training_slope(0) >= type(0))
        training_direction.device(*thread_pool_device) = -gradient;

    // Get learning rate

    optimization_data.epoch == 0
            ? optimization_data.initial_learning_rate = first_learning_rate
            : optimization_data.initial_learning_rate = optimization_data.old_learning_rate;

    const type current_loss = back_propagation.loss;
    const pair<type, type> directional_point = calculate_directional_point(
             batch,
             forward_propagation,
             back_propagation,
             optimization_data,
             current_loss);

    optimization_data.learning_rate = directional_point.first;
    back_propagation.loss = directional_point.second;

    if(abs(optimization_data.learning_rate) > type(0))
    {
        optimization_data.parameters_increment.device(*thread_pool_device)
                = optimization_data.training_direction*optimization_data.learning_rate;

        parameters.device(*thread_pool_device) += optimization_data.parameters_increment;
    }
    else
    {
        const Index parameters_number = parameters.size();

        #pragma omp parallel for

        for(Index i = 0; i < parameters_number; i++)
        {
            if (abs(gradient(i)) < NUMERIC_LIMITS_MIN)
            {
                parameters_increment(i) = type(0);
            }
            else
            {
                parameters_increment(i) = (gradient(i) > type(0)) ? -epsilon : epsilon;
                parameters(i) += parameters_increment(i);
            }
        }

        optimization_data.learning_rate = optimization_data.initial_learning_rate;
    }

    // Update stuff
    old_gradient = gradient;

    optimization_data.old_inverse_hessian = inverse_hessian;

    optimization_data.old_learning_rate = optimization_data.learning_rate;

    // Set parameters

    neural_network->set_parameters(parameters);
}


TrainingResults QuasiNewtonMethod::train()
{
    if (!loss_index || !loss_index->has_neural_network() || !loss_index->has_dataset())
        return TrainingResults();

    TrainingResults results(maximum_epochs_number + 1);

    check();

    if(display) cout << "Training with quasi-Newton method..." << endl;;

    // Data set

    Dataset* dataset = loss_index->get_dataset();

    if (!dataset)
        throw runtime_error("Data set is null.");

    const bool has_selection = dataset->has_selection();

    const string error_type = loss_index->get_name();

    const Index training_samples_number = dataset->get_samples_number("Training");

    const Index selection_samples_number = dataset->get_samples_number("Selection");

    const vector<Index> training_samples_indices = dataset->get_sample_indices("Training");
    const vector<Index> selection_samples_indices = dataset->get_sample_indices("Selection");

    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    ForwardPropagation training_forward_propagation(training_samples_number, neural_network);

    ForwardPropagation selection_forward_propagation(selection_samples_number, neural_network);

    set_names();

    set_scaling();

    // Batch

    Batch training_batch(training_samples_number, dataset);
    // training_batch.fill(training_samples_indices, input_variable_indices, {}, target_variable_indices);
    training_batch.fill(training_samples_indices, input_variable_indices, target_variable_indices);

    Batch selection_batch(selection_samples_number, dataset);
    // selection_batch.fill(selection_samples_indices, input_variable_indices, {}, target_variable_indices);
    selection_batch.fill(selection_samples_indices, input_variable_indices, target_variable_indices);

    // Loss index

    loss_index->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_samples_number, loss_index);
    BackPropagation selection_back_propagation(selection_samples_number, loss_index);

    // Optimization algorithm

    bool stop_training = false;
    bool is_training = true;

    Index selection_failures = 0;

    type old_loss = type(0);
    type loss_decrease = numeric_limits<type>::max();

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time;

    QuasiNewtonMethodData optimization_data(this);

    const Index parameters_number = neural_network->get_parameters_number();
    optimization_data.gradient.resize(parameters_number);

    // Main loop

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

        loss_index->assemble_layers_error_gradient(training_back_propagation,
                                                   optimization_data.gradient);

        results.training_error_history(epoch) = training_back_propagation.error();

        // Update parameters

        update_parameters(training_batch, 
                          training_forward_propagation, 
                          training_back_propagation, 
                          optimization_data);

        // Selection error

        if(has_selection)
        {            
            neural_network->forward_propagate(selection_batch.get_input_pairs(),
                selection_forward_propagation,
                is_training);

            // Loss Index

            loss_index->calculate_error(selection_batch, 
                                        selection_forward_propagation, 
                                        selection_back_propagation);

            results.selection_error_history(epoch) = selection_back_propagation.error();

            if(epoch != 0
            && results.selection_error_history(epoch) > results.selection_error_history(epoch-1))
                selection_failures++;
        }

        elapsed_time = get_elapsed_time(beginning_time);

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_back_propagation.error << endl;
            if(has_selection) cout << "Selection error: " << selection_back_propagation.error << endl;
            cout << "Learning rate: " << optimization_data.learning_rate << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        if(epoch != 0) loss_decrease = old_loss - training_back_propagation.loss;

        old_loss = training_back_propagation.loss;

        stop_training = true;

        if(loss_decrease < minimum_loss_decrease)
        {
            if(display) cout << "Epoch " << epoch << "\nMinimum loss decrease reached (" << minimum_loss_decrease << "): " << loss_decrease << endl;
            results.stopping_condition = OptimizationAlgorithm::StoppingCondition::MinimumLossDecrease;
        }
        else if(results.training_error_history(epoch) < training_loss_goal)
        {
            if(display) cout << "Epoch " << epoch << "\nLoss goal reached: " << results.training_error_history(epoch) << endl;
            results.stopping_condition = OptimizationAlgorithm::StoppingCondition::LossGoal;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << selection_failures << endl;
            results.stopping_condition = OptimizationAlgorithm::StoppingCondition::MaximumSelectionErrorIncreases;
        }
        else if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch << endl;
            results.stopping_condition = OptimizationAlgorithm::StoppingCondition::MaximumEpochsNumber;
        }
        else if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum training time reached: " << write_time(elapsed_time) << endl;
            results.stopping_condition = OptimizationAlgorithm::StoppingCondition::MaximumTime;
        }
        else
        {
            stop_training = false;
        }

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

        if(epoch != 0 && epoch % save_period == 0) neural_network->save(neural_network_file_name);
    }

    set_unscaling();

    if(display) results.print();

    return results;
}


string QuasiNewtonMethod::get_name() const
{
    return "QuasiNewtonMethod";
}


void QuasiNewtonMethod::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("QuasiNewtonMethod");

    add_xml_element(printer, "MinimumLossDecrease", to_string(minimum_loss_decrease));
    add_xml_element(printer, "LossGoal", to_string(training_loss_goal));
    add_xml_element(printer, "MaximumSelectionFailures", to_string(maximum_selection_failures));
    add_xml_element(printer, "MaximumEpochsNumber", to_string(maximum_epochs_number));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));

    printer.CloseElement();
}


Tensor<string, 2> QuasiNewtonMethod::to_string_matrix() const
{
    Tensor<string, 2> string_matrix(6, 2);

    string_matrix.setValues({
    {"Learning rate tolerance", to_string(double(learning_rate_tolerance))},
    {"Minimum loss decrease", to_string(double(minimum_loss_decrease))},
    {"Loss goal", to_string(double(training_loss_goal))},
    {"Maximum selection error increases", to_string(maximum_selection_failures)},
    {"Maximum epochs number", to_string(maximum_epochs_number)},
    {"Maximum time", write_time(maximum_time)}});

    return string_matrix;
}


void QuasiNewtonMethod::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("QuasiNewtonMethod");

    if (!root_element)
        throw runtime_error("Quasi-Newton method element is nullptr.\n");
    
    /*const XMLElement* learning_rate_algorithm_element = root_element->FirstChildElement("LearningRateAlgorithm");
    
    if (!learning_rate_algorithm_element)
        throw runtime_error("Learning rate algorithm element is nullptr.\n");*/
    
    set_minimum_loss_decrease(read_xml_type(root_element, "MinimumLossDecrease"));
    set_loss_goal(read_xml_type(root_element, "LossGoal"));
    set_maximum_selection_failures(read_xml_index(root_element, "MaximumSelectionFailures"));
    set_maximum_epochs_number(read_xml_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
}


QuasiNewtonMethodData::QuasiNewtonMethodData(QuasiNewtonMethod* new_quasi_newton_method)
{
    set(new_quasi_newton_method);
}


void QuasiNewtonMethodData::set(QuasiNewtonMethod* new_quasi_newton_method)
{
    quasi_newton_method = new_quasi_newton_method;

    const LossIndex* loss_index = quasi_newton_method->get_loss_index();

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_number();

    // Neural network data

    parameters.resize(parameters_number);
    old_parameters.resize(parameters_number);

    parameters_difference.resize(parameters_number);

    potential_parameters.resize(parameters_number);
    parameters_increment.resize(parameters_number);

    // Loss index data

    gradient.resize(parameters_number);
    old_gradient.resize(parameters_number);
    old_gradient.setZero();

    gradient_difference.resize(parameters_number);

    inverse_hessian.resize(parameters_number, parameters_number);
    inverse_hessian.setZero();

    old_inverse_hessian.resize(parameters_number, parameters_number);
    old_inverse_hessian.setZero();

    // Optimization algorithm data

    BFGS.resize(parameters_number);

    training_direction.resize(parameters_number);

    old_inverse_hessian_dot_gradient_difference.resize(parameters_number);
}


void QuasiNewtonMethodData::print() const
{
    cout << "Training Direction:" << endl
         << training_direction << endl
         << "Learning rate:" << endl
         << learning_rate << endl;
}


Triplet QuasiNewtonMethod::calculate_bracketing_triplet(
    const Batch& batch,
    ForwardPropagation& forward_propagation,
    BackPropagation& back_propagation,
    QuasiNewtonMethodData& optimization_data)
{
    Triplet triplet;

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    const type regularization_weight = loss_index->get_regularization_weight();

    Tensor<type, 1>& potential_parameters = optimization_data.potential_parameters;

    const Tensor<type, 1>& parameters = optimization_data.parameters;

    const Tensor<type, 1>& training_direction = optimization_data.training_direction;

    // Left point

    triplet.A = { type(0), back_propagation.loss };

    // Right point

    Index count = 0;

    do
    {
        count++;

        triplet.B.first = optimization_data.initial_learning_rate*type(count);

        potential_parameters.device(*thread_pool_device)
                = parameters + training_direction * triplet.B.first;


        neural_network->forward_propagate(batch.get_input_pairs(),
            potential_parameters, forward_propagation);

        loss_index->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index->calculate_regularization(potential_parameters);

        triplet.B.second = back_propagation.error() + regularization_weight * regularization;

    } while(abs(triplet.A.second - triplet.B.second) < loss_tolerance && triplet.A.second != triplet.B.second);

    if(triplet.A.second > triplet.B.second)
    {
        triplet.U = triplet.B;

        triplet.B.first *= golden_ratio;

        potential_parameters.device(*thread_pool_device)
                = parameters + training_direction*triplet.B.first;

        neural_network->forward_propagate(batch.get_input_pairs(),
                                          potential_parameters,
                                          forward_propagation);

        loss_index->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index->calculate_regularization(potential_parameters);

        triplet.B.second = back_propagation.error() + regularization_weight * regularization;

        while(triplet.U.second > triplet.B.second)
        {
            triplet.A = triplet.U;
            triplet.U = triplet.B;

            triplet.B.first *= golden_ratio;

            potential_parameters.device(*thread_pool_device)
                    = parameters + training_direction*triplet.B.first;

            neural_network->forward_propagate(batch.get_input_pairs(),
                                              potential_parameters,
                                              forward_propagation);

            loss_index->calculate_error(batch, forward_propagation, back_propagation);

            const type regularization = loss_index->calculate_regularization(potential_parameters);

            triplet.B.second = back_propagation.error() + regularization_weight * regularization;
        }
    }
    else if(triplet.A.second < triplet.B.second)
    {
        triplet.U.first = triplet.A.first + (triplet.B.first - triplet.A.first)*type(0.382);

        potential_parameters.device(*thread_pool_device)
                = parameters + training_direction*triplet.U.first;

        neural_network->forward_propagate(batch.get_input_pairs(),
                                          potential_parameters,
                                          forward_propagation);

        loss_index->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index->calculate_regularization(potential_parameters);

        triplet.U.second = back_propagation.error() + regularization_weight * regularization;

        while(triplet.A.second < triplet.U.second)
        {
            triplet.B = triplet.U;

            triplet.U.first = triplet.A.first + (triplet.B.first-triplet.A.first)*type(0.382);

            potential_parameters.device(*thread_pool_device)
                    = parameters + training_direction*triplet.U.first;

            neural_network->forward_propagate(batch.get_input_pairs(), potential_parameters, forward_propagation);

            loss_index->calculate_error(batch, forward_propagation, back_propagation);

            const type regularization = loss_index->calculate_regularization(potential_parameters);

            triplet.U.second = back_propagation.error() + regularization_weight * regularization;

            if(triplet.U.first - triplet.A.first <= learning_rate_tolerance)
            {
                triplet.U = triplet.A;
                triplet.B = triplet.A;

                return triplet;
            }
        }
    }

    return triplet;
}


type QuasiNewtonMethod::calculate_learning_rate(const Triplet& triplet) const
{
    const type a = triplet.A.first;
    const type u = triplet.U.first;
    const type b = triplet.B.first;

    const type fa = triplet.A.second;
    const type fu = triplet.U.second;
    const type fb = triplet.B.second;

    const type numerator = (u-a)*(u-a)*(fu-fb) - (u-b)*(u-b)*(fu-fa);

    const type denominator = (u-a)*(fu-fb) - (u-b)*(fu-fa);

    return denominator != type(0)
       ? u - type(0.5) * (numerator / denominator)
       : type(0);
}


pair<type, type> QuasiNewtonMethod::calculate_directional_point(
    const Batch& batch,
    ForwardPropagation& forward_propagation,
    BackPropagation& back_propagation,
    QuasiNewtonMethodData& optimization_data,
    const type& current_loss)
{
    NeuralNetwork* neural_network = loss_index->get_neural_network();
    const type regularization_weight = loss_index->get_regularization_weight();

    type alpha = 1.0;
    const type rho = 0.5;
    const type c = 1e-4F;

    const Tensor<type, 1>& parameters = optimization_data.parameters;
    const Tensor<type, 1>& training_direction = optimization_data.training_direction;
    Tensor<type, 1>& potential_parameters = optimization_data.potential_parameters;

    const type slope = optimization_data.training_slope(0);

    for (int i = 0; i < 20; ++i)
    {
        potential_parameters.device(*thread_pool_device) = parameters + training_direction * alpha;

        neural_network->forward_propagate(batch.get_input_pairs(), potential_parameters, forward_propagation);
        loss_index->calculate_error(batch, forward_propagation, back_propagation);
        const type regularization = loss_index->calculate_regularization(potential_parameters);
        const type new_loss = back_propagation.error() + regularization_weight * regularization;

        if (new_loss <= current_loss + c * alpha * slope)
            return {alpha, new_loss};

        alpha *= rho;
    }

    return {0.0, current_loss};
}


Triplet::Triplet()
{
    A = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
    U = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
    B = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
}


bool Triplet::operator ==(const Triplet &other_triplet) const
{
    return (A == other_triplet.A && U == other_triplet.U && B == other_triplet.B);
}


type Triplet::get_length() const
{
    return abs(B.first - A.first);
}


pair<type, type> Triplet::minimum() const
{
    Tensor<type, 1> losses(3);

    losses.setValues({ A.second, U.second, B.second });

    const Index minimal_index = opennn::minimal_index(losses);

    if (minimal_index == 0) return A;
    else if (minimal_index == 1) return U;
    else return B;
}


string Triplet::struct_to_string() const
{
    ostringstream buffer;

    buffer << "A = (" << A.first << "," << A.second << ")\n"
           << "U = (" << U.first << "," << U.second << ")\n"
           << "B = (" << B.first << "," << B.second << ")" << endl;

    return buffer.str();
}


void Triplet::print() const
{
    cout << struct_to_string()
    << "Length: " << get_length() << endl;
}


void Triplet::check() const
{
    if (U.first < A.first)
        throw runtime_error("U is less than A:\n" + struct_to_string());

    if (U.first > B.first)
        throw runtime_error("U is greater than B:\n" + struct_to_string());

    if (U.second >= A.second)
        throw runtime_error("fU is equal or greater than fA:\n" + struct_to_string());

    if (U.second >= B.second)
        throw runtime_error("fU is equal or greater than fB:\n" + struct_to_string());
}


REGISTER(OptimizationAlgorithm, QuasiNewtonMethod, "QuasiNewtonMethod");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
