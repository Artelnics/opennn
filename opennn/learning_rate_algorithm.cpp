//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E A R N I N G   R A T E   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "dataset.h"
#include "neural_network.h"
#include "loss_index.h"
#include "optimization_algorithm.h"
#include "learning_rate_algorithm.h"

namespace opennn
{

LearningRateAlgorithm::LearningRateAlgorithm(LossIndex* new_loss_index)
    : loss_index(new_loss_index)
{
    set_default();
}


LossIndex* LearningRateAlgorithm::get_loss_index() const
{
    return loss_index;
}


bool LearningRateAlgorithm::has_loss_index() const
{
    return loss_index;
}


const LearningRateAlgorithm::LearningRateMethod& LearningRateAlgorithm::get_learning_rate_method() const
{
    return learning_rate_method;
}


string LearningRateAlgorithm::write_learning_rate_method() const
{
    switch(learning_rate_method)
    {
    case LearningRateMethod::GoldenSection:
        return "GoldenSection";

    case LearningRateMethod::BrentMethod:
        return "BrentMethod";
    default:
        return string();
    }
}


const type& LearningRateAlgorithm::get_learning_rate_tolerance() const
{
    return learning_rate_tolerance;
}


const bool& LearningRateAlgorithm::get_display() const
{
    return display;
}


void LearningRateAlgorithm::set(LossIndex* new_loss_index)
{
    loss_index = new_loss_index;

    set_default();
}


void LearningRateAlgorithm::set_default()
{
    const unsigned int threads_number = thread::hardware_concurrency();

    if(thread_pool != nullptr)
        thread_pool.reset();
    if(thread_pool_device != nullptr)
        thread_pool_device.reset();

    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

    learning_rate_method = LearningRateMethod::BrentMethod;

    learning_rate_tolerance = numeric_limits<type>::epsilon();
    loss_tolerance = numeric_limits<type>::epsilon();
}


void LearningRateAlgorithm::set_loss_index(LossIndex* new_loss_index)
{
    loss_index = new_loss_index;
}


void LearningRateAlgorithm::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr)
        thread_pool.reset();
    if(thread_pool_device != nullptr)
        thread_pool_device.reset();

    thread_pool = make_unique<ThreadPool>(new_threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), new_threads_number);
}


void LearningRateAlgorithm::set_learning_rate_method(
        const LearningRateAlgorithm::LearningRateMethod& new_learning_rate_method)
{
    learning_rate_method = new_learning_rate_method;
}


void LearningRateAlgorithm::set_learning_rate_method(const string& new_learning_rate_method)
{
    if(new_learning_rate_method == "GoldenSection")
        learning_rate_method = LearningRateMethod::GoldenSection;
    else if(new_learning_rate_method == "BrentMethod")
        learning_rate_method = LearningRateMethod::BrentMethod;
    else
        throw runtime_error("Unknown learning rate method: " + new_learning_rate_method + ".\n");
}


void LearningRateAlgorithm::set_learning_rate_tolerance(const type& new_learning_rate_tolerance)
{
    learning_rate_tolerance = new_learning_rate_tolerance;
}


void LearningRateAlgorithm::set_display(const bool& new_display)
{
    display = new_display;
}


pair<type, type> LearningRateAlgorithm::calculate_directional_point(
    const Batch& batch,
    ForwardPropagation& forward_propagation,
    BackPropagation& back_propagation,
    OptimizationAlgorithmData& optimization_data) const
{
    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    const Tensor<type, 1>& parameters = back_propagation.parameters;

    Tensor<type, 1>& potential_parameters = optimization_data.potential_parameters;

    const Tensor<type, 1>& training_direction = optimization_data.training_direction;

    Triplet triplet = calculate_bracketing_triplet(batch,
                                                   forward_propagation,
                                                   back_propagation,
                                                   optimization_data);

    try
    {
        triplet.check();
    }
    catch(const exception& error)
    {
        return triplet.minimum();
    }

    const type regularization_weight = loss_index->get_regularization_weight();

    pair<type, type> V;

    // Reduce the interval

    while(abs(triplet.A.first - triplet.B.first) > learning_rate_tolerance
       || abs(triplet.A.second - triplet.B.second) > loss_tolerance)
    {
        try
        {
            switch(learning_rate_method)
            {
                case LearningRateMethod::GoldenSection: V.first = calculate_golden_section_learning_rate(triplet); break;

                case LearningRateMethod::BrentMethod: V.first = calculate_Brent_method_learning_rate(triplet); break;

                default: break;
            }
        }
        catch(const exception& error)
        {
            cout << error.what() << endl;

            return triplet.minimum();
        }

        // Calculate loss for V

        potential_parameters.device(*thread_pool_device)
                = parameters + training_direction*V.first;
        
        neural_network->forward_propagate(batch.get_input_pairs(), potential_parameters, forward_propagation);

        loss_index->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index->calculate_regularization(potential_parameters);

        V.second = back_propagation.error() + regularization_weight * regularization;

        // Update points

        if(V.first <= triplet.U.first)
        {
            if(V.second >= triplet.U.second)
            {
                triplet.A = V;
            }
            else
            {
                triplet.B = triplet.U;
                triplet.U = V;
            }
        }
        else
        {
            if(V.second >= triplet.U.second)
            {
                triplet.B = V;
            }
            else
            {
                triplet.A = triplet.U;
                triplet.U = V;
            }
        }

        // Check triplet

        try
        {
            triplet.check();
        }
        catch(const exception& error)
        {
            return triplet.minimum();
        }
    }

    return triplet.U;
}


LearningRateAlgorithm::Triplet LearningRateAlgorithm::calculate_bracketing_triplet(
    const Batch& batch,
    ForwardPropagation& forward_propagation,
    BackPropagation& back_propagation,
    OptimizationAlgorithmData& optimization_data) const
{
    Triplet triplet;

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    const type regularization_weight = loss_index->get_regularization_weight();

    Tensor<type, 1>& potential_parameters = optimization_data.potential_parameters;

    const Tensor<type, 1>& parameters = back_propagation.parameters;

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


type LearningRateAlgorithm::calculate_golden_section_learning_rate(const Triplet& triplet) const
{
    const type middle = triplet.A.first + type(0.5)*(triplet.B.first - triplet.A.first);

    const type learning_rate = triplet.U.first < middle
        ? triplet.A.first + type(0.618) * (triplet.B.first - triplet.A.first)
        : triplet.A.first + type(0.382) * (triplet.B.first - triplet.A.first);

    return learning_rate;
}


type LearningRateAlgorithm::calculate_Brent_method_learning_rate(const Triplet& triplet) const
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


void LearningRateAlgorithm::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("LearningRateAlgorithm");

    add_xml_element(printer, "LearningRateMethod", write_learning_rate_method());
    add_xml_element(printer, "LearningRateTolerance", to_string(learning_rate_tolerance));

    printer.CloseElement();
}


void LearningRateAlgorithm::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("LearningRateAlgorithm");

    if(!root_element)
        throw runtime_error("Learning rate algorithm element is nullptr.\n");

    set_learning_rate_method(read_xml_string(root_element, "LearningRateMethod"));
    set_learning_rate_tolerance(read_xml_type(root_element, "LearningRateTolerance"));
}


LearningRateAlgorithm::Triplet::Triplet()
{
    A = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
    U = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
    B = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
}


type LearningRateAlgorithm::Triplet::get_length() const
{
    return abs(B.first - A.first);
}


pair<type, type> LearningRateAlgorithm::Triplet::minimum() const
{
    Tensor<type, 1> losses(3);

    losses.setValues({ A.second, U.second, B.second });

    const Index minimal_index = opennn::minimal_index(losses);

    if (minimal_index == 0) return A;
    else if (minimal_index == 1) return U;
    else return B;
}


string LearningRateAlgorithm::Triplet::struct_to_string() const
{
    ostringstream buffer;

    buffer << "A = (" << A.first << "," << A.second << ")\n"
        << "U = (" << U.first << "," << U.second << ")\n"
        << "B = (" << B.first << "," << B.second << ")" << endl;

    return buffer.str();
}


void LearningRateAlgorithm::Triplet::print() const
{
    cout << struct_to_string()
        << "Length: " << get_length() << endl;
}


void LearningRateAlgorithm::Triplet::check() const
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
