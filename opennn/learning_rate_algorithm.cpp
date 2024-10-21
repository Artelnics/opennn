//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E A R N I N G   R A T E   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "learning_rate_algorithm.h"
#include "back_propagation.h"

namespace opennn
{

LearningRateAlgorithm::LearningRateAlgorithm()
{
    set_default();
}


LearningRateAlgorithm::LearningRateAlgorithm(LossIndex* new_loss_index)
    : loss_index(new_loss_index)
{
    set_default();
}


LearningRateAlgorithm::~LearningRateAlgorithm()
{
    delete thread_pool;
    delete thread_pool_device;
}


LossIndex* LearningRateAlgorithm::get_loss_index() const
{
#ifdef OPENNN_DEBUG
    if(!loss_index)
        throw runtime_error("Loss index pointer is nullptr.\n");
#endif

    return loss_index;
}


bool LearningRateAlgorithm::has_loss_index() const
{
    return loss_index != nullptr;
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


void LearningRateAlgorithm::set()
{
    loss_index = nullptr;

    set_default();
}


void LearningRateAlgorithm::set(LossIndex* new_loss_index)
{
    loss_index = new_loss_index;

    set_default();
}


void LearningRateAlgorithm::set_default()
{
    delete thread_pool;
    delete thread_pool_device;

    const int n = omp_get_max_threads();
    thread_pool = new ThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(thread_pool, n);

    // TRAINING OPERATORS

    learning_rate_method = LearningRateMethod::BrentMethod;

    // TRAINING PARAMETERS

    learning_rate_tolerance = numeric_limits<type>::epsilon();
    loss_tolerance = numeric_limits<type>::epsilon();
}


void LearningRateAlgorithm::set_loss_index(LossIndex* new_loss_index)
{
    loss_index = new_loss_index;
}


void LearningRateAlgorithm::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr) delete thread_pool;
    if(thread_pool_device != nullptr) delete thread_pool_device;

    thread_pool = new ThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(thread_pool, new_threads_number);
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
#ifdef OPENNN_DEBUG

    if(new_learning_rate_tolerance <= type(0))
        throw runtime_error("Tolerance must be greater than 0.\n");

#endif

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

#ifdef OPENNN_DEBUG

    if(loss_index == nullptr)
        throw runtime_error("Pointer to loss index is nullptr.\n");

    if(neural_network == nullptr)
        throw runtime_error("Pointer to neural network is nullptr.\n");

    if(thread_pool_device == nullptr)
        throw runtime_error("Pointer to thread pool device is nullptr.\n");

#endif

    ostringstream buffer;

    // Bracket minimum

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

        optimization_data.potential_parameters.device(*thread_pool_device)
                = back_propagation.parameters + optimization_data.training_direction*V.first;
        
        neural_network->forward_propagate(batch, optimization_data.potential_parameters, forward_propagation);

        loss_index->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index->calculate_regularization(optimization_data.potential_parameters);

        V.second = back_propagation.error + regularization_weight*regularization;

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

#ifdef OPENNN_DEBUG
    if(loss_index == nullptr)
        throw runtime_error("Pointer to loss index is nullptr.\n");
#endif

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

#ifdef OPENNN_DEBUG

    if(neural_network == nullptr)
        throw runtime_error("Pointer to neural network is nullptr.\n");

    if(thread_pool_device == nullptr)
        throw runtime_error("Pointer to thread pool device is nullptr.\n");

    if(optimization_data.initial_learning_rate < type(NUMERIC_LIMITS_MIN))
        throw runtime_error("Initial learning rate is zero.\n");

#endif

    const type regularization_weight = loss_index->get_regularization_weight();

    // Left point

    triplet.A = { type(0), back_propagation.loss };

    // Right point

    Index count = 0;

    do
    {
        count++;

        triplet.B.first = optimization_data.initial_learning_rate*type(count);

        optimization_data.potential_parameters.device(*thread_pool_device)
                = back_propagation.parameters + optimization_data.training_direction * triplet.B.first;
        
        neural_network->forward_propagate(batch, 
            optimization_data.potential_parameters, forward_propagation);

        loss_index->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index->calculate_regularization(optimization_data.potential_parameters);

        triplet.B.second = back_propagation.error + regularization_weight*regularization;

    } while(abs(triplet.A.second - triplet.B.second) < loss_tolerance && triplet.A.second != triplet.B.second);

    if(triplet.A.second > triplet.B.second)
    {
        triplet.U = triplet.B;

        triplet.B.first *= golden_ratio;

        optimization_data.potential_parameters.device(*thread_pool_device)
                = back_propagation.parameters + optimization_data.training_direction*triplet.B.first;
        
        neural_network->forward_propagate(batch,
                                          optimization_data.potential_parameters,
                                          forward_propagation);

        loss_index->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index->calculate_regularization(optimization_data.potential_parameters);

        triplet.B.second = back_propagation.error + regularization_weight*regularization;

        while(triplet.U.second > triplet.B.second)
        {
            triplet.A = triplet.U;
            triplet.U = triplet.B;

            triplet.B.first *= golden_ratio;

            optimization_data.potential_parameters.device(*thread_pool_device)
                    = back_propagation.parameters + optimization_data.training_direction*triplet.B.first;
            
            neural_network->forward_propagate(batch,
                                              optimization_data.potential_parameters,
                                              forward_propagation);

            loss_index->calculate_error(batch, forward_propagation, back_propagation);

            const type regularization = loss_index->calculate_regularization(optimization_data.potential_parameters);

            triplet.B.second = back_propagation.error + regularization_weight*regularization;
        }
    }
    else if(triplet.A.second < triplet.B.second)
    {
        triplet.U.first = triplet.A.first + (triplet.B.first - triplet.A.first)*type(0.382);

        optimization_data.potential_parameters.device(*thread_pool_device)
                = back_propagation.parameters + optimization_data.training_direction*triplet.U.first;
        
        neural_network->forward_propagate(batch,
                                          optimization_data.potential_parameters,
                                          forward_propagation);

        loss_index->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index->calculate_regularization(optimization_data.potential_parameters);

        triplet.U.second = back_propagation.error + regularization_weight*regularization;

        while(triplet.A.second < triplet.U.second)
        {
            triplet.B = triplet.U;

            triplet.U.first = triplet.A.first + (triplet.B.first-triplet.A.first)*type(0.382);

            optimization_data.potential_parameters.device(*thread_pool_device)
                    = back_propagation.parameters + optimization_data.training_direction*triplet.U.first;
            
            neural_network->forward_propagate(batch, optimization_data.potential_parameters, forward_propagation);

            loss_index->calculate_error(batch, forward_propagation, back_propagation);

            const type regularization = loss_index->calculate_regularization(optimization_data.potential_parameters);

            triplet.U.second = back_propagation.error + regularization_weight*regularization;

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

#ifdef OPENNN_DEBUG

    if(learning_rate < triplet.A.first)
    {
        ostringstream buffer;

        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "type calculate_golden_section_learning_rate(const Triplet&) const method.\n"
               << "Learning rate(" << learning_rate << ") is less than left point("
               << triplet.A.first << ").\n";

        throw runtime_error(buffer.str());
    }

    if(learning_rate > triplet.B.first)
    {
        ostringstream buffer;

        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "type calculate_golden_section_learning_rate(const Triplet&) const method.\n"
               << "Learning rate(" << learning_rate << ") is greater than right point("
               << triplet.B.first << ").\n";

        throw runtime_error(buffer.str());
    }

#endif

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


void LearningRateAlgorithm::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Learning rate algorithm

    file_stream.OpenElement("LearningRateAlgorithm");

    // Learning rate method

    file_stream.OpenElement("LearningRateMethod");
    file_stream.PushText(write_learning_rate_method().c_str());
    file_stream.CloseElement();

    // Learning rate tolerance

    file_stream.OpenElement("LearningRateTolerance");
    file_stream.PushText(to_string(learning_rate_tolerance).c_str());
    file_stream.CloseElement();

    // Learning rate algorithm (end tag)

    file_stream.CloseElement();
}


void LearningRateAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("LearningRateAlgorithm");

    if(!root_element)
        throw runtime_error("Learning rate algorithm element is nullptr.\n");

    // Learning rate method

    const tinyxml2::XMLElement* learning_rate_method_element = root_element->FirstChildElement("LearningRateMethod");

    if(learning_rate_method_element)
        set_learning_rate_method(learning_rate_method_element->GetText());

    // Learning rate tolerance

    const tinyxml2::XMLElement* learning_rate_tolerance_element = root_element->FirstChildElement("LearningRateTolerance");

    if(learning_rate_tolerance_element)
        set_learning_rate_tolerance(type(atof(learning_rate_tolerance_element->GetText())));

    // Display warnings

    const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

    if(display_element)
        set_display(display_element->GetText() != string("0"));
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
