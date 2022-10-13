//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E A R N I N G   R A T E   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "learning_rate_algorithm.h"

namespace opennn
{

/// Default constructor.
/// It creates a learning rate algorithm object not associated with any loss index object.
/// It also initializes the class members to their default values.

LearningRateAlgorithm::LearningRateAlgorithm()
{
    set_default();
}


/// Destructor.
/// It creates a learning rate algorithm associated with a loss index.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

LearningRateAlgorithm::LearningRateAlgorithm(LossIndex* new_loss_index_pointer)
    : loss_index_pointer(new_loss_index_pointer)
{
    set_default();
}


/// Destructor.

LearningRateAlgorithm::~LearningRateAlgorithm()
{
    delete thread_pool;
    delete thread_pool_device;
}


/// Returns a pointer to the loss index object
/// to which the learning rate algorithm is associated.
/// If the loss index pointer is nullptr, this method throws an exception.

LossIndex* LearningRateAlgorithm::get_loss_index_pointer() const
{
#ifdef OPENNN_DEBUG

    if(!loss_index_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
               << "LossIndex* get_loss_index_pointer() const method.\n"
               << "Loss index pointer is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    return loss_index_pointer;
}


/// Returns true if this learning rate algorithm has an associated loss index,
/// and false otherwise.

bool LearningRateAlgorithm::has_loss_index() const
{
    if(loss_index_pointer)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns the learning rate method used for training.

const LearningRateAlgorithm::LearningRateMethod& LearningRateAlgorithm::get_learning_rate_method() const
{
    return learning_rate_method;
}


/// Returns a string with the name of the learning rate method to be used.

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


/// Returns true if messages from this class can be displayed on the screen, or false if messages from
/// this class can't be displayed on the screen.

const bool& LearningRateAlgorithm::get_display() const
{
    return display;
}


/// Sets the loss index pointer to nullptr.
/// It also sets the rest of the members to their default values.

void LearningRateAlgorithm::set()
{
    loss_index_pointer = nullptr;

    set_default();
}


/// Sets a new loss index pointer.
/// It also sets the rest of the members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

void LearningRateAlgorithm::set(LossIndex* new_loss_index_pointer)
{
    loss_index_pointer = new_loss_index_pointer;

    set_default();
}


/// Sets the members of the learning rate algorithm to their default values.

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


/// Sets a pointer to a loss index object to be associated with the optimization algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void LearningRateAlgorithm::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
    loss_index_pointer = new_loss_index_pointer;
}


void LearningRateAlgorithm::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr) delete this->thread_pool;
    if(thread_pool_device != nullptr) delete this->thread_pool_device;

    thread_pool = new ThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(thread_pool, new_threads_number);
}


/// Sets a new learning rate method to be used for training.
/// @param new_learning_rate_method Learning rate method.

void LearningRateAlgorithm::set_learning_rate_method(
        const LearningRateAlgorithm::LearningRateMethod& new_learning_rate_method)
{
    learning_rate_method = new_learning_rate_method;
}


/// Sets the method for obtaining the learning rate from a string with the name of the method.
/// @param new_learning_rate_method Name of learning rate method("Fixed", "GoldenSection", "BrentMethod").

void LearningRateAlgorithm::set_learning_rate_method(const string& new_learning_rate_method)
{
    if(new_learning_rate_method == "GoldenSection")
    {
        learning_rate_method = LearningRateMethod::GoldenSection;
    }
    else if(new_learning_rate_method == "BrentMethod")
    {
        learning_rate_method = LearningRateMethod::BrentMethod;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
               << "void set_method(const string&) method.\n"
               << "Unknown learning rate method: " << new_learning_rate_method << ".\n";

        throw invalid_argument(buffer.str());
    }
}


/// Sets a new tolerance value to be used in line minimization.
/// @param new_learning_rate_tolerance Tolerance value in line minimization.

void LearningRateAlgorithm::set_learning_rate_tolerance(const type& new_learning_rate_tolerance)
{
#ifdef OPENNN_DEBUG

    if(new_learning_rate_tolerance <= static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
               << "void set_learning_rate_tolerance(const type&) method.\n"
               << "Tolerance must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    // Set loss tolerance

    learning_rate_tolerance = new_learning_rate_tolerance;
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void LearningRateAlgorithm::set_display(const bool& new_display)
{
    display = new_display;
}


/// Returns a vector with two elements:
///(i) the learning rate calculated by means of the corresponding algorithm, and
///(ii) the loss for that learning rate.

pair<type,type> LearningRateAlgorithm::calculate_directional_point(
    const DataSetBatch& batch,
    NeuralNetworkForwardPropagation& forward_propagation,
    LossIndexBackPropagation& back_propagation,
    OptimizationAlgorithmData& optimization_data) const
{
    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

#ifdef OPENNN_DEBUG

    if(loss_index_pointer == nullptr)
    {
        ostringstream buffer;

        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "pair<type, 1> calculate_directional_point() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(neural_network_pointer == nullptr)
    {
        ostringstream buffer;

        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "Tensor<type, 1> calculate_directional_point() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(thread_pool_device == nullptr)
    {
        ostringstream buffer;

        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "pair<type, 1> calculate_directional_point() const method.\n"
               << "Pointer to thread pool device is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

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
    catch(const invalid_argument& error)
    {
//        cout << "Triplet bracketing" << endl;

//        cout << error.what() << endl;

//        cout << "!";

        return triplet.minimum();
    }

    const type regularization_weight = loss_index_pointer->get_regularization_weight();

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
        catch(const invalid_argument& error)
        {
            cout << error.what() << endl;

            return triplet.minimum();
        }

        // Calculate loss for V

        optimization_data.potential_parameters.device(*thread_pool_device)
                = back_propagation.parameters + optimization_data.training_direction*V.first;

        neural_network_pointer->forward_propagate(batch, optimization_data.potential_parameters, forward_propagation);

        loss_index_pointer->calculate_errors(batch, forward_propagation, back_propagation);
        loss_index_pointer->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index_pointer->calculate_regularization(optimization_data.potential_parameters);

        V.second = back_propagation.error + regularization_weight*regularization;

        // Update points

        if(V.first <= triplet.U.first)
        {
            if(V.second >= triplet.U.second)
            {
                triplet.A = V;
            }
            else if(V.second <= triplet.U.second)
            {
                triplet.B = triplet.U;
                triplet.U = V;
            }
        }
        else if(V.first >= triplet.U.first)
        {
            if(V.second >= triplet.U.second)
            {
                triplet.B = V;
            }
            else if(V.second <= triplet.U.second)
            {
                triplet.A = triplet.U;
                triplet.U = V;
            }
        }
        else
        {
            buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
                   << "Tensor<type, 1> calculate_Brent_method_directional_point() const method.\n"
                   << "Unknown set:\n"
                   << "A = (" << triplet.A.first << "," << triplet.A.second << ")\n"
                   << "B = (" << triplet.B.first << "," << triplet.B.second << ")\n"
                   << "U = (" << triplet.U.first << "," << triplet.U.second << ")\n"
                   << "V = (" << V.first << "," << V.second << ")\n";

            throw invalid_argument(buffer.str());
        }

        // Check triplet

        try
        {
            triplet.check();
        }
        catch(const invalid_argument& error)
        {
            return triplet.minimum();
        }
    }

    return triplet.U;
}


/// Returns bracketing triplet.
/// This algorithm is used by line minimization algorithms.

LearningRateAlgorithm::Triplet LearningRateAlgorithm::calculate_bracketing_triplet(
    const DataSetBatch& batch,
    NeuralNetworkForwardPropagation& forward_propagation,
    LossIndexBackPropagation& back_propagation,
    OptimizationAlgorithmData& optimization_data) const
{
    Triplet triplet;

#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(loss_index_pointer == nullptr)
    {
        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "Triplet calculate_bracketing_triplet() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw invalid_argument(buffer.str());
    }
#endif

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

#ifdef OPENNN_DEBUG

    if(neural_network_pointer == nullptr)
    {
        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "Triplet calculate_bracketing_triplet() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(thread_pool_device == nullptr)
    {
        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "Triplet calculate_bracketing_triplet() const method.\n"
               << "Pointer to thread pool device is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(is_zero(optimization_data.training_direction))
    {
        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "Triplet calculate_bracketing_triplet() const method.\n"
               << "Training direction is zero.\n";

        throw invalid_argument(buffer.str());
    }

    if(optimization_data.initial_learning_rate < type(NUMERIC_LIMITS_MIN))
    {
        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "Triplet calculate_bracketing_triplet() const method.\n"
               << "Initial learning rate is zero.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    const type loss = back_propagation.loss;

    const type regularization_weight = loss_index_pointer->get_regularization_weight();

    // Left point

    triplet.A.first = type(0);
    triplet.A.second = loss;

    // Right point       

    Index count = 0;

    do
    {
        count++;

        triplet.B.first = optimization_data.initial_learning_rate*static_cast<type>(count);

        optimization_data.potential_parameters.device(*thread_pool_device)
                = back_propagation.parameters + optimization_data.training_direction*triplet.B.first;

        neural_network_pointer->forward_propagate(batch, optimization_data.potential_parameters, forward_propagation);

        loss_index_pointer->calculate_errors(batch, forward_propagation, back_propagation);
        loss_index_pointer->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index_pointer->calculate_regularization(optimization_data.potential_parameters);

        triplet.B.second = back_propagation.error + regularization_weight*regularization;

    } while(abs(triplet.A.second - triplet.B.second) < loss_tolerance);


    if(triplet.A.second > triplet.B.second)
    {
        triplet.U = triplet.B;

        triplet.B.first *= golden_ratio;

        optimization_data.potential_parameters.device(*thread_pool_device)
                = back_propagation.parameters + optimization_data.training_direction*triplet.B.first;

        neural_network_pointer->forward_propagate(batch, optimization_data.potential_parameters, forward_propagation);

        loss_index_pointer->calculate_errors(batch, forward_propagation, back_propagation);
        loss_index_pointer->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index_pointer->calculate_regularization(optimization_data.potential_parameters);

        triplet.B.second = back_propagation.error + regularization_weight*regularization;

        while(triplet.U.second > triplet.B.second)
        {
            triplet.A = triplet.U;
            triplet.U = triplet.B;

            triplet.B.first *= golden_ratio;

            optimization_data.potential_parameters.device(*thread_pool_device)
                    = back_propagation.parameters + optimization_data.training_direction*triplet.B.first;

            neural_network_pointer->forward_propagate(batch, optimization_data.potential_parameters, forward_propagation);

            loss_index_pointer->calculate_errors(batch, forward_propagation, back_propagation);
            loss_index_pointer->calculate_error(batch, forward_propagation, back_propagation);

            const type regularization = loss_index_pointer->calculate_regularization(optimization_data.potential_parameters);

            triplet.B.second = back_propagation.error + regularization_weight*regularization;
        }
    }
    else if(triplet.A.second < triplet.B.second)
    {
        triplet.U.first = triplet.A.first + (triplet.B.first - triplet.A.first)*static_cast<type>(0.382);

        optimization_data.potential_parameters.device(*thread_pool_device)
                = back_propagation.parameters + optimization_data.training_direction*triplet.U.first;

        neural_network_pointer->forward_propagate(batch, optimization_data.potential_parameters, forward_propagation);

        loss_index_pointer->calculate_errors(batch, forward_propagation, back_propagation);
        loss_index_pointer->calculate_error(batch, forward_propagation, back_propagation);

        const type regularization = loss_index_pointer->calculate_regularization(optimization_data.potential_parameters);

        triplet.U.second = back_propagation.error + regularization_weight*regularization;

        while(triplet.A.second < triplet.U.second)
        {
            triplet.B = triplet.U;

            triplet.U.first = triplet.A.first + (triplet.B.first-triplet.A.first)*static_cast<type>(0.382);

            optimization_data.potential_parameters.device(*thread_pool_device)
                    = back_propagation.parameters + optimization_data.training_direction*triplet.U.first;

            neural_network_pointer->forward_propagate(batch, optimization_data.potential_parameters, forward_propagation);

            loss_index_pointer->calculate_errors(batch, forward_propagation, back_propagation);
            loss_index_pointer->calculate_error(batch, forward_propagation, back_propagation);

            const type regularization = loss_index_pointer->calculate_regularization(optimization_data.potential_parameters);

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


/// Calculates the golden section point within a minimum interval defined by three points.
/// @param triplet Triplet containing a minimum.

type LearningRateAlgorithm::calculate_golden_section_learning_rate(const Triplet& triplet) const
{
    type learning_rate;

    const type middle = triplet.A.first + static_cast<type>(0.5)*(triplet.B.first - triplet.A.first);

    if(triplet.U.first < middle)
    {
        learning_rate = triplet.A.first + static_cast<type>(0.618)*(triplet.B.first - triplet.A.first);
    }
    else
    {
        learning_rate = triplet.A.first + static_cast<type>(0.382)*(triplet.B.first - triplet.A.first);
    }

#ifdef OPENNN_DEBUG

    if(learning_rate < triplet.A.first)
    {
        ostringstream buffer;

        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "type calculate_golden_section_learning_rate(const Triplet&) const method.\n"
               << "Learning rate(" << learning_rate << ") is less than left point("
               << triplet.A.first << ").\n";

        throw invalid_argument(buffer.str());
    }

    if(learning_rate > triplet.B.first)
    {
        ostringstream buffer;

        buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
               << "type calculate_golden_section_learning_rate(const Triplet&) const method.\n"
               << "Learning rate(" << learning_rate << ") is greater than right point("
               << triplet.B.first << ").\n";

        throw invalid_argument(buffer.str());
    }

#endif

    return learning_rate;
}


/// Returns the minimimal learning rate of a parabola defined by three directional points.
/// @param triplet Triplet containing a minimum.

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

    return u - static_cast<type>(0.5)*numerator/denominator;
}


/// Serializes the learning rate algorithm object into an XML document of the TinyXML library
/// without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void LearningRateAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Learning rate algorithm

    file_stream.OpenElement("LearningRateAlgorithm");

    // Learning rate method

    file_stream.OpenElement("LearningRateMethod");

    file_stream.PushText(write_learning_rate_method().c_str());

    file_stream.CloseElement();

    // Learning rate tolerance

    file_stream.OpenElement("LearningRateTolerance");

    buffer.str("");
    buffer << learning_rate_tolerance;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Learning rate algorithm (end tag)

    file_stream.CloseElement();
}


/// Loads a learning rate algorithm object from an XML-type file.
/// Please mind about the file format, wich is specified in the manual.
/// @param document TinyXML document with the learning rate algorithm members.

void LearningRateAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("LearningRateAlgorithm");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Learning rate algorithm element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Learning rate method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("LearningRateMethod");

        if(element)
        {
            string new_learning_rate_method = element->GetText();

            try
            {
                set_learning_rate_method(new_learning_rate_method);
            }
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Learning rate tolerance
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("LearningRateTolerance");

        if(element)
        {
            const type new_learning_rate_tolerance = static_cast<type>(atof(element->GetText()));

            try
            {
                set_learning_rate_tolerance(new_learning_rate_tolerance);
            }
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Display warnings
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

        if(element)
        {
            const string new_display = element->GetText();

            try
            {
                set_display(new_display != "0");
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
