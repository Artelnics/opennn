//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "neurons_selection.h"

namespace OpenNN
{

/// Default constructor.

NeuronsSelection::NeuronsSelection()
{
    training_strategy_pointer = nullptr;

    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

NeuronsSelection::NeuronsSelection(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;

    set_default();
}


/// File constructor.
/// @param file_name Name of XML order selection file.
/// @todo

NeuronsSelection::NeuronsSelection(const string& file_name)
{
    training_strategy_pointer = nullptr;

//    from_XML(file_name);
}


/// XML constructor.
/// @param neurons_selection_document Pointer to a TinyXML document containing the order selection algorithm data.

NeuronsSelection::NeuronsSelection(const tinyxml2::XMLDocument&)
{
    training_strategy_pointer = nullptr;
//    from_XML(neurons_selection_document);
}


/// Destructor.

NeuronsSelection::~NeuronsSelection()
{
}


/// Returns a pointer to the training strategy object.

TrainingStrategy* NeuronsSelection::get_training_strategy_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!training_strategy_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "DataSet* get_training_strategy_pointer() const method.\n"
               << "Training strategy pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return training_strategy_pointer;
}


/// Returns true if this order selection algorithm has a training strategy associated, and false otherwise.

bool NeuronsSelection::has_training_strategy() const
{
    if(training_strategy_pointer != nullptr)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns the maximum of the hidden perceptrons number used in the order order selection.

const size_t& NeuronsSelection::get_maximum_order() const
{
    return maximum_order;
}


/// Returns the minimum of the hidden perceptrons number used in the order selection.

const size_t& NeuronsSelection::get_minimum_order() const
{
    return minimum_order;
}


/// Returns the number of trials for each network architecture.

const size_t& NeuronsSelection::get_trials_number() const
{
    return trials_number;
}


/// Returns true if the loss index losses are to be reserved, and false otherwise.

const bool& NeuronsSelection::get_reserve_error_data() const
{
    return reserve_error_data;
}


/// Returns true if the loss index selection losses are to be reserved, and false otherwise.

const bool& NeuronsSelection::get_reserve_selection_error_data() const
{
    return reserve_selection_error_data;
}


/// Returns true if the parameters vector of the neural network with minimum selection error is to be reserved, and false otherwise.

const bool& NeuronsSelection::get_reserve_minimal_parameters() const
{
    return reserve_minimal_parameters;
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& NeuronsSelection::get_display() const
{
    return display;
}


/// Returns the goal for the selection error in the order selection algorithm.

const double& NeuronsSelection::get_selection_error_goal() const
{
    return selection_error_goal;
}


/// Returns the maximum number of iterations in the order selection algorithm.

const size_t& NeuronsSelection::get_maximum_iterations_number() const
{
    return maximum_iterations_number;
}


/// Returns the maximum time in the order selection algorithm.

const double& NeuronsSelection::get_maximum_time() const
{
    return maximum_time;
}


/// Return the tolerance of error for the order selection algorithm.

const double& NeuronsSelection::get_tolerance() const
{
    return tolerance;
}


/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void NeuronsSelection::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;
}


/// Sets the members of the order selection object to their default values.

void NeuronsSelection::set_default()
{
    size_t inputs_number;
    size_t outputs_number;

    if(training_strategy_pointer == nullptr
    || !training_strategy_pointer->has_neural_network())
    {
        inputs_number = 0;
        outputs_number = 0;
    }
    else
    {
        inputs_number = training_strategy_pointer->get_neural_network_pointer()->get_inputs_number();
        outputs_number = training_strategy_pointer->get_neural_network_pointer()->get_outputs_number();
    }
    // MEMBERS

    minimum_order = 1;

    // Heuristic value for the maximum_order

    maximum_order = 2*(inputs_number + outputs_number);
    trials_number = 1;

    // Order selection results

    reserve_error_data = true;
    reserve_selection_error_data = true;
    reserve_minimal_parameters = true;

    display = true;

    // Stopping criteria

    selection_error_goal = 0.0;

    maximum_iterations_number = 1000;
    maximum_time = 10000.0;

    tolerance = 0.0;
}


/// Sets the number of the maximum hidden perceptrons for the order selection algorithm.
/// @param new_maximum_order Number of maximum hidden perceptrons.

void NeuronsSelection::set_maximum_order(const size_t& new_maximum_order)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_order <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_maximum_order(const size_t&) method.\n"
               << "maximum_order(" << new_maximum_order << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(new_maximum_order < minimum_order)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_maximum_order(const size_t&) method.\n"
               << "maximum_order(" << new_maximum_order << ") must be equal or greater than minimum_order(" << minimum_order << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_order = new_maximum_order;
}


/// Sets the number of the minimum hidden perceptrons for the order selection algorithm.
/// @param new_minimum_order Number of minimum hidden perceptrons.

void NeuronsSelection::set_minimum_order(const size_t& new_minimum_order)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_order <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_minimum_order(const size_t&) method.\n"
               << "minimum_order(" << new_minimum_order << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(new_minimum_order >= maximum_order)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_minimum_order(const size_t&) method.\n"
               << "minimum_order(" << new_minimum_order << ") must be less than maximum_order(" << maximum_order << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_order = new_minimum_order;
}


/// Sets the number of times that each different neural network is to be trained.
/// @param new_trials_number Number of assays for each set of parameters.

void NeuronsSelection::set_trials_number(const size_t& new_trials_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_trials_number <= 0)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_trials_number(const size_t&) method.\n"
               << "Number of assays must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    trials_number = new_trials_number;
}


/// Sets the reserve flag for the loss data.
/// @param new_reserve_error_data Flag value.

void NeuronsSelection::set_reserve_error_data(const bool& new_reserve_error_data)
{
    reserve_error_data = new_reserve_error_data;
}


/// Sets the reserve flag for the selection error data.
/// @param new_reserve_selection_error_data Flag value.

void NeuronsSelection::set_reserve_selection_error_data(const bool& new_reserve_selection_error_data)
{
    reserve_selection_error_data = new_reserve_selection_error_data;
}


/// Sets the reserve flag for the minimal parameters.
/// @param new_reserve_minimal_parameters Flag value.

void NeuronsSelection::set_reserve_minimal_parameters(const bool& new_reserve_minimal_parameters)
{
    reserve_minimal_parameters = new_reserve_minimal_parameters;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void NeuronsSelection::set_display(const bool& new_display)
{
    display = new_display;
}


/// Sets the selection error goal for the order selection algorithm.
/// @param new_selection_error_goal Goal of the selection error.

void NeuronsSelection::set_selection_error_goal(const double& new_selection_error_goal)
{
#ifdef __OPENNN_DEBUG__

    if(new_selection_error_goal < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_selection_error_goal(const double&) method.\n"
               << "Selection loss goal must be greater or equal than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    selection_error_goal = new_selection_error_goal;
}


/// Sets the maximum iterations number for the order selection algorithm.
/// @param new_maximum_iterations_number Maximum number of iterations.

void NeuronsSelection::set_maximum_iterations_number(const size_t& new_maximum_iterations_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_iterations_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_maximum_iterations_number(const size_t&) method.\n"
               << "Maximum iterations number must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_iterations_number = new_maximum_iterations_number;
}


/// Sets the maximum time for the order selection algorithm.
/// @param new_maximum_time Maximum time for the algorithm.

void NeuronsSelection::set_maximum_time(const double& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_maximum_time(const double&) method.\n"
               << "Maximum time must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_time = new_maximum_time;
}


/// Set the tolerance for the errors in the trainings of the algorithm.
/// @param new_tolerance Value of the tolerance.

void NeuronsSelection::set_tolerance(const double& new_tolerance)
{
#ifdef __OPENNN_DEBUG__

    if(new_tolerance < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void set_tolerance(const double&) method.\n"
               << "Tolerance must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    tolerance = new_tolerance;
}


/// Returns the minimum of the loss and selection error in trials_number trainings.
/// @param order_number Number of neurons in the hidden layer to be trained with.

Vector<double> NeuronsSelection::calculate_losses(const size_t& neurons_number, NeuralNetwork& neural_network)
{
#ifdef __OPENNN_DEBUG__

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "Vector<double> calculate_losses(size_t) method.\n"
               << "Number of hidden neurons must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(trials_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "Vector<double> calculate_losses(size_t) method.\n"
               << "Number of trials must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Neural network stuff

    const size_t trainable_layers_number = neural_network.get_trainable_layers_number();

    const Vector<Layer*> trainable_layers_pointers = neural_network.get_trainable_layers_pointers();

    // Loss index stuff

    double optimum_selection_error = numeric_limits<double>::max();
    double optimum_training_error = numeric_limits<double>::max();
    Vector<double> optimum_parameters;

    // Optimization algorithm stuff

    OptimizationAlgorithm::Results results;

    Vector<double> final_losses(2, numeric_limits<double>::max());

    Vector<double> current_loss(2);

    Vector<double> final_parameters;

    bool flag_training = false;
    bool flag_selection = false;

    for(size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == neurons_number)
        {
            final_losses[0] = training_loss_history[i];
            flag_training = true;
        }
    }

    for(size_t i = 0; i < order_history.size(); i++)
    {
        if(order_history[i] == neurons_number)
        {
            final_losses[1] = selection_error_history[i];
            flag_selection = true;
        }
    }

    if(flag_training && flag_selection)
    {
        return final_losses;
    }

    trainable_layers_pointers[trainable_layers_number-2]->set_neurons_number(neurons_number); // Fix
    trainable_layers_pointers[trainable_layers_number-1]->set_inputs_number(neurons_number); // Fix

    for(size_t i = 0; i < trials_number; i++)
    {
        neural_network.randomize_parameters_normal();

        const OptimizationAlgorithm::Results optimization_algorithm_results = training_strategy_pointer->perform_training();

        const double current_training_error = optimization_algorithm_results.final_training_error;
        const double current_selection_error = optimization_algorithm_results.final_selection_error;
        const Vector<double> current_parameters = optimization_algorithm_results.final_parameters;

        if(current_selection_error < optimum_selection_error)
        {
            optimum_training_error = current_training_error;
            optimum_selection_error = current_selection_error;
            optimum_parameters = current_parameters; // Optimum parameters = final_parameters
        }

        if(display)
        {
            cout << "Trial number: " << i << endl;
            cout << "Training error: " << current_training_error << endl;
            cout << "Selection error: " << current_selection_error << endl;
            cout << "Stopping condition: " << optimization_algorithm_results.write_stopping_condition() << endl << endl;
        }
    }


    // Save results

    final_losses[0] = optimum_training_error;
    final_losses[1] = optimum_selection_error;

    order_history.push_back(neurons_number);

    training_loss_history.push_back(final_losses[0]);

    selection_error_history.push_back(final_losses[1]);

    parameters_history.push_back(optimum_parameters);

    return final_losses;
}


/// Return final training loss and final selection error depending on the training method.
/// @param results Results of the perform_training method.
/*
Vector<double> NeuronsSelection::get_final_losses(const OptimizationAlgorithm::Results& results) const
{
    Vector<double> losses(2);

    switch(training_strategy_pointer->get_optimization_method())
    {
        case TrainingStrategy::GRADIENT_DESCENT:
        {
            losses[0] = results.final_training_error;
            losses[1] = results.final_selection_error;
            return losses;
        }
        case TrainingStrategy::CONJUGATE_GRADIENT:
        {
            losses[0] = results.final_training_error;
            losses[1] = results.final_selection_error;
            return losses;
        }
        case TrainingStrategy::QUASI_NEWTON_METHOD:
        {
            losses[0] = results.final_training_error;
            losses[1] = results.final_selection_error;
            return losses;
        }
        case TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM:
        {
            losses[0] = results.final_training_error;
            losses[1] = results.final_selection_error;
            return losses;
        }
        case TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT:
        {
            losses[0] = results.final_training_error;
            losses[1] = results.final_selection_error;
            return losses;
        }
        case TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION:
        {
            losses[0] = results.final_training_error;
            losses[1] = results.final_selection_error;
            return losses;
        }
//        default:
//        {
//            ostringstream buffer;

//            buffer << "OpenNN Exception: NeuronsSelection class.\n"
//                   << "Vector<double> get_final_losses(const OptimizationAlgorithm::Results) method.\n"
//                   << "Unknown main type method.\n";

//            throw logic_error(buffer.str());
//        }
    }

    // Default

    ostringstream buffer;

    buffer << "OpenNN Exception: NeuronsSelection class.\n"
           << "Vector<double> get_final_losses(const OptimizationAlgorithm::Results) method.\n"
           << "Unknown main type method.\n";

    throw logic_error(buffer.str());
}
*/


/// Return a string with the stopping condition of the training depending on the training method.
/// @param results Results of the perform_training method.

string NeuronsSelection::write_stopping_condition(const OptimizationAlgorithm::Results& results) const
{
    return results.write_stopping_condition();
}


/// Delete the history of the selection error values.

void NeuronsSelection::delete_selection_history()
{
    selection_error_history.set();
}


/// Delete the history of the loss values.

void NeuronsSelection::delete_training_loss_history()
{
    training_loss_history.set();
}


/// Checks that the different pointers needed for performing the order selection are not nullptr.

void NeuronsSelection::check() const
{
    // Optimization algorithm stuff

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Neural network stuff

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(neural_network_pointer->is_empty())
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Multilayer Perceptron is empty.\n";

        throw logic_error(buffer.str());
    }


   if(neural_network_pointer->get_layers_number() == 1)
   {
      buffer << "OpenNN Exception: NeuronsSelection class.\n"
             << "void check() const method.\n"
             << "Number of layers in neural network must be greater than 1.\n";

      throw logic_error(buffer.str());
   }

    // Data set stuff

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const size_t selection_instances_number = data_set_pointer->get_selection_instances_number();

    if(selection_instances_number == 0)
    {
        buffer << "OpenNN Exception: NeuronsSelection class.\n"
               << "void check() const method.\n"
               << "Number of selection instances is zero.\n";

        throw logic_error(buffer.str());
    }

}


/// Return a string with the stopping condition of the Results

string NeuronsSelection::Results::write_stopping_condition() const
{
    switch(stopping_condition)
    {
        case MaximumTime:
        {
            return "MaximumTime";
        }
        case SelectionErrorGoal:
        {
            return "SelectionErrorGoal";
        }
        case MaximumIterations:
        {
            return "MaximumIterations";
        }
        case MaximumSelectionFailures:
        {
            return "MaximumSelectionFailures";
        }
        case AlgorithmFinished:
        {
            return "AlgorithmFinished";
        }
    }

    return string();
}


/// Returns a string representation of the current order selection results structure.

string NeuronsSelection::Results::object_to_string() const
{
   ostringstream buffer;

   // Order history

   if(!neurons_data.empty())
   {
     buffer << "% Order history:\n"
            << neurons_data.to_row_matrix() << "\n";
   }

   // Loss history

   if(!training_loss_data.empty())
   {
       buffer << "% Loss history:\n"
              << training_loss_data.to_row_matrix() << "\n";
   }

   // Selection loss history

   if(!selection_error_data.empty())
   {
       buffer << "% Selection loss history:\n"
              << selection_error_data.to_row_matrix() << "\n";
   }

   // Minimal parameters

   if(!minimal_parameters.empty())
   {
       buffer << "% Minimal parameters:\n"
              << minimal_parameters << "\n";
   }

   // Stopping condition

   buffer << "% Stopping condition\n"
          << write_stopping_condition() << "\n";

   // Optimum selection error

   if(final_selection_error > numeric_limits<double>::epsilon())
   {
       buffer << "% Optimum selection error:\n"
              << final_selection_error << "\n";
   }

   // Final loss

   if(final_training_loss > numeric_limits<double>::epsilon())
   {
       buffer << "% Final loss:\n"
              << final_training_loss << "\n";
   }

   // Optimal order

   if(optimal_neurons_number != 0)
   {
       buffer << "% Optimal order:\n"
              << optimal_neurons_number << "\n";
   }

   // Iterations number


   buffer << "% Number of iterations:\n"
          << iterations_number << "\n";


   // Elapsed time

   buffer << "% Elapsed time:\n"
          << write_elapsed_time(elapsed_time) << "\n";



   return buffer.str();
}
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
