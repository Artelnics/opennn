//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "inputs_selection.h"

namespace opennn
{

/// Default constructor.

InputsSelection::InputsSelection()
{
    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a trainig strategy object.

InputsSelection::InputsSelection(TrainingStrategy* new_training_strategy_pointer)
    : training_strategy_pointer(new_training_strategy_pointer)
{
    set_default();
}


/// Returns a pointer to the training strategy object.

TrainingStrategy* InputsSelection::get_training_strategy_pointer() const
{
#ifdef OPENNN_DEBUG

    if(!training_strategy_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "DataSet* get_training_strategy_pointer() const method.\n"
               << "Training strategy pointer is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    return training_strategy_pointer;
}


/// Returns true if this inputs selection algorithm has a training strategy associated, and false otherwise.

bool InputsSelection::has_training_strategy() const
{
    if(training_strategy_pointer)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns the number of trials for each network architecture.

const Index& InputsSelection::get_trials_number() const
{
    return trials_number;
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& InputsSelection::get_display() const
{
    return display;
}


/// Returns the goal for the selection error in the inputs selection algorithm.

const type& InputsSelection::get_selection_error_goal() const
{
    return selection_error_goal;
}


/// Returns the maximum number of iterations in the inputs selection algorithm.

const Index& InputsSelection::get_maximum_iterations_number() const
{
    return maximum_epochs_number;
}


/// Returns the maximum time in the inputs selection algorithm.

const type& InputsSelection::get_maximum_time() const
{
    return maximum_time;
}


/// Return the maximum correlation for the algorithm.

const type& InputsSelection::get_maximum_correlation() const
{
    return maximum_correlation;
}


/// Return the minimum correlation for the algorithm.

const type& InputsSelection::get_minimum_correlation() const
{
    return minimum_correlation;
}


/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void InputsSelection::set(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;     
}


/// Sets the members of the inputs selection object to their default values.

void InputsSelection::set_default()
{
    trials_number = 1;

    // Stopping criteria

    selection_error_goal = type(0);

    maximum_epochs_number = 1000;

    maximum_correlation = type(1);
    minimum_correlation = type(0);

    maximum_time = type(36000.0);
}


/// Sets the number of times that each different neural network is to be trained.
/// @param new_trials_number Number of trials for each set of parameters.

void InputsSelection::set_trials_number(const Index& new_trials_number)
{
#ifdef OPENNN_DEBUG

    if(new_trials_number <= 0)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_trials_number(const Index&) method.\n"
               << "Number of assays must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    trials_number = new_trials_number;
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void InputsSelection::set_display(const bool& new_display)
{
    display = new_display;
}


/// Sets the selection error goal for the inputs selection algorithm.
/// @param new_selection_error_goal Goal of the selection error.

void InputsSelection::set_selection_error_goal(const type& new_selection_error_goal)
{
#ifdef OPENNN_DEBUG

    if(new_selection_error_goal < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_selection_error_goal(const type&) method.\n"
               << "Selection loss goal must be greater or equal than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    selection_error_goal = new_selection_error_goal;
}


/// Sets the maximum iterations number for the inputs selection algorithm.
/// @param new_maximum_iterations_number Maximum number of epochs.

void InputsSelection::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


/// Sets the maximum time for the inputs selection algorithm.
/// @param new_maximum_time Maximum time for the algorithm.

void InputsSelection::set_maximum_time(const type& new_maximum_time)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_time < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    maximum_time = new_maximum_time;
}


/// Sets the maximum value for the correlations in the inputs selection algorithm.
/// @param new_maximum_correlation Maximum value of the correlations.

void InputsSelection::set_maximum_correlation(const type& new_maximum_correlation)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_correlation < 0 || new_maximum_correlation > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_maximum_correlation(const type&) method.\n"
               << "Maximum correlation must be comprised between 0 and 1.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    maximum_correlation = new_maximum_correlation;
}


/// Sets the minimum value for the correlations in the inputs selection algorithm.
/// @param new_minimum_correlation Minimum value of the correlations.

void InputsSelection::set_minimum_correlation(const type& new_minimum_correlation)
{
#ifdef OPENNN_DEBUG

    if(new_minimum_correlation < 0 || new_minimum_correlation > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void set_minimum_correlation(const type&) method.\n"
               << "Minimum correaltion must be comprised between 0 and 1.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    minimum_correlation = new_minimum_correlation;
}


/// Return a string with the stopping condition of the training depending on the training method.
/// @param results Results of the perform_training method.

string InputsSelection::write_stopping_condition(const TrainingResults& results) const
{
    return results.write_stopping_condition();
}


/// Checks that the different pointers needed for performing the inputs selection are not nullptr.

void InputsSelection::check() const
{
    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    // Neural network

    if(!loss_index_pointer->has_neural_network())
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(neural_network_pointer->is_empty())
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Neural network is empty.\n";

        throw invalid_argument(buffer.str());
    }

    // Data set

    if(!loss_index_pointer->has_data_set())
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();

    if(selection_samples_number == 0)
    {
        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "void check() const method.\n"
               << "Number of selection samples is zero.\n";

        throw invalid_argument(buffer.str());
    }
}


/// Return a string with the stopping condition of the Results

string InputsSelectionResults::write_stopping_condition() const
{
    switch(stopping_condition)
    {
    case InputsSelection::StoppingCondition::MaximumTime:
        return "MaximumTime";

    case InputsSelection::StoppingCondition::SelectionErrorGoal:
        return "SelectionErrorGoal";

    case InputsSelection::StoppingCondition::MaximumInputs:
        return "MaximumInputs";

    case InputsSelection::StoppingCondition::MinimumInputs:
        return "MinimumInputs";

    case InputsSelection::StoppingCondition::MaximumEpochs:
        return "MaximumEpochs";

    case InputsSelection::StoppingCondition::MaximumSelectionFailures:
        return "MaximumSelectionFailures";

    case InputsSelection::StoppingCondition::CorrelationGoal:
        return "CorrelationGoal";
    default:
        return string();
    }
}


/// Writes the time from seconds in format HH:mm:ss.

string InputsSelection::write_time(const type& time) const
{

#ifdef OPENNN_DEBUG

    if(time > static_cast<type>(3600e5))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OptimizationAlgorithm class.\n"
               << "const string write_time(const type& time) const method.\n"
               << "Time must be lower than 10e5 seconds.\n";

        throw invalid_argument(buffer.str());
    }

    if(time < static_cast<type>(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OptimizationAlgorithm class.\n"
               << "const string write_time(const type& time) const method.\n"
               << "Time must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }
#endif

    int hours = static_cast<int>(time) / 3600;
    int seconds = static_cast<int>(time) % 3600;
    int minutes = seconds / 60;
    seconds = seconds % 60;

    ostringstream elapsed_time;

    elapsed_time << setfill('0') << setw(2) << hours << ":"
                 << setfill('0') << setw(2) << minutes << ":"
                 << setfill('0') << setw(2) << seconds << endl;

    return elapsed_time.str();
}


/// Return the index of uses where is the(inputs_number)-th input.
/// @param uses Vector of the uses of the variables.
/// @param inputs_number Index of the input to find.

Index InputsSelection::get_input_index(const Tensor<DataSet::VariableUse, 1>& uses, const Index& inputs_number) const
{
#ifdef OPENNN_DEBUG

    if(uses.size() < inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsSelection class.\n"
               << "const Index get_input_index(const Tensor<DataSet::VariableUse, 1>, const Index) method.\n"
               << "Size of uses vector("<< uses.size() <<") must be greater than " <<  inputs_number << ".\n";

        throw invalid_argument(buffer.str());
    }
#endif

    Index i = 0;

    Index j = 0;

    while(i < uses.size())
    {
        if(uses[i] == DataSet::VariableUse::Input && inputs_number == j)
        {
            break;
        }
        else if(uses[i] == DataSet::VariableUse::Input)
        {
            i++;
            j++;
        }
        else
        {
            i++;
        }
    }
    return i;
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
