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

InputsSelection::InputsSelection(TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);
}


TrainingStrategy* InputsSelection::get_training_strategy() const
{
    return training_strategy;
}


bool InputsSelection::has_training_strategy() const
{
    return training_strategy;
}


const Index& InputsSelection::get_trials_number() const
{
    return trials_number;
}


const bool& InputsSelection::get_display() const
{
    return display;
}


const type& InputsSelection::get_selection_error_goal() const
{
    return selection_error_goal;
}


const Index& InputsSelection::get_maximum_iterations_number() const
{
    return maximum_epochs_number;
}


const type& InputsSelection::get_maximum_time() const
{
    return maximum_time;
}


const type& InputsSelection::get_maximum_correlation() const
{
    return maximum_correlation;
}


const type& InputsSelection::get_minimum_correlation() const
{
    return minimum_correlation;
}


void InputsSelection::set(TrainingStrategy* new_training_strategy)
{
    training_strategy = new_training_strategy;     
}


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


void InputsSelection::set_trials_number(const Index& new_trials_number)
{
    trials_number = new_trials_number;
}


void InputsSelection::set_display(const bool& new_display)
{
    display = new_display;
}


void InputsSelection::set_selection_error_goal(const type& new_selection_error_goal)
{
    selection_error_goal = new_selection_error_goal;
}


void InputsSelection::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
    
}


void InputsSelection::set_maximum_time(const type& new_maximum_time)
{
    maximum_time = new_maximum_time;
}


void InputsSelection::set_maximum_correlation(const type& new_maximum_correlation)
{
    maximum_correlation = new_maximum_correlation;
}


void InputsSelection::set_minimum_correlation(const type& new_minimum_correlation)
{
    minimum_correlation = new_minimum_correlation;
}


string InputsSelection::write_stopping_condition(const TrainingResults& results) const
{
    return results.write_stopping_condition();
}


void InputsSelection::check() const
{
    if(!training_strategy)
        throw runtime_error("Pointer to training strategy is nullptr.\n");

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();

    // Neural network

    if(!loss_index->has_neural_network())
        throw runtime_error("Pointer to neural network is nullptr.\n");

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    if(neural_network->is_empty())
        throw runtime_error("Neural network is empty.\n");

    // Data set

    if(!loss_index->has_data_set())
        throw runtime_error("Pointer to data set is nullptr.\n");

    const DataSet* data_set = loss_index->get_data_set();

    const Index selection_samples_number = data_set->get_samples_number(DataSet::SampleUse::Selection);

    if(selection_samples_number == 0)
        throw runtime_error("Number of selection samples is zero.\n");
}


InputsSelectionResults::InputsSelectionResults(const Index &maximum_epochs_number)
{
    set(maximum_epochs_number);
}


Index InputsSelectionResults::get_epochs_number() const
{
    return training_error_history.size();
}


void InputsSelectionResults::set(const Index& maximum_epochs_number)
{
    training_error_history.resize(maximum_epochs_number);
    training_error_history.setConstant(type(-1));

    selection_error_history.resize(maximum_epochs_number);
    selection_error_history.setConstant(type(-1));

    mean_selection_error_history.resize(maximum_epochs_number);
    mean_selection_error_history.setConstant(type(-1));

    mean_training_error_history.resize(maximum_epochs_number);
    mean_training_error_history.setConstant(type(-1));
}

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

void InputsSelectionResults::resize_history(const Index& new_size)
{
    const Tensor<type, 1> old_training_error_history(training_error_history);
    const Tensor<type, 1> old_selection_error_history(selection_error_history);

    const Tensor<type, 1> old_mean_selection_history(mean_selection_error_history);
    const Tensor<type, 1> old_mean_training_history(mean_training_error_history);

    training_error_history.resize(new_size);
    selection_error_history.resize(new_size);
    mean_training_error_history.resize(new_size);
    mean_selection_error_history.resize(new_size);

    for(Index i = 0; i < new_size; i++)
    {
        training_error_history(i) = old_training_error_history(i);
        selection_error_history(i) = old_selection_error_history(i);
        mean_training_error_history(i) = old_mean_training_history(i);
        mean_selection_error_history(i) = old_mean_selection_history(i);
    }
}


void InputsSelectionResults::print() const
{
    cout << endl
         << "Inputs Selection Results" << endl
         << "Optimal inputs number: " << optimal_input_raw_variables_names.size() << endl
         << "Inputs: " << endl;

    for(size_t i = 0; i < optimal_input_raw_variables_names.size(); i++)
        cout << "   " << optimal_input_raw_variables_names[i] << endl;

    cout << "Optimum training error: " << optimum_training_error << endl
         << "Optimum selection error: " << optimum_selection_error << endl;
}


string InputsSelection::write_time(const type& time) const
{
    const int hours = int(time) / 3600;
    int seconds = int(time) % 3600;
    const int minutes = seconds / 60;
    seconds = seconds % 60;

    ostringstream elapsed_time;

    elapsed_time << setfill('0')
        << setw(2) << hours << ":"
        << setw(2) << minutes << ":"
        << setw(2) << seconds << endl;

    return elapsed_time.str();
}


Index InputsSelection::get_input_index(const Tensor<DataSet::VariableUse, 1>& uses, const Index& inputs_number) const
{
    Index i = 0;
    Index j = 0;

    while(i < uses.size())
    {
        if (uses[i] == DataSet::VariableUse::Input)
        {
            if (j == inputs_number)
                break;

            j++;
        }

        i++;
    }

    return i;
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
