//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "neurons_selection.h"

namespace opennn
{

NeuronsSelection::NeuronsSelection(TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);
}


TrainingStrategy* NeuronsSelection::get_training_strategy() const
{
    return training_strategy;
}


bool NeuronsSelection::has_training_strategy() const
{
    return training_strategy;
}


const Index& NeuronsSelection::get_maximum_neurons() const
{
    return maximum_neurons;
}


const Index& NeuronsSelection::get_minimum_neurons() const
{
    return minimum_neurons;
}


const Index& NeuronsSelection::get_trials_number() const
{
    return trials_number;
}


const bool& NeuronsSelection::get_display() const
{
    return display;
}


const type& NeuronsSelection::get_selection_error_goal() const
{
    return selection_error_goal;
}


const Index& NeuronsSelection::get_maximum_epochs_number() const
{
    return maximum_epochs_number;
}


const type& NeuronsSelection::get_maximum_time() const
{
    return maximum_time;
}


void NeuronsSelection::set(TrainingStrategy* new_training_strategy)
{
    training_strategy = new_training_strategy;

    set_default();
}


void NeuronsSelection::set_training_strategy(TrainingStrategy* new_training_strategy)
{
    training_strategy = new_training_strategy;
}


void NeuronsSelection::set_default()
{
    if (!(training_strategy && training_strategy->get_neural_network()))
        return;

    const Index inputs_number = training_strategy->get_neural_network()->get_inputs_number();
    const Index outputs_number = training_strategy->get_neural_network()->get_outputs_number();

    minimum_neurons = 1;
    maximum_neurons = 2 * (inputs_number + outputs_number);
    trials_number = 1;
    display = true;

    selection_error_goal = type(0);
    maximum_epochs_number = 1000;
    maximum_time = type(3600);
}


void NeuronsSelection::set_maximum_neurons(const Index& new_maximum_neurons)
{
    maximum_neurons = new_maximum_neurons;
}


void NeuronsSelection::set_minimum_neurons(const Index& new_minimum_neurons)
{
    minimum_neurons = new_minimum_neurons;
}


void NeuronsSelection::set_trials_number(const Index& new_trials_number)
{
    trials_number = new_trials_number;
}


void NeuronsSelection::set_display(const bool& new_display)
{
    display = new_display;
}


void NeuronsSelection::set_selection_error_goal(const type& new_selection_error_goal)
{
    selection_error_goal = new_selection_error_goal;
}


void NeuronsSelection::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


void NeuronsSelection::set_maximum_time(const type& new_maximum_time)
{
    maximum_time = new_maximum_time;
}


string NeuronsSelection::write_stopping_condition(const TrainingResults& results) const
{
    return results.write_stopping_condition();
}


void NeuronsSelection::delete_selection_history()
{
    selection_error_history.resize(0);
}


void NeuronsSelection::delete_training_error_history()
{
    training_error_history.resize(0);
}


void NeuronsSelection::check() const
{
    // Optimization algorithm

    if(!training_strategy)
        throw runtime_error("Pointer to training strategy is nullptr.\n");

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();

    if(!loss_index)
        throw runtime_error("Pointer to loss index is nullptr.\n");

    // Neural network

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    if(!neural_network)
        throw runtime_error("Pointer to neural network is nullptr.\n");

    if(neural_network->get_layers_number() == 1)
        throw runtime_error("Number of layers in neural network must be greater than 1.\n");

    // Data set

    const DataSet* data_set = loss_index->get_data_set();

    if(!data_set)
        throw runtime_error("Pointer to data set is nullptr.\n");

    const Index selection_samples_number = data_set->get_samples_number(DataSet::SampleUse::Selection);

    if(selection_samples_number == 0)
        throw runtime_error("Number of selection samples is zero.\n");
}


string NeuronsSelection::write_time(const type& time) const
{
    const int total_seconds = static_cast<int>(time);
    const int hours = total_seconds / 3600;
    const int minutes = (total_seconds % 3600) / 60;
    const int seconds = total_seconds % 60;

    ostringstream elapsed_time;
    elapsed_time << std::setfill('0')
                 << std::setw(2) << hours << ":"
                 << std::setw(2) << minutes << ":"
                 << std::setw(2) << seconds;

    return elapsed_time.str();
}


NeuronsSelectionResults::NeuronsSelectionResults(const Index& maximum_epochs_number)
{
    neurons_number_history.resize(maximum_epochs_number);
    neurons_number_history.setZero();

    training_error_history.resize(maximum_epochs_number);
    training_error_history.setConstant(type(-1));

    selection_error_history.resize(maximum_epochs_number);
    selection_error_history.setConstant(type(-1));

    optimum_training_error = numeric_limits<type>::max();
    optimum_selection_error = numeric_limits<type>::max();
}


void NeuronsSelectionResults::resize_history(const Index& new_size)
{
    const Index old_size = neurons_number_history.size();

    const Tensor<Index, 1> old_neurons_number_history(neurons_number_history);
    const Tensor<type, 1> old_training_error_history(training_error_history);
    const Tensor<type, 1> old_selection_error_history(selection_error_history);

    neurons_number_history.resize(new_size);
    training_error_history.resize(new_size);
    selection_error_history.resize(new_size);

    const Index copy_size = min(old_size, new_size);

    for(Index i = 0; i < copy_size; i++)
    {
        neurons_number_history(i) = old_neurons_number_history(i);
        training_error_history(i) = old_training_error_history(i);
        selection_error_history(i) = old_selection_error_history(i);
    }
}


string NeuronsSelectionResults::write_stopping_condition() const
{
    switch(stopping_condition)
    {
        case NeuronsSelection::StoppingCondition::MaximumTime:
            return "MaximumTime";

        case NeuronsSelection::StoppingCondition::SelectionErrorGoal:
            return "SelectionErrorGoal";

        case NeuronsSelection::StoppingCondition::MaximumEpochs:
            return "MaximumEpochs";

        case NeuronsSelection::StoppingCondition::MaximumSelectionFailures:
            return "MaximumSelectionFailures";

        case NeuronsSelection::StoppingCondition::MaximumNeurons:
            return "MaximumNeurons";

        default:
            return string();
    }
}


void NeuronsSelectionResults::print() const
{
    cout << endl
         << "Neurons Selection Results" << endl
         << "Optimal neurons number: " << optimal_neurons_number << endl
         << "Optimum training error: " << optimum_training_error << endl
         << "Optimum selection error: " << optimum_selection_error << endl;
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
