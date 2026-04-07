//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "dataset.h"
#include "neural_network.h"
#include "optimizer.h"
#include "training_strategy.h"
#include "neuron_selection.h"

namespace opennn
{

NeuronSelection::NeuronSelection(TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);
}


TrainingStrategy* NeuronSelection::get_training_strategy() const
{
    return training_strategy;
}


bool NeuronSelection::has_training_strategy() const
{
    return training_strategy;
}


bool NeuronSelection::get_display() const
{
    return display;
}


void NeuronSelection::set(TrainingStrategy* new_training_strategy)
{
    training_strategy = new_training_strategy;

    set_default();
}


void NeuronSelection::set_training_strategy(TrainingStrategy* new_training_strategy)
{
    training_strategy = new_training_strategy;
}


void NeuronSelection::set_default()
{
    if(!(training_strategy && training_strategy->get_neural_network()))
        return;

    const Index inputs_number = training_strategy->get_neural_network()->get_inputs_number();
    const Index outputs_number = training_strategy->get_neural_network()->get_outputs_number();

    minimum_neurons = 1;
    maximum_neurons = 2 * (inputs_number + outputs_number);
    trials_number = 1;
    display = true;

    validation_error_goal = type(0);
    maximum_epochs = 1000;
    maximum_time = type(3600);
}


void NeuronSelection::set_maximum_neurons(const Index new_maximum_neurons)
{
    maximum_neurons = new_maximum_neurons;
}


void NeuronSelection::set_minimum_neurons(const Index new_minimum_neurons)
{
    minimum_neurons = new_minimum_neurons;
}


void NeuronSelection::set_trials_number(const Index new_trials_number)
{
    trials_number = new_trials_number;
}


void NeuronSelection::set_display(bool new_display)
{
    display = new_display;
}


void NeuronSelection::set_validation_error_goal(const type new_validation_error_goal)
{
    validation_error_goal = new_validation_error_goal;
}


void NeuronSelection::set_maximum_epochs(const Index new_maximum_epochs)
{
    maximum_epochs = new_maximum_epochs;
}


void NeuronSelection::set_maximum_validation_failures(const Index new_maximum_validation_failures)
{
    maximum_validation_failures = new_maximum_validation_failures;
}


void NeuronSelection::set_maximum_time(const type new_maximum_time)
{
    maximum_time = new_maximum_time;
}


string NeuronSelection::write_stopping_condition(const TrainingResults& results) const
{
    return results.write_stopping_condition();
}



void NeuronSelection::check() const
{
    // Optimization algorithm

    if(!training_strategy)
        throw runtime_error("Pointer to training strategy is nullptr.\n");

    // Loss index

    const Loss* loss = training_strategy->get_loss();

    if(!loss)
        throw runtime_error("Pointer to loss index is nullptr.\n");

    // Neural network

    const NeuralNetwork* neural_network = loss->get_neural_network();

    if(!neural_network)
        throw runtime_error("Pointer to neural network is nullptr.\n");

    if(neural_network->get_layers_number() == 1)
        throw runtime_error("Number of layers in neural network must be greater than 1.\n");

    // Dataset

    const Dataset* dataset = loss->get_dataset();

    if(!dataset)
        throw runtime_error("Pointer to dataset is nullptr.\n");

    const Index validation_samples_number = dataset->get_samples_number("Validation");

    if(validation_samples_number == 0)
        throw runtime_error("Number of selection samples is zero.\n");
}


void NeuronSelection::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        return;

    XMLPrinter printer;
    to_XML(printer);
    file << printer.CStr();
}


void NeuronSelection::load(const filesystem::path& file_name)
{
    from_XML(load_xml_file(file_name));
}


NeuronsSelectionResults::NeuronsSelectionResults(const Index maximum_epochs)
{
    neurons_number_history.resize(maximum_epochs);
    neurons_number_history.setZero();

    training_error_history.resize(maximum_epochs);
    training_error_history.setConstant(type(-1));

    validation_error_history.resize(maximum_epochs);
    validation_error_history.setConstant(type(-1));

    optimum_training_error = MAX;
    optimum_validation_error = MAX;
}


void NeuronsSelectionResults::resize_history(const Index new_size)
{
    const Index old_size = neurons_number_history.size();

    const VectorI old_neurons_number_history(neurons_number_history);
    const VectorR old_training_error_history(training_error_history);
    const VectorR old_validation_error_history(validation_error_history);

    neurons_number_history.resize(new_size);
    training_error_history.resize(new_size);
    validation_error_history.resize(new_size);

    const Index copy_size = min(old_size, new_size);

    for(Index i = 0; i < copy_size; i++)
    {
        neurons_number_history(i) = old_neurons_number_history(i);
        training_error_history(i) = old_training_error_history(i);
        validation_error_history(i) = old_validation_error_history(i);
    }
}


string NeuronsSelectionResults::write_stopping_condition() const
{
    switch(stopping_condition)
    {
        case NeuronSelection::StoppingCondition::MaximumTime:
            return "MaximumTime";

        case NeuronSelection::StoppingCondition::SelectionErrorGoal:
            return "SelectionErrorGoal";

        case NeuronSelection::StoppingCondition::MaximumEpochs:
            return "MaximumEpochs";

        case NeuronSelection::StoppingCondition::MaximumSelectionFailures:
            return "MaximumSelectionFailures";

        case NeuronSelection::StoppingCondition::MaximumNeurons:
            return "MaximumNeurons";

        default:
            return string();
    }
}


void NeuronsSelectionResults::print() const
{
    cout << endl
         << "Neuron Validation Results" << endl
         << "Optimal neurons number: " << optimal_neurons_number << endl
         << "Optimum training error: " << optimum_training_error << endl
         << "Optimum selection error: " << optimum_validation_error << endl;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
