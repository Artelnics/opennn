//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "neural_network.h"
#include "training_strategy.h"
#include "neuron_selection.h"

namespace opennn
{

NeuronSelection::NeuronSelection(TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);
}

void NeuronSelection::set(TrainingStrategy* new_training_strategy)
{
    training_strategy = new_training_strategy;

    set_default();
}

void NeuronSelection::set_default()
{
    if (!training_strategy || !training_strategy->get_neural_network())
        return;

    const NeuralNetwork* neural_network = training_strategy->get_neural_network();

    minimum_neurons = 1;
    maximum_neurons = 2 * (neural_network->get_inputs_number() + neural_network->get_outputs_number());
    trials_number = 1;
    display = true;

    validation_error_goal = 0.0f;
    maximum_epochs = 1000;
    maximum_time = 3600.0f;
}

void NeuronSelection::save(const filesystem::path& file_name) const
{
    save_json_file(file_name, *this);
}

void NeuronSelection::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
}

NeuronsSelectionResult::NeuronsSelectionResult(const Index maximum_epochs)
{
    neurons_number_history = VectorI::Zero(maximum_epochs);

    training_error_history = VectorR::Constant(maximum_epochs, -1.0f);
    validation_error_history = VectorR::Constant(maximum_epochs, -1.0f);

    optimum_training_error = MAX;
    optimum_validation_error = MAX;
}

void NeuronsSelectionResult::resize_history(const Index new_size)
{
    neurons_number_history.conservativeResize(new_size);
    training_error_history.conservativeResize(new_size);
    validation_error_history.conservativeResize(new_size);
}

void NeuronsSelectionResult::print() const
{
    cout << "\n"
         << "Neuron Selection Results" << "\n"
         << "Optimal neurons number: " << optimal_neurons_number << "\n"
         << "Optimum training error: " << optimum_training_error << "\n"
         << "Optimum validation error: " << optimum_validation_error << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
