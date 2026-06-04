//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "dataset.h"
#include "optimizer.h"
#include "training_strategy.h"
#include "inputs_selection.h"

namespace opennn
{

InputsSelection::InputsSelection(TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);
}

void InputsSelection::check() const
{
    throw_if(!training_strategy, "training strategy is not set.");

    // Loss

    const Loss* loss = training_strategy->get_loss();

    throw_if(!loss, "loss is not set.");

    // Neural network

    const NeuralNetwork* neural_network = loss->get_neural_network();

    throw_if(!neural_network, "neural network is not set.");

    throw_if(neural_network->is_empty(), "Neural network is empty.\n");

    // Dataset

    const Dataset* dataset = loss->get_dataset();

    throw_if(!dataset, "dataset is not set.");

    const Index validation_samples_number = dataset->get_samples_number("Validation");

    throw_if(validation_samples_number == 0, "Number of validation samples is zero.\n");
}

InputsSelectionResults::InputsSelectionResults(const Index maximum_epochs)
{
    set(maximum_epochs);
}

Index InputsSelectionResults::get_epochs_number() const
{
    return training_error_history.size();
}

void InputsSelectionResults::set(const Index maximum_epochs)
{
    training_error_history = VectorR::Constant(maximum_epochs, -1.0f);
    validation_error_history = VectorR::Constant(maximum_epochs, -1.0f);
    mean_validation_error_history = VectorR::Constant(maximum_epochs, -1.0f);
    mean_training_error_history = VectorR::Constant(maximum_epochs, -1.0f);
}

string InputsSelectionResults::write_stopping_condition() const
{
    using enum InputsSelection::StoppingCondition;
    switch (stopping_condition)
    {
    case MaximumTime:
        return "MaximumTime";

    case ValidationErrorGoal:
        return "ValidationErrorGoal";

    case MaximumInputs:
        return "MaximumInputs";

    case MaximumEpochs:
        return "MaximumEpochs";

    case MaximumValidationFailures:
        return "MaximumValidationFailures";

    default:
        return {};
    }
}

void InputsSelectionResults::resize_history(const Index new_size)
{
    training_error_history.conservativeResize(new_size);
    validation_error_history.conservativeResize(new_size);
    mean_training_error_history.conservativeResize(new_size);
    mean_validation_error_history.conservativeResize(new_size);
}

void InputsSelectionResults::print() const
{
    cout << "\n"
         << "Input Selection Results" << "\n"
         << "Optimal inputs number: " << optimal_input_variable_names.size() << "\n"
         << "Inputs: " << "\n";

    for (const string& name : optimal_input_variable_names)
        cout << "   " << name << "\n";

    cout << "Optimum training error: " << optimum_training_error << "\n"
         << "Optimum validation error: " << optimum_validation_error << "\n";
}

void InputsSelection::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    throw_if(!file.is_open(), format("Cannot open file: {}", file_name.string()));

    JsonWriter printer;
    to_JSON(printer);
    file << printer.c_str();
}

void InputsSelection::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
