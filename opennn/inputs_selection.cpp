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


string InputsSelection::write_stopping_condition(const TrainingResults& results) const
{
    return results.write_stopping_condition();
}


void InputsSelection::check() const
{
    if(!training_strategy)
        throw runtime_error("Pointer to training strategy is nullptr.\n");

    // Loss index

    const Loss* loss = training_strategy->get_loss();

    // Neural network

    if(!loss->get_neural_network())
        throw runtime_error("Pointer to neural network is nullptr.\n");

    const NeuralNetwork* neural_network = loss->get_neural_network();

    if(neural_network->is_empty())
        throw runtime_error("Neural network is empty.\n");

    // Dataset

    if(!loss->get_dataset())
        throw runtime_error("Pointer to dataset is nullptr.\n");

    const Dataset* dataset = loss->get_dataset();

    const Index validation_samples_number = dataset->get_samples_number("Validation");

    if(validation_samples_number == 0)
        throw runtime_error("Number of selection samples is zero.\n");
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
    training_error_history.resize(maximum_epochs);
    training_error_history.setConstant(type(-1));

    validation_error_history.resize(maximum_epochs);
    validation_error_history.setConstant(type(-1));

    mean_validation_error_history.resize(maximum_epochs);
    mean_validation_error_history.setConstant(type(-1));

    mean_training_error_history.resize(maximum_epochs);
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

    case InputsSelection::StoppingCondition::MaximumEpochs:
        return "MaximumEpochs";

    case InputsSelection::StoppingCondition::MaximumSelectionFailures:
        return "MaximumSelectionFailures";

    default:
        return string();
    }
}


void InputsSelectionResults::resize_history(const Index new_size)
{
    const VectorR old_training_error_history(training_error_history);
    const VectorR old_validation_error_history(validation_error_history);

    const VectorR old_mean_selection_history(mean_validation_error_history);
    const VectorR old_mean_training_history(mean_training_error_history);

    training_error_history.resize(new_size);
    validation_error_history.resize(new_size);
    mean_training_error_history.resize(new_size);
    mean_validation_error_history.resize(new_size);

    for(Index i = 0; i < new_size; i++)
    {
        training_error_history(i) = old_training_error_history(i);
        validation_error_history(i) = old_validation_error_history(i);
        mean_training_error_history(i) = old_mean_training_history(i);
        mean_validation_error_history(i) = old_mean_selection_history(i);
    }
}


void InputsSelectionResults::print() const
{
    cout << endl
         << "Input Validation Results" << endl
         << "Optimal inputs number: " << optimal_input_variable_names.size() << endl
         << "Inputs: " << endl;

    for(size_t i = 0; i < optimal_input_variable_names.size(); i++)
        cout << "   " << optimal_input_variable_names[i] << endl;

    cout << "Optimum training error: " << optimum_training_error << endl
         << "Optimum selection error: " << optimum_validation_error << endl;
}


void InputsSelection::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        return;

    XMLPrinter printer;
    to_XML(printer);
    file << printer.CStr();
}


void InputsSelection::load(const filesystem::path& file_name)
{
    from_XML(load_xml_file(file_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
