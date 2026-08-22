//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/model_selection/inputs_selection.h"

#include "opennn/dataset/dataset.h"
#include "opennn/neural_network/neural_network.h"

namespace opennn
{

InputsSelection::InputsSelection(TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);
}

void InputsSelection::configure_neural_network_inputs(NeuralNetwork* neural_network, Dataset* dataset, Index input_features_number) const
{
    dataset->resize_input_shape(input_features_number);
    neural_network->set_input_shape(dataset->get_input_shape());
    neural_network->set_input_variables(dataset->get_model_input_variables());

    neural_network->compile();
}

InputsSelectionResult::InputsSelectionResult(const Index maximum_epochs)
{
    set(maximum_epochs);
}

void InputsSelectionResult::set(const Index maximum_epochs)
{
    training_error_history = VectorR::Constant(maximum_epochs, QUIET_NAN);
    validation_error_history = VectorR::Constant(maximum_epochs, QUIET_NAN);
    mean_validation_error_history = VectorR::Constant(maximum_epochs, QUIET_NAN);
    mean_training_error_history = VectorR::Constant(maximum_epochs, QUIET_NAN);
}

void InputsSelectionResult::resize_history(const Index new_size)
{
    training_error_history.conservativeResize(new_size);
    validation_error_history.conservativeResize(new_size);
    mean_training_error_history.conservativeResize(new_size);
    mean_validation_error_history.conservativeResize(new_size);
}

void InputsSelectionResult::print() const
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
    save_json_file(file_name, *this);
}

void InputsSelection::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
