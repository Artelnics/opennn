//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/model_selection/input_selection.h"

#include "opennn/dataset/dataset.h"
#include "opennn/model_selection/selection_utilities.h"
#include "opennn/network/network.h"

namespace opennn
{

InputSelection::InputSelection(Training* new_training)
{
    set(new_training);
}

void InputSelection::configure_network_inputs(Network* network, Dataset* dataset, Index input_features_number) const
{
    dataset->resize_input_shape(input_features_number);
    network->set_input_shape(dataset->get_input_shape());
    network->set_input_variables(dataset->get_model_input_variables());

    network->compile();
}

void InputSelection::install_optimal_inputs(Network* network,
                                             Dataset* dataset,
                                             const vector<Index>& optimal_input_indices,
                                             const vector<Index>& target_indices,
                                             const vector<Index>& time_indices) const
{
    dataset->set_variable_indices(optimal_input_indices, target_indices);

    if (time_indices.size() == 1)
        dataset->set_variable_role(time_indices[0], "Time");

    configure_network_inputs(network, dataset,
                                    dataset->get_features_number(VariableRole::Input));

    apply_input_scaling(network, capture_input_scaling(dataset));
}

InputSelectionResult::InputSelectionResult(const Index maximum_epochs)
{
    set(maximum_epochs);
}

void InputSelectionResult::set(const Index maximum_epochs)
{
    training_error_history = VectorR::Constant(maximum_epochs, QUIET_NAN);
    validation_error_history = VectorR::Constant(maximum_epochs, QUIET_NAN);
    mean_validation_error_history = VectorR::Constant(maximum_epochs, QUIET_NAN);
    mean_training_error_history = VectorR::Constant(maximum_epochs, QUIET_NAN);
}

void InputSelectionResult::resize_history(const Index new_size)
{
    training_error_history.conservativeResize(new_size);
    validation_error_history.conservativeResize(new_size);
    mean_training_error_history.conservativeResize(new_size);
    mean_validation_error_history.conservativeResize(new_size);
}

void InputSelectionResult::print() const
{
    logging::info() << "\n"
         << "Input Selection Results" << "\n"
         << "Optimal inputs number: " << optimal_input_variable_names.size() << "\n"
         << "Inputs: " << "\n";

    for (const string& name : optimal_input_variable_names)
        logging::info() << "   " << name << "\n";

    logging::info() << "Optimum training error: " << optimum_training_error << "\n"
         << "Optimum validation error: " << optimum_validation_error << "\n";
}

void InputSelection::save(const filesystem::path& file_name) const
{
    save_json_file(file_name, *this);
}

void InputSelection::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
