//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/model_selection/model_selection.h"

#include "opennn/registry.h"

namespace opennn
{

ModelSelection::ModelSelection(Training* new_training)
{
    set(new_training);

    set_default();
}

void ModelSelection::set(Training* new_training)
{
    training = new_training;
    neurons_selection.set_training(new_training);
    if (input_selection) input_selection->set(new_training);
}

void ModelSelection::set_default()
{
    neurons_selection.set(training);

    set_input_selection("GrowingInputs");
}

void ModelSelection::set_input_selection(const string& new_input_selection)
{
    input_selection = create_input_selection(new_input_selection);

    input_selection->set(training);
}

void ModelSelection::to_JSON(JsonWriter& printer) const
{
    printer.open_element("ModelSelection");

    printer.open_element("NeuronSelection");

    add_json_field(printer, "NeuronsSelectionMethod", neurons_selection.get_name());

    neurons_selection.to_JSON(printer);

    printer.close_element();

    printer.open_element("InputSelection");

    add_json_field(printer, "InputSelectionMethod", input_selection->get_name());

    input_selection->to_JSON(printer);

    printer.close_element();

    printer.close_element();
}

void ModelSelection::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "ModelSelection");

    const Json* neurons_selection_element = require_json_field(root_element, "NeuronSelection");

    const string selection_method = read_json_string(neurons_selection_element, "NeuronsSelectionMethod");

    const Json* neurons_selection_method_element = neurons_selection_element->find(selection_method.c_str());

    throw_if(!neurons_selection_method_element,
             "{} element is nullptr.\n", selection_method);

    neurons_selection.set(training);
    neurons_selection.from_JSON(JsonDocument::wrap(selection_method, *neurons_selection_method_element));

    const Json* input_selection_element = require_json_field(root_element, "InputSelection");

    const string inputs_method = read_json_string(input_selection_element, "InputSelectionMethod");

    const Json* input_selection_method_element = input_selection_element->find(inputs_method.c_str());

    throw_if(!input_selection_method_element,
             "{} element is nullptr.\n", inputs_method);

    set_input_selection(inputs_method);
    input_selection->from_JSON(JsonDocument::wrap(inputs_method, *input_selection_method_element));
}

void ModelSelection::save(const filesystem::path& file_name) const
{
    save_json_file(file_name, *this);
}

void ModelSelection::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
