//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "model_selection.h"

namespace opennn
{

ModelSelection::ModelSelection(TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);

    set_default();
}

void ModelSelection::set_default()
{
    set_neurons_selection("GrowingNeurons");

    set_inputs_selection("GrowingInputs");
}

void ModelSelection::set_neurons_selection(const string& new_neurons_selection)
{
    neurons_selection = Registry<NeuronSelection>::instance().create(new_neurons_selection);

    neurons_selection->set(training_strategy);
}

void ModelSelection::set_inputs_selection(const string& new_inputs_selection)
{
    inputs_selection = Registry<InputsSelection>::instance().create(new_inputs_selection);

    inputs_selection->set(training_strategy);
}

NeuronsSelectionResult ModelSelection::perform_neurons_selection()
{
    return neurons_selection->perform_neurons_selection();
}

InputsSelectionResult ModelSelection::perform_input_selection()
{
    return inputs_selection->perform_input_selection();
}

void ModelSelection::to_JSON(JsonWriter& printer) const
{
    printer.open_element("ModelSelection");

    printer.open_element("NeuronSelection");

    add_json_field(printer, "NeuronsSelectionMethod", neurons_selection->get_name());

    neurons_selection->to_JSON(printer);

    printer.close_element();

    printer.open_element("InputsSelection");

    add_json_field(printer, "InputsSelectionMethod", inputs_selection->get_name());

    inputs_selection->to_JSON(printer);

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
             format("{} element is nullptr.\n", selection_method));

    set_neurons_selection(selection_method);
    neurons_selection->from_JSON(JsonDocument::wrap(selection_method, *neurons_selection_method_element));


    const Json* inputs_selection_element = require_json_field(root_element, "InputsSelection");

    const string inputs_method = read_json_string(inputs_selection_element, "InputsSelectionMethod");

    const Json* inputs_selection_method_element = inputs_selection_element->find(inputs_method.c_str());

    throw_if(!inputs_selection_method_element,
             format("{} element is nullptr.\n", inputs_method));

    set_inputs_selection(inputs_method);
    inputs_selection->from_JSON(JsonDocument::wrap(inputs_method, *inputs_selection_method_element));
}

void ModelSelection::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    throw_if(!file.is_open(), format("Cannot open file: {}", file_name.string()));

    JsonWriter printer;
    to_JSON(printer);
    file << printer.c_str();
}

void ModelSelection::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
