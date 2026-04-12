//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "loss.h"
#include "model_selection.h"
#include "training_strategy.h"

namespace opennn
{

ModelSelection::ModelSelection(TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);
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

void ModelSelection::check() const
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

    if(neural_network->is_empty())
        throw runtime_error("Multilayer Dense is empty.\n");

    // Dataset

    const Dataset* dataset = loss->get_dataset();

    if(!dataset)
        throw runtime_error("Pointer to dataset is nullptr.\n");

    const Index validation_samples_number = dataset->get_samples_number("Validation");

    if(validation_samples_number == 0)
        throw runtime_error("Number of selection samples is zero.\n");
}

NeuronsSelectionResults ModelSelection::perform_neurons_selection()
{
    return neurons_selection->perform_neurons_selection();
}

InputsSelectionResults ModelSelection::perform_input_selection()
{
    return inputs_selection->perform_input_selection();
}

void ModelSelection::to_XML(XmlPrinter& printer) const
{
    printer.open_element("ModelSelection");

    printer.open_element("NeuronSelection");

    add_xml_element(printer, "NeuronsSelectionMethod", neurons_selection->get_name());

    neurons_selection->to_XML(printer);

    printer.close_element();

    printer.open_element("InputsSelection");

    add_xml_element(printer, "InputsSelectionMethod", inputs_selection->get_name());

    inputs_selection->to_XML(printer);

    printer.close_element();

    printer.close_element();
}

void ModelSelection::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "ModelSelection");

    // Neuron selection

    const XmlElement* neurons_selection_element = require_xml_element(root_element, "NeuronSelection");

    const string selection_method = read_xml_string(neurons_selection_element, "NeuronsSelectionMethod");
    
    const XmlElement* neurons_selection_method_element = neurons_selection_element->first_child_element(selection_method.c_str());

    if (neurons_selection_method_element)
    {
        set_neurons_selection(selection_method);

        XmlDocument selection_method_document;
        XmlNode* neurons_selection_method_element_copy = neurons_selection_method_element->deep_clone(&selection_method_document);
        selection_method_document.insert_end_child(neurons_selection_method_element_copy);

        neurons_selection->from_XML(selection_method_document);
    }
    else throw runtime_error(selection_method + " element is nullptr.\n");

    // Input Validation

    const XmlElement* inputs_selection_element = require_xml_element(root_element, "InputsSelection");

    const string inputs_method = read_xml_string(inputs_selection_element, "InputsSelectionMethod");

    const XmlElement* inputs_selection_method_element = inputs_selection_element->first_child_element(inputs_method.c_str());

    if (inputs_selection_method_element)
    {
        set_inputs_selection(inputs_method);

        XmlDocument inputs_method_document;
        XmlNode* inputs_selection_method_element_copy = inputs_selection_method_element->deep_clone(&inputs_method_document);
        inputs_method_document.insert_end_child(inputs_selection_method_element_copy);

        inputs_selection->from_XML(inputs_method_document);
    }
    else throw runtime_error(inputs_method + " element is nullptr.\n");
}

void ModelSelection::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open file: " + file_name.string());

    XmlPrinter printer;
    to_XML(printer);
    file << printer.c_str();
}

void ModelSelection::load(const filesystem::path& file_name)
{
    from_XML(load_xml_file(file_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
