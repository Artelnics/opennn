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

ModelSelection::ModelSelection(const TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);
}


TrainingStrategy* ModelSelection::get_training_strategy() const
{
    return training_strategy;
}


bool ModelSelection::has_training_strategy() const
{
    return training_strategy;
}


NeuronSelection* ModelSelection::get_neurons_selection() const
{
    return neurons_selection.get();
}


InputsSelection* ModelSelection::get_inputs_selection() const
{
    return inputs_selection.get();
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


void ModelSelection::set(const TrainingStrategy* new_training_strategy)
{
    training_strategy = const_cast<TrainingStrategy*>(new_training_strategy);
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


void ModelSelection::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("ModelSelection");

    printer.OpenElement("NeuronSelection");

    add_xml_element(printer, "NeuronsSelectionMethod", neurons_selection->get_name());

    neurons_selection->to_XML(printer);

    printer.CloseElement();

    printer.OpenElement("InputsSelection");

    add_xml_element(printer, "InputsSelectionMethod", inputs_selection->get_name());

    inputs_selection->to_XML(printer);

    printer.CloseElement();

    printer.CloseElement();
}


void ModelSelection::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = get_xml_root(document, "ModelSelection");

    // Neuron selection

    const XMLElement* neurons_selection_element = root_element->FirstChildElement("NeuronSelection");
    if(!neurons_selection_element) throw runtime_error("Neuron selection element is nullptr.\n");

    const string selection_method = read_xml_string(neurons_selection_element, "NeuronsSelectionMethod");
    
    const XMLElement* neurons_selection_method_element = neurons_selection_element->FirstChildElement(selection_method.c_str());

    if (neurons_selection_method_element)
    {
        set_neurons_selection(selection_method);

        XMLDocument selection_method_document;
        XMLNode* neurons_selection_method_element_copy = neurons_selection_method_element->DeepClone(&selection_method_document);
        selection_method_document.InsertEndChild(neurons_selection_method_element_copy);

        neurons_selection->from_XML(selection_method_document);
    }
    else throw runtime_error(selection_method + " element is nullptr.\n");

    // Input Validation

    const XMLElement* inputs_selection_element = root_element->FirstChildElement("InputsSelection");
    if(!inputs_selection_element) throw runtime_error("Input selection element is nullptr.\n");

    const string inputs_method = read_xml_string(inputs_selection_element, "InputsSelectionMethod");

    const XMLElement* inputs_selection_method_element = inputs_selection_element->FirstChildElement(inputs_method.c_str());

    if (inputs_selection_method_element)
    {
        set_inputs_selection(inputs_method);

        XMLDocument inputs_method_document;
        XMLNode* inputs_selection_method_element_copy = inputs_selection_method_element->DeepClone(&inputs_method_document);
        inputs_method_document.InsertEndChild(inputs_selection_method_element_copy);

        inputs_selection->from_XML(inputs_method_document);
    }
    else throw runtime_error(inputs_method + " element is nullptr.\n");
}


void ModelSelection::print() const
{
    cout << get_neurons_selection() << endl;
    cout << get_inputs_selection() << endl;
}


void ModelSelection::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        return;

    XMLPrinter printer;
    to_XML(printer);
    file << printer.CStr();
}


void ModelSelection::load(const filesystem::path& file_name)
{
    from_XML(load_xml_file(file_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
