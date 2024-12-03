//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "model_selection.h"

namespace opennn
{

ModelSelection::ModelSelection(TrainingStrategy* new_training_strategy)
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


const ModelSelection::NeuronsSelectionMethod& ModelSelection::get_neurons_selection_method() const
{
    return neurons_selection_method;
}


const ModelSelection::InputsSelectionMethod& ModelSelection::get_inputs_selection_method() const
{
    return inputs_selection_method;
}


GrowingNeurons* ModelSelection::get_growing_neurons()
{
    return &growing_neurons;
}


GrowingInputs* ModelSelection::get_growing_inputs()
{
    return &growing_inputs;
}


GeneticAlgorithm* ModelSelection::get_genetic_algorithm()
{
    return &genetic_algorithm;
}


void ModelSelection::set_default()
{
    set_neurons_selection_method(NeuronsSelectionMethod::GROWING_NEURONS);

    set_inputs_selection_method(InputsSelectionMethod::GROWING_INPUTS);

    display = true;
}


void ModelSelection::set_display(const bool& new_display)
{
    display = new_display;

    // Neurons selection

    growing_neurons.set_display(new_display);

    // Inputs selection

    growing_inputs.set_display(new_display);

    genetic_algorithm.set_display(new_display);

}


void ModelSelection::set_neurons_selection_method(const ModelSelection::NeuronsSelectionMethod& new_neurons_selection_method)
{
    neurons_selection_method = new_neurons_selection_method;
}


void ModelSelection::set_neurons_selection_method(const string& new_neurons_selection_method)
{
    if(new_neurons_selection_method == "GROWING_NEURONS")
        set_neurons_selection_method(NeuronsSelectionMethod::GROWING_NEURONS);
    else
        throw runtime_error("Unknown neurons selection type: " + new_neurons_selection_method + ".\n");
}


void ModelSelection::set_inputs_selection_method(const ModelSelection::InputsSelectionMethod& new_inputs_selection_method)
{
    inputs_selection_method = new_inputs_selection_method;
}


void ModelSelection::set_inputs_selection_method(const string& new_inputs_selection_method)
{
    if(new_inputs_selection_method == "GROWING_INPUTS")
        set_inputs_selection_method(InputsSelectionMethod::GROWING_INPUTS);
    else if(new_inputs_selection_method == "GENETIC_ALGORITHM")
        set_inputs_selection_method(InputsSelectionMethod::GENETIC_ALGORITHM);
    else
        throw runtime_error("Unknown inputs selection type: " + new_inputs_selection_method + ".\n");
}


void ModelSelection::set(TrainingStrategy* new_training_strategy)
{
    training_strategy = new_training_strategy;

    // Neurons selection

    growing_neurons.set_training_strategy(new_training_strategy);

    // Inputs selection

    growing_inputs.set(new_training_strategy);
    genetic_algorithm.set(new_training_strategy);

    set_default();
}


void ModelSelection::check() const
{
    // Optimization algorithm

    if(!training_strategy)
        throw runtime_error("Pointer to training strategy is nullptr.\n");

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();

    if(!loss_index)
        throw runtime_error("Pointer to loss index is nullptr.\n");

    // Neural network

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    if(!neural_network)
        throw runtime_error("Pointer to neural network is nullptr.\n");

    if(neural_network->is_empty())
        throw runtime_error("Multilayer Perceptron is empty.\n");

    // Data set

    const DataSet* data_set = loss_index->get_data_set();

    if(!data_set)
        throw runtime_error("Pointer to data set is nullptr.\n");

    const Index selection_samples_number = data_set->get_samples_number(DataSet::SampleUse::Selection);

    if(selection_samples_number == 0)
        throw runtime_error("Number of selection samples is zero.\n");
}


NeuronsSelectionResults ModelSelection::perform_neurons_selection()
{
    if(neurons_selection_method == NeuronsSelectionMethod::GROWING_NEURONS)
        return growing_neurons.perform_neurons_selection();
    else
        return NeuronsSelectionResults();
}


InputsSelectionResults ModelSelection::perform_inputs_selection()
{
    switch(inputs_selection_method)
    {
    case InputsSelectionMethod::GROWING_INPUTS:
        return growing_inputs.perform_inputs_selection();

    case InputsSelectionMethod::GENETIC_ALGORITHM:
        return genetic_algorithm.perform_inputs_selection();
    }

    return InputsSelectionResults();
}


void ModelSelection::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("ModelSelection");

    printer.OpenElement("NeuronsSelection");
    add_xml_element(printer, "NeuronsSelectionMethod", write_neurons_selection_method());
    growing_neurons.to_XML(printer);
    printer.CloseElement();  

    printer.OpenElement("InputsSelection");
    add_xml_element(printer, "InputsSelectionMethod", write_inputs_selection_method());
    growing_inputs.to_XML(printer);
    genetic_algorithm.to_XML(printer);
    printer.CloseElement(); 

    printer.CloseElement();
}


void ModelSelection::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("ModelSelection");
    
    if (!root_element) 
        throw runtime_error("Model Selection element is nullptr.\n");
    
    const XMLElement* neurons_selection_element = root_element->FirstChildElement("NeuronsSelection");

    if (neurons_selection_element) 
    {
        set_neurons_selection_method(read_xml_string(neurons_selection_element, "NeuronsSelectionMethod"));

        const XMLElement* growing_neurons_element = neurons_selection_element->FirstChildElement("GrowingNeurons");
        if (growing_neurons_element) {
            XMLDocument growing_neurons_document;
            growing_neurons_document.InsertFirstChild(growing_neurons_element->DeepClone(&growing_neurons_document));
            growing_neurons.from_XML(growing_neurons_document);
        }
    }

    const XMLElement* inputs_selection_element = root_element->FirstChildElement("InputsSelection");
    if (inputs_selection_element) {
        set_inputs_selection_method(read_xml_string(inputs_selection_element, "InputsSelectionMethod"));

        const XMLElement* growing_inputs_element = inputs_selection_element->FirstChildElement("GrowingInputs");
        if (growing_inputs_element) {
            XMLDocument growing_inputs_document;
            growing_inputs_document.InsertFirstChild(growing_inputs_element->DeepClone(&growing_inputs_document));
            growing_inputs.from_XML(growing_inputs_document);
        }

        const XMLElement* genetic_algorithm_element = inputs_selection_element->FirstChildElement("GeneticAlgorithm");
        if (genetic_algorithm_element) {
            XMLDocument genetic_algorithm_document;
            genetic_algorithm_document.InsertFirstChild(genetic_algorithm_element->DeepClone(&genetic_algorithm_document));
            genetic_algorithm.from_XML(genetic_algorithm_document);
        }
    }
}


string ModelSelection::write_neurons_selection_method() const
{
    if(neurons_selection_method ==  NeuronsSelectionMethod::GROWING_NEURONS)
        return "GROWING_NEURONS";
    else
        return string();
}


string ModelSelection::write_inputs_selection_method() const
{
    switch(inputs_selection_method)
    {
    case InputsSelectionMethod::GROWING_INPUTS:
        return "GROWING_INPUTS";

    case InputsSelectionMethod::GENETIC_ALGORITHM:
        return "GENETIC_ALGORITHM";
    default:
        return string();
    }
}


void ModelSelection::print() const
{
//    cout << to_string();
}


void ModelSelection::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    XMLPrinter printer;
    to_XML(printer);
    file << printer.CStr();
}


void ModelSelection::load(const filesystem::path& file_name)
{
    XMLDocument document;

    if(document.LoadFile(file_name.u8string().c_str()))
        throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

    from_XML(document);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
