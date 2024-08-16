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

/// Default constructor.

ModelSelection::ModelSelection()
{
    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy Pointer to a training strategy object.

ModelSelection::ModelSelection(TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);

    set_default();
}


/// Returns a pointer to the training strategy object.

TrainingStrategy* ModelSelection::get_training_strategy() const
{
#ifdef OPENNN_DEBUG

    if(!training_strategy)
        throw runtime_error("Training strategy pointer is nullptr.\n");

#endif

    return training_strategy;
}


/// Returns true if this model selection has a training strategy associated,
/// and false otherwise.

bool ModelSelection::has_training_strategy() const
{   
    if(training_strategy)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns the type of algorithm for the neurons selection.

const ModelSelection::NeuronsSelectionMethod& ModelSelection::get_neurons_selection_method() const
{
    return neurons_selection_method;
}


/// Returns the type of algorithm for the inputs selection.

const ModelSelection::InputsSelectionMethod& ModelSelection::get_inputs_selection_method() const
{
    return inputs_selection_method;
}


/// Returns a pointer to the growing neurons selection algorithm.

GrowingNeurons* ModelSelection::get_growing_neurons()
{
    return &growing_neurons;
}


/// Returns a pointer to the growing inputs selection algorithm.

GrowingInputs* ModelSelection::get_growing_inputs()
{
    return &growing_inputs;
}


/// Returns a pointer to the genetic inputs selection algorithm.

GeneticAlgorithm* ModelSelection::get_genetic_algorithm()
{
    return &genetic_algorithm;
}


/// Sets the members of the model selection object to their default values.

void ModelSelection::set_default()
{
    set_neurons_selection_method(NeuronsSelectionMethod::GROWING_NEURONS);

    set_inputs_selection_method(InputsSelectionMethod::GROWING_INPUTS);

    display = true;
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void ModelSelection::set_display(const bool& new_display)
{
    display = new_display;

    // Neurons selection

    growing_neurons.set_display(new_display);

    // Inputs selection

    growing_inputs.set_display(new_display);

    genetic_algorithm.set_display(new_display);

}


/// Sets a new method for selecting the order which have more impact on the targets.
/// @param new_neurons_selection_method Method for selecting the order(NO_NEURONS_SELECTION, growing_neurons, GOLDEN_SECTION, SIMULATED_ANNEALING).

void ModelSelection::set_neurons_selection_method(const ModelSelection::NeuronsSelectionMethod& new_neurons_selection_method)
{
    neurons_selection_method = new_neurons_selection_method;
}


/// Sets a new neurons selection algorithm from a string.
/// @param new_neurons_selection_method String with the neurons selection type.

void ModelSelection::set_neurons_selection_method(const string& new_neurons_selection_method)
{
    if(new_neurons_selection_method == "GROWING_NEURONS")
    {
        set_neurons_selection_method(NeuronsSelectionMethod::GROWING_NEURONS);
    }
    else
    {
        throw runtime_error("Unknown neurons selection type: " + new_neurons_selection_method + ".\n");
    }
}


/// Sets a new method for selecting the inputs which have more impact on the targets.
/// @param new_inputs_selection_method Method for selecting the inputs(GROWING_INPUTS, GENETIC_ALGORITHM).

void ModelSelection::set_inputs_selection_method(const ModelSelection::InputsSelectionMethod& new_inputs_selection_method)
{
    inputs_selection_method = new_inputs_selection_method;
}


/// Sets a new inputs selection algorithm from a string.
/// @param new_inputs_selection_method String with the inputs selection type.

void ModelSelection::set_inputs_selection_method(const string& new_inputs_selection_method)
{
    if(new_inputs_selection_method == "GROWING_INPUTS")
    {
        set_inputs_selection_method(InputsSelectionMethod::GROWING_INPUTS);
    }
    else if(new_inputs_selection_method == "GENETIC_ALGORITHM")
    {
        set_inputs_selection_method(InputsSelectionMethod::GENETIC_ALGORITHM);
    }
    else
    {
        throw runtime_error("Unknown inputs selection type: " + new_inputs_selection_method + ".\n");
    }
}


/// Sets a new training strategy pointer.
/// @param new_training_strategy Pointer to a training strategy object.

void ModelSelection::set(TrainingStrategy* new_training_strategy)
{
    training_strategy = new_training_strategy;

    // Neurons selection

    growing_neurons.set_training_strategy(new_training_strategy);

    // Inputs selection

    growing_inputs.set(new_training_strategy);
    genetic_algorithm.set(new_training_strategy);
}


/// Checks that the different pointers needed for performing the model selection are not nullptr.

void ModelSelection::check() const
{

    // Optimization algorithm

    ostringstream buffer;

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

    const Index selection_samples_number = data_set->get_selection_samples_number();

    if(selection_samples_number == 0)
        throw runtime_error("Number of selection samples is zero.\n");
}


/// Perform the neurons selection, returns a structure with the results of the neurons selection.
/// It also set the neural network of the training strategy pointer with the optimum parameters.

NeuronsSelectionResults ModelSelection::perform_neurons_selection()
{
    if(neurons_selection_method == NeuronsSelectionMethod::GROWING_NEURONS)
    {
        return growing_neurons.perform_neurons_selection();
    }
    else
    {
        return NeuronsSelectionResults();
    }
}


/// Perform the inputs selection, returns a structure with the results of the inputs selection.
/// It also set the neural network of the training strategy pointer with the optimum parameters.

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


/// Serializes the model selection object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void ModelSelection::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Model selection

    file_stream.OpenElement("ModelSelection");

    // Neurons selection

    file_stream.OpenElement("NeuronsSelection");

    file_stream.OpenElement("NeuronsSelectionMethod");
    file_stream.PushText(write_neurons_selection_method().c_str());
    file_stream.CloseElement();

    growing_neurons.write_XML(file_stream);

    file_stream.CloseElement();

    // Inputs selection

    file_stream.OpenElement("InputsSelection");

    file_stream.OpenElement("InputsSelectionMethod");
    file_stream.PushText(write_inputs_selection_method().c_str());
    file_stream.CloseElement();

    growing_inputs.write_XML(file_stream);
    genetic_algorithm.write_XML(file_stream);

    file_stream.CloseElement();

    // Model selection (end tag)

    file_stream.CloseElement();
}


/// Loads the members of this model selection object from an XML document.
/// @param document XML document of the TinyXML library.

void ModelSelection::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("ModelSelection");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Model Selection element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Neurons Selection

    const tinyxml2::XMLElement* neurons_selection_element = root_element->FirstChildElement("NeuronsSelection");

    if(neurons_selection_element)
    {
        // Neurons selection method

        const tinyxml2::XMLElement* neurons_selection_method_element = neurons_selection_element->FirstChildElement("NeuronsSelectionMethod");

        set_neurons_selection_method(neurons_selection_method_element->GetText());

        // Growing neurons

        const tinyxml2::XMLElement* growing_neurons_element = neurons_selection_element->FirstChildElement("GrowingNeurons");

        if(growing_neurons_element)
        {
            tinyxml2::XMLDocument growing_neurons_document;

            tinyxml2::XMLElement* growing_neurons_element_copy = growing_neurons_document.NewElement("GrowingNeurons");

            for(const tinyxml2::XMLNode* nodeFor=growing_neurons_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
            {
                tinyxml2::XMLNode* copy = nodeFor->DeepClone(&growing_neurons_document );
                growing_neurons_element_copy->InsertEndChild(copy );
            }

            growing_neurons_document.InsertEndChild(growing_neurons_element_copy);

            growing_neurons.from_XML(growing_neurons_document);
        }
    }

    // Inputs Selection
    {
        const tinyxml2::XMLElement* inputs_selection_element = root_element->FirstChildElement("InputsSelection");

        if(inputs_selection_element)
        {
            const tinyxml2::XMLElement* inputs_selection_method_element = inputs_selection_element->FirstChildElement("InputsSelectionMethod");

            set_inputs_selection_method(inputs_selection_method_element->GetText());

            // Growing inputs

            const tinyxml2::XMLElement* growing_inputs_element = inputs_selection_element->FirstChildElement("GrowingInputs");

            if(growing_inputs_element)
            {
                tinyxml2::XMLDocument growing_inputs_document;

                tinyxml2::XMLElement* growing_inputs_element_copy = growing_inputs_document.NewElement("GrowingInputs");

                for(const tinyxml2::XMLNode* nodeFor=growing_inputs_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&growing_inputs_document );
                    growing_inputs_element_copy->InsertEndChild(copy );
                }

                growing_inputs_document.InsertEndChild(growing_inputs_element_copy);

                growing_inputs.from_XML(growing_inputs_document);
            }


            // Genetic algorithm

            const tinyxml2::XMLElement* genetic_algorithm_element = inputs_selection_element->FirstChildElement("GeneticAlgorithm");

            if(genetic_algorithm_element)
            {
                tinyxml2::XMLDocument genetic_algorithm_document;

                tinyxml2::XMLElement* genetic_algorithm_element_copy = genetic_algorithm_document.NewElement("GeneticAlgorithm");

                for(const tinyxml2::XMLNode* nodeFor=genetic_algorithm_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&genetic_algorithm_document );
                    genetic_algorithm_element_copy->InsertEndChild(copy );
                }

                genetic_algorithm_document.InsertEndChild(genetic_algorithm_element_copy);

                genetic_algorithm.from_XML(genetic_algorithm_document);
            }
        }
    }
}


string ModelSelection::write_neurons_selection_method() const
{
    if(neurons_selection_method ==  NeuronsSelectionMethod::GROWING_NEURONS)
    {
        return "GROWING_NEURONS";
    }
    else
    {
        return string();
    }
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


/// Prints to the screen the XML representation of this model selection object.

void ModelSelection::print() const
{
//    cout << to_string();
}


/// Saves the model selection members to an XML file.
/// @param file_name Name of model selection XML file.

void ModelSelection::save(const string& file_name) const
{
    FILE * file = fopen(file_name.c_str(), "w");

    if(file)
    {
        tinyxml2::XMLPrinter printer(file);
        write_XML(printer);
        fclose(file);
    }
}


/// Loads the model selection members from an XML file.
/// @param file_name Name of model selection XML file.

void ModelSelection::load(const string& file_name)
{
    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw runtime_error(buffer.str());
    }

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
