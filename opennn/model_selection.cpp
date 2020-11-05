//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "model_selection.h"

namespace OpenNN
{

/// Default constructor.

ModelSelection::ModelSelection()
{
    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

ModelSelection::ModelSelection(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;

    set_default();
}


/// Destructor.

ModelSelection::~ModelSelection()
{
}


/// Returns a pointer to the training strategy object.

TrainingStrategy* ModelSelection::get_training_strategy_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!training_strategy_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "TrainingStrategy* get_training_strategy_pointer() const method.\n"
               << "Training strategy pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return training_strategy_pointer;
}


/// Returns true if this model selection has a training strategy associated,
/// and false otherwise.

bool ModelSelection::has_training_strategy() const
{
    if(training_strategy_pointer)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns the type of algorithm for the order selection.

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

GrowingNeurons* ModelSelection::get_growing_neurons_pointer()
{
    return &growing_neurons;
}


/// Returns a pointer to the growing inputs selection algorithm.

GrowingInputs* ModelSelection::get_growing_inputs_pointer()
{
    return &growing_inputs;
}


/// Returns a pointer to the pruning inputs selection algorithm.

PruningInputs* ModelSelection::get_pruning_inputs_pointer()
{
    return &pruning_inputs;
}


/// Returns a pointer to the genetic inputs selection algorithm.

GeneticAlgorithm* ModelSelection::get_genetic_algorithm_pointer()
{
    return &genetic_algorithm;
}


/// Sets the members of the model selection object to their default values.

void ModelSelection::set_default()
{
    set_neurons_selection_method(GROWING_NEURONS);

    set_inputs_selection_method(GROWING_INPUTS);

    display = true;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void ModelSelection::set_display(const bool& new_display)
{
    display = new_display;

    switch(inputs_selection_method)
    {
    case NO_INPUTS_SELECTION:
    {
        // do nothing

        break;
    }
    case GROWING_INPUTS:
    {
        growing_inputs.set_display(new_display);

        break;
    }
    case PRUNING_INPUTS:
    {
        pruning_inputs.set_display(new_display);

        break;
    }
    case GENETIC_ALGORITHM:
    {
        genetic_algorithm.set_display(new_display);

        break;
    }
    }

    switch(neurons_selection_method)
    {
    case NO_NEURONS_SELECTION:
    {
        // do nothing

        break;
    }
    case GROWING_NEURONS:
    {
        growing_neurons.set_display(new_display);

        break;
    }
    }
}


/// Sets a new method for selecting the order which have more impact on the targets.
/// @param new_neurons_selection_method Method for selecting the order(NO_NEURONS_SELECTION, growing_neurons, GOLDEN_SECTION, SIMULATED_ANNEALING).

void ModelSelection::set_neurons_selection_method(const ModelSelection::NeuronsSelectionMethod& new_neurons_selection_method)
{
    neurons_selection_method = new_neurons_selection_method;
}


/// Sets a new order selection algorithm from a string.
/// @param new_neurons_selection_method String with the order selection type.

void ModelSelection::set_neurons_selection_method(const string& new_neurons_selection_method)
{
    if(new_neurons_selection_method == "NO_NEURONS_SELECTION")
    {
        set_neurons_selection_method(NO_NEURONS_SELECTION);
    }
    else if(new_neurons_selection_method == "GROWING_NEURONS")
    {
        set_neurons_selection_method(GROWING_NEURONS);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_neurons_selection_method(const string&) method.\n"
               << "Unknown order selection type: " << new_neurons_selection_method << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets a new method for selecting the inputs which have more impact on the targets.
/// @param new_inputs_selection_method Method for selecting the inputs(NO_INPUTS_SELECTION, GROWING_INPUTS, PRUNING_INPUTS, GENETIC_ALGORITHM).

void ModelSelection::set_inputs_selection_method(const ModelSelection::InputsSelectionMethod& new_inputs_selection_method)
{
    inputs_selection_method = new_inputs_selection_method;
}


/// Sets a new inputs selection algorithm from a string.
/// @param new_inputs_selection_method String with the inputs selection type.

void ModelSelection::set_inputs_selection_method(const string& new_inputs_selection_method)
{
    if(new_inputs_selection_method == "NO_INPUTS_SELECTION")
    {
        set_inputs_selection_method(NO_INPUTS_SELECTION);
    }
    else if(new_inputs_selection_method == "GROWING_INPUTS")
    {
        set_inputs_selection_method(GROWING_INPUTS);
    }
    else if(new_inputs_selection_method == "PRUNING_INPUTS")
    {
        set_inputs_selection_method(PRUNING_INPUTS);
    }
    else if(new_inputs_selection_method == "GENETIC_ALGORITHM")
    {
        set_inputs_selection_method(GENETIC_ALGORITHM);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void set_inputs_selection_method(const string&) method.\n"
               << "Unknown inputs selection type: " << new_inputs_selection_method << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets a new approximation method.
/// If it is set to true the problem will be taken as a approximation;
/// if it is set to false the problem will be taken as a classification.
/// @param new_approximation Approximation value.

void ModelSelection::set_approximation(const bool& new_approximation)
{
    switch(inputs_selection_method)
    {
    case NO_INPUTS_SELECTION:
    {
        // do nothing

        break;
    }
    case GROWING_INPUTS:
    {
        growing_inputs.set_approximation(new_approximation);

        break;
    }
    case PRUNING_INPUTS:
    {
        pruning_inputs.set_approximation(new_approximation);

        break;
    }
    case GENETIC_ALGORITHM:
    {
        genetic_algorithm.set_approximation(new_approximation);

        break;
    }
    }
}


/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void ModelSelection::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;

    growing_neurons.set_training_strategy_pointer(new_training_strategy_pointer);

    growing_inputs.set_training_strategy_pointer(new_training_strategy_pointer);
    pruning_inputs.set_training_strategy_pointer(new_training_strategy_pointer);
    genetic_algorithm.set_training_strategy_pointer(new_training_strategy_pointer);
}


/// Checks that the different pointers needed for performing the model selection are not nullptr.

void ModelSelection::check() const
{

    // Optimization algorithm

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Neural network

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to neural network is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(neural_network_pointer->is_empty())
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Multilayer Perceptron is empty.\n";

        throw logic_error(buffer.str());
    }

    // Data set

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw logic_error(buffer.str());
    }

//

    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();

    if(selection_samples_number == 0)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Number of selection samples is zero.\n";

        throw logic_error(buffer.str());
    }
}


/// Perform the order selection, returns a structure with the results of the order selection.
/// It also set the neural network of the training strategy pointer with the optimum parameters.

ModelSelection::Results ModelSelection::perform_neurons_selection()
{
    Results results;

    TrainingStrategy* ts = get_training_strategy_pointer();

    switch(neurons_selection_method)
    {
    case NO_NEURONS_SELECTION:
    {
        break;
    }
    case GROWING_NEURONS:
    {
        growing_neurons.set_display(display);

        growing_neurons.set_training_strategy_pointer(ts);

        results.growing_neurons_results_pointer = growing_neurons.perform_neurons_selection();

        break;
    }
    }

    return results;
}


/// Perform the inputs selection, returns a structure with the results of the inputs selection.
/// It also set the neural network of the training strategy pointer with the optimum parameters.

ModelSelection::Results ModelSelection::perform_inputs_selection()
{
    Results results;

    TrainingStrategy* ts = get_training_strategy_pointer();

    switch(inputs_selection_method)
    {
    case NO_INPUTS_SELECTION:
    {
        break;
    }
    case GROWING_INPUTS:
    {
        growing_inputs.set_display(display);

        growing_inputs.set_training_strategy_pointer(ts);

        results.growing_inputs_results_pointer = growing_inputs.perform_inputs_selection();

        break;
    }
    case PRUNING_INPUTS:
    {
        pruning_inputs.set_display(display);

        pruning_inputs.set_training_strategy_pointer(ts);

        results.pruning_inputs_results_pointer = pruning_inputs.perform_inputs_selection();

        break;
    }
    case GENETIC_ALGORITHM:
    {
        genetic_algorithm.set_display(display);

        genetic_algorithm.set_training_strategy_pointer(ts);

        results.genetic_algorithm_results_pointer = genetic_algorithm.perform_inputs_selection();

        break;
    }
    }

    return results;
}


/// Perform inputs selection and order selection.
/// @todo

ModelSelection::Results ModelSelection::perform_model_selection()
{
    perform_inputs_selection();

    return perform_neurons_selection();
}


/// Serializes the model selection object into a XML document of the TinyXML library without keep the DOM tree in memory.
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
    pruning_inputs.write_XML(file_stream);
    genetic_algorithm.write_XML(file_stream);

    file_stream.CloseElement();

    // Model selection (end tag)

    file_stream.CloseElement();
}


/// Loads the members of this model selection object from a XML document.
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

        throw logic_error(buffer.str());
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


            // Pruning inputs

            const tinyxml2::XMLElement* pruning_inputs_element = inputs_selection_element->FirstChildElement("PruningInputs");

            if(pruning_inputs_element)
            {
                tinyxml2::XMLDocument pruning_inputs_document;

                tinyxml2::XMLElement* pruning_inputs_element_copy = pruning_inputs_document.NewElement("PruningInputs");

                for(const tinyxml2::XMLNode* nodeFor=pruning_inputs_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&pruning_inputs_document );
                    pruning_inputs_element_copy->InsertEndChild(copy );
                }

                pruning_inputs_document.InsertEndChild(pruning_inputs_element_copy);

                pruning_inputs.from_XML(pruning_inputs_document);
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
    switch (neurons_selection_method)
    {
    case NO_NEURONS_SELECTION:
        return "NO_NEURONS_SELECTION";

    case GROWING_NEURONS:
        return "GROWING_NEURONS";
    }
}


string ModelSelection::write_inputs_selection_method() const
{
    switch (inputs_selection_method)
    {
    case NO_INPUTS_SELECTION:
        return "NO_INPUTS_SELECTION";

    case GROWING_INPUTS:
        return "GROWING_INPUTS";

    case PRUNING_INPUTS:
        return "PRUNING_INPUTS";

    case GENETIC_ALGORITHM:
        return "GENETIC_ALGORITHM";
    }
}


/// Prints to the screen the XML representation of this model selection object.

void ModelSelection::print() const
{
//    cout << to_string();
}


/// Saves the model selection members to a XML file.
/// @param file_name Name of model selection XML file.

void ModelSelection::save(const string& file_name) const
{
    FILE *pFile;
//    int err;

//    err = fopen_s(&pFile, file_name.c_str(), "w");
    pFile = fopen(file_name.c_str(), "w");

    tinyxml2::XMLPrinter document(pFile);

    write_XML(document);

    fclose(pFile);
}


/// Loads the model selection members from a XML file.
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

        throw logic_error(buffer.str());
    }

    from_XML(document);
}

/// Results constructor.

ModelSelection::Results::Results()
{
    growing_neurons_results_pointer = nullptr;

    growing_inputs_results_pointer = nullptr;

    pruning_inputs_results_pointer = nullptr;

    genetic_algorithm_results_pointer = nullptr;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
