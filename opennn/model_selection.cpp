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
    training_strategy_pointer = nullptr;
    incremental_neurons_pointer = nullptr;
    growing_inputs_pointer = nullptr;
    pruning_inputs_pointer = nullptr;
    genetic_algorithm_pointer = nullptr;

    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

ModelSelection::ModelSelection(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;
    incremental_neurons_pointer = nullptr;
    growing_inputs_pointer = nullptr;
    pruning_inputs_pointer = nullptr;
    genetic_algorithm_pointer = nullptr;

    set_default();
}


/// File constructor.
/// @param file_name Name of XML model selection file.

ModelSelection::ModelSelection(const string& file_name)
{
   training_strategy_pointer = nullptr;
   incremental_neurons_pointer = nullptr;
   growing_inputs_pointer = nullptr;
   pruning_inputs_pointer = nullptr;
   genetic_algorithm_pointer = nullptr;

   load(file_name);
}


/// XML constructor. 
/// @param model_selection_document Pointer to a TinyXML document containing the model selection data.

ModelSelection::ModelSelection(const tinyxml2::XMLDocument& model_selection_document)
{
    training_strategy_pointer = nullptr;
    incremental_neurons_pointer = nullptr;
    growing_inputs_pointer = nullptr;
    pruning_inputs_pointer = nullptr;
    genetic_algorithm_pointer = nullptr;

    from_XML(model_selection_document);
}


/// Destructor.

ModelSelection::~ModelSelection()
{
    // Delete inputs selection algorithms

    delete growing_inputs_pointer;
    delete pruning_inputs_pointer;
    delete genetic_algorithm_pointer;

    // Delete order selection algorithms

    delete incremental_neurons_pointer;
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

const ModelSelection::OrderSelectionMethod& ModelSelection::get_neurons_selection_method() const
{
    return neurons_selection_method;
}


/// Returns the type of algorithm for the inputs selection.

const ModelSelection::InputsSelectionMethod& ModelSelection::get_inputs_selection_method() const
{
    return inputs_selection_method;
}


/// Returns a pointer to the incremental order selection algorithm.

IncrementalNeurons* ModelSelection::get_incremental_neurons_pointer() const
{
    return incremental_neurons_pointer;
}


/// Returns a pointer to the growing inputs selection algorithm.

GrowingInputs* ModelSelection::get_growing_inputs_pointer() const
{
    return growing_inputs_pointer;
}


/// Returns a pointer to the pruning inputs selection algorithm.

PruningInputs* ModelSelection::get_pruning_inputs_pointer() const
{
    return pruning_inputs_pointer;
}


/// Returns a pointer to the genetic inputs selection algorithm.

GeneticAlgorithm* ModelSelection::get_genetic_algorithm_pointer() const
{
    return genetic_algorithm_pointer;
}


/// Sets the members of the model selection object to their default values.

void ModelSelection::set_default()
{
    set_neurons_selection_method(INCREMENTAL_NEURONS);

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
        growing_inputs_pointer->set_display(new_display);

        break;
    }
    case PRUNING_INPUTS:
    {
        pruning_inputs_pointer->set_display(new_display);

        break;
    }
    case GENETIC_ALGORITHM:
    {
        genetic_algorithm_pointer->set_display(new_display);

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
    case INCREMENTAL_NEURONS:
    {
        incremental_neurons_pointer->set_display(new_display);

        break;
    }
    }
}


/// Sets a new method for selecting the order which have more impact on the targets.
/// @param new_neurons_selection_method Method for selecting the order(NO_NEURONS_SELECTION, INCREMENTAL_NEURONS, GOLDEN_SECTION, SIMULATED_ANNEALING).

void ModelSelection::set_neurons_selection_method(const ModelSelection::OrderSelectionMethod& new_neurons_selection_method)
{
    destruct_neurons_selection();

    neurons_selection_method = new_neurons_selection_method;

    switch(new_neurons_selection_method)
    {
        case NO_NEURONS_SELECTION:
        {
            // do nothing

            break;
        }
        case INCREMENTAL_NEURONS:
        {

            if(training_strategy_pointer == nullptr)
            {
                incremental_neurons_pointer = new IncrementalNeurons();
            }
            else
            {
                incremental_neurons_pointer = new IncrementalNeurons(training_strategy_pointer);
            }

//            if(training_strategy_pointer != nullptr)
//            {
//                incremental_neurons_pointer->set_training_strategy_pointer(training_strategy_pointer);
//            }

            break;
        }
    }
}


/// Sets a new order selection algorithm from a string.
/// @param new_neurons_selection_method String with the order selection type.

void ModelSelection::set_neurons_selection_method(const string& new_neurons_selection_method)
{
    if(new_neurons_selection_method == "NO_NEURONS_SELECTION")
    {
        set_neurons_selection_method(NO_NEURONS_SELECTION);
    }
    else if(new_neurons_selection_method == "INCREMENTAL_NEURONS")
    {
        set_neurons_selection_method(INCREMENTAL_NEURONS);
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
    destruct_inputs_selection();

    inputs_selection_method = new_inputs_selection_method;

    switch(new_inputs_selection_method)
    {
        case NO_INPUTS_SELECTION:
        {
           // do nothing

            break;
        }
        case GROWING_INPUTS:
        {
            growing_inputs_pointer = new GrowingInputs;
/*
            if(training_strategy_pointer == nullptr)
            {
                incremental_neurons_pointer = new IncrementalNeurons();
            }
            else
            {
                incremental_neurons_pointer = new IncrementalNeurons(training_strategy_pointer);
            }
*/
            if(training_strategy_pointer != nullptr) growing_inputs_pointer->set_training_strategy_pointer(training_strategy_pointer);

            break;
        }
        case PRUNING_INPUTS:
        {
            pruning_inputs_pointer = new PruningInputs;

            if(training_strategy_pointer != nullptr) pruning_inputs_pointer->set_training_strategy_pointer(training_strategy_pointer);

            break;
        }
        case GENETIC_ALGORITHM:
        {
            genetic_algorithm_pointer = new GeneticAlgorithm;

            if(training_strategy_pointer != nullptr) genetic_algorithm_pointer->set_training_strategy_pointer(training_strategy_pointer);

            break;
        }
    }
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
        growing_inputs_pointer->set_approximation(new_approximation);

        break;
    }
    case PRUNING_INPUTS:
    {
        pruning_inputs_pointer->set_approximation(new_approximation);

        break;
    }
    case GENETIC_ALGORITHM:
    {
        genetic_algorithm_pointer->set_approximation(new_approximation);

        break;
    }
    }
}


/// Sets a new training strategy pointer.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

void ModelSelection::set_training_strategy_pointer(TrainingStrategy* new_training_strategy_pointer)
{
    training_strategy_pointer = new_training_strategy_pointer;

    switch(neurons_selection_method)
    {
        case NO_NEURONS_SELECTION:
        {
            // do nothing

            break;
        }
        case INCREMENTAL_NEURONS:
        {
            incremental_neurons_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
            break;
        }
    }

    switch(inputs_selection_method)
    {
        case NO_INPUTS_SELECTION:
        {
            // do nothing

            break;
        }
        case GROWING_INPUTS:
        {
            growing_inputs_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
            break;
        }
        case PRUNING_INPUTS:
        {
            pruning_inputs_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
            break;
        }
        case GENETIC_ALGORITHM:
        {
            genetic_algorithm_pointer->set_training_strategy_pointer(new_training_strategy_pointer);
            break;
        }
    }
}


/// This method deletes the order selection algorithm object which composes this model selection object.

void ModelSelection::destruct_neurons_selection()
{
    delete incremental_neurons_pointer;

    incremental_neurons_pointer = nullptr;

    neurons_selection_method = NO_NEURONS_SELECTION;
}


/// This method deletes the inputs selection algorithm object which composes this model selection object.

void ModelSelection::destruct_inputs_selection()
{
    delete growing_inputs_pointer;
    delete pruning_inputs_pointer;
    delete genetic_algorithm_pointer;

    growing_inputs_pointer = nullptr;
    pruning_inputs_pointer = nullptr;
    genetic_algorithm_pointer = nullptr;

    inputs_selection_method = NO_INPUTS_SELECTION;
}


/// Checks that the different pointers needed for performing the model selection are not nullptr.

void ModelSelection::check() const
{

    // Optimization algorithm stuff

    ostringstream buffer;

    if(!training_strategy_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to training strategy is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index stuff

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Neural network stuff

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

    // Data set stuff

    const DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw logic_error(buffer.str());
    }

//

    const size_t selection_instances_number = data_set_pointer->get_selection_instances_number();

    if(selection_instances_number == 0)
    {
        buffer << "OpenNN Exception: ModelSelection class.\n"
               << "void check() const method.\n"
               << "Number of selection instances is zero.\n";

        throw logic_error(buffer.str());
    }
}


/// Perform the order selection, returns a structure with the results of the order selection.
/// It also set the neural network of the training strategy pointer with the optimum parameters.

ModelSelection::Results ModelSelection::perform_neurons_selection() const
{
    Results results;

    switch(neurons_selection_method)
    {
        case NO_NEURONS_SELECTION:
        {
            break;
        }
        case INCREMENTAL_NEURONS:
        {        
            incremental_neurons_pointer->set_display(display);

            results.incremental_neurons_results_pointer = incremental_neurons_pointer->perform_neurons_selection();

            break;
        }
    }

    return results;
}


/// Perform the inputs selection, returns a structure with the results of the inputs selection.
/// It also set the neural network of the training strategy pointer with the optimum parameters.

ModelSelection::Results ModelSelection::perform_inputs_selection() const
{
    Results results;

    switch(inputs_selection_method)
    {
        case NO_INPUTS_SELECTION:
        {
            break;
        }
        case GROWING_INPUTS:
        {
            growing_inputs_pointer->set_display(display);

            results.growing_inputs_results_pointer = growing_inputs_pointer->perform_inputs_selection();

            break;
        }
        case PRUNING_INPUTS:
        {
            pruning_inputs_pointer->set_display(display);

            results.pruning_inputs_results_pointer = pruning_inputs_pointer->perform_inputs_selection();

            break;
        }
        case GENETIC_ALGORITHM:
        {
            genetic_algorithm_pointer->set_display(display);

            results.genetic_algorithm_results_pointer = genetic_algorithm_pointer->perform_inputs_selection();

            break;
        }
    }

    return results;
}


/// Perform inputs selection and order selection.
/// @todo

ModelSelection::Results ModelSelection::perform_model_selection() const
{
    perform_inputs_selection();

    return perform_neurons_selection();
}


/// Serializes the model selection object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document. 

tinyxml2::XMLDocument* ModelSelection::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Training strategy

    tinyxml2::XMLElement* model_selection_element = document->NewElement("ModelSelection");

    document->InsertFirstChild(model_selection_element);

    // Inputs Selection

    switch(inputs_selection_method)
    {
    case NO_INPUTS_SELECTION:
    {
        tinyxml2::XMLElement* inputs_selection_element = document->NewElement("InputsSelection");
        model_selection_element->LinkEndChild(inputs_selection_element);

        inputs_selection_element->SetAttribute("Type", "NO_INPUTS_SELECTION");
    }
        break;

    case GROWING_INPUTS:
    {
        tinyxml2::XMLElement* inputs_selection_element = document->NewElement("InputsSelection");
        model_selection_element->LinkEndChild(inputs_selection_element);

        inputs_selection_element->SetAttribute("Type", "GROWING_INPUTS");

        const tinyxml2::XMLDocument* growing_inputs_document = growing_inputs_pointer->to_XML();

        const tinyxml2::XMLElement* growing_inputs_element = growing_inputs_document->FirstChildElement("GrowingInputs");

        for(const tinyxml2::XMLNode* nodeFor=growing_inputs_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone(document );
            inputs_selection_element->InsertEndChild(copy );
        }

        delete growing_inputs_document;
    }
        break;

    case PRUNING_INPUTS:
    {
        tinyxml2::XMLElement* inputs_selection_element = document->NewElement("InputsSelection");
        model_selection_element->LinkEndChild(inputs_selection_element);

        inputs_selection_element->SetAttribute("Type", "PRUNING_INPUTS");

        const tinyxml2::XMLDocument* pruning_inputs_document = pruning_inputs_pointer->to_XML();

        const tinyxml2::XMLElement* pruning_inputs_element = pruning_inputs_document->FirstChildElement("PruningInputs");

        for(const tinyxml2::XMLNode* nodeFor=pruning_inputs_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone(document );
            inputs_selection_element->InsertEndChild(copy );
        }

        delete pruning_inputs_document;
    }
        break;

    case GENETIC_ALGORITHM:
    {
        tinyxml2::XMLElement* inputs_selection_element = document->NewElement("InputsSelection");
        model_selection_element->LinkEndChild(inputs_selection_element);

        inputs_selection_element->SetAttribute("Type", "GENETIC_ALGORITHM");

        const tinyxml2::XMLDocument* genetic_algorithm_document = genetic_algorithm_pointer->to_XML();

        const tinyxml2::XMLElement* genetic_algorithm_element = genetic_algorithm_document->FirstChildElement("GeneticAlgorithm");

        for(const tinyxml2::XMLNode* nodeFor=genetic_algorithm_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone(document );
            inputs_selection_element->InsertEndChild(copy );
        }

        delete genetic_algorithm_document;
    }
        break;
    }

    // Order Selection

    switch(neurons_selection_method)
    {
    case NO_NEURONS_SELECTION:
    {
        tinyxml2::XMLElement* neurons_selection_element = document->NewElement("NeuronsSelection");
        model_selection_element->LinkEndChild(neurons_selection_element);

        neurons_selection_element->SetAttribute("Type", "NO_NEURONS_SELECTION");
    }
        break;

    case INCREMENTAL_NEURONS:
    {
        tinyxml2::XMLElement* neurons_selection_element = document->NewElement("NeuronsSelection");
        model_selection_element->LinkEndChild(neurons_selection_element);

        neurons_selection_element->SetAttribute("Type", "INCREMENTAL_NEURONS");

        const tinyxml2::XMLDocument* incremental_order_document = incremental_neurons_pointer->to_XML();

        const tinyxml2::XMLElement* incremental_order_element = incremental_order_document->FirstChildElement("IncrementalNeurons");

        for(const tinyxml2::XMLNode* nodeFor=incremental_order_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone(document );
            neurons_selection_element->InsertEndChild(copy );
        }

        delete incremental_order_document;
    }
        break;
    }

    return document;
}


/// Serializes the model selection object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void ModelSelection::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("ModelSelection");

    // Inputs Selection

    switch(inputs_selection_method)
    {
    case NO_INPUTS_SELECTION:
    {
        file_stream.OpenElement("InputsSelection");

        file_stream.PushAttribute("Type", "NO_INPUTS_SELECTION");

        file_stream.CloseElement();
    }
        break;

    case GROWING_INPUTS:
    {
        file_stream.OpenElement("InputsSelection");

        file_stream.PushAttribute("Type", "GROWING_INPUTS");

        growing_inputs_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;

    case PRUNING_INPUTS:
    {
        file_stream.OpenElement("InputsSelection");

        file_stream.PushAttribute("Type", "PRUNING_INPUTS");

        pruning_inputs_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;

    case GENETIC_ALGORITHM:
    {
        file_stream.OpenElement("InputsSelection");

        file_stream.PushAttribute("Type", "GENETIC_ALGORITHM");

        genetic_algorithm_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;
    }

    // Order Selection

    switch(neurons_selection_method)
    {
    case NO_NEURONS_SELECTION:
    {
        file_stream.OpenElement("NeuronsSelection");

        file_stream.PushAttribute("Type", "NO_NEURONS_SELECTION");

        file_stream.CloseElement();
    }
        break;

    case INCREMENTAL_NEURONS:
    {
        file_stream.OpenElement("NeuronsSelection");

        file_stream.PushAttribute("Type", "INCREMENTAL_NEURONS");

        incremental_neurons_pointer->write_XML(file_stream);

        file_stream.CloseElement();
    }
        break;
    }

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

    // Inputs Selection
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("InputsSelection");

        if(element)
        {
            const string new_inputs_selection_method = element->Attribute("Type");

            set_inputs_selection_method(new_inputs_selection_method);

            switch(inputs_selection_method)
            {
            case NO_INPUTS_SELECTION:
            {
                // do nothing
            }
                break;
            case GROWING_INPUTS:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* growing_element = new_document.NewElement("GrowingInputs");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    growing_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(growing_element);

                growing_inputs_pointer->from_XML(new_document);
            }
                break;
            case PRUNING_INPUTS:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* pruning_element = new_document.NewElement("PruningInputs");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    pruning_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(pruning_element);

                pruning_inputs_pointer->from_XML(new_document);
            }
                break;
            case GENETIC_ALGORITHM:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* genetic_element = new_document.NewElement("GeneticAlgorithm");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    genetic_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(genetic_element);

                genetic_algorithm_pointer->from_XML(new_document);
            }
                break;
            }
        }
    }

    // Order Selection
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("NeuronsSelection");

        if(element)
        {
            const string new_neurons_selection_method = element->Attribute("Type");

            set_neurons_selection_method(new_neurons_selection_method);

            switch(neurons_selection_method)
            {
            case NO_NEURONS_SELECTION:
            {
                // do nothing
            }
                break;
            case INCREMENTAL_NEURONS:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* incremental_element = new_document.NewElement("IncrementalNeurons");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling() ) {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    incremental_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(incremental_element);

                incremental_neurons_pointer->from_XML(new_document);
            }
                break;
            }
        }
    }
}


/// Prints to the screen the XML representation of this model selection object. 

void ModelSelection::print() const
{
    cout << to_XML();
}


/// Saves the model selection members to a XML file. 
/// @param file_name Name of model selection XML file. 

void ModelSelection::save(const string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
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
    incremental_neurons_results_pointer = nullptr;

    growing_inputs_results_pointer = nullptr;

    pruning_inputs_results_pointer = nullptr;

    genetic_algorithm_results_pointer = nullptr;
}


/// Saves the results structure to a data file.
/// @param file_name Name of model selection results data file.

void ModelSelection::Results::save(const string& file_name) const
{
    ofstream file(file_name.c_str());

    file << "% Model Selection Results\n";

    file << "\n% Order Selection Results\n";

    if(incremental_neurons_results_pointer)
    {
        file << incremental_neurons_results_pointer->object_to_string();
    }

    file << "\n% Inputs Selection Results\n";

    if(growing_inputs_results_pointer)
    {
        file << growing_inputs_results_pointer->object_to_string();
    }

    if(pruning_inputs_results_pointer)
    {
        file << pruning_inputs_results_pointer->object_to_string();
    }

    if(genetic_algorithm_results_pointer)
    {
        file << genetic_algorithm_results_pointer->object_to_string();
    }

    file.close();
}


/// @todo

Vector<NeuralNetwork> ModelSelection::perform_k_fold_cross_validation(const size_t& k) const
{
    DataSet* data_set_pointer = training_strategy_pointer->get_loss_index_pointer()->get_data_set_pointer();

    const Vector<size_t> selection_instances_indices = data_set_pointer->get_selection_instances_indices();

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_neural_network_pointer();
    LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(k < 2)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_k_fold_cross_validation(const size_t&).\n"
              << "Number of iterations must be grater or equal than 2.\n";

       throw logic_error(buffer.str());
    }

    if(!data_set_pointer || data_set_pointer->has_data())
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_k_fold_cross_validation(const size_t&).\n"
              << "There is no data set assigned.\n";

       throw logic_error(buffer.str());
    }

    if(!neural_network_pointer)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_k_fold_cross_validation(const size_t&).\n"
              << "There is no neural network assigned.\n";

       throw logic_error(buffer.str());
    }
#endif

    //

    const Vector<DataSet::InstanceUse> original_uses = data_set_pointer->get_instances_uses();

    data_set_pointer->split_instances_random(1,0,0);

    Vector<double> minimum_error_parameters;
    double minimum_error = 1.0;

    Vector<NeuralNetwork> neural_network_ensemble(k);
    double cross_validation_error = 0.0;

    for(size_t i = 0; i < k; i++)
    {
        data_set_pointer->set_k_fold_cross_validation_instances_uses(k,i);

        neural_network_pointer->randomize_parameters_normal();

        training_strategy_pointer->perform_training();

        data_set_pointer->set_testing_to_selection_instances();

        const double current_error = loss_index_pointer->calculate_selection_error();

        if(i == 0 || current_error < minimum_error)
        {
            minimum_error = current_error;
            minimum_error_parameters = neural_network_pointer->get_parameters();
        }

        neural_network_ensemble[i].set(*neural_network_pointer);
        cross_validation_error += current_error;

        if(display)
        {
            cout << "Iteration: " << i << "/" << k << endl;
            cout << "Current error: " << current_error << endl;
        }
    }

    if(display)
    {
        cout << "Cross validation error: " << cross_validation_error/k << endl;
    }

    data_set_pointer->set_instances_uses(original_uses);

    neural_network_pointer->set_parameters(minimum_error_parameters);

    return neural_network_ensemble;
}


/// @todo

Vector<NeuralNetwork> ModelSelection::perform_random_cross_validation(const size_t& k, const double& selection_ratio) const
{
    DataSet* data_set_pointer = training_strategy_pointer->get_loss_index_pointer()->get_data_set_pointer();

    const Vector<size_t> selection_instances_indices = data_set_pointer->get_selection_instances_indices();

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_neural_network_pointer();
    LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(k < 2)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_random_cross_validation(const size_t&, const double&).\n"
              << "Number of iterations must be grater or equal than 2.\n";

       throw logic_error(buffer.str());
    }

    if(selection_ratio <= 0.0 || selection_ratio >= 1.0)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_random_cross_validation(const size_t&, const double&).\n"
              << "The ratio of testing instances must be between 0.0 and 1.0.\n";

       throw logic_error(buffer.str());
    }

    if(!data_set_pointer || data_set_pointer->has_data())
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_random_cross_validation(const size_t&, const double&).\n"
              << "There is no data set assigned.\n";

       throw logic_error(buffer.str());
    }

    if(!neural_network_pointer)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_random_cross_validation(const size_t&, const double&).\n"
              << "There is no neural network assigned.\n";

       throw logic_error(buffer.str());
    }
#endif

    //

    const Vector<DataSet::InstanceUse> original_uses = data_set_pointer->get_instances_uses();

    data_set_pointer->split_instances_random(1,0,0);

    Vector<double> minimum_error_parameters;
    double minimum_error = 1.0;

    Vector<NeuralNetwork> neural_network_ensemble(k);
    double cross_validation_error = 0.0;

    for(size_t i = 0; i < k; i++)
    {
        data_set_pointer->split_instances_random(1-selection_ratio,0.0,selection_ratio);

        neural_network_pointer->randomize_parameters_normal();

        training_strategy_pointer->perform_training();

        data_set_pointer->set_testing_to_selection_instances();

        const double current_error = loss_index_pointer->calculate_selection_error();

        if(i == 0 || current_error < minimum_error)
        {
            minimum_error = current_error;
            minimum_error_parameters = neural_network_pointer->get_parameters();
        }

        neural_network_ensemble[i].set(*neural_network_pointer);
        cross_validation_error += current_error;

        if(display)
        {
            cout << "Iteration: " << i << "/" << k << endl;
            cout << "Current error: " << current_error << endl;
        }
    }

    if(display)
    {
        cout << "Cross validation error: " << cross_validation_error/k << endl;
    }

    data_set_pointer->set_instances_uses(original_uses);
    neural_network_pointer->set_parameters(minimum_error_parameters);

    return neural_network_ensemble;
}


/// @todo Check this method.

Vector<NeuralNetwork> ModelSelection::perform_positives_cross_validation() const
{
    DataSet* data_set_pointer = training_strategy_pointer->get_loss_index_pointer()->get_data_set_pointer();

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_neural_network_pointer();

    LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!data_set_pointer || data_set_pointer->has_data())
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_positives_cross_validation().\n"
              << "There is no data set assigned.\n";

       throw logic_error(buffer.str());
    }

    if(!neural_network_pointer)
    {
       buffer << "OpenNN Exception: ModelSelection class.\n"
              << "Vector<NeuralNetwork> perform_positives_cross_validation().\n"
              << "There is no neural network assigned.\n";

       throw logic_error(buffer.str());
    }
#endif

    //

    const Vector<size_t> training_indices = data_set_pointer->get_training_instances_indices();

    const Vector<DataSet::InstanceUse> original_uses = data_set_pointer->get_instances_uses();

    const Vector<size_t> inputs_indices = data_set_pointer->get_input_variables_indices();

    const size_t target_index = data_set_pointer->get_target_variables_indices()[0];

    const Vector<size_t> positives_instances_indices = data_set_pointer->get_variable_data(target_index).get_indices_greater_than(0.5);

    const size_t positives_instances_number = positives_instances_indices.size();

    data_set_pointer->split_instances_random(1,0,0);

    Vector<double> minimum_error_parameters;
    double minimum_error = 1.0;

    Vector<NeuralNetwork> neural_network_ensemble(positives_instances_number);
    double cross_validation_error = 0.0;

    for(size_t i = 0; i < positives_instances_number; i++)
    {
        const size_t current_selection_instance_index = positives_instances_indices[i];
        const Vector<double> current_selection_instance = data_set_pointer->get_instance_data(current_selection_instance_index);
        const double targets = current_selection_instance[target_index];
        const Vector<double> current_inputs_selection_instance = current_selection_instance.get_subvector(inputs_indices);

        data_set_pointer->set_instance_use(current_selection_instance_index, DataSet::Testing);
        neural_network_pointer->randomize_parameters_normal();

        training_strategy_pointer->perform_training();

        const double outputs = neural_network_pointer->calculate_outputs(current_inputs_selection_instance.to_tensor({1}))(0,0);

        const double current_error = abs(targets-outputs);
        const double current_loss = loss_index_pointer->calculate_training_loss();

        if(i == 0 || current_error < minimum_error)
        {
            minimum_error = current_error;
            minimum_error_parameters = neural_network_pointer->get_parameters();
        }

        data_set_pointer->set_instance_use(current_selection_instance_index, DataSet::Training);

        neural_network_ensemble[i].set(*neural_network_pointer);
        cross_validation_error += current_error;

        if(display)
        {
            cout << "Iteration: " << i+1 << "/" << positives_instances_number << endl;
            cout << "instance index: " << current_selection_instance_index << endl;
            cout << "Output data: " << outputs << endl;
            cout << "Current loss: " << current_loss << endl;
            cout << "Current error: " << current_error << endl;
        }
    }

    if(display)
    {
        cout << "Cross validation error: " << cross_validation_error/positives_instances_number << endl;
    }

    data_set_pointer->set_instances_uses(original_uses);
    neural_network_pointer->set_parameters(minimum_error_parameters);

    return neural_network_ensemble;

//    return Vector<NeuralNetwork>();

}
}
