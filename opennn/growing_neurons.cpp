//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   N E U R O N S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "growing_neurons.h"

namespace opennn
{

/// Default constructor.

GrowingNeurons::GrowingNeurons()
    : NeuronsSelection()
{
    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a gradient descent object.

GrowingNeurons::GrowingNeurons(TrainingStrategy* new_training_strategy_pointer)
    : NeuronsSelection(new_training_strategy_pointer)
{
    set_default();
}


/// Returns the number of the hidden perceptrons pointed in each iteration of the growing neurons algorithm.

const Index& GrowingNeurons::get_step() const
{
    return neurons_increment;
}


/// Returns the maximum number of selection failures in the model neurons selection algorithm.

const Index& GrowingNeurons::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


/// Sets the members of the model selection object to their default values:

void GrowingNeurons::set_default()
{
    minimum_neurons = 1;

    maximum_neurons = 10;

    trials_number = 3;

    neurons_increment = 1;

    maximum_selection_failures = 100;

    maximum_time = type(3600);
}


/// Sets the number of the hidden perceptrons pointed in each iteration of the growing algorithm
/// in the model neurons selection process.
/// @param new_step number of hidden perceptrons pointed.

void GrowingNeurons::set_neurons_increment(const Index& new_neurons_increment)
{
#ifdef OPENNN_DEBUG

    if(new_neurons_increment <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingNeurons class.\n"
               << "void set_neurons_increment(const Index&) method.\n"
               << "New_step(" << new_neurons_increment << ") must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    neurons_increment = new_neurons_increment;
}


/// Sets the maximum selection failures for the growing neurons selection algorithm.
/// @param new_maximum_selection_failures Maximum number of selection failures in the growing neurons selection algorithm.

void GrowingNeurons::set_maximum_selection_failures(const Index& new_maximum_selection_failures)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_selection_failures <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingNeurons class.\n"
               << "void set_maximum_selection_failures(const Index&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_selection_failures;
}


/// Perform neurons selection with the growing neurons method.

NeuronsSelectionResults GrowingNeurons::perform_neurons_selection()
{
    #ifdef OPENNN_DEBUG

    if(!training_strategy_pointer)
    {
         ostringstream buffer;

         buffer << "OpenNN Exception: growing_neurons class.\n"
                << "TrainingStrategy* training_strategy_pointer const method.\n"
                << "training_strategy_pointer is nullptr.\n";

         throw invalid_argument(buffer.str());
    }

    #endif

    NeuronsSelectionResults neurons_selection_results(maximum_epochs_number);

    if(display) cout << "Performing growing neurons selection..." << endl;

    // Neural network    

    NeuralNetwork* neural_network = training_strategy_pointer->get_neural_network_pointer();

    const Index trainable_layers_number = neural_network->get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers_pointers = neural_network->get_trainable_layers_pointers();

    Index neurons_number;

    // Loss index

    type previous_selection_error = numeric_limits<type>::max();

    // Optimization algorithm

    Index selection_failures = 0;

    bool end = false;

    time_t beginning_time;
    time_t current_time;

    type elapsed_time = type(0);

    TrainingResults training_results;

    training_strategy_pointer->set_display(false);

    time(&beginning_time);

    // Main loop

    for(Index epoch = 0; epoch < maximum_epochs_number; epoch++)
    {               
        if(display) cout << endl << "Neurons selection epoch: " << epoch << endl;

        // Neural network

        neurons_number = minimum_neurons + epoch*neurons_increment;

        trainable_layers_pointers(trainable_layers_number-2)->set_neurons_number(neurons_number);

        trainable_layers_pointers(trainable_layers_number-1)->set_inputs_number(neurons_number);

        neurons_selection_results.neurons_number_history(epoch) = neurons_number;

        // Loss index

        type minimum_training_error = numeric_limits<type>::max();
        type minimum_selection_error = numeric_limits<type>::max();

        for(Index trial = 0; trial < trials_number; trial++)
        {
            neural_network->set_parameters_random();

            training_results = training_strategy_pointer->perform_training();            

            if(display)
            {
                cout << "Trial: " << trial+1 << endl;
                cout << "Training error: " << training_results.get_training_error() << endl;
                cout << "Selection error: " << training_results.get_selection_error() << endl;
            }

            if(training_results.get_selection_error() < minimum_selection_error)
            {
                minimum_training_error = training_results.get_training_error();
                minimum_selection_error = training_results.get_selection_error();

                neurons_selection_results.training_error_history(epoch) = minimum_training_error;
                neurons_selection_results.selection_error_history(epoch) = minimum_selection_error;
            }

            if(minimum_selection_error < neurons_selection_results.optimum_selection_error)
            {
                neurons_selection_results.optimal_neurons_number = neurons_number;
                neurons_selection_results.optimal_parameters = neural_network->get_parameters();

                neurons_selection_results.optimum_training_error = minimum_training_error;
                neurons_selection_results.optimum_selection_error = minimum_selection_error;                                
            }
        }

        if(display)
        {
            cout << "Neurons number: " << neurons_number << endl;
            cout << "Training error: " << training_results.get_training_error() << endl;
            cout << "Selection error: " << training_results.get_selection_error() << endl;

            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        if(neurons_selection_results.optimum_selection_error > previous_selection_error) selection_failures++;

        previous_selection_error = neurons_selection_results.optimum_selection_error;

        time(&current_time);

        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            end = true;

            if(display) cout << "Epoch " << epoch << endl << "Maximum time reached: " << write_time(elapsed_time) << endl;

            neurons_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::MaximumTime;
        }

        if(training_results.get_selection_error() <= selection_error_goal)
        {
            end = true;

            if(display) cout << "Epoch " << epoch << endl <<  "Selection error goal reached: " << training_results.get_selection_error() << endl;

            neurons_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::SelectionErrorGoal;
        }

        if(epoch >= maximum_epochs_number)
        {
            end = true;

            if(display) cout << "Epoch " << epoch << endl <<  "Maximum number of epochs reached: " << epoch << endl;

            neurons_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::MaximumEpochs;
        }

        if(selection_failures >= maximum_selection_failures)
        {
            end = true;

            if(display) cout << "Epoch " << epoch << endl <<  "Maximum selection failures reached: " << selection_failures << endl;

            neurons_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::MaximumSelectionFailures;
        }

        if(neurons_number >= maximum_neurons)
        {
            end = true;

            if(display) cout << "Epoch " << epoch << endl <<  "Maximum number of neurons reached: " << neurons_number << endl;

            neurons_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::MaximumNeurons;
        }

        if(end)
        {
            neurons_selection_results.resize_history(epoch+1);

            neurons_selection_results.elapsed_time = write_time(elapsed_time);

            break;
        }
    }

    // Save neural network

    trainable_layers_pointers[trainable_layers_number-1]->set_inputs_number(neurons_selection_results.optimal_neurons_number);
    trainable_layers_pointers[trainable_layers_number-2]->set_neurons_number(neurons_selection_results.optimal_neurons_number);

    neural_network->set_parameters(neurons_selection_results.optimal_parameters);

    if(display) neurons_selection_results.print();

    return neurons_selection_results;
}


/// This method writes a matrix of strings the most representative atributes.

Tensor<string, 2> GrowingNeurons::to_string_matrix() const
{
    ostringstream buffer;

    Tensor<string, 1> labels(8);
    Tensor<string, 1> values(8);

    // Minimum neurons number

    labels(0) = "Minimum neurons";

    buffer.str("");
    buffer << minimum_neurons;

    values(0) = buffer.str();

    // Maximum order

    labels(1) = "Maximum neurons";

    buffer.str("");
    buffer << maximum_neurons;

    values(1) = buffer.str();

    // Step

    labels(2) = "Step";

    buffer.str("");
    buffer << neurons_increment;

    values(2) = buffer.str();

    // Trials number

    labels(3) = "Trials number";

    buffer.str("");
    buffer << trials_number;

    values(3) = buffer.str();

    // Selection loss goal

    labels(4) = "Selection loss goal";

    buffer.str("");
    buffer << selection_error_goal;

    values(4) = buffer.str();

    // Maximum selection failures

    labels(5) = "Maximum selection failures";

    buffer.str("");
    buffer << maximum_selection_failures;

    values(5) = buffer.str();

    // Maximum iterations number

    labels(6) = "Maximum iterations number";

    buffer.str("");
    buffer << maximum_epochs_number;

    values(6) = buffer.str();

    // Maximum time

    labels(7) = "Maximum time";

    buffer.str("");
    buffer << maximum_time;

    values(7) = buffer.str();

    const Index rows_number = labels.size();
    const Index columns_number = 2;

    Tensor<string, 2> string_matrix(rows_number, columns_number);

    string_matrix.chip(0, 1) = labels;
    string_matrix.chip(1, 1) = values;

    return string_matrix;
}


/// Serializes the growing neurons object into an XML document of the TinyXML library without
/// keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GrowingNeurons::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("GrowingNeurons");

    // Minimum order

    file_stream.OpenElement("MinimumNeurons");

    buffer.str("");
    buffer << minimum_neurons;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum order

    file_stream.OpenElement("MaximumNeurons");

    buffer.str("");
    buffer << maximum_neurons;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Step

    file_stream.OpenElement("Step");

    buffer.str("");
    buffer << neurons_increment;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Trials number

    file_stream.OpenElement("TrialsNumber");

    buffer.str("");
    buffer << trials_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Selection error goal

    file_stream.OpenElement("SelectionErrorGoal");

    buffer.str("");
    buffer << selection_error_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum selection failures

    file_stream.OpenElement("MaximumSelectionFailures");

    buffer.str("");
    buffer << maximum_selection_failures;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this growing neurons object.
/// @param document TinyXML document containing the member data.

void GrowingNeurons::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GrowingNeurons");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingNeurons class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "GrowingNeurons element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Minimum neurons
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumNeurons");

        if(element)
        {
            const Index new_minimum_neurons = static_cast<Index>(atoi(element->GetText()));

            try
            {
                minimum_neurons = new_minimum_neurons;
            }
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum neurons
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumNeurons");

        if(element)
        {
            const Index new_maximum_neurons = static_cast<Index>(atoi(element->GetText()));

            try
            {
                maximum_neurons = new_maximum_neurons;
            }
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Step
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Step");

        if(element)
        {
            const Index new_step = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_neurons_increment(new_step);
            }
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Trials number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("TrialsNumber");

        if(element)
        {
            const Index new_trials_number = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_trials_number(new_trials_number);
            }
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Selection error goal
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("SelectionErrorGoal");

        if(element)
        {
            const type new_selection_error_goal = static_cast<type>(atof(element->GetText()));

            try
            {
                set_selection_error_goal(new_selection_error_goal);
            }
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum selection failures
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumSelectionFailures");

        if(element)
        {
            const Index new_maximum_selection_failures = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_maximum_selection_failures(new_maximum_selection_failures);
            }
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum time
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumTime");

        if(element)
        {
            const type new_maximum_time = type(atoi(element->GetText()));

            try
            {
                set_maximum_time(new_maximum_time);
            }
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
    }
}


/// Saves to an XML-type file the members of the growing neurons object.
/// @param file_name Name of growing neurons XML-type file.

void GrowingNeurons::save(const string& file_name) const
{
    FILE * file = fopen(file_name.c_str(), "w");

    if(file)
    {
        tinyxml2::XMLPrinter printer(file);
        write_XML(printer);
        fclose(file);
    }
}


/// Loads a growing neurons object from an XML-type file.
/// @param file_name Name of growing neurons XML-type file.

void GrowingNeurons::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingNeurons class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw invalid_argument(buffer.str());
    }

    from_XML(document);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
