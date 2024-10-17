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

GrowingNeurons::GrowingNeurons()
    : NeuronsSelection()
{
    set_default();
}


GrowingNeurons::GrowingNeurons(TrainingStrategy* new_training_strategy)
    : NeuronsSelection(new_training_strategy)
{
    set_default();
}


const Index& GrowingNeurons::get_step() const
{
    return neurons_increment;
}


const Index& GrowingNeurons::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


void GrowingNeurons::set_default()
{
    minimum_neurons = 1;

    maximum_neurons = 10;

    trials_number = 3;

    neurons_increment = 1;

    maximum_selection_failures = 100;

    maximum_time = type(3600);
}


void GrowingNeurons::set_neurons_increment(const Index& new_neurons_increment)
{
    neurons_increment = new_neurons_increment;
}


void GrowingNeurons::set_maximum_selection_failures(const Index& new_maximum_selection_failures)
{
    maximum_selection_failures = new_maximum_selection_failures;
}


NeuronsSelectionResults GrowingNeurons::perform_neurons_selection()
{
    #ifdef OPENNN_DEBUG

    if(!training_strategy)
         throw runtime_error("training_strategy is nullptr.\n");

    #endif

    NeuronsSelectionResults neurons_selection_results(maximum_epochs_number);

    if(display) cout << "Performing growing neurons selection..." << endl;

    // Neural network    

    NeuralNetwork* neural_network = training_strategy->get_neural_network();
/*
    const Index trainable_layers_number = neural_network->get_trainable_layers_number();

    const vector<Layer*> trainable_layers = neural_network->get_trainable_layers();

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

    training_strategy->set_display(false);

    time(&beginning_time);

    // Main loop

    for(Index epoch = 0; epoch < maximum_epochs_number; epoch++)
    {               
        if(display) cout << endl << "Neurons selection epoch: " << epoch << endl;

        // Neural network

        neurons_number = minimum_neurons + epoch*neurons_increment;

        trainable_layers[trainable_layers_number-2]->set_neurons_number(neurons_number);

        trainable_layers[trainable_layers_number-1]->set_inputs_number(neurons_number);

        neurons_selection_results.neurons_number_history(epoch) = neurons_number;

        // Loss index

        type minimum_training_error = numeric_limits<type>::max();
        type minimum_selection_error = numeric_limits<type>::max();

        for(Index trial = 0; trial < trials_number; trial++)
        {
            neural_network->set_parameters_random();

            training_results = training_strategy->perform_training();

            if(display)
            {
                cout << "Trial: " << trial+1 << endl
                     << "Training error: " << training_results.get_training_error() << endl
                     << "Selection error: " << training_results.get_selection_error() << endl;
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
            cout << "Neurons number: " << neurons_number << endl
                 << "Training error: " << training_results.get_training_error() << endl
                 << "Selection error: " << training_results.get_selection_error() << endl
                 << "Elapsed time: " << write_time(elapsed_time) << endl;

        if(neurons_selection_results.optimum_selection_error > previous_selection_error) selection_failures++;

        previous_selection_error = neurons_selection_results.optimum_selection_error;

        time(&current_time);

        elapsed_time = type(difftime(current_time, beginning_time));

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

    trainable_layers[trainable_layers_number-1]->set_inputs_number(neurons_selection_results.optimal_neurons_number);
    trainable_layers[trainable_layers_number-2]->set_neurons_number(neurons_selection_results.optimal_neurons_number);

    neural_network->set_parameters(neurons_selection_results.optimal_parameters);

    if(display) neurons_selection_results.print();
*/
    return neurons_selection_results;
}


Tensor<string, 2> GrowingNeurons::to_string_matrix() const
{
    Tensor<string, 1> labels(8);
    Tensor<string, 1> values(8);

    labels(0) = "Minimum neurons";
    values(0) = to_string(minimum_neurons);

    labels(1) = "Maximum neurons";
    values(1) = to_string(maximum_neurons);

    labels(2) = "NeuronsIncrement";
    values(2) = to_string(neurons_increment);

    labels(3) = "Trials number";
    values(3) = to_string(trials_number);

    labels(4) = "Selection loss goal";
    values(4) = to_string(selection_error_goal);

    labels(5) = "Maximum selection failures";
    values(5) = to_string(maximum_selection_failures);

    labels(6) = "Maximum iterations number";
    values(6) = to_string(maximum_epochs_number);

    labels(7) = "Maximum time";
    values(7) = to_string(maximum_time);

    const Index rows_number = labels.size();
    const Index raw_variables_number = 2;

    Tensor<string, 2> string_matrix(rows_number, raw_variables_number);

    string_matrix.chip(0, 1) = labels;
    string_matrix.chip(1, 1) = values;

    return string_matrix;
}


void GrowingNeurons::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("GrowingNeurons");

    // Minimum order

    file_stream.OpenElement("MinimumNeurons");
    file_stream.PushText(to_string(minimum_neurons).c_str());
    file_stream.CloseElement();

    // Maximum order

    file_stream.OpenElement("MaximumNeurons");
    file_stream.PushText(to_string(maximum_neurons).c_str());
    file_stream.CloseElement();

    // Step

    file_stream.OpenElement("NeuronsIncrement");
    file_stream.PushText(to_string(neurons_increment).c_str());
    file_stream.CloseElement();

    // Trials number

    file_stream.OpenElement("TrialsNumber");
    file_stream.PushText(to_string(trials_number).c_str());
    file_stream.CloseElement();

    // Selection error goal

    file_stream.OpenElement("SelectionErrorGoal");
    file_stream.PushText(to_string(selection_error_goal).c_str());
    file_stream.CloseElement();

    // Maximum selection failures

    file_stream.OpenElement("MaximumSelectionFailures");
    file_stream.PushText(to_string(maximum_selection_failures).c_str());
    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");
    file_stream.PushText(to_string(maximum_time).c_str());
    file_stream.CloseElement();

    file_stream.CloseElement();
}


void GrowingNeurons::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GrowingNeurons");

    if(!root_element)
        throw runtime_error("GrowingNeurons element is nullptr.\n");

    // Minimum neurons

    const tinyxml2::XMLElement* minimum_neurons_element = root_element->FirstChildElement("MinimumNeurons");

    if(minimum_neurons_element)
        minimum_neurons = Index(atoi(minimum_neurons_element->GetText()));

    // Maximum neurons

    const tinyxml2::XMLElement* maximum_neurons_element = root_element->FirstChildElement("MaximumNeurons");

    if(maximum_neurons_element)
        maximum_neurons = Index(atoi(maximum_neurons_element->GetText()));

    // Neurons increment

    const tinyxml2::XMLElement* neurons_increment_element = root_element->FirstChildElement("NeuronsIncrement");

    if(neurons_increment_element)
        set_neurons_increment(Index(atoi(neurons_increment_element->GetText())));

    // Trials number

    const tinyxml2::XMLElement* trials_number_element = root_element->FirstChildElement("TrialsNumber");

    if(trials_number_element)
        set_trials_number(Index(atoi(trials_number_element->GetText())));

    // Selection error goal

    const tinyxml2::XMLElement* selection_error_goal_element = root_element->FirstChildElement("SelectionErrorGoal");

    if(selection_error_goal_element)
        set_selection_error_goal(type(atof(selection_error_goal_element->GetText())));

    // Maximum selection failures

    const tinyxml2::XMLElement* maximum_selection_failures_element = root_element->FirstChildElement("MaximumSelectionFailures");

    if(maximum_selection_failures_element)
        set_maximum_selection_failures(Index(atoi(maximum_selection_failures_element->GetText())));

    // Maximum time

    const tinyxml2::XMLElement* maximum_time_element = root_element->FirstChildElement("MaximumTime");

    if(maximum_time_element)
        set_maximum_time(type(atoi(maximum_time_element->GetText())));
}


void GrowingNeurons::save(const string& file_name) const
{
    FILE * file = fopen(file_name.c_str(), "w");

    if(!file) return;

    tinyxml2::XMLPrinter printer(file);
    to_XML(printer);
    fclose(file);
}


void GrowingNeurons::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
        throw runtime_error("Cannot load XML file " + file_name + ".\n");

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
