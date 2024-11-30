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
    NeuronsSelectionResults neurons_selection_results(maximum_epochs_number);

    if(display) cout << "Performing growing neurons selection..." << endl;

    // Neural network    

    NeuralNetwork* neural_network = training_strategy->get_neural_network();

    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

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
        if(display) cout << endl << "Growing neurons epoch: " << epoch << endl;

        // Neural network

        neurons_number = minimum_neurons + epoch*neurons_increment;

        neural_network->get_layer(last_trainable_layer_index - 1).get()->set_output_dimensions({ neurons_number });
        neural_network->get_layer(last_trainable_layer_index).get()->set_input_dimensions({ neurons_number });

        neurons_selection_results.neurons_number_history(epoch) = neurons_number;

        // Loss index

        type minimum_training_error = numeric_limits<type>::max();
        type minimum_selection_error = numeric_limits<type>::max();

        for(Index trial = 0; trial < trials_number; trial++)
        {

            neural_network->set_parameters_random();

            training_results = training_strategy->perform_training();

            if(display)
                cout << "Trial: " << trial+1 << endl
                     << "Training error: " << training_results.get_training_error() << endl
                     << "Selection error: " << training_results.get_selection_error() << endl;

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

        if(neurons_selection_results.optimum_selection_error > previous_selection_error) 
            selection_failures++;

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

            if(display) cout << "Epoch " << epoch << endl <<  "Maximum epochs number reached: " << epoch << endl;

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

    neural_network->get_layer(last_trainable_layer_index - 1).get()->set_output_dimensions({ neurons_number });
    neural_network->get_layer(last_trainable_layer_index).get()->set_input_dimensions({ neurons_number });

    neural_network->set_parameters(neurons_selection_results.optimal_parameters);

    if(display) neurons_selection_results.print();

    return neurons_selection_results;
}


Tensor<string, 2> GrowingNeurons::to_string_matrix() const
{
    Tensor<string, 2> string_matrix(8, 2);

    string_matrix.setValues({
    {"Minimum neurons", to_string(minimum_neurons)},
    {"Maximum neurons", to_string(maximum_neurons)},
    {"NeuronsIncrement", to_string(neurons_increment)},
    {"Trials number", to_string(trials_number)},
    {"Selection loss goal", to_string(selection_error_goal)},
    {"Maximum selection failures", to_string(maximum_selection_failures)},
    {"Maximum iterations number", to_string(maximum_epochs_number)},
    {"Maximum time", to_string(maximum_time)}});

    return string_matrix;
}


void GrowingNeurons::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("GrowingNeurons");

    add_xml_element(printer, "MinimumNeurons", to_string(minimum_neurons));
    add_xml_element(printer, "MaximumNeurons", to_string(maximum_neurons));
    add_xml_element(printer, "NeuronsIncrement", to_string(neurons_increment));
    add_xml_element(printer, "TrialsNumber", to_string(trials_number));
    add_xml_element(printer, "SelectionErrorGoal", to_string(selection_error_goal));
    add_xml_element(printer, "MaximumSelectionFailures", to_string(maximum_selection_failures));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));

    printer.CloseElement();
}


void GrowingNeurons::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("GrowingNeurons");

    if(!root_element)
        throw runtime_error("GrowingNeurons element is nullptr.\n");

    minimum_neurons = read_xml_index(root_element, "MinimumNeurons");
    maximum_neurons = read_xml_index(root_element, "MaximumNeurons");
    set_neurons_increment(read_xml_index(root_element, "NeuronsIncrement"));
    set_trials_number(read_xml_index(root_element, "TrialsNumber"));
    set_selection_error_goal(read_xml_type(root_element, "SelectionErrorGoal"));
    set_maximum_selection_failures(read_xml_index(root_element, "MaximumSelectionFailures"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
}


void GrowingNeurons::save(const string& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    XMLPrinter printer;
    to_XML(printer);
    file << printer.CStr();
}


void GrowingNeurons::load(const string& file_name)
{
    set_default();

    XMLDocument document;

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
