//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   N E U R O N S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "neural_network.h"
#include "optimizer.h"
#include "training_strategy.h"
#include "growing_neurons.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

GrowingNeurons::GrowingNeurons(TrainingStrategy* new_training_strategy)
    : NeuronSelection(new_training_strategy)
{
    set_default();
}

void GrowingNeurons::set_default()
{
    name = "GrowingNeurons";

    minimum_neurons = 1;
    maximum_neurons = 10;
    trials_number = 3;
    neurons_increment = 1;
    maximum_validation_failures = 100;
    maximum_time = type(3600);
}

void GrowingNeurons::set_neurons_increment(const Index new_neurons_increment)
{
    neurons_increment = new_neurons_increment;
}

NeuronsSelectionResults GrowingNeurons::perform_neurons_selection()
{
    NeuronsSelectionResults neuron_selection_results(maximum_epochs);

    if(display) cout << "Performing growing neuron selection..." << "\n";

    // Neural network    

    NeuralNetwork* neural_network = training_strategy->get_neural_network();

    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    Index neurons_number = 0;

    // Loss index

    type previous_validation_error = MAX;

    // Optimization algorithm

    Index validation_failures = 0;

    bool end = false;

    time_t beginning_time;
    time_t current_time;

    type elapsed_time = type(0);

    TrainingResults training_results;

    time(&beginning_time);

    // Main loop

    for(Index epoch = 0; epoch < maximum_epochs; ++epoch)
    {
        if(display) cout << "\nGrowing neurons epoch: " << epoch << "\n";

        // Neural network

        neurons_number = minimum_neurons + epoch*neurons_increment;

        neural_network->get_layer(last_trainable_layer_index - 1)->set_output_shape({ neurons_number });
        neural_network->get_layer(last_trainable_layer_index)->set_input_shape({ neurons_number });

        //neural_network->print();
        // throw runtime_error("Checking the network");

        neuron_selection_results.neurons_number_history(epoch) = neurons_number;

        // Loss index

        type minimum_training_error = MAX;
        type minimum_validation_error = MAX;

        for(Index trial = 0; trial < trials_number; ++trial)
        {
            neural_network->set_parameters_random();

            training_results = training_strategy->train();

            if(display)
                cout << "Trial: " << trial+1 << "\n"
                     << "Training error: " << training_results.get_training_error() << "\n"
                     << "Validation error: " << training_results.get_validation_error() << "\n";

            if(training_results.get_validation_error() < minimum_validation_error)
            {
                minimum_training_error = training_results.get_training_error();
                minimum_validation_error = training_results.get_validation_error();

                neuron_selection_results.training_error_history(epoch) = minimum_training_error;
                neuron_selection_results.validation_error_history(epoch) = minimum_validation_error;
            }

            if(minimum_validation_error < neuron_selection_results.optimum_validation_error)
            {
                neuron_selection_results.optimal_neurons_number = neurons_number;
                //neural_network->get_parameters(neuron_selection_results.optimal_parameters);
                neuron_selection_results.optimum_training_error = minimum_training_error;
                neuron_selection_results.optimum_validation_error = minimum_validation_error;                                
            }
        }

        if(display)
            cout << "Neurons number: " << neurons_number << "\n"
                 << "Training error: " << training_results.get_training_error() << "\n"
                 << "Validation error: " << training_results.get_validation_error() << "\n"
                 << "Elapsed time: " << get_time(elapsed_time) << "\n";

        if(neuron_selection_results.optimum_validation_error > previous_validation_error) 
            ++validation_failures;

        previous_validation_error = neuron_selection_results.optimum_validation_error;

        time(&current_time);

        elapsed_time = type(difftime(current_time,beginning_time));

        // Stopping criteria

        end = true;

        if(elapsed_time >= maximum_time)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum time reached: " << get_time(elapsed_time) << "\n";
            neuron_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::MaximumTime;
        }
        else if(training_results.get_validation_error() <= validation_error_goal)
        {
            if(display) cout << "Epoch " << epoch << "\nSelection error goal reached: " << training_results.get_validation_error() << "\n";
            neuron_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::SelectionErrorGoal;
        }
        else if(epoch >= maximum_epochs)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch << "\n";
            neuron_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::MaximumEpochs;
        }
        else if(validation_failures >= maximum_validation_failures)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << validation_failures << "\n";
            neuron_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::MaximumSelectionFailures;
        }
        else if(neurons_number >= maximum_neurons)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum number of neurons reached: " << neurons_number << "\n";
            neuron_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::MaximumNeurons;
        }
        else
        {
            end = false;
        }

        if(end)
        {
            neuron_selection_results.elapsed_time = get_time(elapsed_time);

            neuron_selection_results.resize_history(epoch+1);

            break;
        }
    }

    // Save neural network

    cout << "Parameters number: " << neuron_selection_results.optimal_parameters.size() << "\n";

    neural_network->get_layer(last_trainable_layer_index - 1)->set_output_shape({ neuron_selection_results.optimal_neurons_number });
    neural_network->get_layer(last_trainable_layer_index)->set_input_shape({ neuron_selection_results.optimal_neurons_number });
    neural_network->set_parameters(neuron_selection_results.optimal_parameters);

    if(display) neuron_selection_results.print();

    return neuron_selection_results;
}

void GrowingNeurons::to_XML(XmlPrinter& printer) const
{
    printer.open_element("GrowingNeurons");

    write_xml_properties(printer, {
        {"MinimumNeurons", to_string(minimum_neurons)},
        {"MaximumNeurons", to_string(maximum_neurons)},
        {"NeuronsIncrement", to_string(neurons_increment)},
        {"TrialsNumber", to_string(trials_number)},
        {"SelectionErrorGoal", to_string(validation_error_goal)},
        {"MaximumSelectionFailures", to_string(maximum_validation_failures)},
        {"MaximumTime", to_string(maximum_time)}
    });

    printer.close_element();
}

void GrowingNeurons::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "GrowingNeurons");

    set_minimum_neurons(read_xml_index(root_element, "MinimumNeurons"));
    set_maximum_neurons(read_xml_index(root_element, "MaximumNeurons"));
    set_neurons_increment(read_xml_index(root_element, "NeuronsIncrement"));
    set_trials_number(read_xml_index(root_element, "TrialsNumber"));
    set_validation_error_goal(read_xml_type(root_element, "SelectionErrorGoal"));
    set_maximum_validation_failures(read_xml_index(root_element, "MaximumSelectionFailures"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
}

REGISTER(NeuronSelection, GrowingNeurons, "GrowingNeurons");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
