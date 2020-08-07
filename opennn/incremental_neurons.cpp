//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N C R E M E N T A L   N E U R O N S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "incremental_neurons.h"

namespace OpenNN
{

/// Default constructor.

IncrementalNeurons::IncrementalNeurons()
    : NeuronsSelection()
{
    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a gradient descent object.

IncrementalNeurons::IncrementalNeurons(TrainingStrategy* new_training_strategy_pointer)
    : NeuronsSelection(new_training_strategy_pointer)
{
    set_default();
}


/// XML constructor.
/// @param incremental_neurons_document Pointer to a TinyXML document containing the incremental order data.

IncrementalNeurons::IncrementalNeurons(const tinyxml2::XMLDocument& incremental_neurons_document)
    : NeuronsSelection(incremental_neurons_document)
{
    from_XML(incremental_neurons_document);
}


/// File constructor.
/// @param file_name Name of XML incremental order file.

IncrementalNeurons::IncrementalNeurons(const string& file_name)
    : NeuronsSelection(file_name)
{
    load(file_name);
}


/// Destructor.

IncrementalNeurons::~IncrementalNeurons()
{
}


/// Returns the number of the hidden perceptrons pointed in each iteration of the Incremental algorithm.

const Index& IncrementalNeurons::get_step() const
{
    return step;
}


/// Returns the maximum number of selection failures in the model order selection algorithm.

const Index& IncrementalNeurons::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


/// Sets the members of the model selection object to their default values:

void IncrementalNeurons::set_default()
{
    step = 1;

    maximum_selection_failures = 10;
}


/// Sets the number of the hidden perceptrons pointed in each iteration of the Incremental algorithm
/// in the model order selection process.
/// @param new_step number of hidden perceptrons pointed.

void IncrementalNeurons::set_step(const Index& new_step)
{
#ifdef __OPENNN_DEBUG__

    if(new_step <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalNeurons class.\n"
               << "void set_step(const Index&) method.\n"
               << "New_step(" << new_step << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    step = new_step;
}


/// Sets the maximum selection failures for the Incremental order selection algorithm.
/// @param new_maximum_loss_failures Maximum number of selection failures in the Incremental order selection algorithm.

void IncrementalNeurons::set_maximum_selection_failures(const Index& new_maximum_loss_failures)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_loss_failures <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalNeurons class.\n"
               << "void set_maximum_selection_failures(const Index&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_loss_failures;
}


/// Perform the neurons selection with the Incremental method.

IncrementalNeurons::IncrementalNeuronsResults* IncrementalNeurons::perform_neurons_selection()
{
    IncrementalNeuronsResults* results = new IncrementalNeuronsResults();

    if(display)
    {
        cout << "Performing Incremental neurons selection..." << endl;
        cout.flush();
    }

    // Neural network

    NeuralNetwork* neural_network = training_strategy_pointer->get_neural_network_pointer();

    const Index trainable_layers_number = neural_network->get_trainable_layers_number();

    const Tensor<Layer*, 1> trainable_layers_pointers = neural_network->get_trainable_layers_pointers();

    // Loss index

    type prev_selection_error = numeric_limits<type>::max();

    Tensor<type, 1> optimal_parameters;

    type optimum_training_error = 0;
    type optimum_selection_error = 0;

    type current_training_loss = 0;
    type current_selection_error = 0;

    Tensor<type, 1> current_parameters;

    // Optimization algorithm

    Index optimal_neurons_number = 0;

    Index neurons_number = minimum_neurons;
    Index iterations = 0;
    Index selection_failures = 0;

    bool end = false;

    time_t beginning_time, current_time;
    type elapsed_time = 0;

    time(&beginning_time);

    // Main loop

    for(Index i = 0; i < maximum_neurons; i++)
    {
        // Set new neurons number

        trainable_layers_pointers(trainable_layers_number-2)->set_neurons_number(neurons_number);
        trainable_layers_pointers(trainable_layers_number-1)->set_inputs_number(neurons_number);
        results->neurons_data = insert_index_result(neurons_number, results->neurons_data);

        // Loss index

        type optimum_selection_error_trial = numeric_limits<type>::max();
        type optimum_training_error_trial = numeric_limits<type>::max();
        Tensor<type, 1> optimum_parameters_trial;

        for(Index i = 0; i < trials_number; i++)
        {
            neural_network->set_parameters_random();

            const OptimizationAlgorithm::Results optimization_algorithm_results
                    = training_strategy_pointer->perform_training();

            const type current_training_error_trial = optimization_algorithm_results.final_training_error;
            const type current_selection_error_trial = optimization_algorithm_results.final_selection_error;
            const Tensor<type, 1> current_parameters_trial = optimization_algorithm_results.final_parameters;

            if(current_selection_error_trial < optimum_selection_error_trial)
            {
                optimum_training_error_trial = current_training_error_trial;
                optimum_selection_error_trial = current_selection_error_trial;
                optimum_parameters_trial = current_parameters_trial;
            }

            if(display)
            {
                cout << "Trial number: " << i << endl;
                cout << "Training error: " << current_training_error_trial << endl;
                cout << "Selection error: " << current_selection_error_trial << endl;
                cout << "Stopping condition: " << optimization_algorithm_results.write_stopping_condition() << endl;
            }
        }

        current_training_loss = optimum_training_error_trial;
        current_selection_error = optimum_selection_error_trial;
        current_parameters = optimum_parameters_trial;

        time(&current_time);

        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        if(reserve_training_error_data)
        {
            results->training_error_data = insert_result(current_training_loss, results->training_error_data);
        }

        if(reserve_selection_error_data)
        {
            results->selection_error_data = insert_result(current_selection_error, results->selection_error_data);
        }

        if(iterations == 0
                ||(optimum_selection_error > current_selection_error
                   && abs(optimum_selection_error - current_selection_error) > tolerance))
        {
            optimal_neurons_number = neurons_number;
            optimum_training_error = current_training_loss;
            optimum_selection_error = current_selection_error;
            optimal_parameters = current_parameters;
        }
        else if(prev_selection_error < current_selection_error)
        {
            selection_failures++;
        }

        prev_selection_error = current_selection_error;
        iterations++;

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            end = true;

            if(display) cout << "Maximum time reached." << endl;

            results->stopping_condition = IncrementalNeurons::MaximumTime;
        }
        else if(current_selection_error <= selection_error_goal)
        {
            end = true;

            if(display) cout << "Selection loss reached." << endl;

            results->stopping_condition = IncrementalNeurons::SelectionErrorGoal;
        }
        else if(iterations >= maximum_epochs_number)
        {
            end = true;

            if(display) cout << "Maximum number of epochs reached." << endl;

            results->stopping_condition = IncrementalNeurons::MaximumIterations;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            end = true;

            if(display) cout << "Maximum selection failures (" << selection_failures << ") reached." << endl;

            results->stopping_condition = IncrementalNeurons::MaximumSelectionFailures;
        }
        else if(neurons_number == maximum_neurons)
        {
            end = true;

            if(display) cout << "Algorithm finished." << endl;

            results->stopping_condition = IncrementalNeurons::AlgorithmFinished;
        }

        if(display)
        {
            cout << "Iteration: " << iterations << endl
                 << "Hidden neurons number: " << neurons_number << endl
                 << "Training loss: " << current_training_loss << endl
                 << "Selection error: " << current_selection_error << endl
                 << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;
        }

        if(end) break;

        neurons_number += step;
    }

    if(display)
    {
        cout << endl
             << "Optimal order: " << optimal_neurons_number <<  endl
             << "Optimum selection error: " << optimum_selection_error << endl
             << "Corresponding training error: " << optimum_training_error << endl;
    }

    // Save neural network

    trainable_layers_pointers[trainable_layers_number-1]->set_inputs_number(optimal_neurons_number);
    trainable_layers_pointers[trainable_layers_number-2]->set_neurons_number(optimal_neurons_number);

    neural_network->set_parameters(optimal_parameters);

    // Save results

    if(reserve_minimal_parameters)
    {
        results->minimal_parameters = optimal_parameters;
    }

    results->optimal_neurons_number = optimal_neurons_number;
    results->final_selection_error = optimum_selection_error;
    results->final_training_error = optimum_training_error;
    results->iterations_number = iterations;
    results->elapsed_time = elapsed_time;

    return results;
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> IncrementalNeurons::to_string_matrix() const
{
    /*
        ostringstream buffer;

        Tensor<string, 1> labels;
        Tensor<string, 1> values;

       // Minimum order

       labels.push_back("Minimum order");

       buffer.str("");
       buffer << minimum_order;

       values.push_back(buffer.str());

       // Maximum order

       labels.push_back("Maximum order");

       buffer.str("");
       buffer << maximum_order;

       values.push_back(buffer.str());

       // Step

       labels.push_back("Step");

       buffer.str("");
       buffer << step;

       values.push_back(buffer.str());

       // Trials number

       labels.push_back("Trials number");

       buffer.str("");
       buffer << trials_number;

       values.push_back(buffer.str());

       // Tolerance

       labels.push_back("Tolerance");

       buffer.str("");
       buffer << tolerance;

       values.push_back(buffer.str());

       // Selection loss goal

       labels.push_back("Selection loss goal");

       buffer.str("");
       buffer << selection_error_goal;

       values.push_back(buffer.str());

       // Maximum selection failures

       labels.push_back("Maximum selection failures");

       buffer.str("");
       buffer << maximum_selection_failures;

       values.push_back(buffer.str());

       // Maximum iterations number

       labels.push_back("Maximum iterations number");

       buffer.str("");
       buffer << maximum_iterations_number;

       values.push_back(buffer.str());

       // Maximum time

       labels.push_back("Maximum time");

       buffer.str("");
       buffer << maximum_time;

       values.push_back(buffer.str());

       // Plot training error history

       labels.push_back("Plot training error history");

       buffer.str("");

       if(reserve_training_error_data)
       {
           buffer << "true";
       }
       else
       {
           buffer << "false";
       }

       values.push_back(buffer.str());

       // Plot selection error history

       labels.push_back("Plot selection error history");

       buffer.str("");

       if(reserve_selection_error_data)
       {
           buffer << "true";
       }
       else
       {
           buffer << "false";
       }

       values.push_back(buffer.str());

       const Index rows_number = labels.size();
       const Index columns_number = 2;

       Tensor<string, 2> string_matrix(rows_number, columns_number);

       string_matrix.set_column(0, labels, "name");
       string_matrix.set_column(1, values, "value");

        return string_matrix;
    */

    return Tensor<string, 2>();
}


/// Serializes the incremental order object into a XML document of the TinyXML library without
/// keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void IncrementalNeurons::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    //file_stream.OpenElement("IncrementalNeurons");

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
    buffer << step;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Trials number

    file_stream.OpenElement("TrialsNumber");

    buffer.str("");
    buffer << trials_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Tolerance

    file_stream.OpenElement("Tolerance");

    buffer.str("");
    buffer << tolerance;

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

    // Reserve training erro history

    file_stream.OpenElement("ReserveTrainingErrorHistory");

    buffer.str("");
    buffer << reserve_training_error_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection error data

    file_stream.OpenElement("ReserveSelectionErrorHistory");

    buffer.str("");
    buffer << reserve_selection_error_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this incremental order object.
/// @param document TinyXML document containing the member data.

void IncrementalNeurons::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("IncrementalNeurons");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalNeurons class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "IncrementalNeurons element is nullptr.\n";

        throw logic_error(buffer.str());
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
            catch(const logic_error& e)
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
            catch(const logic_error& e)
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
                set_step(new_step);
            }
            catch(const logic_error& e)
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
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Tolerance
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Tolerance");

        if(element)
        {
            const Index new_tolerance = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_tolerance(new_tolerance);
            }
            catch(const logic_error& e)
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
            catch(const logic_error& e)
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
            catch(const logic_error& e)
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
            const type new_maximum_time = atoi(element->GetText());

            try
            {
                set_maximum_time(new_maximum_time);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }


    // Reserve training error history
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveTrainingErrorHistory");

        if(element)
        {
            const string new_reserve_training_error_data = element->GetText();

            try
            {
                set_reserve_training_error_data(new_reserve_training_error_data != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve selection error history
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

        if(element)
        {
            const string new_reserve_selection_error_data = element->GetText();

            try
            {
                set_reserve_selection_error_data(new_reserve_selection_error_data != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

}


/// Saves to a XML-type file the members of the incremental order object.
/// @param file_name Name of incremental order XML-type file.

void IncrementalNeurons::save(const string& file_name) const
{
//    tinyxml2::XMLDocument* document = to_XML();

//    document->SaveFile(file_name.c_str());

//    delete document;
}


/// Loads a incremental order object from a XML-type file.
/// @param file_name Name of incremental order XML-type file.

void IncrementalNeurons::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalNeurons class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw logic_error(buffer.str());
    }

    from_XML(document);
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
