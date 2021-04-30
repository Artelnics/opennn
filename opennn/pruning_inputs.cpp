//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R U N I N G   I N P U T S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pruning_inputs.h"

namespace OpenNN
{

/// Default constructor.

PruningInputs::PruningInputs()
    : InputsSelection()
{
    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

PruningInputs::PruningInputs(TrainingStrategy* new_training_strategy_pointer)
    : InputsSelection(new_training_strategy_pointer)
{
    set_default();
}


/// Destructor.

PruningInputs::~PruningInputs()
{
}


/// Returns the minimum number of inputs in the pruning inputs selection algorithm.

const Index& PruningInputs::get_minimum_inputs_number() const
{
    return minimum_inputs_number;
}


/// Returns the maximum number of inputs in the pruning inputs selection algorithm.

const Index& PruningInputs::get_maximum_inputs_number() const
{
    return maximum_inputs_number;
}


/// Returns the maximum number of selection failures in the pruning inputs algorithm.

const Index& PruningInputs::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


/// Sets the members of the pruning inputs object to their default values.

void PruningInputs::set_default()
{
    Index inputs_number;

    if(training_strategy_pointer == nullptr || !training_strategy_pointer->has_neural_network())
    {
        maximum_selection_failures = 100;

        maximum_inputs_number = 20;
    }
    else
    {
        inputs_number = training_strategy_pointer->get_neural_network_pointer()->get_inputs_number();
        maximum_selection_failures = 100;//static_cast<Index>(max(3.,inputs_number/5.));

        maximum_inputs_number = inputs_number;
    }

    minimum_inputs_number = 1;

    minimum_correlation = 0.0;

    trials_number = 3;

    maximum_epochs_number = 1000;

    maximum_time = 3600.0;
}


/// Sets the minimum inputs for the pruning inputs algorithm.
/// @param new_minimum_inputs_number Minimum number of inputs in the pruning inputs algorithm.

void PruningInputs::set_minimum_inputs_number(const Index& new_minimum_inputs_number)
{
#ifdef OPENNN_DEBUG

    if(new_minimum_inputs_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void set_minimum_inputs_number(const Index&) method.\n"
               << "Minimum inputs number must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_inputs_number = new_minimum_inputs_number;
}


/// Sets the maximum inputs for the pruning inputs algorithm.
/// @param new_maximum_inputs_number Maximum number of inputs in the pruning inputs algorithm.

void PruningInputs::set_maximum_inputs_number(const Index& new_maximum_inputs_number)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_inputs_number <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void set_maximum_inputs_number(const Index&) method.\n"
               << "Maximum inputs number must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_inputs_number = new_maximum_inputs_number;
}


/// Sets the maximum selection failures for the pruning inputs algorithm.
/// @param new_maximum_selection_failures Maximum number of selection failures in the pruning inputs algorithm.

void PruningInputs::set_maximum_selection_failures(const Index& new_maximum_selection_failures)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_selection_failures <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void set_maximum_selection_failures(const Index&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_selection_failures;
}


/// Perform the inputs selection with the pruning inputs method.

InputsSelectionResults PruningInputs::perform_inputs_selection()
{

#ifdef OPENNN_DEBUG

    check();

#endif

    InputsSelectionResults results(maximum_epochs_number);

    if(display) cout << "Performing pruning inputs selection..." << endl;

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    type previus_selection_error = numeric_limits<type>::max();

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Tensor<Index, 1> target_columns_indices = data_set_pointer->get_target_columns_indices();

    const Tensor<string, 1> columns_names = data_set_pointer->get_columns_names();   

    Tensor<string, 1> input_columns_names;

    const Tensor<type, 2> correlations = data_set_pointer->calculate_input_target_columns_correlations_values();

    const Tensor<type, 1> total_correlations = correlations.abs().sum(rows_sum);

    Tensor<Index, 1> correlations_rank_ascending = data_set_pointer->get_input_columns_indices();

    sort(correlations_rank_ascending.data(),
         correlations_rank_ascending.data() + correlations_rank_ascending.size(),
         [&](Index i, Index j){return total_correlations[i] < total_correlations[j];});

    // Neural network

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_neural_network_pointer();

    // Training strategy

    training_strategy_pointer->set_display(false);

    Index selection_failures = 0;

    TrainingResults training_results;

    // Model selection

    time_t beginning_time, current_time;
    type elapsed_time = 0;

    time(&beginning_time);

    bool stop = false;

    for(Index epoch = 0; epoch < maximum_epochs_number; epoch++)
    {
        if(epoch > 0) data_set_pointer->set_column_use(correlations_rank_ascending[epoch], DataSet::UnusedVariable);

        const Index input_columns_number = data_set_pointer->get_input_columns_number();
        const Index input_variables_number = data_set_pointer->get_input_variables_number();

        neural_network_pointer->set_inputs_number(input_variables_number);

        if(display)
        {
            cout << endl;
            cout << "Epoch: " << epoch << endl;
            cout << "Input columns number: " << input_columns_number << endl;
            cout << "Inputs: " << endl;

            input_columns_names = data_set_pointer->get_input_columns_names();

            for(Index i = 0; i < input_columns_number; i++) cout << "   " << input_columns_names(i) << endl;
        }

        // Trial

        for(Index trial = 0; trial < trials_number; trial++)
        {
            neural_network_pointer->set_parameters_random();

            training_results = training_strategy_pointer->perform_training();

//            if(training_results.final_selection_error < results.optimum_selection_error)
            {
                results.optimal_input_columns_indices = data_set_pointer->get_input_columns_indices();
                results.optimal_input_columns_names = data_set_pointer->get_input_columns_names();

                // Neural network

//                results.optimal_parameters = training_results.parameters;

                // Loss index

//                results.optimum_training_error = training_results.final_training_error;
//                results.optimum_selection_error = training_results.final_selection_error;
            }

            if(display)
            {
                cout << "Trial number: " << trial+1 << endl;
//                cout << "   Training error: " << training_results.final_training_error << endl;
//                cout << "   Selection error: " << training_results.final_selection_error << endl;
            }
        }

        if(results.optimum_selection_error >= previus_selection_error) selection_failures++;

        previus_selection_error = results.optimum_selection_error;

//        if(reserve_training_errors)results.training_errors(epoch) = training_results.final_training_error;

//        if(reserve_selection_errors) results.selection_errors(epoch) = training_results.final_selection_error;

        time(&current_time);

        elapsed_time = static_cast<type>( difftime(current_time,beginning_time));

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            stop = true;

            if(display) cout << "Maximum time reached." << endl;

            results.stopping_condition = InputsSelection::MaximumTime;
        }
        else if(results.optimum_selection_error <= selection_error_goal)
        {
            stop = true;

            if(display) cout << "Selection loss reached." << endl;

            results.stopping_condition = InputsSelection::SelectionErrorGoal;
        }
        else if(epoch >= maximum_epochs_number)
        {
            stop = true;

            if(display) cout << "Maximum number of epochs reached." << endl;

            results.stopping_condition = InputsSelection::MaximumEpochs;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            stop = true;

            if(display) cout << "Maximum selection failures (" << selection_failures << ") reached." << endl;

            results.stopping_condition = InputsSelection::MaximumSelectionFailures;
        }
        else if(input_columns_number <= minimum_inputs_number
             || input_columns_number == 1)
        {
            stop = true;

            if(display) cout << "Minimum inputs (" << minimum_inputs_number << ") reached." << endl;

            results.stopping_condition = InputsSelection::MinimumInputs;
        }

        if(stop)
        {
            // Save results

            results.epochs_number = epoch + 1;
            results.elapsed_time = write_elapsed_time(elapsed_time);

            break;
        }
    }

    // Set data set stuff

    data_set_pointer->set_input_target_columns(results.optimal_input_columns_indices, target_columns_indices);

    const Tensor<Scaler, 1> input_variables_scalers = data_set_pointer->get_input_variables_scalers();

    const Tensor<Descriptives, 1> input_variables_descriptives =  data_set_pointer->scale_input_variables();

    // Set neural network stuff

    neural_network_pointer->set_inputs_number(data_set_pointer->get_input_variables_number());

    neural_network_pointer->set_inputs_names(data_set_pointer->get_input_variables_names());

    if(neural_network_pointer->has_scaling_layer())
        neural_network_pointer->get_scaling_layer_pointer()->set(input_variables_descriptives, input_variables_scalers);

    neural_network_pointer->set_parameters(results.optimal_parameters);

    if(display) results.print();

    return results;
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> PruningInputs::to_string_matrix() const
{
    ostringstream buffer;

    Tensor<string, 1> labels(11);
    Tensor<string, 1> values(11);

    // Trials number

    labels(0) = "Trials number";

    buffer.str("");
    buffer << trials_number;

    values(0) = buffer.str();

    // Selection loss goal

    labels(2) = "Selection loss goal";

    buffer.str("");
    buffer << selection_error_goal;

    values(2) = buffer.str();

    // Maximum selection failures

    labels(3) = "Maximum selection failures";

    buffer.str("");
    buffer << maximum_selection_failures;

    values(3) = buffer.str();

    // Minimum inputs number

    labels(4) = "Minimum inputs number";

    buffer.str("");
    buffer << minimum_inputs_number;

    values(4) = buffer.str();

    // Minimum correlation

    labels(5) = "Minimum correlation";

    buffer.str("");
    buffer << minimum_correlation;

    values(5) = buffer.str();

    // Maximum correlation

    labels(6) = "Maximum correlation";

    buffer.str("");
    buffer << maximum_correlation;

    values(6) = buffer.str();

    // Maximum iterations number

    labels(7) = "Maximum iterations number";

    buffer.str("");
    buffer << maximum_epochs_number;

    values(7) = buffer.str();

    // Maximum time

    labels(8) = "Maximum time";

    buffer.str("");
    buffer << maximum_time;

    values(8) = buffer.str();

    // Plot training loss history

    labels(9) = "Plot training loss history";

    buffer.str("");

    if(reserve_training_errors)
    {
        buffer << "true";
    }
    else
    {
        buffer << "false";
    }

    values(9) = buffer.str();

    // Plot selection error history

    labels(10) = "Plot selection error hitory";

    buffer.str("");

    if(reserve_selection_errors)
    {
        buffer << "true";
    }
    else
    {
        buffer << "false";
    }

    values(10) = buffer.str();

    const Index rows_number = labels.size();
    const Index columns_number = 2;

    Tensor<string, 2> string_matrix(rows_number, columns_number);

    string_matrix.chip(0, 1) = labels;
    string_matrix.chip(1, 1) = values;

     return string_matrix;
}


/// Serializes the pruning inputs object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void PruningInputs::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("PruningInputs");

    // Trials number

    file_stream.OpenElement("TrialsNumber");

    buffer.str("");
    buffer << trials_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // selection error goal

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

    // Minimum inputs number

    file_stream.OpenElement("MinimumInputsNumber");

    buffer.str("");
    buffer << minimum_inputs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum inputs number

    file_stream.OpenElement("MaximumInputsNumber");

    buffer.str("");
    buffer << maximum_inputs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum correlation

    file_stream.OpenElement("MinimumCorrelation");

    buffer.str("");
    buffer << minimum_correlation;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum correlation

    file_stream.OpenElement("MaximumCorrelation");

    buffer.str("");
    buffer << maximum_correlation;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum iterations

    file_stream.OpenElement("MaximumEpochsNumber");

    buffer.str("");
    buffer << maximum_epochs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve loss data

    file_stream.OpenElement("ReserveTrainingErrorHistory");

    buffer.str("");
    buffer << reserve_training_errors;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection error data

    file_stream.OpenElement("ReserveSelectionErrorHistory");

    buffer.str("");
    buffer << reserve_selection_errors;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this pruning inputs object.
/// @param document TinyXML document containing the member data.

void PruningInputs::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("PruningInputs");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "PruningInputs element is nullptr.\n";

        throw logic_error(buffer.str());
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

    // Reserve loss data
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

    // Reserve selection error data
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

    // Display
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

        if(element)
        {
            const string new_display = element->GetText();

            try
            {
                set_display(new_display != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // selection error goal
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

    // Maximum iterations number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumEpochsNumber");

        if(element)
        {
            const Index new_maximum_iterations_number = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_maximum_iterations_number(new_maximum_iterations_number);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum correlation
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumCorrelation");

        if(element)
        {
            const type new_maximum_correlation = static_cast<type>(atof(element->GetText()));

            try
            {
                set_maximum_correlation(new_maximum_correlation);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Minimum correlation
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumCorrelation");

        if(element)
        {
            const type new_minimum_correlation = static_cast<type>(atof(element->GetText()));

            try
            {
                set_minimum_correlation(new_minimum_correlation);
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
            const type new_maximum_time = static_cast<type>( atoi(element->GetText()));

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

    // Minimum inputs number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumInputsNumber");

        if(element)
        {
            const Index new_minimum_inputs_number = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_minimum_inputs_number(new_minimum_inputs_number);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum inputs number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumInputsNumber");

        if(element)
        {
            const Index new_maximum_inputs_number = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_maximum_inputs_number(new_maximum_inputs_number);
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
}


/// Saves to a XML-type file the members of the pruning inputs object.
/// @param file_name Name of pruning inputs XML-type file.

void PruningInputs::save(const string& file_name) const
{
    FILE * file = fopen(file_name.c_str(), "w");

    tinyxml2::XMLPrinter printer(file);

    write_XML(printer);

    fclose(file);
}


/// Loads a pruning inputs object from a XML-type file.
/// @param file_name Name of pruning inputs XML-type file.

void PruningInputs::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PruningInputs class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw logic_error(buffer.str());
    }

    from_XML(document);
}
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
