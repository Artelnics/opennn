//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "growing_inputs.h"

namespace opennn
{

/// Default constructor.

GrowingInputs::GrowingInputs()
    : InputsSelection()
{
    set_default();
}


/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

GrowingInputs::GrowingInputs(TrainingStrategy* new_training_strategy_pointer)
    : InputsSelection(new_training_strategy_pointer)
{
    set_default();
}


/// Returns the maximum number of inputs in the growing inputs selection algorithm.

const Index& GrowingInputs::get_maximum_inputs_number() const
{
    return maximum_inputs_number;
}


/// Returns the minimum number of inputs in the growing inputs selection algorithm.

const Index& GrowingInputs::get_minimum_inputs_number() const
{
    return minimum_inputs_number;
}


/// Returns the maximum number of selection failures in the growing inputs selection algorithm.

const Index& GrowingInputs::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


/// Sets the members of the growing inputs object to their default values.

void GrowingInputs::set_default()
{
    maximum_selection_failures = 100;

    if(training_strategy_pointer == nullptr || !training_strategy_pointer->has_neural_network())
    {
        maximum_inputs_number = 100;
    }
    else
    {
        training_strategy_pointer->get_neural_network_pointer()->get_display();

        const Index inputs_number = training_strategy_pointer->get_data_set_pointer()->get_input_columns_number();

        maximum_selection_failures = 100;

        maximum_inputs_number = inputs_number;
    }

    minimum_inputs_number = 1;

    minimum_correlation = type(0);

    trials_number = 3;

    maximum_epochs_number = 1000;

    maximum_time = type(3600.0);
}


/// Sets the maximum inputs number for the growing inputs selection algorithm.
/// @param new_maximum_inputs_number Maximum inputs number in the growing inputs selection algorithm.

void GrowingInputs::set_maximum_inputs_number(const Index& new_maximum_inputs_number)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_inputs_number <= 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingInputs class.\n"
               << "void set_maximum_selection_failures(const Index&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    const Index inputs_number = training_strategy_pointer->get_data_set_pointer()->get_input_columns_number();

    maximum_inputs_number = min(new_maximum_inputs_number,inputs_number);
}


/// Sets the minimum inputs number for the growing inputs selection algorithm.
/// @param new_minimum_inputs_number Minimum inputs number in the growing inputs selection algorithm.

void GrowingInputs::set_minimum_inputs_number(const Index& new_minimum_inputs_number)
{
#ifdef OPENNN_DEBUG

    if(new_minimum_inputs_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingInputs class.\n"
               << "void set_minimum_inputs_number(const Index&) method.\n"
               << "Minimum inputs number must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    minimum_inputs_number = new_minimum_inputs_number;
}


/// Sets the maximum selection failures for the growing inputs selection algorithm.
/// @param new_maximum_selection_failures Maximum number of selection failures in the growing inputs selection algorithm.

void GrowingInputs::set_maximum_selection_failures(const Index& new_maximum_selection_failures)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_selection_failures <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingInputs class.\n"
               << "void set_maximum_selection_failures(const Index&) method.\n"
               << "Maximum selection failures must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    maximum_selection_failures = new_maximum_selection_failures;
}


/// Perform inputs selection with the growing inputs method.

InputsSelectionResults GrowingInputs::perform_inputs_selection()
{
#ifdef OPENNN_DEBUG

    check();

#endif

    InputsSelectionResults inputs_selection_results(maximum_epochs_number);

    if(display) cout << "Performing growing inputs selection..." << endl;

    // Loss index

    const LossIndex* loss_index_pointer = training_strategy_pointer->get_loss_index_pointer();

    type previus_selection_error = numeric_limits< type>::max();
    type previus_training_error = numeric_limits< type>::max();

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Tensor<Index, 1> target_columns_indices = data_set_pointer->get_target_columns_indices();

    const Index original_input_columns_number = data_set_pointer->get_input_columns_number();

    const Tensor<string, 1> columns_names = data_set_pointer->get_columns_names();

    Tensor<string, 1> input_columns_names;

    const Tensor<type, 2> correlations = get_correlation_values(data_set_pointer->calculate_input_target_columns_correlations());

    const Tensor<type, 1> total_correlations = correlations.abs().sum(rows_sum);

    Tensor<Index, 1> correlations_rank_descending = data_set_pointer->get_input_columns_indices();

    sort(correlations_rank_descending.data(),
         correlations_rank_descending.data() + correlations_rank_descending.size(),
         [&](Index i, Index j){return total_correlations[i] > total_correlations[j];});

    data_set_pointer->set_input_columns_unused();

    Index column_index = 0;

    // Neural network

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_neural_network_pointer();

    // Training strategy

    training_strategy_pointer->set_display(false);

    Index selection_failures = 0;

    TrainingResults training_results;

    // Model selection

    time_t beginning_time;
    time_t current_time;
    type elapsed_time = type(0);

    time(&beginning_time);

    bool stop = false;

    for(Index i = 0; i < maximum_epochs_number; i++)
    {
        data_set_pointer->set_column_use(correlations_rank_descending[column_index], DataSet::VariableUse::Input);

        Index input_columns_number = data_set_pointer->get_input_columns_number();
        Index input_variables_number = data_set_pointer->get_input_variables_number();

        if(input_columns_number >= minimum_inputs_number)
        {
            long long epoch = input_columns_number - minimum_inputs_number + 1;
            neural_network_pointer->set_inputs_number(input_variables_number);

            if(display)
            {
                cout << endl;
                cout << "Epoch: " << epoch << endl;
                cout << "Input columns number: " << input_columns_number << endl;
                cout << "Inputs: " << endl;

                input_columns_names = data_set_pointer->get_input_columns_names();

                for(Index j = 0; j < input_columns_number; j++) cout << "   " << input_columns_names(j) << endl;
            }

            type minimum_training_error = numeric_limits<type>::max();
            type minimum_selection_error = numeric_limits<type>::max();

            for(Index j = 0; j < trials_number; j++)
            {
                neural_network_pointer->set_parameters_random();

                if(data_set_pointer->has_nan())
                {
                    data_set_pointer->scrub_missing_values();
                }

                training_results = training_strategy_pointer->perform_training();

                if(training_results.get_selection_error() < minimum_selection_error)
                {
                    minimum_training_error = training_results.get_training_error();
                    minimum_selection_error = training_results.get_selection_error();

                    inputs_selection_results.training_error_history(input_columns_number-1) = minimum_training_error;
                    inputs_selection_results.selection_error_history(input_columns_number-1) = minimum_selection_error;
                }

                if(training_results.get_selection_error() < inputs_selection_results.optimum_selection_error)
                {
                    // Neural network

                    inputs_selection_results.optimal_input_columns_indices = data_set_pointer->get_input_columns_indices();
                    inputs_selection_results.optimal_input_columns_names = data_set_pointer->get_input_columns_names();

                    inputs_selection_results.optimal_parameters = neural_network_pointer->get_parameters();

                    // Loss index

                    inputs_selection_results.optimum_training_error = training_results.get_training_error();
                    inputs_selection_results.optimum_selection_error = training_results.get_selection_error();
                }

                if(display)
                {
                    cout << "Trial number: " << j+1 << endl;
                    cout << "   Training error: " << training_results.get_training_error() << endl;
                    cout << "   Selection error: " << training_results.get_selection_error() << endl;
                }
            }

            if(previus_training_error < minimum_training_error)
            {
                cout << "Selection failure" << endl;

                selection_failures++;

                data_set_pointer->set_column_use(correlations_rank_descending[column_index], DataSet::VariableUse::Unused);

                input_columns_number += -1;
            }
            else
            {
                previus_training_error = minimum_training_error;
                previus_selection_error = minimum_selection_error;

                inputs_selection_results.training_error_history(input_columns_number) = minimum_training_error;
                inputs_selection_results.selection_error_history(input_columns_number) = minimum_selection_error;
            }

            time(&current_time);

            elapsed_time = static_cast<type>(difftime(current_time,beginning_time));

            // Stopping criteria

            if(elapsed_time >= maximum_time)
            {
                stop = true;

                if(display) cout << "Epoch " << epoch << endl << "Maximum time reached: " << write_time(elapsed_time) << endl;

                inputs_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumTime;
            }
            else if(inputs_selection_results.optimum_selection_error <= selection_error_goal)
            {
                stop = true;

                if(display) cout << "\nSelection error reached: " << inputs_selection_results.optimum_selection_error << endl;

                inputs_selection_results.stopping_condition = InputsSelection::StoppingCondition::SelectionErrorGoal;
            }
            else if(epoch >= maximum_epochs_number)
            {
                stop = true;

                if(display) cout << "\nMaximum number of epochs reached." << endl;

                inputs_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumEpochs;
            }
            else if(selection_failures >= maximum_selection_failures)
            {
                stop = true;

                if(display) cout << "\nMaximum selection failures ("<<selection_failures<<") reached." << endl;

                inputs_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumSelectionFailures;
            }
            else if(input_columns_number >= maximum_inputs_number || input_columns_number >= original_input_columns_number)
            {
                stop = true;

                if(display) cout << "\nMaximum inputs (" << input_columns_number << ") reached." << endl;

                inputs_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
            }
            else if(column_index >= correlations_rank_descending.size() - 1 )
            {
                stop = true;

                if(display) cout << "\nAll the columns has been used." << endl;

                inputs_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
            }

            if(stop)
            {
                inputs_selection_results.elapsed_time = write_time(elapsed_time);

                inputs_selection_results.resize_history(input_columns_number);

                break;
            }
        }

        column_index++;

    }

    // Set data set stuff

    data_set_pointer->set_input_target_columns(inputs_selection_results.optimal_input_columns_indices, target_columns_indices);

    const Tensor<Scaler, 1> input_variables_scalers = data_set_pointer->get_input_variables_scalers();

    const Tensor<Descriptives, 1> input_variables_descriptives = data_set_pointer->calculate_input_variables_descriptives();

    set_maximum_inputs_number(data_set_pointer->get_input_columns_number());

    // Set neural network stuff

    neural_network_pointer->set_inputs_number(data_set_pointer->get_input_variables_number());

    neural_network_pointer->set_inputs_names(data_set_pointer->get_input_variables_names());

    if(neural_network_pointer->has_scaling_layer())
        neural_network_pointer->get_scaling_layer_pointer()->set(input_variables_descriptives, input_variables_scalers);

    neural_network_pointer->set_parameters(inputs_selection_results.optimal_parameters);

    if(display) inputs_selection_results.print();

    return inputs_selection_results;
}


/// This method writes a matrix of strings the most representative atributes.

Tensor<string, 2> GrowingInputs::to_string_matrix() const
{
    ostringstream buffer;

    Tensor<string, 1> labels(8);
    Tensor<string, 1> values(8);

    // Trials number

    labels(0) = "Trials number";

    buffer.str("");
    buffer << trials_number;

    values(0) = buffer.str();

    // Selection loss goal

    labels(1) = "Selection error goal";

    buffer.str("");
    buffer << selection_error_goal;

    values(1) = buffer.str();

    // Maximum selection failures

    labels(2) = "Maximum selection failures";

    buffer.str("");
    buffer << maximum_selection_failures;

    values(2) = buffer.str();

    // Maximum inputs number

    labels(3) = "Maximum inputs number";

    buffer.str("");
    buffer << maximum_inputs_number;

    values(3) = buffer.str();

    // Minimum correlation

    labels(4) = "Minimum correlations";

    buffer.str("");
    buffer << minimum_correlation;

    values(4) = buffer.str();

    // Maximum correlation

    labels(5) = "Maximum correlation";

    buffer.str("");
    buffer << maximum_correlation;

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


/// Serializes the growing inputs object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GrowingInputs::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("GrowingInputs");

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

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this growing inputs object.
/// @param document TinyXML document containing the member data.

void GrowingInputs::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GrowingInputs");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingInputs class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "GrowingInputs element is nullptr.\n";

        throw invalid_argument(buffer.str());
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
            catch(const invalid_argument& e)
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
            catch(const invalid_argument& e)
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
            const Index new_maximum_epochs_number = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_maximum_epochs_number(new_maximum_epochs_number);
            }
            catch(const invalid_argument& e)
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
            catch(const invalid_argument& e)
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
            const type new_maximum_time = static_cast<type>(atof(element->GetText()));

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
            catch(const invalid_argument& e)
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

}


/// Saves to an XML-type file the members of the growing inputs object.
/// @param file_name Name of growing inputs XML-type file.

void GrowingInputs::save(const string& file_name) const
{
    FILE * file = fopen(file_name.c_str(), "w");

    if(file)
    {
        tinyxml2::XMLPrinter printer(file);
        write_XML(printer);
        fclose(file);
    }
}


/// Loads a growing inputs object from an XML-type file.
/// @param file_name Name of growing inputs XML-type file.

void GrowingInputs::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GrowingInputs class.\n"
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
