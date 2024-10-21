//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "growing_inputs.h"
#include "correlations.h"

namespace opennn
{

GrowingInputs::GrowingInputs()
    : InputsSelection()
{
    set_default();
}


GrowingInputs::GrowingInputs(TrainingStrategy* new_training_strategy)
    : InputsSelection(new_training_strategy)
{
    set_default();
}


const Index& GrowingInputs::get_maximum_inputs_number() const
{
    return maximum_inputs_number;
}


const Index& GrowingInputs::get_minimum_inputs_number() const
{
    return minimum_inputs_number;
}


const Index& GrowingInputs::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


void GrowingInputs::set_default()
{
    maximum_selection_failures = 100;

    if(training_strategy == nullptr || !training_strategy->has_neural_network())
    {
        maximum_inputs_number = 100;
    }
    else
    {
        training_strategy->get_neural_network()->get_display();

        const Index inputs_number = training_strategy->get_data_set()->get_input_raw_variables_number();

        maximum_selection_failures = 100;

        maximum_inputs_number = inputs_number;
    }

    minimum_inputs_number = 1;

    minimum_correlation = type(0);

    trials_number = 3;

    maximum_epochs_number = 1000;

    maximum_time = type(3600.0);
}


void GrowingInputs::set_maximum_inputs_number(const Index& new_maximum_inputs_number)
{
    const Index inputs_number = training_strategy->get_data_set()->get_input_raw_variables_number();

    maximum_inputs_number = min(new_maximum_inputs_number,inputs_number);
}


void GrowingInputs::set_minimum_inputs_number(const Index& new_minimum_inputs_number)
{
    minimum_inputs_number = new_minimum_inputs_number;
}


void GrowingInputs::set_maximum_selection_failures(const Index& new_maximum_selection_failures)
{
    maximum_selection_failures = new_maximum_selection_failures;
}


InputsSelectionResults GrowingInputs::perform_inputs_selection()
{
#ifdef OPENNN_DEBUG

    check();

#endif

    InputsSelectionResults inputs_selection_results(maximum_epochs_number);

    if(display) cout << "Performing growing inputs selection..." << endl;

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();

//    type previus_selection_error = numeric_limits< type>::max();
    type previus_training_error = numeric_limits< type>::max();

    // Data set

    DataSet* data_set = loss_index->get_data_set();

    const Tensor<Index, 1> target_raw_variables_indices = data_set->get_target_raw_variables_indices();

    const Index original_input_raw_variables_number = data_set->get_input_raw_variables_number();

    const Tensor<string, 1> raw_variables_names = data_set->get_raw_variable_names();

    Tensor<string, 1> input_raw_variables_names;

    const Tensor<type, 2> correlations = get_correlation_values(data_set->calculate_input_target_raw_variable_pearson_correlations());

    const Tensor<type, 1> total_correlations = correlations.abs().chip(0,1);

    Tensor<Index, 1> correlations_indexes(original_input_raw_variables_number);

    initialize_sequential(correlations_indexes);

    sort(correlations_indexes.data(),
         correlations_indexes.data() + correlations_indexes.size(),
         [&](Index i, Index j){return total_correlations[i] > total_correlations[j];});

    Tensor<Index, 1> input_raw_variables_indices = data_set->get_input_raw_variables_indices();

    Tensor<Index, 1> correlations_rank_descending(input_raw_variables_indices.size());

    for(Index i = 0; i < correlations_rank_descending.size(); i++) correlations_rank_descending(i) = input_raw_variables_indices(correlations_indexes[i]);

    data_set->set_input_raw_variables_unused();

    Index raw_variable_index = 0;

    // Neural network

    NeuralNetwork* neural_network = training_strategy->get_neural_network();

    // Training strategy

    training_strategy->set_display(false);

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
        data_set->set_raw_variable_use(correlations_rank_descending[raw_variable_index], DataSet::VariableUse::Input);

        Index input_raw_variables_number = data_set->get_input_raw_variables_number();
        Index input_variables_number = data_set->get_input_variables_number();

        if(input_raw_variables_number >= minimum_inputs_number)
        {
            long long epoch = input_raw_variables_number - minimum_inputs_number + 1;
            neural_network->set_inputs_number(input_variables_number);

            if(display)
            {
                cout << endl;
                cout << "Epoch: " << epoch << endl;
                cout << "Input raw_variables number: " << input_raw_variables_number << endl;
                cout << "Inputs: " << endl;

                input_raw_variables_names = data_set->get_input_raw_variable_names();

                for(Index j = 0; j < input_raw_variables_number; j++) cout << "   " << input_raw_variables_names(j) << endl;
            }

            type minimum_training_error = numeric_limits<type>::max();
            type minimum_selection_error = numeric_limits<type>::max();

            for(Index j = 0; j < trials_number; j++)
            {
                neural_network->set_parameters_random();

                if(data_set->has_nan())
                {
                    data_set->scrub_missing_values();
                }

                training_results = training_strategy->perform_training();

                if(training_results.get_selection_error() < minimum_selection_error)
                {
                    minimum_training_error = training_results.get_training_error();
                    minimum_selection_error = training_results.get_selection_error();

                    inputs_selection_results.training_error_history(input_raw_variables_number-1) = minimum_training_error;
                    inputs_selection_results.selection_error_history(input_raw_variables_number-1) = minimum_selection_error;
                }

                if(training_results.get_selection_error() < inputs_selection_results.optimum_selection_error)
                {
                    // Neural network

                    inputs_selection_results.optimal_input_raw_variables_indices = data_set->get_input_raw_variables_indices();
                    inputs_selection_results.optimal_input_raw_variables_names = data_set->get_input_raw_variable_names();

                    inputs_selection_results.optimal_parameters = neural_network->get_parameters();

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

                data_set->set_raw_variable_use(correlations_rank_descending[raw_variable_index], DataSet::VariableUse::None);

                input_raw_variables_number += -1;
            }
            else
            {
                previus_training_error = minimum_training_error;
//                previus_selection_error = minimum_selection_error;

                inputs_selection_results.training_error_history(input_raw_variables_number) = minimum_training_error;
                inputs_selection_results.selection_error_history(input_raw_variables_number) = minimum_selection_error;
            }

            time(&current_time);

            elapsed_time = type(difftime(current_time,beginning_time));

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
            else if(input_raw_variables_number >= maximum_inputs_number || input_raw_variables_number >= original_input_raw_variables_number)
            {
                stop = true;

                if(display) cout << "\nMaximum inputs (" << input_raw_variables_number << ") reached." << endl;

                inputs_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
            }
            else if(raw_variable_index >= correlations_rank_descending.size() - 1 )
            {
                stop = true;

                if(display) cout << "\nAll the raw_variables has been used." << endl;

                inputs_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
            }

            if(stop)
            {
                inputs_selection_results.elapsed_time = write_time(elapsed_time);

                inputs_selection_results.resize_history(input_raw_variables_number);

                break;
            }
        }

        raw_variable_index++;

    }

    // Set data set stuff

    data_set->set_input_target_raw_variables_indices(inputs_selection_results.optimal_input_raw_variables_indices, target_raw_variables_indices);

    const Tensor<Scaler, 1> input_variables_scalers = data_set->get_input_variables_scalers();

    const Tensor<Descriptives, 1> input_variables_descriptives = data_set->calculate_input_variables_descriptives();

    set_maximum_inputs_number(data_set->get_input_raw_variables_number());

    // Set neural network stuff

    neural_network->set_inputs_number(data_set->get_input_variables_number());

    neural_network->set_inputs_names(data_set->get_input_variables_names());

    if(neural_network->has_scaling_layer_2d())
    {
        ScalingLayer2D* scaling_layer_2d =   neural_network->get_scaling_layer_2d();
        scaling_layer_2d->set_descriptives(input_variables_descriptives);
        scaling_layer_2d->set_scalers(input_variables_scalers);
    }

    neural_network->set_parameters(inputs_selection_results.optimal_parameters);

    if(display) inputs_selection_results.print();

    return inputs_selection_results;
}


Tensor<string, 2> GrowingInputs::to_string_matrix() const
{
    Tensor<string, 1> labels(8);
    Tensor<string, 1> values(8);

    labels(0) = "Trials number";
    values(0) = to_string(trials_number);

    labels(1) = "Selection error goal";
    values(1) = to_string(selection_error_goal);

    labels(2) = "Maximum selection failures";
    values(2) = to_string(maximum_selection_failures);

    labels(3) = "Maximum inputs number";
    values(3) = to_string(maximum_inputs_number);

    labels(4) = "Minimum correlations";
    values(4) = to_string(minimum_correlation);

    labels(5) = "Maximum correlation";
    values(5) = to_string(maximum_correlation);

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


void GrowingInputs::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("GrowingInputs");

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

    // Minimum inputs number

    file_stream.OpenElement("MinimumInputsNumber");
    file_stream.PushText(to_string(minimum_inputs_number).c_str());
    file_stream.CloseElement();

    // Maximum inputs number

    file_stream.OpenElement("MaximumInputsNumber");
    file_stream.PushText(to_string(maximum_inputs_number).c_str());
    file_stream.CloseElement();

    // Minimum correlation

    file_stream.OpenElement("MinimumCorrelation");
    file_stream.PushText(to_string(minimum_correlation).c_str());
    file_stream.CloseElement();

    // Maximum correlation

    file_stream.OpenElement("MaximumCorrelation");
    file_stream.PushText(to_string(maximum_correlation).c_str());
    file_stream.CloseElement();

    // Maximum iterations

    file_stream.OpenElement("MaximumEpochsNumber");
    file_stream.PushText(to_string(maximum_epochs_number).c_str());
    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");
    file_stream.PushText(to_string(maximum_time).c_str());
    file_stream.CloseElement();

    file_stream.CloseElement();
}


void GrowingInputs::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GrowingInputs");

    if(!root_element)
        throw runtime_error("GrowingInputs element is nullptr.\n");

    // Trials number

    const tinyxml2::XMLElement* trials_number_element = root_element->FirstChildElement("TrialsNumber");

    if(trials_number_element)
        set_trials_number(Index(atoi(trials_number_element->GetText())));

    // Display

    const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

    if(display_element)
        set_display(display_element->GetText() != string("0"));

    // Selection error goal

    const tinyxml2::XMLElement* selection_error_goal_element = root_element->FirstChildElement("SelectionErrorGoal");

    if(selection_error_goal_element)
        set_selection_error_goal(type(atof(selection_error_goal_element->GetText())));

    // Maximum epochs number

    const tinyxml2::XMLElement* maximum_epochs_number_element = root_element->FirstChildElement("MaximumEpochsNumber");

    if(maximum_epochs_number_element)
        set_maximum_epochs_number(Index(atoi(maximum_epochs_number_element->GetText())));

    // Maximum correlation

    const tinyxml2::XMLElement* maximum_correlation_element = root_element->FirstChildElement("MaximumCorrelation");

    if(maximum_correlation_element)
        set_maximum_correlation(type(atof(maximum_correlation_element->GetText())));

    // Minimum correlation

    const tinyxml2::XMLElement* minimum_correlation_element = root_element->FirstChildElement("MinimumCorrelation");

    if(minimum_correlation_element)
        set_minimum_correlation(type(atof(minimum_correlation_element->GetText())));

    // Maximum time

    const tinyxml2::XMLElement* maximum_time_element = root_element->FirstChildElement("MaximumTime");

    if(maximum_time_element)
        set_maximum_time(type(atof(maximum_time_element->GetText())));

    // Minimum inputs number

    const tinyxml2::XMLElement* minimum_inputs_number_element = root_element->FirstChildElement("MinimumInputsNumber");

    if(minimum_inputs_number_element)
        set_minimum_inputs_number(Index(atoi(minimum_inputs_number_element->GetText())));

    // Maximum inputs number

    const tinyxml2::XMLElement* maximum_inputs_number_element = root_element->FirstChildElement("MaximumInputsNumber");

    if(maximum_inputs_number_element)
        set_maximum_inputs_number(Index(atoi(maximum_inputs_number_element->GetText())));

    // Maximum selection failures

    const tinyxml2::XMLElement* maximum_selection_failures_element = root_element->FirstChildElement("MaximumSelectionFailures");

    if(maximum_selection_failures_element)
        set_maximum_selection_failures(Index(atoi(maximum_selection_failures_element->GetText())));
}


void GrowingInputs::save(const string& file_name) const
{
    FILE * file = fopen(file_name.c_str(), "w");

    if(!file) return;

    tinyxml2::XMLPrinter printer(file);
    to_XML(printer);
    fclose(file);
}


void GrowingInputs::load(const string& file_name)
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
