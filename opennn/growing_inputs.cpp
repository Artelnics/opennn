//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "time_series_dataset.h"
#include "growing_inputs.h"
#include "correlations.h"
#include "scaling_layer_2d.h"
#include "scaling_layer_3d.h"
#include "optimization_algorithm.h"
#include "training_strategy.h"

namespace opennn
{

GrowingInputs::GrowingInputs(const TrainingStrategy* new_training_strategy)
    : InputsSelection(new_training_strategy)
{
    set_default();
}


const Index& GrowingInputs::get_minimum_inputs_number() const
{
    return minimum_inputs_number;
}


const Index& GrowingInputs::get_maximum_inputs_number() const
{
    return maximum_inputs_number;
}


const Index& GrowingInputs::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


void GrowingInputs::set_default()
{
    name = "GrowingInputs";

    maximum_selection_failures = 100;
    minimum_inputs_number = 1;
    trials_number = 3;
    maximum_epochs_number = 1000;
    maximum_time = type(3600);

    training_strategy && training_strategy->has_neural_network()
        ? maximum_inputs_number = training_strategy->get_dataset()->get_raw_variables_number("Input")
        : maximum_inputs_number = 50;
}


void GrowingInputs::set_minimum_inputs_number(const Index& new_minimum_inputs_number)
{
    minimum_inputs_number = new_minimum_inputs_number;
}


void GrowingInputs::set_maximum_inputs_number(const Index& new_maximum_inputs_number)
{
    const Index inputs_number = training_strategy->get_dataset()->get_raw_variables_number("Input");

    maximum_inputs_number = (inputs_number == 0)
                                ? new_maximum_inputs_number
                                : min(new_maximum_inputs_number, inputs_number);
}


void GrowingInputs::set_minimum_correlation(const type& new_minimum_correlation)
{
    minimum_correlation = new_minimum_correlation;
}


void GrowingInputs::set_maximum_correlation(const type& new_maximum_correlation)
{
    maximum_correlation = new_maximum_correlation;
}


void GrowingInputs::set_maximum_selection_failures(const Index& new_maximum_selection_failures)
{
    maximum_selection_failures = new_maximum_selection_failures;
}


InputsSelectionResults GrowingInputs::perform_input_selection()
{
    // Dataset

    Dataset* dataset = training_strategy->get_dataset();
    const Index original_input_raw_variables_number = dataset->get_raw_variables_number("Input");

    if(dataset->has_nan())
        dataset->scrub_missing_values();

    if(display) cout << "Performing growing inputs selection..." << endl;

    InputsSelectionResults input_selection_results(original_input_raw_variables_number);

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();
    training_strategy->get_optimization_algorithm()->set_display(false);

    type previus_selection_error = numeric_limits<type>::max();
    type previus_training_error = numeric_limits<type>::max();

    TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);
    const vector<Index> target_raw_variable_indices = dataset->get_raw_variable_indices("Target");
    const vector<Index> time_raw_variable_indices = dataset->get_raw_variable_indices("Time");
    const vector<string> raw_variable_names = dataset->get_raw_variable_names();
    vector<string> input_raw_variable_names;

    if(display) cout << "Calculating correlations..." << endl;

    const Tensor<type, 2> correlations = get_correlation_values(dataset->calculate_input_target_raw_variable_pearson_correlations());
    const Tensor<type, 1> total_correlations = correlations.abs().chip(0, 1);

    vector<Index> correlation_indices(original_input_raw_variables_number);
    iota(correlation_indices.begin(), correlation_indices.end(), 0);

    sort(correlation_indices.data(),
        correlation_indices.data() + correlation_indices.size(),
        [&](Index i, Index j) {return total_correlations[i] > total_correlations[j]; });

    const vector<Index> input_raw_variable_indices = dataset->get_raw_variable_indices("Input");

    Tensor<Index, 1> correlations_rank_descending(input_raw_variable_indices.size());

    for(Index i = 0; i < correlations_rank_descending.size(); i++)
        correlations_rank_descending(i) = input_raw_variable_indices[correlation_indices[i]];

    dataset->set_input_raw_variables_unused();

    Index raw_variable_index = 0;

    // Neural network

    NeuralNetwork* neural_network = training_strategy->get_neural_network();

    // Training strategy

    Index selection_failures = 0;
    TrainingResults training_results;

    // Model selection

    time_t beginning_time;
    time_t current_time;
    type elapsed_time = type(0);
    time(&beginning_time);

    bool stop = false;
    Index epoch = 0;

    while(!stop)
    {
        if(raw_variable_index >= correlations_rank_descending.size())
        {
            if (display) cout << "\nAll the raw_variables has been used." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
            stop = true;
            continue;
        }

        const Index current_raw_var_index = correlations_rank_descending[raw_variable_index];
        const string current_use = dataset->get_raw_variables()[current_raw_var_index].use;

        if(current_use == "InputTarget")
            dataset->set_raw_variable_use(current_raw_var_index, "InputTarget");
        else
            dataset->set_raw_variable_use(current_raw_var_index, "Input");

        const Index input_raw_variables_number = dataset->get_raw_variables_number("Input");
        const Index input_variables_number = dataset->get_variables_number("Input");

        if(input_raw_variables_number < minimum_inputs_number)
        {
            raw_variable_index++;
            continue;
        }

        if(time_series_dataset)
        {
            const Index past_time_steps = time_series_dataset->get_past_time_steps();
            neural_network->set_input_dimensions({ past_time_steps, input_variables_number });
            dataset->set_dimensions("Input", { past_time_steps, input_variables_number });
        }
        else
        {
            neural_network->set_input_dimensions({ input_variables_number });
            dataset->set_dimensions("Input", { input_variables_number });
        }

        if(display)
        {
            cout << endl
                << "Epoch: " << epoch + 1 << endl
                << "Input raw_variables number: " << input_raw_variables_number << endl
                << "Inputs: " << endl;

            input_raw_variable_names = dataset->get_raw_variable_names("Input");
            print_vector(input_raw_variable_names);
        }

        type minimum_training_error = numeric_limits<type>::max();
        type minimum_selection_error = numeric_limits<type>::max();

        for(Index j = 0; j < trials_number; j++)
        {
            neural_network->set_parameters_random();
            training_results = training_strategy->train();

            if(training_results.get_selection_error() < minimum_selection_error)
            {
                minimum_training_error = training_results.get_training_error();
                minimum_selection_error = training_results.get_selection_error();
            }

            if(minimum_selection_error < input_selection_results.optimum_selection_error)
            {
                input_selection_results.optimal_input_raw_variables_indices = dataset->get_raw_variable_indices("Input");
                input_selection_results.optimal_input_raw_variable_names = dataset->get_raw_variable_names("Input");
                neural_network->get_parameters(input_selection_results.optimal_parameters);
                input_selection_results.optimum_training_error = training_results.get_training_error();
                input_selection_results.optimum_selection_error = training_results.get_selection_error();
            }

            if(display)
                cout << "Trial number: " << j + 1 << endl
                << "   Training error: " << training_results.get_training_error() << endl
                << "   Selection error: " << training_results.get_selection_error() << endl;
        }

        if(previus_training_error < minimum_training_error)
        {
            if(display) cout << "Selection failure" << endl;
            selection_failures++;

            if(dataset->get_raw_variables()[current_raw_var_index].use == "InputTarget")
                dataset->set_raw_variable_use(current_raw_var_index, "Target");
            else
                dataset->set_raw_variable_use(current_raw_var_index, "None");
        }
        else
        {
            previus_training_error = minimum_training_error;
            previus_selection_error = minimum_selection_error;

            input_selection_results.training_error_history(epoch) = minimum_training_error;
            input_selection_results.selection_error_history(epoch) = minimum_selection_error;

            epoch++;
        }

        raw_variable_index++;
        time(&current_time);
        elapsed_time = type(difftime(current_time, beginning_time));

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum time reached: " << write_time(elapsed_time) << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumTime;
            stop = true;
        }
        else if(input_selection_results.optimum_selection_error <= selection_error_goal)
        {
            if(display) cout << "\nSelection error reached: " << input_selection_results.optimum_selection_error << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::SelectionErrorGoal;
            stop = true;
        }
        else if(epoch >= maximum_epochs_number)
        {
            if(display) cout << "\nMaximum number of epochs reached." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumEpochs;
            stop = true;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "\nMaximum selection failures (" << selection_failures << ") reached." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumSelectionFailures;
            stop = true;
        }
        else if(dataset->get_raw_variables_number("Input") >= maximum_inputs_number)
        {
            if(display) cout << "\nMaximum inputs (" << dataset->get_raw_variables_number("Input") << ") reached." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
            stop = true;
        }
    }

    input_selection_results.elapsed_time = write_time(elapsed_time);
    input_selection_results.resize_history(epoch);

    // Set dataset

    dataset->set_raw_variable_indices(input_selection_results.optimal_input_raw_variables_indices,
        target_raw_variable_indices);

    const Index optimal_processed_variables_number = dataset->get_variables_number("Input");

    if(time_series_dataset)
    {
        const Index past_time_steps = time_series_dataset->get_past_time_steps();
        dataset->set_dimensions("Input", { past_time_steps, optimal_processed_variables_number });
        neural_network->set_input_dimensions({ past_time_steps, optimal_processed_variables_number });

        if(time_raw_variable_indices.size() == 1)
            dataset->set_raw_variable_use(time_raw_variable_indices[0], "Time");
    }
    else
    {
        dataset->set_dimensions("Input", { optimal_processed_variables_number });
        neural_network->set_input_dimensions({ optimal_processed_variables_number });
    }

    dataset->print();

    const vector<string> input_variable_scalers = dataset->get_variable_scalers("Input");
    const vector<Descriptives> input_variable_descriptives = dataset->calculate_variable_descriptives("Input");

    set_maximum_inputs_number(dataset->get_raw_variables_number("Input"));

    // Set neural network

    if(time_series_dataset)
    {
        vector<string> final_input_names;
        const vector<string> base_names = dataset->get_raw_variable_names("Input");
        const Index time_steps = time_series_dataset->get_past_time_steps();
        final_input_names.reserve(base_names.size() * time_steps);
        for(const string& base_name : base_names)
        {
            for(Index j = 0; j < time_steps; j++)
            {
                string name = (base_name.empty() ? "variable" : base_name) + "_lag" + to_string(j);
                final_input_names.push_back(name);
            }
        }
        neural_network->set_input_names(final_input_names);
    }
    else
        neural_network->set_input_names(dataset->get_variable_names("Input"));

    if(neural_network->has("Scaling2d"))
    {
        Scaling2d* scaling_layer_2d = static_cast<Scaling2d*>(neural_network->get_first("Scaling2d"));
        scaling_layer_2d->set_descriptives(input_variable_descriptives);
        scaling_layer_2d->set_scalers(input_variable_scalers);
    }
    else if(neural_network->has("Scaling3d"))
    {
        Scaling3d* scaling_layer_3d = static_cast<Scaling3d*>(neural_network->get_first("Scaling3d"));
        scaling_layer_3d->set_descriptives(input_variable_descriptives);
        scaling_layer_3d->set_scalers(input_variable_scalers);
    }

    neural_network->set_parameters(input_selection_results.optimal_parameters);

    if(display) input_selection_results.print();

    return input_selection_results;
}


Tensor<string, 2> GrowingInputs::to_string_matrix() const
{
    Tensor<string, 2> string_matrix(8, 2);

    string_matrix.setValues({
    {"Trials number", to_string(trials_number)},
    {"Selection error goal", to_string(selection_error_goal)},
    {"Maximum selection failures", to_string(maximum_selection_failures)},
    {"Maximum inputs number", to_string(maximum_inputs_number)},
    {"Minimum correlations", to_string(minimum_correlation)},
    {"Maximum correlation", to_string(maximum_correlation)},
    {"Maximum iterations number", to_string(maximum_epochs_number)},
    {"Maximum time", to_string(maximum_time)}});

    return string_matrix;
}


void GrowingInputs::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("GrowingInputs");

    add_xml_element(printer, "TrialsNumber", to_string(trials_number));
    add_xml_element(printer, "SelectionErrorGoal", to_string(selection_error_goal));
    add_xml_element(printer, "MaximumSelectionFailures", to_string(maximum_selection_failures));
    add_xml_element(printer, "MinimumInputsNumber", to_string(minimum_inputs_number));
    add_xml_element(printer, "MaximumInputsNumber", to_string(maximum_inputs_number));
    add_xml_element(printer, "MinimumCorrelation", to_string(minimum_correlation));
    add_xml_element(printer, "MaximumCorrelation", to_string(maximum_correlation));
    add_xml_element(printer, "MaximumEpochsNumber", to_string(maximum_epochs_number));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));

    printer.CloseElement();  
}


void GrowingInputs::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("GrowingInputs");

    if(!root_element)
        throw runtime_error("GrowingInputs element is nullptr.\n");

    set_trials_number(read_xml_index(root_element, "TrialsNumber"));
    set_selection_error_goal(read_xml_type(root_element, "SelectionErrorGoal"));
    set_maximum_epochs_number(read_xml_index(root_element, "MaximumEpochsNumber"));
    set_maximum_correlation(read_xml_type(root_element, "MaximumCorrelation"));
    set_minimum_correlation(read_xml_type(root_element, "MinimumCorrelation"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
    set_minimum_inputs_number(read_xml_index(root_element, "MinimumInputsNumber"));
    set_maximum_inputs_number(read_xml_index(root_element, "MaximumInputsNumber"));
    set_maximum_selection_failures(read_xml_index(root_element, "MaximumSelectionFailures"));
}


void GrowingInputs::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    XMLPrinter printer;
    to_XML(printer);
    file << printer.CStr();
}


void GrowingInputs::load(const filesystem::path& file_name)
{
    set_default();

    XMLDocument document;

    if (document.LoadFile(file_name.string().c_str()))
        throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

    from_XML(document);
}

REGISTER(InputsSelection, GrowingInputs, "GrowingInputs");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
