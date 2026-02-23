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
#include "scaling_layer.h"
#include "scaling_layer.h"
#include "optimization_algorithm.h"
#include "training_strategy.h"

namespace opennn
{

GrowingInputs::GrowingInputs(const TrainingStrategy* new_training_strategy)
    : InputsSelection(new_training_strategy)
{
    set_default();
}


Index GrowingInputs::get_minimum_inputs_number() const
{
    return minimum_inputs_number;
}


Index GrowingInputs::get_maximum_inputs_number() const
{
    return maximum_inputs_number;
}


Index GrowingInputs::get_maximum_validation_failures() const
{
    return maximum_validation_failures;
}


void GrowingInputs::set_default()
{
    name = "GrowingInputs";

    maximum_validation_failures = 100;
    minimum_inputs_number = 1;
    trials_number = 3;
    maximum_epochs = 1000;
    maximum_time = type(3600);

    training_strategy && training_strategy->has_neural_network()
        ? maximum_inputs_number = training_strategy->get_dataset()->get_variables_number("Input")
        : maximum_inputs_number = 50;
}


void GrowingInputs::set_minimum_inputs_number(const Index new_minimum_inputs_number)
{
    minimum_inputs_number = new_minimum_inputs_number;
}


void GrowingInputs::set_maximum_inputs_number(const Index new_maximum_inputs_number)
{
    const Index inputs_number = training_strategy->get_dataset()->get_variables_number("Input");

    maximum_inputs_number = (inputs_number == 0)
                                ? new_maximum_inputs_number
                                : min(new_maximum_inputs_number, inputs_number);
}


void GrowingInputs::set_minimum_correlation(const type new_minimum_correlation)
{
    minimum_correlation = new_minimum_correlation;
}


void GrowingInputs::set_maximum_correlation(const type new_maximum_correlation)
{
    maximum_correlation = new_maximum_correlation;
}


void GrowingInputs::set_maximum_validation_failures(const Index new_maximum_validation_failures)
{
    maximum_validation_failures = new_maximum_validation_failures;
}


InputsSelectionResults GrowingInputs::perform_input_selection()
{
    // Dataset

    Dataset* dataset = training_strategy->get_dataset();
    const Index original_input_variables_number = dataset->get_variables_number("Input");

    if(dataset->has_nan())
        dataset->scrub_missing_values();

    if(display) cout << "Performing growing input selection..." << endl;

    InputsSelectionResults input_selection_results(original_input_variables_number);

    // Loss index

    const Loss* loss_index = training_strategy->get_loss_index();
    training_strategy->get_optimization_algorithm()->set_display(false);

    type previus_validation_error = numeric_limits<type>::max();
    type previus_training_error = numeric_limits<type>::max();

    TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");
    const vector<Index> time_variable_indices = dataset->get_variable_indices("Time");
    const vector<string> variable_names = dataset->get_variable_names();
    vector<string> input_variable_names;

    if(display) cout << "Calculating correlations..." << endl;

    const MatrixR correlations = get_correlation_values(dataset->calculate_input_target_variable_pearson_correlations());
    const VectorR total_correlations = correlations.col(0).array().abs();

    vector<Index> correlation_indices(original_input_variables_number);
    iota(correlation_indices.begin(), correlation_indices.end(), 0);

    sort(correlation_indices.data(),
        correlation_indices.data() + correlation_indices.size(),
        [&](Index i, Index j) {return total_correlations[i] > total_correlations[j]; });

    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");

    VectorI correlations_rank_descending(input_variable_indices.size());

    for(Index i = 0; i < correlations_rank_descending.size(); i++)
        correlations_rank_descending(i) = input_variable_indices[correlation_indices[i]];

    dataset->set_input_variables_unused();

    Index variable_index = 0;

    // Neural network

    NeuralNetwork* neural_network = training_strategy->get_neural_network();

    // Training strategy

    Index validation_failures = 0;
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
        if(variable_index >= correlations_rank_descending.size())
        {
            if (display) cout << "\nAll the variables has been used." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
            stop = true;
            continue;
        }

        const Index current_variable_index = correlations_rank_descending[variable_index];
        const string current_use = dataset->get_variables()[current_variable_index].role;

        if(current_use == "InputTarget")
            dataset->set_variable_role(current_variable_index, "InputTarget");
        else
            dataset->set_variable_role(current_variable_index, "Input");

        const Index input_variables_number = dataset->get_variables_number("Input");
        const Index input_features_number = dataset->get_features_number("Input");

        if(input_variables_number < minimum_inputs_number)
        {
            variable_index++;
            continue;
        }

        if(time_series_dataset)
        {
            const Index past_time_steps = time_series_dataset->get_past_time_steps();
            neural_network->set_input_shape({ past_time_steps, input_features_number });
            dataset->set_shape("Input", { past_time_steps, input_features_number });
        }
        else
        {
            neural_network->set_input_shape({ input_features_number });
            dataset->set_shape("Input", { input_features_number });
        }

        if(display)
        {
            cout << endl
                << "Epoch: " << epoch + 1 << endl
                << "Input variables number: " << input_variables_number << endl
                << "Inputs: " << endl;

            input_variable_names = dataset->get_variable_names("Input");
            cout << input_variable_names;
        }

        type minimum_training_error = numeric_limits<type>::max();
        type minimum_validation_error = numeric_limits<type>::max();

        for(Index j = 0; j < trials_number; j++)
        {
            neural_network->set_parameters_random();
            training_results = training_strategy->train();

            if(training_results.get_validation_error() < minimum_validation_error)
            {
                minimum_training_error = training_results.get_training_error();
                minimum_validation_error = training_results.get_validation_error();
            }

            if(minimum_validation_error < input_selection_results.optimum_validation_error)
            {
                input_selection_results.optimal_input_variables_indices = dataset->get_variable_indices("Input");
                input_selection_results.optimal_input_variable_names = dataset->get_variable_names("Input");
                //neural_network->get_parameters(input_selection_results.optimal_parameters);
                input_selection_results.optimum_training_error = training_results.get_training_error();
                input_selection_results.optimum_validation_error = training_results.get_validation_error();
            }

            if(display)
                cout << "Trial number: " << j + 1 << endl
                << "   Training error: " << training_results.get_training_error() << endl
                << "   Validation error: " << training_results.get_validation_error() << endl;
        }

        if(previus_training_error < minimum_training_error)
        {
            if(display) cout << "Validation failure" << endl;
            validation_failures++;

            if(dataset->get_variables()[current_variable_index].role == "InputTarget")
                dataset->set_variable_role(current_variable_index, "Target");
            else
                dataset->set_variable_role(current_variable_index, "None");
        }
        else
        {
            previus_training_error = minimum_training_error;
            previus_validation_error = minimum_validation_error;

            input_selection_results.training_error_history(epoch) = minimum_training_error;
            input_selection_results.validation_error_history(epoch) = minimum_validation_error;

            epoch++;
        }

        variable_index++;
        time(&current_time);
        elapsed_time = type(difftime(current_time, beginning_time));

        // Stopping criteria

        if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum time reached: " << write_time(elapsed_time) << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumTime;
            stop = true;
        }
        else if(input_selection_results.optimum_validation_error <= validation_error_goal)
        {
            if(display) cout << "\nSelection error reached: " << input_selection_results.optimum_validation_error << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::SelectionErrorGoal;
            stop = true;
        }
        else if(epoch >= maximum_epochs)
        {
            if(display) cout << "\nMaximum number of epochs reached." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumEpochs;
            stop = true;
        }
        else if(validation_failures >= maximum_validation_failures)
        {
            if(display) cout << "\nMaximum selection failures (" << validation_failures << ") reached." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumSelectionFailures;
            stop = true;
        }
        else if(dataset->get_variables_number("Input") >= maximum_inputs_number)
        {
            if(display) cout << "\nMaximum inputs (" << dataset->get_variables_number("Input") << ") reached." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
            stop = true;
        }
    }

    input_selection_results.elapsed_time = write_time(elapsed_time);
    input_selection_results.resize_history(epoch);

    // Set dataset

    dataset->set_variable_indices(input_selection_results.optimal_input_variables_indices,
        target_variable_indices);

    const Index optimal_processed_variables_number = dataset->get_features_number("Input");

    if(time_series_dataset)
    {
        const Index past_time_steps = time_series_dataset->get_past_time_steps();
        dataset->set_shape("Input", { past_time_steps, optimal_processed_variables_number });
        neural_network->set_input_shape({ past_time_steps, optimal_processed_variables_number });

        if(time_variable_indices.size() == 1)
            dataset->set_variable_role(time_variable_indices[0], "Time");
    }
    else
    {
        dataset->set_shape("Input", { optimal_processed_variables_number });
        neural_network->set_input_shape({ optimal_processed_variables_number });
    }

    dataset->print();

    const vector<string> input_variable_scalers = dataset->get_feature_scalers("Input");
    const vector<Descriptives> input_variable_descriptives = dataset->calculate_feature_descriptives("Input");

    set_maximum_inputs_number(dataset->get_variables_number("Input"));

    // Set neural network

    if(time_series_dataset)
    {
        vector<string> final_feature_names;
        const vector<string> base_names = dataset->get_variable_names("Input");
        const Index time_steps = time_series_dataset->get_past_time_steps();
        final_feature_names.reserve(base_names.size() * time_steps);
        for(const string& base_name : base_names)
        {
            for(Index j = 0; j < time_steps; j++)
            {
                string name = (base_name.empty() ? "variable" : base_name) + "_lag" + to_string(j);
                final_feature_names.push_back(name);
            }
        }
        neural_network->set_feature_names(final_feature_names);
    }
    else
        neural_network->set_feature_names(dataset->get_feature_names("Input"));

    if(neural_network->has("Scaling2d"))
    {
        Scaling<2>* scaling_layer = static_cast<Scaling<2>*>(neural_network->get_first("Scaling2d"));
        scaling_layer->set_descriptives(input_variable_descriptives);
        scaling_layer->set_scalers(input_variable_scalers);
    }
    else if(neural_network->has("Scaling3d"))
    {
        Scaling<3>* scaling_layer = static_cast<Scaling<3>*>(neural_network->get_first("Scaling3d"));
        scaling_layer->set_descriptives(input_variable_descriptives);
        scaling_layer->set_scalers(input_variable_scalers);
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
    {"Validation error goal", to_string(validation_error_goal)},
    {"Maximum selection failures", to_string(maximum_validation_failures)},
    {"Maximum inputs number", to_string(maximum_inputs_number)},
    {"Minimum correlations", to_string(minimum_correlation)},
    {"Maximum correlation", to_string(maximum_correlation)},
    {"Maximum iterations number", to_string(maximum_epochs)},
    {"Maximum time", to_string(maximum_time)}});

    return string_matrix;
}


void GrowingInputs::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("GrowingInputs");

    add_xml_element(printer, "TrialsNumber", to_string(trials_number));
    add_xml_element(printer, "SelectionErrorGoal", to_string(validation_error_goal));
    add_xml_element(printer, "MaximumSelectionFailures", to_string(maximum_validation_failures));
    add_xml_element(printer, "MinimumInputsNumber", to_string(minimum_inputs_number));
    add_xml_element(printer, "MaximumInputsNumber", to_string(maximum_inputs_number));
    add_xml_element(printer, "MinimumCorrelation", to_string(minimum_correlation));
    add_xml_element(printer, "MaximumCorrelation", to_string(maximum_correlation));
    add_xml_element(printer, "MaximumEpochsNumber", to_string(maximum_epochs));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));

    printer.CloseElement();  
}


void GrowingInputs::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("GrowingInputs");

    if(!root_element)
        throw runtime_error("GrowingInputs element is nullptr.\n");

    set_trials_number(read_xml_index(root_element, "TrialsNumber"));
    set_validation_error_goal(read_xml_type(root_element, "SelectionErrorGoal"));
    set_maximum_epochs(read_xml_index(root_element, "MaximumEpochsNumber"));
    set_maximum_correlation(read_xml_type(root_element, "MaximumCorrelation"));
    set_minimum_correlation(read_xml_type(root_element, "MinimumCorrelation"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
    set_minimum_inputs_number(read_xml_index(root_element, "MinimumInputsNumber"));
    set_maximum_inputs_number(read_xml_index(root_element, "MaximumInputsNumber"));
    set_maximum_validation_failures(read_xml_index(root_element, "MaximumSelectionFailures"));
}


void GrowingInputs::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
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
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
