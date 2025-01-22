//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "growing_inputs.h"
#include "correlations.h"
#include "scaling_layer_2d.h"

namespace opennn
{

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

    if(!training_strategy || !training_strategy->has_neural_network())
    {
        maximum_inputs_number = 100;
    }
    else
    {
        training_strategy->get_neural_network()->get_display();

        const Index inputs_number = training_strategy->get_data_set()->get_raw_variables_number(DataSet::VariableUse::Input);

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
    const Index inputs_number = training_strategy->get_data_set()->get_raw_variables_number(DataSet::VariableUse::Input);

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
    InputsSelectionResults input_selection_results(maximum_epochs_number);

    if(display) cout << "Performing growing inputs selection..." << endl;

    // Loss index

    const LossIndex* loss_index = training_strategy->get_loss_index();

//    type previus_selection_error = numeric_limits< type>::max();
    type previus_training_error = numeric_limits< type>::max();

    // Data set

    DataSet* data_set = loss_index->get_data_set();

    const vector<Index> target_raw_variable_indices = data_set->get_raw_variable_indices(DataSet::VariableUse::Target);

    const Index original_input_raw_variables_number = data_set->get_raw_variables_number(DataSet::VariableUse::Input);

    const vector<string> raw_variable_names = data_set->get_raw_variable_names();

    vector<string> input_raw_variables_names;

    const Tensor<type, 2> correlations = get_correlation_values(data_set->calculate_input_target_raw_variable_pearson_correlations());

    const Tensor<type, 1> total_correlations = correlations.abs().chip(0,1);

    vector<Index> correlation_indices(original_input_raw_variables_number);
    iota(correlation_indices.begin(), correlation_indices.end(), 0);

    sort(correlation_indices.data(),
         correlation_indices.data() + correlation_indices.size(),
         [&](Index i, Index j){return total_correlations[i] > total_correlations[j];});

    const vector<Index> input_raw_variable_indices = data_set->get_raw_variable_indices(DataSet::VariableUse::Input);

    Tensor<Index, 1> correlations_rank_descending(input_raw_variable_indices.size());

    for(Index i = 0; i < correlations_rank_descending.size(); i++) 
        correlations_rank_descending(i) = input_raw_variable_indices[correlation_indices[i]];

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

        Index input_raw_variables_number = data_set->get_raw_variables_number(DataSet::VariableUse::Input);
        Index input_variables_number = data_set->get_variables_number(DataSet::VariableUse::Input);

        if (input_raw_variables_number < minimum_inputs_number)
        {
            // @todo

            continue;
        }
            
        Index epoch = input_raw_variables_number - minimum_inputs_number + 1;
        neural_network->set_input_dimensions({ input_variables_number });

        if(display)
        {
            cout << endl
                    << "Epoch: " << epoch << endl
                    << "Input raw_variables number: " << input_raw_variables_number << endl
                    << "Inputs: " << endl;

            input_raw_variables_names = data_set->get_raw_variable_names(DataSet::VariableUse::Input);

            print_vector(input_raw_variables_names);
        }

        type minimum_training_error = numeric_limits<type>::max();
        type minimum_selection_error = numeric_limits<type>::max();

        for(Index j = 0; j < trials_number; j++)
        {
            neural_network->set_parameters_random();

            if(data_set->has_nan())
                data_set->scrub_missing_values();

            training_results = training_strategy->perform_training();

            if(training_results.get_selection_error() < minimum_selection_error)
            {
                minimum_training_error = training_results.get_training_error();
                minimum_selection_error = training_results.get_selection_error();

                input_selection_results.training_error_history(input_raw_variables_number-1) = minimum_training_error;
                input_selection_results.selection_error_history(input_raw_variables_number-1) = minimum_selection_error;
            }

            if(training_results.get_selection_error() < input_selection_results.optimum_selection_error)
            {
                // Neural network

                input_selection_results.optimal_input_raw_variables_indices = data_set->get_raw_variable_indices(DataSet::VariableUse::Input);
                input_selection_results.optimal_input_raw_variable_names = data_set->get_raw_variable_names(DataSet::VariableUse::Input);

                input_selection_results.optimal_parameters = neural_network->get_parameters();

                // Loss index

                input_selection_results.optimum_training_error = training_results.get_training_error();
                input_selection_results.optimum_selection_error = training_results.get_selection_error();
            }

            if(display)
                cout << "Trial number: " << j+1 << endl
                        << "   Training error: " << training_results.get_training_error() << endl
                        << "   Selection error: " << training_results.get_selection_error() << endl;           
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

            input_selection_results.training_error_history(input_raw_variables_number) = minimum_training_error;
            input_selection_results.selection_error_history(input_raw_variables_number) = minimum_selection_error;
        }

        time(&current_time);

        elapsed_time = type(difftime(current_time,beginning_time));

        // Stopping criteria

        stop = true;

        if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum time reached: " << write_time(elapsed_time) << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumTime;
        }
        else if(input_selection_results.optimum_selection_error <= selection_error_goal)
        {
            if(display) cout << "\nSelection error reached: " << input_selection_results.optimum_selection_error << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::SelectionErrorGoal;
        }
        else if(epoch >= maximum_epochs_number)
        {
            if(display) cout << "\nMaximum number of epochs reached." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumEpochs;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "\nMaximum selection failures ("<<selection_failures<<") reached." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumSelectionFailures;
        }
        else if(input_raw_variables_number >= maximum_inputs_number || input_raw_variables_number >= original_input_raw_variables_number)
        {
            if(display) cout << "\nMaximum inputs (" << input_raw_variables_number << ") reached." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
        }
        else if(raw_variable_index >= correlations_rank_descending.size() - 1 )
        {
            if(display) cout << "\nAll the raw_variables has been used." << endl;
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
        }
        else
        {
            stop = false;
        }

        if(stop)
        {
            input_selection_results.elapsed_time = write_time(elapsed_time);

            input_selection_results.resize_history(input_raw_variables_number);

            break;
        }
            

        raw_variable_index++;        
    }

    // Set data set stuff

    data_set->set_input_target_raw_variable_indices(input_selection_results.optimal_input_raw_variables_indices, 
                                                    target_raw_variable_indices);

    data_set->print();
    
    const vector<Scaler> input_variable_scalers = data_set->get_variable_scalers(DataSet::VariableUse::Input);

    const vector<Descriptives> input_variable_descriptives = data_set->calculate_variable_descriptives(DataSet::VariableUse::Input);

    set_maximum_inputs_number(data_set->get_raw_variables_number(DataSet::VariableUse::Input));

    // Set neural network stuff

    neural_network->set_input_dimensions({ data_set->get_variables_number(DataSet::VariableUse::Input) });

    neural_network->set_input_names(data_set->get_variable_names(DataSet::VariableUse::Input));

    if(neural_network->has(Layer::Type::Scaling2D))
    {
        ScalingLayer2D* scaling_layer_2d = static_cast<ScalingLayer2D*>(neural_network->get_first(Layer::Type::Scaling2D));
        scaling_layer_2d->set_descriptives(input_variable_descriptives);
        scaling_layer_2d->set_scalers(input_variable_scalers);
    }
/*
    neural_network->set_parameters(input_selection_results.optimal_parameters);
*/
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
    set_display(read_xml_bool(root_element, "Display"));
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
