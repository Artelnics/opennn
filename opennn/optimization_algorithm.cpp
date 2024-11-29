//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "optimization_algorithm.h"

namespace opennn
{

OptimizationAlgorithm::OptimizationAlgorithm(LossIndex* new_loss_index)
{
    const unsigned int threads_number = thread::hardware_concurrency();
    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

    set(new_loss_index);
}


LossIndex* OptimizationAlgorithm::get_loss_index() const
{
    return loss_index;
}


string OptimizationAlgorithm::get_hardware_use() const
{
    return hardware_use;
}


void OptimizationAlgorithm::set_hardware_use(const string& new_hardware_use)
{
    hardware_use = new_hardware_use;
}


bool OptimizationAlgorithm::has_loss_index() const
{
    return loss_index;
}


const bool& OptimizationAlgorithm::get_display() const
{
    return display;
}


const Index& OptimizationAlgorithm::get_display_period() const
{
    return display_period;
}


const Index& OptimizationAlgorithm::get_save_period() const
{
    return save_period;
}


const string& OptimizationAlgorithm::get_neural_network_file_name() const
{
    return neural_network_file_name;
}


void OptimizationAlgorithm::set(LossIndex* new_loss_index)
{
    loss_index = new_loss_index;

    set_default();
}


void OptimizationAlgorithm::set_threads_number(const int& new_threads_number)
{
    thread_pool = make_unique<ThreadPool>(new_threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), new_threads_number);
}


void OptimizationAlgorithm::set_loss_index(LossIndex* new_loss_index)
{
    loss_index = new_loss_index;
}


void OptimizationAlgorithm::set_display(const bool& new_display)
{
    display = new_display;
}


void OptimizationAlgorithm::set_display_period(const Index& new_display_period)
{
    display_period = new_display_period;
}


void OptimizationAlgorithm::set_save_period(const Index& new_save_period)
{
    save_period = new_save_period;
}


void OptimizationAlgorithm::set_neural_network_file_name(const string& new_neural_network_file_name)
{
    neural_network_file_name = new_neural_network_file_name;
}


// BoxPlot OptimizationAlgorithm::calculate_distances_box_plot(type* & new_inputs_data, Tensor<Index,1>& input_dimensions,
//                                                             type* & new_outputs_data, Tensor<Index,1>& output_dimensions)
// {
//     const Index samples_number = input_dimensions(0);
//     const Index inputs_number = input_dimensions(1);

//     TensorMap<Tensor<type, 2>> inputs(new_inputs_data, samples_number, inputs_number);
//     TensorMap<Tensor<type, 2>> outputs(new_outputs_data, output_dimensions[0], output_dimensions(1));

//     Tensor<type, 1> distances(samples_number);
//     Index distance_index = 0;

//     for(Index i = 0; i < samples_number; i++)
//     {
//         Tensor<type, 1> input_row = inputs.chip(i, 0);
//         Tensor<type, 1> output_row = outputs.chip(i, 0);

//         const type distance = l2_distance(input_row, output_row)/inputs_number;

//         if(!isnan(distance))
//             distances(distance_index++) = l2_distance(input_row, output_row)/inputs_number;
//     }

//     return box_plot(distances);
// }


void OptimizationAlgorithm::set_default()
{
    display = true;

    display_period = 10;

    save_period = UINT_MAX;

    neural_network_file_name = "neural_network.xml";
}


void OptimizationAlgorithm::check() const
{
}


void OptimizationAlgorithm::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("OptimizationAlgorithm");

    add_xml_element(printer, "Display", std::to_string(display));

    printer.CloseElement();
}


void OptimizationAlgorithm::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("OptimizationAlgorithm");

    if(!root_element)
        throw runtime_error("Optimization algorithm element is nullptr.\n");

    set_display(read_xml_bool(root_element, "Display"));
}


Tensor<string, 2> OptimizationAlgorithm::to_string_matrix() const
{
    return Tensor<string, 2>();
}


void OptimizationAlgorithm::print() const
{
}


void OptimizationAlgorithm::save(const string& file_name) const
{
    try
    {
        ofstream file(file_name);

        if (file.is_open())
        {
            XMLPrinter printer;
            to_XML(printer);

            file << printer.CStr();

            file.close();
        }
        else
        {
            throw runtime_error("Cannot open file: " + file_name);
        }
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;
    }
}


void OptimizationAlgorithm::load(const string& file_name)
{
    set_default();

    XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
        throw runtime_error("Cannot load XML file " + file_name + ".\n");

    from_XML(document);
}


type OptimizationAlgorithm::get_elapsed_time(const time_t &beginning_time)
{
    time_t current_time;
    time(&current_time);
    return type(difftime(current_time, beginning_time));
}


void OptimizationAlgorithm::set_neural_network_variable_names()
{
    DataSet* data_set = loss_index->get_data_set();

    const vector<string> input_names = data_set->get_variable_names(DataSet::VariableUse::Input);
    const vector<string> target_names = data_set->get_variable_names(DataSet::VariableUse::Target);

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    neural_network->set_input_names(input_names);
    neural_network->set_output_namess(target_names);
}


TrainingResults::TrainingResults(const Index& epochs_number)
{
    training_error_history.resize(1 + epochs_number);
    training_error_history.setConstant(type(-1.0));

    selection_error_history.resize(1 + epochs_number);
    selection_error_history.setConstant(type(-1.0));
}


string TrainingResults::write_stopping_condition() const
{
    switch(stopping_condition)
    {
    case OptimizationAlgorithm::StoppingCondition::None:
        return "None";

    case OptimizationAlgorithm::StoppingCondition::MinimumLossDecrease:
        return "Minimum loss decrease";

    case OptimizationAlgorithm::StoppingCondition::LossGoal:
        return "Loss goal";

    case OptimizationAlgorithm::StoppingCondition::MaximumSelectionErrorIncreases:
        return "Maximum selection error increases";

    case OptimizationAlgorithm::StoppingCondition::MaximumEpochsNumber:
        return "Maximum epochs number";

    case OptimizationAlgorithm::StoppingCondition::MaximumTime:
        return "Maximum training time";

    default:
        return string();
    }
}


type TrainingResults::get_training_error()
{
    const Index size = training_error_history.size();

    return training_error_history(size - 1);
}


type TrainingResults::get_selection_error() const
{
    const Index size = selection_error_history.size();

    return selection_error_history(size - 1);
}


Index TrainingResults::get_epochs_number() const
{
    return training_error_history.size() - 1;
}


void TrainingResults::resize_training_error_history(const Index& new_size)
{
    if(training_error_history.size() == 0)
    {
        training_error_history.resize(new_size);

        return;
    }

    const Tensor<type, 1> old_training_error_history = training_error_history;

    training_error_history.resize(new_size);

    for(Index i = 0; i < new_size; i++)
        training_error_history(i) = old_training_error_history(i);
}


void TrainingResults::resize_selection_error_history(const Index& new_size)
{
    if(selection_error_history.size() == 0)
    {
        selection_error_history.resize(new_size);
        return;
    }

    const Tensor<type, 1> old_selection_error_history = selection_error_history;

    selection_error_history.resize(new_size);

    const Index minimum_size = min(new_size, old_selection_error_history.size());

    for (Index i = 0; i < minimum_size; ++i)
        selection_error_history(i) = old_selection_error_history(i);

}


string OptimizationAlgorithm::write_time(const type& time) const
{
    const int hours = int(time) / 3600;
    int seconds = int(time) % 3600;
    const int minutes = seconds / 60;
    seconds = seconds % 60;

    ostringstream elapsed_time;

    elapsed_time << setfill('0')
        << setw(2) << hours << ":"
        << setw(2) << minutes << ":"
        << setw(2) << seconds << endl;

    return elapsed_time.str();
}


void TrainingResults::save(const string& file_name) const
{
    const Tensor<string, 2> final_results = write_final_results();

    ofstream file(file_name);

    if(!file) return;

    for(Index i = 0; i < final_results.dimension(0); i++)
        file << final_results(i,0) << "; " << final_results(i,1) << "\n";

    file.close();
}


void TrainingResults::print(const string &message)
{
    const Index epochs_number = training_error_history.size();

    cout << message << endl
         << "Training results" << endl
         << "Epochs number: " << epochs_number-1 << endl
         << "Training error: " << training_error_history(epochs_number-1) << endl;

    if(abs(training_error_history(epochs_number-1) + type(1)) < type(NUMERIC_LIMITS_MIN))
        cout << "Selection error: " << selection_error_history(epochs_number-1) << endl;

    cout << "Stopping condition: " << write_stopping_condition() << endl;
}


Tensor<string, 2> TrainingResults::write_final_results(const Index& precision) const
{
    Tensor<string, 2> final_results(6, 2);

    final_results(0,0) = "Epochs number";
    final_results(1,0) = "Elapsed time";
    final_results(2,0) = "Stopping criterion";
    final_results(3,0) = "Training error";
    final_results(4,0) = "Selection error";

    const Index size = training_error_history.size();

    if(size == 0)
    {
        final_results(0,1) = "NA";
        final_results(1,1) = "NA";
        final_results(2,1) = "NA";
        final_results(3,1) = "NA";
        final_results(4,1) = "NA";

        return final_results;
    }

    final_results(0,1) = to_string(training_error_history.size()-1);
    final_results(1,1) = elapsed_time;
    final_results(2,1) = write_stopping_condition();
    final_results(3,1) = to_string(training_error_history(size-1));

    // Final selection error

    ostringstream buffer;
    buffer.str("");

    selection_error_history.size() == 0
            ? buffer << "NAN"
            : buffer << setprecision(precision) << selection_error_history(size-1);

    final_results(4,1) = buffer.str();

    return final_results;
}


OptimizationAlgorithmData::OptimizationAlgorithmData()
{
}


void OptimizationAlgorithmData::print() const
{
    cout << "Potential parameters:" << endl
         << potential_parameters << endl
         << "Training direction:" << endl
         << training_direction << endl
         << "Initial learning rate:" << endl
         << initial_learning_rate << endl;
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
