//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <fstream>
//#include <algorithm>
//#include <functional>
//#include <limits>
#include <cmath>
//#include <ctime>
#include <iomanip>

#include "tensors.h"
#include "optimization_algorithm.h"

namespace opennn
{

OptimizationAlgorithm::OptimizationAlgorithm()
{
    const int n = omp_get_max_threads();
    thread_pool = new ThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(thread_pool, n);

    set_default();
}


OptimizationAlgorithm::OptimizationAlgorithm(LossIndex* new_loss_index)
    : loss_index(new_loss_index)
{
    const int n = omp_get_max_threads();
    thread_pool = new ThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(thread_pool, n);

    set_default();
}


OptimizationAlgorithm::~OptimizationAlgorithm()
{
    delete thread_pool;
    delete thread_pool_device;
}


LossIndex* OptimizationAlgorithm::get_loss_index() const
{
#ifdef OPENNN_DEBUG

    if(!loss_index)
        throw runtime_error("Loss index pointer is nullptr.\n");

#endif

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
    if(loss_index)
    {
        return true;
    }
    else
    {
        return false;
    }
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


void OptimizationAlgorithm::set()
{
    loss_index = nullptr;

    set_default();
}


void OptimizationAlgorithm::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr) delete thread_pool;
    if(thread_pool_device != nullptr) delete thread_pool_device;

    thread_pool = new ThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(thread_pool, new_threads_number);
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


BoxPlot OptimizationAlgorithm::calculate_distances_box_plot(type* & new_inputs_data, Tensor<Index,1>& input_dimensions,
                                                            type* & new_outputs_data, Tensor<Index,1>& output_dimensions)
{
    const Index samples_number = input_dimensions(0);
    const Index inputs_number = input_dimensions(1);

    TensorMap<Tensor<type, 2>> inputs(new_inputs_data, samples_number, inputs_number);
    TensorMap<Tensor<type, 2>> outputs(new_outputs_data, output_dimensions[0], output_dimensions(1));

    Tensor<type, 1> distances(samples_number);
    Index distance_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        Tensor<type, 1> input_row = inputs.chip(i, 0);
        Tensor<type, 1> output_row = outputs.chip(i, 0);

        const type distance = l2_distance(input_row, output_row)/inputs_number;

        if(!isnan(distance))
        {
            distances(distance_index) = l2_distance(input_row, output_row)/inputs_number;
            distance_index++;
        }
    }

    return box_plot(distances);
}


void OptimizationAlgorithm::set_default()
{
    display = true;

    display_period = 10;

    save_period = UINT_MAX;

    neural_network_file_name = "neural_network.xml";
}


void OptimizationAlgorithm::check() const
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(!loss_index)
        throw runtime_error("Pointer to loss index is nullptr.\n");

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    if(neural_network == nullptr)
        throw runtime_error("Pointer to neural network is nullptr.\n");

#endif
}


void OptimizationAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("OptimizationAlgorithm");

    // Display

    file_stream.OpenElement("Display");

    buffer.str("");
    buffer << display;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    file_stream.CloseElement();
}


void OptimizationAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("OptimizationAlgorithm");

    if(!root_element)
        throw runtime_error("Optimization algorithm element is nullptr.\n");

    // Display
    {
        const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

        if(display_element)
        {
            const string new_display_string = display_element->GetText();

            try
            {
                set_display(new_display_string != "0");
            }
            catch(const exception& e)
            {
                cerr << e.what() << endl;
            }
        }
    }
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
    FILE * file = fopen(file_name.c_str(), "w");

    if(file)
    {
        tinyxml2::XMLPrinter printer(file);
        write_XML(printer);
        fclose(file);
    }
}


void OptimizationAlgorithm::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
        throw runtime_error("Cannot load XML file " + file_name + ".\n");

    from_XML(document);
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
    case OptimizationAlgorithm::StoppingCondition::MinimumLossDecrease:
        return "Minimum loss decrease";

    case OptimizationAlgorithm::StoppingCondition::LossGoal:
        return "Loss goal";

    case OptimizationAlgorithm::StoppingCondition::MaximumSelectionErrorIncreases:
        return "Maximum selection error increases";

    case OptimizationAlgorithm::StoppingCondition::MaximumEpochsNumber:
        return "Maximum number of epochs";

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
    {
        training_error_history(i) = old_training_error_history(i);
    }
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

    for(Index i = 0; i < new_size; i++)
    {
        selection_error_history(i) = old_selection_error_history(i);
    }
}


string OptimizationAlgorithm::write_time(const type& time) const
{

#ifdef OPENNN_DEBUG

    if(time > type(3600e5))
        throw runtime_error("Time must be lower than 10e5 seconds.\n");

    if(time < type(0))
        throw runtime_error("Time must be greater than 0.\n");

#endif

    const int hours = int(time) / 3600;
    int seconds = int(time) % 3600;
    const int minutes = seconds / 60;
    seconds = seconds % 60;

    ostringstream elapsed_time;

    elapsed_time << setfill('0') << setw(2) << hours << ":"
                 << setfill('0') << setw(2) << minutes << ":"
                 << setfill('0') << setw(2) << seconds;

    return elapsed_time.str();
}


void TrainingResults::save(const string& file_name) const
{
    Tensor<string, 2> final_results = write_final_results();

    ofstream file;
    file.open(file_name);

    if(file)
    {
        for(Index i = 0; i < final_results.dimension(0); i++)
        {
            file << final_results(i,0) << "; " << final_results(i,1) << "\n";
        }

        file.close();
    }

}


Tensor<string, 2> TrainingResults::write_final_results(const Index& precision) const
{
    ostringstream buffer;

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

    // Epochs number

    buffer.str("");
    buffer << training_error_history.size()-1;

    final_results(0,1) = buffer.str();

    // Elapsed time

    buffer.str("");
    buffer << setprecision(precision) << elapsed_time;

    final_results(1,1) = buffer.str();

    // Stopping criteria

    final_results(2,1) = write_stopping_condition();

    // Final training error

    buffer.str("");
    buffer << setprecision(precision) << training_error_history(size-1);

    final_results(3,1) = buffer.str();

    // Final selection error

    buffer.str("");

    selection_error_history.size() == 0
            ? buffer << "NAN"
                        : buffer << setprecision(precision) << selection_error_history(size-1);

    final_results(4,1) = buffer.str();


    return final_results;
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
