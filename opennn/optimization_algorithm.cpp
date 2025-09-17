//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_dataset.h"
#include "time_series_dataset.h"
#include "pch.h"
#include "optimization_algorithm.h"
#include "scaling_layer_2d.h"
#include "scaling_layer_3d.h"
#include "unscaling_layer.h"
#include "language_dataset.h"
#include "transformer.h"
#include "loss_index.h"

namespace opennn
{

OptimizationAlgorithm::OptimizationAlgorithm(const LossIndex* new_loss_index)
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


void OptimizationAlgorithm::set(const LossIndex* new_loss_index)
{
    loss_index = const_cast<LossIndex*>(new_loss_index);
}


void OptimizationAlgorithm::set_threads_number(const int& new_threads_number)
{
    thread_pool.reset();
    thread_pool_device.reset();

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


void OptimizationAlgorithm::check() const
{
    if (!loss_index)
        throw runtime_error("loss_index is nullptr.\n");
}


string OptimizationAlgorithm::get_name() const
{
    return string();
}


void OptimizationAlgorithm::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("OptimizationAlgorithm");

    add_xml_element(printer, "Display", to_string(display));

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


void OptimizationAlgorithm::save(const filesystem::path& file_name) const
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
            throw runtime_error("Cannot open file: " + file_name.string());
        }
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;
    }
}


void OptimizationAlgorithm::load(const filesystem::path& file_name)
{
    XMLDocument document;

    if (document.LoadFile(file_name.string().c_str()))
        throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

    from_XML(document);
}


type OptimizationAlgorithm::get_elapsed_time(const time_t &beginning_time)
{
    time_t current_time;
    time(&current_time);
    return type(difftime(current_time, beginning_time));
}


void OptimizationAlgorithm::set_names()
{
    Dataset* dataset = loss_index->get_dataset();

    const vector<string> input_names = dataset->get_variable_names("Input");
    const vector<string> target_names = dataset->get_variable_names("Target");

    const Index input_variables_number = dataset->get_variables_number("Input");
    const Index target_variables_number = dataset->get_variables_number("Target");

    const vector<Dataset::RawVariable> raw_variables = dataset->get_raw_variables();
    const vector<Index> input_variable_indices = dataset->get_raw_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_raw_variable_indices("Input");

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    vector<string> input_variable_names;
    vector<string> target_variable_names;

    TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);

    for(Index i = 0; i < input_variables_number; i++)
    {
        if(time_series_dataset)
        {
            const Index time_steps = time_series_dataset->get_past_time_steps();

            if(input_names[i] == "")
                for(Index j = 0; j < time_steps; j++)
                    input_variable_names.push_back("variable_" + to_string(i + 1) + "_lag" + to_string(j));
            else
                for(Index j = 0; j < time_steps; j++)
                    input_variable_names.push_back(input_names[i] + "_lag" + to_string(j));
        }
        else
        {
            if(input_names[i] == "")
                input_variable_names.push_back("variable_" + to_string(i + 1));
            else
                input_variable_names.push_back(input_names[i]);
        }
    }

    for(Index i = 0; i < target_variables_number; i++)
    {
        if(target_names[i] == "")
        {
            auto input_iterator = find(input_variable_indices.begin(), input_variable_indices.end(), target_variable_indices[i]);

            if(input_iterator == input_variable_indices.end())
                target_variable_names.push_back("variable_" + to_string(input_variables_number + i + 1));
            else
                target_variable_names.push_back("variable_" + to_string(int(*input_iterator) + 1));
        }
        else
            target_variable_names.push_back(target_names[i]);
    }

    neural_network->set_input_names(input_variable_names);
    neural_network->set_output_names(target_variable_names);
}


void OptimizationAlgorithm::set_scaling()
{
    Dataset* dataset = loss_index->get_dataset();
    NeuralNetwork* neural_network = loss_index->get_neural_network();

    vector<Descriptives> input_descriptives;
    vector<Scaler> input_scalers;
    bool input_has_been_scaled = false;

    if (neural_network->has("Scaling2d"))
    {
        input_scalers = dataset->get_variable_scalers("Input");
        input_descriptives = dataset->scale_variables("Input");
        input_has_been_scaled = true;

        Scaling2d* scaling_layer_2d = static_cast<Scaling2d*>(neural_network->get_first("Scaling2d"));
        scaling_layer_2d->set_descriptives(input_descriptives);
        scaling_layer_2d->set_scalers(input_scalers);
    }
    else if (neural_network->has("Scaling3d"))
    {
        TimeSeriesDataset* time_series_dataset = static_cast<TimeSeriesDataset*>(dataset);
        input_scalers = time_series_dataset->get_variable_scalers("Input");
        input_descriptives = time_series_dataset->scale_variables("Input");
        input_has_been_scaled = true;

        Scaling3d* scaling_layer_3d = static_cast<Scaling3d*>(neural_network->get_first("Scaling3d"));
        scaling_layer_3d->set_descriptives(input_descriptives);
        scaling_layer_3d->set_scalers(input_scalers);
    }
    else if (neural_network->has("Scaling4d"))
    {
        ImageDataset* image_dataset = static_cast<ImageDataset*>(dataset);
        image_dataset->scale_variables("Input");
        input_has_been_scaled = true;
    }

    if (neural_network->has("Unscaling"))
    {
        Unscaling* unscaling_layer = static_cast<Unscaling*>(neural_network->get_first("Unscaling"));

        const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
        const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");

        vector<Descriptives> target_descriptives;
        vector<Scaler> target_scalers;

        for (const Index& target_index : target_variable_indices)
        {
            auto target_position_inputs = std::find(input_variable_indices.begin(), input_variable_indices.end(), target_index);

            if (target_position_inputs != input_variable_indices.end())
            {
                if (!input_has_been_scaled)
                    throw std::runtime_error("Configuration error: Unscaling layer exists for a target that is also an input, but no input scaling layer was found.");

                const Index input_index = std::distance(input_variable_indices.begin(), target_position_inputs);
                target_descriptives.push_back(input_descriptives[input_index]);
                target_scalers.push_back(input_scalers[input_index]);
            }
            else
            {
                const vector<Descriptives> all_target_descriptives = dataset->scale_variables("Target");
                const vector<Scaler> all_target_scalers = dataset->get_variable_scalers("Target");

                auto target_position_targets = std::find(target_variable_indices.begin(), target_variable_indices.end(), target_index);
                const Index target_position = std::distance(target_variable_indices.begin(), target_position_targets);

                target_descriptives.push_back(all_target_descriptives[target_position]);
                target_scalers.push_back(all_target_scalers[target_position]);
            }
        }

        if (static_cast<Index>(target_descriptives.size()) != unscaling_layer->get_outputs_number()) {
            throw std::runtime_error("Unscaling setup error: Mismatch between number of target variables and unscaling layer neurons.");
        }

        unscaling_layer->set_descriptives(target_descriptives);
        unscaling_layer->set_scalers(target_scalers);
    }
}


void OptimizationAlgorithm::set_unscaling()
{
    Dataset* dataset = loss_index->get_dataset();
    NeuralNetwork* neural_network = loss_index->get_neural_network();

    if (neural_network->has("Scaling2d"))
    {
        Scaling2d* layer = static_cast<Scaling2d*>(neural_network->get_first("Scaling2d"));
        dataset->unscale_variables("Input", layer->get_descriptives());
    }
    else if (neural_network->has("Scaling3d"))
    {
        //Scaling3d* layer = static_cast<Scaling3d*>(neural_network->get_first("Scaling3d"));
        //dataset->unscale_variables("Input", layer->get_descriptives());
    }
    else if (neural_network->has("Scaling4d"))
    {
        ImageDataset* image_dataset = static_cast<ImageDataset*>(dataset);
        image_dataset->unscale_variables("Input");
    }

    if (neural_network->has("Unscaling"))
    {
        const vector<Index> input_indices = dataset->get_variable_indices("Input");
        const vector<Index> target_indices = dataset->get_variable_indices("Target");

        vector<Descriptives> targets_to_unscale_descriptives;

        Unscaling* unscaling_layer = static_cast<Unscaling*>(neural_network->get_first("Unscaling"));
        const vector<Descriptives> all_unscaling_descriptives = unscaling_layer->get_descriptives();

        for (size_t i = 0; i < target_indices.size(); ++i) {
            const Index& target_index = target_indices[i];

            auto target_position = std::find(input_indices.begin(), input_indices.end(), target_index);

            if (target_position == input_indices.end())
                targets_to_unscale_descriptives.push_back(all_unscaling_descriptives[i]);
        }

        if (!targets_to_unscale_descriptives.empty())
            dataset->unscale_variables("Target", all_unscaling_descriptives);
    }
}


void OptimizationAlgorithm::set_vocabularies()
{
    Dataset* dataset = loss_index->get_dataset();

    if(!is_instance_of<LanguageDataset>(dataset))
        return;

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    if(!is_instance_of<Transformer>(neural_network))
        return;

    LanguageDataset* language_dataset = static_cast<LanguageDataset*>(dataset);

    const vector<string>& input_vocabulary = language_dataset->get_input_vocabulary();
    const vector<string>& target_vocabulary = language_dataset->get_target_vocabulary();

    Transformer* transformer = static_cast<Transformer*>(neural_network);

    transformer->set_input_vocabulary(input_vocabulary);
    transformer->set_output_vocabulary(target_vocabulary);
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


type TrainingResults::get_training_error() const
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

    const Index copy_size = min(old_training_error_history.size(), new_size);

    for(Index i = 0; i < copy_size; i++)
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
                 << setw(2) << seconds;

    return elapsed_time.str();
}


void TrainingResults::save(const filesystem::path& file_name) const
{
    const Tensor<string, 2> override_results = write_override_results();

    ofstream file(file_name);

    if(!file) return;

    for(Index i = 0; i < override_results.dimension(0); i++)
        file << override_results(i,0) << "; " << override_results(i,1) << "\n";

    file.close();
}


void TrainingResults::print(const string &message) const
{
    const Index epochs_number = training_error_history.size();

    cout << message << endl
         << "Training results" << endl
         << "Epochs number: " << epochs_number - 1 << endl
         << "Training error: " << training_error_history(epochs_number - 1) << endl;
    if (selection_error_history.size() > 0)
        cout << "Selection error: " << selection_error_history(epochs_number - 1) << endl;
    cout << "Stopping condition: " << write_stopping_condition() << endl;
}


Tensor<string, 2> TrainingResults::write_override_results(const Index& precision) const
{
    Tensor<string, 2> override_results(5, 2);

    override_results(0,0) = "Epochs number";
    override_results(1,0) = "Elapsed time";
    override_results(2,0) = "Stopping criterion";
    override_results(3,0) = "Training error";
    override_results(4,0) = "Selection error";

    const Index size = training_error_history.size();

    if(size == 0)
    {
        override_results(0,1) = "NA";
        override_results(1,1) = "NA";
        override_results(2,1) = "NA";
        override_results(3,1) = "NA";
        override_results(4,1) = "NA";

        return override_results;
    }

    override_results(0,1) = to_string(training_error_history.size()-1);
    override_results(1,1) = elapsed_time;
    override_results(2,1) = write_stopping_condition();
    override_results(3,1) = to_string(training_error_history(size-1));

    // Final selection error

    ostringstream buffer;
    buffer.str("");

    selection_error_history.size() == 0
        ? buffer << "NAN"
        : buffer << setprecision(precision) << selection_error_history(size-1);

    override_results(4,1) = buffer.str();

    return override_results;
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
