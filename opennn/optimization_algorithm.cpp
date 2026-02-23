//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "image_dataset.h"
#include "time_series_dataset.h"
#include "language_dataset.h"
#include "scaling_layer.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"
#include "loss_index.h"
#include "optimization_algorithm.h"

namespace opennn
{

Optimizer::Optimizer(const Loss* new_loss)
{
    set(new_loss);
}


Loss* Optimizer::get_loss_index() const
{
    return loss_index;
}


string Optimizer::get_hardware_use() const
{
    return hardware_use;
}


void Optimizer::set_hardware_use(const string& new_hardware_use)
{
    hardware_use = new_hardware_use;
}


bool Optimizer::has_loss_index() const
{
    return loss_index;
}


bool Optimizer::get_display() const
{
    return display;
}


Index Optimizer::get_display_period() const
{
    return display_period;
}


Index Optimizer::get_save_period() const
{
    return save_period;
}


const string& Optimizer::get_neural_network_file_name() const
{
    return neural_network_file_name;
}


void Optimizer::set(const Loss* new_loss)
{
    loss_index = const_cast<Loss*>(new_loss);
}


void Optimizer::set_loss_index(Loss* new_loss)
{
    loss_index = new_loss;
}


void Optimizer::set_display(bool new_display)
{
    display = new_display;
}


void Optimizer::set_display_period(const Index new_display_period)
{
    display_period = new_display_period;
}


void Optimizer::set_save_period(const Index new_save_period)
{
    save_period = new_save_period;
}


void Optimizer::set_neural_network_file_name(const string& new_neural_network_file_name)
{
    neural_network_file_name = new_neural_network_file_name;
}


void Optimizer::check() const
{
    if(!loss_index)
        throw runtime_error("loss_index is nullptr.\n");
}


string Optimizer::get_name() const
{
    return name;
}


void Optimizer::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Optimizer");

    add_xml_element(printer, "Display", to_string(display));

    printer.CloseElement();
}


void Optimizer::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("Optimizer");

    if(!root_element)
        throw runtime_error("Optimization algorithm element is nullptr.\n");

    set_display(read_xml_bool(root_element, "Display"));
}


Tensor<string, 2> Optimizer::to_string_matrix() const
{
    return Tensor<string, 2>();
}


void Optimizer::print() const
{
}


void Optimizer::save(const filesystem::path& file_name) const
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


void Optimizer::load(const filesystem::path& file_name)
{
    XMLDocument document;

    if (document.LoadFile(file_name.string().c_str()))
        throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

    from_XML(document);
}


type Optimizer::get_elapsed_time(const time_t &beginning_time)
{
    time_t current_time;
    time(&current_time);
    return type(difftime(current_time, beginning_time));
}


void Optimizer::set_names()
{
    Dataset* dataset = loss_index->get_dataset();

    const vector<string> input_names = dataset->get_feature_names("Input");
    const vector<string> target_names = dataset->get_feature_names("Target");

    const Index input_features_number = dataset->get_features_number("Input");
    const Index target_features_number = dataset->get_features_number("Target");

    const vector<Dataset::Variable> variables = dataset->get_variables();
    const vector<Index> input_feature_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_variable_indices("Input");

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    vector<string> input_variable_names;
    vector<string> target_variable_names;

    TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);

    for(Index i = 0; i < input_features_number; i++)
    {
        if(time_series_dataset)
        {
            const Index time_steps = time_series_dataset->get_past_time_steps();

            if(input_names[i].empty())
                for(Index j = 0; j < time_steps; j++)
                    input_variable_names.push_back("variable_" + to_string(i + 1) + "_lag" + to_string(j));
            else
                for(Index j = 0; j < time_steps; j++)
                    input_variable_names.push_back(input_names[i] + "_lag" + to_string(j));
        }
        else
        {
            if(input_names[i].empty())
                input_variable_names.push_back("variable_" + to_string(i + 1));
            else
                input_variable_names.push_back(input_names[i]);
        }
    }

    for(Index i = 0; i < target_features_number; i++)
    {
        string current_target_name;

        if(target_names[i].empty())
        {
            auto input_iterator = find(input_feature_indices.begin(), input_feature_indices.end(), target_feature_indices[i]);

            if(input_iterator == input_feature_indices.end())
                current_target_name = "variable_" + to_string(input_features_number + i + 1);
            else
                current_target_name = "variable_" + to_string(int(*input_iterator) + 1);
        }
        else
            current_target_name = target_names[i];

        if(time_series_dataset)
            current_target_name += "_ahead";

        target_variable_names.push_back(current_target_name);
    }

    neural_network->set_feature_names(input_variable_names);
    neural_network->set_output_names(target_variable_names);
}


void Optimizer::set_scaling()
{
    Dataset* dataset = loss_index->get_dataset();
    NeuralNetwork* neural_network = loss_index->get_neural_network();

    // Scaling layer

    vector<Descriptives> input_variable_descriptives;
    vector<string> input_variable_scalers;

    if(neural_network->has("Scaling2d"))
    {
        input_variable_scalers = dataset->get_feature_scalers("Input");
        input_variable_descriptives = dataset->scale_features("Input");

        Scaling<2>* scaling_layer = static_cast<Scaling<2>*>(neural_network->get_first("Scaling2d"));
        scaling_layer->set_descriptives(input_variable_descriptives);
        scaling_layer->set_scalers(input_variable_scalers);
    }
    else if(neural_network->has("Scaling3d"))
    {
        TimeSeriesDataset* time_series_dataset = static_cast<TimeSeriesDataset*>(dataset);
        input_variable_scalers = time_series_dataset->get_feature_scalers("Input");
        input_variable_descriptives = time_series_dataset->scale_features("Input");

        Scaling<3>* scaling_layer = static_cast<Scaling<3>*>(neural_network->get_first("Scaling3d"));
        scaling_layer->set_descriptives(input_variable_descriptives);
        scaling_layer->set_scalers(input_variable_scalers);
    }
    else if (neural_network->has("Scaling4d"))
    {
        ImageDataset* image_dataset = static_cast<ImageDataset*>(dataset);

        image_dataset->scale_features("Input");

        if (neural_network->get_first("Scaling4d"))
        {
            Scaling<4>* scaling_layer = static_cast<Scaling<4>*>(neural_network->get_first("Scaling4d"));
            scaling_layer->set_scalers("ImageMinMax");
        }
    }

    if(!neural_network->has("Unscaling"))
        return;

    // Unscaling layer

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    bool has_pure_targets = false;
    for(const Index t_idx : target_feature_indices)
    {
        bool is_input = false;
        for(const Index i_idx : input_feature_indices)
        {
            if(t_idx == i_idx)
            {
                is_input = true;
                break;
            }
        }
        if(!is_input)
        {
            has_pure_targets = true;
            break;
        }
    }

    vector<Descriptives> target_variable_descriptives;
    vector<string> target_variable_scalers;

    if(has_pure_targets)
    {
        target_variable_descriptives = dataset->scale_features("Target");
        target_variable_scalers = dataset->get_feature_scalers("Target");
    }

    vector<Descriptives> unscaling_layer_descriptives;
    vector<string> unscaling_layer_scalers;

    for(size_t i = 0; i < target_feature_indices.size(); ++i)
    {
        Index target_index = target_feature_indices[i];

        auto it = find(input_feature_indices.begin(), input_feature_indices.end(), target_index);

        if(it != input_feature_indices.end())
        {
            const Index input_pos = distance(input_feature_indices.begin(), it);

            unscaling_layer_descriptives.push_back(input_variable_descriptives[input_pos]);
            unscaling_layer_scalers.push_back(input_variable_scalers[input_pos]);
        }
        else
        {
            unscaling_layer_descriptives.push_back(target_variable_descriptives[i]);
            unscaling_layer_scalers.push_back(target_variable_scalers[i]);
        }
    }

    Unscaling* unscaling_layer = static_cast<Unscaling*>(neural_network->get_first("Unscaling"));

    if(static_cast<Index>(unscaling_layer_descriptives.size()) != unscaling_layer->get_outputs_number())
        throw runtime_error("Unscaling setup error: Mismatch between number of target variables and unscaling layer neurons.");

    unscaling_layer->set_descriptives(unscaling_layer_descriptives);
    unscaling_layer->set_scalers(unscaling_layer_scalers);
}


void Optimizer::set_unscaling()
{
    Dataset* dataset = loss_index->get_dataset();
    NeuralNetwork* neural_network = loss_index->get_neural_network();

    // Scaling layer

    if(neural_network->has("Scaling2d"))
    {
        Scaling<2>* layer = static_cast<Scaling<2>*>(neural_network->get_first("Scaling2d"));
        dataset->unscale_features("Input", layer->get_descriptives());
    }
    else if(neural_network->has("Scaling3d"))
    {
        Scaling<3>* layer = static_cast<Scaling<3>*>(neural_network->get_first("Scaling3d"));
        dataset->unscale_features("Input", layer->get_descriptives());
    }
    else if(neural_network->has("Scaling4d"))
    {
        ImageDataset* image_dataset = static_cast<ImageDataset*>(dataset);
        image_dataset->unscale_features("Input");
    }

    if(!neural_network->has("Unscaling"))
        return;

    // Unscaling layer

    const Unscaling* unscaling_layer = static_cast<Unscaling*>(neural_network->get_first("Unscaling"));
    const vector<Descriptives>& all_target_descriptives = unscaling_layer->get_descriptives();

    const vector<Index> input_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_indices = dataset->get_feature_indices("Target");

    vector<Descriptives> unscaled_targets_descriptives;

    for(size_t i = 0; i < target_indices.size(); ++i)
    {
        bool is_input = false;

        for(const Index input_idx : input_indices)
        {
            if(target_indices[i] == input_idx)
            {
                is_input = true;
                break;
            }
        }

        if(!is_input)
            unscaled_targets_descriptives.push_back(all_target_descriptives[i]);
    }

    if(!unscaled_targets_descriptives.empty())
        dataset->unscale_features("Target", unscaled_targets_descriptives);
}


void Optimizer::set_vocabularies()
{
    Dataset* dataset = loss_index->get_dataset();

    if(!is_instance_of<LanguageDataset>(dataset))
        return;

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    if(!neural_network->has("Embedding"))
        return;

    LanguageDataset* language_dataset = static_cast<LanguageDataset*>(dataset);

    const vector<string>& input_vocabulary = language_dataset->get_input_vocabulary();
    const vector<string>& target_vocabulary = language_dataset->get_target_vocabulary();

    neural_network->set_input_vocabulary(input_vocabulary);
    neural_network->set_output_vocabulary(target_vocabulary);
}


TrainingResults::TrainingResults(const Index epochs_number)
{
    training_error_history.resize(1 + epochs_number);
    training_error_history.setConstant(type(-1.0));

    validation_error_history.resize(1 + epochs_number);
    validation_error_history.setConstant(type(-1.0));
}


string TrainingResults::write_stopping_condition() const
{
    switch(stopping_condition)
    {
    case Optimizer::StoppingCondition::None:
        return "None";

    case Optimizer::StoppingCondition::MinimumLossDecrease:
        return "Minimum loss decrease";

    case Optimizer::StoppingCondition::LossGoal:
        return "Loss goal";

    case Optimizer::StoppingCondition::MaximumSelectionErrorIncreases:
        return "Maximum selection error increases";

    case Optimizer::StoppingCondition::MaximumEpochsNumber:
        return "Maximum epochs number";

    case Optimizer::StoppingCondition::MaximumTime:
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


type TrainingResults::get_validation_error() const
{
    const Index size = validation_error_history.size();

    return validation_error_history(size - 1);
}


Index TrainingResults::get_epochs_number() const
{
    return training_error_history.size() - 1;
}


void TrainingResults::resize_training_error_history(const Index new_size)
{
    if(training_error_history.size() == 0)
    {
        training_error_history.resize(new_size);

        return;
    }

    const VectorR old_training_error_history = training_error_history;

    training_error_history.resize(new_size);

    const Index copy_size = min(old_training_error_history.size(), new_size);

    for(Index i = 0; i < copy_size; i++)
        training_error_history(i) = old_training_error_history(i);
}


void TrainingResults::resize_validation_error_history(const Index new_size)
{
    if(validation_error_history.size() == 0)
    {
        validation_error_history.resize(new_size);
        return;
    }

    const VectorR old_validation_error_history = validation_error_history;

    validation_error_history.resize(new_size);

    const Index minimum_size = min(new_size, old_validation_error_history.size());

    for(Index i = 0; i < minimum_size; ++i)
        validation_error_history(i) = old_validation_error_history(i);

}


string Optimizer::write_time(const type time) const
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
    if (validation_error_history.size() > 0)
        cout << "Validation error: " << validation_error_history(epochs_number - 1) << endl;
    cout << "Stopping condition: " << write_stopping_condition() << endl;
}


Tensor<string, 2> TrainingResults::write_override_results(const Index precision) const
{
    Tensor<string, 2> override_results(5, 2);

    override_results(0,0) = "Epochs number";
    override_results(1,0) = "Elapsed time";
    override_results(2,0) = "Stopping criterion";
    override_results(3,0) = "Training error";
    override_results(4,0) = "Validation error";

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

    validation_error_history.size() == 0
        ? buffer << "NAN"
        : buffer << setprecision(precision) << validation_error_history(size-1);

    override_results(4,1) = buffer.str();

    return override_results;
}


OptimizerData::OptimizerData()
{
}


void OptimizerData::print() const
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
