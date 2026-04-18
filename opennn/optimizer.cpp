//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_dataset.h"
#include "time_series_dataset.h"
#include "language_dataset.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"
#include "loss.h"
#include "optimizer.h"
#include "variable.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Optimizer::Optimizer(Loss* new_loss)
{
    set(new_loss);
}

void Optimizer::check() const
{
    if(!loss)
        throw runtime_error("Optimizer error: loss is not set.");
}

void Optimizer::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Optimizer");

    add_xml_element(printer, "Display", to_string(display));

    printer.close_element();
}

void Optimizer::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "Optimizer");

    set_display(read_xml_bool(root_element, "Display"));
}

void Optimizer::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open file: " + file_name.string());

    XmlPrinter printer;
    to_XML(printer);
    file << printer.c_str();
}

void Optimizer::load(const filesystem::path& file_name)
{
    from_XML(load_xml_file(file_name));
}

type Optimizer::get_elapsed_time(const time_t &beginning_time)
{
    time_t current_time;
    time(&current_time);
    return type(difftime(current_time, beginning_time));
}

void Optimizer::set_names()
{
    check();

    const Dataset* dataset = loss->get_dataset();

    const vector<Variable> input_variables = dataset->get_variables("Input");
    const vector<Variable> target_variables = dataset->get_variables("Target");

    NeuralNetwork* neural_network = loss->get_neural_network();

    neural_network->set_input_variables(input_variables);
    neural_network->set_output_variables(target_variables);
}

void Optimizer::set_scaling()
{
    check();

    Dataset* dataset = loss->get_dataset();
    NeuralNetwork* neural_network = loss->get_neural_network();

    // Scaling layer

    vector<Descriptives> input_variable_descriptives;
    vector<string> input_variable_scalers;

    if(neural_network->has(LayerType::Scaling2d))
    {
        input_variable_scalers = dataset->get_feature_scalers("Input");
        input_variable_descriptives = dataset->scale_features("Input");

        auto* scaling_layer = dynamic_cast<Scaling<2>*>(neural_network->get_first(LayerType::Scaling2d));
        if(!scaling_layer) throw runtime_error("Expected Scaling<2> layer.");
        scaling_layer->set_descriptives(input_variable_descriptives);
        scaling_layer->set_scalers(input_variable_scalers);
    }
    else if(neural_network->has(LayerType::Scaling3d))
    {
        auto* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);
        if(!time_series_dataset) throw runtime_error("Expected TimeSeriesDataset.");
        input_variable_scalers = time_series_dataset->get_feature_scalers("Input");
        input_variable_descriptives = time_series_dataset->scale_features("Input");

        auto* scaling_layer = dynamic_cast<Scaling<3>*>(neural_network->get_first(LayerType::Scaling3d));
        if(!scaling_layer) throw runtime_error("Expected Scaling<3> layer.");
        scaling_layer->set_descriptives(input_variable_descriptives);
        scaling_layer->set_scalers(input_variable_scalers);
    }
    else if (neural_network->has(LayerType::Scaling4d))
    {
        auto* image_dataset = dynamic_cast<ImageDataset*>(dataset);
        if(!image_dataset) throw runtime_error("Expected ImageDataset.");

        image_dataset->scale_features("Input");

        auto* scaling_layer = dynamic_cast<Scaling<4>*>(neural_network->get_first(LayerType::Scaling4d));
        if(scaling_layer)
            scaling_layer->set_scalers("ImageMinMax");
    }

    if(!neural_network->has(LayerType::Unscaling))
        return;

    // Unscaling layer

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    const bool has_pure_targets = any_of(target_feature_indices.begin(), target_feature_indices.end(),
        [&](Index t) { return find(input_feature_indices.begin(), input_feature_indices.end(), t) == input_feature_indices.end(); });

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
        const Index target_index = target_feature_indices[i];

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

    auto* unscaling_layer = dynamic_cast<Unscaling*>(neural_network->get_first(LayerType::Unscaling));
    if(!unscaling_layer) throw runtime_error("Expected Unscaling layer.");

    if(ssize(unscaling_layer_descriptives) != unscaling_layer->get_outputs_number())
        throw runtime_error("Unscaling setup error: Mismatch between number of target variables and unscaling layer neurons.");

    unscaling_layer->set_descriptives(unscaling_layer_descriptives);
    unscaling_layer->set_scalers(unscaling_layer_scalers);
}

void Optimizer::set_unscaling()
{
    check();

    const Dataset* dataset = loss->get_dataset();
    const NeuralNetwork* neural_network = loss->get_neural_network();
/*
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
*/
}

bool Optimizer::check_stopping_condition(TrainingResults& results,
                                          const Index epoch,
                                          const type elapsed_time,
                                          const type training_error,
                                          const Index validation_failures) const
{
    if(training_error < training_loss_goal)
    {
        if(display) cout << "Epoch " << epoch << "\nLoss goal reached: " << training_error << "\n";
        results.stopping_condition = StoppingCondition::LossGoal;
    }
    else if(validation_failures >= maximum_validation_failures)
    {
        if(display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << validation_failures << "\n";
        results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
    }
    else if(epoch == maximum_epochs)
    {
        if(display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch << "\n";
        results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
    }
    else if(elapsed_time >= maximum_time)
    {
        if(display) cout << "Epoch " << epoch << "\nMaximum training time reached: " << get_time(elapsed_time) << "\n";
        results.stopping_condition = StoppingCondition::MaximumTime;
    }
    else
    {
        return false;
    }

    return true;
}

void Optimizer::write_common_xml(XmlPrinter& printer) const
{
    write_xml_properties(printer, {
        {"LossGoal", to_string(training_loss_goal)},
        {"MaximumSelectionFailures", to_string(maximum_validation_failures)},
        {"MaximumEpochsNumber", to_string(maximum_epochs)},
        {"MaximumTime", to_string(maximum_time)}
    });
}

void Optimizer::read_common_xml(const XmlElement* root_element)
{
    set_loss_goal(read_xml_type(root_element, "LossGoal"));
    set_maximum_validation_failures(read_xml_index(root_element, "MaximumSelectionFailures"));
    set_maximum_epochs(read_xml_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
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
    training_error_history.conservativeResize(new_size);
}

void TrainingResults::resize_validation_error_history(const Index new_size)
{
    validation_error_history.conservativeResize(new_size);
}

void TrainingResults::save(const filesystem::path& file_name) const
{
    const Tensor<string, 2> override_results = write_override_results();

    ofstream file(file_name);

    if(!file) return;

    for(Index i = 0; i < override_results.dimension(0); ++i)
        file << override_results(i,0) << "; " << override_results(i,1) << "\n";

    file.close();
}

void TrainingResults::print(const string &message) const
{
    const Index epochs_number = training_error_history.size();

    cout << message << "\n"
         << "Training results" << "\n"
         << "Epochs number: " << epochs_number - 1 << "\n"
         << "Training error: " << training_error_history(epochs_number - 1) << "\n";
    if (validation_error_history.size() > 0)
        cout << "Validation error: " << validation_error_history(epochs_number - 1) << "\n";
    cout << "Stopping condition: " << write_stopping_condition() << "\n";
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

void OptimizerData::print() const
{
    cout << "Potential parameters:" << "\n"
         << potential_parameters << "\n"
         << "Training direction:" << "\n"
         << training_direction << "\n"
         << "Initial learning rate:" << "\n"
         << initial_learning_rate << "\n";
}



TrainingResults Optimizer::train_cuda()
{
    throw runtime_error("train_cuda() is not implemented for " + name + ". "
                        "Use an optimizer with GPU support (AdaptiveMomentEstimation or StochasticGradientDescent).");
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
