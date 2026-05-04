//   OpenNN: Open Neural Networks Library+
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "loss.h"
#include "optimizer.h"
#include "training_strategy.h"
#include "adaptive_moment_estimation.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "dense_layer.h"

namespace opennn
{

TrainingStrategy::TrainingStrategy(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    set(new_neural_network, new_dataset);
}

void TrainingStrategy::set(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    neural_network = new_neural_network;
    dataset = new_dataset;

    set_default();
}

void TrainingStrategy::set_loss(const string& new_loss)
{
    loss = make_unique<Loss>(neural_network, dataset);
    loss->set_error(new_loss);

    if (optimizer)
        optimizer->set(loss.get());
}

void TrainingStrategy::set_optimization_algorithm(const string& new_optimization_algorithm)
{
    optimizer = Registry<Optimizer>::instance().create(new_optimization_algorithm);

    optimizer->set(loss.get());
}

void TrainingStrategy::set_default()
{
    if (!get_neural_network())
        return;

    // Forecasting

    if (neural_network->has(LayerType::Recurrent))
    {
        set_loss("MeanSquaredError");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        return;
    }

    // Image Classification

    if (neural_network->has(LayerType::Convolutional))
    {
        set_loss("CrossEntropy");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        dynamic_cast<AdaptiveMomentEstimation*>(optimizer.get())->set_maximum_epochs(100);
        return;
    }

    // Transformer: signaled by *any* Dense layer that consumes a rank-2 (seq, feat)
    // input â€” i.e. the layers formerly known as Dense3d.
    bool has_seq_dense = false;
    for (const auto& layer : neural_network->get_layers())
    {
        const auto* dense = dynamic_cast<const Dense*>(layer.get());
        if (dense && dense->get_input_shape().rank == 2) { has_seq_dense = true; break; }
    }
    if (has_seq_dense)
    {
        set_loss("CrossEntropy");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        dynamic_cast<AdaptiveMomentEstimation*>(optimizer.get())->set_learning_rate(0.0001);
        return;
    }

    // Text Classification

    if (neural_network->has(LayerType::Embedding) || neural_network->has(LayerType::MultiHeadAttention))
    {
        set_loss("CrossEntropy");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        dynamic_cast<AdaptiveMomentEstimation*>(optimizer.get())->set_maximum_epochs(100);
        return;
    }

    // Classification

    const Activation::Function output_activation = neural_network->get_output_activation();

    if (output_activation == Activation::Function::Softmax)
    {
        // Multi-class
        set_loss("CrossEntropy");
        set_optimization_algorithm("QuasiNewtonMethod");
        return;
    }

    if (output_activation == Activation::Function::Sigmoid)
    {
        // Binary
        set_loss("WeightedSquaredError");
        set_optimization_algorithm("QuasiNewtonMethod");
        return;
    }

    // Approximation (default)

    set_loss("MeanSquaredError");
    set_optimization_algorithm("QuasiNewtonMethod");
}

TrainingResults TrainingStrategy::train()
{
    if (!get_neural_network())
        throw runtime_error("neural network is not set.");

    if (!get_dataset())
        throw runtime_error("dataset is not set.");

    if (!loss->get_neural_network() || !loss->get_dataset())
        throw runtime_error("loss is not set.");

    if (!optimizer->get_loss())
        throw runtime_error("optimizer is not set.");

    if (neural_network->has(LayerType::Recurrent))
        fix_forecasting();

    return optimizer->train();
}


void TrainingStrategy::fix_forecasting()
{
/*
    const Index past_time_steps = 0;

    if (neural_network->has("Recurrent"))
        past_time_steps = static_cast<Recurrent*>(neural_network->get_first(Recurrent))->get_timesteps();
    else
        return;

    Index batch_size = 0;

    if (optimization_method == OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION)
        batch_size = adaptive_moment_estimation.get_samples_number();
    else if (optimization_method == OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT)
        batch_size = stochastic_gradient_descent.get_samples_number();
    else
        return;

    if (batch_size%past_time_steps == 0)
        return;

    const Index constant = past_time_steps > batch_size
        ? 1
        : Index(batch_size/past_time_steps);

    if (optimization_method == OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION)
        adaptive_moment_estimation.set_batch_size(constant*past_time_steps);
    else if (optimization_method == OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT)
        stochastic_gradient_descent.set_batch_size(constant*past_time_steps);
*/
}

void TrainingStrategy::to_JSON(JsonWriter& printer) const
{

    printer.open_element("TrainingStrategy");

    printer.open_element("Loss");

    add_json_field(printer, "Error", loss->get_name());

    loss->to_JSON(printer);

    loss->regularization_to_JSON(printer);

    printer.close_element();

    printer.open_element("Optimizer");

    add_json_field(printer, "OptimizationMethod", optimizer->get_name());

    optimizer->to_JSON(printer);

    printer.close_element();

    add_json_field(printer, "Display", to_string(optimizer->get_display()));

    printer.close_element();
}

void TrainingStrategy::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "TrainingStrategy");

    // Loss

    const Json* loss_element = require_json_field(root_element, "Loss");

    // Loss method

    const string loss_method = read_json_string(loss_element, "Error");

    const Json* loss_method_element = loss_element->first_child(loss_method.c_str());

    if (loss_method_element)
    {
        set_loss(loss_method);
        loss->from_JSON(JsonDocument::wrap(loss_method, *loss_method_element));
    }
    else throw runtime_error(loss_method + " element is nullptr.\n");

    // Optimization algorithm

    const Json* optimization_algorithm_element = require_json_field(root_element, "Optimizer");

    // Optimization method

    const string optimization_method = read_json_string(optimization_algorithm_element, "OptimizationMethod");

    const Json* optimization_method_element = optimization_algorithm_element->first_child(optimization_method.c_str());

    if (optimization_method_element)
    {
        set_optimization_algorithm(optimization_method);
        optimizer->from_JSON(JsonDocument::wrap(optimization_method, *optimization_method_element));
    }
    else throw runtime_error(optimization_method + " element is nullptr.\n");

    // Regularization

    const Json* regularization_element = loss_element->first_child("Regularization");

    if (regularization_element)
        loss->regularization_from_JSON(JsonDocument::wrap("Regularization", *regularization_element));

    // Display

    optimizer->set_display(read_json_bool(root_element, "Display"));
}

void TrainingStrategy::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        throw runtime_error("Cannot open file: " + file_name.string());

    JsonWriter printer;
    to_JSON(printer);
    file << printer.c_str();
}

void TrainingStrategy::load(const filesystem::path& file_name)
{
    set_default();

    from_JSON(load_json_file(file_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
