//   OpenNN: Open Neural Networks Library
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


    if (neural_network->has(LayerType::Recurrent)
        || neural_network->has(LayerType::LongShortTermMemory))
    {
        set_loss("MeanSquaredError");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        return;
    }


    if (neural_network->has(LayerType::Convolutional))
    {
        set_loss("CrossEntropy");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        dynamic_cast<AdaptiveMomentEstimation*>(optimizer.get())->set_maximum_epochs(100);
        return;
    }

    const auto& layers = neural_network->get_layers();
    const bool has_seq_dense = ranges::any_of(layers, [](const auto& layer) {
        const auto* dense = dynamic_cast<const Dense*>(layer.get());
        return dense && dense->get_input_shape().rank == 2;
    });
    if (has_seq_dense)
    {
        set_loss("CrossEntropyError3d");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        dynamic_cast<AdaptiveMomentEstimation*>(optimizer.get())->set_learning_rate(0.0001f);
        return;
    }


    if (neural_network->has(LayerType::Embedding) || neural_network->has(LayerType::MultiHeadAttention))
    {
        set_loss("CrossEntropy");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        dynamic_cast<AdaptiveMomentEstimation*>(optimizer.get())->set_maximum_epochs(100);
        return;
    }


    const ActivationFunction output_activation = neural_network->get_output_activation();

    if (output_activation == ActivationFunction::Softmax)
    {
        set_loss("CrossEntropy");
        set_optimization_algorithm("QuasiNewtonMethod");
        return;
    }

    if (output_activation == ActivationFunction::Sigmoid)
    {
        set_loss("WeightedSquaredError");
        set_optimization_algorithm("QuasiNewtonMethod");
        return;
    }

    set_loss("MeanSquaredError");
    set_optimization_algorithm("AdaptiveMomentEstimation");
}

TrainingResult TrainingStrategy::train()
{
    throw_if(!get_neural_network(), "neural network is not set.");

    throw_if(!get_dataset(), "dataset is not set.");

    throw_if(!loss->get_neural_network() || !loss->get_dataset(), "loss is not set.");

    throw_if(!optimizer->get_loss(), "optimizer is not set.");

    return optimizer->train();
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


    const Json* loss_element = require_json_field(root_element, "Loss");


    const string loss_method = read_json_string(loss_element, "Error");

    const Json* loss_method_element = loss_element->find(loss_method.c_str());

    throw_if(!loss_method_element, format("{} element is nullptr.\n", loss_method));

    set_loss(loss_method);
    loss->from_JSON(JsonDocument::wrap(loss_method, *loss_method_element));


    const Json* optimization_algorithm_element = require_json_field(root_element, "Optimizer");


    const string optimization_method = read_json_string(optimization_algorithm_element, "OptimizationMethod");

    const Json* optimization_method_element = optimization_algorithm_element->find(optimization_method.c_str());

    throw_if(!optimization_method_element, format("{} element is nullptr.\n", optimization_method));

    set_optimization_algorithm(optimization_method);
    optimizer->from_JSON(JsonDocument::wrap(optimization_method, *optimization_method_element));


    const Json* regularization_element = loss_element->find("Regularization");

    if (regularization_element)
        loss->regularization_from_JSON(JsonDocument::wrap("Regularization", *regularization_element));


    optimizer->set_display(read_json_bool(root_element, "Display"));
}

void TrainingStrategy::save(const filesystem::path& file_name) const
{
    save_json_file(file_name, *this);
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
