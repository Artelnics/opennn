//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/training_strategy/training_strategy.h"

#include "opennn/registry.h"

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

    if (!neural_network)
    {
        optimizer.reset();
        return loss.reset();
    }

    set_default();
}

void TrainingStrategy::set_dataset(Dataset* new_dataset)
{
    dataset = new_dataset;
    if (loss) loss->set_dataset(new_dataset);
}

void TrainingStrategy::set_neural_network(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
    if (!neural_network)
    {
        optimizer.reset();
        loss.reset();
    }
    else if (loss)
        loss->set_neural_network(new_neural_network);
    else
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
    optimizer = create_optimizer(new_optimization_algorithm);

    optimizer->set(loss.get());
}

void TrainingStrategy::set_default()
{
    if (!get_neural_network())
        return;

    const char* loss_name = "MeanSquaredError";
    const char* optimizer_name = "AdaptiveMomentEstimation";

    switch (neural_network->get_task())
    {
        case NetworkTask::Classification:
            loss_name = neural_network->get_outputs_number() == 1
                      ? "WeightedSquaredError"
                      : "CrossEntropy";
            optimizer_name = "QuasiNewtonMethod";
            break;

        case NetworkTask::ImageClassification:
        case NetworkTask::ObjectDetection:
        case NetworkTask::TextClassification:
            loss_name = "CrossEntropy";
            break;

        case NetworkTask::LanguageModeling:
            loss_name = "CrossEntropyError3d";
            break;

        case NetworkTask::Generic:
        case NetworkTask::Approximation:
        case NetworkTask::Forecasting:
        case NetworkTask::AutoAssociation:
            break;
    }

    set_loss(loss_name);
    set_optimization_algorithm(optimizer_name);
    optimizer->configure_for_task(neural_network->get_task());
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

    add_json_field(printer, "Display", optimizer->get_display());

    printer.close_element();
}

void TrainingStrategy::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "TrainingStrategy");

    const Json* loss_element = require_json_field(root_element, "Loss");

    const string loss_method = read_json_string(loss_element, "Error");

    const Json* loss_method_element = loss_element->find(loss_method.c_str());

    throw_if(!loss_method_element, "{} element is nullptr.\n", loss_method);

    set_loss(loss_method);
    loss->from_JSON(JsonDocument::wrap(loss_method, *loss_method_element));

    const Json* optimization_algorithm_element = require_json_field(root_element, "Optimizer");

    const string optimization_method = read_json_string(optimization_algorithm_element, "OptimizationMethod");

    const Json* optimization_method_element = optimization_algorithm_element->find(optimization_method.c_str());

    throw_if(!optimization_method_element, "{} element is nullptr.\n", optimization_method);

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
