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

    if(optimizer)
        optimizer->set(loss.get());
}

void TrainingStrategy::set_optimization_algorithm(const string& new_optimization_algorithm)
{
    optimizer = Registry<Optimizer>::instance().create(new_optimization_algorithm);

    optimizer->set(loss.get());
}

void TrainingStrategy::set_default()
{
    if(!get_neural_network())
        return;

    // Forecasting

    if(neural_network->has(LayerType::Recurrent))
    {
        set_loss("MeanSquaredError");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        return;
    }

    // Image Classification

    if(neural_network->has(LayerType::Convolutional))
    {
        set_loss("CrossEntropy");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        dynamic_cast<AdaptiveMomentEstimation*>(optimizer.get())->set_maximum_epochs(100);
        return;
    }

    // Transformer

    if(neural_network->has(LayerType::Dense3d))
    {
        set_loss("CrossEntropy");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        dynamic_cast<AdaptiveMomentEstimation*>(optimizer.get())->set_learning_rate(0.0001);
        return;
    }

    // Text Classification

    if(neural_network->has(LayerType::Embedding) || neural_network->has(LayerType::MultiHeadAttention))
    {
        set_loss("WeightedSquaredError");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        dynamic_cast<AdaptiveMomentEstimation*>(optimizer.get())->set_maximum_epochs(100);
        return;
    }

    // Approximation (default)

    set_loss("MeanSquaredError");
    set_optimization_algorithm("QuasiNewtonMethod");
}

TrainingResults TrainingStrategy::train()
{
    if(!get_neural_network())
        throw runtime_error("Neural network is null.");

    if(!get_dataset())
        throw runtime_error("Dataset is null.");

    if(!loss->get_neural_network() || !loss->get_dataset())
        throw runtime_error("Loss index is wrong.");

    if(!optimizer->get_loss())
        throw runtime_error("Optimization algorithm is wrong.");

    if(neural_network->has(LayerType::Recurrent))
        fix_forecasting();

    return optimizer->train();
}


TrainingResults TrainingStrategy::train_cuda()
{
    if(!get_neural_network())
        throw runtime_error("Neural network is null.");

    if(!get_dataset())
        throw runtime_error("Dataset is null.");

    return optimizer->train_cuda();
}


void TrainingStrategy::fix_forecasting()
{
/*
    const Index past_time_steps = 0;

    if(neural_network->has("Recurrent"))
        past_time_steps = static_cast<Recurrent*>(neural_network->get_first(Recurrent))->get_timesteps();
    else
        return;

    Index batch_size = 0;

    if(optimization_method == OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION)
        batch_size = adaptive_moment_estimation.get_samples_number();
    else if(optimization_method == OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT)
        batch_size = stochastic_gradient_descent.get_samples_number();
    else
        return;

    if(batch_size%past_time_steps == 0)
        return;

    const Index constant = past_time_steps > batch_size
        ? 1
        : Index(batch_size/past_time_steps);

    if(optimization_method == OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION)
        adaptive_moment_estimation.set_batch_size(constant*past_time_steps);
    else if(optimization_method == OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT)
        stochastic_gradient_descent.set_batch_size(constant*past_time_steps);
*/
}

void TrainingStrategy::to_XML(XmlPrinter& printer) const
{

    printer.open_element("TrainingStrategy");

    printer.open_element("Loss");

    add_xml_element(printer, "Error", loss->get_name());

    loss->to_XML(printer);

    loss->write_regularization_XML(printer);

    printer.close_element();

    printer.open_element("Optimizer");

    add_xml_element(printer, "OptimizationMethod", optimizer->get_name());

    optimizer->to_XML(printer);

    printer.close_element();

    add_xml_element(printer, "Display", to_string(optimizer->get_display()));

    printer.close_element();
}

void TrainingStrategy::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "TrainingStrategy");

    // Loss

    const XmlElement* loss_element = require_xml_element(root_element, "Loss");

    // Loss method

    const string loss_method = read_xml_string(loss_element, "Error");

    const XmlElement* loss_method_element = loss_element->first_child_element(loss_method.c_str());

    if(loss_method_element)
    {
        set_loss(loss_method);

        XmlDocument loss_method_document;
        loss_method_document.insert_first_child(loss_method_element->deep_clone(&loss_method_document));
        loss->from_XML(loss_method_document);
    }
    else throw runtime_error(loss_method + " element is nullptr.\n");

    // Optimization algorithm

    const XmlElement* optimization_algorithm_element = require_xml_element(root_element, "Optimizer");

    // Optimization method

    const string optimization_method = read_xml_string(optimization_algorithm_element, "OptimizationMethod");

    const XmlElement* optimization_method_element = optimization_algorithm_element->first_child_element(optimization_method.c_str());

    if(optimization_method_element)
    {
        set_optimization_algorithm(optimization_method);

        XmlDocument optimization_method_document;
        optimization_method_document.insert_first_child(optimization_method_element->deep_clone(&optimization_method_document));
        optimizer->from_XML(optimization_method_document);
    }
    else throw runtime_error(optimization_method + " element is nullptr.\n");

    // Regularization

    const XmlElement* regularization_element = loss_element->first_child_element("Regularization");

    if (regularization_element)
    {
        XmlDocument regularization_document;
        regularization_document.insert_first_child(regularization_element->deep_clone(&regularization_document));
        loss->regularization_from_XML(regularization_document);
    }

    // Display

    optimizer->set_display(read_xml_bool(root_element, "Display"));
}

void TrainingStrategy::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open file: " + file_name.string());

    XmlPrinter printer;
    to_XML(printer);
    file << printer.c_str();
}

void TrainingStrategy::load(const filesystem::path& file_name)
{
    set_default();

    from_XML(load_xml_file(file_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
