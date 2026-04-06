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
#include "quasi_newton_method.h"
#include "adaptive_moment_estimation.h"
#include "dense_layer.h"

namespace opennn
{

TrainingStrategy::TrainingStrategy(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
{
    set(new_neural_network, new_dataset);
}


Dataset* TrainingStrategy::get_dataset()
{
    return dataset;
}


NeuralNetwork* TrainingStrategy::get_neural_network() const
{
    return neural_network;
}


Loss* TrainingStrategy::get_loss() const
{
    return loss.get();
}


Optimizer* TrainingStrategy::get_optimization_algorithm() const
{
    return optimizer.get();
}


bool TrainingStrategy::has_neural_network() const
{
    return neural_network;
}


bool TrainingStrategy::has_dataset() const
{
    return dataset;
}


void TrainingStrategy::set(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
{
    neural_network = const_cast<NeuralNetwork*>(new_neural_network);
    dataset = const_cast<Dataset*>(new_dataset);

    set_default();
}


void TrainingStrategy::set_loss(const string& new_loss)
{
    loss = Registry<Loss>::instance().create(new_loss);

    loss->set(neural_network, dataset);

    if(optimizer){
        if(optimizer->get_name() == "QuasiNewtonMethod")
            static_cast<QuasiNewtonMethod*>(optimizer.get())->set_loss(loss.get());
        else
            optimizer->set(loss.get());
    }
}


void TrainingStrategy::set_optimization_algorithm(const string& new_optimization_algorithm)
{
    optimizer = Registry<Optimizer>::instance().create(new_optimization_algorithm);

    optimizer->set(loss.get());
}


void TrainingStrategy::set_dataset(const Dataset* new_dataset)
{
    dataset = const_cast<Dataset*>(new_dataset);
}


void TrainingStrategy::set_neural_network(const NeuralNetwork* new_neural_network)
{
    neural_network = const_cast<NeuralNetwork*>(new_neural_network);
}


void TrainingStrategy::set_default()
{
    if(!has_neural_network())
        return;

    // Forecasting

    if(neural_network->has("Recurrent"))
    {
        set_loss("MeanSquaredError");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        return;
    }

    // Image Classification

    if(neural_network->has("Convolutional"))
    {
        set_loss("CrossEntropyError2d");
        set_optimization_algorithm("AdaptiveMomentEstimation");

        AdaptiveMomentEstimation* adaptive_moment_estimation = dynamic_cast<AdaptiveMomentEstimation*>(get_optimization_algorithm());
        adaptive_moment_estimation->set_maximum_epochs(100);
        adaptive_moment_estimation->set_display_period(10);

        return;
    }

    string output_activation = "Linear";

    const Index layers_number = neural_network->get_layers_number();

    for(Index i = layers_number - 1; i >= 0; i--)
    {
        const unique_ptr<Layer>& layer = neural_network->get_layer(i);

        if(layer->get_name() == "Dense2d")
        {
            /*
            const Dense<2>* dense_layer = static_cast<const Dense<2>*>(layer.get());
            output_activation = dense_layer->get_activation_function();
            break;
*/
        }
    }

    // Transformer

    if(neural_network->has("Dense3d"))
    {
        set_loss("CrossEntropyError3d");
        set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = static_cast<AdaptiveMomentEstimation*>(optimizer.get());
        adam->set_learning_rate(0.0001);
        return;
    }

    // Text Classification

    if(neural_network->has("Embedding") || neural_network->has("MultiHeadAttention"))
    {
        if(output_activation == "Softmax")
            set_loss("CrossEntropyError2d");
        else
            set_loss("WeightedSquaredError");

        set_optimization_algorithm("AdaptiveMomentEstimation");

        AdaptiveMomentEstimation* adaptive_moment_estimation = dynamic_cast<AdaptiveMomentEstimation*>(get_optimization_algorithm());
        adaptive_moment_estimation->set_maximum_epochs(100);
        adaptive_moment_estimation->set_display_period(10);

        return;
    }

    // Multiple classification

    if(output_activation == "Softmax")
    {
        set_loss("CrossEntropyError2d");
        set_optimization_algorithm("QuasiNewtonMethod");
    }

    // Binary classification

    else if(output_activation == "Sigmoid")
    {
        set_loss("WeightedSquaredError");
        set_optimization_algorithm("QuasiNewtonMethod");
    }

    // Approximation

    else
    {
        set_loss("MeanSquaredError");
        set_optimization_algorithm("QuasiNewtonMethod");
    }
}


TrainingResults TrainingStrategy::train()
{
    if(!has_neural_network())
        throw runtime_error("Neural network is null.");

    if(!has_dataset())
        throw runtime_error("Dataset is null.");

    if(!loss->has_neural_network() || !loss->has_dataset())
        throw runtime_error("Loss index is wrong.");

    if(!optimizer->has_loss())
        throw runtime_error("Optimization algorithm is wrong.");

    if(neural_network->has("Recurrent"))
        fix_forecasting();

    return optimizer->train();
}


void TrainingStrategy::fix_forecasting()
{
    Index past_time_steps = 0;
/*
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


void TrainingStrategy::print() const
{
    cout << "Training strategy object" << endl
         << "Loss: " << loss->get_name() << endl
         << "Optimization algorithm: " << optimizer->get_name() << endl;
}


void TrainingStrategy::to_XML(XMLPrinter& printer) const
{

    printer.OpenElement("TrainingStrategy");

    printer.OpenElement("Loss");

    add_xml_element(printer, "Error", loss->get_name());

    loss->to_XML(printer);

    loss->write_regularization_XML(printer);

    printer.CloseElement();

    printer.OpenElement("Optimizer");

    add_xml_element(printer, "OptimizationMethod", optimizer->get_name());

    optimizer->to_XML(printer);

    printer.CloseElement();

    add_xml_element(printer, "Display", to_string(optimizer->get_display()));

    printer.CloseElement();
}


void TrainingStrategy::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = get_xml_root(document, "TrainingStrategy");

    // Loss

    const XMLElement* loss_element = root_element->FirstChildElement("Loss");
    if(!loss_element) throw runtime_error("Loss element is nullptr.\n");

    // Loss method

    const string loss_method = read_xml_string(loss_element, "Error");

    const XMLElement* loss_method_element = loss_element->FirstChildElement(loss_method.c_str());

    if(loss_method_element)
    {
        set_loss(loss_method);

        XMLDocument loss_method_document;
        loss_method_document.InsertFirstChild(loss_method_element->DeepClone(&loss_method_document));
        loss->from_XML(loss_method_document);
    }
    else throw runtime_error(loss_method + " element is nullptr.\n");

    // Optimization algorithm

    const XMLElement* optimization_algorithm_element = root_element->FirstChildElement("Optimizer");
    if(!optimization_algorithm_element) throw runtime_error("Optimizer element is nullptr.\n");

    // Optimization method

    const string optimization_method = read_xml_string(optimization_algorithm_element, "OptimizationMethod");

    const XMLElement* optimization_method_element = optimization_algorithm_element->FirstChildElement(optimization_method.c_str());

    if(optimization_method_element)
    {
        set_optimization_algorithm(optimization_method);

        XMLDocument optimization_method_document;
        optimization_method_document.InsertFirstChild(optimization_method_element->DeepClone(&optimization_method_document));
        optimizer->from_XML(optimization_method_document);
    }
    else throw runtime_error(optimization_method + " element is nullptr.\n");

    // Regularization

    const XMLElement* regularization_element = loss_element->FirstChildElement("Regularization");

    if (regularization_element)
    {
        XMLDocument regularization_document;
        regularization_document.InsertFirstChild(regularization_element->DeepClone(&regularization_document));
        loss->regularization_from_XML(regularization_document);
    }

    // Display

    optimizer->set_display(read_xml_bool(root_element, "Display"));
}


void TrainingStrategy::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        return;

    XMLPrinter printer;
    to_XML(printer);
    file << printer.CStr();
}


void TrainingStrategy::load(const filesystem::path& file_name)
{
    set_default();

    from_XML(load_xml_file(file_name));
}


#ifdef CUDA

TrainingResults TrainingStrategy::train_cuda()
{
    if(!has_neural_network())
        throw runtime_error("Neural network is null.");

    if(!has_dataset())
        throw runtime_error("Dataset is null.");

    if(!loss->has_neural_network() || !loss->has_dataset())
        throw runtime_error("Loss is wrong.");

    if(!optimizer->has_loss())
        throw runtime_error("Optimization algorithm is wrong.");

    if (neural_network->has("Recurrent"))
        fix_forecasting();

    return optimizer->train_cuda();
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
