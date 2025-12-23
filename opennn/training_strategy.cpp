//   OpenNN: Open Neural Networks Library+
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "loss_index.h"
#include "optimization_algorithm.h"
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


LossIndex* TrainingStrategy::get_loss_index() const
{
    return loss_index.get();
}


OptimizationAlgorithm* TrainingStrategy::get_optimization_algorithm() const
{
    return optimization_algorithm.get();
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


void TrainingStrategy::set_loss_index(const string& new_loss_index)
{
    loss_index = Registry<LossIndex>::instance().create(new_loss_index);

    loss_index->set(neural_network, dataset);

    if(optimization_algorithm){
        if(optimization_algorithm->get_name() == "QuasiNewtonMethod")
            static_cast<QuasiNewtonMethod*>(optimization_algorithm.get())->set_loss_index(loss_index.get());
        else
            optimization_algorithm->set(loss_index.get());
    }
}


void TrainingStrategy::set_optimization_algorithm(const string& new_optimization_algorithm)
{
    optimization_algorithm = Registry<OptimizationAlgorithm>::instance().create(new_optimization_algorithm);

    optimization_algorithm->set(loss_index.get());
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
        set_loss_index("MeanSquaredError");
        set_optimization_algorithm("AdaptiveMomentEstimation");
        return;
    }

    // Image Classification

    if(neural_network->has("Convolutional"))
    {
        set_loss_index("CrossEntropyError2d");
        set_optimization_algorithm("AdaptiveMomentEstimation");

        AdaptiveMomentEstimation* adaptive_moment_estimation = dynamic_cast<AdaptiveMomentEstimation*>(get_optimization_algorithm());
        adaptive_moment_estimation->set_maximum_epochs_number(100);
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
            const Dense2d* dense_layer = static_cast<const Dense2d*>(layer.get());
            output_activation = dense_layer->get_activation_function();
            break;
        }
    }

    // Text Classification

    if(neural_network->has("Embedding") || neural_network->has("MultiHeadAttention"))
    {
        if(output_activation == "Softmax")
            set_loss_index("CrossEntropyError2d");
        else
            set_loss_index("WeightedSquaredError");

        set_optimization_algorithm("AdaptiveMomentEstimation");

        AdaptiveMomentEstimation* adaptive_moment_estimation = dynamic_cast<AdaptiveMomentEstimation*>(get_optimization_algorithm());
        adaptive_moment_estimation->set_maximum_epochs_number(100);
        adaptive_moment_estimation->set_display_period(10);

        return;
    }

    // Multiple classification

    if(output_activation == "Softmax")
    {
        set_loss_index("CrossEntropyError2d");
        set_optimization_algorithm("QuasiNewtonMethod");
    }

    // Binary classification

    else if(output_activation == "Logistic")
    {
        set_loss_index("WeightedSquaredError");
        set_optimization_algorithm("QuasiNewtonMethod");
    }

    // Approximation

    else
    {
        set_loss_index("MeanSquaredError");
        set_optimization_algorithm("QuasiNewtonMethod");
    }
}


TrainingResults TrainingStrategy::train()
{
    if(!has_neural_network())
        throw runtime_error("Neural network is null.");

    if(!has_dataset())
        throw runtime_error("Dataset is null.");

    if(!loss_index->has_neural_network() || !loss_index->has_dataset())
        throw runtime_error("Loss index is wrong.");

    if(!optimization_algorithm->has_loss_index())
        throw runtime_error("Optimization algorithm is wrong.");

    if(neural_network->has("Recurrent"))
        fix_forecasting();

    return optimization_algorithm->train();
}


void TrainingStrategy::fix_forecasting()
{
    /*
    Index past_time_steps = 0;

    if(neural_network->has(Recurrent))
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
         << "Loss index: " << loss_index->get_name() << endl
         << "Optimization algorithm: " << optimization_algorithm->get_name() << endl;
}


void TrainingStrategy::to_XML(XMLPrinter& printer) const
{

    printer.OpenElement("TrainingStrategy");

    printer.OpenElement("LossIndex");

    add_xml_element(printer, "LossMethod", loss_index->get_name());

    loss_index->to_XML(printer);

    loss_index->write_regularization_XML(printer);

    printer.CloseElement();

    printer.OpenElement("OptimizationAlgorithm");

    add_xml_element(printer, "OptimizationMethod", optimization_algorithm->get_name());

    optimization_algorithm->to_XML(printer);

    printer.CloseElement();

    add_xml_element(printer, "Display", to_string(optimization_algorithm->get_display()));

    printer.CloseElement();
}


void TrainingStrategy::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("TrainingStrategy");
    if (!root_element) throw runtime_error("TrainingStrategy element is nullptr.\n");

    // Loss index

    const XMLElement* loss_index_element = root_element->FirstChildElement("LossIndex");
    if (!loss_index_element) throw runtime_error("Loss index element is nullptr.\n");

    // Loss method

    const string loss_method = read_xml_string(loss_index_element, "LossMethod");

    const XMLElement* loss_method_element = loss_index_element->FirstChildElement(loss_method.c_str());

    if(loss_method_element)
    {
        set_loss_index(loss_method);

        XMLDocument loss_method_document;
        loss_method_document.InsertFirstChild(loss_method_element->DeepClone(&loss_method_document));
        loss_index->from_XML(loss_method_document);
    }
    else throw runtime_error(loss_method + " element is nullptr.\n");

    // Optimization algorithm

    const XMLElement* optimization_algorithm_element = root_element->FirstChildElement("OptimizationAlgorithm");
    if (!optimization_algorithm_element) throw runtime_error("OptimizationAlgorithm element is nullptr.\n");

    // Optimization method

    const string optimization_method = read_xml_string(optimization_algorithm_element, "OptimizationMethod");

    const XMLElement* optimization_method_element = optimization_algorithm_element->FirstChildElement(optimization_method.c_str());

    if(optimization_method_element)
    {
        set_optimization_algorithm(optimization_method);

        XMLDocument optimization_method_document;
        optimization_method_document.InsertFirstChild(optimization_method_element->DeepClone(&optimization_method_document));
        optimization_algorithm->from_XML(optimization_method_document);
    }
    else throw runtime_error(optimization_method + " element is nullptr.\n");

    // Regularization

    const XMLElement* regularization_element = loss_index_element->FirstChildElement("Regularization");

    if (regularization_element)
    {
        XMLDocument regularization_document;
        regularization_document.InsertFirstChild(regularization_element->DeepClone(&regularization_document));
        loss_index->regularization_from_XML(regularization_document);
    }

    // Display

    optimization_algorithm->set_display(read_xml_bool(root_element, "Display"));
}


void TrainingStrategy::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    XMLPrinter printer;
    to_XML(printer);
    file << printer.CStr();
}


void TrainingStrategy::load(const filesystem::path& file_name)
{
    set_default();

    XMLDocument document;

    if (document.LoadFile(file_name.string().c_str()))
        throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

    from_XML(document);
}


#ifdef OPENNN_CUDA

TrainingResults TrainingStrategy::train_cuda()
{
    if (!has_neural_network())
        throw runtime_error("Neural network is null.");

    if (!has_dataset())
        throw runtime_error("Dataset is null.");

    if (!loss_index->has_neural_network() || !loss_index->has_dataset())
        throw runtime_error("Loss index is wrong.");

    if (!optimization_algorithm->has_loss_index())
        throw runtime_error("Optimization algorithm is wrong.");

    neural_network->create_cuda();

    get_loss_index()->create_cuda();

    get_optimization_algorithm()->create_cuda();

    if (neural_network->has("Recurrent"))
        fix_forecasting();

    return optimization_algorithm->train_cuda();
}

#endif

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
