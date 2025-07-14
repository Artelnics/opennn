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
#include "quasi_newton_method.h"
#include "training_strategy.h"

namespace opennn
{

TrainingStrategy::TrainingStrategy(NeuralNetwork* new_neural_network, Dataset* new_data_set)
{
    set(new_neural_network, new_data_set);
}


Dataset* TrainingStrategy::get_data_set()
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


bool TrainingStrategy::has_data_set() const
{
    return dataset;
}


void TrainingStrategy::set(NeuralNetwork* new_neural_network, Dataset* new_data_set)
{
    neural_network = new_neural_network;
    dataset = new_data_set;

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


void TrainingStrategy::set_data_set(Dataset* new_data_set)
{
    dataset = new_data_set;
}


void TrainingStrategy::set_neural_network(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
}


void TrainingStrategy::set_default()
{
    if(!has_neural_network()) return;

    set_loss_index("MeanSquaredError");
    set_optimization_algorithm("AdaptiveMomentEstimation");
}


TrainingResults TrainingStrategy::perform_training()
{
    if(!has_neural_network())
        throw runtime_error("Neural network is null.");

    if(!has_data_set())
        throw runtime_error("Data set is null.");

    if(!loss_index->has_neural_network() || !loss_index->has_data_set())
        throw runtime_error("Loss index is wrong.");

    if(!optimization_algorithm->has_loss_index())
        throw runtime_error("Optimization algorithm is wrong.");

    if(neural_network->has("Recurrent"))
        fix_forecasting();

    optimization_algorithm->set_display(true);

    return optimization_algorithm->perform_training();
}


void TrainingStrategy::fix_forecasting()
{
/*
    Index time_steps = 0;

    if(neural_network->has(Recurrent))
        time_steps = static_cast<Recurrent*>(neural_network->get_first(Recurrent))->get_timesteps();
    else
        return;

    Index batch_size = 0;

    if(optimization_method == OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION)
        batch_size = adaptive_moment_estimation.get_samples_number();
    else if(optimization_method == OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT)
        batch_size = stochastic_gradient_descent.get_samples_number();
    else
        return;

    if(batch_size%time_steps == 0)
        return;

    const Index constant = time_steps > batch_size
        ? 1
        : Index(batch_size/time_steps);

    if(optimization_method == OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION)
        adaptive_moment_estimation.set_batch_size(constant*time_steps);
    else if(optimization_method == OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT)
        stochastic_gradient_descent.set_batch_size(constant*time_steps);
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
        XMLElement* loss_index_element_copy = loss_method_document.NewElement(loss_method.c_str());

        for (const XMLNode* node = loss_index_element->FirstChild(); node; node = node->NextSibling())
            loss_index_element_copy->InsertEndChild(node->DeepClone(&loss_method_document));

        loss_method_document.InsertEndChild(loss_index_element_copy);
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
        XMLElement* optimization_method_element_copy = optimization_method_document.NewElement(optimization_method.c_str());

        for (const XMLNode* node = optimization_method_element->FirstChild(); node; node = node->NextSibling())
            optimization_method_element_copy->InsertEndChild(node->DeepClone(&optimization_method_document));

        optimization_method_document.InsertEndChild(optimization_method_element_copy);
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

TrainingResults TrainingStrategy::perform_training_cuda()
{
    if (!has_neural_network())
        throw runtime_error("Neural network is null.");

    if (!has_data_set())
        throw runtime_error("Data set is null.");
            
    if(!optimization_algorithm->has_loss_index())
        throw runtime_error("Optimization algorithm is wrong.");

    neural_network->create_cuda();

    get_loss_index()->create_cuda();

    get_optimization_algorithm()->create_cuda();

    if (neural_network->has("Recurrent"))
        fix_forecasting();

    return optimization_algorithm->perform_training_cuda();
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
