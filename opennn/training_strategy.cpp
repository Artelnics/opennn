//   OpenNN: Open Neural Networks Library+
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "recurrent_layer.h"
#include "optimization_algorithm.h"
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


LossIndex* TrainingStrategy::get_loss_index()
{
    return loss_index.get();
}


OptimizationAlgorithm* TrainingStrategy::get_optimization_algorithm()
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

    if(optimization_algorithm)
        optimization_algorithm->set(loss_index.get());
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

    const NeuralNetwork::ModelType model_type = neural_network->get_model_type();
/*
    if(model_type == NeuralNetwork::ModelType::Classification
    || model_type == NeuralNetwork::ModelType::ImageClassification)
        set_loss_index("CrossEntropyError2d");
    else if(model_type == NeuralNetwork::ModelType::TextClassification)
        set_loss_index("CrossEntropyError3d");
    else
        set_loss_index("NormalizedSquaredError");
*/

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

    if(neural_network->has(Layer::Type::Recurrent))
        fix_forecasting();

    return optimization_algorithm->perform_training();
}


void TrainingStrategy::fix_forecasting()
{
/*
    Index time_steps = 0;

    if(neural_network->has(Layer::Type::Recurrent))
        time_steps = static_cast<Recurrent*>(neural_network->get_first(Layer::Type::Recurrent))->get_timesteps();
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
/*
    printer.OpenElement("TrainingStrategy");

    printer.OpenElement("LossIndex");

    add_xml_element(printer, "LossMethod", write_loss_method());

    mean_squared_error.to_XML(printer);
    normalized_squared_error.to_XML(printer);
    Minkowski_error.to_XML(printer);
    cross_entropy_error_2d.to_XML(printer);
    weighted_squared_error.to_XML(printer);

    switch (loss_method) {
    case LossMethod::MEAN_SQUARED_ERROR:
        mean_squared_error.write_regularization_XML(printer);
        break;
    case LossMethod::NORMALIZED_SQUARED_ERROR:
        normalized_squared_error.write_regularization_XML(printer);
        break;
    case LossMethod::MINKOWSKI_ERROR:
        Minkowski_error.write_regularization_XML(printer);
        break;
    case LossMethod::CROSS_ENTROPY_ERROR_2D:
        cross_entropy_error_2d.write_regularization_XML(printer);
        break;
    case LossMethod::WEIGHTED_SQUARED_ERROR:
        weighted_squared_error.write_regularization_XML(printer);
        break;
    default:
        break;
    }

    printer.CloseElement();  

    printer.OpenElement("OptimizationAlgorithm");

    add_xml_element(printer, "OptimizationMethod", write_optimization_method());

    stochastic_gradient_descent.to_XML(printer);
    adaptive_moment_estimation.to_XML(printer);
    quasi_Newton_method.to_XML(printer);
    Levenberg_Marquardt_algorithm.to_XML(printer);

    printer.CloseElement();  

    add_xml_element(printer, "Display", to_string(get_display())); 

    printer.CloseElement();
*/
}


void TrainingStrategy::from_XML(const XMLDocument& document)
{
/*
    const XMLElement* root_element = document.FirstChildElement("TrainingStrategy");
    if (!root_element) throw runtime_error("TrainingStrategy element is nullptr.\n");

    const XMLElement* loss_index_element = root_element->FirstChildElement("LossIndex");
    if (!loss_index_element) throw runtime_error("Loss index element is nullptr.\n");

    // Loss method

    set_loss_method(read_xml_string(loss_index_element, "LossMethod"));

    // Minkowski error

    const XMLElement* minkowski_error_element = loss_index_element->FirstChildElement("MinkowskiError");

    if (minkowski_error_element)
    {
        XMLDocument minkowski_document;
        XMLElement* minkowski_error_element_copy = minkowski_document.NewElement("MinkowskiError");

        for (const XMLNode* node = minkowski_error_element->FirstChild(); node; node = node->NextSibling())
            minkowski_error_element_copy->InsertEndChild(node->DeepClone(&minkowski_document));

        minkowski_document.InsertEndChild(minkowski_error_element_copy);
        Minkowski_error.from_XML(minkowski_document);
    }

    // Cross entropy error

    const XMLElement* cross_entropy_element = loss_index_element->FirstChildElement("CrossEntropyError2d");

    if (cross_entropy_element)
    {
        XMLDocument cross_entropy_document;
        XMLElement* cross_entropy_error_element_copy = cross_entropy_document.NewElement("CrossEntropyError2d");

        for (const XMLNode* node = cross_entropy_element->FirstChild(); node; node = node->NextSibling())
            cross_entropy_error_element_copy->InsertEndChild(node->DeepClone(&cross_entropy_document));

        cross_entropy_document.InsertEndChild(cross_entropy_error_element_copy);
        cross_entropy_error_2d.from_XML(cross_entropy_document);
    }

    // Weighted squared error

    const XMLElement* weighted_squared_error_element = loss_index_element->FirstChildElement("WeightedSquaredError");

    if (weighted_squared_error_element)
    {
        XMLDocument weighted_squared_error_document;
        XMLElement* weighted_squared_error_element_copy = weighted_squared_error_document.NewElement("WeightedSquaredError");

        for (const XMLNode* node = weighted_squared_error_element->FirstChild(); node; node = node->NextSibling())
            weighted_squared_error_element_copy->InsertEndChild(node->DeepClone(&weighted_squared_error_document));

        weighted_squared_error_document.InsertEndChild(weighted_squared_error_element_copy);
        weighted_squared_error.from_XML(weighted_squared_error_document);
    }

    // Regularization

    const XMLElement* regularization_element = loss_index_element->FirstChildElement("Regularization");

    if (regularization_element)
    {
        XMLDocument regularization_document;
        regularization_document.InsertFirstChild(regularization_element->DeepClone(&regularization_document));
        get_loss_index()->regularization_from_XML(regularization_document);
    }

    // Optimization algorithm

    const XMLElement* optimization_algorithm_element = root_element->FirstChildElement("OptimizationAlgorithm");
    if (!optimization_algorithm_element) throw runtime_error("OptimizationAlgorithm element is nullptr.\n");

    // Optimization method

    set_optimization_method(read_xml_string(optimization_algorithm_element, "OptimizationMethod"));

    // Stochastic gradient descent

    const XMLElement* stochastic_gradient_descent_element = optimization_algorithm_element->FirstChildElement("StochasticGradientDescent");

    if (stochastic_gradient_descent_element)
    {
        XMLDocument stochastic_gradient_document;
        XMLElement* stochastic_gradient_element_copy = stochastic_gradient_document.NewElement("StochasticGradientDescent");

        for (const XMLNode* node = stochastic_gradient_descent_element->FirstChild(); node; node = node->NextSibling())
            stochastic_gradient_element_copy->InsertEndChild(node->DeepClone(&stochastic_gradient_document));

        stochastic_gradient_document.InsertEndChild(stochastic_gradient_element_copy);
        stochastic_gradient_descent.from_XML(stochastic_gradient_document);
    }

    // Adaptive moment estimation

    const XMLElement* adaptive_moment_element = optimization_algorithm_element->FirstChildElement("AdaptiveMomentEstimation");
    if (adaptive_moment_element) {
        XMLDocument adaptive_moment_document;
        XMLElement* adaptive_moment_element_copy = adaptive_moment_document.NewElement("AdaptiveMomentEstimation");

        for (const XMLNode* node = adaptive_moment_element->FirstChild(); node; node = node->NextSibling())
            adaptive_moment_element_copy->InsertEndChild(node->DeepClone(&adaptive_moment_document));

        adaptive_moment_document.InsertEndChild(adaptive_moment_element_copy);
        adaptive_moment_estimation.from_XML(adaptive_moment_document);
    }

    // Quasi-Newton method

    const XMLElement* quasi_newton_element = optimization_algorithm_element->FirstChildElement("QuasiNewtonMethod");
    if (quasi_newton_element) {
        XMLDocument quasi_newton_document;
        XMLElement* quasi_newton_element_copy = quasi_newton_document.NewElement("QuasiNewtonMethod");

        for (const XMLNode* node = quasi_newton_element->FirstChild(); node; node = node->NextSibling())
            quasi_newton_element_copy->InsertEndChild(node->DeepClone(&quasi_newton_document));

        quasi_newton_document.InsertEndChild(quasi_newton_element_copy);
        quasi_Newton_method.from_XML(quasi_newton_document);
    }

    // Levenberg-Marquardt

    const XMLElement* levenberg_marquardt_element = optimization_algorithm_element->FirstChildElement("LevenbergMarquardt");
    if (levenberg_marquardt_element) {
        XMLDocument levenberg_document;
        XMLElement* levenberg_element_copy = levenberg_document.NewElement("LevenbergMarquardt");

        for (const XMLNode* node = levenberg_marquardt_element->FirstChild(); node; node = node->NextSibling())
            levenberg_element_copy->InsertEndChild(node->DeepClone(&levenberg_document));

        levenberg_document.InsertEndChild(levenberg_element_copy);
        Levenberg_Marquardt_algorithm.from_XML(levenberg_document);
    }

    // Display

    set_display(read_xml_bool(root_element, "Display"));
*/
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

    if (neural_network->has(Layer::Type::Recurrent))
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
