//   OpenNN: Open Neural Networks Library+
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "training_strategy.h"
#include "optimization_algorithm.h"
#include "recurrent_layer.h"
#include "long_short_term_memory_layer.h"

namespace opennn
{

TrainingStrategy::TrainingStrategy(NeuralNetwork* new_neural_network, DataSet* new_data_set)
{
    set(new_neural_network, new_data_set);
}


DataSet* TrainingStrategy::get_data_set()
{
    return data_set;
}


NeuralNetwork* TrainingStrategy::get_neural_network() const
{
    return neural_network;
}


LossIndex* TrainingStrategy::get_loss_index()
{
    switch(loss_method)
    {
        case LossMethod::MEAN_SQUARED_ERROR: return &mean_squared_error;

        case LossMethod::NORMALIZED_SQUARED_ERROR: return &normalized_squared_error;

        case LossMethod::MINKOWSKI_ERROR: return &Minkowski_error;

        case LossMethod::WEIGHTED_SQUARED_ERROR: return &weighted_squared_error;

        case LossMethod::CROSS_ENTROPY_ERROR: return &cross_entropy_error;

        case LossMethod::CROSS_ENTROPY_ERROR_3D: return &cross_entropy_error_3d;

        default: throw runtime_error("Unknown loss method.");
    }
}


OptimizationAlgorithm* TrainingStrategy::get_optimization_algorithm()
{
    switch(optimization_method)
    {
    case OptimizationMethod::CONJUGATE_GRADIENT: return &conjugate_gradient;

    case OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT: return &stochastic_gradient_descent;

    case OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION: return &adaptive_moment_estimation;

    case OptimizationMethod::QUASI_NEWTON_METHOD: return &quasi_Newton_method;

    case OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM: return &Levenberg_Marquardt_algorithm;

    default: return nullptr;
    }
}


bool TrainingStrategy::has_neural_network() const
{
    return neural_network;
}


bool TrainingStrategy::has_data_set() const
{
    return data_set;
}


ConjugateGradient* TrainingStrategy::get_conjugate_gradient()
{
    return &conjugate_gradient;
}


QuasiNewtonMethod* TrainingStrategy::get_quasi_Newton_method()
{
    return &quasi_Newton_method;
}


LevenbergMarquardtAlgorithm* TrainingStrategy::get_Levenberg_Marquardt_algorithm()
{
    return &Levenberg_Marquardt_algorithm;
}


StochasticGradientDescent* TrainingStrategy::get_stochastic_gradient_descent()
{
    return &stochastic_gradient_descent;
}


AdaptiveMomentEstimation* TrainingStrategy::get_adaptive_moment_estimation()
{
    return &adaptive_moment_estimation;
}


MeanSquaredError* TrainingStrategy::get_mean_squared_error()
{
    return &mean_squared_error;
}


NormalizedSquaredError* TrainingStrategy::get_normalized_squared_error()
{

    return &normalized_squared_error;
}


MinkowskiError* TrainingStrategy::get_Minkowski_error()
{
    return &Minkowski_error;
}


CrossEntropyError* TrainingStrategy::get_cross_entropy_error()
{
    return &cross_entropy_error;
}


WeightedSquaredError* TrainingStrategy::get_weighted_squared_error()
{
    return &weighted_squared_error;
}


const TrainingStrategy::LossMethod& TrainingStrategy::get_loss_method() const
{
    return loss_method;
}


const TrainingStrategy::OptimizationMethod& TrainingStrategy::get_optimization_method() const
{
    return optimization_method;
}


string TrainingStrategy::write_loss_method() const
{
    switch(loss_method)
    {
    case LossMethod::MEAN_SQUARED_ERROR:
        return "MEAN_SQUARED_ERROR";

    case LossMethod::NORMALIZED_SQUARED_ERROR:
        return "NORMALIZED_SQUARED_ERROR";

    case LossMethod::MINKOWSKI_ERROR:
        return "MINKOWSKI_ERROR";

    case LossMethod::WEIGHTED_SQUARED_ERROR:
        return "WEIGHTED_SQUARED_ERROR";

    case LossMethod::CROSS_ENTROPY_ERROR:
        return "CROSS_ENTROPY_ERROR";

    default:
        return string();
    }
}


string TrainingStrategy::write_optimization_method() const
{
    switch (optimization_method)
    { 
    case OptimizationMethod::CONJUGATE_GRADIENT:
        return "CONJUGATE_GRADIENT";
    
    case OptimizationMethod::QUASI_NEWTON_METHOD:
        return "QUASI_NEWTON_METHOD";
    
    case OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM:
        return "LEVENBERG_MARQUARDT_ALGORITHM";
    
    case OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT:
        return "STOCHASTIC_GRADIENT_DESCENT";
    
    case OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION:
        return "ADAPTIVE_MOMENT_ESTIMATION";
    
    default:
    
        throw runtime_error("Unknown optimization method.\n");
    }
}


string TrainingStrategy::write_optimization_method_text() const
{
    switch (optimization_method)
    {
    case OptimizationMethod::CONJUGATE_GRADIENT:
        return "conjugate gradient";

    case OptimizationMethod::QUASI_NEWTON_METHOD:
        return "quasi-Newton method";

    case OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM:
        return "Levenberg-Marquardt algorithm";

    case OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT:
        return "stochastic gradient descent";

    case OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION:
        return "adaptive moment estimation";

    default:
        throw runtime_error("Unknown main type.\n");
    }
}


string TrainingStrategy::write_loss_method_text() const
{
    switch(loss_method)
    {
    case LossMethod::MEAN_SQUARED_ERROR:
        return "Mean squared error";

    case LossMethod::NORMALIZED_SQUARED_ERROR:
        return "Normalized squared error";

    case LossMethod::MINKOWSKI_ERROR:
        return "Minkowski error";

    case LossMethod::WEIGHTED_SQUARED_ERROR:
        return "Weighted squared error";

    case LossMethod::CROSS_ENTROPY_ERROR:
        return "Cross entropy error";

    default:
        return string();
    }
}


const bool& TrainingStrategy::get_display() const
{
    return display;
}


void TrainingStrategy::set(NeuralNetwork* new_neural_network, DataSet* new_data_set)
{
    neural_network = new_neural_network;
    data_set = new_data_set;

    
    set_default();

    mean_squared_error.set(new_neural_network, new_data_set);
    normalized_squared_error.set(new_neural_network, new_data_set);
    cross_entropy_error.set(new_neural_network, new_data_set);
    cross_entropy_error_3d.set(new_neural_network, new_data_set);
    weighted_squared_error.set(new_neural_network, new_data_set);
    Minkowski_error.set(new_neural_network, new_data_set);

    LossIndex* new_loss_index = get_loss_index();

    conjugate_gradient.set_loss_index(new_loss_index);
    stochastic_gradient_descent.set_loss_index(new_loss_index);
    adaptive_moment_estimation.set_loss_index(new_loss_index);
    quasi_Newton_method.set_loss_index(new_loss_index);
    Levenberg_Marquardt_algorithm.set_loss_index(new_loss_index);
}


void TrainingStrategy::set_loss_method(const string& new_loss_method)
{
    if(new_loss_method == "MEAN_SQUARED_ERROR")
        set_loss_method(LossMethod::MEAN_SQUARED_ERROR);
    else if(new_loss_method == "NORMALIZED_SQUARED_ERROR")
        set_loss_method(LossMethod::NORMALIZED_SQUARED_ERROR);
    else if(new_loss_method == "MINKOWSKI_ERROR")
        set_loss_method(LossMethod::MINKOWSKI_ERROR);
    else if(new_loss_method == "WEIGHTED_SQUARED_ERROR")
        set_loss_method(LossMethod::WEIGHTED_SQUARED_ERROR);
    else if(new_loss_method == "CROSS_ENTROPY_ERROR")
        set_loss_method(LossMethod::CROSS_ENTROPY_ERROR);
    else
        throw runtime_error("Unknown loss method: " + new_loss_method + ".\n");
}


void TrainingStrategy::set_loss_method(const LossMethod& new_loss_method)
{
    loss_method = new_loss_method;

    set_loss_index(get_loss_index());
}


void TrainingStrategy::set_optimization_method(const OptimizationMethod& new_optimization_method)
{
    optimization_method = new_optimization_method;
}


void TrainingStrategy::set_optimization_method(const string& new_optimization_method)
{
    if(new_optimization_method == "CONJUGATE_GRADIENT")
        set_optimization_method(OptimizationMethod::CONJUGATE_GRADIENT);
    else if(new_optimization_method == "QUASI_NEWTON_METHOD")
        set_optimization_method(OptimizationMethod::QUASI_NEWTON_METHOD);
    else if(new_optimization_method == "LEVENBERG_MARQUARDT_ALGORITHM")
        set_optimization_method(OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);
    else if(new_optimization_method == "STOCHASTIC_GRADIENT_DESCENT")
        set_optimization_method(OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
    else if(new_optimization_method == "ADAPTIVE_MOMENT_ESTIMATION")
        set_optimization_method(OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
    else
        throw runtime_error("Unknown main type: " + new_optimization_method + ".\n");
}


void TrainingStrategy::set_threads_number(const int& new_threads_number)
{
    mean_squared_error.set_threads_number(new_threads_number);
    normalized_squared_error.set_threads_number(new_threads_number);
    Minkowski_error.set_threads_number(new_threads_number);
    weighted_squared_error.set_threads_number(new_threads_number);
    cross_entropy_error.set_threads_number(new_threads_number);

    conjugate_gradient.set_threads_number(new_threads_number);
    quasi_Newton_method.set_threads_number(new_threads_number);
    Levenberg_Marquardt_algorithm.set_threads_number(new_threads_number);
    stochastic_gradient_descent.set_threads_number(new_threads_number);
    adaptive_moment_estimation.set_threads_number(new_threads_number);
}


void TrainingStrategy::set_data_set(DataSet* new_data_set)
{
    data_set = new_data_set;

    mean_squared_error.set_data_set(new_data_set);

    normalized_squared_error.set_data_set(new_data_set);

    cross_entropy_error.set_data_set(new_data_set);
    cross_entropy_error_3d.set_data_set(new_data_set);

    weighted_squared_error.set_data_set(new_data_set);

    Minkowski_error.set_data_set(new_data_set);
}


void TrainingStrategy::set_neural_network(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;

    mean_squared_error.set_neural_network(new_neural_network);
    normalized_squared_error.set_neural_network(new_neural_network);
    cross_entropy_error.set_neural_network(new_neural_network);
    cross_entropy_error_3d.set_neural_network(new_neural_network);
    weighted_squared_error.set_neural_network(new_neural_network);
    Minkowski_error.set_neural_network(new_neural_network);
}


void TrainingStrategy::set_loss_index(LossIndex* new_loss_index)
{
    conjugate_gradient.set_loss_index(new_loss_index);
    stochastic_gradient_descent.set_loss_index(new_loss_index);
    adaptive_moment_estimation.set_loss_index(new_loss_index);
    quasi_Newton_method.set_loss_index(new_loss_index);
    Levenberg_Marquardt_algorithm.set_loss_index(new_loss_index);
}


void TrainingStrategy::set_display(const bool& new_display)
{
    display = new_display;

    // Loss index

    mean_squared_error.set_display(display);
    normalized_squared_error.set_display(display);
    cross_entropy_error.set_display(display);
    weighted_squared_error.set_display(display);
    Minkowski_error.set_display(display);

    // Optimization algorithm

    conjugate_gradient.set_display(display);
    stochastic_gradient_descent.set_display(display);
    adaptive_moment_estimation.set_display(display);
    quasi_Newton_method.set_display(display);
    Levenberg_Marquardt_algorithm.set_display(display);
}


void TrainingStrategy::set_loss_goal(const type&  new_loss_goal)
{
    conjugate_gradient.set_loss_goal(new_loss_goal);
    quasi_Newton_method.set_loss_goal(new_loss_goal);
    Levenberg_Marquardt_algorithm.set_loss_goal(new_loss_goal);
}


void TrainingStrategy::set_maximum_selection_failures(const Index&  maximum_selection_failures)
{
    conjugate_gradient.set_maximum_selection_failures(maximum_selection_failures);
    quasi_Newton_method.set_maximum_selection_failures(maximum_selection_failures);
    Levenberg_Marquardt_algorithm.set_maximum_selection_failures(maximum_selection_failures);
}


void TrainingStrategy::set_maximum_epochs_number(const int & maximum_epochs_number)
{
    conjugate_gradient.set_maximum_epochs_number(maximum_epochs_number);
    stochastic_gradient_descent.set_maximum_epochs_number(maximum_epochs_number);
    adaptive_moment_estimation.set_maximum_epochs_number(maximum_epochs_number);
    quasi_Newton_method.set_maximum_epochs_number(maximum_epochs_number);
    Levenberg_Marquardt_algorithm.set_maximum_epochs_number(maximum_epochs_number);
}


void TrainingStrategy::set_display_period(const int & display_period)
{
    get_optimization_algorithm()->set_display_period(display_period);
}


void TrainingStrategy::set_maximum_time(const type&  maximum_time)
{
    conjugate_gradient.set_maximum_time(maximum_time);
    stochastic_gradient_descent.set_maximum_time(maximum_time);
    adaptive_moment_estimation.set_maximum_time(maximum_time);
    quasi_Newton_method.set_maximum_time(maximum_time);
    Levenberg_Marquardt_algorithm.set_maximum_time(maximum_time);
}


void TrainingStrategy::set_default()
{
    loss_method = LossMethod::MEAN_SQUARED_ERROR;

    optimization_method = OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION;

    if(has_neural_network())
        if(neural_network->get_model_type() == NeuralNetwork::ModelType::Classification)
            loss_method = LossMethod::CROSS_ENTROPY_ERROR;
}


TrainingResults TrainingStrategy::perform_training()
{
    if(!has_neural_network())
        throw runtime_error("Neural network is null.");

    if(!has_data_set())
        throw runtime_error("Data set is null.");

    if(neural_network->has(Layer::Type::Recurrent)
    || neural_network->has(Layer::Type::LongShortTermMemory))
        fix_forecasting();

    set_display(display);

    switch(optimization_method)
    {  
        case OptimizationMethod::CONJUGATE_GRADIENT:
            return conjugate_gradient.perform_training();

        case OptimizationMethod::QUASI_NEWTON_METHOD:
            return quasi_Newton_method.perform_training();
        
        case OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM:
            return Levenberg_Marquardt_algorithm.perform_training();
        
        case OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT:
            return stochastic_gradient_descent.perform_training();

        case OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION:
            return adaptive_moment_estimation.perform_training();
        
        default:
            return TrainingResults(0);
    }
}


void TrainingStrategy::fix_forecasting()
{
    Index time_steps = 0;

    if(neural_network->has(Layer::Type::Recurrent))
        time_steps = static_cast<RecurrentLayer*>(neural_network->get_first(Layer::Type::Recurrent))->get_timesteps();
    else if(neural_network->has(Layer::Type::LongShortTermMemory))
        time_steps = static_cast<LongShortTermMemoryLayer*>(neural_network->get_first(Layer::Type::LongShortTermMemory))->get_timesteps();
    else
        return;

    Index batch_samples_number = 0;

    if(optimization_method == OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION)
        batch_samples_number = adaptive_moment_estimation.get_samples_number();
    else if(optimization_method == OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT)
        batch_samples_number = stochastic_gradient_descent.get_samples_number();
    else
        return;

    if(batch_samples_number%time_steps == 0)
        return;

    const Index constant = time_steps > batch_samples_number
        ? 1
        : Index(batch_samples_number/time_steps);

    if(optimization_method == OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION)
        adaptive_moment_estimation.set_batch_samples_number(constant*time_steps);
    else if(optimization_method == OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT)
        stochastic_gradient_descent.set_batch_samples_number(constant*time_steps);

}


void TrainingStrategy::print() const
{
    cout << "Training strategy object" << endl
         << "Loss index: " << write_loss_method() << endl
         << "Optimization algorithm: " << write_optimization_method() << endl;
}


void TrainingStrategy::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("TrainingStrategy");

    printer.OpenElement("LossIndex");

    add_xml_element(printer, "LossMethod", write_loss_method());

    mean_squared_error.to_XML(printer);
    normalized_squared_error.to_XML(printer);
    Minkowski_error.to_XML(printer);
    cross_entropy_error.to_XML(printer);
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
    case LossMethod::CROSS_ENTROPY_ERROR:
        cross_entropy_error.write_regularization_XML(printer);
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

    conjugate_gradient.to_XML(printer);
    stochastic_gradient_descent.to_XML(printer);
    adaptive_moment_estimation.to_XML(printer);
    quasi_Newton_method.to_XML(printer);
    Levenberg_Marquardt_algorithm.to_XML(printer);

    printer.CloseElement();  

    add_xml_element(printer, "Display", to_string(get_display())); 

    printer.CloseElement();
}


void TrainingStrategy::from_XML(const XMLDocument& document)
{
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

    const XMLElement* cross_entropy_element = loss_index_element->FirstChildElement("CrossEntropyError");

    if (cross_entropy_element)
    {
        XMLDocument cross_entropy_document;
        XMLElement* cross_entropy_error_element_copy = cross_entropy_document.NewElement("CrossEntropyError");

        for (const XMLNode* node = cross_entropy_element->FirstChild(); node; node = node->NextSibling())
            cross_entropy_error_element_copy->InsertEndChild(node->DeepClone(&cross_entropy_document));

        cross_entropy_document.InsertEndChild(cross_entropy_error_element_copy);
        cross_entropy_error.from_XML(cross_entropy_document);
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

    // Conjugate gradient

    const XMLElement* conjugate_gradient_element = optimization_algorithm_element->FirstChildElement("ConjugateGradient");
    if (conjugate_gradient_element) {
        XMLDocument conjugate_gradient_document;
        XMLElement* conjugate_gradient_element_copy = conjugate_gradient_document.NewElement("ConjugateGradient");

        for (const XMLNode* node = conjugate_gradient_element->FirstChild(); node; node = node->NextSibling())
            conjugate_gradient_element_copy->InsertEndChild(node->DeepClone(&conjugate_gradient_document));

        conjugate_gradient_document.InsertEndChild(conjugate_gradient_element_copy);
        conjugate_gradient.from_XML(conjugate_gradient_document);
    }

    // Stochastic gradient descent

    const XMLElement* stochastic_gradient_descent_element = optimization_algorithm_element->FirstChildElement("StochasticGradientDescent");
    if (stochastic_gradient_descent_element) {
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
