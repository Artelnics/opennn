//   OpenNN: Open Neural Networks Library+
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "training_strategy.h"
#include "optimization_algorithm.h"

namespace opennn
{

TrainingStrategy::TrainingStrategy()
{
    set_loss_method(LossMethod::NORMALIZED_SQUARED_ERROR);

    set_optimization_method(OptimizationMethod::QUASI_NEWTON_METHOD);

    LossIndex* loss_index = get_loss_index();

    set_loss_index(loss_index);

    set_default();

}


TrainingStrategy::TrainingStrategy(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : data_set(new_data_set),
       neural_network(new_neural_network)
{

    set_loss_method(LossMethod::NORMALIZED_SQUARED_ERROR);

    set_optimization_method(OptimizationMethod::QUASI_NEWTON_METHOD);

    set_loss_index_neural_network(neural_network);

    set_loss_index_data_set(data_set);

    set_default();
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
        case LossMethod::SUM_SQUARED_ERROR: return &sum_squared_error;

        case LossMethod::MEAN_SQUARED_ERROR: return &mean_squared_error;

        case LossMethod::NORMALIZED_SQUARED_ERROR: return &normalized_squared_error;

        case LossMethod::MINKOWSKI_ERROR: return &Minkowski_error;

        case LossMethod::WEIGHTED_SQUARED_ERROR: return &weighted_squared_error;

        case LossMethod::CROSS_ENTROPY_ERROR: return &cross_entropy_error;

        case LossMethod::CROSS_ENTROPY_ERROR_3D: return &cross_entropy_error_3d;

        default: return nullptr;
    }
}


OptimizationAlgorithm* TrainingStrategy::get_optimization_algorithm()
{
    switch(optimization_method)
    {
    case OptimizationMethod::GRADIENT_DESCENT: return &gradient_descent;

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
    return neural_network != nullptr;
}


bool TrainingStrategy::has_data_set() const
{
    return data_set != nullptr;
}


GradientDescent* TrainingStrategy::get_gradient_descent()
{
    return &gradient_descent;
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


SumSquaredError* TrainingStrategy::get_sum_squared_error()
{
    return &sum_squared_error;
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
    case LossMethod::SUM_SQUARED_ERROR:
        return "SUM_SQUARED_ERROR";

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
    case OptimizationMethod::GRADIENT_DESCENT:
        return "GRADIENT_DESCENT";
    
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
    case OptimizationMethod::GRADIENT_DESCENT:
        return "gradient descent";

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
    case LossMethod::SUM_SQUARED_ERROR:
        return "Sum squared error";

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


void TrainingStrategy::set()
{
    set_optimization_method(OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

    set_default();
}


void TrainingStrategy::set(NeuralNetwork* new_neural_network, DataSet* new_data_set)
{
    set_neural_network(new_neural_network);

    set_data_set(new_data_set);
}


void TrainingStrategy::set_loss_method(const string& new_loss_method)
{
    if(new_loss_method == "SUM_SQUARED_ERROR")
        set_loss_method(LossMethod::SUM_SQUARED_ERROR);
    else if(new_loss_method == "MEAN_SQUARED_ERROR")
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
    if(new_optimization_method == "GRADIENT_DESCENT")
        set_optimization_method(OptimizationMethod::GRADIENT_DESCENT);
    else if(new_optimization_method == "CONJUGATE_GRADIENT")
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
    set_loss_index_threads_number(new_threads_number);

    set_optimization_algorithm_threads_number(new_threads_number);
}


void TrainingStrategy::set_data_set(DataSet* new_data_set)
{
    data_set = new_data_set;

    set_loss_index_data_set(data_set);
}


void TrainingStrategy::set_neural_network(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;

    set_loss_index_neural_network(neural_network);
}


void TrainingStrategy::set_loss_index_threads_number(const int& new_threads_number)
{
    sum_squared_error.set_threads_number(new_threads_number);
    mean_squared_error.set_threads_number(new_threads_number);
    normalized_squared_error.set_threads_number(new_threads_number);
    Minkowski_error.set_threads_number(new_threads_number);
    weighted_squared_error.set_threads_number(new_threads_number);
    cross_entropy_error.set_threads_number(new_threads_number);
}


void TrainingStrategy::set_optimization_algorithm_threads_number(const int& new_threads_number)
{
    gradient_descent.set_threads_number(new_threads_number);
    conjugate_gradient.set_threads_number(new_threads_number);
    quasi_Newton_method.set_threads_number(new_threads_number);
    Levenberg_Marquardt_algorithm.set_threads_number(new_threads_number);
    stochastic_gradient_descent.set_threads_number(new_threads_number);
    adaptive_moment_estimation.set_threads_number(new_threads_number);
}


void TrainingStrategy::set_loss_index(LossIndex* new_loss_index)
{
    gradient_descent.set_loss_index(new_loss_index);
    conjugate_gradient.set_loss_index(new_loss_index);
    stochastic_gradient_descent.set_loss_index(new_loss_index);
    adaptive_moment_estimation.set_loss_index(new_loss_index);
    quasi_Newton_method.set_loss_index(new_loss_index);
    Levenberg_Marquardt_algorithm.set_loss_index(new_loss_index);
}


void TrainingStrategy::set_loss_index_data_set(DataSet* new_data_set)
{
    sum_squared_error.set_data_set(new_data_set);
    mean_squared_error.set_data_set(new_data_set);
    normalized_squared_error.set_data_set(new_data_set);
    cross_entropy_error.set_data_set(new_data_set);
    cross_entropy_error_3d.set_data_set(new_data_set);
    weighted_squared_error.set_data_set(new_data_set);
    Minkowski_error.set_data_set(new_data_set);
}


void TrainingStrategy::set_loss_index_neural_network(NeuralNetwork* new_neural_network)
{
    sum_squared_error.set_neural_network(new_neural_network);
    mean_squared_error.set_neural_network(new_neural_network);
    normalized_squared_error.set_neural_network(new_neural_network);
    cross_entropy_error.set_neural_network(new_neural_network);
    cross_entropy_error_3d.set_neural_network(new_neural_network);
    weighted_squared_error.set_neural_network(new_neural_network);
    Minkowski_error.set_neural_network(new_neural_network);
}


void TrainingStrategy::set_display(const bool& new_display)
{
    display = new_display;

    // Loss index

    sum_squared_error.set_display(display);
    mean_squared_error.set_display(display);
    normalized_squared_error.set_display(display);
    cross_entropy_error.set_display(display);
    weighted_squared_error.set_display(display);
    Minkowski_error.set_display(display);

    // Optimization algorithm

    gradient_descent.set_display(display);
    conjugate_gradient.set_display(display);
    stochastic_gradient_descent.set_display(display);
    adaptive_moment_estimation.set_display(display);
    quasi_Newton_method.set_display(display);
    Levenberg_Marquardt_algorithm.set_display(display);
}


void TrainingStrategy::set_loss_goal(const type&  new_loss_goal)
{
    gradient_descent.set_loss_goal(new_loss_goal);
    conjugate_gradient.set_loss_goal(new_loss_goal);
    quasi_Newton_method.set_loss_goal(new_loss_goal);
    Levenberg_Marquardt_algorithm.set_loss_goal(new_loss_goal);
}


void TrainingStrategy::set_maximum_selection_failures(const Index&  maximum_selection_failures)
{
    gradient_descent.set_maximum_selection_failures(maximum_selection_failures);
    conjugate_gradient.set_maximum_selection_failures(maximum_selection_failures);
    quasi_Newton_method.set_maximum_selection_failures(maximum_selection_failures);
    Levenberg_Marquardt_algorithm.set_maximum_selection_failures(maximum_selection_failures);
}


void TrainingStrategy::set_maximum_epochs_number(const int & maximum_epochs_number)
{
    gradient_descent.set_maximum_epochs_number(maximum_epochs_number);
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
    gradient_descent.set_maximum_time(maximum_time);
    conjugate_gradient.set_maximum_time(maximum_time);
    stochastic_gradient_descent.set_maximum_time(maximum_time);
    adaptive_moment_estimation.set_maximum_time(maximum_time);
    quasi_Newton_method.set_maximum_time(maximum_time);
    Levenberg_Marquardt_algorithm.set_maximum_time(maximum_time);
}


void TrainingStrategy::set_default() const
{
}


TrainingResults TrainingStrategy::perform_training()
{    
    if(neural_network->has_recurrent_layer()
    || neural_network->has_long_short_term_memory_layer())
        fix_forecasting();

    set_display(display);

    switch(optimization_method)
    {
        case OptimizationMethod::GRADIENT_DESCENT:
            return gradient_descent.perform_training();
       
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

    if(neural_network->has_recurrent_layer())
        time_steps = neural_network->get_recurrent_layer()->get_timesteps();
    else if(neural_network->has_long_short_term_memory_layer())
        time_steps = neural_network->get_long_short_term_memory_layer()->get_timesteps();
    else
        return;

    Index batch_samples_number = 0;

    if(optimization_method == OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION)
        batch_samples_number = adaptive_moment_estimation.get_batch_samples_number();
    else if(optimization_method == OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT)
        batch_samples_number = stochastic_gradient_descent.get_batch_samples_number();
    else
        return;

    if(batch_samples_number%time_steps == 0)
    {
        return;
    }
    else
    {
        const Index constant = time_steps > batch_samples_number 
            ? 1 
            : Index(batch_samples_number/time_steps);

        if(optimization_method == OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION)
            adaptive_moment_estimation.set_batch_samples_number(constant*time_steps);
        else if(optimization_method == OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT)
            stochastic_gradient_descent.set_batch_samples_number(constant*time_steps);
    }
}


void TrainingStrategy::print() const
{
    cout << "Training strategy object" << endl
         << "Loss index: " << write_loss_method() << endl
         << "Optimization algorithm: " << write_optimization_method() << endl;
}


void TrainingStrategy::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("TrainingStrategy");

    // Loss index

    file_stream.OpenElement("LossIndex");

    // Loss method

    file_stream.OpenElement("LossMethod");
    file_stream.PushText(write_loss_method().c_str());
    file_stream.CloseElement();

    mean_squared_error.to_XML(file_stream);
    normalized_squared_error.to_XML(file_stream);
    Minkowski_error.to_XML(file_stream);
    cross_entropy_error.to_XML(file_stream);
    weighted_squared_error.to_XML(file_stream);

    switch(loss_method)
    {
    case LossMethod::MEAN_SQUARED_ERROR : mean_squared_error.write_regularization_XML(file_stream); break;
    case LossMethod::NORMALIZED_SQUARED_ERROR : normalized_squared_error.write_regularization_XML(file_stream); break;
    case LossMethod::MINKOWSKI_ERROR : Minkowski_error.write_regularization_XML(file_stream); break;
    case LossMethod::CROSS_ENTROPY_ERROR : cross_entropy_error.write_regularization_XML(file_stream); break;
    case LossMethod::WEIGHTED_SQUARED_ERROR : weighted_squared_error.write_regularization_XML(file_stream); break;
    case LossMethod::SUM_SQUARED_ERROR : sum_squared_error.write_regularization_XML(file_stream); break;
    default: break;
    }

    file_stream.CloseElement();

    // Optimization algorithm

    file_stream.OpenElement("OptimizationAlgorithm");

    file_stream.OpenElement("OptimizationMethod");
    file_stream.PushText(write_optimization_method().c_str());
    file_stream.CloseElement();

    gradient_descent.to_XML(file_stream);
    conjugate_gradient.to_XML(file_stream);
    stochastic_gradient_descent.to_XML(file_stream);
    adaptive_moment_estimation.to_XML(file_stream);
    quasi_Newton_method.to_XML(file_stream);
    Levenberg_Marquardt_algorithm.to_XML(file_stream);

    file_stream.CloseElement();

    // Close TrainingStrategy

    file_stream.CloseElement();
}


void TrainingStrategy::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("TrainingStrategy");

    if(!root_element)
        throw runtime_error("Training strategy element is nullptr.\n");

    // Loss index

    const tinyxml2::XMLElement* loss_index_element = root_element->FirstChildElement("LossIndex");

    if(loss_index_element)
    {
        const tinyxml2::XMLElement* loss_method_element = loss_index_element->FirstChildElement("LossMethod");

        set_loss_method(loss_method_element->GetText());

        // Minkowski error

        const tinyxml2::XMLElement* Minkowski_error_element = loss_index_element->FirstChildElement("MinkowskiError");

        if(Minkowski_error_element)
        {
            tinyxml2::XMLDocument new_document;

            tinyxml2::XMLElement* Minkowski_error_element_copy = new_document.NewElement("MinkowskiError");

            for(const tinyxml2::XMLNode* nodeFor=Minkowski_error_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                Minkowski_error_element_copy->InsertEndChild(nodeFor->DeepClone(&new_document));

            new_document.InsertEndChild(Minkowski_error_element_copy);

            Minkowski_error.from_XML(new_document);
        }

        // Cross entropy error

        const tinyxml2::XMLElement* cross_entropy_element = loss_index_element->FirstChildElement("CrossEntropyError");

        if(cross_entropy_element)
        {
            tinyxml2::XMLDocument new_document;

            tinyxml2::XMLElement* cross_entropy_error_element_copy = new_document.NewElement("CrossEntropyError");

            for(const tinyxml2::XMLNode* nodeFor=loss_index_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                cross_entropy_error_element_copy->InsertEndChild(nodeFor->DeepClone(&new_document));

            new_document.InsertEndChild(cross_entropy_error_element_copy);

            cross_entropy_error.from_XML(new_document);
        }

        // Weighted squared error

        const tinyxml2::XMLElement* weighted_squared_error_element = loss_index_element->FirstChildElement("WeightedSquaredError");

        if(weighted_squared_error_element)
        {
            tinyxml2::XMLDocument new_document;

            tinyxml2::XMLElement* weighted_squared_error_element_copy = new_document.NewElement("WeightedSquaredError");

            for(const tinyxml2::XMLNode* nodeFor=weighted_squared_error_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                weighted_squared_error_element_copy->InsertEndChild(nodeFor->DeepClone(&new_document));

            new_document.InsertEndChild(weighted_squared_error_element_copy);

            weighted_squared_error.from_XML(new_document);
        }

        // Regularization

        const tinyxml2::XMLElement* regularization_element = loss_index_element->FirstChildElement("Regularization");

        if(regularization_element)
        {
            tinyxml2::XMLDocument regularization_document;
            regularization_document.InsertFirstChild(regularization_element->DeepClone(&regularization_document));
            get_loss_index()->regularization_from_XML(regularization_document);
        }
    }

    // Optimization algorithm

    const tinyxml2::XMLElement* optimization_algorithm_element = root_element->FirstChildElement("OptimizationAlgorithm");

    if(optimization_algorithm_element)
    {
        const tinyxml2::XMLElement* optimization_method_element = optimization_algorithm_element->FirstChildElement("OptimizationMethod");

        set_optimization_method(optimization_method_element->GetText());

        // Gradient descent

        const tinyxml2::XMLElement* gradient_descent_element = optimization_algorithm_element->FirstChildElement("GradientDescent");

        if(gradient_descent_element)
        {
            tinyxml2::XMLDocument gradient_descent_document;

            tinyxml2::XMLElement* gradient_descent_element_copy = gradient_descent_document.NewElement("GradientDescent");

            for(const tinyxml2::XMLNode* nodeFor=gradient_descent_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                gradient_descent_element_copy->InsertEndChild(nodeFor->DeepClone(&gradient_descent_document));
     
            gradient_descent_document.InsertEndChild(gradient_descent_element_copy);

            gradient_descent.from_XML(gradient_descent_document);
        }

        // Conjugate gradient

        const tinyxml2::XMLElement* conjugate_gradient_element = optimization_algorithm_element->FirstChildElement("ConjugateGradient");

        if(conjugate_gradient_element)
        {
            tinyxml2::XMLDocument conjugate_gradient_document;

            tinyxml2::XMLElement* conjugate_gradient_element_copy = conjugate_gradient_document.NewElement("ConjugateGradient");

            for(const tinyxml2::XMLNode* nodeFor=conjugate_gradient_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                conjugate_gradient_element_copy->InsertEndChild(nodeFor->DeepClone(&conjugate_gradient_document));

            conjugate_gradient_document.InsertEndChild(conjugate_gradient_element_copy);

            conjugate_gradient.from_XML(conjugate_gradient_document);
        }

        // Stochastic gradient

        const tinyxml2::XMLElement* stochastic_gradient_descent_element = optimization_algorithm_element->FirstChildElement("StochasticGradientDescent");

        if(stochastic_gradient_descent_element)
        {
            tinyxml2::XMLDocument stochastic_gradient_descent_document;

            tinyxml2::XMLElement* stochastic_gradient_descent_element_copy = stochastic_gradient_descent_document.NewElement("StochasticGradientDescent");

            for(const tinyxml2::XMLNode* nodeFor=stochastic_gradient_descent_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                stochastic_gradient_descent_element_copy->InsertEndChild(nodeFor->DeepClone(&stochastic_gradient_descent_document));

            stochastic_gradient_descent_document.InsertEndChild(stochastic_gradient_descent_element_copy);

            stochastic_gradient_descent.from_XML(stochastic_gradient_descent_document);
        }

        // Adaptive moment estimation

        const tinyxml2::XMLElement* adaptive_moment_estimation_element = optimization_algorithm_element->FirstChildElement("AdaptiveMomentEstimation");

        if(adaptive_moment_estimation_element)
        {
            tinyxml2::XMLDocument adaptive_moment_estimation_document;

            tinyxml2::XMLElement* adaptive_moment_estimation_element_copy = adaptive_moment_estimation_document.NewElement("AdaptiveMomentEstimation");

            for(const tinyxml2::XMLNode* nodeFor=adaptive_moment_estimation_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                adaptive_moment_estimation_element_copy->InsertEndChild(nodeFor->DeepClone(&adaptive_moment_estimation_document));

            adaptive_moment_estimation_document.InsertEndChild(adaptive_moment_estimation_element_copy);

            adaptive_moment_estimation.from_XML(adaptive_moment_estimation_document);
        }

        // Quasi-Newton method

        const tinyxml2::XMLElement* quasi_Newton_method_element = optimization_algorithm_element->FirstChildElement("QuasiNewtonMethod");

        if(quasi_Newton_method_element)
        {
            tinyxml2::XMLDocument quasi_Newton_document;

            tinyxml2::XMLElement* quasi_newton_method_element_copy = quasi_Newton_document.NewElement("QuasiNewtonMethod");

            for(const tinyxml2::XMLNode* nodeFor=quasi_Newton_method_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                quasi_newton_method_element_copy->InsertEndChild(nodeFor->DeepClone(&quasi_Newton_document));

            quasi_Newton_document.InsertEndChild(quasi_newton_method_element_copy);

            quasi_Newton_method.from_XML(quasi_Newton_document);
        }

        // Levenberg Marquardt

        const tinyxml2::XMLElement* Levenberg_Marquardt_element = optimization_algorithm_element->FirstChildElement("LevenbergMarquardt");

        if(Levenberg_Marquardt_element)
        {
            tinyxml2::XMLDocument Levenberg_Marquardt_document;
            tinyxml2::XMLElement* levenberg_marquardt_algorithm_element_copy = Levenberg_Marquardt_document.NewElement("LevenbergMarquardt");

            for(const tinyxml2::XMLNode* nodeFor=Levenberg_Marquardt_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                levenberg_marquardt_algorithm_element_copy->InsertEndChild(nodeFor->DeepClone(&Levenberg_Marquardt_document));

            Levenberg_Marquardt_document.InsertEndChild(levenberg_marquardt_algorithm_element_copy);

            Levenberg_Marquardt_algorithm.from_XML(Levenberg_Marquardt_document);
        }
    }

    // Display

    const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

    if(display_element)
        set_display(display_element->GetText() != string("0"));
}


void TrainingStrategy::save(const string& file_name) const
{
    FILE * file = fopen(file_name.c_str(), "w");

    if(file)
    {
        tinyxml2::XMLPrinter printer(file);
        to_XML(printer);
        fclose(file);
    }
}


void TrainingStrategy::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
        throw runtime_error("Cannot load XML file " + file_name + ".\n");

    from_XML(document);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
