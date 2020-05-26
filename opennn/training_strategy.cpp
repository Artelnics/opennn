//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "training_strategy.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a training strategy object not associated to any loss index object.
/// It also constructs the main optimization algorithm object.

TrainingStrategy::TrainingStrategy()
{
    data_set_pointer = nullptr;

    neural_network_pointer = nullptr;

    set_loss_method(NORMALIZED_SQUARED_ERROR);

    set_optimization_method(QUASI_NEWTON_METHOD);

    set_default();
}


/// Pointer constuctor.
/// It creates a training strategy object not associated to any loss index object.
/// It also loads the members of this object from NeuralNetwork and DataSet class.

TrainingStrategy::TrainingStrategy(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
{
    data_set_pointer = new_data_set_pointer;

    neural_network_pointer = new_neural_network_pointer;

    set_optimization_method(QUASI_NEWTON_METHOD);

    set_loss_method(NORMALIZED_SQUARED_ERROR);

    set_default();
}


/// XML constructor.
/// It creates a training strategy object not associated to any loss index object.
/// It also loads the members of this object from a XML document.
/// @param document Document of the TinyXML library.

TrainingStrategy::TrainingStrategy(const tinyxml2::XMLDocument& document)
{
    set_optimization_method(QUASI_NEWTON_METHOD);

    set_default();

    from_XML(document);
}


/// File constructor.
/// It creates a training strategy object associated to a loss index object.
/// It also loads the members of this object from a XML file.
/// @param file_name Name of training strategy XML file.

TrainingStrategy::TrainingStrategy(const string& file_name)
{
    set_optimization_method(QUASI_NEWTON_METHOD);

    set_default();

    load(file_name);
}


/// Destructor.
/// This destructor deletes the loss index and optimization algorithm objects.

TrainingStrategy::~TrainingStrategy()
{
}


/// Returns a pointer to the NeuralNetwork class.

NeuralNetwork* TrainingStrategy::get_neural_network_pointer() const
{
    return neural_network_pointer;
}


/// Returns a pointer to the LossIndex class.

LossIndex* TrainingStrategy::get_loss_index_pointer() const
{
/*
    if(loss_method == SUM_SQUARED_ERROR && sum_squared_error_pointer != nullptr) return sum_squared_error_pointer;
    else if(loss_method == MEAN_SQUARED_ERROR && mean_squared_error_pointer != nullptr) return mean_squared_error_pointer;
    else if(loss_method == NORMALIZED_SQUARED_ERROR && normalized_squared_error_pointer != nullptr) return normalized_squared_error_pointer;
    else if(loss_method == MINKOWSKI_ERROR && Minkowski_error_pointer != nullptr) return Minkowski_error_pointer;
    else if(loss_method == WEIGHTED_SQUARED_ERROR && weighted_squared_error_pointer != nullptr) return weighted_squared_error_pointer;
    else if(loss_method == CROSS_ENTROPY_ERROR && cross_entropy_error_pointer != nullptr) return cross_entropy_error_pointer;
    else return nullptr;
*/
    return nullptr;
}


/// Returns a pointer to the OptimizationAlgorithm class.

OptimizationAlgorithm* TrainingStrategy::get_optimization_algorithm_pointer() const
{
/*
    if(optimization_method == GRADIENT_DESCENT && gradient_descent_pointer != nullptr) return gradient_descent_pointer;
    else if(optimization_method == CONJUGATE_GRADIENT && conjugate_gradient_pointer != nullptr) return conjugate_gradient_pointer;
    else if(optimization_method == QUASI_NEWTON_METHOD && quasi_Newton_method_pointer != nullptr) return quasi_Newton_method_pointer;
    else if(optimization_method == LEVENBERG_MARQUARDT_ALGORITHM && Levenberg_Marquardt_algorithm_pointer != nullptr) return Levenberg_Marquardt_algorithm_pointer;
    else if(optimization_method == STOCHASTIC_GRADIENT_DESCENT && stochastic_gradient_descent_pointer != nullptr) return stochastic_gradient_descent_pointer;
    else if(optimization_method == ADAPTIVE_MOMENT_ESTIMATION && adaptive_moment_estimation_pointer != nullptr) return adaptive_moment_estimation_pointer;
    else return nullptr;
*/
    return nullptr;
}


bool TrainingStrategy::has_neural_network() const
{
    if(neural_network_pointer == nullptr) return false;

    return true;
}


bool TrainingStrategy::has_data_set() const
{
    if(data_set_pointer == nullptr) return false;

    return true;
}


/// Return true if contain loss index, and false otherwise.

bool TrainingStrategy::has_loss_index() const
{
    if(get_loss_index_pointer() == nullptr)
    {
        return false;
    }
    else
    {
        return true;
    }
}


bool TrainingStrategy::has_optimization_algorithm() const
{
    if(get_optimization_algorithm_pointer() == nullptr)
    {
        return false;
    }
    else
    {
        return true;
    }
}


/// Returns a pointer to the gradient descent main algorithm.
/// It also throws an exception if that pointer is nullptr.

GradientDescent* TrainingStrategy::get_gradient_descent_pointer()
{
    return &gradient_descent;
}


/// Returns a pointer to the conjugate gradient main algorithm.
/// It also throws an exception if that pointer is nullptr.

ConjugateGradient* TrainingStrategy::get_conjugate_gradient_pointer()
{
    return &conjugate_gradient;
}


/// Returns a pointer to the Newton method main algorithm.
/// It also throws an exception if that pointer is nullptr.

QuasiNewtonMethod* TrainingStrategy::get_quasi_Newton_method_pointer()
{
    return &quasi_Newton_method;
}


/// Returns a pointer to the Levenberg-Marquardt main algorithm.
/// It also throws an exception if that pointer is nullptr.

LevenbergMarquardtAlgorithm* TrainingStrategy::get_Levenberg_Marquardt_algorithm_pointer()
{
    return &Levenberg_Marquardt_algorithm;
}


/// Returns a pointer to the stochastic gradient descent main algorithm.
/// It also throws an exception if that pointer is nullptr.

StochasticGradientDescent* TrainingStrategy::get_stochastic_gradient_descent_pointer()
{
    return &stochastic_gradient_descent;
}


/// Returns a pointer to the adaptive moment estimation main algorithm.
/// It also throws an exception if that pointer is nullptr.

AdaptiveMomentEstimation* TrainingStrategy::get_adaptive_moment_estimation_pointer()
{
    return &adaptive_moment_estimation;
}


/// Returns a pointer to the sum squared error which is used as error.
/// If that object does not exists, an exception is thrown.

SumSquaredError* TrainingStrategy::get_sum_squared_error_pointer()
{
    return &sum_squared_error;
}


/// Returns a pointer to the mean squared error which is used as error.
/// If that object does not exists, an exception is thrown.

MeanSquaredError* TrainingStrategy::get_mean_squared_error_pointer()
{
    return &mean_squared_error;
}


/// Returns a pointer to the normalized squared error which is used as error.
/// If that object does not exists, an exception is thrown.

NormalizedSquaredError* TrainingStrategy::get_normalized_squared_error_pointer()
{

    return &normalized_squared_error;
}



/// Returns a pointer to the Minkowski error which is used as error.
/// If that object does not exists, an exception is thrown.

MinkowskiError* TrainingStrategy::get_Minkowski_error_pointer()
{

    return &Minkowski_error;
}


/// Returns a pointer to the cross entropy error which is used as error.
/// If that object does not exists, an exception is thrown.

CrossEntropyError* TrainingStrategy::get_cross_entropy_error_pointer()
{
    return &cross_entropy_error;
}


/// Returns a pointer to the weighted squared error which is used as error.
/// If that object does not exists, an exception is thrown.

WeightedSquaredError* TrainingStrategy::get_weighted_squared_error_pointer()
{
    return &weighted_squared_error;
}

/// Returns the type of the main loss algorithm composing this training strategy object.

const TrainingStrategy::LossMethod& TrainingStrategy::get_loss_method() const
{
    return loss_method;
}


/// Returns the type of the main optimization algorithm composing this training strategy object.

const TrainingStrategy::OptimizationMethod& TrainingStrategy::get_optimization_method() const
{
    return optimization_method;
}


/// Returns a string with the type of the main loss algorithm composing this training strategy object.

string TrainingStrategy::write_loss_method() const
{
    switch(loss_method)
    {
    case SUM_SQUARED_ERROR:
        return "SUM_SQUARED_ERROR";

    case MEAN_SQUARED_ERROR:
        return "MEAN_SQUARED_ERROR";

    case NORMALIZED_SQUARED_ERROR:
        return "NORMALIZED_SQUARED_ERROR";

    case MINKOWSKI_ERROR:
        return "MINKOWSKI_ERROR";

    case WEIGHTED_SQUARED_ERROR:
        return "WEIGHTED_SQUARED_ERROR";

    case CROSS_ENTROPY_ERROR:
        return "CROSS_ENTROPY_ERROR";
    }

    return string();
}


/// Returns a string with the type of the main optimization algorithm composing this training strategy object.
/// If that object does not exists, an exception is thrown.

string TrainingStrategy::write_optimization_method() const
{
    if(optimization_method == GRADIENT_DESCENT)
    {
        return "GRADIENT_DESCENT";
    }
    else if(optimization_method == CONJUGATE_GRADIENT)
    {
        return "CONJUGATE_GRADIENT";
    }
    else if(optimization_method == QUASI_NEWTON_METHOD)
    {
        return "QUASI_NEWTON_METHOD";
    }
    else if(optimization_method == LEVENBERG_MARQUARDT_ALGORITHM)
    {
        return "LEVENBERG_MARQUARDT_ALGORITHM";
    }
    else if(optimization_method == STOCHASTIC_GRADIENT_DESCENT)
    {
        return "STOCHASTIC_GRADIENT_DESCENT";
    }
    else if(optimization_method == ADAPTIVE_MOMENT_ESTIMATION)
    {
        return "ADAPTIVE_MOMENT_ESTIMATION";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "string write_optimization_method() const method.\n"
               << "Unknown main type.\n";

        throw logic_error(buffer.str());
    }
}


/// Returns a string with the main type in text format.
/// If that object does not exists, an exception is thrown.

string TrainingStrategy::write_optimization_method_text() const
{
    if(optimization_method == GRADIENT_DESCENT)
    {
        return "gradient descent";
    }
    else if(optimization_method == CONJUGATE_GRADIENT)
    {
        return "conjugate gradient";
    }
    else if(optimization_method == QUASI_NEWTON_METHOD)
    {
        return "quasi-Newton method";
    }
    else if(optimization_method == LEVENBERG_MARQUARDT_ALGORITHM)
    {
        return "Levenberg-Marquardt algorithm";
    }
    else if(optimization_method == STOCHASTIC_GRADIENT_DESCENT)
    {
        return "stochastic gradient descent";
    }
    else if(optimization_method == ADAPTIVE_MOMENT_ESTIMATION)
    {
        return "adaptive moment estimation";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "string write_optimization_method_text() const method.\n"
               << "Unknown main type.\n";

        throw logic_error(buffer.str());
    }
}


/// Returns a string with the main loss method type in text format.

string TrainingStrategy::write_loss_method_text() const
{
    switch(loss_method)
    {
    case SUM_SQUARED_ERROR:
    {
        return "Sum squared error";
    }

    case MEAN_SQUARED_ERROR:
    {
        return "Mean squared error";
    }

    case NORMALIZED_SQUARED_ERROR:
    {
        return "Normalized squared error";
    }

    case MINKOWSKI_ERROR:
    {
        return "Minkowski error";
    }

    case WEIGHTED_SQUARED_ERROR:
    {
        return "Weighted squared error";
    }

    case CROSS_ENTROPY_ERROR:
    {
        return "Cross entropy error";
    }
    }

    return string();
}


/// Returns true if messages from this class can be displayed on the screen, or false if messages from
/// this class can't be displayed on the screen.

const bool& TrainingStrategy::get_display() const
{
    return display;
}


/// Sets the loss index pointer to nullptr.
/// It also destructs the loss index and the optimization algorithm.
/// Finally, it sets the rest of members to their default values.

void TrainingStrategy::set()
{
    set_optimization_method(QUASI_NEWTON_METHOD);

    set_default();
}


/// Sets the loss index method.
/// If that object does not exists, an exception is thrown.
/// @param new_loss_method String with the name of the new method.

void TrainingStrategy::set_loss_method(const string& new_loss_method)
{
    if(new_loss_method == "SUM_SQUARED_ERROR")
    {
        set_loss_method(SUM_SQUARED_ERROR);
    }
    else if(new_loss_method == "MEAN_SQUARED_ERROR")
    {
        set_loss_method(MEAN_SQUARED_ERROR);
    }
    else if(new_loss_method == "NORMALIZED_SQUARED_ERROR")
    {
        set_loss_method(NORMALIZED_SQUARED_ERROR);
    }
    else if(new_loss_method == "MINKOWSKI_ERROR")
    {
        set_loss_method(MINKOWSKI_ERROR);
    }
    else if(new_loss_method == "WEIGHTED_SQUARED_ERROR")
    {
        set_loss_method(WEIGHTED_SQUARED_ERROR);
    }
    else if(new_loss_method == "CROSS_ENTROPY_ERROR")
    {
        set_loss_method(CROSS_ENTROPY_ERROR);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "void set_loss_method(const string&) method.\n"
               << "Unknown loss method: " << new_loss_method << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets the loss index method.
/// If that object does not exists, an exception is thrown.
/// @param new_loss_method New method type.

void TrainingStrategy::set_loss_method(const LossMethod& new_loss_method)
{
    LossIndex::RegularizationMethod regularization_method;

    loss_method = new_loss_method;
}


/// Sets a new type of main optimization algorithm.
/// @param new_optimization_method Type of main optimization algorithm.

void TrainingStrategy::set_optimization_method(const OptimizationMethod& new_optimization_method)
{
    optimization_method = new_optimization_method;
}


/// Sets a new main optimization algorithm from a string containing the type.
/// @param new_optimization_method String with the type of main optimization algorithm.

void TrainingStrategy::set_optimization_method(const string& new_optimization_method)
{
    if(new_optimization_method == "GRADIENT_DESCENT")
    {
        set_optimization_method(GRADIENT_DESCENT);
    }
    else if(new_optimization_method == "CONJUGATE_GRADIENT")
    {
        set_optimization_method(CONJUGATE_GRADIENT);
    }
    else if(new_optimization_method == "QUASI_NEWTON_METHOD")
    {
        set_optimization_method(QUASI_NEWTON_METHOD);
    }
    else if(new_optimization_method == "LEVENBERG_MARQUARDT_ALGORITHM")
    {
        set_optimization_method(LEVENBERG_MARQUARDT_ALGORITHM);
    }
    else if(new_optimization_method == "STOCHASTIC_GRADIENT_DESCENT")
    {
        set_optimization_method(STOCHASTIC_GRADIENT_DESCENT);
    }
    else if(new_optimization_method == "ADAPTIVE_MOMENT_ESTIMATION")
    {
        set_optimization_method(ADAPTIVE_MOMENT_ESTIMATION);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "void set_optimization_method(const string&) method.\n"
               << "Unknown main type: " << new_optimization_method << ".\n";

        throw logic_error(buffer.str());
    }
}


void TrainingStrategy::set_thread_pool_device(ThreadPoolDevice* new_thread_pool_device)
{
    get_loss_index_pointer()->set_thread_pool_device(new_thread_pool_device);

    get_optimization_algorithm_pointer()->set_thread_pool_device(new_thread_pool_device);
}


/// Sets a pointer to a loss index object to be associated to the training strategy.
/// @param new_loss_index_pointer Pointer to a loss index object.

void TrainingStrategy::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
    // Main

    switch(optimization_method)
    {
    case GRADIENT_DESCENT:
    {
        gradient_descent.set_loss_index_pointer(new_loss_index_pointer);
    }
    break;

    case CONJUGATE_GRADIENT:
    {
        conjugate_gradient.set_loss_index_pointer(new_loss_index_pointer);
    }
    break;

    case QUASI_NEWTON_METHOD:
    {
        quasi_Newton_method.set_loss_index_pointer(new_loss_index_pointer);
    }
    break;

    case LEVENBERG_MARQUARDT_ALGORITHM:
    {
        Levenberg_Marquardt_algorithm.set_loss_index_pointer(new_loss_index_pointer);
    }
    break;

    case STOCHASTIC_GRADIENT_DESCENT:
    {
        stochastic_gradient_descent.set_loss_index_pointer(new_loss_index_pointer);
    }
    break;

    case ADAPTIVE_MOMENT_ESTIMATION:
    {
        adaptive_moment_estimation.set_loss_index_pointer(new_loss_index_pointer);
    }
    break;
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void TrainingStrategy::set_display(const bool& new_display)
{
    display = new_display;

    switch(optimization_method)
    {
    case GRADIENT_DESCENT:
    {
        gradient_descent.set_display(display);
    }
    break;

    case CONJUGATE_GRADIENT:
    {
        conjugate_gradient.set_display(display);
    }
    break;

    case QUASI_NEWTON_METHOD:
    {
        quasi_Newton_method.set_display(display);
    }
    break;

    case LEVENBERG_MARQUARDT_ALGORITHM:
    {
        Levenberg_Marquardt_algorithm.set_display(display);
    }
    break;

    case STOCHASTIC_GRADIENT_DESCENT:
    {
        stochastic_gradient_descent.set_display(display);
    }
    break;

    case ADAPTIVE_MOMENT_ESTIMATION:
    {
        adaptive_moment_estimation.set_display(display);
    }
    break;
    }
}


/// Sets the members of the training strategy object to their default values:
/// <ul>
/// <li> Display: true.
/// </ul>

void TrainingStrategy::set_default()
{
    display = true;
}


/// This is the most important method of this class.
/// It optimizes the loss index of a neural network.
/// This method also returns a structure with the results from training.

OptimizationAlgorithm::Results TrainingStrategy::perform_training()
{
#ifdef __OPENNN_DEBUG__

//    check_loss_index();

//    check_optimization_algorithms();

#endif

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer())
    {
        if(!check_forecasting())
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: TrainingStrategy class.\n"
                   << "OptimizationAlgorithm::Results TrainingStrategy::perform_training() const method.\n"
                   << "The batch size must be multiple of timesteps.\n";

            throw logic_error(buffer.str());
        }
    }

    OptimizationAlgorithm::Results results;

    // Main

    switch(optimization_method)
    {
    case GRADIENT_DESCENT:
    {
        gradient_descent.set_display(display);

        results = gradient_descent.perform_training();

    }
    break;

    case CONJUGATE_GRADIENT:
    {
        conjugate_gradient.set_display(display);

        results = conjugate_gradient.perform_training();
    }
    break;

    case QUASI_NEWTON_METHOD:
    {
        quasi_Newton_method.set_display(display);

        results = quasi_Newton_method.perform_training();
    }
    break;

    case LEVENBERG_MARQUARDT_ALGORITHM:
    {
        Levenberg_Marquardt_algorithm.set_display(display);

        results = Levenberg_Marquardt_algorithm.perform_training();
    }
    break;

    case STOCHASTIC_GRADIENT_DESCENT:
    {
        stochastic_gradient_descent.set_display(display);

        results = stochastic_gradient_descent.perform_training();

    }
    break;

    case ADAPTIVE_MOMENT_ESTIMATION:
    {
        adaptive_moment_estimation.set_display(display);

        results = adaptive_moment_estimation.perform_training();
    }
    break;
    }

    return results;
}


/// Perfom the training with the selected method.

void TrainingStrategy::perform_training_void()
{
#ifdef __OPENNN_DEBUG__

//    check_loss_index();

//    check_optimization_algorithms();

#endif

    switch(optimization_method)
    {
    case GRADIENT_DESCENT:
    {
        gradient_descent.set_display(display);

        gradient_descent.perform_training_void();
    }
    break;

    case CONJUGATE_GRADIENT:
    {
        conjugate_gradient.set_display(display);

        conjugate_gradient.perform_training_void();
    }
    break;

    case QUASI_NEWTON_METHOD:
    {
        quasi_Newton_method.set_display(display);

        quasi_Newton_method.perform_training_void();
    }
    break;

    case LEVENBERG_MARQUARDT_ALGORITHM:
    {
        Levenberg_Marquardt_algorithm.set_display(display);

        Levenberg_Marquardt_algorithm.perform_training_void();
    }
    break;

    case STOCHASTIC_GRADIENT_DESCENT:
    {
        stochastic_gradient_descent.set_display(display);

        stochastic_gradient_descent.perform_training_void();
    }
    break;


    case ADAPTIVE_MOMENT_ESTIMATION:
    {
        adaptive_moment_estimation.set_display(display);

        adaptive_moment_estimation.perform_training_void();
    }
    break;
    }
}


/// Check the time steps and the batch size in forecasting problems
/// @todo

bool TrainingStrategy::check_forecasting() const
{
/*
    const Index batch_instances_number = data_set.get_batch_instances_number();
    Index timesteps = 0;

    if(neural_network.has_recurrent_layer())
    {
        timesteps = neural_network.get_recurrent_layer_pointer()->get_timesteps();
    }
    else if(neural_network.has_long_short_term_memory_layer())
    {
        timesteps = neural_network.get_long_short_term_memory_layer_pointer()->get_timesteps();
    }

    if(batch_instances_number%timesteps == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
*/
    return false;
}


/// Returns a string representation of the training strategy.

string TrainingStrategy::object_to_string() const
{
    ostringstream buffer;

    buffer << "Training strategy\n";

    // Main

    buffer << "Loss method: " << write_loss_method() << "\n";

    buffer << "Training method: " << write_optimization_method() << "\n";

    switch(optimization_method)
    {
    case GRADIENT_DESCENT:

        buffer << gradient_descent.object_to_string();

        break;

    case CONJUGATE_GRADIENT:

        buffer << conjugate_gradient.object_to_string();

        break;

    case QUASI_NEWTON_METHOD:

        buffer << quasi_Newton_method.object_to_string();

        break;

    case LEVENBERG_MARQUARDT_ALGORITHM:

        buffer << Levenberg_Marquardt_algorithm.object_to_string();

        break;

    case STOCHASTIC_GRADIENT_DESCENT:

        buffer << stochastic_gradient_descent.object_to_string();

        break;

    case ADAPTIVE_MOMENT_ESTIMATION:

        buffer << adaptive_moment_estimation.object_to_string();

        break;
    }

    return buffer.str();
}


/// Prints to the screen the string representation of the training strategy object.

void TrainingStrategy::print() const
{
    cout << object_to_string();
}


/// Returns a default string representation in XML-type format of the optimization algorithm object.
/// This containts the training operators, the training parameters, stopping criteria and other stuff.

tinyxml2::XMLDocument* TrainingStrategy::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Training strategy

    tinyxml2::XMLElement* training_strategy_element = document->NewElement("TrainingStrategy");

    document->InsertFirstChild(training_strategy_element);

    // Main

    switch(optimization_method)
    {
    case GRADIENT_DESCENT:
    {
        tinyxml2::XMLElement* main_element = document->NewElement("Main");
        training_strategy_element->LinkEndChild(main_element);

        main_element->SetAttribute("Type", "GRADIENT_DESCENT");

        const tinyxml2::XMLDocument* gradient_descent_document = gradient_descent.to_XML();

        const tinyxml2::XMLElement* gradient_descent_element = gradient_descent_document->FirstChildElement("GradientDescent");

        for(const tinyxml2::XMLNode* nodeFor=gradient_descent_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
        {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone(document );
            main_element->InsertEndChild(copy );
        }

        delete gradient_descent_document;
    }
    break;

    case CONJUGATE_GRADIENT:
    {
        tinyxml2::XMLElement* main_element = document->NewElement("Main");
        training_strategy_element->LinkEndChild(main_element);

        main_element->SetAttribute("Type", "CONJUGATE_GRADIENT");

        const tinyxml2::XMLDocument* conjugate_gradient_document = conjugate_gradient.to_XML();

        const tinyxml2::XMLElement* conjugate_gradient_element = conjugate_gradient_document->FirstChildElement("ConjugateGradient");

        for(const tinyxml2::XMLNode* nodeFor=conjugate_gradient_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
        {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone(document );
            main_element->InsertEndChild(copy );
        }

        delete conjugate_gradient_document;
    }
    break;

    case QUASI_NEWTON_METHOD:
    {
        tinyxml2::XMLElement* main_element = document->NewElement("Main");
        training_strategy_element->LinkEndChild(main_element);

        main_element->SetAttribute("Type", "QUASI_NEWTON_METHOD");

        const tinyxml2::XMLDocument* quasi_Newton_method_document = quasi_Newton_method.to_XML();

        const tinyxml2::XMLElement* quasi_Newton_method_element = quasi_Newton_method_document->FirstChildElement("QuasiNewtonMethod");

        for(const tinyxml2::XMLNode* nodeFor=quasi_Newton_method_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
        {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone(document );
            main_element->InsertEndChild(copy );
        }

        delete quasi_Newton_method_document;
    }
    break;

    case LEVENBERG_MARQUARDT_ALGORITHM:
    {
        tinyxml2::XMLElement* main_element = document->NewElement("Main");
        training_strategy_element->LinkEndChild(main_element);

        main_element->SetAttribute("Type", "LEVENBERG_MARQUARDT_ALGORITHM");

        const tinyxml2::XMLDocument* Levenberg_Marquardt_algorithm_document = Levenberg_Marquardt_algorithm.to_XML();

        const tinyxml2::XMLElement* Levenberg_Marquardt_algorithm_element = Levenberg_Marquardt_algorithm_document->FirstChildElement("LevenbergMarquardtAlgorithm");

        for(const tinyxml2::XMLNode* nodeFor=Levenberg_Marquardt_algorithm_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
        {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone(document );
            main_element->InsertEndChild(copy );
        }

        delete Levenberg_Marquardt_algorithm_document;
    }
    break;

    case STOCHASTIC_GRADIENT_DESCENT:
    {
        tinyxml2::XMLElement* main_element = document->NewElement("Main");
        training_strategy_element->LinkEndChild(main_element);

        main_element->SetAttribute("Type", "STOCHASTIC_GRADIENT_DESCENT");

        const tinyxml2::XMLDocument* stochastic_gradient_descent_document = stochastic_gradient_descent.to_XML();

        const tinyxml2::XMLElement* stochastic_gradient_descent_element = stochastic_gradient_descent_document->FirstChildElement("StochasticGradientDescent");

        for(const tinyxml2::XMLNode* nodeFor = stochastic_gradient_descent_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
        {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone(document );
            main_element->InsertEndChild(copy );
        }

        delete stochastic_gradient_descent_document;
    }
    break;

    case ADAPTIVE_MOMENT_ESTIMATION:
    {
        tinyxml2::XMLElement* main_element = document->NewElement("Main");
        training_strategy_element->LinkEndChild(main_element);

        main_element->SetAttribute("Type", "ADAPTIVE_MOMENT_ESTIMATION");

        const tinyxml2::XMLDocument* adaptive_moment_estimation_document = adaptive_moment_estimation.to_XML();

        const tinyxml2::XMLElement* adaptive_moment_estimation_element = adaptive_moment_estimation_document->FirstChildElement("AdaptiveMomentEstimation");

        for(const tinyxml2::XMLNode* nodeFor = adaptive_moment_estimation_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
        {
            tinyxml2::XMLNode* copy = nodeFor->DeepClone(document );
            main_element->InsertEndChild(copy );
        }

        delete adaptive_moment_estimation_document;
    }

    break;
    }

    // Display
//   {
//      element = document->NewElement("Display");
//      training_strategy_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << display;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

    return document;
}


/// Serializes the training strategy object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void TrainingStrategy::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("TrainingStrategy");

    // Loss index

    switch(loss_method)
    {
    case SUM_SQUARED_ERROR:
    {
        file_stream.OpenElement("LossIndex");

        sum_squared_error.write_XML(file_stream);

        file_stream.CloseElement();
    }
    break;

    case MEAN_SQUARED_ERROR:
    {
        file_stream.OpenElement("LossIndex");

        mean_squared_error.write_XML(file_stream);

        file_stream.CloseElement();
    }
    break;

    case NORMALIZED_SQUARED_ERROR:
    {
        file_stream.OpenElement("LossIndex");

        normalized_squared_error.write_XML(file_stream);

        file_stream.CloseElement();
    }
    break;

    case MINKOWSKI_ERROR:
    {
        file_stream.OpenElement("LossIndex");

        Minkowski_error.write_XML(file_stream);

        file_stream.CloseElement();
    }
    break;

    case CROSS_ENTROPY_ERROR:
    {
        file_stream.OpenElement("LossIndex");

        cross_entropy_error.write_XML(file_stream);

        file_stream.CloseElement();
    }
    break;

    case WEIGHTED_SQUARED_ERROR:
    {
        file_stream.OpenElement("LossIndex");

        weighted_squared_error.write_XML(file_stream);

        file_stream.CloseElement();
    }
    }

    switch(optimization_method)
    {
    case GRADIENT_DESCENT:
    {
        file_stream.OpenElement("Main");

        file_stream.PushAttribute("Type", "GRADIENT_DESCENT");

        gradient_descent.write_XML(file_stream);

        file_stream.CloseElement();
    }
    break;

    case CONJUGATE_GRADIENT:
    {
        file_stream.OpenElement("Main");

        file_stream.PushAttribute("Type", "CONJUGATE_GRADIENT");

        conjugate_gradient.write_XML(file_stream);

        file_stream.CloseElement();
    }
    break;

    case QUASI_NEWTON_METHOD:
    {
        file_stream.OpenElement("Main");

        file_stream.PushAttribute("Type", "QUASI_NEWTON_METHOD");

        quasi_Newton_method.write_XML(file_stream);

        file_stream.CloseElement();
    }
    break;

    case LEVENBERG_MARQUARDT_ALGORITHM:
    {
        file_stream.OpenElement("Main");

        file_stream.PushAttribute("Type", "LEVENBERG_MARQUARDT_ALGORITHM");

        Levenberg_Marquardt_algorithm.write_XML(file_stream);

        file_stream.CloseElement();
    }
    break;

    case STOCHASTIC_GRADIENT_DESCENT:
    {
        file_stream.OpenElement("Main");

        file_stream.PushAttribute("Type", "STOCHASTIC_GRADIENT_DESCENT");

        stochastic_gradient_descent.write_XML(file_stream);

        file_stream.CloseElement();
    }
    break;


    case ADAPTIVE_MOMENT_ESTIMATION:
    {
        file_stream.OpenElement("Main");

        file_stream.PushAttribute("Type", "ADAPTIVE_MOMENT_ESTIMATION");

        adaptive_moment_estimation.write_XML(file_stream);

        file_stream.CloseElement();
    }
    break;

    }

    file_stream.CloseElement();
}


/// Loads the members of this training strategy object from a XML document.
/// @param document XML document of the TinyXML library.

void TrainingStrategy::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("TrainingStrategy");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Training strategy element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Loss index
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("LossIndex");

        if(element)
        {
            const tinyxml2::XMLElement* error_element = element->FirstChildElement("Error");

            const string new_loss_method = error_element->Attribute("Type");

            set_loss_method(new_loss_method);

            switch(loss_method)
            {
            case SUM_SQUARED_ERROR:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* sum_squared_error_element = new_document.NewElement("SumSquaredError");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    sum_squared_error_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(sum_squared_error_element);

                sum_squared_error.from_XML(new_document);
            }
            break;

            case MEAN_SQUARED_ERROR:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* mean_squared_error_element = new_document.NewElement("MeanSquaredError");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    mean_squared_error_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(mean_squared_error_element);

                mean_squared_error.from_XML(new_document);
            }
            break;

            case NORMALIZED_SQUARED_ERROR:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* normalized_squared_error_element = new_document.NewElement("NormalizedSquaredError");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    normalized_squared_error_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(normalized_squared_error_element);

                normalized_squared_error.from_XML(new_document);
            }
            break;

            case MINKOWSKI_ERROR:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* Minkowski_error_element = new_document.NewElement("MinkowskiError");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    Minkowski_error_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(Minkowski_error_element);

                Minkowski_error.from_XML(new_document);
            }
            break;

            case CROSS_ENTROPY_ERROR:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* cross_entropy_error_element = new_document.NewElement("CrossEntropyError");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    cross_entropy_error_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(cross_entropy_error_element);

                cross_entropy_error.from_XML(new_document);
            }
            break;

            case WEIGHTED_SQUARED_ERROR:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* weighted_squared_error_element = new_document.NewElement("WeightedSquaredError");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    weighted_squared_error_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(weighted_squared_error_element);

                weighted_squared_error.from_XML(new_document);
            }
            }
        }

    }

    // Main
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Main");

        if(element)
        {
            const string new_optimization_method = element->Attribute("Type");

            set_optimization_method(new_optimization_method);

            switch(optimization_method)
            {
            case GRADIENT_DESCENT:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* gradient_descent_element = new_document.NewElement("GradientDescent");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    gradient_descent_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(gradient_descent_element);

                gradient_descent.from_XML(new_document);
            }
            break;

            case CONJUGATE_GRADIENT:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* conjugate_gradient_element = new_document.NewElement("ConjugateGradient");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    conjugate_gradient_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(conjugate_gradient_element);

                conjugate_gradient.from_XML(new_document);
            }
            break;

            case QUASI_NEWTON_METHOD:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* quasi_newton_method_element = new_document.NewElement("QuasiNewtonMethod");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    quasi_newton_method_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(quasi_newton_method_element);

                quasi_Newton_method.from_XML(new_document);
            }
            break;

            case LEVENBERG_MARQUARDT_ALGORITHM:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* levenberg_marquardt_algorithm_element = new_document.NewElement("LevenbergMarquardtAlgorithm");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    levenberg_marquardt_algorithm_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(levenberg_marquardt_algorithm_element);

                Levenberg_Marquardt_algorithm.from_XML(new_document);
            }
            break;

            case STOCHASTIC_GRADIENT_DESCENT:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* stochastic_gradient_descent_element = new_document.NewElement("StochasticGradientDescent");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    stochastic_gradient_descent_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(stochastic_gradient_descent_element);

                stochastic_gradient_descent.from_XML(new_document);
            }
            break;

            case ADAPTIVE_MOMENT_ESTIMATION:
            {
                tinyxml2::XMLDocument new_document;

                tinyxml2::XMLElement* adaptive_moment_estimation_element = new_document.NewElement("AdaptiveMomentEstimation");

                for(const tinyxml2::XMLNode* nodeFor=element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
                {
                    tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                    adaptive_moment_estimation_element->InsertEndChild(copy );
                }

                new_document.InsertEndChild(adaptive_moment_estimation_element);

                adaptive_moment_estimation.from_XML(new_document);
            }
            break;

            }
        }
    }

    // Display
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

        if(element)
        {
            const string new_display = element->GetText();

            try
            {
                set_display(new_display != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

}


/// Saves to a XML-type file the members of the optimization algorithm object.
/// @param file_name Name of optimization algorithm XML-type file.

void TrainingStrategy::save(const string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
}


/// Loads a gradient descent object from a XML-type file.
/// Please mind about the file format, wich is specified in the User's Guide.
/// @param file_name Name of optimization algorithm XML-type file.

void TrainingStrategy::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw logic_error(buffer.str());
    }

    from_XML(document);
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
