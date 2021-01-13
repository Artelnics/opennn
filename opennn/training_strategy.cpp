//   OpenNN: Open Neural Networks Library+
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "training_strategy.h"
#include "optimization_algorithm.h"

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

    LossIndex* loss_index_pointer = get_loss_index_pointer();

    set_loss_index_pointer(loss_index_pointer);

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

    set_loss_index_data_set_pointer(data_set_pointer);
    set_loss_index_neural_network_pointer(neural_network_pointer);

    LossIndex* loss_index_pointer = get_loss_index_pointer();
    set_loss_index_pointer(loss_index_pointer);

    set_default();

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

LossIndex* TrainingStrategy::get_loss_index_pointer()
{
    switch (loss_method)
    {
        case SUM_SQUARED_ERROR: return &sum_squared_error;

        case MEAN_SQUARED_ERROR: return &mean_squared_error;

        case NORMALIZED_SQUARED_ERROR: return &normalized_squared_error;

        case MINKOWSKI_ERROR: return &Minkowski_error;

        case WEIGHTED_SQUARED_ERROR: return &weighted_squared_error;

        case CROSS_ENTROPY_ERROR: return &cross_entropy_error;
    }

    return nullptr;
}


/// Returns a pointer to the OptimizationAlgorithm class.

OptimizationAlgorithm* TrainingStrategy::get_optimization_algorithm_pointer()
{
    switch (optimization_method)
    {
        case GRADIENT_DESCENT: return &gradient_descent;

        case CONJUGATE_GRADIENT: return &conjugate_gradient;

        case STOCHASTIC_GRADIENT_DESCENT: return &stochastic_gradient_descent;

        case ADAPTIVE_MOMENT_ESTIMATION: return &adaptive_moment_estimation;

        case QUASI_NEWTON_METHOD: return &quasi_Newton_method;

        case LEVENBERG_MARQUARDT_ALGORITHM: return &Levenberg_Marquardt_algorithm;
    }

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
    loss_method = new_loss_method;

    set_loss_index_pointer(get_loss_index_pointer());
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


void TrainingStrategy::set_threads_number(const int& new_threads_number)
{
    set_loss_index_threads_number(new_threads_number);

    set_optimization_algorithm_threads_number(new_threads_number);
}


void TrainingStrategy::set_data_set_pointer(DataSet* new_data_set_pointer)
{
    data_set_pointer = new_data_set_pointer;

    set_loss_index_data_set_pointer(data_set_pointer);
}


void TrainingStrategy::set_neural_network_pointer(NeuralNetwork* new_neural_network_pointer)
{
    neural_network_pointer = new_neural_network_pointer;

    set_loss_index_neural_network_pointer(neural_network_pointer);
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


/// Sets a pointer to a loss index object to be associated to the training strategy.
/// @param new_loss_index_pointer Pointer to a loss index object.

void TrainingStrategy::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
    gradient_descent.set_loss_index_pointer(new_loss_index_pointer);
    conjugate_gradient.set_loss_index_pointer(new_loss_index_pointer);
    stochastic_gradient_descent.set_loss_index_pointer(new_loss_index_pointer);
    adaptive_moment_estimation.set_loss_index_pointer(new_loss_index_pointer);
    quasi_Newton_method.set_loss_index_pointer(new_loss_index_pointer);
    Levenberg_Marquardt_algorithm.set_loss_index_pointer(new_loss_index_pointer);
}


void TrainingStrategy::set_loss_index_data_set_pointer(DataSet* new_data_set_pointer)
{
    sum_squared_error.set_data_set_pointer(new_data_set_pointer);
    mean_squared_error.set_data_set_pointer(new_data_set_pointer);
    normalized_squared_error.set_data_set_pointer(new_data_set_pointer);
    cross_entropy_error.set_data_set_pointer(new_data_set_pointer);
    weighted_squared_error.set_data_set_pointer(new_data_set_pointer);
    Minkowski_error.set_data_set_pointer(new_data_set_pointer);
}


void TrainingStrategy::set_loss_index_neural_network_pointer(NeuralNetwork* new_neural_network_pointer)
{
    sum_squared_error.set_neural_network_pointer(new_neural_network_pointer);
    mean_squared_error.set_neural_network_pointer(new_neural_network_pointer);
    normalized_squared_error.set_neural_network_pointer(new_neural_network_pointer);
    cross_entropy_error.set_neural_network_pointer(new_neural_network_pointer);
    weighted_squared_error.set_neural_network_pointer(new_neural_network_pointer);
    Minkowski_error.set_neural_network_pointer(new_neural_network_pointer);
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void TrainingStrategy::set_display(const bool& new_display)
{
    display = new_display;
    gradient_descent.set_display(display);
    conjugate_gradient.set_display(display);
    stochastic_gradient_descent.set_display(display);
    adaptive_moment_estimation.set_display(display);
    quasi_Newton_method.set_display(display);
    Levenberg_Marquardt_algorithm.set_display(display);
}


void TrainingStrategy::set_loss_goal(const type & new_loss_goal)
{
    gradient_descent.set_loss_goal(new_loss_goal);
    conjugate_gradient.set_loss_goal(new_loss_goal);
    quasi_Newton_method.set_loss_goal(new_loss_goal);
    Levenberg_Marquardt_algorithm.set_loss_goal(new_loss_goal);
}


void TrainingStrategy::set_maximum_selection_error_increases(const Index & maximum_selection_error_increases)
{
    gradient_descent.set_maximum_selection_error_increases(maximum_selection_error_increases);
    conjugate_gradient.set_maximum_selection_error_increases(maximum_selection_error_increases);
    quasi_Newton_method.set_maximum_selection_error_increases(maximum_selection_error_increases);
    Levenberg_Marquardt_algorithm.set_maximum_selection_error_increases(maximum_selection_error_increases);
}


void TrainingStrategy::set_reserve_selection_error_history(const bool& reserve_selection_error)
{
    gradient_descent.set_reserve_selection_error_history(reserve_selection_error);
    conjugate_gradient.set_reserve_selection_error_history(reserve_selection_error);
    stochastic_gradient_descent.set_reserve_selection_error_history(reserve_selection_error);
    adaptive_moment_estimation.set_reserve_selection_error_history(reserve_selection_error);
    quasi_Newton_method.set_reserve_selection_error_history(reserve_selection_error);
    Levenberg_Marquardt_algorithm.set_reserve_selection_error_history(reserve_selection_error);
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
    get_optimization_algorithm_pointer()->set_display_period(display_period);
}


void TrainingStrategy::set_maximum_time(const type & maximum_time)
{
    gradient_descent.set_maximum_time(maximum_time);
    conjugate_gradient.set_maximum_time(maximum_time);
    stochastic_gradient_descent.set_maximum_time(maximum_time);
    adaptive_moment_estimation.set_maximum_time(maximum_time);
    quasi_Newton_method.set_maximum_time(maximum_time);
    Levenberg_Marquardt_algorithm.set_maximum_time(maximum_time);
}


/// Sets the members of the training strategy object to their default values:
/// <ul>
/// <li> Display: true.
/// </ul>

void TrainingStrategy::set_default()
{
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

        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "OptimizationAlgorithm::Results TrainingStrategy::perform_training() const method.\n"
               << "Long Short Term Memory Layer and Recurrent Layer are not available yet. Both of them will be included in future versions.\n";

        throw logic_error(buffer.str());

        if(!check_forecasting())
        {

            ostringstream buffer;

            buffer << "OpenNN Exception: TrainingStrategy class.\n"
                   << "OptimizationAlgorithm::Results TrainingStrategy::perform_training() const method.\n"
                   << "The batch size must be multiple of timesteps.\n";

            throw logic_error(buffer.str());
        }
    }

    if(neural_network_pointer->has_convolutional_layer())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingStrategy class.\n"
               << "OptimizationAlgorithm::Results TrainingStrategy::perform_training() const method.\n"
               << "Convolutional Layer is not available yet. It will be included in future versions.\n";

        throw logic_error(buffer.str());
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

    Index timesteps = 0;

    if(neural_network_pointer->has_recurrent_layer())
    {
        timesteps = neural_network_pointer->get_recurrent_layer_pointer()->get_timesteps();
    }
    else if(neural_network_pointer->has_long_short_term_memory_layer())
    {
        timesteps = neural_network_pointer->get_long_short_term_memory_layer_pointer()->get_timesteps();
    }


//    const Index batch_samples_number = data_set.get_batch_samples_number();

//    if(batch_samples_number%timesteps == 0)
//    {
//        return true;
//    }
//    else
//    {
//        return false;
//    }

    return true;
}


/// Prints to the screen the string representation of the training strategy object.

void TrainingStrategy::print() const
{
}


/// Serializes the training strategy object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void TrainingStrategy::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("TrainingStrategy");

    // Loss index

    file_stream.OpenElement("LossIndex");

    // Loss method

    file_stream.OpenElement("LossMethod");
    file_stream.PushText(write_loss_method().c_str());
    file_stream.CloseElement();

    mean_squared_error.write_XML(file_stream);
    normalized_squared_error.write_XML(file_stream);
    Minkowski_error.write_XML(file_stream);
    cross_entropy_error.write_XML(file_stream);
    weighted_squared_error.write_XML(file_stream);

    switch(loss_method)
    {
    case MEAN_SQUARED_ERROR : mean_squared_error.write_regularization_XML(file_stream); break;
    case NORMALIZED_SQUARED_ERROR : normalized_squared_error.write_regularization_XML(file_stream); break;
    case MINKOWSKI_ERROR : Minkowski_error.write_regularization_XML(file_stream); break;
    case CROSS_ENTROPY_ERROR : cross_entropy_error.write_regularization_XML(file_stream); break;
    case WEIGHTED_SQUARED_ERROR : weighted_squared_error.write_regularization_XML(file_stream); break;
    case SUM_SQUARED_ERROR : sum_squared_error.write_regularization_XML(file_stream); break;
    }

    file_stream.CloseElement();

    // Optimization algorithm

    file_stream.OpenElement("OptimizationAlgorithm");

    file_stream.OpenElement("OptimizationMethod");
    file_stream.PushText(write_optimization_method().c_str());
    file_stream.CloseElement();

    gradient_descent.write_XML(file_stream);
    conjugate_gradient.write_XML(file_stream);
    stochastic_gradient_descent.write_XML(file_stream);
    adaptive_moment_estimation.write_XML(file_stream);
    quasi_Newton_method.write_XML(file_stream);
    Levenberg_Marquardt_algorithm.write_XML(file_stream);

    file_stream.CloseElement();

    // Close TrainingStrategy

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

    const tinyxml2::XMLElement* loss_index_element = root_element->FirstChildElement("LossIndex");

    if(loss_index_element)
    {
        const tinyxml2::XMLElement* loss_method_element = loss_index_element->FirstChildElement("LossMethod");

        set_loss_method(loss_method_element->GetText());

        // Mean squared error

        //            const tinyxml2::XMLElement* mean_squared_error_element = loss_index_element->FirstChildElement("MeanSquaredError");

        // Normalized squared error

        //            const tinyxml2::XMLElement* normalized_squared_error_element = loss_index_element->FirstChildElement("NormalizedSquaredError");

        // Minkowski error

        const tinyxml2::XMLElement* Minkowski_error_element = loss_index_element->FirstChildElement("MinkowskiError");

        if(Minkowski_error_element)
        {
            tinyxml2::XMLDocument new_document;

            tinyxml2::XMLElement* Minkowski_error_element_copy = new_document.NewElement("MinkowskiError");

            for(const tinyxml2::XMLNode* nodeFor=Minkowski_error_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
            {
                tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                Minkowski_error_element_copy->InsertEndChild(copy );
            }

            new_document.InsertEndChild(Minkowski_error_element_copy);

            Minkowski_error.from_XML(new_document);
        }
        else
        {
            Minkowski_error.set_Minkowski_parameter(1.5);
        }

        // Cross entropy error

        const tinyxml2::XMLElement* cross_entropy_element = loss_index_element->FirstChildElement("CrossEntropyError");

        if(cross_entropy_element)
        {
            tinyxml2::XMLDocument new_document;

            tinyxml2::XMLElement* cross_entropy_error_element_copy = new_document.NewElement("CrossEntropyError");

            for(const tinyxml2::XMLNode* nodeFor=loss_index_element->FirstChild(); nodeFor; nodeFor=nodeFor->NextSibling())
            {
                tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                cross_entropy_error_element_copy->InsertEndChild(copy );
            }

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
            {
                tinyxml2::XMLNode* copy = nodeFor->DeepClone(&new_document );
                weighted_squared_error_element_copy->InsertEndChild(copy );
            }

            new_document.InsertEndChild(weighted_squared_error_element_copy);

            weighted_squared_error.from_XML(new_document);
        }
        else
        {
            weighted_squared_error.set_positives_weight(1);
            weighted_squared_error.set_negatives_weight(1);
        }

        // Regularization

        const tinyxml2::XMLElement* regularization_element = loss_index_element->FirstChildElement("Regularization");

        if(regularization_element)
        {
            tinyxml2::XMLDocument regularization_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = regularization_element->DeepClone(&regularization_document);

            regularization_document.InsertFirstChild(element_clone);

            get_loss_index_pointer()->regularization_from_XML(regularization_document);
        }
    }

    cout << "Loss index loaded" << endl;

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
            {
                tinyxml2::XMLNode* copy = nodeFor->DeepClone(&gradient_descent_document );
                gradient_descent_element_copy->InsertEndChild(copy );
            }

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
            {
                tinyxml2::XMLNode* copy = nodeFor->DeepClone(&conjugate_gradient_document );
                conjugate_gradient_element_copy->InsertEndChild(copy );
            }

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
            {
                tinyxml2::XMLNode* copy = nodeFor->DeepClone(&stochastic_gradient_descent_document );
                stochastic_gradient_descent_element_copy->InsertEndChild(copy );
            }

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
            {
                tinyxml2::XMLNode* copy = nodeFor->DeepClone(&adaptive_moment_estimation_document );
                adaptive_moment_estimation_element_copy->InsertEndChild(copy );
            }

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
            {
                tinyxml2::XMLNode* copy = nodeFor->DeepClone(&quasi_Newton_document );
                quasi_newton_method_element_copy->InsertEndChild(copy );
            }

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
            {
                tinyxml2::XMLNode* copy = nodeFor->DeepClone(&Levenberg_Marquardt_document );
                levenberg_marquardt_algorithm_element_copy->InsertEndChild(copy );
            }

            Levenberg_Marquardt_document.InsertEndChild(levenberg_marquardt_algorithm_element_copy);

            Levenberg_Marquardt_algorithm.from_XML(Levenberg_Marquardt_document);
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

    FILE *pFile;

    pFile = fopen(file_name.c_str(), "w");

    tinyxml2::XMLPrinter document(pFile);

    write_XML(document);

    fclose(pFile);
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
