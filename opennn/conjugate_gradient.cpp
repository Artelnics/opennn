//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N J U G A T E   G R A D I E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "conjugate_gradient.h"
#include "training_strategy.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a conjugate gradient optimization algorithm object not associated to any loss index object.
/// It also initializes the class members to their default values.

ConjugateGradient::ConjugateGradient()
    : OptimizationAlgorithm()
{
    set_default();
}


/// Destructor.
/// It creates a conjugate gradient optimization algorithm associated to a loss index object.
/// It also initializes the rest of class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

ConjugateGradient::ConjugateGradient(LossIndex* new_loss_index_pointer)
    : OptimizationAlgorithm(new_loss_index_pointer)
{
    learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);

    set_default();
}


/// Destructor.

ConjugateGradient::~ConjugateGradient()
{
}


/// Returns a constant reference to the learning rate algorithm object inside the conjugate gradient method object.

const LearningRateAlgorithm& ConjugateGradient::get_learning_rate_algorithm() const
{
    return learning_rate_algorithm;
}


/// Returns a pointer to the learning rate algorithm object inside the conjugate gradient method object.

LearningRateAlgorithm* ConjugateGradient::get_learning_rate_algorithm_pointer()
{
    return &learning_rate_algorithm;
}


/// Returns the conjugate gradient training direction method used for training.

const ConjugateGradient::TrainingDirectionMethod& ConjugateGradient::get_training_direction_method() const
{
    return training_direction_method;
}


/// Returns a string with the name of the training direction.

string ConjugateGradient::write_training_direction_method() const
{
    switch(training_direction_method)
    {
    case PR:
        return "PR";

    case FR:
        return "FR";
    }

    return string();
}


/// Returns the minimum norm of the parameter increment vector used as a stopping criteria when training.

const type& ConjugateGradient::get_minimum_parameters_increment_norm() const
{
    return minimum_parameters_increment_norm;    
}


/// Returns the minimum loss improvement during training.

const type& ConjugateGradient::get_minimum_loss_decrease() const
{
    return minimum_loss_decrease;
}


/// Returns the goal value for the loss.
/// This is used as a stopping criterion when training a neural network

const type& ConjugateGradient::get_loss_goal() const
{
    return training_loss_goal;
}


/// Returns the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a neural network

const type& ConjugateGradient::get_gradient_norm_goal() const
{
    return gradient_norm_goal;
}


/// Returns the maximum number of selection error increases during the training process.

const Index& ConjugateGradient::get_maximum_selection_error_increases() const
{
    return maximum_selection_error_increases;
}


/// Returns the maximum number of epochs for training.

const Index& ConjugateGradient::get_maximum_epochs_number() const
{
    return maximum_epochs_number;
}


/// Returns the maximum training time.

const type& ConjugateGradient::get_maximum_time() const
{
    return maximum_time;
}


/// Returns true if the final model will be the neural network with the minimum selection error, false otherwise.

const bool& ConjugateGradient::get_choose_best_selection() const
{
    return choose_best_selection;
}


/// Returns true if the error history vector is to be reserved, and false otherwise.

const bool& ConjugateGradient::get_reserve_training_error_history() const
{
    return reserve_training_error_history;
}


/// Returns true if the selection error history vector is to be reserved, and false otherwise.

const bool& ConjugateGradient::get_reserve_selection_error_history() const
{
    return reserve_selection_error_history;
}


/// Sets a pointer to a loss index object to be associated to the conjugate gradient object.
/// It also sets that loss index to the learning rate algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void ConjugateGradient::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
    loss_index_pointer = new_loss_index_pointer;

    learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);
}


/// Sets a new training direction method to be used for training.
/// @param new_training_direction_method Conjugate gradient training direction method.

void ConjugateGradient::set_training_direction_method
(const ConjugateGradient::TrainingDirectionMethod& new_training_direction_method)
{
    training_direction_method = new_training_direction_method;
}


/// Sets a new conjugate gradient training direction from a string representation.
/// Possible values are:
/// <ul>
/// <li> "PR"
/// <li> "FR"
/// </ul>
/// @param new_training_direction_method_name String with the name of the training direction method.

void ConjugateGradient::set_training_direction_method(const string& new_training_direction_method_name)
{
    if(new_training_direction_method_name == "PR")
    {
        training_direction_method = PR;

    }
    else if(new_training_direction_method_name == "FR")
    {
        training_direction_method = FR;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "void set_training_direction_method(const string&) method.\n"
               << "Unknown training direction method: " << new_training_direction_method_name << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Makes the training history of all variables to reseved or not in memory when training.
/// <ul>
/// <li> Parameters.
/// <li> Parameters norm.
/// <li> loss.
/// <li> Gradient.
/// <li> Gradient norm.
/// <li> selection error.
/// <li> Training direction.
/// <li> Training direction norm.
/// <li> Learning rate.
/// </ul>
///
/// @param new_reserve_all_training_history True if all training history variables are to be reserved,
/// false otherwise.

void ConjugateGradient::set_reserve_all_training_history(const bool& new_reserve_all_training_history)
{
    reserve_training_error_history = new_reserve_all_training_history;
    reserve_selection_error_history = new_reserve_all_training_history;
}


/// Sets the default values into a conjugate gradient object.
/// Training operators:
/// <ul>
/// <li> Training direction method = Polak-Ribiere;
/// <li> Learning rate method = Brent;
/// </ul>
/// Training parameters:
/// <ul>
/// <li> First learning rate: 1.0.
/// <li> Bracketing factor: 2.0.
/// <li> Learning rate tolerance: 1.0e-3.
/// </ul>
/// Stopping criteria:
/// <ul>
/// <li> Loss goal: -numeric_limits<type>::max().
/// <li> Gradient norm goal: 0.0.
/// <li> Maximum training time: 1.0e6.
/// <li> Maximum number of epochs: 100.
/// </ul>
/// User stuff:
/// <ul>
/// <li> Warning learning rate: 1.0e6.
/// <li> Error learning rate: 1.0e12.
/// <li> Display: true.
/// <li> Display period: 10.
/// <li> Save period: 0.
/// </ul>
/// Reserve:
/// <ul>
/// <li> Reserve training error history: false.
/// <li> Reserve training direction norm history: false.
/// </ul>
///

void ConjugateGradient::set_default()
{
    // Stopping criteria

    minimum_parameters_increment_norm = 0;

    minimum_loss_decrease = 0;
    training_loss_goal = 0;
    gradient_norm_goal = 0;
    maximum_selection_error_increases = 1000000;

    maximum_epochs_number = 1000;
    maximum_time = 3600.0;

    choose_best_selection = false;

    // TRAINING HISTORY

    reserve_training_error_history = true;
    reserve_selection_error_history = true;

    // UTILITIES

    display_period = 5;

    training_direction_method = FR;
}


/// Sets a new value for the minimum parameters increment norm stopping criterion.
/// @param new_minimum_parameters_increment_norm Value of norm of parameters increment norm used to stop training.

void ConjugateGradient::set_minimum_parameters_increment_norm(const type& new_minimum_parameters_increment_norm)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_parameters_increment_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "void new_minimum_parameters_increment_norm(const type&) method.\n"
               << "Minimum parameters increment norm must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set error learning rate

    minimum_parameters_increment_norm = new_minimum_parameters_increment_norm;
}


/// Sets a new minimum loss improvement during training.
/// @param new_minimum_loss_decrease Minimum improvement in the loss between two iterations.

void ConjugateGradient::set_minimum_loss_decrease(const type& new_minimum_loss_decrease)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_loss_decrease < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "void set_minimum_loss_decrease(const type&) method.\n"
               << "Minimum loss improvement must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set minimum loss improvement

    minimum_loss_decrease = new_minimum_loss_decrease;
}


/// Sets a new goal value for the loss.
/// This is used as a stopping criterion when training a neural network
/// @param new_loss_goal Goal value for the loss.

void ConjugateGradient::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


/// Sets a new the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a neural network
/// @param new_gradient_norm_goal Goal value for the norm of the error function gradient.

void ConjugateGradient::set_gradient_norm_goal(const type& new_gradient_norm_goal)
{
#ifdef __OPENNN_DEBUG__

    if(new_gradient_norm_goal < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "void set_gradient_norm_goal(const type&) method.\n"
               << "Gradient norm goal must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set gradient norm goal

    gradient_norm_goal = new_gradient_norm_goal;
}


/// Sets a new maximum number of selection error increases.
/// @param new_maximum_selection_error_increases Maximum number of epochs in which the selection evalutation increases.

void ConjugateGradient::set_maximum_selection_error_increases(const Index& new_maximum_selection_error_increases)
{
    maximum_selection_error_increases = new_maximum_selection_error_increases;
}


/// Sets a maximum number of epochs for training.
/// @param new_maximum_iterations_number Maximum number of epochs for training.

void ConjugateGradient::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


/// Sets a new maximum training time.
/// @param new_maximum_time Maximum training time.

void ConjugateGradient::set_maximum_time(const type& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set maximum time

    maximum_time = new_maximum_time;
}


/// Makes the minimum selection error neural network of all the iterations to be returned or not.
/// @param new_choose_best_selection True if the final model will be the neural network with the minimum selection error, false otherwise.

void ConjugateGradient::set_choose_best_selection(const bool& new_choose_best_selection)
{
    choose_best_selection = new_choose_best_selection;
}


/// Makes the error history vector to be reseved or not in memory.
/// @param new_reserve_training_error_history True if the loss history vector is to be reserved, false otherwise.

void ConjugateGradient::set_reserve_training_error_history(const bool& new_reserve_training_error_history)
{
    reserve_training_error_history = new_reserve_training_error_history;
}


/// Makes the selection error history to be reserved or not in memory.
/// This is a vector.
/// @param new_reserve_selection_error_history True if the selection error history is to be reserved, false otherwise.

void ConjugateGradient::set_reserve_selection_error_history(const bool& new_reserve_selection_error_history)
{
    reserve_selection_error_history = new_reserve_selection_error_history;
}


/// Sets a new number of iterations between the training saving progress.
/// @param new_save_period
/// Number of iterations between the training saving progress.

void ConjugateGradient::set_save_period(const Index& new_save_period)
{
#ifdef __OPENNN_DEBUG__

    if(new_save_period <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "void set_save_period(const type&) method.\n"
               << "Save period must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    save_period = new_save_period;
}


/// Returns the Fletcher-Reeves parameter used to calculate the training direction.
/// @param old_gradient Previous error function gradient.
/// @param gradient: Current error function gradient.

type ConjugateGradient::calculate_FR_parameter(const Tensor<type, 1>& old_gradient, const Tensor<type, 1>& gradient) const
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "type calculate_FR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"

               << "Loss index pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    const Index old_gradient_size = old_gradient.size();

    if(old_gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "type calculate_FR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of old gradient(" << old_gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

    const Index gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "type calculate_FR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    type FR_parameter = 0;

    Tensor<type, 0> numerator;
    Tensor<type, 0> denominator;

    numerator.device(*thread_pool_device) = gradient.contract(gradient, AT_B);
    denominator.device(*thread_pool_device) = old_gradient.contract(old_gradient, AT_B);

    // Prevent a possible division by 0

    if(abs(denominator(0)) < numeric_limits<type>::min())
    {
        FR_parameter = 0;
    }
    else
    {
        FR_parameter = numerator(0)/denominator(0);
    }

    // Bound the Fletcher-Reeves parameter between 0 and 1

    if(FR_parameter < static_cast<type>(0.0))
    {
        FR_parameter = 0;
    }
    else if(FR_parameter > static_cast<type>(1.0))
    {
        FR_parameter = 1;
    }

    return FR_parameter;
}


/// Returns the Polak-Ribiere parameter used to calculate the training direction.
/// @param old_gradient Previous error function gradient.
/// @param gradient Current error function gradient.

type ConjugateGradient::calculate_PR_parameter(const Tensor<type, 1>& old_gradient, const Tensor<type, 1>& gradient) const
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "type calculate_PR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"

               << "Loss index pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    const Index old_gradient_size = old_gradient.size();

    if(old_gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "type calculate_PR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of old gradient(" << old_gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

    const Index gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "type calculate_PR_parameter(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    type PR_parameter = 0;

    Tensor<type, 0> numerator;
    Tensor<type, 0> denominator;

    numerator.device(*thread_pool_device) = (gradient-old_gradient).contract(gradient, AT_B);
    denominator.device(*thread_pool_device) = old_gradient.contract(old_gradient, AT_B);

    // Prevent a possible division by 0

    if(abs(denominator(0)) < numeric_limits<type>::min())
    {
        PR_parameter = 0;
    }
    else
    {
        PR_parameter = numerator(0)/denominator(0);
    }

    // Bound the Polak-Ribiere parameter between 0 and 1

    if(PR_parameter < static_cast<type>(0.0))
    {
        PR_parameter = 0;
    }
    else if(PR_parameter > static_cast<type>(1.0))
    {
        PR_parameter = 1.0;
    }

    return PR_parameter;

}


/// Returns the training direction using the Polak-Ribiere update.
/// @param old_gradient Previous error function gradient.
/// @param gradient Current error function gradient.
/// @param old_training_direction Previous training direction vector.

void ConjugateGradient::calculate_PR_training_direction(const Tensor<type, 1>& old_gradient,
                                                        const Tensor<type, 1>& gradient,
                                                        const Tensor<type, 1>& old_training_direction,
                                                        Tensor<type, 1>& training_direction) const
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_PR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Loss index pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    const Index old_gradient_size = old_gradient.size();

    if(old_gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_PR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of old gradient(" << old_gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

    const Index gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_PR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

    const Index old_training_direction_size = old_training_direction.size();

    if(old_training_direction_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_PR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of old training direction(" << old_training_direction_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const type PR_parameter = calculate_PR_parameter(old_gradient, gradient);

    training_direction.device(*thread_pool_device) = -gradient + old_training_direction*PR_parameter;
}


/// Returns the training direction using the Fletcher-Reeves update.
/// @param old_gradient Previous error function gradient.
/// @param gradient Current error function gradient.
/// @param old_training_direction Previous training direction vector.

void ConjugateGradient::calculate_FR_training_direction(const Tensor<type, 1>& old_gradient,
                                                        const Tensor<type, 1>& gradient,
                                                        const Tensor<type, 1>& old_training_direction,
                                                        Tensor<type, 1>& training_direction) const
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_FR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Loss index pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    const Index old_gradient_size = old_gradient.size();

    if(old_gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_FR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of old gradient(" << old_gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

    const Index gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_FR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

    const Index old_training_direction_size = old_training_direction.size();

    if(old_training_direction_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_FR_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of old training direction(" << old_training_direction_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const type FR_parameter = calculate_FR_parameter(old_gradient, gradient);

    training_direction.device(*thread_pool_device) = -gradient + old_training_direction*FR_parameter;
}


///
////// \brief ConjugateGradient::calculate_gradient_descent_training_direction
////// \param gradient
////// \param training_direction

void ConjugateGradient::calculate_gradient_descent_training_direction(const Tensor<type, 1>& gradient,
                                                                      Tensor<type, 1>& training_direction) const
{
    training_direction.device(*thread_pool_device) = -gradient;
}

/// Returns the conjugate gradient training direction.
/// @param old_gradient Gradient vector in the previous iteration.
/// @param gradient Current gradient vector.
/// @param old_training_direction Training direction in the previous iteration.

void ConjugateGradient::calculate_conjugate_gradient_training_direction(const Tensor<type, 1>& old_gradient,
                                                                        const Tensor<type, 1>& gradient,
                                                                        const Tensor<type, 1>& old_training_direction,
                                                                        Tensor<type, 1>& training_direction) const
{

#ifdef __OPENNN_DEBUG__
    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    ostringstream buffer;

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Loss index pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const Index old_gradient_size = old_gradient.size();

    if(old_gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of old gradient(" << old_gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

    const Index gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of gradient(" << gradient_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

    const Index old_training_direction_size = old_training_direction.size();

    if(old_training_direction_size != parameters_number)
    {
        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "Tensor<type, 1> calculate_training_direction(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of old training direction(" << old_training_direction_size << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(training_direction_method)
    {
    case FR:
        calculate_FR_training_direction(old_gradient, gradient, old_training_direction, training_direction);

        return;
    case PR:
        calculate_PR_training_direction(old_gradient, gradient, old_training_direction, training_direction);
        return;
    }
}


/// Trains a neural network with an associated loss index according to the conjugate gradient algorithm.
/// Training occurs according to the training operators, training parameters and stopping criteria.

OptimizationAlgorithm::Results ConjugateGradient::perform_training()
{
    check();

    // Start training

    if(display) cout << "Training with conjugate gradient...\n";

    Results results;
    results.resize_training_history(maximum_epochs_number+1);

    // Elapsed time

    time_t beginning_time, current_time;
    time(&beginning_time);
    type elapsed_time = 0;

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Index training_samples_number = data_set_pointer->get_training_samples_number();
    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();
    const bool has_selection = data_set_pointer->has_selection();

    Tensor<Index, 1> training_samples_indices = data_set_pointer->get_training_samples_indices();
    Tensor<Index, 1> selection_samples_indices = data_set_pointer->get_selection_samples_indices();
    Tensor<Index, 1> inputs_indices = data_set_pointer->get_input_variables_indices();
    Tensor<Index, 1> target_indices = data_set_pointer->get_target_variables_indices();

    DataSet::Batch training_batch(training_samples_number, data_set_pointer);
    DataSet::Batch selection_batch(selection_samples_number, data_set_pointer);

    training_batch.fill(training_samples_indices, inputs_indices, target_indices);
    selection_batch.fill(selection_samples_indices, inputs_indices, target_indices);

    training_samples_indices.resize(0);
    selection_samples_indices.resize(0);
    inputs_indices.resize(0);
    target_indices.resize(0);

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    type parameters_norm = 0;

    NeuralNetwork::ForwardPropagation training_forward_propagation(training_samples_number, neural_network_pointer);
    NeuralNetwork::ForwardPropagation selection_forward_propagation(selection_samples_number, neural_network_pointer);

    // Loss index

    string information;

    LossIndex::BackPropagation training_back_propagation(training_samples_number, loss_index_pointer);
    LossIndex::BackPropagation selection_back_propagation(selection_samples_number, loss_index_pointer);

    // Optimization algorithm

    type old_training_loss = 0;
    type training_loss_decrease = 0;

    type gradient_norm = 0;

    type selection_error = 0;
    type old_selection_error = 0;

    type learning_rate = 0;

    type minimum_selection_error = numeric_limits<type>::max();
    Tensor<type, 1> minimal_selection_parameters;

    bool stop_training = false;

    Index selection_error_increases = 0;

    GGOptimizationData optimization_data(this);

    if(has_selection)
    {
        minimal_selection_parameters = optimization_data.parameters;
        results.resize_selection_history(maximum_epochs_number + 1);
    }

    // Main loop

    for(Index epoch = 0; epoch < maximum_epochs_number; epoch++)
    {
        optimization_data.epoch = epoch;

        // Neural network

        parameters_norm = l2_norm(optimization_data.parameters);

        neural_network_pointer->forward_propagate(training_batch, training_forward_propagation);

        // Loss index

        loss_index_pointer->back_propagate(training_batch, training_forward_propagation, training_back_propagation);

        gradient_norm = l2_norm(training_back_propagation.gradient);

        if(has_selection)
        {
            neural_network_pointer->forward_propagate(selection_batch, selection_forward_propagation);

            loss_index_pointer->calculate_error(selection_batch, selection_forward_propagation, selection_back_propagation);

            selection_error = selection_back_propagation.error;

            if(epoch == 0)
            {
                minimum_selection_error = selection_error;
            }
            else if(selection_error > old_selection_error)
            {
                selection_error_increases++;
            }
            else if(selection_error <= minimum_selection_error)
            {
                minimum_selection_error = selection_error;
                minimal_selection_parameters = optimization_data.parameters;
            }
        }

        // Optimization algorithm

        update_epoch(training_batch, training_forward_propagation, training_back_propagation, optimization_data);

        // Training history

        if(reserve_training_error_history)
        {
            results.training_error_history(epoch) = training_back_propagation.loss;
        }

        if(has_selection && reserve_selection_error_history)
        {
            results.selection_error_history(epoch) = selection_error;
        }

        // Stopping Criteria       

        if(epoch != 0) training_loss_decrease = training_back_propagation.loss - old_training_loss;
        old_training_loss = training_back_propagation.loss;

        time(&current_time);
        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        if(optimization_data.parameters_increment_norm <= minimum_parameters_increment_norm)
        {
            if(display)
            {
                cout << "Epoch " << epoch+1 << ": Minimum parameters increment norm reached.\n";
                cout << "Parameters increment norm: " << optimization_data.parameters_increment_norm << endl;
            }

            stop_training = true;

            results.stopping_condition = MinimumParametersIncrementNorm;
        }

        else if(epoch != 0 && abs(training_loss_decrease) <= minimum_loss_decrease)
        {
            if(display)
            {
                cout << "Epoch " << epoch+1 << ": Minimum loss decrease (" << minimum_loss_decrease << ") reached.\n"
                     << "Loss decrease: " << training_loss_decrease << endl;
            }

            stop_training = true;

            results.stopping_condition = MinimumLossDecrease;
        }

        else if(training_back_propagation.loss <= training_loss_goal)
        {
            if(display) cout << "Epoch " << epoch+1 << ": Loss goal reached.\n";

            stop_training = true;

            results.stopping_condition = LossGoal;
        }

        else if(gradient_norm <= gradient_norm_goal)
        {
            if(display) cout << "Epoch " << epoch+1 << ": Gradient norm goal reached.\n";

            stop_training = true;

            results.stopping_condition = GradientNormGoal;
        }

        else if(has_selection
                && selection_error_increases >= maximum_selection_error_increases)
        {
            if(display)
            {
                cout << "Epoch " << epoch+1 << ": Maximum selection error increases reached.\n"
                     << "Selection error increases: " << selection_error_increases << endl;
            }

            stop_training = true;

            results.stopping_condition = MaximumSelectionErrorIncreases;
        }

        else if(epoch == maximum_epochs_number)
        {
            if(display)
            {
                cout << "Epoch " << epoch+1 << ": Maximum number of epochs reached.\n";
            }

            stop_training = true;

            results.stopping_condition = MaximumEpochsNumber;
        }

        else if(elapsed_time >= maximum_time)
        {
            if(display)
            {
                cout << "Epoch " << epoch+1 << ": Maximum training time reached.\n";
            }

            stop_training = true;

            results.stopping_condition = MaximumTime;
        }

        if(epoch != 0 && epoch % save_period == 0)
        {
            neural_network_pointer->save(neural_network_file_name);
        }

        if(stop_training)
        {
            if(display)
            {
                cout << "Parameters norm: " << parameters_norm << "\n"
                     << "Training error: " << training_back_propagation.error << "\n"
                     << "Gradient norm: " << gradient_norm << "\n"
                     << "Learning rate: " << learning_rate << "\n"
                     << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

                if(has_selection) cout << "Selection error: " << selection_error << endl;
            }

            results.resize_training_error_history(epoch+1);
            if(has_selection) results.resize_selection_error_history(epoch+1);

            results.final_parameters = optimization_data.parameters;
            results.final_parameters_norm = parameters_norm;

            results.final_training_error = training_back_propagation.error;
            if(has_selection) results.final_selection_error = selection_error;

            results.final_gradient_norm = gradient_norm;

            results.elapsed_time = write_elapsed_time(elapsed_time);

            results.epochs_number = epoch;

            break;
        }

        if(display)
        if(epoch == 0
        ||((epoch+1) % display_period) == 0)
        {
            cout << "Epoch " << epoch+1 << ";\n"
                 << "Parameters norm: " << parameters_norm << "\n"
                 << "Training error: " << training_back_propagation.error << "\n"
                 << "Gradient norm: " << gradient_norm << "\n"
                 << "Learning rate: " << optimization_data.learning_rate << "\n"
                 << "Elapsed time: " << write_elapsed_time(elapsed_time) << "\n";

            if(has_selection) cout << "Selection error: " << selection_error << endl;
        }

        // Set new parameters

        neural_network_pointer->set_parameters(optimization_data.parameters);

        // Update stuff

        if(has_selection) old_selection_error = selection_error;
    }

    if(has_selection && choose_best_selection)
    {
        neural_network_pointer->set_parameters(minimal_selection_parameters);

//        neural_network_pointer->forward_propagate(training_batch, training_forward_propagation);

//        loss_index_pointer->back_propagate(training_batch, training_forward_propagation, training_back_propagation);

//        training_loss = training_back_propagation.loss;

//        selection_error = minimum_selection_error;
    }

    results.final_parameters = optimization_data.parameters;
    results.final_parameters_norm = parameters_norm;

    results.final_training_error = training_back_propagation.error;
    if(has_selection) results.final_selection_error = selection_error;

    results.final_gradient_norm = gradient_norm;

    results.elapsed_time = write_elapsed_time(elapsed_time);

    return results;
}


/// Perform the training.

void ConjugateGradient::perform_training_void()
{
    perform_training();
}

/// Write a string with best algorithm type for the model.

string ConjugateGradient::write_optimization_algorithm_type() const
{
    return "CONJUGATE_GRADIENT";
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> ConjugateGradient::to_string_matrix() const
{
    Tensor<string, 2> labels_values(12, 2);

    // Training direction method

    labels_values(0,0) = "Training direction method";

    labels_values(0,1) = write_training_direction_method();

    // Learning rate method

    labels_values(1,0) = "Learning rate method";

    labels_values(1,1) = learning_rate_algorithm.write_learning_rate_method();

    // Learning rate tolerance

    labels_values(2,0) = "Learning rate tolerance";

    labels_values(2,1) = std::to_string(learning_rate_algorithm.get_learning_rate_tolerance());

    // Minimum parameters increment norm

    labels_values(3,0) = "Minimum parameters increment norm";

    labels_values(3,1) = std::to_string(minimum_parameters_increment_norm);

    // Minimum loss decrease

    labels_values(4,0) = "Minimum loss decrease";

    labels_values(4,1) = std::to_string(minimum_loss_decrease);

    // Loss goal

    labels_values(5,0) = "Loss goal";

    labels_values(5,1) = std::to_string(training_loss_goal);

    // Gradient norm goal

    labels_values(6,0) = "Gradient norm goal";

    labels_values(6,1) = std::to_string(gradient_norm_goal);

    // Maximum selection error increases

    labels_values(7,0) = "Maximum selection error increases";

    labels_values(7,1) = std::to_string(maximum_selection_error_increases);

    // Maximum epochs number

    labels_values(8,0) = "Maximum epochs number";

    labels_values(8,1) = std::to_string(maximum_epochs_number);

    // Maximum time

    labels_values(9,0) = "Maximum time";

    labels_values(9,1) = std::to_string(maximum_time);

    // Reserve training error history

    labels_values(10,0) = "Reserve training error history";

    if(reserve_training_error_history)
    {
        labels_values(10,1) = "true";
    }
    else
    {
        labels_values(10,1) = "false";
    }

    // Reserve selection error history

    labels_values(11,0) = "Reserve selection error history";

    if(reserve_training_error_history)
    {
        labels_values(11,1) = "true";
    }
    else
    {
        labels_values(11,1) = "false";
    }

    return labels_values;
}


/// Serializes the conjugate gradient object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this element.

void ConjugateGradient::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("ConjugateGradient");

    // Training direction method

    {
        file_stream.OpenElement("TrainingDirectionMethod");

        file_stream.PushText(write_training_direction_method().c_str());

        file_stream.CloseElement();
    }

    // Learning rate algorithm

    learning_rate_algorithm.write_XML(file_stream);

   // Return minimum selection error neural network

    {
        file_stream.OpenElement("ReturnMinimumSelectionErrorNN");

        buffer.str("");
        buffer << choose_best_selection;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Minimum parameters increment norm

    {
        file_stream.OpenElement("MinimumParametersIncrementNorm");

        buffer.str("");
        buffer << minimum_parameters_increment_norm;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Minimum loss decrease

    {
        file_stream.OpenElement("MinimumLossDecrease");

        buffer.str("");
        buffer << minimum_loss_decrease;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Loss goal

    {
        file_stream.OpenElement("LossGoal");

        buffer.str("");
        buffer << training_loss_goal;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Gradient norm goal

    {
        file_stream.OpenElement("GradientNormGoal");

        buffer.str("");
        buffer << gradient_norm_goal;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Maximum selection error increases

    {
        file_stream.OpenElement("MaximumSelectionErrorIncreases");

        buffer.str("");
        buffer << maximum_selection_error_increases;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Maximum iterations number

    {
        file_stream.OpenElement("MaximumEpochsNumber");

        buffer.str("");
        buffer << maximum_epochs_number;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Maximum time

    {
        file_stream.OpenElement("MaximumTime");

        buffer.str("");
        buffer << maximum_time;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Reserve training error history

    {
        file_stream.OpenElement("ReserveTrainingErrorHistory");

        buffer.str("");
        buffer << reserve_training_error_history;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Reserve selection error history

    {
        file_stream.OpenElement("ReserveSelectionErrorHistory");

        buffer.str("");
        buffer << reserve_selection_error_history;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Hardware use
    {
        file_stream.OpenElement("HardwareUse");

        buffer.str("");
        buffer << hardware_use;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    file_stream.CloseElement();
}


/// Deserializes the conjugate gradient object from a XML document of the TinyXML library.
/// @param document TinyXML document containing the member data.

void ConjugateGradient::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("ConjugateGradient");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConjugateGradient class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Conjugate gradient element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Training direction method
    {
        const tinyxml2::XMLElement* training_direction_method_element = root_element->FirstChildElement("TrainingDirectionMethod");

        if(training_direction_method_element)
        {
            const string new_training_direction_method = training_direction_method_element->GetText();

            try
            {
                set_training_direction_method(new_training_direction_method);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Learning rate algorithm
    {
        const tinyxml2::XMLElement* learning_rate_algorithm_element = root_element->FirstChildElement("LearningRateAlgorithm");

        if(learning_rate_algorithm_element)
        {
            tinyxml2::XMLDocument learning_rate_algorithm_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = learning_rate_algorithm_element->DeepClone(&learning_rate_algorithm_document);

            learning_rate_algorithm_document.InsertFirstChild(element_clone);

            learning_rate_algorithm.from_XML(learning_rate_algorithm_document);
        }
    }

    // Return minimum selection error neural network

    const tinyxml2::XMLElement* choose_best_selection_element = root_element->FirstChildElement("ReturnMinimumSelectionErrorNN");

    if(choose_best_selection_element)
    {
        string new_choose_best_selection = choose_best_selection_element->GetText();

        try
        {
            set_choose_best_selection(new_choose_best_selection != "0");
        }
        catch(const logic_error& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Minimum parameters increment norm
    {
        const tinyxml2::XMLElement* minimum_parameters_increment_norm_element = root_element->FirstChildElement("MinimumParametersIncrementNorm");

        if(minimum_parameters_increment_norm_element)
        {
            const type new_minimum_parameters_increment_norm = static_cast<type>(atof(minimum_parameters_increment_norm_element->GetText()));

            try
            {
                set_minimum_parameters_increment_norm(new_minimum_parameters_increment_norm);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Minimum loss decrease
    {
        const tinyxml2::XMLElement* minimum_loss_decrease_element = root_element->FirstChildElement("MinimumLossDecrease");

        if(minimum_loss_decrease_element)
        {
            const type new_minimum_loss_decrease = static_cast<type>(atof(minimum_loss_decrease_element->GetText()));

            try
            {
                set_minimum_loss_decrease(new_minimum_loss_decrease);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Loss goal
    {
        const tinyxml2::XMLElement* loss_goal_element = root_element->FirstChildElement("LossGoal");

        if(loss_goal_element)
        {
            const type new_loss_goal = static_cast<type>(atof(loss_goal_element->GetText()));

            try
            {
                set_loss_goal(new_loss_goal);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Gradient norm goal
    {
        const tinyxml2::XMLElement* gradient_norm_goal_element = root_element->FirstChildElement("GradientNormGoal");

        if(gradient_norm_goal_element)
        {
            const type new_gradient_norm_goal = static_cast<type>(atof(gradient_norm_goal_element->GetText()));

            try
            {
                set_gradient_norm_goal(new_gradient_norm_goal);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum selection error increases
    {
        const tinyxml2::XMLElement* maximum_selection_error_increases_element = root_element->FirstChildElement("MaximumSelectionErrorIncreases");

        if(maximum_selection_error_increases_element)
        {
            const Index new_maximum_selection_error_increases = static_cast<Index>(atoi(maximum_selection_error_increases_element->GetText()));

            try
            {
                set_maximum_selection_error_increases(new_maximum_selection_error_increases);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum epochs number
    {
        const tinyxml2::XMLElement* maximum_iterations_number_element = root_element->FirstChildElement("MaximumEpochsNumber");

        if(maximum_iterations_number_element)
        {
            const Index new_maximum_iterations_number = static_cast<Index>(atoi(maximum_iterations_number_element->GetText()));

            try
            {
                set_maximum_epochs_number(new_maximum_iterations_number);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum time
    {
        const tinyxml2::XMLElement* maximum_time_element = root_element->FirstChildElement("MaximumTime");

        if(maximum_time_element)
        {
            const type new_maximum_time = static_cast<type>(atof(maximum_time_element->GetText()));

            try
            {
                set_maximum_time(new_maximum_time);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve training error history
    {
        const tinyxml2::XMLElement* reserve_training_error_history_element = root_element->FirstChildElement("ReserveTrainingErrorHistory");

        if(reserve_training_error_history_element)
        {
            const string new_reserve_training_error_history = reserve_training_error_history_element->GetText();

            try
            {
                set_reserve_training_error_history(new_reserve_training_error_history != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve selection error history
    {
        const tinyxml2::XMLElement* reserve_selection_error_history_element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

        if(reserve_selection_error_history_element)
        {
            const string new_reserve_selection_error_history = reserve_selection_error_history_element->GetText();

            try
            {
                set_reserve_selection_error_history(new_reserve_selection_error_history != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Reserve selection error history
    {
        const tinyxml2::XMLElement* reserve_selection_error_history_element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

        if(reserve_selection_error_history_element)
        {
            const string new_reserve_selection_error_history = reserve_selection_error_history_element->GetText();

            try
            {
                set_reserve_selection_error_history(new_reserve_selection_error_history != "0");
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

      // Display period
      {
         const tinyxml2::XMLElement* display_period_element = root_element->FirstChildElement("DisplayPeriod");

         if(display_period_element)
         {
            const Index new_display_period = static_cast<Index>(atoi(display_period_element->GetText()));

            try
            {
               set_display_period(new_display_period);
            }
            catch(const logic_error& e)
            {
               cerr << e.what() << endl;
            }
         }
      }

         // Save period
         {
             const tinyxml2::XMLElement* element = root_element->FirstChildElement("SavePeriod");

             if(element)
             {
                const Index new_save_period = static_cast<Index>(atoi(element->GetText()));

                try
                {
                   set_save_period(new_save_period);
                }
                catch(const logic_error& e)
                {
                   cerr << e.what() << endl;
                }
             }
         }

         // Neural network file name
         {
             const tinyxml2::XMLElement* element = root_element->FirstChildElement("NeuralNetworkFileName");

             if(element)
             {
                const string new_neural_network_file_name = element->GetText();

                try
                {
                   set_neural_network_file_name(new_neural_network_file_name);
                }
                catch(const logic_error& e)
                {
                   cerr << e.what() << endl;
                }
             }
         }

      // Display
      {
         const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

         if(display_element)
         {
            const string new_display = display_element->GetText();

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

    // Hardware use
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("HardwareUse");

        if(element)
        {
            const string new_hardware_use = element->GetText();

            try
            {
                set_hardware_use(new_hardware_use);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

}


////// \brief ConjugateGradient::update_epoch
////// \param batch
////// \param forward_propagation
////// \param back_propagation
////// \param optimization_data
void ConjugateGradient::update_epoch(
        const DataSet::Batch& batch,
        NeuralNetwork::ForwardPropagation& forward_propagation,
        LossIndex::BackPropagation& back_propagation,
        GGOptimizationData& optimization_data)
{      
    const Index parameters_number = optimization_data.parameters.dimension(0);

    if(optimization_data.epoch == 0 || optimization_data.epoch % parameters_number == 0)
    {
        calculate_gradient_descent_training_direction(
                    back_propagation.gradient,
                    optimization_data.training_direction);
    }
    else
    {       
        calculate_conjugate_gradient_training_direction(
                    optimization_data.old_gradient,
                    back_propagation.gradient,
                    optimization_data.old_training_direction,
                    optimization_data.training_direction);
    }

    optimization_data.training_slope.device(*thread_pool_device)
            = (back_propagation.gradient).contract(optimization_data.training_direction, AT_B);

    if(optimization_data.training_slope(0) >= 0)
    {
        calculate_gradient_descent_training_direction(
                    back_propagation.gradient,
                    optimization_data.training_direction);

        cout << "Epoch " << optimization_data.epoch << ": Gradient descent training direction" << endl;
    }

    // Get initial learning rate

    optimization_data.initial_learning_rate = 0;

    optimization_data.epoch == 0
            ? optimization_data.initial_learning_rate = first_learning_rate
            : optimization_data.initial_learning_rate = optimization_data.old_learning_rate;

    pair<type,type> directional_point = learning_rate_algorithm.calculate_directional_point(
         batch,
         forward_propagation,
         back_propagation,
         optimization_data);

    optimization_data.learning_rate = directional_point.first;

    optimization_data.parameters_increment.device(*thread_pool_device)
            = optimization_data.training_direction*optimization_data.learning_rate;

    optimization_data.parameters_increment_norm = l2_norm(optimization_data.parameters_increment);

    optimization_data.parameters.device(*thread_pool_device) += optimization_data.parameters_increment;

    // Update stuff

    optimization_data.old_gradient = back_propagation.gradient;

    optimization_data.old_training_direction = optimization_data.training_direction;
    optimization_data.old_learning_rate = optimization_data.learning_rate;
}



ConjugateGradient::GGOptimizationData::GGOptimizationData(): OptimizationData()
{
}


ConjugateGradient::GGOptimizationData::GGOptimizationData(ConjugateGradient* new_conjugate_gradient_pointer) : OptimizationData()
{
    set(new_conjugate_gradient_pointer);
}


ConjugateGradient::GGOptimizationData::~GGOptimizationData()
{

}


void ConjugateGradient::GGOptimizationData::set(ConjugateGradient* new_conjugate_gradient_pointer)
{
    conjugate_gradient_pointer = new_conjugate_gradient_pointer;

    LossIndex* loss_index_pointer = conjugate_gradient_pointer->get_loss_index_pointer();

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    parameters.resize(parameters_number);
    parameters = neural_network_pointer->get_parameters();

    potential_parameters.resize(parameters_number);

    parameters_increment.resize(parameters_number);

    old_gradient.resize(parameters_number);

    training_direction.resize(parameters_number);
    old_training_direction.resize(parameters_number);
}


void ConjugateGradient::GGOptimizationData::print() const
{
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
