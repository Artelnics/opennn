//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "quasi_newton_method.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a quasi-Newton method optimization algorithm not associated to any loss index.
/// It also initializes the class members to their default values.

QuasiNewtonMethod::QuasiNewtonMethod()
    : OptimizationAlgorithm()
{
    set_default();
}


/// Loss index constructor.
/// It creates a quasi-Newton method optimization algorithm associated to a loss index.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

QuasiNewtonMethod::QuasiNewtonMethod(LossIndex* new_loss_index_pointer)
    : OptimizationAlgorithm(new_loss_index_pointer)
{
    learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);

    set_default();
}


/// XML constructor.
/// It creates a quasi-Newton method optimization algorithm not associated to any loss index.
/// It also initializes the class members to their default values.

QuasiNewtonMethod::QuasiNewtonMethod(const tinyxml2::XMLDocument& document)
    : OptimizationAlgorithm(document)
{
    set_default();
}


/// Destructor.
/// It does not delete any object.

QuasiNewtonMethod::~QuasiNewtonMethod()
{
}


/// Returns a constant reference to the learning rate algorithm object inside the quasi-Newton method object.

const LearningRateAlgorithm& QuasiNewtonMethod::get_learning_rate_algorithm() const
{
    return learning_rate_algorithm;
}


/// Returns a pointer to the learning rate algorithm object inside the quasi-Newton method object.

LearningRateAlgorithm* QuasiNewtonMethod::get_learning_rate_algorithm_pointer()
{
    return &learning_rate_algorithm;
}


/// Returns the method for approximating the inverse hessian matrix to be used when training.

const QuasiNewtonMethod::InverseHessianApproximationMethod& QuasiNewtonMethod::get_inverse_hessian_approximation_method() const
{
    return inverse_hessian_approximation_method;
}


/// Returns the name of the method for the approximation of the inverse hessian.

string QuasiNewtonMethod::write_inverse_hessian_approximation_method() const
{
    switch(inverse_hessian_approximation_method)
    {
    case DFP:
        return "DFP";

    case BFGS:
        return "BFGS";
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
           << "string write_inverse_hessian_approximation_method() const method.\n"
           << "Unknown inverse hessian approximation method.\n";

    throw logic_error(buffer.str());
}


/// Returns the minimum value for the norm of the parameters vector at wich a warning message is written to the screen.

const type& QuasiNewtonMethod::get_warning_parameters_norm() const
{
    return warning_parameters_norm;
}


/// Returns the minimum value for the norm of the gradient vector at wich a warning message is written to the screen.

const type& QuasiNewtonMethod::get_warning_gradient_norm() const
{
    return warning_gradient_norm;
}


/// Returns the training rate value at wich a warning message is written to the screen during line minimization.

const type& QuasiNewtonMethod::get_warning_learning_rate() const
{
    return warning_learning_rate;
}


/// Returns the value for the norm of the parameters vector at wich an error message is written to the screen
/// and the program exits.

const type& QuasiNewtonMethod::get_error_parameters_norm() const
{
    return error_parameters_norm;
}


/// Returns the value for the norm of the gradient vector at wich an error message is written
/// to the screen and the program exits.

const type& QuasiNewtonMethod::get_error_gradient_norm() const
{
    return error_gradient_norm;
}


/// Returns the training rate value at wich the line minimization algorithm is assumed to fail when
/// bracketing a minimum.

const type& QuasiNewtonMethod::get_error_learning_rate() const
{
    return error_learning_rate;
}


const Index& QuasiNewtonMethod::get_epochs_number() const
{
    return epochs_number;
}


/// Returns the minimum norm of the parameter increment vector used as a stopping criteria when training.

const type& QuasiNewtonMethod::get_minimum_parameters_increment_norm() const
{
    return minimum_parameters_increment_norm;
}


/// Returns the minimum loss improvement during training.

const type& QuasiNewtonMethod::get_minimum_loss_decrease() const
{
    return minimum_loss_decrease;
}


/// Returns the goal value for the loss.
/// This is used as a stopping criterion when training a neural network

const type& QuasiNewtonMethod::get_loss_goal() const
{
    return training_loss_goal;
}


/// Returns the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a neural network

const type& QuasiNewtonMethod::get_gradient_norm_goal() const
{
    return gradient_norm_goal;
}


/// Returns the maximum number of selection error increases during the training process.

const Index& QuasiNewtonMethod::get_maximum_selection_error_increases() const
{
    return maximum_selection_error_increases;
}


/// Returns the maximum number of epochs for training.

const Index& QuasiNewtonMethod::get_maximum_epochs_number() const
{
    return maximum_epochs_number;
}


/// Returns the maximum training time.

const type& QuasiNewtonMethod::get_maximum_time() const
{
    return maximum_time;
}


/// Returns true if the final model will be the neural network with the minimum selection error, false otherwise.

const bool& QuasiNewtonMethod::get_choose_best_selection() const
{
    return choose_best_selection;
}


/// Returns true if the selection error decrease stopping criteria has to be taken in account, false otherwise.

const bool& QuasiNewtonMethod::get_apply_early_stopping() const
{
    return apply_early_stopping;
}


/// Returns true if the error history vector is to be reserved, and false otherwise.

const bool& QuasiNewtonMethod::get_reserve_training_error_history() const
{
    return reserve_training_error_history;
}


/// Returns true if the selection error history vector is to be reserved, and false otherwise.

const bool& QuasiNewtonMethod::get_reserve_selection_error_history() const
{
    return reserve_selection_error_history;
}


/// Sets a pointer to a loss index object to be associated to the quasi-Newton method object.
/// It also sets that loss index to the learning rate algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void QuasiNewtonMethod::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
    loss_index_pointer = new_loss_index_pointer;

    learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);
}


/// Sets a new inverse hessian approximatation method value.
/// @param new_inverse_hessian_approximation_method Inverse hessian approximation method value.

void QuasiNewtonMethod::set_inverse_hessian_approximation_method(
    const QuasiNewtonMethod::InverseHessianApproximationMethod& new_inverse_hessian_approximation_method)
{
    inverse_hessian_approximation_method = new_inverse_hessian_approximation_method;
}


/// Sets a new method for approximating the inverse of the hessian matrix from a string containing the name.
/// Possible values are:
/// <ul>
/// <li> "DFP"
/// <li> "BFGS"
/// </ul>
/// @param new_inverse_hessian_approximation_method_name Name of inverse hessian approximation method.

void QuasiNewtonMethod::set_inverse_hessian_approximation_method(const string& new_inverse_hessian_approximation_method_name)
{
    if(new_inverse_hessian_approximation_method_name == "DFP")
    {
        inverse_hessian_approximation_method = DFP;
    }
    else if(new_inverse_hessian_approximation_method_name == "BFGS")
    {
        inverse_hessian_approximation_method = BFGS;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void set_inverse_hessian_approximation_method(const string&) method.\n"
               << "Unknown inverse hessian approximation method: " << new_inverse_hessian_approximation_method_name << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Makes the training history of all variables to reseved or not in memory.
/// @param new_reserve_all_training_history True if the training history of all variables is to be reserved,
/// false otherwise.

void QuasiNewtonMethod::set_reserve_all_training_history(const bool& new_reserve_all_training_history)
{
    reserve_training_error_history = new_reserve_all_training_history;

    reserve_selection_error_history = new_reserve_all_training_history;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void QuasiNewtonMethod::set_display(const bool& new_display)
{
    display = new_display;
}


void QuasiNewtonMethod::set_default()
{
    inverse_hessian_approximation_method = BFGS;

    learning_rate_algorithm.set_default();

    // TRAINING PARAMETERS

    warning_parameters_norm = 1.0e6;
    warning_gradient_norm = 1.0e3;
    warning_learning_rate = 1.0e3;

    error_parameters_norm = 1.0e6;
    error_gradient_norm = 1.0e6;
    error_learning_rate = 1.0e6;

    // Stopping criteria

    minimum_parameters_increment_norm = 0;

    minimum_loss_decrease = 0;
    training_loss_goal = 0;
    gradient_norm_goal = 0;
    maximum_selection_error_increases = 1000000;

    maximum_epochs_number = 1000;
    maximum_time = 3600.0;

    choose_best_selection = false;
    apply_early_stopping = false;

    // TRAINING HISTORY

    reserve_training_error_history = true;
    reserve_selection_error_history = false;

    // UTILITIES

    display = true;
    display_period = 5;
}


/// Sets a new value for the parameters vector norm at which a warning message is written to the
/// screen.
/// @param new_warning_parameters_norm Warning norm of parameters vector value.

void QuasiNewtonMethod::set_warning_parameters_norm(const type& new_warning_parameters_norm)
{
#ifdef __OPENNN_DEBUG__

    if(new_warning_parameters_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void set_warning_parameters_norm(const type&) method.\n"
               << "Warning parameters norm must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set warning parameters norm

    warning_parameters_norm = new_warning_parameters_norm;
}


/// Sets a new value for the gradient vector norm at which
/// a warning message is written to the screen.
/// @param new_warning_gradient_norm Warning norm of gradient vector value.

void QuasiNewtonMethod::set_warning_gradient_norm(const type& new_warning_gradient_norm)
{
#ifdef __OPENNN_DEBUG__

    if(new_warning_gradient_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void set_warning_gradient_norm(const type&) method.\n"
               << "Warning gradient norm must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set warning gradient norm

    warning_gradient_norm = new_warning_gradient_norm;
}


/// Sets a new training rate value at wich a warning message is written to the screen during line
/// minimization.
/// @param new_warning_learning_rate Warning training rate value.

void QuasiNewtonMethod::set_warning_learning_rate(const type& new_warning_learning_rate)
{
#ifdef __OPENNN_DEBUG__

    if(new_warning_learning_rate < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void set_warning_learning_rate(const type&) method.\n"
               << "Warning training rate must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    warning_learning_rate = new_warning_learning_rate;
}


/// Sets a new value for the parameters vector norm at which an error message is written to the
/// screen and the program exits.
/// @param new_error_parameters_norm Error norm of parameters vector value.

void QuasiNewtonMethod::set_error_parameters_norm(const type& new_error_parameters_norm)
{
#ifdef __OPENNN_DEBUG__

    if(new_error_parameters_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void set_error_parameters_norm(const type&) method.\n"
               << "Error parameters norm must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set error parameters norm

    error_parameters_norm = new_error_parameters_norm;
}


/// Sets a new value for the gradient vector norm at which an error message is written to the screen
/// and the program exits.
/// @param new_error_gradient_norm Error norm of gradient vector value.

void QuasiNewtonMethod::set_error_gradient_norm(const type& new_error_gradient_norm)
{
#ifdef __OPENNN_DEBUG__

    if(new_error_gradient_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void set_error_gradient_norm(const type&) method.\n"
               << "Error gradient norm must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set error gradient norm

    error_gradient_norm = new_error_gradient_norm;
}


/// Sets a new training rate value at wich a the line minimization algorithm is assumed to fail when
/// bracketing a minimum.
/// @param new_error_learning_rate Error training rate value.

void QuasiNewtonMethod::set_error_learning_rate(const type& new_error_learning_rate)
{
#ifdef __OPENNN_DEBUG__

    if(new_error_learning_rate < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void set_error_learning_rate(const type&) method.\n"
               << "Error training rate must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set error training rate

    error_learning_rate = new_error_learning_rate;
}


/// Sets a new value for the minimum parameters increment norm stopping criterion.
/// @param new_minimum_parameters_increment_norm Value of norm of parameters increment norm used to stop training.

void QuasiNewtonMethod::set_minimum_parameters_increment_norm(const type& new_minimum_parameters_increment_norm)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_parameters_increment_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void new_minimum_parameters_increment_norm(const type&) method.\n"
               << "Minimum parameters increment norm must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set error training rate

    minimum_parameters_increment_norm = new_minimum_parameters_increment_norm;
}


/// Sets a new minimum loss improvement during training.
/// @param new_minimum_loss_decrease Minimum improvement in the loss between two epochs.

void QuasiNewtonMethod::set_minimum_loss_decrease(const type& new_minimum_loss_decrease)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_loss_decrease < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
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

void QuasiNewtonMethod::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


/// Sets a new the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a neural network
/// @param new_gradient_norm_goal Goal value for the norm of the error function gradient.

void QuasiNewtonMethod::set_gradient_norm_goal(const type& new_gradient_norm_goal)
{
#ifdef __OPENNN_DEBUG__

    if(new_gradient_norm_goal < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
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

void QuasiNewtonMethod::set_maximum_selection_error_increases(const Index& new_maximum_selection_error_increases)
{
    maximum_selection_error_increases = new_maximum_selection_error_increases;
}


/// Sets a new maximum number of epochs number.
/// @param new_maximum_epochs_number Maximum number of epochs in which the selection evalutation decreases.

void QuasiNewtonMethod::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


/// Sets a new maximum training time.
/// @param new_maximum_time Maximum training time.

void QuasiNewtonMethod::set_maximum_time(const type& new_maximum_time)
{
#ifdef __OPENNN_DEBUG__

    if(new_maximum_time < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set maximum time

    maximum_time = new_maximum_time;
}


/// Makes the minimum selection error neural network of all the epochs to be returned or not.
/// @param new_choose_best_selection True if the final model will be the neural network with the minimum selection error,
/// false otherwise.

void QuasiNewtonMethod::set_choose_best_selection(const bool& new_choose_best_selection)
{
    choose_best_selection = new_choose_best_selection;
}


/// Makes the selection error decrease stopping criteria has to be taken in account or not.
/// @param new_apply_early_stopping True if the selection error decrease stopping criteria has to be taken in account,
/// false otherwise.

void QuasiNewtonMethod::set_apply_early_stopping(const bool& new_apply_early_stopping)
{
    apply_early_stopping = new_apply_early_stopping;
}


/// Makes the error history vector to be reseved or not in memory.
/// @param new_reserve_training_error_history True if the loss history vector is to be reserved, false otherwise.

void QuasiNewtonMethod::set_reserve_training_error_history(const bool& new_reserve_training_error_history)
{
    reserve_training_error_history = new_reserve_training_error_history;
}


/// Makes the selection error history to be reserved or not in memory.
/// This is a vector.
/// @param new_reserve_selection_error_history True if the selection error history is to be reserved, false otherwise.

void QuasiNewtonMethod::set_reserve_selection_error_history(const bool& new_reserve_selection_error_history)
{
    reserve_selection_error_history = new_reserve_selection_error_history;
}


/// Sets a new number of epochs between the training showing progress.
/// @param new_display_period
/// Number of epochs between the training showing progress.

void QuasiNewtonMethod::set_display_period(const Index& new_display_period)
{
#ifdef __OPENNN_DEBUG__

    if(new_display_period == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void set_display_period(const Index&) method.\n"
               << "Display period must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    display_period = new_display_period;
}


/// Calculates an approximation of the inverse hessian, accoring to the method used.
/// @param old_parameters Another point of the error function.
/// @param parameters Current point of the error function
/// @param old_gradient Gradient at the other point.
/// @param gradient Gradient at the current point.
/// @param old_inverse_hessian Inverse hessian at the other point of the error function.

Tensor<type, 2> QuasiNewtonMethod::calculate_inverse_hessian_approximation(
    const Tensor<type, 1>& old_parameters, const Tensor<type, 1>& parameters,
    const Tensor<type, 1>& old_gradient, const Tensor<type, 1>& gradient,
    const Tensor<type, 2>& old_inverse_hessian) const
{
    switch(inverse_hessian_approximation_method)
    {
    case DFP:
        return calculate_DFP_inverse_hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_hessian);

    case BFGS:
        return calculate_BFGS_inverse_hessian(old_parameters, parameters, old_gradient, gradient, old_inverse_hessian);
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
           << "Tensor<type, 1> calculate_inverse_hessian_approximation(const Tensor<type, 1>&, "
           "const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>&) method.\n"
           << "Unknown inverse hessian approximation method.\n";

    throw logic_error(buffer.str());
}


const Tensor<type, 2> QuasiNewtonMethod::kronecker_product(Tensor<type, 1> & left_matrix, Tensor<type, 1> & right_matrix) const
{
    // Transform Tensors into Dense matrix

    auto ml = Eigen::Map<Eigen::Matrix<type,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor >>
            (left_matrix.data(),left_matrix.dimension(0), 1);
    auto mr = Eigen::Map<Eigen::Matrix<type,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
            (right_matrix.data(),right_matrix.dimension(0), 1);

    // Kronecker Product

    auto product = kroneckerProduct(ml,mr).eval();

    // Matrix into a Tensor

    TensorMap< Tensor<type, 2> > direct_matrix(product.data(), left_matrix.size(), left_matrix.size());

    return direct_matrix;
}


/// This method calculates the kronecker product between two matrix.
/// Its return a direct matrix.
/// @param left_matrix Matrix to be porudct.
/// @param right_matrix Matrix to be product.

const Tensor<type, 2> QuasiNewtonMethod::kronecker_product(Tensor<type, 2>& left_matrix, Tensor<type, 2>& right_matrix) const
{
    // Transform Tensors into Dense matrix

    auto ml = Eigen::Map<Eigen::Matrix<type,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor >>
            (left_matrix.data(),left_matrix.dimension(0),left_matrix.dimension(1));

    auto mr = Eigen::Map<Eigen::Matrix<type,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
            (right_matrix.data(),right_matrix.dimension(0),right_matrix.dimension(1));

    // Kronecker Product

    auto product = kroneckerProduct(ml,mr).eval();

    // Matrix into a Tensor

    TensorMap< Tensor<type, 2> > direct_matrix(product.data(), product.rows(), product.cols());

    return direct_matrix;

}


/// Returns an approximation of the inverse hessian matrix according to the Davidon-Fletcher-Powel
/// (DFP) algorithm.
/// @param old_parameters A previous set of parameters.
/// @param old_gradient The gradient of the error function for that previous set of parameters.
/// @param old_inverse_hessian The hessian of the error function for that previous set of parameters.
/// @param parameters Actual set of parameters.
/// @param gradient The gradient of the error function for the actual set of parameters.

Tensor<type, 2> QuasiNewtonMethod::calculate_DFP_inverse_hessian(const Tensor<type, 1>& old_parameters,
        const Tensor<type, 1>& parameters,
        const Tensor<type, 1>& old_gradient,
        const Tensor<type, 1>& gradient,
        const Tensor<type, 2>& old_inverse_hessian) const
{
    ostringstream buffer;

#ifdef __OPENNN_DEBUG__

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    const Index old_parameters_size = old_parameters.size();
    const Index parameters_size = parameters.size();

    if(old_parameters_size != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_DFP_inverse_hessian() method.\n"
               << "Size of old parameters vector must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }
    else if(parameters_size != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_DFP_inverse_hessian() method.\n"
               << "Size of parameters vector must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }

    const Index old_gradient_size = old_gradient.size();
    const Index gradient_size = gradient.size();

    if(old_gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_DFP_inverse_hessian() method.\n"
               << "Size of old gradient vector must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }
    else if(gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_DFP_inverse_hessian() method.\n"
               << "Size of gradient vector must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }

    const Index rows_number = old_inverse_hessian.dimension(0);
    const Index columns_number = old_inverse_hessian.dimension(1);

    if(rows_number != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_DFP_inverse_hessian() method.\n"
               << "Number of rows in old inverse hessian must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }
    else if(columns_number != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_DFP_inverse_hessian() method.\n"
               << "Number of columns in old inverse hessian must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Parameters difference Vector

    /*
       if(parameters_difference.abs() < numeric_limits<type>::min())
       {
          buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
                 << "Tensor<type, 2> calculate_DFP_inverse_hessian() method.\n"
                 << "Parameters difference vector is zero.\n";

          throw logic_error(buffer.str());
       }

       // Gradient difference Vector



       if(gradient_difference.abs() < 1.0e-50)
       {
          buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
                 << "Tensor<type, 2> calculate_DFP_inverse_hessian() method.\n"
                 << "Gradient difference vector is zero.\n";

          throw logic_error(buffer.str());
       }

       if(absolute_value(old_inverse_hessian) < 1.0e-50)
       {
          buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
                 << "Tensor<type, 2> calculate_DFP_inverse_hessian() method.\n"
                 << "Old inverse hessian matrix is zero.\n";

          throw logic_error(buffer.str());
       }



//   const type parameters_dot_gradient = dot(parameters_difference, gradient_difference);

//   dot(dot(gradient_difference, old_inverse_hessian), gradient_difference)

    if(abs(parameters_dot_gradient(0)) < static_cast<type>(1.0e-50))
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_DFP_inverse_hessian() method.\n"
               << "Denominator of first term is zero.\n";

        throw logic_error(buffer.str());
    }
    else if(abs(gradient_dot_hesian_dot_gradient(0)) < static_cast<type>(1.0e-50))
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_DFP_inverse_hessian() method.\n"
               << "Denominator of second term is zero.\n";

        throw logic_error(buffer.str());
    }
*/

    // Dots

    Tensor<type, 1> parameters_difference = parameters - old_parameters;

    const Tensor<type, 1> gradient_difference = gradient - old_gradient;

    const Tensor<type, 0> parameters_dot_gradient = parameters_difference.contract(gradient_difference, AT_B); // Ok

    Tensor<type, 1> hessian_dot_gradient_difference
        = old_inverse_hessian.contract(gradient_difference, A_B); // Ok

    Tensor<type, 0> gradient_dot_hessian_dot_gradient
        = gradient_difference.contract(hessian_dot_gradient_difference, AT_B); // Ok , auto?

//    const Tensor<type, 1> gradient_dot_hessian = gradient_difference.contract(old_inverse_hessian, product_vector_matrix); // Only for exceptions and repeated above

//    const Tensor<type, 0> gradient_dot_hesian_dot_gradient
//        = gradient_dot_hessian.contract(gradient_difference,AT_B); // Only for exceptions and repeated above

    // Calculates Approximation

    Tensor<type, 2> inverse_hessian_approximation = old_inverse_hessian; // TensorMap?

    inverse_hessian_approximation += kronecker_product(parameters_difference, parameters_difference)/parameters_dot_gradient(0); // Ok

    inverse_hessian_approximation -= kronecker_product(hessian_dot_gradient_difference, hessian_dot_gradient_difference)/ gradient_dot_hessian_dot_gradient(0); // Ok

    return inverse_hessian_approximation;

    /*
       inverse_hessian_approximation += direct(parameters_difference, parameters_difference)/parameters_dot_gradient;

       inverse_hessian_approximation -= direct(hessian_dot_gradient_difference, hessian_dot_gradient_difference)
                /(gradient_dot_gradient(0)); //dot(gradient_difference, hessian_dot_gradient_difference);
    */
}


/// Returns an approximation of the inverse hessian matrix according to the
/// Broyden-Fletcher-Goldfarb-Shanno(BGFS) algorithm.
/// @param old_parameters A previous set of parameters.
/// @param old_gradient The gradient of the error function for that previous set of parameters.
/// @param old_inverse_hessian The hessian of the error function for that previous set of parameters.
/// @param parameters Actual set of parameters.
/// @param gradient The gradient of the error function for the actual set of parameters.

Tensor<type, 2> QuasiNewtonMethod::calculate_BFGS_inverse_hessian(
    const Tensor<type, 1>& old_parameters, const Tensor<type, 1>& parameters,
    const Tensor<type, 1>& old_gradient, const Tensor<type, 1>& gradient, const Tensor<type, 2>& old_inverse_hessian) const
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    const Index old_parameters_size = old_parameters.size();
    const Index parameters_size = parameters.size();

    if(old_parameters_size != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_BFGS_inverse_hessian() method.\n"
               << "Size of old parameters vector must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }
    else if(parameters_size != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_BFGS_inverse_hessian() method.\n"
               << "Size of parameters vector must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }

    const Index old_gradient_size = old_gradient.size();

    if(old_gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_BFGS_inverse_hessian() method."
               << endl
               << "Size of old gradient vector must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }

    const Index gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_BFGS_inverse_hessian() method." << endl
               << "Size of gradient vector must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }

    const Index rows_number = old_inverse_hessian.dimension(0);
    const Index columns_number = old_inverse_hessian.dimension(1);

    if(rows_number != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_BFGS_inverse_hessian() method.\n"
               << "Number of rows in old inverse hessian must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }

    if(columns_number != parameters_number)
    {
        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 2> calculate_BFGS_inverse_hessian() method.\n"
               << "Number of columns in old inverse hessian must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Parameters difference Vector


//   if(absolute_value(parameters_difference) < 1.0e-50)
//   {
//       ostringstream buffer;

//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Tensor<type, 2> calculate_BFGS_inverse_hessian() method.\n"
//             << "Parameters difference vector is zero.\n";

//      throw logic_error(buffer.str());
//   }

    // Gradient difference Vector

//   if(absolute_value(gradient_difference) < numeric_limits<type>::min())
//   {
//       ostringstream buffer;

//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Tensor<type, 2> calculate_BFGS_inverse_hessian() method.\n"
//             << "Gradient difference vector is zero.\n";

//      throw logic_error(buffer.str());
//   }

//   if(absolute_value(old_inverse_hessian) < 1.0e-50)
//   {
//       ostringstream buffer;

//      buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
//             << "Tensor<type, 2> calculate_BFGS_inverse_hessian() method.\n"
//             << "Old inverse hessian matrix is zero.\n";

//	  throw logic_error(buffer.str());
//   }

    // BGFS Vector

    Tensor<type, 1> parameters_difference = parameters - old_parameters;

    const Tensor<type, 1> gradient_difference = gradient - old_gradient;

    const Tensor<type, 0> parameters_dot_gradient = parameters_difference.contract(gradient_difference, AT_B);

    Tensor<type, 1> hessian_dot_gradient = old_inverse_hessian.contract(gradient_difference, A_B);

    const Tensor<type, 0> gradient_dot_hessian_dot_gradient = gradient_difference.contract(hessian_dot_gradient, AT_B);

    Tensor<type, 1> BFGS = parameters_difference/parameters_dot_gradient(0)
                               - hessian_dot_gradient/gradient_dot_hessian_dot_gradient(0);

    // Calculates Approximation

    Tensor<type, 2> inverse_hessian_approximation = old_inverse_hessian;

    inverse_hessian_approximation += kronecker_product(parameters_difference, parameters_difference)/parameters_dot_gradient(0); // Ok

    inverse_hessian_approximation -= kronecker_product(hessian_dot_gradient, hessian_dot_gradient)/gradient_dot_hessian_dot_gradient(0); // Ok

    inverse_hessian_approximation += kronecker_product(BFGS, BFGS)*(gradient_dot_hessian_dot_gradient(0)); // Ok

    return inverse_hessian_approximation;
}


/// Trains a neural network with an associated loss index according to the quasi-Newton method.
/// Training occurs according to the training operators, training parameters and stopping criteria.

OptimizationAlgorithm::Results QuasiNewtonMethod::perform_training()
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Start training

    if(display) cout << "Training with quasi-Newton method...\n";

    Results results;

    results.resize_training_history(maximum_epochs_number);

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Index training_instances_number = data_set_pointer->get_training_instances_number();
    const Index selection_instances_number = data_set_pointer->get_selection_instances_number();

    Tensor<Index, 1> training_instances_indices = data_set_pointer->get_training_instances_indices();
    Tensor<Index, 1> selection_instances_indices = data_set_pointer->get_selection_instances_indices();
    const Tensor<Index, 1> inputs_indices = data_set_pointer->get_input_variables_indices();
    const Tensor<Index, 1> target_indices = data_set_pointer->get_target_variables_indices();

    const bool has_selection = data_set_pointer->has_selection();

    DataSet::Batch training_batch(training_instances_number, data_set_pointer);
    DataSet::Batch selection_batch(selection_instances_number, data_set_pointer);

    training_batch.fill(training_instances_indices, inputs_indices, target_indices);
    selection_batch.fill(selection_instances_indices, inputs_indices, target_indices);

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    type parameters_norm = 0;

//    type parameters_increment_norm = 0;

    NeuralNetwork::ForwardPropagation training_forward_propagation(training_instances_number, neural_network_pointer);
    NeuralNetwork::ForwardPropagation selection_forward_propagation(selection_instances_number, neural_network_pointer);

    // Loss index

    type training_error = 0;

    type gradient_norm = 0;

    type selection_error = numeric_limits<type>::max();
    type old_selection_error = numeric_limits<type>::max();

    LossIndex::BackPropagation training_back_propagation(training_instances_number, loss_index_pointer);
    LossIndex::BackPropagation selection_back_propagation(selection_instances_number, loss_index_pointer);

    // Optimization algorithm

    Tensor<type, 0> training_slope;

    Tensor<type, 1> minimal_selection_parameters;

    type minimum_selection_error = numeric_limits<type>::max();

    bool stop_training = false;

    Index selection_failures = 0;

    time_t beginning_time, current_time;
    time(&beginning_time);
    type elapsed_time;

    OptimizationData optimization_data(this);

    // Main loop

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        optimization_data.epoch = epoch;

        // Neural network

        parameters_norm = l2_norm(optimization_data.parameters);

        if(display && parameters_norm >= warning_parameters_norm)
        {
            cout << "OpenNN Warning: Parameters norm is " << parameters_norm << ".\n";
        }

        neural_network_pointer->forward_propagate(training_batch, training_forward_propagation);

        // Loss index

        loss_index_pointer->back_propagate(training_batch, training_forward_propagation, training_back_propagation);

        loss_index_pointer->calculate_error(training_batch, training_forward_propagation, training_back_propagation);

        training_error = training_back_propagation.loss;

        gradient_norm = l2_norm(training_back_propagation.gradient);

        if(display && gradient_norm >= warning_gradient_norm)
        {
            cout << "OpenNN Warning: Gradient norm is " << gradient_norm << ".\n";
        }

        // Optimization data

        update_epoch(training_batch, training_forward_propagation, training_back_propagation, optimization_data);

        neural_network_pointer->set_parameters(optimization_data.parameters);

        // Selection error

        if(has_selection)
        {
            selection_error = 0;

            neural_network_pointer->forward_propagate(selection_batch, selection_forward_propagation);

            // Loss Index

            loss_index_pointer->calculate_error(selection_batch, selection_forward_propagation, selection_back_propagation);

            selection_error = selection_back_propagation.loss;

            if(selection_error > old_selection_error)
            {
                selection_failures++;
            }
            else if(selection_error < minimum_selection_error)
            {
                minimum_selection_error = selection_error;

                minimal_selection_parameters = optimization_data.parameters;
            }

            if(reserve_selection_error_history) results.selection_error_history(epoch) = selection_error;
        }

        // Training history

        if(reserve_training_error_history) results.training_error_history(epoch) = training_error;

        // Stopping Criteria

        time(&current_time);
        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        if(optimization_data.parameters_increment_norm <= minimum_parameters_increment_norm)
        {
            if(display)
            {
               cout << "Epoch " << epoch << ": Minimum parameters increment norm reached.\n"
                    << "Parameters increment norm: " << optimization_data.parameters_increment_norm << endl;
            }

            stop_training = true;

            results.stopping_condition = MinimumParametersIncrementNorm;
        }
        else if(epoch != 0 &&
                (training_back_propagation.loss - optimization_data.old_training_loss) >= minimum_loss_decrease)
        {
            if(display)
            {
               cout << "Epoch " << epoch << ": Minimum loss decrease (" << minimum_loss_decrease << ") reached.\n"
                    << "Loss decrease: " << training_back_propagation.loss - optimization_data.old_training_loss <<  endl;
            }

            stop_training = true;

            results.stopping_condition = MinimumLossDecrease;
        }
        else if(training_back_propagation.loss <= training_loss_goal)
        {
            if(display)
            {
                cout << "Epoch " << epoch << ": Loss goal reached.\n";
            }

            stop_training = true;

            results.stopping_condition = LossGoal;
        }
        else if(gradient_norm <= gradient_norm_goal)
        {
            if(display)
            {
                cout << "Iteration " << epoch << ": Gradient norm goal reached.\n";
            }

            stop_training = true;

            results.stopping_condition = GradientNormGoal;
        }
        else if(apply_early_stopping && selection_failures >= maximum_selection_error_increases)
        {
            if(display)
            {
                cout << "Epoch " << epoch << ": Maximum selection error increases reached.\n"
                     << "Selection loss increases: "<< selection_failures << endl;
            }

            stop_training = true;

            results.stopping_condition = MaximumSelectionErrorIncreases;
        }
        else if(epoch == maximum_epochs_number)
        {
            if(display)
            {
                cout << "Epoch " << epoch << ": Maximum number of epochs reached.\n";
            }

            stop_training = true;

            results.stopping_condition = MaximumEpochsNumber;
        }
        else if(elapsed_time >= maximum_time)
        {
            if(display)
            {
                cout << "Epoch " << epoch << ": Maximum training time reached.\n";
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
            results.final_parameters = optimization_data.parameters;
            results.final_parameters_norm = parameters_norm;
            results.final_training_error = training_error;
            results.final_selection_error = selection_error;

            results.final_gradient_norm = gradient_norm;

            results.elapsed_time = elapsed_time;

            results.epochs_number = epoch;

            results.resize_error_history(epoch+1);

            if(display)
            {
                cout << "Parameters norm: " << parameters_norm << "\n"
                     << "Training error: " << training_error <<  "\n"
                     << "Gradient norm: " << gradient_norm <<  "\n"
                     << loss_index_pointer->write_information()
                     << "Training rate: " << optimization_data.learning_rate <<  "\n"
                     << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

                if(has_selection)
                {
                    cout << "Selection error: " << selection_error << endl;
                }
            }

            break;
        }
        else if(display && epoch % display_period == 0)
        {
            cout << "Epoch " << epoch << ";\n"
                 << "Parameters norm: " << parameters_norm << "\n"
                 << "Training error: " << training_error << "\n"
                 << "Gradient norm: " << gradient_norm << "\n"
                 << loss_index_pointer->write_information()
                 << "Training rate: " << optimization_data.learning_rate << "\n"
                 << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

            if(has_selection)
            {
                cout << "Selection error: " << selection_error << endl;
            }
        }

        optimization_data.old_training_loss = training_back_propagation.loss;

        if(stop_training) break;
    }

    if(choose_best_selection)
    {
        //optimization_data.parameters = minimal_selection_parameters;
        //parameters_norm = l2_norm(parameters);

        neural_network_pointer->set_parameters(minimal_selection_parameters);

        //neural_network_pointer->forward_propagate(training_batch, training_forward_propagation);

        //loss_index_pointer->back_propagate(training_batch, training_forward_propagation, training_back_propagation);

        //training_loss = training_back_propagation.loss;

        //selection_error = minimum_selection_error;
    }

    return results;
}


void QuasiNewtonMethod::perform_training_void()
{
    perform_training();
}


string QuasiNewtonMethod::write_optimization_algorithm_type() const
{
    return "QUASI_NEWTON_METHOD";
}


/// Returns a XML-type string representation of this quasi-Newton method object.
/// It contains the training methods and parameters chosen, as well as
/// the stopping criteria and other user stuff concerning the quasi-Newton method object.

tinyxml2::XMLDocument* QuasiNewtonMethod::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Quasi-Newton method

    tinyxml2::XMLElement* root_element = document->NewElement("QuasiNewtonMethod");

    document->InsertFirstChild(root_element);

    tinyxml2::XMLElement* element = nullptr;
    tinyxml2::XMLText* text = nullptr;

    // Inverse hessian approximation method
    {
        element = document->NewElement("InverseHessianApproximationMethod");
        root_element->LinkEndChild(element);

        text = document->NewText(write_inverse_hessian_approximation_method().c_str());
        element->LinkEndChild(text);
    }


    // Training rate algorithm
    {
        const tinyxml2::XMLDocument* learning_rate_algorithm_document = learning_rate_algorithm.to_XML();

        const tinyxml2::XMLElement* learning_rate_algorithm_element = learning_rate_algorithm_document->FirstChildElement("LearningRateAlgorithm");

        tinyxml2::XMLNode* node = learning_rate_algorithm_element->DeepClone(document);

        root_element->InsertEndChild(node);

        delete learning_rate_algorithm_document;
    }

    // Return minimum selection error neural network

    element = document->NewElement("ReturnMinimumSelectionErrorNN");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << choose_best_selection;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Apply early stopping

    element = document->NewElement("ApplyEarlyStopping");
    root_element->LinkEndChild(element);

    buffer.str("");
    buffer << apply_early_stopping;

    text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);

    // Warning parameters norm
//   {
//   element = document->NewElement("WarningParametersNorm");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << warning_parameters_norm;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

    // Warning gradient norm
//   {
//   element = document->NewElement("WarningGradientNorm");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << warning_gradient_norm;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

    // Warning training rate
//   {
//   element = document->NewElement("WarningLearningRate");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << warning_learning_rate;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

    // Error parameters norm
//   {
//   element = document->NewElement("ErrorParametersNorm");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << error_parameters_norm;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

    // Error gradient norm
//   {
//   element = document->NewElement("ErrorGradientNorm");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << error_gradient_norm;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

    // Error training rate
//   {
//   element = document->NewElement("ErrorLearningRate");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << error_learning_rate;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

    // Minimum parameters increment norm
    {
        element = document->NewElement("MinimumParametersIncrementNorm");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << minimum_parameters_increment_norm;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Minimum loss decrease
    {
        element = document->NewElement("MinimumLossDecrease");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << minimum_loss_decrease;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Loss goal
    {
        element = document->NewElement("LossGoal");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << training_loss_goal;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Gradient norm goal
    {
        element = document->NewElement("GradientNormGoal");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << gradient_norm_goal;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Maximum selection error increases
    {
        element = document->NewElement("MaximumSelectionErrorIncreases");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << maximum_selection_error_increases;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Maximum iterations number
    {
        element = document->NewElement("MaximumEpochsNumber");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << maximum_epochs_number;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Maximum time
    {
        element = document->NewElement("MaximumTime");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << maximum_time;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Reserve training error history
    {
        element = document->NewElement("ReserveTrainingErrorHistory");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << reserve_training_error_history;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Reserve selection error history
    {
        element = document->NewElement("ReserveSelectionErrorHistory");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << reserve_selection_error_history;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }


    // Reserve gradient history
//   {
//   element = document->NewElement("ReserveGradientHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_gradient_history;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }


    // Reserve selection error history
//   {
//   element = document->NewElement("ReserveSelectionErrorHistory");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_selection_error_history;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

    // Display period
//   {
//   element = document->NewElement("DisplayPeriod");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << display_period;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

    // Save period
//   {
//       element = document->NewElement("SavePeriod");
//       root_element->LinkEndChild(element);

//       buffer.str("");
//       buffer << save_period;

//       text = document->NewText(buffer.str().c_str());
//       element->LinkEndChild(text);
//   }

    // Neural network file name
//   {
//       element = document->NewElement("NeuralNetworkFileName");
//       root_element->LinkEndChild(element);

//       text = document->NewText(neural_network_file_name.c_str());
//       element->LinkEndChild(text);
//   }

    // Display
//   {
//   element = document->NewElement("Display");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << display;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

    return document;
}


/// Serializes the quasi Newton method object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void QuasiNewtonMethod::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

//    file_stream.OpenElement("QuasiNewtonMethod");

    // Inverse hessian approximation method

    file_stream.OpenElement("InverseHessianApproximationMethod");

    file_stream.PushText(write_inverse_hessian_approximation_method().c_str());

    file_stream.CloseElement();

    // Training rate algorithm

    learning_rate_algorithm.write_XML(file_stream);

    // Return minimum selection error neural network

    file_stream.OpenElement("ReturnMinimumSelectionErrorNN");

    buffer.str("");
    buffer << choose_best_selection;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Apply early stopping

    file_stream.OpenElement("ApplyEarlyStopping");

    buffer.str("");
    buffer << apply_early_stopping;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum parameters increment norm

    file_stream.OpenElement("MinimumParametersIncrementNorm");

    buffer.str("");
    buffer << minimum_parameters_increment_norm;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum loss decrease

    file_stream.OpenElement("MinimumLossDecrease");

    buffer.str("");
    buffer << minimum_loss_decrease;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Loss goal

    file_stream.OpenElement("LossGoal");

    buffer.str("");
    buffer << training_loss_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Gradient norm goal

    file_stream.OpenElement("GradientNormGoal");

    buffer.str("");
    buffer << gradient_norm_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum selection error increases

    file_stream.OpenElement("MaximumSelectionErrorIncreases");

    buffer.str("");
    buffer << maximum_selection_error_increases;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum iterations number

    file_stream.OpenElement("MaximumEpochsNumber");

    buffer.str("");
    buffer << maximum_epochs_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve training error history

    file_stream.OpenElement("ReserveTrainingErrorHistory");

    buffer.str("");
    buffer << reserve_training_error_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection error history

    file_stream.OpenElement("ReserveSelectionErrorHistory");

    buffer.str("");
    buffer << reserve_selection_error_history;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();
}


string QuasiNewtonMethod::object_to_string() const
{
    ostringstream buffer;

    buffer << "Quasi-Newton method\n";

    return buffer.str();
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> QuasiNewtonMethod::to_string_matrix() const
{
    ostringstream buffer;

    Tensor<string, 1> labels;
    Tensor<string, 1> values;

    // Inverse hessian approximation method
    /*
        labels.push_back("Inverse hessian approximation method");

        const string inverse_hessian_approximation_method_string = write_inverse_hessian_approximation_method();

        values.push_back(inverse_hessian_approximation_method_string);

       // Training rate method

       labels.push_back("Training rate method");

       const string learning_rate_method = learning_rate_algorithm.write_learning_rate_method();

       values.push_back(learning_rate_method);

       // Loss tolerance

       labels.push_back("Loss tolerance");

       buffer.str("");
       buffer << learning_rate_algorithm.get_loss_tolerance();

       values.push_back(buffer.str());

       // Minimum parameters increment norm

       labels.push_back("Minimum parameters increment norm");

       buffer.str("");
       buffer << minimum_parameters_increment_norm;

       values.push_back(buffer.str());

       // Minimum loss decrease

       labels.push_back("Minimum loss decrease");

       buffer.str("");
       buffer << minimum_loss_decrease;

       values.push_back(buffer.str());

       // Loss goal

       labels.push_back("Loss goal");

       buffer.str("");
       buffer << training_loss_goal;

       values.push_back(buffer.str());

       // Gradient norm goal

       labels.push_back("Gradient norm goal");

       buffer.str("");
       buffer << gradient_norm_goal;

       values.push_back(buffer.str());

       // Maximum selection error increases

       labels.push_back("Maximum selection error increases");

       buffer.str("");
       buffer << maximum_selection_error_increases;

       values.push_back(buffer.str());

       // Maximum iterations number

       labels.push_back("Maximum iterations number");

       buffer.str("");
       buffer << maximum_epochs_number;

       values.push_back(buffer.str());

       // Maximum time

       labels.push_back("Maximum time");

       buffer.str("");
       buffer << maximum_time;

       values.push_back(buffer.str());

       // Reserve training error history

       labels.push_back("Reserve training error history");

       buffer.str("");

       if(reserve_training_error_history)
       {
           buffer << "true";
       }
       else
       {
           buffer << "false";
       }

       values.push_back(buffer.str());

       // Reserve selection error history

       labels.push_back("Reserve selection error history");

       buffer.str("");

       if(reserve_selection_error_history)
       {
           buffer << "true";
       }
       else
       {
           buffer << "false";
       }

       values.push_back(buffer.str());

       const Index rows_number = labels.size();
       const Index columns_number = 2;

       Tensor<string, 2> string_matrix(rows_number, columns_number);

       string_matrix.set_column(0, labels, "name");
       string_matrix.set_column(1, values, "value");

        return string_matrix;
    */
    return Tensor<string, 2>();
}


void QuasiNewtonMethod::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("QuasiNewtonMethod");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Quasi-Newton method element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Inverse hessian approximation method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("InverseHessianApproximationMethod");

        if(element)
        {
            const string new_inverse_hessian_approximation_method = element->GetText();

            try
            {
                set_inverse_hessian_approximation_method(new_inverse_hessian_approximation_method);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Learning rate algorithm
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("LearningRateAlgorithm");

        if(element)
        {
            tinyxml2::XMLDocument learning_rate_algorithm_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&learning_rate_algorithm_document);

            learning_rate_algorithm_document.InsertFirstChild(element_clone);

            learning_rate_algorithm.from_XML(learning_rate_algorithm_document);
        }
    }
    /*
       // Warning parameters norm
       {
           const tinyxml2::XMLElement* element = root_element->FirstChildElement("WarningParametersNorm");

           if(element)
           {
              const type new_warning_parameters_norm = static_cast<type>(atof(element->GetText()));

              try
              {
                 set_warning_parameters_norm(new_warning_parameters_norm);
              }
              catch(const logic_error& e)
              {
                 cerr << e.what() << endl;
              }
           }
       }

       // Warning gradient norm
       {
           const tinyxml2::XMLElement* element = root_element->FirstChildElement("WarningGradientNorm");

           if(element)
           {
              const type new_warning_gradient_norm = static_cast<type>(atof(element->GetText()));

              try
              {
                 set_warning_gradient_norm(new_warning_gradient_norm);
              }
              catch(const logic_error& e)
              {
                 cerr << e.what() << endl;
              }
           }
       }

       // Warning training rate
       {
           const tinyxml2::XMLElement* element = root_element->FirstChildElement("WarningLearningRate");

           if(element)
           {
              const type new_warning_learning_rate = static_cast<type>(atof(element->GetText()));

              try
              {
                 set_warning_learning_rate(new_warning_learning_rate);
              }
              catch(const logic_error& e)
              {
                 cerr << e.what() << endl;
              }
           }
       }

       // Error parameters norm
       {
           const tinyxml2::XMLElement* element = root_element->FirstChildElement("ErrorParametersNorm");

           if(element)
           {
              const type new_error_parameters_norm = static_cast<type>(atof(element->GetText()));

              try
              {
                 set_error_parameters_norm(new_error_parameters_norm);
              }
              catch(const logic_error& e)
              {
                 cerr << e.what() << endl;
              }
           }
       }

       // Error gradient norm
       {
           const tinyxml2::XMLElement* element = root_element->FirstChildElement("ErrorGradientNorm");

           if(element)
           {
              const type new_error_gradient_norm = static_cast<type>(atof(element->GetText()));

              try
              {
                 set_error_gradient_norm(new_error_gradient_norm);
              }
              catch(const logic_error& e)
              {
                 cerr << e.what() << endl;
              }
           }
       }

       // Error training rate
       {
           const tinyxml2::XMLElement* element = root_element->FirstChildElement("ErrorLearningRate");

           if(element)
           {
              const type new_error_learning_rate = static_cast<type>(atof(element->GetText()));

              try
              {
                 set_error_learning_rate(new_error_learning_rate);
              }
              catch(const logic_error& e)
              {
                 cerr << e.what() << endl;
              }
           }
       }
    */
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

    // Apply early stopping

    const tinyxml2::XMLElement* apply_early_stopping_element = root_element->FirstChildElement("ApplyEarlyStopping");

    if(apply_early_stopping_element)
    {
        string new_apply_early_stopping = apply_early_stopping_element->GetText();

        try
        {
            set_apply_early_stopping(new_apply_early_stopping != "0");
        }
        catch(const logic_error& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Minimum parameters increment norm
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumParametersIncrementNorm");

        if(element)
        {
            const type new_minimum_parameters_increment_norm = static_cast<type>(atof(element->GetText()));

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
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumLossDecrease");

        if(element)
        {
            const type new_minimum_loss_decrease = static_cast<type>(atof(element->GetText()));

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
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("LossGoal");

        if(element)
        {
            const type new_loss_goal = static_cast<type>(atof(element->GetText()));

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
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("GradientNormGoal");

        if(element)
        {
            const type new_gradient_norm_goal = static_cast<type>(atof(element->GetText()));

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
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumSelectionErrorIncreases");

        if(element)
        {
            const Index new_maximum_selection_error_increases = static_cast<Index>(atoi(element->GetText()));

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
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumEpochsNumber");

        if(element)
        {
            const Index new_maximum_epochs_number = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_maximum_epochs_number(new_maximum_epochs_number);
            }
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum time
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumTime");

        if(element)
        {
            const type new_maximum_time = static_cast<type>(atof(element->GetText()));

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
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveTrainingErrorHistory");

        if(element)
        {
            const string new_reserve_training_error_history = element->GetText();

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
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

        if(element)
        {
            const string new_reserve_selection_error_history = element->GetText();

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
    /*{
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("DisplayPeriod");

        if(element)
        {
           const Index new_display_period = static_cast<Index>(atoi(element->GetText()));

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
    */
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

