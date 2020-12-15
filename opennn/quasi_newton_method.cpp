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

    // Stopping criteria

    minimum_parameters_increment_norm = static_cast<type>(1.0e-3);

    minimum_loss_decrease = static_cast<type>(0.0);
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

    display = true;
    display_period = 5;
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

    // Set error learning rate

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



void QuasiNewtonMethod::initialize_inverse_hessian_approximation(QNMOptimizationData& optimization_data) const
{
    optimization_data.inverse_hessian.setZero();

    const Index parameters_number = optimization_data.parameters.size();

    for(Index i = 0; i < parameters_number; i++) optimization_data.inverse_hessian(i,i) = 1.0;

}

/// Calculates an approximation of the inverse hessian, accoring to the method used.
/// @param old_parameters Another point of the error function.
/// @param parameters Current point of the error function
/// @param old_gradient Gradient at the other point.
/// @param gradient Gradient at the current point.
/// @param old_inverse_hessian Inverse hessian at the other point of the error function.

void QuasiNewtonMethod::calculate_inverse_hessian_approximation(const LossIndex::BackPropagation& back_propagation,
                                                                QNMOptimizationData& optimization_data) const
{
    switch(inverse_hessian_approximation_method)
    {
    case DFP:
        calculate_DFP_inverse_hessian(back_propagation, optimization_data);

        return;

    case BFGS:
        calculate_BFGS_inverse_hessian(back_propagation, optimization_data);

        return;
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

void QuasiNewtonMethod::calculate_DFP_inverse_hessian(const LossIndex::BackPropagation& back_propagation,
                                                      QNMOptimizationData& optimization_data) const
{
    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    // Dots

    Tensor<type, 0> parameters_difference_dot_gradient_difference;

    parameters_difference_dot_gradient_difference.device(*thread_pool_device)
            = optimization_data.parameters_difference.contract(optimization_data.gradient_difference, AT_B); // Ok

    optimization_data.old_inverse_hessian_dot_gradient_difference.device(*thread_pool_device)
            = optimization_data.old_inverse_hessian.contract(optimization_data.gradient_difference, A_B); // Ok

    Tensor<type, 0> gradient_dot_hessian_dot_gradient;

    gradient_dot_hessian_dot_gradient.device(*thread_pool_device)
            = optimization_data.gradient_difference.contract(optimization_data.old_inverse_hessian_dot_gradient_difference, AT_B); // Ok , auto?

    // Calculates Approximation

    optimization_data.inverse_hessian = optimization_data.old_inverse_hessian; // TensorMap?

    optimization_data.inverse_hessian
            += kronecker_product(optimization_data.parameters_difference, optimization_data.parameters_difference)
            /parameters_difference_dot_gradient_difference(0); // Ok

    optimization_data.inverse_hessian
            -= kronecker_product(optimization_data.old_inverse_hessian_dot_gradient_difference, optimization_data.old_inverse_hessian_dot_gradient_difference)
            / gradient_dot_hessian_dot_gradient(0); // Ok
}


/// Returns an approximation of the inverse hessian matrix according to the
/// Broyden-Fletcher-Goldfarb-Shanno(BGFS) algorithm.
/// @param old_parameters A previous set of parameters.
/// @param old_gradient The gradient of the error function for that previous set of parameters.
/// @param old_inverse_hessian The hessian of the error function for that previous set of parameters.
/// @param parameters Actual set of parameters.
/// @param gradient The gradient of the error function for the actual set of parameters.

void QuasiNewtonMethod::calculate_BFGS_inverse_hessian(const LossIndex::BackPropagation& back_propagation,
                                                       QNMOptimizationData& optimization_data) const
{
    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    Tensor<type, 0> parameters_difference_dot_gradient_difference;

    parameters_difference_dot_gradient_difference.device(*thread_pool_device)
            = optimization_data.parameters_difference.contract(optimization_data.gradient_difference, AT_B);


    optimization_data.old_inverse_hessian_dot_gradient_difference.device(*thread_pool_device)
            = optimization_data.old_inverse_hessian.contract(optimization_data.gradient_difference, A_B);

    Tensor<type, 0> gradient_dot_hessian_dot_gradient;

    gradient_dot_hessian_dot_gradient.device(*thread_pool_device)
            = optimization_data.gradient_difference.contract(optimization_data.old_inverse_hessian_dot_gradient_difference, AT_B);

    Tensor<type, 1> BFGS(parameters_number);

    BFGS.device(*thread_pool_device)
            = optimization_data.parameters_difference/parameters_difference_dot_gradient_difference(0)
            - optimization_data.old_inverse_hessian_dot_gradient_difference/gradient_dot_hessian_dot_gradient(0);

    // Calculates Approximation

    optimization_data.inverse_hessian = optimization_data.old_inverse_hessian;

    optimization_data.inverse_hessian
            += kronecker_product(optimization_data.parameters_difference, optimization_data.parameters_difference)
            / parameters_difference_dot_gradient_difference(0); // Ok

    optimization_data.inverse_hessian
            -= kronecker_product(optimization_data.old_inverse_hessian_dot_gradient_difference, optimization_data.old_inverse_hessian_dot_gradient_difference)
            / gradient_dot_hessian_dot_gradient(0); // Ok

    optimization_data.inverse_hessian
            += kronecker_product(BFGS, BFGS)*(gradient_dot_hessian_dot_gradient(0)); // Ok
}



////// \brief QuasiNewtonMethod::update_epoch
////// \param batch
////// \param forward_propagation
////// \param back_propagation
////// \param optimization_data
void QuasiNewtonMethod::update_epoch(
        const DataSet::Batch& batch,
        NeuralNetwork::ForwardPropagation& forward_propagation,
        LossIndex::BackPropagation& back_propagation,
        QNMOptimizationData& optimization_data)
{
    #ifdef __OPENNN_DEBUG__

        check();

    #endif

    optimization_data.old_training_loss = back_propagation.loss;

    optimization_data.parameters_difference.device(*thread_pool_device)
            = optimization_data.parameters - optimization_data.old_parameters;

    optimization_data.gradient_difference.device(*thread_pool_device)
            = back_propagation.gradient - optimization_data.old_gradient;

    if(optimization_data.epoch == 0
    || is_zero(optimization_data.parameters_difference)
    || is_zero(optimization_data.gradient_difference))
    {
//        if(is_zero(optimization_data.parameters_difference)) cout << "parameters_difference" << endl;
//        if(is_zero(optimization_data.gradient_difference)) cout << "gradient_difference" << endl;

        initialize_inverse_hessian_approximation(optimization_data);
    }
    else
    {
        calculate_inverse_hessian_approximation(back_propagation, optimization_data);
    }

    // Optimization algorithm

    optimization_data.training_direction.device(*thread_pool_device)
            = -optimization_data.inverse_hessian.contract(back_propagation.gradient, A_B);

    // Calculate training slope

    optimization_data.training_slope.device(*thread_pool_device)
            = back_propagation.gradient.contract(optimization_data.training_direction, AT_B);

    // Check for a descent direction

    if(optimization_data.training_slope(0) >= 0)
    {
        cout << "Training slope is greater than zero." << endl;

        optimization_data.training_direction.device(*thread_pool_device) = -back_propagation.gradient;
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

    /// @todo ?
    // Reset training direction when learning rate is 0

    if(optimization_data.epoch != 0 && abs(optimization_data.learning_rate) < numeric_limits<type>::min())
    {
        optimization_data.training_direction.device(*thread_pool_device) = -back_propagation.gradient;

        directional_point = learning_rate_algorithm.calculate_directional_point(
                    batch,
                    forward_propagation,
                    back_propagation,
                    optimization_data);

        optimization_data.learning_rate = directional_point.first;
    }

    optimization_data.parameters_increment.device(*thread_pool_device)
            = optimization_data.training_direction*optimization_data.learning_rate;

    optimization_data.parameters_increment_norm = l2_norm(optimization_data.parameters_increment);

    optimization_data.old_parameters = optimization_data.parameters;

    optimization_data.parameters.device(*thread_pool_device) += optimization_data.parameters_increment;

    // Update stuff

    optimization_data.old_gradient = back_propagation.gradient;

    optimization_data.old_inverse_hessian = optimization_data.inverse_hessian;

    optimization_data.old_learning_rate = optimization_data.learning_rate;

    back_propagation.loss = directional_point.second;
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

    type gradient_norm = 0;

    type old_selection_error = numeric_limits<type>::max();

    LossIndex::BackPropagation training_back_propagation(training_samples_number, loss_index_pointer);
    LossIndex::BackPropagation selection_back_propagation(selection_samples_number, loss_index_pointer);

    // Optimization algorithm

    Tensor<type, 1> minimal_selection_parameters;

    type minimum_selection_error = numeric_limits<type>::max();

    bool stop_training = false;

    Index selection_failures = 0;

    time_t beginning_time, current_time;
    time(&beginning_time);
    type elapsed_time;

    QNMOptimizationData optimization_data(this);

    if(has_selection) results.resize_selection_history(maximum_epochs_number);

    // Main loop

    for(Index epoch = 0; epoch < maximum_epochs_number; epoch++)
    {
        optimization_data.epoch = epoch;

        // Neural network

        parameters_norm = l2_norm(optimization_data.parameters);

        neural_network_pointer->forward_propagate(training_batch, training_forward_propagation);

        loss_index_pointer->back_propagate(training_batch, training_forward_propagation, training_back_propagation);

        gradient_norm = l2_norm(training_back_propagation.gradient);

        // Selection error

        if(has_selection)
        {
            neural_network_pointer->forward_propagate(selection_batch, selection_forward_propagation);

            // Loss Index

            loss_index_pointer->calculate_error(selection_batch, selection_forward_propagation, selection_back_propagation);

            if(selection_back_propagation.error > old_selection_error)
            {
                selection_failures++;
            }
            else if(selection_back_propagation.error < minimum_selection_error)
            {
                minimum_selection_error = selection_back_propagation.error;

                minimal_selection_parameters = optimization_data.parameters;
            }

            if(reserve_selection_error_history) results.selection_error_history(epoch) = selection_back_propagation.error;
        }

        // Optimization data

        update_epoch(training_batch, training_forward_propagation, training_back_propagation, optimization_data);

        #ifdef __OPENNN_DEBUG__

        if(::isnan(training_back_propagation.error)){
            ostringstream buffer;

            buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
                   << "type perform_training() mehtod.\n"
                   << "Error is NAN.\n";

            throw logic_error(buffer.str());
        }
        #endif

        neural_network_pointer->set_parameters(optimization_data.parameters);

        // Training history

        if(reserve_training_error_history) results.training_error_history(epoch) = training_back_propagation.error;

        // Stopping Criteria

        time(&current_time);
        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        if(optimization_data.parameters_increment_norm <= minimum_parameters_increment_norm)
        {
            if(display)
            {
               cout << "Epoch " << epoch+1 << ": Minimum parameters increment norm reached.\n"
                    << "Parameters increment norm: " << optimization_data.parameters_increment_norm << endl;
            }

            stop_training = true;

            results.stopping_condition = MinimumParametersIncrementNorm;
        }
        else if(epoch != 0 &&
                training_back_propagation.loss - optimization_data.old_training_loss >= minimum_loss_decrease)
        {
            if(display)
            {
               cout << "Epoch " << epoch+1 << ": Minimum loss decrease (" << minimum_loss_decrease << ") reached.\n"
                    << "Loss decrease: " << training_back_propagation.loss - optimization_data.old_training_loss <<  endl;
            }

            stop_training = true;

            results.stopping_condition = MinimumLossDecrease;
        }
        else if(training_back_propagation.loss <= training_loss_goal)
        {
            if(display)
            {
                cout << "Epoch " << epoch+1 << ": Loss goal reached.\n";
            }

            stop_training = true;

            results.stopping_condition = LossGoal;
        }
        else if(gradient_norm <= gradient_norm_goal)
        {
            if(display)
            {
                cout << "Iteration " << epoch+1 << ": Gradient norm goal reached.\n";
            }

            stop_training = true;

            results.stopping_condition = GradientNormGoal;
        }
        else if(selection_failures >= maximum_selection_error_increases)
        {
            if(display)
            {
                cout << "Epoch " << epoch+1 << ": Maximum selection error increases reached.\n"
                     << "Selection loss increases: "<< selection_failures << endl;
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
            results.final_parameters = optimization_data.parameters;
            results.final_parameters_norm = parameters_norm;
            results.final_training_error = training_back_propagation.error;
            results.final_selection_error = selection_back_propagation.error;

            results.final_gradient_norm = gradient_norm;

            results.elapsed_time = write_elapsed_time(elapsed_time);

            results.epochs_number = epoch;

            results.resize_training_error_history(epoch+1);
            if(has_selection) results.resize_selection_error_history(epoch+1);

            if(display)
            {
                cout << "Parameters norm: " << parameters_norm << "\n"
                     << "Training error: " << training_back_propagation.error <<  "\n"
                     << "Gradient norm: " << gradient_norm <<  "\n"
                     << "Learning rate: " << optimization_data.learning_rate <<  "\n"
                     << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

                if(has_selection)
                {
                    cout << "Selection error: " << selection_back_propagation.error << endl;
                }
            }

            break;
        }
        else if((display && epoch == 0) || (display && (epoch+1) % display_period == 0))
        {
            cout << "Epoch " << epoch+1 << ";\n"
                 << "Parameters norm: " << parameters_norm << "\n"
                 << "Training error: " << training_back_propagation.error << "\n"
                 << "Gradient norm: " << gradient_norm << "\n"
                 << "Learning rate: " << optimization_data.learning_rate << "\n"
                 << "Elapsed time: " << write_elapsed_time(elapsed_time) << endl;

            if(has_selection)
            {
                cout << "Selection error: " << selection_back_propagation.error << endl;
            }
        }

        old_selection_error = selection_back_propagation.error;

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


/// Serializes the quasi Newton method object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void QuasiNewtonMethod::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("QuasiNewtonMethod");

    // Inverse hessian approximation method

    file_stream.OpenElement("InverseHessianApproximationMethod");

    file_stream.PushText(write_inverse_hessian_approximation_method().c_str());

    file_stream.CloseElement();

    // Learning rate algorithm

    learning_rate_algorithm.write_XML(file_stream);

    // Return minimum selection error neural network

    file_stream.OpenElement("ReturnMinimumSelectionErrorNN");

    buffer.str("");
    buffer << choose_best_selection;

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

    // Hardware use

    file_stream.OpenElement("HardwareUse");

    buffer.str("");
    buffer << hardware_use;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    file_stream.CloseElement();
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> QuasiNewtonMethod::to_string_matrix() const
{
    Tensor<string, 2> labels_values(12, 2);

    // Inverse hessian approximation method

    labels_values(0,0) = "Inverse hessian approximation method";

    const string inverse_hessian_approximation_method_string = write_inverse_hessian_approximation_method();

    labels_values(0,1) = inverse_hessian_approximation_method_string;

    // Learning rate method

    labels_values(1,0) = "Learning rate method";

    const string learning_rate_method = learning_rate_algorithm.write_learning_rate_method();

    labels_values(1,1) = "learning_rate_method";

    // Loss tolerance

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

    if(reserve_selection_error_history)
    {
        labels_values(11,1) = "true";
    }
    else
    {
        labels_values(11,1) = "false";
    }

    return labels_values;
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

