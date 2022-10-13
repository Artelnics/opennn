//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "quasi_newton_method.h"

namespace opennn
{

/// Default constructor.
/// It creates a quasi-Newton method optimization algorithm not associated with any loss index.
/// It also initializes the class members to their default values.

QuasiNewtonMethod::QuasiNewtonMethod()
    : OptimizationAlgorithm()
{
    set_default();
}


/// Loss index constructor.
/// It creates a quasi-Newton method optimization algorithm associated with a loss index.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

QuasiNewtonMethod::QuasiNewtonMethod(LossIndex* new_loss_index_pointer)
    : OptimizationAlgorithm(new_loss_index_pointer)
{
    learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);

    set_default();
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
    case InverseHessianApproximationMethod::DFP:
        return "DFP";

    case InverseHessianApproximationMethod::BFGS:
        return "BFGS";

    default:
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "string write_inverse_hessian_approximation_method() const method.\n"
               << "Unknown inverse hessian approximation method.\n";

        throw invalid_argument(buffer.str());
    }
}


const Index& QuasiNewtonMethod::get_epochs_number() const
{
    return epochs_number;
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


/// Returns the maximum number of selection error increases during the training process.

const Index& QuasiNewtonMethod::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
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


/// Sets a pointer to a loss index object to be associated with the quasi-Newton method object.
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
        inverse_hessian_approximation_method = InverseHessianApproximationMethod::DFP;
    }
    else if(new_inverse_hessian_approximation_method_name == "BFGS")
    {
        inverse_hessian_approximation_method = InverseHessianApproximationMethod::BFGS;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void set_inverse_hessian_approximation_method(const string&) method.\n"
               << "Unknown inverse hessian approximation method: " << new_inverse_hessian_approximation_method_name << ".\n";

        throw invalid_argument(buffer.str());
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void QuasiNewtonMethod::set_display(const bool& new_display)
{
    display = new_display;
}


void QuasiNewtonMethod::set_default()
{
    inverse_hessian_approximation_method = InverseHessianApproximationMethod::BFGS;

    learning_rate_algorithm.set_default();

    // Stopping criteria

    minimum_loss_decrease = type(0);
    training_loss_goal = type(0);
    maximum_selection_failures = numeric_limits<Index>::max();

    maximum_epochs_number = 1000;
    maximum_time = type(3600.0);

    // UTILITIES

    display = true;
    display_period = 10;
}


/// Sets a new minimum loss improvement during training.
/// @param new_minimum_loss_decrease Minimum improvement in the loss between two epochs.

void QuasiNewtonMethod::set_minimum_loss_decrease(const type& new_minimum_loss_decrease)
{
    minimum_loss_decrease = new_minimum_loss_decrease;
}


/// Sets a new goal value for the loss.
/// This is used as a stopping criterion when training a neural network
/// @param new_loss_goal Goal value for the loss.

void QuasiNewtonMethod::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


/// Sets a new maximum number of selection error increases.
/// @param new_maximum_selection_failures Maximum number of epochs in which the selection evalutation increases.

void QuasiNewtonMethod::set_maximum_selection_failures(const Index& new_maximum_selection_failures)
{
    maximum_selection_failures = new_maximum_selection_failures;
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
#ifdef OPENNN_DEBUG

    if(new_maximum_time < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be equal or greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    // Set maximum time

    maximum_time = new_maximum_time;
}


void QuasiNewtonMethod::initialize_inverse_hessian_approximation(QuasiNewtonMehtodData& optimization_data) const
{
    optimization_data.inverse_hessian.setZero();

    const Index parameters_number = optimization_data.inverse_hessian.dimension(0);

    for(Index i = 0; i < parameters_number; i++) optimization_data.inverse_hessian(i,i) = type(1);
}


/// Calculates an approximation of the inverse hessian, accoring to the method used.
/// @param old_parameters Another point of the error function.
/// @param parameters Current point of the error function
/// @param old_gradient Gradient at the other point.
/// @param gradient Gradient at the current point.
/// @param old_inverse_hessian Inverse hessian at the other point of the error function.

void QuasiNewtonMethod::calculate_inverse_hessian_approximation(QuasiNewtonMehtodData& optimization_data) const
{
    switch(inverse_hessian_approximation_method)
    {
    case InverseHessianApproximationMethod::DFP:
        calculate_DFP_inverse_hessian(optimization_data);
        return;

    case InverseHessianApproximationMethod::BFGS:
        calculate_BFGS_inverse_hessian(optimization_data);
        return;

    default:
        ostringstream buffer;

        buffer << "OpenNN Exception: QuasiNewtonMethod class.\n"
               << "Tensor<type, 1> calculate_inverse_hessian_approximation(const Tensor<type, 1>&, "
               "const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>&) method.\n"
               << "Unknown inverse hessian approximation method.\n";

        throw invalid_argument(buffer.str());
    }

}


Tensor<type, 2> QuasiNewtonMethod::kronecker_product(Tensor<type, 1>& left_matrix, Tensor<type, 1>& right_matrix) const
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


/// This method calculates the kronecker product between two matrices.
/// Its return a direct matrix.
/// @param left_matrix Matrix.
/// @param right_matrix Matrix.

Tensor<type, 2> QuasiNewtonMethod::kronecker_product(Tensor<type, 2>& left_matrix, Tensor<type, 2>& right_matrix) const
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

void QuasiNewtonMethod::calculate_DFP_inverse_hessian(QuasiNewtonMehtodData& optimization_data) const
{
    // Dots

    Tensor<type, 0> parameters_difference_dot_gradient_difference;

    parameters_difference_dot_gradient_difference.device(*thread_pool_device)
            = optimization_data.parameters_difference.contract(optimization_data.gradient_difference, AT_B);

    optimization_data.old_inverse_hessian_dot_gradient_difference.device(*thread_pool_device)
            = optimization_data.old_inverse_hessian.contract(optimization_data.gradient_difference, A_B);

    Tensor<type, 0> gradient_dot_hessian_dot_gradient;

    gradient_dot_hessian_dot_gradient.device(*thread_pool_device)
            = optimization_data.gradient_difference.contract(optimization_data.old_inverse_hessian_dot_gradient_difference, AT_B); // Ok , auto?

    // Calculates Approximation

    optimization_data.inverse_hessian = optimization_data.old_inverse_hessian;

    optimization_data.inverse_hessian
            += kronecker_product(optimization_data.parameters_difference, optimization_data.parameters_difference)
            /parameters_difference_dot_gradient_difference(0);

    optimization_data.inverse_hessian
            -= kronecker_product(optimization_data.old_inverse_hessian_dot_gradient_difference, optimization_data.old_inverse_hessian_dot_gradient_difference)
            / gradient_dot_hessian_dot_gradient(0);
}


/// Returns an approximation of the inverse hessian matrix according to the
/// Broyden-Fletcher-Goldfarb-Shanno(BGFS) algorithm.
/// @param old_parameters A previous set of parameters.
/// @param old_gradient The gradient of the error function for that previous set of parameters.
/// @param old_inverse_hessian The hessian of the error function for that previous set of parameters.
/// @param parameters Actual set of parameters.
/// @param gradient The gradient of the error function for the actual set of parameters.
/// @todo Add thread pool.

void QuasiNewtonMethod::calculate_BFGS_inverse_hessian(QuasiNewtonMehtodData& optimization_data) const
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


/// \brief QuasiNewtonMethod::update_parameters
/// \param batch
/// \param forward_propagation
/// \param back_propagation
/// \param optimization_data

void QuasiNewtonMethod::update_parameters(
        const DataSetBatch& batch,
        NeuralNetworkForwardPropagation& forward_propagation,
        LossIndexBackPropagation& back_propagation,
        QuasiNewtonMehtodData& optimization_data) const
{
    #ifdef OPENNN_DEBUG

        check();

    #endif


    optimization_data.parameters_difference.device(*thread_pool_device)
            = back_propagation.parameters - optimization_data.old_parameters;

    optimization_data.gradient_difference.device(*thread_pool_device)
            = back_propagation.gradient - optimization_data.old_gradient;

    optimization_data.old_parameters = back_propagation.parameters; // do not move above

    // Get training direction

    if(optimization_data.epoch == 0
    || is_zero(optimization_data.parameters_difference)
    || is_zero(optimization_data.gradient_difference))
    {
        initialize_inverse_hessian_approximation(optimization_data);
    }
    else
    {
        calculate_inverse_hessian_approximation(optimization_data);
    }

    optimization_data.training_direction.device(*thread_pool_device)
            = -optimization_data.inverse_hessian.contract(back_propagation.gradient, A_B);

    optimization_data.training_slope.device(*thread_pool_device)
            = back_propagation.gradient.contract(optimization_data.training_direction, AT_B);

    if(optimization_data.training_slope(0) >= type(0))
    {
        optimization_data.training_direction.device(*thread_pool_device) = -back_propagation.gradient;
    }

     // Get learning rate

    optimization_data.epoch == 0
            ? optimization_data.initial_learning_rate = first_learning_rate
            : optimization_data.initial_learning_rate = optimization_data.old_learning_rate;

    const pair<type,type> directional_point = learning_rate_algorithm.calculate_directional_point(
             batch,
             forward_propagation,
             back_propagation,
             optimization_data);

    optimization_data.learning_rate = directional_point.first;
    back_propagation.loss = directional_point.second;

    if(abs(optimization_data.learning_rate) > type(0))
    {
        optimization_data.parameters_increment.device(*thread_pool_device)
                = optimization_data.training_direction*optimization_data.learning_rate;

        back_propagation.parameters.device(*thread_pool_device) += optimization_data.parameters_increment;
    }
    else
    {
        const Index parameters_number = back_propagation.parameters.size();

        for(Index i = 0; i < parameters_number; i++)
        {
            if(abs(back_propagation.gradient(i)) < type(NUMERIC_LIMITS_MIN))
            {
                optimization_data.parameters_increment(i) = type(0);
            }
            else if(back_propagation.gradient(i) > type(0))
            {
                back_propagation.parameters(i) -= numeric_limits<type>::epsilon();

                optimization_data.parameters_increment(i) = -numeric_limits<type>::epsilon();
            }
            else if(back_propagation.gradient(i) < type(0))
            {
                back_propagation.parameters(i) += numeric_limits<type>::epsilon();

                optimization_data.parameters_increment(i) = numeric_limits<type>::epsilon();
            }
        }

        optimization_data.learning_rate = optimization_data.initial_learning_rate;
    }

    // Update stuff

    optimization_data.old_gradient = back_propagation.gradient;

    optimization_data.old_inverse_hessian = optimization_data.inverse_hessian;

    optimization_data.old_learning_rate = optimization_data.learning_rate;

    // Set parameters

    NeuralNetwork* neural_network_pointer = forward_propagation.neural_network_pointer;

    neural_network_pointer->set_parameters(back_propagation.parameters);
}


/// Trains a neural network with an associated loss index according to the quasi-Newton method.
/// Training occurs according to the training operators, training parameters and stopping criteria.

TrainingResults QuasiNewtonMethod::perform_training()
{
#ifdef OPENNN_DEBUG

    check();

#endif

    // Start training

    if(display) cout << "Training with quasi-Newton method...\n";

    TrainingResults results(maximum_epochs_number+1);

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    // Loss index

    const string error_type = loss_index_pointer->get_error_type();

    const Index training_samples_number = data_set_pointer->get_training_samples_number();

    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();
    const bool has_selection = data_set_pointer->has_selection();

    const Tensor<Index, 1> training_samples_indices = data_set_pointer->get_training_samples_indices();
    const Tensor<Index, 1> selection_samples_indices = data_set_pointer->get_selection_samples_indices();

    const Tensor<Index, 1> input_variables_indices = data_set_pointer->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set_pointer->get_target_variables_indices();

    const Tensor<string, 1> inputs_names = data_set_pointer->get_input_variables_names();
    const Tensor<string, 1> targets_names = data_set_pointer->get_target_variables_names();

    const Tensor<Scaler, 1> input_variables_scalers = data_set_pointer->get_input_variables_scalers();
    const Tensor<Scaler, 1> target_variables_scalers = data_set_pointer->get_target_variables_scalers();

    Tensor<Descriptives, 1> input_variables_descriptives;
    Tensor<Descriptives, 1> target_variables_descriptives;

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    NeuralNetworkForwardPropagation training_forward_propagation(training_samples_number, neural_network_pointer);
    NeuralNetworkForwardPropagation selection_forward_propagation(selection_samples_number, neural_network_pointer);

    neural_network_pointer->set_inputs_names(inputs_names);
    neural_network_pointer->set_outputs_names(targets_names);

    if(neural_network_pointer->has_scaling_layer())
    {
        input_variables_descriptives = data_set_pointer->scale_input_variables();

        ScalingLayer* scaling_layer_pointer = neural_network_pointer->get_scaling_layer_pointer();
        scaling_layer_pointer->set(input_variables_descriptives, input_variables_scalers);
    }

    if(neural_network_pointer->has_unscaling_layer())
    {
        target_variables_descriptives = data_set_pointer->scale_target_variables();

        UnscalingLayer* unscaling_layer_pointer = neural_network_pointer->get_unscaling_layer_pointer();
        unscaling_layer_pointer->set(target_variables_descriptives, target_variables_scalers);
    }

    if(neural_network_pointer->has_flatten_layer())
    {
        FlattenLayer* flatten_layer_pointer = neural_network_pointer->get_flatten_layer_pointer();
        //flatten_layer_pointer->set(target_variables_descriptives, target_variables_scalers);
    }

    DataSetBatch training_batch(training_samples_number, data_set_pointer);
    training_batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    DataSetBatch selection_batch(selection_samples_number, data_set_pointer);
    selection_batch.fill(selection_samples_indices, input_variables_indices, target_variables_indices);

    // Loss index

    loss_index_pointer->set_normalization_coefficient();

    LossIndexBackPropagation training_back_propagation(training_samples_number, loss_index_pointer);
    LossIndexBackPropagation selection_back_propagation(selection_samples_number, loss_index_pointer);

    // Optimization algorithm

    bool stop_training = false;

    Index selection_failures = 0;

    type old_loss = type(0);
    type loss_decrease = numeric_limits<type>::max();

    time_t beginning_time;
    time_t current_time;
    time(&beginning_time);
    type elapsed_time;

    QuasiNewtonMehtodData optimization_data(this);

    // Main loop

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        optimization_data.epoch = epoch;

        // Neural network

        neural_network_pointer->forward_propagate(training_batch, training_forward_propagation);

        // Loss index

        loss_index_pointer->back_propagate(training_batch, training_forward_propagation, training_back_propagation);
        results.training_error_history(epoch) = training_back_propagation.error;

        // Update parameters

        update_parameters(training_batch, training_forward_propagation, training_back_propagation, optimization_data);

        // Selection error

        if(has_selection)
        {
            neural_network_pointer->forward_propagate(selection_batch, selection_forward_propagation);

            // Loss Index

            loss_index_pointer->calculate_errors(selection_batch, selection_forward_propagation, selection_back_propagation);
            loss_index_pointer->calculate_error(selection_batch, selection_forward_propagation, selection_back_propagation);

            results.selection_error_history(epoch) = selection_back_propagation.error;

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) selection_failures++;
        }

        time(&current_time);
        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_back_propagation.error << endl;
            if(has_selection) cout << "Selection error: " << selection_back_propagation.error << endl;
            cout << "Learning rate: " << optimization_data.learning_rate << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        if(epoch != 0) loss_decrease = old_loss - training_back_propagation.loss;

        if(loss_decrease < minimum_loss_decrease)
        {
            if(display) cout << "Epoch " << epoch << endl << "Minimum loss decrease reached: " << loss_decrease << endl;

            stop_training = true;

            results.stopping_condition = OptimizationAlgorithm::StoppingCondition::MinimumLossDecrease;
        }

        old_loss = training_back_propagation.loss;

        if(training_back_propagation.loss <= training_loss_goal)
        {
            if(display) cout << "Epoch " << epoch << endl << "Loss goal reached: " << training_back_propagation.loss << endl;

            stop_training = true;

            results.stopping_condition = OptimizationAlgorithm::StoppingCondition::LossGoal;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum selection failures reached: " << selection_failures << endl;

            stop_training = true;

            results.stopping_condition = OptimizationAlgorithm::StoppingCondition::MaximumSelectionErrorIncreases;
        }
        else if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum number of epochs reached: " << epoch << endl;

            stop_training = true;

            results.stopping_condition = OptimizationAlgorithm::StoppingCondition::MaximumEpochsNumber;
        }
        else if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum training time reached: " << write_time(elapsed_time) << endl;

            stop_training = true;

            results.stopping_condition = OptimizationAlgorithm::StoppingCondition::MaximumTime;
        }

        if(stop_training)
        {
            results.loss = training_back_propagation.loss;

            results.loss_decrease = loss_decrease;

            results.selection_failures = selection_failures;

            results.resize_training_error_history(epoch+1);

            if(has_selection) results.resize_selection_error_history(epoch+1);
            else results.resize_selection_error_history(0);

            results.elapsed_time = write_time(elapsed_time);

            break;
        }

        if(epoch != 0 && epoch % save_period == 0) neural_network_pointer->save(neural_network_file_name);

        if(stop_training) break;
    }

    data_set_pointer->unscale_input_variables(input_variables_descriptives);

    if(neural_network_pointer->has_unscaling_layer())
        data_set_pointer->unscale_target_variables(target_variables_descriptives);

    if(display) results.print();

    return results;
}


string QuasiNewtonMethod::write_optimization_algorithm_type() const
{
    return "QUASI_NEWTON_METHOD";
}


/// Serializes the quasi Newton method object into an XML document of the TinyXML library without keeping the DOM tree in memory.
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

    // Maximum selection error increases

    file_stream.OpenElement("MaximumSelectionErrorIncreases");

    buffer.str("");
    buffer << maximum_selection_failures;

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

    // Hardware use

    file_stream.OpenElement("HardwareUse");

    buffer.str("");
    buffer << hardware_use;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    file_stream.CloseElement();
}


/// This method writes a matrix of strings the most representative atributes.

Tensor<string, 2> QuasiNewtonMethod::to_string_matrix() const
{
    Tensor<string, 2> labels_values(8, 2);

    // Inverse hessian approximation method

    labels_values(0,0) = "Inverse hessian approximation method";
    labels_values(0,1) = write_inverse_hessian_approximation_method();

    // Learning rate method

    labels_values(1,0) = "Learning rate method";
    labels_values(1,1) = learning_rate_algorithm.write_learning_rate_method();

    // Loss tolerance

    labels_values(2,0) = "Learning rate tolerance";
    labels_values(2,1) = to_string(double(learning_rate_algorithm.get_learning_rate_tolerance()));

    // Minimum loss decrease

    labels_values(3,0) = "Minimum loss decrease";
    labels_values(3,1) = to_string(double(minimum_loss_decrease));

    // Loss goal

    labels_values(4,0) = "Loss goal";
    labels_values(4,1) = to_string(double(training_loss_goal));

    // Maximum selection error increases

    labels_values(5,0) = "Maximum selection error increases";
    labels_values(5,1) = to_string(maximum_selection_failures);

    // Maximum epochs number

    labels_values(6,0) = "Maximum epochs number";
    labels_values(6,1) = to_string(maximum_epochs_number);

    // Maximum time

    labels_values(7,0) = "Maximum time";
    labels_values(7,1) = write_time(maximum_time);

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

        throw invalid_argument(buffer.str());
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
            catch(const invalid_argument& e)
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
            catch(const invalid_argument& e)
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
            catch(const invalid_argument& e)
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
            const Index new_maximum_selection_failures = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_maximum_selection_failures(new_maximum_selection_failures);
            }
            catch(const invalid_argument& e)
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
            catch(const invalid_argument& e)
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
            catch(const invalid_argument& e)
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
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
