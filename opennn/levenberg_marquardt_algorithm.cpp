//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "levenberg_marquardt_algorithm.h"
#include "neural_network_forward_propagation.h"

namespace opennn
{

/// Default constructor.
/// It creates a Levenberg-Marquardt optimization algorithm object not associated with any loss index object.
/// It also initializes the class members to their default values.

LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithm()
    : OptimizationAlgorithm()
{
    set_default();
}


/// Loss index constructor.
/// It creates a Levenberg-Marquardt optimization algorithm object associated associated with a given loss index object.
/// It also initializes the class members to their default values.
/// @param new_loss_index Pointer to an external loss index object.

LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithm(LossIndex* new_loss_index)
    : OptimizationAlgorithm(new_loss_index)
{
    set_default();
}


/// Returns the minimum loss improvement during training.

const type& LevenbergMarquardtAlgorithm::get_minimum_loss_decrease() const
{
    return minimum_loss_decrease;
}


/// Returns the goal value for the loss.
/// This is used as a stopping criterion when training a neural network.

const type& LevenbergMarquardtAlgorithm::get_loss_goal() const
{
    return training_loss_goal;
}


/// Returns the maximum number of selection failures during the training process.

const Index& LevenbergMarquardtAlgorithm::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


/// Returns the maximum number of iterations for training.

const Index& LevenbergMarquardtAlgorithm::get_maximum_epochs_number() const
{
    return maximum_epochs_number;
}


/// Returns the maximum training time.

const type& LevenbergMarquardtAlgorithm::get_maximum_time() const
{
    return maximum_time;
}


/// Returns the damping parameter for the hessian approximation.

const type& LevenbergMarquardtAlgorithm::get_damping_parameter() const
{
    return damping_parameter;
}


/// Returns the damping parameter factor(beta in the User's Guide) for the hessian approximation.

const type& LevenbergMarquardtAlgorithm::get_damping_parameter_factor() const
{
    return damping_parameter_factor;
}


/// Returns the minimum damping parameter allowed in the algorithm.

const type& LevenbergMarquardtAlgorithm::get_minimum_damping_parameter() const
{
    return minimum_damping_parameter;
}


/// Returns the maximum damping parameter allowed in the algorithm.

const type& LevenbergMarquardtAlgorithm::get_maximum_damping_parameter() const
{
    return maximum_damping_parameter;
}


/// Sets the following default values for the Levenberg-Marquardt algorithm:
/// Training parameters:
/// <ul>
/// <li> Levenberg-Marquardt parameter: 0.001.
/// </ul>
/// Stopping criteria:
/// <ul>
/// <li> Loss goal: 1.0e-6.
/// <li> Maximum training time: 1000 secondata_set.
/// <li> Maximum number of epochs: 1000.
/// </ul>
/// User stuff:
/// <ul>
/// <li> Iterations between showing progress: 10.
/// </ul>

void LevenbergMarquardtAlgorithm::set_default()
{
    // Stopping criteria

    minimum_loss_decrease = type(0);
    training_loss_goal = type(0);
    maximum_selection_failures = 1000;

    maximum_epochs_number = 1000;
    maximum_time = type(3600.0);

    // UTILITIES

    display_period = 10;

    // Training parameters

    damping_parameter = type(1.0e-3);

    damping_parameter_factor = type(10.0);

    minimum_damping_parameter = type(1.0e-6);
    maximum_damping_parameter = type(1.0e6);
}


/// Sets a new damping parameter(lambda in the User's Guide) for the hessian approximation.
/// @param new_damping_parameter Damping parameter value.

void LevenbergMarquardtAlgorithm::set_damping_parameter(const type& new_damping_parameter)
{
    if(new_damping_parameter <= minimum_damping_parameter)
    {
        damping_parameter = minimum_damping_parameter;
    }
    else if(new_damping_parameter >= maximum_damping_parameter)
    {
        damping_parameter = maximum_damping_parameter;
    }
    else
    {
        damping_parameter = new_damping_parameter;
    }
}


/// Sets a new damping parameter factor(beta in the User's Guide) for the hessian approximation.
/// @param new_damping_parameter_factor Damping parameter factor value.

void LevenbergMarquardtAlgorithm::set_damping_parameter_factor(const type& new_damping_parameter_factor)
{
#ifdef OPENNN_DEBUG

    if(new_damping_parameter_factor <= type(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << endl
               << "void set_damping_parameter_factor(const type&) method." << endl
               << "Damping parameter factor must be greater than zero." << endl;

        throw runtime_error(buffer.str());
    }

#endif

    damping_parameter_factor = new_damping_parameter_factor;
}


/// Sets a new minimum damping parameter allowed in the algorithm.
/// @param new_minimum_damping_parameter Minimum damping parameter value.

void LevenbergMarquardtAlgorithm::set_minimum_damping_parameter(const type& new_minimum_damping_parameter)
{
#ifdef OPENNN_DEBUG

    if(new_minimum_damping_parameter <= type(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << endl
               << "void set_minimum_damping_parameter(const type&) method." << endl
               << "Minimum damping parameter must be greater than zero." << endl;

        throw runtime_error(buffer.str());
    }

#endif

    minimum_damping_parameter = new_minimum_damping_parameter;
}


/// Sets a new maximum damping parameter allowed in the algorithm.
/// @param new_maximum_damping_parameter Maximum damping parameter value.

void LevenbergMarquardtAlgorithm::set_maximum_damping_parameter(const type& new_maximum_damping_parameter)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_damping_parameter <= type(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << endl
               << "void set_maximum_damping_parameter(const type&) method." << endl
               << "Maximum damping parameter must be greater than zero." << endl;

        throw runtime_error(buffer.str());
    }

#endif

    maximum_damping_parameter = new_maximum_damping_parameter;
}


/// Sets a new minimum loss improvement during training.
/// @param new_minimum_loss_decrease Minimum improvement in the loss between two iterations.

void LevenbergMarquardtAlgorithm::set_minimum_loss_decrease(const type& new_minimum_loss_decrease)
{
    minimum_loss_decrease = new_minimum_loss_decrease;
}


/// Sets a new goal value for the loss.
/// This is used as a stopping criterion when training a neural network.
/// @param new_loss_goal Goal value for the loss.

void LevenbergMarquardtAlgorithm::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


/// Sets a new maximum number of selection error increases.
/// @param new_maximum_selection_failures Maximum number of epochs in which the
/// selection evalutation increases.

void LevenbergMarquardtAlgorithm::set_maximum_selection_failures(
        const Index& new_maximum_selection_failures)
{
    maximum_selection_failures = new_maximum_selection_failures;
}


/// Sets a maximum number of iterations for training.
/// @param new_maximum_epochs_number Maximum number of epochs for training.

void LevenbergMarquardtAlgorithm::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


/// Sets a new maximum training time.
/// @param new_maximum_time Maximum training time.

void LevenbergMarquardtAlgorithm::set_maximum_time(const type& new_maximum_time)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_time < type(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OptimizationAlgorithm class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be equal or greater than 0.\n";

        throw runtime_error(buffer.str());
    }

#endif

    maximum_time = new_maximum_time;
}


/// Checks that the Levenberg-Marquard object is ok for training.
/// In particular, it checks that:
/// <ul>
/// <li> The loss index pointer associated with the optimization algorithm is not nullptr,
/// <li> The neural network associated with that loss index is neither nullptr.
/// <li> The data set associated with that loss index is neither nullptr.
/// </ul>
/// If that checkings are not hold, an exception is thrown.

void LevenbergMarquardtAlgorithm::check() const
{
    ostringstream buffer;

    if(!loss_index)
    {
        buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class.\n"
               << "void check() const method.\n"
               << "Pointer to loss index is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const DataSet* data_set = loss_index->get_data_set();

    if(!data_set)
    {
        buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << endl
               << "void check() const method.\n"
               << "The loss funcional has no data set." << endl;

        throw runtime_error(buffer.str());
    }

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    if(!neural_network)
    {
        buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class." << endl
               << "void check() const method.\n"
               << "Pointer to neural network is nullptr." << endl;

        throw runtime_error(buffer.str());
    }
}


/// Trains a neural network with an associated loss index according to the Levenberg-Marquardt algorithm.
/// Training occurs according to the training parameters.

TrainingResults LevenbergMarquardtAlgorithm::perform_training()
{
    if(loss_index->get_error_type() == "MINKOWSKI_ERROR")
    {
        throw runtime_error("Levenberg-Marquard algorithm cannot work with Minkowski error.");
    }
    else if(loss_index->get_error_type() == "CROSS_ENTROPY_ERROR")
    {
        throw runtime_error("Levenberg-Marquard algorithm cannot work with cross-entropy error.");
    }
    else if(loss_index->get_error_type() == "WEIGHTED_SQUARED_ERROR")
    {
        throw runtime_error("Levenberg-Marquard algorithm is not implemented yet with weighted squared error.");
    }

    ostringstream buffer;

    // Control sentence

#ifdef OPENNN_DEBUG

    check();

#endif

    // Start training

    if(display) cout << "Training with Levenberg-Marquardt algorithm...\n";

    TrainingResults results(maximum_epochs_number+1);

    // Data set

    DataSet* data_set = loss_index->get_data_set();

    const bool has_selection = data_set->has_selection();

    const Index training_samples_number = data_set->get_training_samples_number();
    const Index selection_samples_number = data_set->get_selection_samples_number();

    const Tensor<Index, 1> training_samples_indices = data_set->get_training_samples_indices();
    const Tensor<Index, 1> selection_samples_indices = data_set->get_selection_samples_indices();

    const Tensor<Index, 1> input_variables_indices = data_set->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set->get_target_variables_indices();

    const Tensor<string, 1> inputs_names = data_set->get_input_variables_names();
    const Tensor<string, 1> targets_names = data_set->get_target_variables_names();

    const Tensor<Scaler, 1> input_variables_scalers = data_set->get_input_variables_scalers();
    const Tensor<Scaler, 1> target_variables_scalers = data_set->get_target_variables_scalers();

    Tensor<Descriptives, 1> input_variables_descriptives;
    Tensor<Descriptives, 1> target_variables_descriptives;

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    neural_network->set_inputs_names(inputs_names);
    neural_network->set_outputs_names(targets_names);

    if(neural_network->has_scaling_layer())
    {
        input_variables_descriptives = data_set->scale_input_variables();

        ScalingLayer2D* scaling_layer_2d = neural_network->get_scaling_layer_2d();
        scaling_layer_2d->set(input_variables_descriptives, input_variables_scalers);
    }

    if(neural_network->has_unscaling_layer())
    {
        target_variables_descriptives = data_set->scale_target_variables();

        UnscalingLayer* unscaling_layer = neural_network->get_unscaling_layer();
        unscaling_layer->set(target_variables_descriptives, target_variables_scalers);
    }

    Batch training_batch(training_samples_number, data_set);
    training_batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    Batch selection_batch(selection_samples_number, data_set);
    selection_batch.fill(selection_samples_indices, input_variables_indices, target_variables_indices);

    ForwardPropagation training_forward_propagation(training_samples_number, neural_network);
    ForwardPropagation selection_forward_propagation(selection_samples_number, neural_network);

    // Loss index

    loss_index->set_normalization_coefficient();

    type old_loss = type(0);
    type loss_decrease = numeric_limits<type>::max();

    Index selection_failures = 0;

    BackPropagationLM training_back_propagation_lm(training_samples_number, loss_index);
    BackPropagationLM selection_back_propagation_lm(selection_samples_number, loss_index);

    // Training strategy stuff

    bool stop_training = false;
    bool is_training = true;

    time_t beginning_time;
    time_t current_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    LevenbergMarquardtAlgorithmData optimization_data(this);

    // Main loop

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        optimization_data.epoch = epoch;

        // Neural network
        
        neural_network->forward_propagate(training_batch.get_inputs_pair(),
                                          training_forward_propagation,
                                          is_training);
        
        // Loss index

        loss_index->back_propagate_lm(training_batch,
                                      training_forward_propagation,
                                      training_back_propagation_lm);
        
        results.training_error_history(epoch) = training_back_propagation_lm.error;

        if(has_selection)
        {           
            neural_network->forward_propagate(selection_batch.get_inputs_pair(),
                                                      selection_forward_propagation,
                                                      is_training);

            loss_index->calculate_errors_lm(selection_batch,
                                                    selection_forward_propagation,
                                                    selection_back_propagation_lm);

            loss_index->calculate_squared_errors_lm(selection_batch,
                                                            selection_forward_propagation,
                                                            selection_back_propagation_lm);

            loss_index->calculate_error_lm(selection_batch,
                                                   selection_forward_propagation,
                                                   selection_back_propagation_lm);

            results.selection_error_history(epoch) = selection_back_propagation_lm.error;

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) selection_failures++;
        }

        // Elapsed time

        time(&current_time);
        elapsed_time = type(difftime(current_time, beginning_time));

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_back_propagation_lm.error << endl;
            if(has_selection) cout << "Selection error: " << selection_back_propagation_lm.error << endl;
            cout << "Damping parameter: " << damping_parameter << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        if(results.training_error_history(epoch) < training_loss_goal)
        {
            stop_training = true;

            results.stopping_condition = StoppingCondition::LossGoal;

            if(display) cout << "Epoch " << epoch << "\nLoss goal reached: " << results.training_error_history(epoch) << endl;
        }

        if(epoch != 0) loss_decrease = old_loss - training_back_propagation_lm.loss;

        if(loss_decrease < minimum_loss_decrease)
        {
            if(display) cout << "Epoch " << epoch << endl << "Minimum loss decrease reached: " << loss_decrease << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MinimumLossDecrease;
        }

        old_loss = training_back_propagation_lm.loss;

        if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << "Maximum selection failures reached: " << selection_failures << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }

        if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum number of epochs reached: " << epoch << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
        }

        if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << "Maximum training time reached: " << elapsed_time << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumTime;
        }

        if(stop_training)
        {
            results.loss = training_back_propagation_lm.loss;

            results.loss_decrease = loss_decrease;

            results.selection_failures = selection_failures;

            results.resize_training_error_history(epoch+1);

            if(has_selection) results.resize_selection_error_history(epoch+1);
            else results.resize_selection_error_history(0);

            results.elapsed_time = write_time(elapsed_time);

            break;
        }

        if(epoch != 0 && epoch%save_period == 0) neural_network->save(neural_network_file_name);

        update_parameters(training_batch,
                          training_forward_propagation,
                          training_back_propagation_lm,
                          optimization_data);
    }

    if(neural_network->has_scaling_layer())
        data_set->unscale_input_variables(input_variables_descriptives);

    if(neural_network->has_unscaling_layer())
        data_set->unscale_target_variables(target_variables_descriptives);

    if(display) results.print();

    return results;
}


/// \brief LevenbergMarquardtAlgorithm::update_parameters
/// \param batch
/// \param forward_propagation
/// \param back_propagation
/// \param back_propagation_lm
/// \param optimization_data

void LevenbergMarquardtAlgorithm::update_parameters(const Batch& batch,
                                                    ForwardPropagation& forward_propagation,
                                                    BackPropagationLM& back_propagation_lm,
                                                    LevenbergMarquardtAlgorithmData& optimization_data)
{

    const Tensor<pair<type*, dimensions>, 1> inputs_pair = batch.get_inputs_pair();

    const type regularization_weight = loss_index->get_regularization_weight();

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    Tensor<type, 1>& parameters = back_propagation_lm.parameters;

    type& error = back_propagation_lm.error;
    type& loss = back_propagation_lm.loss;

    const Tensor<type, 1>& gradient = back_propagation_lm.gradient;
    Tensor<type, 2>& hessian = back_propagation_lm.hessian;

    Tensor<type, 1>& potential_parameters = optimization_data.potential_parameters;
    Tensor<type, 1>& parameters_increment = optimization_data.parameters_increment;

    const Index parameters_number = parameters.size();

    bool success = false;

    do
    {
        sum_diagonal(hessian, damping_parameter);

        parameters_increment = perform_Householder_QR_decomposition(hessian, type(-1)*gradient);

        potential_parameters.device(*thread_pool_device) = parameters + parameters_increment;
        
        neural_network->forward_propagate(inputs_pair,
                                          potential_parameters,
                                          forward_propagation);

        loss_index->calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        loss_index->calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        loss_index->calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        type new_loss;

        try
        {
            new_loss = error + regularization_weight*loss_index->calculate_regularization(potential_parameters);

        }catch(exception)
        {
            new_loss = loss;
        }

        if(new_loss < loss) // succesfull step
        {
            set_damping_parameter(damping_parameter/damping_parameter_factor);

            parameters = potential_parameters;

            loss = new_loss;

            success = true;

            break;
        }
        else
        {
            sum_diagonal(hessian, -damping_parameter);

            set_damping_parameter(damping_parameter*damping_parameter_factor);
        }

    }while(damping_parameter < maximum_damping_parameter);

    if(!success)
    {
        #pragma omp parallel for

        for(Index i = 0; i < parameters_number; i++)
        {
            if(abs(gradient(i)) < type(NUMERIC_LIMITS_MIN))
            {
                parameters_increment(i) = type(0);
            }
            else if(gradient(i) > type(0))
            {
                parameters(i) -= numeric_limits<type>::epsilon();

                parameters_increment(i) = -numeric_limits<type>::epsilon();
            }
            else if(gradient(i) < type(0))
            {
                parameters(i) += numeric_limits<type>::epsilon();

                parameters_increment(i) = numeric_limits<type>::epsilon();
            }
        }
    }

    // Set parameters

    neural_network->set_parameters(parameters);
}


/// Writes the optimization algorithm type.

string LevenbergMarquardtAlgorithm::write_optimization_algorithm_type() const
{
    return "LEVENBERG_MARQUARDT_ALGORITHM";
}


/// This method writes a matrix of strings the most representative atributes.

Tensor<string, 2> LevenbergMarquardtAlgorithm::to_string_matrix() const
{
    Tensor<string, 2> labels_values(7, 2);

    // Damping parameter factor

    labels_values(0,0) = "Damping parameter factor";
    labels_values(0,1) = std::to_string(double(damping_parameter_factor));

    // Minimum loss decrease

    labels_values(2,0) = "Minimum loss decrease";
    labels_values(2,1) = std::to_string(double(minimum_loss_decrease));

    // Loss goal

    labels_values(3,0) = "Loss goal";
    labels_values(3,1) = std::to_string(double(training_loss_goal));

    // Maximum selection error increases

    labels_values(4,0) = "Maximum selection error increases";
    labels_values(4,1) = std::to_string(maximum_selection_failures);

    // Maximum epochs number

    labels_values(5,0) = "Maximum epochs number";
    labels_values(5,1) = std::to_string(maximum_epochs_number);

    // Maximum time

    labels_values(6,0) = "Maximum time";
    labels_values(6,1) = write_time(maximum_time);

    return labels_values;
}


/// Serializes the Levenberg Marquardt algorithm object into an XML document of the TinyXML library
/// without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void LevenbergMarquardtAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("LevenbergMarquardt");

    // Damping paramterer factor.

    file_stream.OpenElement("DampingParameterFactor");

    buffer.str("");
    buffer << damping_parameter_factor;

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


/// Loads a Levenberg-Marquardt method object from an XML document.
/// Please mind about the format, wich is specified in the OpenNN manual.
/// @param document TinyXML document containint the object data.

void LevenbergMarquardtAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("LevenbergMarquardt");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LevenbergMarquardtAlgorithm class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Levenberg-Marquardt algorithm element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Damping parameter factor

    const tinyxml2::XMLElement* damping_parameter_factor_element
            = root_element->FirstChildElement("DampingParameterFactor");

    if(damping_parameter_factor_element)
    {
        const type new_damping_parameter_factor = type(atof(damping_parameter_factor_element->GetText()));

        try
        {
            set_damping_parameter_factor(new_damping_parameter_factor);
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Minimum loss decrease

    const tinyxml2::XMLElement* minimum_loss_decrease_element = root_element->FirstChildElement("MinimumLossDecrease");

    if(minimum_loss_decrease_element)
    {
        const type new_minimum_loss_decrease = type(atof(minimum_loss_decrease_element->GetText()));

        try
        {
            set_minimum_loss_decrease(new_minimum_loss_decrease);
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Loss goal

    const tinyxml2::XMLElement* loss_goal_element = root_element->FirstChildElement("LossGoal");

    if(loss_goal_element)
    {
        const type new_loss_goal = type(atof(loss_goal_element->GetText()));

        try
        {
            set_loss_goal(new_loss_goal);
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Maximum selection error increases

    const tinyxml2::XMLElement* maximum_selection_failures_element
            = root_element->FirstChildElement("MaximumSelectionErrorIncreases");

    if(maximum_selection_failures_element)
    {
        const Index new_maximum_selection_failures
                = Index(atoi(maximum_selection_failures_element->GetText()));

        try
        {
            set_maximum_selection_failures(new_maximum_selection_failures);
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Maximum epochs number

    const tinyxml2::XMLElement* maximum_epochs_number_element = root_element->FirstChildElement("MaximumEpochsNumber");

    if(maximum_epochs_number_element)
    {
        const Index new_maximum_epochs_number = Index(atoi(maximum_epochs_number_element->GetText()));

        try
        {
            set_maximum_epochs_number(new_maximum_epochs_number);
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Maximum time

    const tinyxml2::XMLElement* maximum_time_element = root_element->FirstChildElement("MaximumTime");

    if(maximum_time_element)
    {
        const type new_maximum_time = type(atof(maximum_time_element->GetText()));

        try
        {
            set_maximum_time(new_maximum_time);
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
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
            catch(const exception& e)
            {
                cerr << e.what() << endl;
            }
        }
    }
}

void LevenbergMarquardtAlgorithmData::set(LevenbergMarquardtAlgorithm* new_Levenberg_Marquardt_method)
{
    Levenberg_Marquardt_algorithm = new_Levenberg_Marquardt_method;

    const LossIndex* loss_index = Levenberg_Marquardt_algorithm->get_loss_index();

    const NeuralNetwork* neural_network = loss_index->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_number();

    // Neural network data

    old_parameters.resize(parameters_number);

    parameters_difference.resize(parameters_number);

    potential_parameters.resize(parameters_number);
    parameters_increment.resize(parameters_number);
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
