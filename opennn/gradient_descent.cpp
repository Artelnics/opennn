//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R A D I E N T   D E S C E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "gradient_descent.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a gradient descent optimization algorithm not associated to any loss index object.
/// It also initializes the class members to their default values.

GradientDescent::GradientDescent()
    : OptimizationAlgorithm()
{
    set_default();
}


/// Loss index constructor.
/// It creates a gradient descent optimization algorithm associated to a loss index.
/// It also initializes the class members to their default values.
/// @param new_loss_index_pointer Pointer to a loss index object.

GradientDescent::GradientDescent(LossIndex* new_loss_index_pointer)
    : OptimizationAlgorithm(new_loss_index_pointer)
{
    learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);

    set_default();
}


/// Destructor.

GradientDescent::~GradientDescent()
{
}


/// Returns a constant reference to the learning rate algorithm object inside the gradient descent object.

const LearningRateAlgorithm& GradientDescent::get_learning_rate_algorithm() const
{
    return learning_rate_algorithm;
}


/// Returns a pointer to the learning rate algorithm object inside the gradient descent object.

LearningRateAlgorithm* GradientDescent::get_learning_rate_algorithm_pointer()
{
    return &learning_rate_algorithm;
}


/// Returns the hardware used. Default: Multi-core

string GradientDescent::get_hardware_use() const
{
    return hardware_use;
}


/// Returns the minimum norm of the parameter increment vector used as a stopping criteria when training.

const type& GradientDescent::get_minimum_parameters_increment_norm() const
{
    return minimum_parameters_increment_norm;
}


/// Returns the minimum loss improvement during training.

const type& GradientDescent::get_minimum_loss_decrease() const
{
    return minimum_loss_decrease;
}


/// Returns the goal value for the loss.
/// This is used as a stopping criterion when training a neural network.

const type& GradientDescent::get_loss_goal() const
{
    return training_loss_goal;
}


/// Returns the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a neural network.

const type& GradientDescent::get_gradient_norm_goal() const
{
    return gradient_norm_goal;
}


/// Returns the maximum number of selection error increases during the training process.

const Index& GradientDescent::get_maximum_selection_failures() const
{
    return maximum_selection_failures;
}


/// Returns the maximum number of iterations for training.

const Index& GradientDescent::get_maximum_epochs_number() const
{
    return maximum_epochs_number;
}


/// Returns the maximum training time.

const type& GradientDescent::get_maximum_time() const
{
    return maximum_time;
}


/// Sets a pointer to a loss index object to be associated to the gradient descent object.
/// It also sets that loss index to the learning rate algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void GradientDescent::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
    loss_index_pointer = new_loss_index_pointer;

    learning_rate_algorithm.set_loss_index_pointer(new_loss_index_pointer);
}


void GradientDescent::set_default()
{
    // Stopping criteria

    minimum_parameters_increment_norm = static_cast<type>(0.0);

    minimum_loss_decrease = -numeric_limits<type>::max();

    training_loss_goal = 0;
    gradient_norm_goal = 0;
    maximum_selection_failures = 100;

    maximum_epochs_number = 1000;
    maximum_time = 3600;

    // UTILITIES

    display_period = 10;
}


/// Set the a new maximum for the epochs number.
/// @param new_maximum_epochs number New maximum epochs number.

void GradientDescent::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_epochs_number < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void set_maximum_epochs_number(const type&) method.\n"
               << "Maximum epochs number must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    maximum_epochs_number = new_maximum_epochs_number;
}


/// Sets a new value for the minimum parameters increment norm stopping criterion.
/// @param new_minimum_parameters_increment_norm Value of norm of parameters increment norm used to stop training.

void GradientDescent::set_minimum_parameters_increment_norm(const type& new_minimum_parameters_increment_norm)
{
#ifdef OPENNN_DEBUG

    if(new_minimum_parameters_increment_norm < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
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

void GradientDescent::set_minimum_loss_decrease(const type& new_minimum_loss_decrease)
{
    minimum_loss_decrease = new_minimum_loss_decrease;
}


/// Sets a new goal value for the loss.
/// This is used as a stopping criterion when training a neural network.
/// @param new_loss_goal Goal value for the loss.

void GradientDescent::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


/// Sets a new the goal value for the norm of the error function gradient.
/// This is used as a stopping criterion when training a neural network.
/// @param new_gradient_norm_goal Goal value for the norm of the error function gradient.

void GradientDescent::set_gradient_norm_goal(const type& new_gradient_norm_goal)
{
#ifdef OPENNN_DEBUG

    if(new_gradient_norm_goal < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void set_gradient_norm_goal(const type&) method.\n"
               << "Gradient norm goal must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set gradient norm goal

    gradient_norm_goal = new_gradient_norm_goal;
}


/// Sets a new maximum number of selection error increases.
/// @param new_maximum_selection_failures Maximum number of epochs in which the selection evalutation
/// increases.

void GradientDescent::set_maximum_selection_failures(const Index& new_maximum_selection_failures)
{
    maximum_selection_failures = new_maximum_selection_failures;
}


/// Sets a new maximum training time.
/// @param new_maximum_time Maximum training time.

void GradientDescent::set_maximum_time(const type& new_maximum_time)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_time < static_cast<type>(0.0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set maximum time

    maximum_time = new_maximum_time;
}


/// Returns the gradient descent training direction,
/// which is the negative of the normalized gradient.
/// @param gradient Loss index gradient.

void GradientDescent::calculate_training_direction(const Tensor<type, 1>& gradient, Tensor<type, 1>& training_direction) const
{
#ifdef OPENNN_DEBUG

    ostringstream buffer;

    if(!loss_index_pointer)
    {
        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "Tensor<type, 1> calculate_training_direction(const Tensor<type, 1>&) const method.\n"
               << "Loss index pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    const Index gradient_size = gradient.size();

    if(gradient_size != parameters_number)
    {
        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "Tensor<type, 1> calculate_training_direction(const Tensor<type, 1>&) const method.\n"
               << "Size of gradient(" << gradient_size
               << ") is not equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    training_direction.device(*thread_pool_device) = -gradient;
}


/// \brief GradientDescent::update_parameters
/// \param batch
/// \param forward_propagation
/// \param back_propagation
/// \param optimization_data

void GradientDescent::update_parameters(
        const DataSetBatch& batch,
        NeuralNetworkForwardPropagation& forward_propagation,
        LossIndexBackPropagation& back_propagation,
        GradientDescentData& optimization_data)
{
    calculate_training_direction(back_propagation.gradient, optimization_data.training_direction);

    if(is_zero(optimization_data.training_direction))
        throw logic_error("Training direction is zero");

    // Get initial learning_rate

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

    if(abs(optimization_data.learning_rate) > 0)
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
            if(abs(back_propagation.gradient(i)) < numeric_limits<type>::min())
            {
                back_propagation.parameters(i) = back_propagation.parameters(i);

                optimization_data.parameters_increment(i) = 0;
            }
            else if(back_propagation.gradient(i) > 0)
            {
                back_propagation.parameters(i)
                        = nextafter(back_propagation.parameters(i), back_propagation.parameters(i)-1);

                back_propagation.parameters(i) -= numeric_limits<type>::epsilon();
//                        = nextafter(back_propagation.parameters(i), back_propagation.parameters(i)-1);

                optimization_data.parameters_increment(i) = -numeric_limits<type>::epsilon();
            }
            else if(back_propagation.gradient(i) < 0)
            {
                back_propagation.parameters(i) += numeric_limits<type>::epsilon();
//                        = nextafter(back_propagation.parameters(i), back_propagation.parameters(i)+1);

                optimization_data.parameters_increment(i) = numeric_limits<type>::epsilon();
            }
        }

        optimization_data.learning_rate = optimization_data.old_learning_rate;
    }

    optimization_data.parameters_increment_norm = l2_norm(thread_pool_device, optimization_data.parameters_increment);

    // Update parameters

    optimization_data.old_learning_rate = optimization_data.learning_rate;

    forward_propagation.neural_network_pointer->set_parameters(back_propagation.parameters);

}


/// Trains a neural network with an associated loss index,
/// according to the gradient descent method.
/// Training occurs according to the training parameters and stopping criteria.
/// It returns a results structure with the history and the final values of the reserved variables.

TrainingResults GradientDescent::perform_training()
{
    TrainingResults results(maximum_epochs_number+1);

#ifdef OPENNN_DEBUG
    check();
#endif

    // Start training

    if(display) cout << "Training with gradient descent...\n";

    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Index training_samples_number = data_set_pointer->get_training_samples_number();
    const Index selection_samples_number = data_set_pointer->get_selection_samples_number();

    const bool has_selection = data_set_pointer->has_selection();

    const Tensor<Index, 1> training_samples_indices = data_set_pointer->get_training_samples_indices();
    const Tensor<Index, 1> selection_samples_indices = data_set_pointer->get_selection_samples_indices();

    const Tensor<Index, 1> inputs_indices = data_set_pointer->get_input_variables_indices();
    const Tensor<Index, 1> target_indices = data_set_pointer->get_target_variables_indices();

    const Tensor<string, 1> inputs_names = data_set_pointer->get_input_variables_names();
    const Tensor<string, 1> targets_names = data_set_pointer->get_target_variables_names();

    const Tensor<Scaler, 1> input_variables_scalers = data_set_pointer->get_input_variables_scalers();
    const Tensor<Scaler, 1> target_variables_scalers = data_set_pointer->get_target_variables_scalers();

    const Tensor<Descriptives, 1> input_variables_descriptives =  data_set_pointer->scale_input_variables();
    Tensor<Descriptives, 1> target_variables_descriptives;

    DataSetBatch training_batch(training_samples_number, data_set_pointer);
    DataSetBatch selection_batch(selection_samples_number, data_set_pointer);

    training_batch.fill(training_samples_indices, inputs_indices, target_indices);
    selection_batch.fill(selection_samples_indices, inputs_indices, target_indices);

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    neural_network_pointer->set_inputs_names(inputs_names);
    neural_network_pointer->set_outputs_names(targets_names);

    if(neural_network_pointer->has_scaling_layer())
    {
        ScalingLayer* scaling_layer_pointer = neural_network_pointer->get_scaling_layer_pointer();
        scaling_layer_pointer->set(input_variables_descriptives, input_variables_scalers);
    }

    if(neural_network_pointer->has_unscaling_layer())
    {
        target_variables_descriptives = data_set_pointer->scale_target_variables();

        UnscalingLayer* unscaling_layer_pointer = neural_network_pointer->get_unscaling_layer_pointer();
        unscaling_layer_pointer->set(target_variables_descriptives, target_variables_scalers);
    }

    NeuralNetworkForwardPropagation training_forward_propagation(training_samples_number, neural_network_pointer);
    NeuralNetworkForwardPropagation selection_forward_propagation(selection_samples_number, neural_network_pointer);

    // Loss index

    const string error_type = loss_index_pointer->get_error_type();

    loss_index_pointer->set_normalization_coefficient();

    type gradient_norm = 0;

    LossIndexBackPropagation training_back_propagation(training_samples_number, loss_index_pointer);
    LossIndexBackPropagation selection_back_propagation(selection_samples_number, loss_index_pointer);

    // Optimization algorithm

    GradientDescentData optimization_data(this);

    Index selection_failures = 0;

    type parameters_increment_norm = numeric_limits<type>::max();

    bool stop_training = false;

    type old_loss = 0;
    type loss_decrease = numeric_limits<type>::max();

    // Main loop

    time_t beginning_time, current_time;
    time(&beginning_time);
    type elapsed_time = 0;

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        optimization_data.epoch = epoch;

        // Neural network

        neural_network_pointer->forward_propagate(training_batch, training_forward_propagation);

        // Loss index

        loss_index_pointer->back_propagate(training_batch, training_forward_propagation, training_back_propagation);
        results.training_error_history(epoch) = training_back_propagation.error;

        gradient_norm = l2_norm(thread_pool_device, training_back_propagation.gradient);

        if(has_selection)
        {
            neural_network_pointer->forward_propagate(selection_batch, selection_forward_propagation);

            loss_index_pointer->calculate_errors(selection_batch, selection_forward_propagation, selection_back_propagation);
            loss_index_pointer->calculate_error(selection_batch, selection_forward_propagation, selection_back_propagation);

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) selection_failures++;
        }

        // Optimization algorithm

        time(&current_time);
        elapsed_time = static_cast<type>(difftime(current_time, beginning_time));

        // Print progress

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_back_propagation.error << endl;
            if(has_selection) cout << "Selection error: " << selection_back_propagation.error << endl;
            cout << "Gradient norm: " << gradient_norm << endl;
            cout << "Learning rate: " << optimization_data.learning_rate << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        // Stopping Criteria

        if(training_back_propagation.loss <= training_loss_goal)
        {
            if(display)
                cout << "Epoch " << epoch << endl << "Loss goal reached: " << training_back_propagation.loss << endl;

            stop_training = true;

            results.stopping_condition = LossGoal;
        }

        else if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum selection failures reached: " << selection_failures << endl;

            stop_training = true;

            results.stopping_condition = MaximumSelectionErrorIncreases;
        }

        else if(gradient_norm <= gradient_norm_goal)
        {
            if(display) cout << "Epoch " << epoch << endl << "Gradient norm goal reached: " << gradient_norm << endl;

            stop_training = true;

            results.stopping_condition = GradientNormGoal;
        }

        else if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum number of epochs reached: " << epoch << endl;

            stop_training = true;

            results.stopping_condition = MaximumEpochsNumber;
        }

        if(optimization_data.parameters_increment_norm <= minimum_parameters_increment_norm)
        {
            if(display) cout << "Epoch " << epoch << endl << "Minimum parameters increment norm reached: " << parameters_increment_norm << endl;

            stop_training = true;

            results.stopping_condition = MinimumParametersIncrementNorm;
        }        

        else if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum training time reached: " << elapsed_time;

            stop_training = true;

            results.stopping_condition = MaximumTime;
        }

        if(epoch != 0) loss_decrease = old_loss - training_back_propagation.loss;

        if(loss_decrease < minimum_loss_decrease)
        {
            if(display) cout << "Epoch " << epoch << endl << "Minimum loss decrease reached: " << loss_decrease << endl;

            stop_training = true;

            results.stopping_condition = MinimumLossDecrease;
        }

        old_loss = training_back_propagation.loss;

        if(stop_training)
        {
            results.resize_training_error_history(epoch+1);

            if(has_selection) results.resize_selection_error_history(epoch+1);
            else results.resize_selection_error_history(0);

            results.gradient_norm = gradient_norm;

            results.elapsed_time = write_time(elapsed_time);

            break;
        }

        if(epoch != 0 && epoch%save_period == 0) neural_network_pointer->save(neural_network_file_name);

        update_parameters(training_batch, training_forward_propagation, training_back_propagation, optimization_data);
    }

    data_set_pointer->unscale_input_variables(input_variables_descriptives);

    if(neural_network_pointer->has_unscaling_layer())
        data_set_pointer->unscale_target_variables(target_variables_descriptives);

    if(display) results.print();

    return results;
}


string GradientDescent::write_optimization_algorithm_type() const
{
    return "GRADIENT_DESCENT";
}


/// Writes as matrix of strings the most representative atributes.

Tensor<string, 2> GradientDescent::to_string_matrix() const
{
    Tensor<string, 2> labels_values(9, 2);

    // Learning rate method

    labels_values(0,0) = "Learning rate method";

    const string learning_rate_method = learning_rate_algorithm.write_learning_rate_method();

    labels_values(0,1) = learning_rate_method;

    // Loss tolerance

    labels_values(1,0) = "Learning rate tolerance";

    labels_values(1,1) = to_string(learning_rate_algorithm.get_learning_rate_tolerance());

    // Minimum parameters increment norm

    labels_values(2,0) = "Minimum parameters increment norm";

    labels_values(2,1) = to_string(minimum_parameters_increment_norm);

    // Minimum loss decrease

    labels_values(3,0) = "Minimum loss decrease";

    labels_values(3,1) = to_string(minimum_loss_decrease);

    // Loss goal

    labels_values(4,0) = "Loss goal";

    labels_values(4,1) = to_string(training_loss_goal);

    // Gradient norm goal

    labels_values(5,0) = "Gradient norm goal";

    labels_values(5,1) = to_string(gradient_norm_goal);

    // Maximum selection error increases

    labels_values(6,0) = "Maximum selection error increases";

    labels_values(6,1) = to_string(maximum_selection_failures);

    // Maximum epochs number

    labels_values(7,0) = "Maximum epochs number";

    labels_values(7,1) = to_string(maximum_epochs_number);

    // Maximum time

    labels_values(8,0) = "Maximum time";

    labels_values(8,1) = write_time(maximum_time);

    return labels_values;
}


/// Serializes the gradient descent object into a XML document of the TinyXML library
/// without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void GradientDescent::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Learning rate algorithm

    file_stream.OpenElement("GradientDescent");

    learning_rate_algorithm.write_XML(file_stream);

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
    buffer << maximum_selection_failures;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum epochs number

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


void GradientDescent::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("GradientDescent");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: GradientDescent class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Gradient descent element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Learning rate algorithm
    {
        const tinyxml2::XMLElement* learning_rate_algorithm_element
                = root_element->FirstChildElement("LearningRateAlgorithm");

        if(learning_rate_algorithm_element)
        {
            tinyxml2::XMLDocument learning_rate_algorithm_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = learning_rate_algorithm_element->DeepClone(&learning_rate_algorithm_document);

            learning_rate_algorithm_document.InsertFirstChild(element_clone);

            learning_rate_algorithm.from_XML(learning_rate_algorithm_document);
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
            cout << "MinimumLossDecrease" << endl;
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
            const Index new_maximum_selection_failures = static_cast<Index>(atoi(element->GetText()));

            try
            {
                set_maximum_selection_failures(new_maximum_selection_failures);
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
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
