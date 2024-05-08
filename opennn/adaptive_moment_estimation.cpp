 //   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "language_data_set.h"
#include "cross_entropy_error_3d.h"
#include "neural_network_forward_propagation.h"
#include "adaptive_moment_estimation.h"
#include "back_propagation.h"

namespace opennn
{

/// Default constructor.
/// It creates an adaptive moment estimation optimization algorithm not associated with any loss index object.
/// It also initializes the class members to their default values.

AdaptiveMomentEstimation::AdaptiveMomentEstimation()
    :OptimizationAlgorithm()
{
     set_default();
}


/// Loss index constructor.
/// It creates an adaptive moment estimation optimization algorithm associated with a loss index.
/// It also initializes the class members to their default values.
/// @param new_loss_index Pointer to a loss index object.

AdaptiveMomentEstimation::AdaptiveMomentEstimation(LossIndex* new_loss_index)
    : OptimizationAlgorithm(new_loss_index)
{
    set_default();
}


/// Destructor.

//AdaptiveMomentEstimation::~AdaptiveMomentEstimation()
//{
//}


/// Returns batch samples number.

Index AdaptiveMomentEstimation::get_batch_samples_number() const
{
    return batch_samples_number;
}


/// Returns beta 1.

const type& AdaptiveMomentEstimation::get_beta_1() const
{
    return beta_1;
}


/// Returns beta 2.

const type& AdaptiveMomentEstimation::get_beta_2() const
{
    return beta_2;
}


/// Returns epsilon.

const type& AdaptiveMomentEstimation::get_epsilon() const
{
    return epsilon;
}


/// Returns the initial learning rate.

const type& AdaptiveMomentEstimation::get_learning_rate() const
{
    return learning_rate;
}


/// Returns the goal value for the loss.
/// This is a stopping criterion when training a neural network.

const type& AdaptiveMomentEstimation::get_loss_goal() const
{
    return training_loss_goal;
}


/// Returns the maximum training time.

const type& AdaptiveMomentEstimation::get_maximum_time() const
{
    return maximum_time;
}


/// Set number of samples in each batch. Default 1000.
/// @param new_batch_sumples_number New value for batch samples number.

void AdaptiveMomentEstimation::set_batch_samples_number(const Index& new_batch_samples_number)
{
    batch_samples_number = new_batch_samples_number;
}


/// Sets beta 1 generally close to 1.
/// @param new_beta_1 New value for beta 1.

void AdaptiveMomentEstimation::set_beta_1(const type& new_beta_1)
{
    beta_1= new_beta_1;
}


/// Sets beta 2 generally close to 1.
/// @param new_beta_2 New value for beta 2.

void AdaptiveMomentEstimation::set_beta_2(const type& new_beta_2)
{
    beta_2= new_beta_2;
}


/// Sets adaptive moment estimation optimization algorithm to default.

void AdaptiveMomentEstimation::set_default()
{
    display_period = 100;
}


/// Sets epsilon.
/// @param epsilon New epsilon value.

void AdaptiveMomentEstimation::set_epsilon(const type& new_epsilon)
{
    epsilon= new_epsilon;
}


/// Sets a new learning rate.
/// @param new_learning_rate New learning rate.

void AdaptiveMomentEstimation::set_learning_rate(const type& new_learning_rate)
{
    learning_rate= new_learning_rate;
}


/// Sets a new custom learning rate.

void AdaptiveMomentEstimation::set_custom_learning_rate(const type& parameter)
{
    use_custom_learning_rate = true;

    learning_rate = pow(parameter, -0.5);
}


/// Sets a new goal value for the loss.
/// This is a stopping criterion when training a neural network.
/// @param new_loss_goal Goal value for the loss.

void AdaptiveMomentEstimation::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


/// Sets a new goal value for the accuracy.
/// This is a stopping criterion when training a neural network.
/// @param new_accuracy_goal Goal value for the accuracy.

void AdaptiveMomentEstimation::set_accuracy_goal(const type& new_accuracy_goal)
{
    training_accuracy_goal = new_accuracy_goal;
}


/// Sets a pointer to a loss index object to be associated with the gradient descent object.
/// It also sets that loss index to the learning rate algorithm.
/// @param new_loss_index Pointer to a loss index object.

void AdaptiveMomentEstimation::set_loss_index(LossIndex* new_loss_index)
{
    loss_index = new_loss_index;
}


/// Set the a new maximum for the epochs number.
/// @param new_maximum_epochs number New maximum epochs number.

void AdaptiveMomentEstimation::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_epochs_number < type(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: AdaptiveMomentEstimation class.\n"
               << "void set_maximum_epochs_number(const type&) method.\n"
               << "Maximum epochs number must be equal or greater than 0.\n";

        throw runtime_error(buffer.str());
    }

#endif

    // Set maximum_epochs number

    maximum_epochs_number = new_maximum_epochs_number;
}


/// Sets a new maximum training time.
/// @param new_maximum_time New maximum training time.

void AdaptiveMomentEstimation::set_maximum_time(const type& new_maximum_time)
{
#ifdef OPENNN_DEBUG

    if(new_maximum_time < type(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: AdaptiveMomentEstimation class.\n"
               << "void set_maximum_time(const type&) method.\n"
               << "Maximum time must be equal or greater than 0.\n";

        throw runtime_error(buffer.str());
    }

#endif

    // Set maximum time

    maximum_time = new_maximum_time;
}


/// Trains a neural network with an associated loss index,
/// according to the gradient descent method.
/// Training occurs according to the training parameters and stopping criteria.
/// It returns a results structure with the history and the final values of the reserved variables.

TrainingResults AdaptiveMomentEstimation::perform_training()
{
    TrainingResults results(maximum_epochs_number + 1);
    
    check();

    // Start training

    if(display) cout << "Training with adaptive moment estimation \"Adam\" ...\n";

    // Data set

    DataSet* data_set = loss_index->get_data_set();

    const bool has_selection = data_set->has_selection();

    bool is_language_model = false;
    if (is_instance_of<LanguageDataSet>(data_set))  is_language_model = true;

    bool is_classification_model = false;
    if (is_instance_of<CrossEntropyError3D>(loss_index))  is_classification_model = true;

    const Tensor<Index, 1> input_variables_indices = data_set->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set->get_target_variables_indices();
    Tensor<Index, 1> context_variables_indices;
    if (is_language_model)
    {
        LanguageDataSet* language_data_set = static_cast<LanguageDataSet*>(data_set);
        context_variables_indices = language_data_set->get_context_variables_indices();
    }

    const Tensor<Index, 1> training_samples_indices = data_set->get_training_samples_indices();
    const Tensor<Index, 1> selection_samples_indices = data_set->get_selection_samples_indices();

    const Tensor<string, 1> inputs_names = data_set->get_input_variables_names();

    const Tensor<string, 1> targets_names = data_set->get_target_variables_names();    

    const Tensor<Scaler, 1> input_variables_scalers = data_set->get_input_variables_scalers();
    const Tensor<Scaler, 1> target_variables_scalers = data_set->get_target_variables_scalers();

    const Tensor<Descriptives, 1> input_variables_descriptives = data_set->scale_input_variables();    

    Tensor<Descriptives, 1> target_variables_descriptives;

    Index training_batch_samples_number = 0;
    Index selection_batch_samples_number = 0;
    
    const Index training_samples_number = data_set->get_training_samples_number();
    const Index selection_samples_number = data_set->get_selection_samples_number();
    
    training_samples_number < batch_samples_number
            ? training_batch_samples_number = training_samples_number
            : training_batch_samples_number = batch_samples_number;

    selection_samples_number < batch_samples_number && selection_samples_number != 0
            ? selection_batch_samples_number = selection_samples_number
            : selection_batch_samples_number = batch_samples_number;
    
    Batch training_batch(training_batch_samples_number, data_set);
    Batch selection_batch(selection_batch_samples_number, data_set);
    
    const Index training_batches_number = training_samples_number/training_batch_samples_number;
    const Index selection_batches_number = selection_samples_number/selection_batch_samples_number;
    
    Tensor<Index, 2> training_batches(training_batches_number, training_batch_samples_number);
    Tensor<Index, 2> selection_batches(selection_batches_number, selection_batch_samples_number);
    
    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    neural_network->set_inputs_names(inputs_names);
    neural_network->set_outputs_names(targets_names);

    if(neural_network->has_scaling_layer())
    {
        ScalingLayer2D* scaling_layer_2d = neural_network->get_scaling_layer_2d();
        scaling_layer_2d->set(input_variables_descriptives, input_variables_scalers);
    }

    if(neural_network->has_unscaling_layer())
    {
        target_variables_descriptives = data_set->scale_target_variables();

        UnscalingLayer* unscaling_layer = neural_network->get_unscaling_layer();
        unscaling_layer->set(target_variables_descriptives, target_variables_scalers);
    }

    ForwardPropagation training_forward_propagation(training_batch_samples_number, neural_network);

    ForwardPropagation selection_forward_propagation(selection_batch_samples_number, neural_network);

    Tensor<pair<type*, dimensions>, 1> inputs_pair;

    // Loss index

    loss_index->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_samples_number, loss_index);
    BackPropagation selection_back_propagation(selection_batch_samples_number, loss_index);

//    type training_loss = type(0);
    type training_error = type(0);
    type training_accuracy = type(0);

    type selection_error = type(0);
    type selection_accuracy = type(0);

    Index selection_failures = 0;

    // Optimization algorithm

    AdaptiveMomentEstimationData optimization_data(this);

    bool stop_training = false;
    bool is_training = true;

    time_t beginning_time;
    time_t current_time;

    time(&beginning_time);
    type elapsed_time = type(0);

    bool shuffle = true;

    if(neural_network->has_long_short_term_memory_layer()
    || neural_network->has_recurrent_layer())
        shuffle = false;

    // Main loop
    
    optimization_data.iteration = 1;

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = data_set->get_batches(training_samples_indices, training_batch_samples_number, shuffle);
        
        const Index batches_number = training_batches.dimension(0);
        
//        training_loss = type(0);
        training_error = type(0);
        if(is_classification_model)    training_accuracy = type(0);
        
        //optimization_data.iteration = 1;
        
        for(Index iteration = 0; iteration < batches_number; iteration++)
        {

            // Data set

            training_batch.fill(training_batches.chip(iteration, 0),
                                input_variables_indices,
                                target_variables_indices,
                                context_variables_indices);
            
            // Neural network
            
            inputs_pair = training_batch.get_inputs_pair();

            neural_network->forward_propagate(inputs_pair,
                                              training_forward_propagation,
                                              is_training);
            
            // Loss index
            
            loss_index->back_propagate(training_batch,
                                       training_forward_propagation,
                                       training_back_propagation);
            
            training_error += training_back_propagation.error;
            if(is_classification_model)   training_accuracy += training_back_propagation.accuracy;
//            training_loss += training_back_propagation.loss;

            update_parameters(training_back_propagation, optimization_data);
            
            //if(display && epoch % display_period == 0)      display_progress_bar(iteration, batches_number - 1);
        }
        
        // Loss

        training_error /= type(batches_number);
        if(is_classification_model)   training_accuracy /= type(batches_number);

        results.training_error_history(epoch) = training_error;
        
        if(has_selection)
        {
            selection_batches = data_set->get_batches(selection_samples_indices, selection_batch_samples_number, shuffle);
            
            selection_error = type(0);
            if(is_classification_model)    selection_accuracy = type(0);
            
            for(Index iteration = 0; iteration < selection_batches_number; iteration++)
            {
                // Data set

                selection_batch.fill(selection_batches.chip(iteration,0),
                                     input_variables_indices,
                                     target_variables_indices,
                                     context_variables_indices);

                // Neural network
                
                inputs_pair = selection_batch.get_inputs_pair();

                neural_network->forward_propagate(inputs_pair,
                                                  selection_forward_propagation,
                                                  is_training);
                
                // Loss

                loss_index->calculate_error(selection_batch,
                                            selection_forward_propagation,
                                            selection_back_propagation);
                
                selection_error += selection_back_propagation.error;
                if(is_classification_model)    selection_accuracy += selection_back_propagation.accuracy;

            }

            selection_error /= type(selection_batches_number);
            if(is_classification_model)    selection_accuracy /= type(selection_batches_number);

            results.selection_error_history(epoch) = selection_error;

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) selection_failures++;

        }
        
        // Elapsed time

        time(&current_time);
        elapsed_time = type(difftime(current_time, beginning_time));

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if (is_classification_model) cout << "Training accuracy: " << training_accuracy << endl;
            if(has_selection) cout << "Selection error: " << selection_error << endl;
            if (has_selection && is_classification_model) cout << "Selection accuracy: " << selection_accuracy << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        // Training history

        if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum number of epochs reached: " << epoch << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
        }

        if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum training time reached: " << write_time(elapsed_time) << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumTime;
        }

        /// @todo loss and error missmatch

        if(results.training_error_history(epoch) < training_loss_goal)
        {
            stop_training = true;

            results.stopping_condition  = StoppingCondition::LossGoal;

            if(display) cout << "Epoch " << epoch << endl << "Loss goal reached: " << results.training_error_history(epoch) << endl;
        }

        if(training_accuracy >= training_accuracy_goal)
        {
            stop_training = true;

            results.stopping_condition  = StoppingCondition::LossGoal;

            if(display) cout << "Epoch " << epoch << endl << "Accuracy goal reached: " << training_accuracy << endl;
        }

        if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << endl << "Maximum selection failures reached: " << selection_failures << endl;

            stop_training = true;

            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }

        if(stop_training)
        {
            results.loss = training_back_propagation.loss;

            results.selection_failures = selection_failures;

            results.resize_training_error_history(epoch+1);

            if(has_selection) results.resize_selection_error_history(epoch+1);
            else results.resize_selection_error_history(0);

            results.elapsed_time = write_time(elapsed_time);

            break;
        }

        if(epoch != 0 && epoch % save_period == 0) neural_network->save(neural_network_file_name);
    }
    
    data_set->unscale_input_variables(input_variables_descriptives);

    if(neural_network->has_unscaling_layer())
        data_set->unscale_target_variables(target_variables_descriptives);

    if(display) results.print();
    
    return results;
}


/// Writes in a matrix of strings the most representative atributes.

Tensor<string, 2> AdaptiveMomentEstimation::to_string_matrix() const
{
    Tensor<string, 2> labels_values(9, 2);

    // Initial learning rate

    labels_values(0,0) = "Learning rate";
    labels_values(0,1) = std::to_string(double(learning_rate));

    // Initial decay

    labels_values(1,0) = "Initial decay";
    labels_values(1,1) = std::to_string(double(initial_decay));

    // Beta 1

    labels_values(2,0) = "Beta 1";
    labels_values(2,1) = std::to_string(double(beta_1));

    // Beta 2

    labels_values(3,0) = "Beta 2";
    labels_values(3,1) = std::to_string(double(beta_2));

    // Epsilon

    labels_values(4,0) = "Epsilon";
    labels_values(4,1) = std::to_string(double(epsilon));

    // Training loss goal

    labels_values(5,0) = "Training loss goal";
    labels_values(5,1) = std::to_string(double(training_loss_goal));

    // Maximum epochs number

    labels_values(6,0) = "Maximum epochs number";
    labels_values(6,1) = std::to_string(maximum_epochs_number);

    // Maximum time

    labels_values(7,0) = "Maximum time";
    labels_values(7,1) = write_time(maximum_time);

    // Batch samples number

    labels_values(8,0) = "Batch samples number";
    labels_values(8,1) = std::to_string(batch_samples_number);

    return labels_values;
}


/// Update iteration parameters.
/// @param back_propagation New loss index back propagation.
/// @param optimization_data New moment estimation data.

void AdaptiveMomentEstimation::update_parameters(BackPropagation& back_propagation,
                                                 AdaptiveMomentEstimationData& optimization_data) const
{
    NeuralNetwork* neural_network = loss_index->get_neural_network();

    Index& iteration = optimization_data.iteration;
    
    const type bias_correction =
            sqrt(type(1) - pow(beta_2, type(iteration))) /
            sqrt(type(1) - pow(beta_2, type(iteration))) /
            (type(1) - pow(beta_1, type(iteration)));

    const Tensor<type, 1>& gradient = back_propagation.gradient;

    Tensor<type, 1>& gradient_exponential_decay = optimization_data.gradient_exponential_decay;

    Tensor<type, 1>& square_gradient_exponential_decay = optimization_data.square_gradient_exponential_decay;

    Tensor<type, 1>& parameters = back_propagation.parameters;

    gradient_exponential_decay.device(*thread_pool_device)
        = gradient * (type(1) - beta_1) + gradient_exponential_decay * beta_1;

    square_gradient_exponential_decay.device(*thread_pool_device)
        = gradient.square() * (type(1) - beta_2) + square_gradient_exponential_decay * beta_2;

    if (!use_custom_learning_rate)
    {
        parameters.device(*thread_pool_device)
            -= (learning_rate * bias_correction) * gradient_exponential_decay / (square_gradient_exponential_decay.sqrt() + epsilon);
    }
    else
    {
        const type warmup_steps = 4000;
        type& step = optimization_data.step;

        const type custom_learning_rate = learning_rate * min(pow(step, -0.5), step * pow(warmup_steps, -1.5));

        parameters.device(*thread_pool_device)
            -= (custom_learning_rate * bias_correction) * gradient_exponential_decay / (square_gradient_exponential_decay.sqrt() + epsilon);

        step++;
    }

    optimization_data.iteration++;

    // Update parameters

    neural_network->set_parameters(parameters);
}


/// Write a string with best algorithm type for the model.

string AdaptiveMomentEstimation::write_optimization_algorithm_type() const
{
    return "ADAPTIVE_MOMENT_ESTIMATION";
}


/// Serializes the adaptive moment estimation object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.
/// @param file_stream.

void AdaptiveMomentEstimation::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("AdaptiveMomentEstimation");

    // Batch size

    file_stream.OpenElement("BatchSize");

    buffer.str("");
    buffer << batch_samples_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Loss goal

    file_stream.OpenElement("LossGoal");

    buffer.str("");
    buffer << training_loss_goal;

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
    buffer << get_hardware_use();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // End element

    file_stream.CloseElement();
}


/// Imports the adaptive moment estimation object from an XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document.
/// @param document.

void AdaptiveMomentEstimation::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("AdaptiveMomentEstimation");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: AdaptiveMomentEstimation class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Adaptive moment estimation element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Batch size

    const tinyxml2::XMLElement* batch_samples_number_element = root_element->FirstChildElement("BatchSize");

    if(batch_samples_number_element)
    {
        const Index new_batch_samples_number = Index(atoi(batch_samples_number_element->GetText()));

        try
        {
            set_batch_samples_number(new_batch_samples_number);
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Loss goal
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("LossGoal");

        if(element)
        {
            const type new_loss_goal = type(atof(element->GetText()));

            try
            {
                set_loss_goal(new_loss_goal);
            }
            catch(const exception& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Maximum eochs number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumEpochsNumber");

        if(element)
        {
            const Index new_maximum_epochs_number = Index(atoi(element->GetText()));

            try
            {
                set_maximum_epochs_number(new_maximum_epochs_number);
            }
            catch(const exception& e)
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
            const type new_maximum_time = type(atof(element->GetText()));

            try
            {
                set_maximum_time(new_maximum_time);
            }
            catch(const exception& e)
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
            catch(const exception& e)
            {
                cerr << e.what() << endl;
            }
        }
    }
}


/// Default constructor

//AdaptiveMomentEstimationData::AdaptiveMomentEstimationData()
//{
//}


/// Adaptive Moment Estimation constructor.
/// It creates an adaptive moment estimation data object associated with an adaptive moment estimation algorithm.
/// @param new_adaptive_moment_estimation Pointer to a adaptive moment estimation object.

AdaptiveMomentEstimationData::AdaptiveMomentEstimationData(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    set(new_adaptive_moment_estimation);
}


/// Sets a new adaptive moment estimation pointer.
/// @param new_adaptive_moment_estimation New adaptive moment estimation pointer.

void AdaptiveMomentEstimationData::set(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    adaptive_moment_estimation = new_adaptive_moment_estimation;

    LossIndex* loss_index = new_adaptive_moment_estimation->get_loss_index();

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_number();

    gradient_exponential_decay.resize(parameters_number);
    gradient_exponential_decay.setZero();

    square_gradient_exponential_decay.resize(parameters_number);
    square_gradient_exponential_decay.setZero();
}


/// Prints on the screen the information about de AdaptiveMomentEstimation data.
/// <ul>
/// <li> Gradient exponential decay.
/// <li> Square gradient exponential decay.
/// </ul>

void AdaptiveMomentEstimationData::print() const
{
    cout << "Gradient exponential decay:" << endl
         <<gradient_exponential_decay << endl;

    cout << "Square gradient exponential decay:" << endl
         << square_gradient_exponential_decay << endl;
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
