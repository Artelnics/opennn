 //   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "cross_entropy_error_3d.h"
#include "adaptive_moment_estimation.h"

namespace opennn
{

AdaptiveMomentEstimation::AdaptiveMomentEstimation(LossIndex* new_loss_index)
    : OptimizationAlgorithm(new_loss_index)
{
    set_default();
}


Index AdaptiveMomentEstimation::get_samples_number() const
{
    return batch_size;
}


const type& AdaptiveMomentEstimation::get_beta_1() const
{
    return beta_1;
}


const type& AdaptiveMomentEstimation::get_beta_2() const
{
    return beta_2;
}


const type& AdaptiveMomentEstimation::get_learning_rate() const
{
    return learning_rate;
}


const type& AdaptiveMomentEstimation::get_loss_goal() const
{
    return training_loss_goal;
}


const type& AdaptiveMomentEstimation::get_maximum_time() const
{
    return maximum_time;
}


void AdaptiveMomentEstimation::set_batch_size(const Index& new_batch_size)
{
    batch_size = new_batch_size;
}


void AdaptiveMomentEstimation::set_beta_1(const type& new_beta_1)
{
    beta_1 = new_beta_1;
}


void AdaptiveMomentEstimation::set_beta_2(const type& new_beta_2)
{
    beta_2= new_beta_2;
}


void AdaptiveMomentEstimation::set_default()
{
    display_period = 100;
}


void AdaptiveMomentEstimation::set_display(const bool& new_display)
{
    display = new_display;
}


void AdaptiveMomentEstimation::set_learning_rate(const type& new_learning_rate)
{
    learning_rate= new_learning_rate;
}


void AdaptiveMomentEstimation::set_custom_learning_rate(const type& parameter)
{
    use_custom_learning_rate = true;

    learning_rate = pow(parameter, -0.5);
}


void AdaptiveMomentEstimation::set_loss_goal(const type& new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


void AdaptiveMomentEstimation::set_accuracy_goal(const type& new_accuracy_goal)
{
    training_accuracy_goal = new_accuracy_goal;
}


void AdaptiveMomentEstimation::set_maximum_epochs_number(const Index& new_maximum_epochs_number)
{
    maximum_epochs_number = new_maximum_epochs_number;
}


void AdaptiveMomentEstimation::set_maximum_time(const type& new_maximum_time)
{
    maximum_time = new_maximum_time;
}


TrainingResults AdaptiveMomentEstimation::perform_training()
{
    if (!loss_index || !loss_index->has_neural_network() || !loss_index->has_data_set())
        return TrainingResults();

    TrainingResults results(maximum_epochs_number + 1);

    check();

    if(display) cout << "Training with adaptive moment estimation \"Adam\" ..." << endl;

    // Data set

    Dataset* dataset = loss_index->get_data_set();

    if(!dataset)
        throw runtime_error("Data set is null.");

    const bool has_selection = dataset->has_selection();
    
    const bool is_classification_model = is_instance_of<CrossEntropyError3d>(loss_index);

    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");
    // const vector<Index> decoder_variable_indices = dataset->get_variable_indices("Decoder");

    const vector<Index> training_samples_indices = dataset->get_sample_indices("Training");
    const vector<Index> selection_samples_indices = dataset->get_sample_indices("Selection");

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index selection_samples_number = dataset->get_samples_number("Selection");

    const Index training_batch_samples_number = min(training_samples_number, batch_size);

    const Index selection_batch_samples_number = (selection_samples_number != 0)
                                                 ? min(selection_samples_number, batch_size)
                                                 : 0;

    const Index training_batches_number = (training_batch_samples_number != 0)
        ? training_samples_number / training_batch_samples_number
        : 0;

    const Index selection_batches_number = (selection_batch_samples_number != 0)
       ? selection_samples_number / selection_batch_samples_number
       : 0;

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> selection_batches(selection_batches_number);

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    set_names();

    set_scaling();

    set_vocabularies();

    Batch training_batch(training_batch_samples_number, dataset);
    unique_ptr<Batch> selection_batch;

    ForwardPropagation training_forward_propagation(training_batch_samples_number, neural_network);
    unique_ptr<ForwardPropagation> selection_forward_propagation;

    // Loss index

    loss_index->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_samples_number, loss_index);
    unique_ptr<BackPropagation> selection_back_propagation;

    if (has_selection)
    {
        selection_batch = make_unique<Batch>(selection_batch_samples_number, dataset);
        selection_forward_propagation = make_unique<ForwardPropagation>(selection_batch_samples_number, neural_network);
        selection_back_propagation = make_unique<BackPropagation>(selection_batch_samples_number, loss_index);
    }

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
    time(&beginning_time);

    type elapsed_time = type(0);

    bool shuffle = true;

    if(neural_network->has("Recurrent"))
        shuffle = false;

    // Main loop

    optimization_data.iteration = 1;

    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = dataset->get_batches(training_samples_indices, training_batch_samples_number, shuffle);

        training_error = type(0);

        if(is_classification_model) training_accuracy = type(0);

        for(Index iteration = 0; iteration < training_batches_number; iteration++)
        {
            // Data set

            training_batch.fill(training_batches[iteration],
                                input_variable_indices,
                                // decoder_variable_indices,
                                target_variable_indices);

            // Neural network

            neural_network->forward_propagate(training_batch.get_input_pairs(),
                                              training_forward_propagation,
                                              is_training);

            // Loss index

            loss_index->back_propagate(training_batch,
                                       training_forward_propagation,
                                       training_back_propagation);

            training_error += training_back_propagation.error();

            if(is_classification_model) training_accuracy += training_back_propagation.accuracy(0);

            update_parameters(training_back_propagation, optimization_data);
        }

        // Loss

        training_error /= type(training_batches_number);
        if(is_classification_model)   
            training_accuracy /= type(training_batches_number);

        results.training_error_history(epoch) = training_error;

        if(has_selection)
        {
            selection_batches = dataset->get_batches(selection_samples_indices, selection_batch_samples_number, shuffle);
            
            selection_error = type(0);
            if(is_classification_model)    selection_accuracy = type(0);

            for(Index iteration = 0; iteration < selection_batches_number; iteration++)
            {
                // Data set

                selection_batch->fill(selection_batches[iteration],
                                      input_variable_indices,
                                      // decoder_variable_indices,
                                      target_variable_indices);

                // Neural network

                neural_network->forward_propagate(selection_batch->get_input_pairs(),
                                                  *selection_forward_propagation,
                                                  is_training);
                
                // Loss

                loss_index->calculate_error(*selection_batch,
                                            *selection_forward_propagation,
                                            *selection_back_propagation);

                selection_error += selection_back_propagation->error();

                if(is_classification_model) 
                    selection_accuracy += selection_back_propagation->accuracy(0);
            }

            selection_error /= type(selection_batches_number);
            if(is_classification_model) selection_accuracy /= type(selection_batches_number);

            results.selection_error_history(epoch) = selection_error;

            if(epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch-1)) selection_failures++;
        }
        
        // Elapsed time

        elapsed_time = get_elapsed_time(beginning_time);

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if(is_classification_model) cout << "Training accuracy: " << training_accuracy << endl;
            if(has_selection) cout << "Selection error: " << selection_error << endl;
            if(has_selection && is_classification_model) cout << "Selection accuracy: " << selection_accuracy << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        stop_training = true;

        if(epoch == maximum_epochs_number)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch << endl;
            results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
        }
        else if(elapsed_time >= maximum_time)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum training time reached: " << write_time(elapsed_time) << endl;
            results.stopping_condition = StoppingCondition::MaximumTime;
        }
        else if(results.training_error_history(epoch) < training_loss_goal)
        {
            results.stopping_condition  = StoppingCondition::LossGoal;
            if(display) cout << "Epoch " << epoch << "\nLoss goal reached: " << results.training_error_history(epoch) << endl;
        }
        else if(training_accuracy >= training_accuracy_goal)
        {
            results.stopping_condition  = StoppingCondition::LossGoal;
            if(display) cout << "Epoch " << epoch << "\nAccuracy goal reached: " << training_accuracy << endl;
        }
        else if(selection_failures >= maximum_selection_failures)
        {
            if(display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << selection_failures << endl;
            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }
        else
        {
            stop_training = false;
        }

        if(stop_training)
        {
            results.loss = training_back_propagation.loss;

            results.selection_failures = selection_failures;

            results.resize_training_error_history(epoch+1);

            results.resize_selection_error_history(has_selection ? epoch + 1 : 0);

            results.elapsed_time = write_time(elapsed_time);

            break;
        }

        if(epoch != 0 && epoch % save_period == 0) neural_network->save(neural_network_file_name);
    }
    
    set_unscaling();

    if(display) results.print();

    return results;
}


Tensor<string, 2> AdaptiveMomentEstimation::to_string_matrix() const
{
    Tensor<string, 2> string_matrix(9, 2);

    string_matrix.setValues({
    {"Learning rate", to_string(double(learning_rate))},
    {"Initial decay", to_string(double(initial_decay))},
    {"Beta 1", to_string(double(beta_1))},
    {"Beta 2", to_string(double(beta_2))},
    {"Epsilon", to_string(double(epsilon))},
    {"Training loss goal", to_string(double(training_loss_goal))},
    {"Maximum epochs number", to_string(maximum_epochs_number)},
    {"Maximum time", write_time(maximum_time)},
    {"Batch samples number", to_string(batch_size)}});

    return string_matrix;
}


void AdaptiveMomentEstimation::update_parameters(BackPropagation& back_propagation,
                                                 AdaptiveMomentEstimationData& optimization_data) const
{
/*
    NeuralNetwork* neural_network = back_propagation.loss_index->get_neural_network();

    Index& iteration = optimization_data.iteration;
    
    const type bias_correction =
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

    type effective_learning_rate = type(learning_rate * bias_correction);

    if(use_custom_learning_rate)
    {
        const type warmup_steps = 4000;
        type& step = optimization_data.step;
        effective_learning_rate *= learning_rate * min(pow(step, -0.5), step * pow(warmup_steps, -1.5));
        step++;
    }

    parameters.device(*thread_pool_device)
        -= effective_learning_rate*gradient_exponential_decay / (square_gradient_exponential_decay.sqrt() + epsilon);

    optimization_data.iteration++;

    // Update parameters
    neural_network->set_parameters(parameters);
*/

    NeuralNetwork* neural_network = back_propagation.loss_index->get_neural_network();
    const Index layers_number = neural_network->get_layers_number();

    optimization_data.iteration++;
    Index& iteration = optimization_data.iteration;

    const type bias_correction_1 = type(1) - pow(beta_1, type(iteration));
    const type bias_correction_2 = type(1) - pow(beta_2, type(iteration));

    for(Index layer_index = 0; layer_index < layers_number; layer_index++)
    {
        Layer* layer = neural_network->get_layer(layer_index).get();

        if (!layer->get_is_trainable())
            continue;

        LayerBackPropagation* layer_back_propagation = back_propagation.neural_network.layers[layer_index].get();

        const vector<pair<type*, Index>>& parameter_pairs = layer->get_parameter_pairs();
        const vector<pair<type*, Index>>& delta_pairs = layer_back_propagation->get_parameter_delta_pairs();

        for(Index parameter_index = 0; parameter_index < parameter_pairs.size(); parameter_index++)
        {
            type* parameter_data = parameter_pairs[parameter_index].first;
            const Index parameter_size = parameter_pairs[parameter_index].second;
            type* delta_data = delta_pairs[parameter_index].first;

            TensorMap<Tensor<type, 1>> parameters(parameter_data, parameter_size);
            TensorMap<Tensor<type, 1>> gradient(delta_data, parameter_size);

            Tensor<type, 1>& gradient_exponential_decay = optimization_data.gradient_exponential_decay[layer_index][parameter_index];
            Tensor<type, 1>& square_gradient_exponential_decay = optimization_data.square_gradient_exponential_decay[layer_index][parameter_index];

            gradient_exponential_decay.device(*thread_pool_device)
                = gradient_exponential_decay * beta_1 + gradient * (type(1) - beta_1);

            square_gradient_exponential_decay.device(*thread_pool_device)
                = square_gradient_exponential_decay * beta_2 + gradient.square() * (type(1) - beta_2);

            Tensor<type, 1> corrected_gradient_exponential_decay = gradient_exponential_decay / bias_correction_1;
            Tensor<type, 1> corrected_square_gradient_exponential_decay = square_gradient_exponential_decay / bias_correction_2;

            parameters.device(*thread_pool_device)
                -= learning_rate * corrected_gradient_exponential_decay / (corrected_square_gradient_exponential_decay.sqrt() + epsilon);
        }
    }
}


string AdaptiveMomentEstimation::get_name() const
{
    return "AdaptiveMomentEstimation";
}


void AdaptiveMomentEstimation::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("AdaptiveMomentEstimation");

    add_xml_element(printer, "BatchSize", to_string(batch_size));
    add_xml_element(printer, "LossGoal", to_string(training_loss_goal));
    add_xml_element(printer, "MaximumEpochsNumber", to_string(maximum_epochs_number));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));
    add_xml_element(printer, "HardwareUse", get_hardware_use());

    printer.CloseElement();
}


void AdaptiveMomentEstimation::from_XML(const XMLDocument& document)
{

    const XMLElement* root_element = document.FirstChildElement("AdaptiveMomentEstimation");

    if(!root_element)
        throw runtime_error("Adaptive moment estimation element is nullptr.\n");

    set_batch_size(read_xml_index(root_element, "BatchSize"));
    set_loss_goal(read_xml_type(root_element, "LossGoal")); 
    set_maximum_epochs_number(read_xml_index(root_element, "MaximumEpochsNumber"));   
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));    
    set_hardware_use(read_xml_string(root_element, "HardwareUse"));
}


AdaptiveMomentEstimationData::AdaptiveMomentEstimationData(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    set(new_adaptive_moment_estimation);
}


void AdaptiveMomentEstimationData::set(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    adaptive_moment_estimation = new_adaptive_moment_estimation;

    LossIndex* loss_index = new_adaptive_moment_estimation->get_loss_index();
    NeuralNetwork* neural_network = loss_index->get_neural_network();

    const Index layers_number = neural_network->get_layers_number();

    gradient_exponential_decay.resize(layers_number);
    square_gradient_exponential_decay.resize(layers_number);

    for (Index i = 0; i < layers_number; i++)
    {
        Layer* layer = neural_network->get_layer(i).get();

        if (!layer->get_is_trainable())
            continue;

        const auto& parameter_pairs = layer->get_parameter_pairs();
        const Index parameter_sets_number = parameter_pairs.size();

        gradient_exponential_decay[i].resize(parameter_sets_number);
        square_gradient_exponential_decay[i].resize(parameter_sets_number);

        for (Index j = 0; j < parameter_sets_number; j++)
        {
            const Index parameter_size = parameter_pairs[j].second;

            gradient_exponential_decay[i][j].resize(parameter_size);
            gradient_exponential_decay[i][j].setZero();

            square_gradient_exponential_decay[i][j].resize(parameter_size);
            square_gradient_exponential_decay[i][j].setZero();
        }
    }
}


void AdaptiveMomentEstimationData::print() const
{
    // cout << "Gradient exponential decay:" << endl
    //      << gradient_exponential_decay << endl
    //      << "Square gradient exponential decay:" << endl
    //      << square_gradient_exponential_decay << endl;
}


#ifdef OPENNN_CUDA

TrainingResults AdaptiveMomentEstimation::perform_training_cuda()
{
    if (!loss_index || !loss_index->has_neural_network() || !loss_index->has_data_set())
        return TrainingResults();

    TrainingResults results(maximum_epochs_number + 1);
    
    check();

    if (display) cout << "Training with adaptive moment estimation \"Adam\" CUDA ...\n";

    // Data set

    Dataset* dataset = loss_index->get_data_set();

    if (!dataset)
        throw runtime_error("Data set is null.");

    const bool has_selection = dataset->has_selection();

    const bool is_classification_model = is_instance_of<CrossEntropyError3d>(loss_index);

    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");
    const vector<Index> decoder_variable_indices = dataset->get_variable_indices("Decoder");

    const vector<Index> training_samples_indices = dataset->get_sample_indices("Training");
    const vector<Index> selection_samples_indices = dataset->get_sample_indices("Selection");

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index selection_samples_number = dataset->get_samples_number("Selection");

    const Index training_batch_samples_number = min(training_samples_number, batch_size);

    const Index selection_batch_samples_number = (selection_samples_number != 0)
        ? min(selection_samples_number, batch_size)
        : 0;

    const Index training_batches_number = (training_batch_samples_number != 0)
        ? training_samples_number / training_batch_samples_number
        : 0;

    const Index selection_batches_number = (selection_batch_samples_number != 0)
        ? selection_samples_number / selection_batch_samples_number
        : 0;

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> selection_batches(selection_batches_number);

    // Neural network

    NeuralNetwork* neural_network = loss_index->get_neural_network();

    set_names();

    set_scaling();

    set_vocabularies();

    BatchCuda training_batch_cuda(training_batch_samples_number, dataset);
    unique_ptr<BatchCuda> selection_batch_cuda;
    
    ForwardPropagationCuda training_forward_propagation_cuda(training_batch_samples_number, neural_network);
    unique_ptr<ForwardPropagationCuda> selection_forward_propagation_cuda;

    neural_network->allocate_parameters_device();
    neural_network->copy_parameters_device();

    // Loss Index

    loss_index->set_normalization_coefficient();

    BackPropagationCuda training_back_propagation_cuda(training_batch_samples_number, loss_index);
    unique_ptr<BackPropagationCuda> selection_back_propagation_cuda;

    if (has_selection)
    {
        selection_batch_cuda = make_unique<BatchCuda>(selection_batch_samples_number, dataset);
        selection_forward_propagation_cuda = make_unique<ForwardPropagationCuda>(selection_batch_samples_number, neural_network);
        selection_back_propagation_cuda = make_unique<BackPropagationCuda>(selection_batch_samples_number, loss_index);
    }

    type training_error = type(0);
    type training_accuracy = type(0);

    type selection_error = type(0);
    type selection_accuracy = type(0);

    Index selection_failures = 0;

    // Optimization algorithm

    ADAMOptimizationDataCuda optimization_data_cuda(this);

    bool stop_training = false;
    bool is_training = true;

    time_t beginning_time;
    time(&beginning_time);

    type elapsed_time = type(0);

    bool shuffle = true;

    if(neural_network->has("Recurrent"))
        shuffle = false;

    // Main loop

    optimization_data_cuda.iteration = 1;

    for (Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    { 
        if (display && epoch % display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = dataset->get_batches(training_samples_indices, training_batch_samples_number, shuffle);

        training_error = type(0);

        if (is_classification_model) training_accuracy = type(0);

        for (Index iteration = 0; iteration < training_batches_number; iteration++)
        {
            // Data set

            training_batch_cuda.fill(training_batches[iteration],
                                     input_variable_indices,
                                     decoder_variable_indices,
                                     target_variable_indices);

            // Neural network

            neural_network->forward_propagate_cuda(training_batch_cuda.get_input_device(),
                                                   training_forward_propagation_cuda,
                                                   is_training);

            // Loss index

            loss_index->back_propagate_cuda(training_batch_cuda,
                                            training_forward_propagation_cuda,
                                            training_back_propagation_cuda);

            training_error += training_back_propagation_cuda.error();

            if (is_classification_model)   training_accuracy += training_back_propagation_cuda.accuracy();

            // Optimization algorithm

            update_parameters_cuda(training_back_propagation_cuda, optimization_data_cuda);
        }

        // Loss

        training_error /= type(training_batches_number);

        if (is_classification_model)
            training_accuracy /= type(training_batches_number);

        results.training_error_history(epoch) = training_error;

        if (has_selection)
        {
            selection_batches = dataset->get_batches(selection_samples_indices, selection_batch_samples_number, shuffle);

            selection_error = type(0);
            if (is_classification_model)    selection_accuracy = type(0);

            for (Index iteration = 0; iteration < selection_batches_number; iteration++)
            {
                // Data set

                selection_batch_cuda->fill(selection_batches[iteration],
                                           input_variable_indices,
                                           decoder_variable_indices,
                                           target_variable_indices);

                // Neural network

                neural_network->forward_propagate_cuda(selection_batch_cuda->get_input_device(),
                                                       *selection_forward_propagation_cuda,
                                                       is_training);

                // Loss

                loss_index->calculate_error_cuda(*selection_batch_cuda,
                                                 *selection_forward_propagation_cuda,
                                                 *selection_back_propagation_cuda);

                selection_error += selection_back_propagation_cuda->error();

                if (is_classification_model)    
                    selection_accuracy += selection_back_propagation_cuda->accuracy();
            }

            selection_error /= type(selection_batches_number);
            if (is_classification_model) selection_accuracy /= type(selection_batches_number);

            results.selection_error_history(epoch) = selection_error;

            if (epoch != 0 && results.selection_error_history(epoch) > results.selection_error_history(epoch - 1)) selection_failures++;
        }

        // Elapsed time

        elapsed_time = get_elapsed_time(beginning_time);

        if (display && epoch % display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if (is_classification_model) cout << "Training accuracy: " << training_accuracy << endl;
            if (has_selection) cout << "Selection error: " << selection_error << endl;
            if (has_selection && is_classification_model) cout << "Selection accuracy: " << selection_accuracy << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        stop_training = true;

        if (epoch == maximum_epochs_number)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch << endl;
            results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
        }
        else if (elapsed_time >= maximum_time)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum training time reached: " << write_time(elapsed_time) << endl;
            results.stopping_condition = StoppingCondition::MaximumTime;
        }
        else if (results.training_error_history(epoch) < training_loss_goal)
        {
            results.stopping_condition = StoppingCondition::LossGoal;
            if (display) cout << "Epoch " << epoch << "\nLoss goal reached: " << results.training_error_history(epoch) << endl;
        }
        else if (training_accuracy >= training_accuracy_goal)
        {
            results.stopping_condition = StoppingCondition::LossGoal;
            if (display) cout << "Epoch " << epoch << "\nAccuracy goal reached: " << training_accuracy << endl;
        }
        else if (selection_failures >= maximum_selection_failures)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << selection_failures << endl;
            results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
        }
        else
        {
            stop_training = false;
        }

        if (stop_training)
        {
            results.loss = training_back_propagation_cuda.loss;

            results.selection_failures = selection_failures;

            results.resize_training_error_history(epoch + 1);

            results.resize_selection_error_history(has_selection ? epoch + 1 : 0);

            results.elapsed_time = write_time(elapsed_time);

            break;
        }

        if (epoch != 0 && epoch % save_period == 0) neural_network->save(neural_network_file_name);
    }

    set_unscaling();

    neural_network->copy_parameters_host();
    neural_network->free_parameters_device();

    if (display) results.print();

    return results;
    
}


void AdaptiveMomentEstimation::update_parameters_cuda(BackPropagationCuda& back_propagation_cuda,
                                                      ADAMOptimizationDataCuda& optimization_data_cuda) const
{
    NeuralNetwork* neural_network = back_propagation_cuda.loss_index->get_neural_network();
    const Index layers_number = neural_network->get_layers_number();

    optimization_data_cuda.iteration++;
    const int iteration = static_cast<int>(optimization_data_cuda.iteration);

    const float bias_correction_1 = 1.0f - powf(beta_1, static_cast<float>(iteration));
    const float bias_correction_2 = 1.0f - powf(beta_2, static_cast<float>(iteration));

    for (Index layer_index = 0; layer_index < layers_number; ++layer_index)
    {
        Layer* layer = neural_network->get_layer(layer_index).get();

        if (!layer->get_is_trainable())
            continue;

        const vector<pair<float*, Index>> parameter_pairs = layer->get_parameter_pair_device();

        LayerBackPropagationCuda* layer_back_prop = back_propagation_cuda.neural_network.layers[layer_index].get();
        const vector<pair<float*, Index>> delta_pairs = layer_back_prop->get_parameter_delta_pair_device();

        assert(parameter_pairs.size() == delta_pairs.size());

        for (Index parameter_index = 0; parameter_index < parameter_pairs.size(); ++parameter_index)
        {
            float* params_d = parameter_pairs[parameter_index].first;
            const Index param_size = parameter_pairs[parameter_index].second;
            const float* grads_d = delta_pairs[parameter_index].first;

            float* m_d = optimization_data_cuda.gradient_exponential_decay[layer_index][parameter_index];
            float* v_d = optimization_data_cuda.square_gradient_exponential_decay[layer_index][parameter_index];

            adam_update_device(
                param_size,
                params_d,
                m_d,
                v_d,
                grads_d,
                beta_1,
                beta_2,
                learning_rate,
                static_cast<float>(epsilon),
                bias_correction_1,
                bias_correction_2
            );
        }
    }
}


ADAMOptimizationDataCuda::ADAMOptimizationDataCuda(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    set(new_adaptive_moment_estimation);
}


void ADAMOptimizationDataCuda::set(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    cout << "ADAMOptimizationDataCuda set (granular):" << endl;
    adaptive_moment_estimation = new_adaptive_moment_estimation;

    NeuralNetwork* neural_network = adaptive_moment_estimation->get_loss_index()->get_neural_network();
    const Index layers_number = neural_network->get_layers_number();

    gradient_exponential_decay.resize(layers_number);
    square_gradient_exponential_decay.resize(layers_number);

    for (Index i = 0; i < layers_number; ++i)
    {
        Layer* layer = neural_network->get_layer(i).get();

        if (!layer->get_is_trainable())
        {
            continue;
        }

        const auto parameter_pairs = layer->get_parameter_pair_device();
        const size_t param_blocks_count = parameter_pairs.size();

        gradient_exponential_decay[i].resize(param_blocks_count, nullptr);
        square_gradient_exponential_decay[i].resize(param_blocks_count, nullptr);

        for (Index j = 0; j < param_blocks_count; ++j)
        {
            const Index param_size = parameter_pairs[j].second;

            if (param_size > 0)
            {
                const size_t memory_size_bytes = param_size * sizeof(float);

                CUDA_MALLOC_AND_REPORT(gradient_exponential_decay[i][j], memory_size_bytes);
                CHECK_CUDA(cudaMemset(gradient_exponential_decay[i][j], 0, memory_size_bytes));

                CUDA_MALLOC_AND_REPORT(square_gradient_exponential_decay[i][j], memory_size_bytes);
                CHECK_CUDA(cudaMemset(square_gradient_exponential_decay[i][j], 0, memory_size_bytes));
            }
        }
    }
}


void ADAMOptimizationDataCuda::free()
{
    for (auto& layer_moments : gradient_exponential_decay)
    {
        for (float*& ptr : layer_moments)
        {
            if (ptr != nullptr)
            {
                CHECK_CUDA(cudaFree(ptr));
                ptr = nullptr;
            }
        }
    }
    gradient_exponential_decay.clear();

    for (auto& layer_moments : square_gradient_exponential_decay)
    {
        for (float*& ptr : layer_moments)
        {
            if (ptr != nullptr)
            {
                CHECK_CUDA(cudaFree(ptr));
                ptr = nullptr;
            }
        }
    }
    square_gradient_exponential_decay.clear();
}


void ADAMOptimizationDataCuda::print() const
{
    cout << "--- ADAM Optimization Data (CUDA) ---" << endl;

    NeuralNetwork* neural_network = adaptive_moment_estimation->get_loss_index()->get_neural_network();
    const Index layers_number = neural_network->get_layers_number();

    for (Index i = 0; i < layers_number; ++i)
    {
        Layer* layer = neural_network->get_layer(i).get();
        if (!layer->get_is_trainable())
        {
            continue;
        }

        cout << "Layer " << i << " (" << layer->get_name() << "):" << endl;

        const auto parameter_pairs = layer->get_parameter_pair_device();

        for (Index j = 0; j < parameter_pairs.size(); ++j)
        {
            const Index param_size = parameter_pairs[j].second;

            if (param_size == 0) continue;

            cout << "  - Parameter Block " << j << " (Size: " << param_size << "):" << endl;

            cout << "    gradient_exponential_decay_host:" << endl << "      ";
            const float* m_device_ptr = gradient_exponential_decay[i][j];
            cout << vector_from_device(m_device_ptr, param_size) << endl;

            cout << "    square_gradient_exponential_decay_host:" << endl << "      ";
            const float* v_device_ptr = square_gradient_exponential_decay[i][j];
            cout << vector_from_device(v_device_ptr, param_size) << endl;
        }
    }
    cout << "-----------------------------------" << endl;
}

#endif

REGISTER(OptimizationAlgorithm, AdaptiveMomentEstimation, "AdaptiveMomentEstimation");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
