 //   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error_3d.h"
#include "adaptive_moment_estimation.h"
#include "forward_propagation.h"
#include "back_propagation.h"

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


const type& AdaptiveMomentEstimation::get_epsilon() const
{
    return epsilon;
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


void AdaptiveMomentEstimation::set_batch_samples_number(const Index& new_batch_size)
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


void AdaptiveMomentEstimation::set_epsilon(const type& new_epsilon)
{
    epsilon = new_epsilon;
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

    if(display) cout << "Training with adaptive moment estimation \"Adam\" ...\n";

    // Data set

    DataSet* data_set = loss_index->get_data_set();

    if(!data_set)
        throw runtime_error("Data set is null.");

    const bool has_selection = data_set->has_selection();
    
    const bool is_classification_model = is_instance_of<CrossEntropyError3D>(loss_index);

    const vector<Index> input_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Target);
    const vector<Index> decoder_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Decoder);

    const vector<Index> training_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Training);
    const vector<Index> selection_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Selection);

    const Index training_samples_number = data_set->get_samples_number(DataSet::SampleUse::Training);
    const Index selection_samples_number = data_set->get_samples_number(DataSet::SampleUse::Selection);

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

    Batch training_batch(training_batch_samples_number, data_set);
    Batch selection_batch(selection_batch_samples_number, data_set);

    ForwardPropagation training_forward_propagation(training_batch_samples_number, neural_network);
    ForwardPropagation selection_forward_propagation(selection_batch_samples_number, neural_network);

    // Loss index

    loss_index->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_samples_number, loss_index);
    BackPropagation selection_back_propagation(selection_batch_samples_number, loss_index);

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

    bool shuffle = 0;

    if(neural_network->has(Layer::Type::Recurrent))
        shuffle = false;

    // Main loop
    
    optimization_data.iteration = 1;
    for(Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = data_set->get_batches(training_samples_indices, training_batch_samples_number, shuffle);

        // cout << "Training_batches phrases:" << endl;
        // print_vector(training_batches);
        // throw runtime_error("");

        training_error = type(0);

        if(is_classification_model) training_accuracy = type(0); 
        
        for(Index iteration = 0; iteration < training_batches_number; iteration++)
        {
            // cout << "Iteration " << iteration << "/" << training_batches_number << endl;

            // Data set
            
            training_batch.fill(training_batches[iteration],
                                input_variable_indices,
                                decoder_variable_indices,
                                target_variable_indices);

            // Neural network
            
            neural_network->forward_propagate(training_batch.get_input_pairs(),
                                              training_forward_propagation,
                                              is_training);

            // cout << "xeee" << endl;

            // Loss index

            loss_index->back_propagate(training_batch,
                                       training_forward_propagation,
                                       training_back_propagation);

            // cout << "xexexexe" << endl;

            // // if(epoch == 50)
            // // {
            // Tensor<type, 1> numerical_gradient = loss_index->calculate_numerical_gradient();

            // // cout << "gradient:\n" << training_back_propagation.gradient << endl;
            // // cerr << "numerical gradient:\n" << numerical_gradient<< endl;
            // // cout << "gradient - numerical gradient :\n" << training_back_propagation.gradient - numerical_gradient << endl;
            // cout << "MHA Gradient - numerical gradient:" << endl;
            // for(Index i = numerical_gradient.size()-4565; i < numerical_gradient.size()-321;i++)
            //     cout << training_back_propagation.gradient(i) - numerical_gradient(i) << " ";

            // throw runtime_error("\nChecking the gradient and numerical gradient.");
            // // }
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
            selection_batches = data_set->get_batches(selection_samples_indices, selection_batch_samples_number, shuffle);
            
            selection_error = type(0);
            if(is_classification_model)    selection_accuracy = type(0);

            for(Index iteration = 0; iteration < selection_batches_number; iteration++)
            {
                // Data set

                selection_batch.fill(selection_batches[iteration],
                                     input_variable_indices,
                                     decoder_variable_indices,
                                     target_variable_indices);

                // Neural network

                neural_network->forward_propagate(selection_batch.get_input_pairs(),
                                                  selection_forward_propagation,
                                                  is_training);
                
                // Loss

                loss_index->calculate_error(selection_batch,
                                            selection_forward_propagation,
                                            selection_back_propagation);
                
                selection_error += selection_back_propagation.error();

                if(is_classification_model) 
                    selection_accuracy += selection_back_propagation.accuracy(0);
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

        // @todo loss and error missmatch

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

    type effective_learning_rate = learning_rate * bias_correction;

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
}


string AdaptiveMomentEstimation::write_optimization_algorithm_type() const
{
    return "ADAPTIVE_MOMENT_ESTIMATION";
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

    set_batch_samples_number(read_xml_index(root_element, "BatchSize"));
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

    const Index parameters_number = neural_network->get_parameters_number();

    gradient_exponential_decay.resize(parameters_number);
    gradient_exponential_decay.setZero();

    square_gradient_exponential_decay.resize(parameters_number);
    square_gradient_exponential_decay.setZero();
}


void AdaptiveMomentEstimationData::print() const
{
    cout << "Gradient exponential decay:" << endl
         << gradient_exponential_decay << endl
         << "Square gradient exponential decay:" << endl
         << square_gradient_exponential_decay << endl;
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

    DataSet* data_set = loss_index->get_data_set();

    if (!data_set)
        throw runtime_error("Data set is null.");

    const bool has_selection = data_set->has_selection();

    const bool is_classification_model = is_instance_of<CrossEntropyError3D>(loss_index);

    const vector<Index> input_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Target);
    const vector<Index> decoder_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Decoder);

    const vector<Index> training_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Training);
    const vector<Index> selection_samples_indices = data_set->get_sample_indices(DataSet::SampleUse::Selection);

    const Index training_samples_number = data_set->get_samples_number(DataSet::SampleUse::Training);
    const Index selection_samples_number = data_set->get_samples_number(DataSet::SampleUse::Selection);

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

    BatchCuda training_batch_cuda(training_batch_samples_number, data_set);
    BatchCuda selection_batch_cuda(selection_batch_samples_number, data_set);

    ForwardPropagationCuda training_forward_propagation_cuda(training_batch_samples_number, neural_network);
    ForwardPropagationCuda selection_forward_propagation_cuda(selection_batch_samples_number, neural_network);

    neural_network->allocate_parameters_device();
    neural_network->copy_parameters_device();
    
    // Loss Index

    loss_index->set_normalization_coefficient();

    BackPropagationCuda training_back_propagation_cuda(training_batch_samples_number, loss_index);
    BackPropagationCuda selection_back_propagation_cuda(selection_batch_samples_number, loss_index);

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

    if(neural_network->has(Layer::Type::Recurrent))
        shuffle = false;

    // Main loop
  
    optimization_data_cuda.iteration = 1;

    for (Index epoch = 0; epoch <= maximum_epochs_number; epoch++)
    { 
        if (display && epoch % display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = data_set->get_batches(training_samples_indices, training_batch_samples_number, shuffle);

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

            neural_network->forward_propagate_cuda(training_batch_cuda.get_input_pairs_device(),
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
            selection_batches = data_set->get_batches(selection_samples_indices, selection_batch_samples_number, shuffle);

            selection_error = type(0);
            if (is_classification_model)    selection_accuracy = type(0);

            for (Index iteration = 0; iteration < selection_batches_number; iteration++)
            {
                // Data set

                selection_batch_cuda.fill(selection_batches[iteration],
                                          input_variable_indices,
                                          decoder_variable_indices,
                                          target_variable_indices);

                // Neural network

                neural_network->forward_propagate_cuda(selection_batch_cuda.get_input_pairs_device(),
                                                       selection_forward_propagation_cuda,
                                                       is_training);

                // Loss

                loss_index->calculate_error_cuda(selection_batch_cuda,
                                                 selection_forward_propagation_cuda,
                                                 selection_back_propagation_cuda);

                selection_error += selection_back_propagation_cuda.error();

                if (is_classification_model)    
                    selection_accuracy += selection_back_propagation_cuda.accuracy();
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

        // @todo loss and error missmatch

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

    if (display) results.print();

    return results;
    
}


void AdaptiveMomentEstimation::update_parameters_cuda(BackPropagationCuda& back_propagation_cuda,
                                                       ADAMOptimizationDataCuda& optimization_data_cuda) const
{
    NeuralNetwork* neural_network = loss_index->get_neural_network();

    Index& iteration = optimization_data_cuda.iteration;

    const type bias_correction =
        sqrt(type(1) - pow(beta_2, type(iteration))) /
        (type(1) - pow(beta_1, type(iteration)));

    const Index parameters_number = optimization_data_cuda.adaptive_moment_estimation->get_loss_index()->get_neural_network()->get_parameters_number();

    type* ones = optimization_data_cuda.ones;

    type* gradient = back_propagation_cuda.gradient;

    type* gradient_exponential_decay = optimization_data_cuda.gradient_exponential_decay;

    type* square_gradient = optimization_data_cuda.square_gradient;

    type* square_gradient_exponential_decay = optimization_data_cuda.square_gradient_exponential_decay;

    type* parameters = back_propagation_cuda.parameters;

    const cudnnTensorDescriptor_t& gradient_tensor_descriptor = back_propagation_cuda.gradient_tensor_descriptor;

    const cudnnOpTensorDescriptor_t& operator_sum_descriptor = back_propagation_cuda.operator_sum_descriptor;

    const cudnnOpTensorDescriptor_t& operator_multiplication_descriptor = back_propagation_cuda.operator_multiplication_descriptor;

    const cudnnOpTensorDescriptor_t& operator_square_root_descriptor = back_propagation_cuda.operator_square_root_descriptor;

    // Gradients

    float alpha = 1.0f;
    const float beta = 0.0f;

    cudnnOpTensor(cudnn_handle,
        operator_multiplication_descriptor,
        &alpha,
        gradient_tensor_descriptor,
        gradient,
        &alpha,
        gradient_tensor_descriptor,
        gradient,
        &beta,
        gradient_tensor_descriptor,
        square_gradient);

    alpha = (type(1) - beta_2);

    cublasSscal(cublas_handle, parameters_number, &alpha, square_gradient, 1);

    alpha = (type(1) - beta_1);

    cublasSscal(cublas_handle, parameters_number, &alpha, gradient, 1);

    alpha = beta_1;

    cublasSscal(cublas_handle, parameters_number, &alpha, gradient_exponential_decay, 1);

    alpha = beta_2;

    cublasSscal(cublas_handle, parameters_number, &alpha, square_gradient_exponential_decay, 1);

    alpha = 1.0f;

    cublasSaxpy(cublas_handle, parameters_number, &alpha, gradient, 1, gradient_exponential_decay, 1);

    cublasSaxpy(cublas_handle, parameters_number, &alpha, square_gradient, 1, square_gradient_exponential_decay, 1);

    // Parameters

    // Numerator -> (learning_rate * bias_correction) * gradient_exponential_decay

    float* numerator = optimization_data_cuda.numerator;

    cudaMemcpy(numerator, gradient_exponential_decay, parameters_number * sizeof(float), cudaMemcpyDeviceToDevice);

    if (!use_custom_learning_rate)
    {
        alpha = bias_correction * learning_rate;

        cublasSscal(cublas_handle, parameters_number, &alpha, numerator, 1);
    }
    else
    {
        const type warmup_steps = 4000;
        type& step = optimization_data_cuda.step;

        alpha = (learning_rate * min(pow(step, -0.5), step * pow(warmup_steps, -1.5))) * bias_correction;

        cublasSscal(cublas_handle, parameters_number, &alpha, numerator, 1);

        step++;
    }

    // Denominator -> (square_gradient_exponential_decay.sqrt() + epsilon)

    float* denominator = optimization_data_cuda.denominator;

    alpha = 1.0f;

    cudnnOpTensor(cudnn_handle,
        operator_square_root_descriptor,
        &alpha,
        gradient_tensor_descriptor,
        square_gradient_exponential_decay,
        &alpha,
        gradient_tensor_descriptor,
        square_gradient_exponential_decay,
        &beta,
        gradient_tensor_descriptor,
        denominator);

    alpha = epsilon;
    
    cublasSaxpy(cublas_handle, parameters_number, &alpha, ones, 1, denominator, 1);

    // parameters -= numerator / denominator

    divide_subtract(parameters_number, parameters, numerator, denominator);

    optimization_data_cuda.iteration++;

    // Update parameters

    neural_network->set_parameters_cuda(parameters);
}


ADAMOptimizationDataCuda::ADAMOptimizationDataCuda(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    set(new_adaptive_moment_estimation);
}


void ADAMOptimizationDataCuda::set(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    adaptive_moment_estimation = new_adaptive_moment_estimation;

    const Index parameters_number = adaptive_moment_estimation->get_loss_index()->get_neural_network()->get_parameters_number();

    // Gradient

    if (cudaMalloc(&square_gradient, parameters_number * sizeof(float)) != cudaSuccess)
        cout << "Square gradient allocation error" << endl;

    if (cudaMalloc(&gradient_exponential_decay, parameters_number * sizeof(float)) != cudaSuccess)
        cout << "gradient_exponential_decay allocation error" << endl;

    if (cudaMalloc(&square_gradient_exponential_decay, parameters_number * sizeof(float)) != cudaSuccess)
        cout << "square_gradient_exponential_decay allocation error" << endl;

    if (cudaMalloc(&last_gradient_exponential_decay, parameters_number * sizeof(float)) != cudaSuccess)
        cout << "last_gradient_exponential_decay allocation error" << endl;

    if (cudaMalloc(&last_square_gradient_exponential_decay, parameters_number * sizeof(float)) != cudaSuccess)
        cout << "last_square_gradient_exponential_decay allocation error" << endl;

    if (cudaMalloc(&numerator, parameters_number * sizeof(float)) != cudaSuccess)
        cout << "numerator allocation error" << endl;

    if (cudaMalloc(&denominator, parameters_number * sizeof(float)) != cudaSuccess)
        cout << "denominator allocation error" << endl;

    // Aux ones

    if (cudaMalloc(&ones, parameters_number * sizeof(float)) != cudaSuccess)
        cout << "aux ones allocation error" << endl;

    for (Index i = 0; i < parameters_number; i++)
        cudaMemcpy(ones + i, &one, sizeof(float), cudaMemcpyHostToDevice);
}


void ADAMOptimizationDataCuda::free()
{
    cudaFree(square_gradient);
    cudaFree(gradient_exponential_decay);
    cudaFree(square_gradient_exponential_decay);
    cudaFree(last_gradient_exponential_decay);
    cudaFree(last_square_gradient_exponential_decay);
    cudaFree(numerator);
    cudaFree(denominator);
    cudaFree(ones);
}


void ADAMOptimizationDataCuda::print() const
{
    const Index parameters_number = adaptive_moment_estimation->get_loss_index()->get_neural_network()->get_parameters_number();

    cout << "gradient_exponential_decay_host:" << endl;
    cout << vector_from_device(gradient_exponential_decay, parameters_number) << endl;

    cout << "square_gradient_exponential_decay_host:" << endl;
    cout << vector_from_device(square_gradient_exponential_decay, parameters_number) << endl;
}

#endif

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
