//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "loss.h"
#include "adaptive_moment_estimation.h"

namespace opennn
{

AdaptiveMomentEstimation::AdaptiveMomentEstimation(Loss* new_loss)
    : Optimizer(new_loss)
{
    set_default();
}


Index AdaptiveMomentEstimation::get_samples_number() const
{
    return batch_size;
}


void AdaptiveMomentEstimation::set_batch_size(const Index new_batch_size)
{
    batch_size = new_batch_size;
}


void AdaptiveMomentEstimation::set_beta_1(const type new_beta_1)
{
    beta_1 = new_beta_1;
}


void AdaptiveMomentEstimation::set_beta_2(const type new_beta_2)
{
    beta_2 = new_beta_2;
}


void AdaptiveMomentEstimation::set_default()
{
    display_period = 100;
    name = "AdaptiveMomentEstimation";
}


void AdaptiveMomentEstimation::set_learning_rate(const type new_learning_rate)
{
    learning_rate = new_learning_rate;
}



TrainingResults AdaptiveMomentEstimation::train()
{
    if(!loss || !loss->get_neural_network() || !loss->get_dataset())
        return TrainingResults();

    TrainingResults results(maximum_epochs + 1);

    check();

    if(display) cout << "Training with adaptive moment estimation \"Adam\" ..." << endl;

    // Dataset

    Dataset* dataset = loss->get_dataset();

    if(!dataset)
        throw runtime_error("Dataset is null.");

    const bool has_validation = dataset->has_validation();

    const bool is_text_classification_model = false/*is_instance_of<CrossEntropyError3d>(loss)*/;

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");
    const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");

    const vector<Index> training_sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> validation_sample_indices = dataset->get_sample_indices("Validation");

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const Index training_batch_size = min(training_samples_number, batch_size);

    const Index validation_batch_size = (validation_samples_number != 0)
        ? min(validation_samples_number, batch_size)
        : 0;

    const Index training_batches_number = (training_batch_size != 0)
        ? training_samples_number / training_batch_size
        : 0;

    const Index validation_batches_number = (validation_batch_size != 0)
       ? validation_samples_number / validation_batch_size
       : 0;

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> validation_batches(validation_batches_number);

    // Neural network

    NeuralNetwork* neural_network = loss->get_neural_network();

    set_names();
    set_scaling();

    Batch training_batch(training_batch_size, dataset);
    Batch validation_batch(validation_batch_size, dataset);

    ForwardPropagation training_forward_propagation(training_batch_size, neural_network);
    ForwardPropagation validation_forward_propagation(validation_batch_size, neural_network);

    // Loss index

    loss->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_size, loss);
    BackPropagation validation_back_propagation(validation_batch_size, loss);

    type training_error = type(0);
    type training_accuracy = type(0);

    type validation_error = type(0);
    type validation_accuracy = type(0);

    Index validation_failures = 0;

    // Optimization algorithm

    AdaptiveMomentEstimationData optimization_data(this);

    bool stop_training = false;
    constexpr bool is_training = true;

    time_t beginning_time;
    time(&beginning_time);

    type elapsed_time = type(0);

    const bool shuffle = !neural_network->has("Recurrent");

    // Main loop
    optimization_data.iteration = 1;

    for(Index epoch = 0; epoch <= maximum_epochs; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;
        
        training_batches = dataset->get_batches(training_sample_indices, training_batch_size, shuffle);
        
        training_error = type(0);

        if(is_text_classification_model) training_accuracy = type(0);

        for(Index iteration = 0; iteration < training_batches_number; iteration++)
        {
            training_back_propagation.gradient.setZero();

            // Dataset

            training_batch.fill(training_batches[iteration],
                                input_feature_indices,
                                decoder_feature_indices,
                                target_feature_indices);

            // Neural network

            neural_network->forward_propagate(training_batch.get_inputs(),
                                              training_forward_propagation,
                                              is_training);

            // Loss index

            loss->back_propagate(training_batch,
                                       training_forward_propagation,
                                       training_back_propagation);

            training_error += training_back_propagation.error;

            if(is_text_classification_model) training_accuracy += training_back_propagation.accuracy(0);

            update_parameters(training_back_propagation, optimization_data);
        }

        // Loss

        training_error /= type(training_batches_number);
        if(is_text_classification_model)
            training_accuracy /= type(training_batches_number);

        results.training_error_history(epoch) = training_error;

        if(has_validation)
        {
            validation_batches = dataset->get_batches(validation_sample_indices, validation_batch_size, shuffle);

            validation_error = type(0);

            if(is_text_classification_model)
                validation_accuracy = type(0);

            for(Index iteration = 0; iteration < validation_batches_number; iteration++)
            {
                // Dataset

                validation_batch.fill(validation_batches[iteration],
                                     input_feature_indices,
                                     decoder_feature_indices,
                                     target_feature_indices);

                // Neural network

                neural_network->forward_propagate(validation_batch.get_inputs(),
                                                  validation_forward_propagation,
                                                  is_training);

                // Loss

                loss->calculate_error(validation_batch,
                                            validation_forward_propagation,
                                            validation_back_propagation);

                validation_error += validation_back_propagation.error;

                if(is_text_classification_model)
                    validation_accuracy += validation_back_propagation.accuracy(0);
            }

            validation_error /= type(validation_batches_number);
            if(is_text_classification_model) validation_accuracy /= type(validation_batches_number);

            results.validation_error_history(epoch) = validation_error;

            if(epoch != 0 && results.validation_error_history(epoch) > results.validation_error_history(epoch-1)) validation_failures++;
        }

        // Elapsed time

        elapsed_time = get_elapsed_time(beginning_time);

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if(is_text_classification_model) cout << "Training accuracy: " << training_accuracy << endl;
            if(has_validation) cout << "Validation error: " << validation_error << endl;
            if(has_validation && is_text_classification_model) cout << "Validation accuracy: " << validation_accuracy << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        stop_training = check_stopping_condition(results, epoch, elapsed_time,
                                                  results.training_error_history(epoch),
                                                  validation_failures);

        if(stop_training)
        {
            results.loss = training_back_propagation.loss_value;

            results.validation_failures = validation_failures;

            results.resize_training_error_history(epoch+1);

            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);

            results.elapsed_time = write_time(elapsed_time);

            break;
        }

    }

    set_unscaling();

    if(display) results.print();

    return results;
}


void AdaptiveMomentEstimation::update_parameters(BackPropagation& back_propagation,
                                                 AdaptiveMomentEstimationData& optimization_data) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    optimization_data.iteration++;
    const type iteration = static_cast<type>(optimization_data.iteration);

    const type bias_correction_1 = type(1) - pow(beta_1, iteration);
    const type bias_correction_2 = type(1) - pow(beta_2, iteration);

    VectorR& parameters = neural_network->get_parameters();

    VectorR& gradient_exponential_decay = optimization_data.gradient_exponential_decay;
    VectorR& square_gradient_exponential_decay = optimization_data.square_gradient_exponential_decay;

    const VectorR& gradient = back_propagation.gradient;

    gradient_exponential_decay.array() = beta_1 * gradient_exponential_decay.array() + (type(1) - beta_1) * gradient.array();
    square_gradient_exponential_decay.array() = beta_2 * square_gradient_exponential_decay.array() + (type(1) - beta_2) * gradient.array().square();

    parameters.array() -= learning_rate * (gradient_exponential_decay.array() / bias_correction_1) /
                          ((square_gradient_exponential_decay.array() / bias_correction_2).sqrt() + EPSILON);
}


void AdaptiveMomentEstimation::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("AdaptiveMomentEstimation");

    add_xml_element(printer, "BatchSize", to_string(batch_size));
    write_common_xml(printer);

    printer.CloseElement();
}


void AdaptiveMomentEstimation::from_XML(const XMLDocument& document)
{

    const XMLElement* root_element = get_xml_root(document, "AdaptiveMomentEstimation");

    set_batch_size(read_xml_index(root_element, "BatchSize"));
    read_common_xml(root_element);
}


AdaptiveMomentEstimationData::AdaptiveMomentEstimationData(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    set(new_adaptive_moment_estimation);
}


void AdaptiveMomentEstimationData::set(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    adaptive_moment_estimation = new_adaptive_moment_estimation;

    Loss* loss = new_adaptive_moment_estimation->get_loss();
    NeuralNetwork* neural_network = loss->get_neural_network();

    const Index parameters_number = neural_network->get_parameters().size();

    gradient_exponential_decay.resize(parameters_number);
    gradient_exponential_decay.setZero();

    square_gradient_exponential_decay.resize(parameters_number);
    square_gradient_exponential_decay.setZero();
}


void AdaptiveMomentEstimationData::print() const
{
    // cout << "Gradient exponential decay:" << endl
    //      << gradient_exponential_decay << endl
    //      << "Square gradient exponential decay:" << endl
    //      << square_gradient_exponential_decay << endl;
}


#ifdef CUDA

TrainingResults AdaptiveMomentEstimation::train_cuda()
{
    if(!loss || !loss->get_neural_network() || !loss->get_dataset())
        return TrainingResults();

    TrainingResults results(maximum_epochs + 1);
    
    check();

    if (display) cout << "Training with adaptive moment estimation \"Adam\" CUDA ...\n";

    Dataset* dataset = loss->get_dataset();
    if(!dataset) throw runtime_error("Dataset is null.");

    const bool has_validation = dataset->has_validation();
    const bool is_text_classification_model = is_instance_of<CrossEntropyError3d>(loss);

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    const vector<Index> training_sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> validation_sample_indices = dataset->get_sample_indices("Validation");

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const Index training_batch_size = min(training_samples_number, batch_size);
    const Index validation_batch_size = (validation_samples_number != 0) ? min(validation_samples_number, batch_size) : 0;

    const Index training_batches_number = (training_batch_size != 0) ? training_samples_number / training_batch_size : 0;
    const Index validation_batches_number = (validation_batch_size != 0) ? validation_samples_number / validation_batch_size : 0;

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> validation_batches(validation_batches_number);

    NeuralNetwork* neural_network = loss->get_neural_network();

    set_names();
    set_scaling();

    const int PREFETCH_BATCHES = 3;
    
    ThreadSafeQueue<BatchCuda*> empty_training_queue;
    ThreadSafeQueue<BatchCuda*> ready_training_queue;
    vector<unique_ptr<BatchCuda>> training_batch_pool;

    for (int i = 0; i < PREFETCH_BATCHES; i++) 
    {
        training_batch_pool.push_back(make_unique<BatchCuda>(training_batch_size, dataset));
        empty_training_queue.push(training_batch_pool.back().get());
    }

    ThreadSafeQueue<BatchCuda*> empty_validation_queue;
    ThreadSafeQueue<BatchCuda*> ready_validation_queue;
    vector<unique_ptr<BatchCuda>> validation_batch_pool;

    if (has_validation) 
    {
        for (int i = 0; i < PREFETCH_BATCHES; i++) 
        {
            validation_batch_pool.push_back(make_unique<BatchCuda>(validation_batch_size, dataset));
            empty_validation_queue.push(validation_batch_pool.back().get());
        }
    }

    cudaStream_t memory_stream;
    cudaStreamCreate(&memory_stream);
    cudaEvent_t batch_ready_event;
    cudaEventCreate(&batch_ready_event);

    ForwardPropagationCuda training_forward_propagation(training_batch_size, neural_network);
    unique_ptr<ForwardPropagationCuda> validation_forward_propagation;

    neural_network->copy_parameters_device();
    loss->set_normalization_coefficient();

    BackPropagationCuda training_back_propagation(training_batch_size, loss);
    unique_ptr<BackPropagationCuda> validation_back_propagation;

    if (has_validation) 
    {
        validation_forward_propagation = make_unique<ForwardPropagationCuda>(validation_batch_size, neural_network);
        validation_back_propagation = make_unique<BackPropagationCuda>(validation_batch_size, loss);
    }

    type training_error = type(0);
    type training_accuracy = type(0);
    type validation_error = type(0);
    type validation_accuracy = type(0);
    Index validation_failures = 0;

    ADAMOptimizationDataCuda optimization_data(this);

    bool stop_training = false;
    constexpr bool is_training = true;
    bool shuffle = true;

    if(neural_network->has("Recurrent")) shuffle = false;

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time = type(0);
    optimization_data.iteration = 1;

    for(Index epoch = 0; epoch <= maximum_epochs; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = dataset->get_batches(training_sample_indices, training_batch_size, shuffle);
        training_error = type(0);
        if (is_text_classification_model) training_accuracy = type(0);
        
        std::thread training_worker([&]() 
        {
            for(Index iteration = 0; iteration < training_batches_number; iteration++) 
            {
                BatchCuda* batch = empty_training_queue.pop();
                batch->fill_host(training_batches[iteration],
                                 input_feature_indices,
                                 decoder_feature_indices,
                                 target_feature_indices);
                ready_training_queue.push(batch);
            }
        });

        for(Index iteration = 0; iteration < training_batches_number; iteration++)
        {
            training_back_propagation.neural_network.gradients.fill(0.0f);

            BatchCuda* current_batch = ready_training_queue.pop();

            current_batch->copy_device_async(training_batches[iteration].size(), memory_stream);
            cudaEventRecord(batch_ready_event, memory_stream);
            cudaStreamWaitEvent(0, batch_ready_event, 0);

            neural_network->forward_propagate(current_batch->get_inputs_device(), training_forward_propagation, is_training);

            loss->back_propagate(*current_batch, training_forward_propagation, training_back_propagation);

            training_error += training_back_propagation.error;

            if (is_text_classification_model)
                training_accuracy += training_back_propagation.accuracy();
            
            update_parameters(training_back_propagation, optimization_data);

            cudaStreamSynchronize(0);

            empty_training_queue.push(current_batch);
        }

        training_worker.join();

        training_error /= type(training_batches_number);
        if (is_text_classification_model) training_accuracy /= type(training_batches_number);
        results.training_error_history(epoch) = training_error;

        if (has_validation)
        {
            validation_batches = dataset->get_batches(validation_sample_indices, validation_batch_size, shuffle);
            validation_error = type(0);
            if (is_text_classification_model) validation_accuracy = type(0);

            std::thread validation_worker([&]() 
            {
                for(Index iteration = 0; iteration < validation_batches_number; iteration++) 
                {
                    BatchCuda* batch = empty_validation_queue.pop();
                    batch->fill_host(validation_batches[iteration],
                                     input_feature_indices,
                                     decoder_feature_indices,
                                     target_feature_indices);
                    ready_validation_queue.push(batch);
                }
            });

            for(Index iteration = 0; iteration < validation_batches_number; iteration++)
            {
                BatchCuda* current_batch = ready_validation_queue.pop();

                current_batch->copy_device_async(validation_batches[iteration].size(), memory_stream);
                cudaEventRecord(batch_ready_event, memory_stream);
                cudaStreamWaitEvent(0, batch_ready_event, 0);

                neural_network->forward_propagate(current_batch->get_inputs_device(), *validation_forward_propagation, is_training);
                loss->calculate_error(*current_batch, *validation_forward_propagation, *validation_back_propagation);

                validation_error += validation_back_propagation->error;
                if (is_text_classification_model) validation_accuracy += validation_back_propagation->accuracy();

                cudaStreamSynchronize(0);

                empty_validation_queue.push(current_batch);
            }

            validation_worker.join();

            validation_error /= type(validation_batches_number);
            if (is_text_classification_model) validation_accuracy /= type(validation_batches_number);
            results.validation_error_history(epoch) = validation_error;

            if (epoch != 0 && results.validation_error_history(epoch) > results.validation_error_history(epoch - 1)) 
                validation_failures++;
        }

        elapsed_time = get_elapsed_time(beginning_time);

        if (display && epoch % display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if (is_text_classification_model) cout << "Training accuracy: " << training_accuracy << endl;
            if (has_validation) cout << "Validation error: " << validation_error << endl;
            if (has_validation && is_text_classification_model) cout << "Validation accuracy: " << validation_accuracy << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        stop_training = check_stopping_condition(results, epoch, elapsed_time,
                                                  results.training_error_history(epoch),
                                                  validation_failures);

        if (stop_training) 
        {
            results.loss = training_back_propagation.loss_value;
            results.validation_failures = validation_failures;
            results.resize_training_error_history(epoch + 1);
            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);
            results.elapsed_time = write_time(elapsed_time);
            break;
        }

    }

    cudaStreamDestroy(memory_stream);
    cudaEventDestroy(batch_ready_event);

    neural_network->copy_parameters_host();
    set_unscaling();

    if (display) results.print();

    return results;
}


void AdaptiveMomentEstimation::update_parameters(BackPropagationCuda& back_propagation,
                                                 ADAMOptimizationDataCuda& optimization_data) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_device().size();

    float* parameters_device_data = neural_network->get_parameters_device().data;
    const float* gradients_device = back_propagation.neural_network.gradients.data;

    optimization_data.iteration++;
    const int iteration = static_cast<int>(optimization_data.iteration);

    const float bias_correction_1 = 1.0f - powf(beta_1, static_cast<float>(iteration));
    const float bias_correction_2 = 1.0f - powf(beta_2, static_cast<float>(iteration));

    adam_update_device(
        parameters_number,
        parameters_device_data,
        optimization_data.gradient_exponential_decay.data,
        optimization_data.square_gradient_exponential_decay.data,
        gradients_device,
        beta_1,
        beta_2,
        learning_rate,
        EPSILON,
        bias_correction_1,
        bias_correction_2);
}


ADAMOptimizationDataCuda::ADAMOptimizationDataCuda(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    set(new_adaptive_moment_estimation);
}


void ADAMOptimizationDataCuda::set(AdaptiveMomentEstimation* new_adaptive_moment_estimation)
{
    adaptive_moment_estimation = new_adaptive_moment_estimation;

    NeuralNetwork* neural_network = adaptive_moment_estimation->get_loss()->get_neural_network();
    const Index parameters_number = neural_network->get_parameters().size();

    gradient_exponential_decay.resize({parameters_number});
    square_gradient_exponential_decay.resize({parameters_number});

    CHECK_CUDA(cudaMemset(gradient_exponential_decay.data, 0, parameters_number * sizeof(float)));
    CHECK_CUDA(cudaMemset(square_gradient_exponential_decay.data, 0, parameters_number * sizeof(float)));
}


void ADAMOptimizationDataCuda::print() const
{
    cout << "--- ADAM Optimization Data (CUDA) ---" << endl;

    NeuralNetwork* neural_network = adaptive_moment_estimation->get_loss()->get_neural_network();

    cout << "-----------------------------------" << endl;
}

#endif

REGISTER(Optimizer, AdaptiveMomentEstimation, "AdaptiveMomentEstimation");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
