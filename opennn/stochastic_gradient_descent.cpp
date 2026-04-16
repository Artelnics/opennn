//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "neural_network.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "loss.h"
#include "batch.h"
#include "stochastic_gradient_descent.h"

namespace opennn
{

StochasticGradientDescent::StochasticGradientDescent(Loss* new_loss)
    : Optimizer(new_loss)
{
    set_default();
}

void StochasticGradientDescent::set_default()
{
    name = "StochasticGradientDescent";

    // TRAINING OPERATORS

    initial_learning_rate = type(0.001);
    initial_decay = type(0.001);
    momentum = type(0);
    nesterov = false;

    // Stopping criteria

    training_loss_goal = type(0);
    maximum_time = type(3600);
    maximum_epochs = 1000;

    // UTILITIES

    display_period = 100;
}

void StochasticGradientDescent::set_batch_size(const Index new_batch_size)
{
    batch_size = new_batch_size;
}

Index StochasticGradientDescent::get_samples_number() const
{
    return batch_size;
}

void StochasticGradientDescent::set_initial_learning_rate(const type new_learning_rate)
{
    initial_learning_rate = new_learning_rate;
}

void StochasticGradientDescent::set_initial_decay(const type new_decay)
{
    initial_decay = new_decay;
}

void StochasticGradientDescent::set_momentum(const type new_momentum)
{
    momentum = new_momentum;
}

void StochasticGradientDescent::set_nesterov(bool new_nesterov_momentum)
{
    nesterov = new_nesterov_momentum;
}

void StochasticGradientDescent::update_parameters(BackPropagation& back_propagation,
                                                  StochasticGradientDescentData& optimization_data,
                                                  type current_learning_rate) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
    {
        const Index parameters_number = neural_network->get_parameters_size();

        sgd_update_device(
            parameters_number,
            neural_network->get_parameters_device(),
            optimization_data.parameter_updates.device(),
            back_propagation.gradient.device(),
            current_learning_rate,
            momentum,
            nesterov);
        return;
    }
#endif

    VectorR& parameters = neural_network->get_parameters();

    const VectorR& gradient = back_propagation.gradient.vector;

    VectorR& parameter_updates = optimization_data.parameter_updates.vector;
    VectorR& last_parameter_updates = optimization_data.last_parameter_updates.vector;

    parameter_updates = gradient * (-current_learning_rate);

    if (momentum > type(0))
    {
        parameter_updates += momentum * last_parameter_updates;
        last_parameter_updates = parameter_updates;
    }

    parameters += nesterov
        ? parameter_updates * momentum - gradient * current_learning_rate
        : parameter_updates;
}

TrainingResults StochasticGradientDescent::train()
{
    if(!loss || !loss->get_neural_network() || !loss->get_dataset())
        return TrainingResults();

    TrainingResults results(maximum_epochs+1);

    check();

    // Start training

    if(display) cout << "Training with stochastic gradient descent (SGD)...\n";

    // Dataset

    const Dataset* dataset = loss->get_dataset();

    const bool has_validation = dataset->has_validation();

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

    Batch training_batch(training_batch_size, dataset);
    Batch validation_batch(validation_batch_size, dataset);

    const Index training_batches_number = (training_batch_size != 0)
                                              ? training_samples_number / training_batch_size
                                              : 0;

    const Index validation_batches_number = (validation_batch_size != 0)
                                               ? validation_samples_number / validation_batch_size
                                               : 0;

    vector<vector<Index>> training_batches(training_batches_number);

    vector<Index> const training_batch_indices(training_batch_size);
    vector<Index> const selection_batch_indices(training_batch_size);

    // Neural network

    NeuralNetwork* neural_network = loss->get_neural_network();

    set_names();

    set_scaling();

    ForwardPropagation training_forward_propagation(training_batch_size, neural_network);
    ForwardPropagation validation_forward_propagation(validation_batch_size, neural_network);

    // Loss index

    loss->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_size, loss);
    BackPropagation validation_back_propagation(validation_batch_size, loss);

    //type training_loss = type(0);
    type training_error = type(0);
    type validation_error = type(0);

    Index validation_failures = 0;

    // Optimization algorithm

    StochasticGradientDescentData optimization_data(this);

    bool stop_training = false;
    constexpr bool is_training = true;

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    const bool shuffle = !neural_network->has(LayerType::Recurrent);

    vector<vector<Index>> validation_batches;

    // Main loop

    for(Index epoch = 0; epoch <= maximum_epochs; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << "\n";

        dataset->get_batches(training_sample_indices, training_batch_size, shuffle, training_batches);

        const type current_learning_rate = initial_learning_rate / (type(1) + type(epoch) * initial_decay);

        //training_loss = type(0);
        training_error = type(0);

        optimization_data.iteration = 0;

        for(Index iteration = 0; iteration < training_batches_number; iteration++)
        {
            optimization_data.iteration++;

            // Dataset

            training_batch.fill(training_batches[iteration],
                                input_feature_indices,
                                decoder_feature_indices,
                                target_feature_indices,
                                true);

            // Neural network

            neural_network->forward_propagate(training_batch.get_inputs(),
                                              training_forward_propagation,
                                              is_training);

            // Loss index

            loss->back_propagate(training_batch,
                                       training_forward_propagation,
                                       training_back_propagation);

            results.training_error_history(epoch) = training_back_propagation.error;

            training_error += training_back_propagation.error;
            //training_loss += training_back_propagation.loss;

            // Gradient

            update_parameters(training_back_propagation, optimization_data, current_learning_rate);

            //if(display && epoch % display_period == 0)      display_progress_bar(iteration, batches_number - 1);
        }

        // Loss

        training_error /= type(training_batches_number);

        results.training_error_history(epoch) = training_error;

        if(has_validation)
        {
            dataset->get_batches(validation_sample_indices, validation_batch_size, shuffle, validation_batches);

            validation_error = type(0);

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
                                                  false);

                // Loss

                loss->calculate_error(validation_batch,
                                            validation_forward_propagation,
                                            validation_back_propagation);

                validation_error += validation_back_propagation.error;
            }

            validation_error /= type(validation_batches_number);

            results.validation_error_history(epoch) = validation_error;

            if(epoch != 0 && results.validation_error_history(epoch) > results.validation_error_history(epoch-1)) validation_failures++;
        }

        // Elapsed time

        elapsed_time = get_elapsed_time(beginning_time);

        if(display && epoch%display_period == 0)
        {
            cout << "Training error: " << training_error << "\n";
            if(has_validation) cout << "Validation error: " << validation_error << "\n"<<endl;
            cout << "Elapsed time: " << get_time(elapsed_time) << "\n";
        }

        // Stopping criteria

        stop_training = check_stopping_condition(results, epoch, elapsed_time,
                                                  results.training_error_history(epoch),
                                                  validation_failures);

        if(stop_training)
        {
            results.loss = training_back_propagation.loss_value;
            results.validation_failures = validation_failures;
            results.elapsed_time = get_time(elapsed_time);

            results.resize_training_error_history(epoch+1);
            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);

            break;
        }

        // Update stuff

    }

    set_unscaling();

    if(display) results.print();

    return results;
}

void StochasticGradientDescent::to_XML(XmlPrinter& printer) const
{
    printer.open_element("StochasticGradientDescent");

    add_xml_element(printer, "BatchSize", to_string(batch_size));
    add_xml_element(printer, "ApplyMomentum", to_string(momentum > type(0)));
    write_common_xml(printer);

    printer.close_element();
}

void StochasticGradientDescent::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "StochasticGradientDescent");

    set_batch_size(read_xml_index(root_element, "BatchSize"));

    const bool apply_momentum = read_xml_bool(root_element, "ApplyMomentum");
    set_momentum(apply_momentum ? type(0.9) : type(0));

    read_common_xml(root_element);
}

StochasticGradientDescentData::StochasticGradientDescentData(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    set(new_stochastic_gradient_descent);
}

void StochasticGradientDescentData::set(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    stochastic_gradient_descent = new_stochastic_gradient_descent;

    const Loss* loss = stochastic_gradient_descent->get_loss();

    const NeuralNetwork* neural_network = loss->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_size();

    parameter_updates.resize(parameters_number);
    parameter_updates.setZero();

    last_parameter_updates.resize(parameters_number);
    last_parameter_updates.setZero();

#ifdef OPENNN_WITH_CUDA
    parameter_updates.resize_device(parameters_number);
    parameter_updates.setZero_device();
    last_parameter_updates.resize_device(parameters_number);
    last_parameter_updates.setZero_device();
#endif
}

#ifdef OPENNN_WITH_CUDA

TrainingResults StochasticGradientDescent::train_cuda()
{
    if(!loss || !loss->get_neural_network() || !loss->get_dataset())
        return TrainingResults();

    TrainingResults results(maximum_epochs + 1);

    check();

    if(display) cout << "Training with stochastic gradient descent (SGD) CUDA..." << "\n";

    Dataset* dataset = loss->get_dataset();

    const bool has_validation = dataset->has_validation();

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");
    const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");

    const vector<Index> training_sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> validation_sample_indices = dataset->get_sample_indices("Validation");

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const Index training_batch_size = min(training_samples_number, batch_size);
    const Index validation_batch_size = (validation_samples_number != 0)
        ? min(validation_samples_number, batch_size) : 0;

    const Index training_batches_number = (training_batch_size != 0)
        ? training_samples_number / training_batch_size : 0;
    const Index validation_batches_number = (validation_batch_size != 0)
        ? validation_samples_number / validation_batch_size : 0;

    vector<vector<Index>> training_batches(training_batches_number);

    NeuralNetwork* neural_network = loss->get_neural_network();

    set_names();
    set_scaling();


    neural_network->copy_parameters_device();
    neural_network->link_parameters_device();
    neural_network->copy_states_device();
    neural_network->link_states_device();

    const int PREFETCH_BATCHES = 3;

    ThreadSafeQueue<Batch*> empty_training_queue;
    ThreadSafeQueue<Batch*> ready_training_queue;
    vector<unique_ptr<Batch>> training_batch_pool;

    for(int i = 0; i < PREFETCH_BATCHES; i++)
    {
        training_batch_pool.push_back(make_unique<Batch>(training_batch_size, dataset));
        empty_training_queue.push(training_batch_pool.back().get());
    }

    ThreadSafeQueue<Batch*> empty_validation_queue;
    ThreadSafeQueue<Batch*> ready_validation_queue;
    vector<unique_ptr<Batch>> validation_batch_pool;

    if(has_validation)
    {
        for(int i = 0; i < PREFETCH_BATCHES; i++)
        {
            validation_batch_pool.push_back(make_unique<Batch>(validation_batch_size, dataset));
            empty_validation_queue.push(validation_batch_pool.back().get());
        }
    }

    cudaStream_t memory_stream;
    cudaStreamCreate(&memory_stream);
    cudaEvent_t batch_ready_event[2];
    cudaEventCreate(&batch_ready_event[0]);
    cudaEventCreate(&batch_ready_event[1]);

    ForwardPropagation training_forward_propagation(training_batch_size, neural_network);
    training_forward_propagation.allocate_device();

    loss->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_size, loss);
    training_back_propagation.allocate_device();

    ForwardPropagation validation_forward_propagation(validation_batch_size, neural_network);
    validation_forward_propagation.allocate_device();

    BackPropagation validation_back_propagation(validation_batch_size, loss);
    validation_back_propagation.allocate_device();

    StochasticGradientDescentData optimization_data(this);

    type training_error = type(0);
    type validation_error = type(0);
    Index validation_failures = 0;

    bool stop_training = false;
    constexpr bool is_training = true;
    const bool shuffle = !neural_network->has(LayerType::Recurrent);

    vector<vector<Index>> validation_batches;

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time = type(0);
    optimization_data.iteration = 1;

    for(Index epoch = 0; epoch <= maximum_epochs; epoch++)
    {
        if(display && epoch % display_period == 0) cout << "Epoch: " << epoch << "\n";

        dataset->get_batches(training_sample_indices, training_batch_size, shuffle, training_batches);

        const type current_learning_rate = initial_learning_rate / (type(1) + type(epoch) * initial_decay);

        training_error = type(0);

        std::thread training_worker([&]()
        {
            for(Index iteration = 0; iteration < training_batches_number; iteration++)
            {
                Batch* batch = empty_training_queue.pop();
                batch->fill_host(training_batches[iteration],
                                 input_feature_indices,
                                 decoder_feature_indices,
                                 target_feature_indices);
                ready_training_queue.push(batch);
            }
        });

        Batch* next_batch = nullptr;
        if(training_batches_number > 0)
        {
            next_batch = ready_training_queue.pop();
            next_batch->copy_device_async(training_batches[0].size(), memory_stream);
            cudaEventRecord(batch_ready_event[0], memory_stream);
        }

        for(Index iteration = 0; iteration < training_batches_number; iteration++)
        {
            Batch* current_batch = next_batch;
            next_batch = nullptr;

            cudaStreamWaitEvent(0, batch_ready_event[iteration % 2], 0);

            if(iteration + 1 < training_batches_number)
            {
                next_batch = ready_training_queue.pop();
                next_batch->copy_device_async(training_batches[iteration + 1].size(), memory_stream);
                cudaEventRecord(batch_ready_event[(iteration + 1) % 2], memory_stream);
            }

            neural_network->forward_propagate(current_batch->get_inputs_device(),
                                              training_forward_propagation,
                                              is_training);

            loss->back_propagate(*current_batch,
                                 training_forward_propagation,
                                 training_back_propagation);

            training_error += training_back_propagation.error;

            update_parameters(training_back_propagation, optimization_data, current_learning_rate);

            cudaStreamSynchronize(0);

            empty_training_queue.push(current_batch);
        }

        training_worker.join();

        training_error /= type(training_batches_number);
        results.training_error_history(epoch) = training_error;

        if(has_validation)
        {
            dataset->get_batches(validation_sample_indices, validation_batch_size, shuffle, validation_batches);
            validation_error = type(0);

            std::thread validation_worker([&]()
            {
                for(Index iteration = 0; iteration < validation_batches_number; iteration++)
                {
                    Batch* batch = empty_validation_queue.pop();
                    batch->fill_host(validation_batches[iteration],
                                     input_feature_indices,
                                     decoder_feature_indices,
                                     target_feature_indices);
                    ready_validation_queue.push(batch);
                }
            });

            Batch* next_val_batch = nullptr;
            if(validation_batches_number > 0)
            {
                next_val_batch = ready_validation_queue.pop();
                next_val_batch->copy_device_async(validation_batches[0].size(), memory_stream);
                cudaEventRecord(batch_ready_event[0], memory_stream);
            }

            for(Index iteration = 0; iteration < validation_batches_number; iteration++)
            {
                Batch* current_batch = next_val_batch;
                next_val_batch = nullptr;

                cudaStreamWaitEvent(0, batch_ready_event[iteration % 2], 0);

                if(iteration + 1 < validation_batches_number)
                {
                    next_val_batch = ready_validation_queue.pop();
                    next_val_batch->copy_device_async(validation_batches[iteration + 1].size(), memory_stream);
                    cudaEventRecord(batch_ready_event[(iteration + 1) % 2], memory_stream);
                }

                neural_network->forward_propagate(current_batch->get_inputs_device(),
                                                  validation_forward_propagation,
                                                  false);

                loss->calculate_error(*current_batch,
                                      validation_forward_propagation,
                                      validation_back_propagation);

                validation_error += validation_back_propagation.error;

                cudaStreamSynchronize(0);

                empty_validation_queue.push(current_batch);
            }

            validation_worker.join();

            validation_error /= type(validation_batches_number);
            results.validation_error_history(epoch) = validation_error;

            if(epoch != 0 && results.validation_error_history(epoch) > results.validation_error_history(epoch - 1))
                validation_failures++;
        }

        elapsed_time = get_elapsed_time(beginning_time);

        if(display && epoch % display_period == 0)
        {
            cout << "Training error: " << training_error << "\n";
            if(has_validation) cout << "Validation error: " << validation_error << "\n";
            cout << "Elapsed time: " << get_time(elapsed_time) << "\n";
        }

        stop_training = check_stopping_condition(results, epoch, elapsed_time,
                                                  results.training_error_history(epoch),
                                                  validation_failures);

        if(stop_training)
        {
            results.loss = training_back_propagation.loss_value;
            results.validation_failures = validation_failures;
            results.resize_training_error_history(epoch + 1);
            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);
            results.elapsed_time = get_time(elapsed_time);
            break;
        }
    }

    cudaStreamDestroy(memory_stream);
    cudaEventDestroy(batch_ready_event[0]);
    cudaEventDestroy(batch_ready_event[1]);

    neural_network->copy_parameters_host();
    neural_network->link_parameters_cpu();
    neural_network->copy_states_host();
    neural_network->link_states_cpu();

    set_unscaling();

    if(display) results.print();

    return results;
}

#endif

REGISTER(Optimizer, StochasticGradientDescent, "StochasticGradientDescent");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
