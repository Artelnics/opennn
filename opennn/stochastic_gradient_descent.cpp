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
#include "loss.h"
#include "stochastic_gradient_descent.h"

namespace opennn
{

StochasticGradientDescent::StochasticGradientDescent(const Loss* new_loss)
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


void StochasticGradientDescent::set_loss_goal(const type new_loss_goal)
{
    training_loss_goal = new_loss_goal;
}


void StochasticGradientDescent::update_parameters(BackPropagation& back_propagation,
                                                  StochasticGradientDescentData& optimization_data,
                                                  type learning_rate) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    VectorR& parameters = neural_network->get_parameters();

    VectorR& gradient = back_propagation.gradient;

    VectorR& parameter_updates = optimization_data.parameter_updates;
    VectorR& last_parameter_updates = optimization_data.last_parameter_updates;

    if (momentum <= type(0))
    {
        parameter_updates = gradient * (-learning_rate);
        parameters += parameter_updates;
    }
    else if (momentum > type(0) && !nesterov)
    {
        parameter_updates = gradient*(-learning_rate) + momentum*last_parameter_updates;
        last_parameter_updates = parameter_updates;
        parameters += parameter_updates;
    }
    else if (momentum > type(0) && nesterov)
    {
        parameter_updates = gradient*(-learning_rate) + momentum*last_parameter_updates;
        last_parameter_updates = parameter_updates;
        parameters += parameter_updates*momentum - gradient*learning_rate;
    }
}


TrainingResults StochasticGradientDescent::train()
{
    if(!loss || !loss->has_neural_network() || !loss->has_dataset())
        return TrainingResults();

    TrainingResults results(maximum_epochs+1);

    check();

    // Start training

    if(display) cout << "Training with stochastic gradient descent (SGD)...\n";

    // Dataset

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
    vector<vector<Index>> validation_batches(validation_batches_number);

    vector<Index> training_batch_indices(training_batch_size);
    vector<Index> selection_batch_indices(training_batch_size);

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
    bool is_training = true;

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    bool shuffle = !neural_network->has("Recurrent");

    // Main loop

    for(Index epoch = 0; epoch <= maximum_epochs; epoch++)
    {
        if(display && epoch%display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = dataset->get_batches(training_sample_indices, training_batch_size, shuffle);

        const Index batches_number = training_batches.size();

        const type current_learning_rate = initial_learning_rate / (type(1) + type(epoch) * initial_decay);

        //training_loss = type(0);
        training_error = type(0);

        optimization_data.iteration = 0;

        for(Index iteration = 0; iteration < batches_number; iteration++)
        {
            optimization_data.iteration++;

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

            results.training_error_history(epoch) = training_back_propagation.error;

            training_error += training_back_propagation.error;
            //training_loss += training_back_propagation.loss;

            // Gradient

            update_parameters(training_back_propagation, optimization_data, current_learning_rate);

            //if(display && epoch % display_period == 0)      display_progress_bar(iteration, batches_number - 1);
        }

        // Loss

        training_error /= type(batches_number);

        results.training_error_history(epoch) = training_error;

        if(has_validation)
        {
            validation_batches = dataset->get_batches(validation_sample_indices, validation_batch_size, shuffle);

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
                                                  is_training);

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
            cout << "Training error: " << training_error << endl;
            if(has_validation) cout << "Validation error: " << validation_error << endl<<endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        // Stopping criteria

        stop_training = check_stopping_condition(results, epoch, elapsed_time,
                                                  results.training_error_history(epoch),
                                                  validation_failures, training_loss_goal,
                                                  maximum_validation_failures);

        if(stop_training)
        {
            results.loss = training_back_propagation.loss_value;
            results.validation_failures = validation_failures;
            results.elapsed_time = write_time(elapsed_time);

            results.resize_training_error_history(epoch+1);
            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);

            break;
        }

        // Update stuff

        if(epoch != 0 && epoch%save_period == 0)
            neural_network->save(neural_network_file_name);
    }

    set_unscaling();

    if(display) results.print();

    return results;
}


void StochasticGradientDescent::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("StochasticGradientDescent");

    add_xml_element(printer, "BatchSize", to_string(batch_size));
    add_xml_element(printer, "ApplyMomentum", to_string(momentum > type(0)));
    add_xml_element(printer, "LossGoal", to_string(training_loss_goal));
    add_xml_element(printer, "MaximumEpochsNumber", to_string(maximum_epochs));
    add_xml_element(printer, "MaximumTime", to_string(maximum_time));
    add_xml_element(printer, "HardwareUse", hardware_use);

    printer.CloseElement();
}


void StochasticGradientDescent::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = get_xml_root(document, "StochasticGradientDescent");

    set_batch_size(read_xml_index(root_element, "BatchSize"));

    const bool apply_momentum = read_xml_bool(root_element, "ApplyMomentum");
    set_momentum(apply_momentum ? type(0.9) : type(0));

    set_loss_goal(read_xml_type(root_element, "LossGoal"));
    set_maximum_epochs(read_xml_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_xml_type(root_element, "MaximumTime"));
    set_hardware_use(read_xml_string(root_element, "HardwareUse"));
}


StochasticGradientDescentData::StochasticGradientDescentData(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    set(new_stochastic_gradient_descent);
}


void StochasticGradientDescentData::set(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    stochastic_gradient_descent = new_stochastic_gradient_descent;

    const Loss* loss = stochastic_gradient_descent->get_loss();

    NeuralNetwork* neural_network = loss->get_neural_network();

    const Index parameters_number = neural_network->get_parameters().size();

    parameter_updates.resize(parameters_number);
    parameter_updates.setZero();

    last_parameter_updates.resize(parameters_number);
    last_parameter_updates.setZero();
}


#ifdef CUDA

TrainingResults StochasticGradientDescent::train_cuda()
{
    if(!loss || !loss->has_neural_network() || !loss->has_dataset())
        return TrainingResults();

    TrainingResults results(maximum_epochs + 1);

    check();

    if (display) cout << "Training with stochastic gradient descent (SGD) CUDA...\n";

    // Dataset

    Dataset* dataset = loss->get_dataset();

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

    // Optimization algorithm
    SGDOptimizationDataCuda optimization_data(this);

    bool stop_training = false;
    bool is_training = true;
    bool shuffle = !neural_network->has("Recurrent");

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time = type(0);
    optimization_data.iteration = 1;

    // Main loop
    for(Index epoch = 0; epoch <= maximum_epochs; epoch++)
    {
        if (display && epoch % display_period == 0) cout << "Epoch: " << epoch << endl;

        training_batches = dataset->get_batches(training_sample_indices, training_batch_size, shuffle);

        const type current_learning_rate = initial_learning_rate / (type(1) + type(epoch) * initial_decay);

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
            BatchCuda* current_batch = ready_training_queue.pop();

            current_batch->copy_device_async(training_batches[iteration].size(), memory_stream);
            cudaEventRecord(batch_ready_event, memory_stream);
            cudaStreamWaitEvent(0, batch_ready_event, 0);

            neural_network->forward_propagate(current_batch->get_inputs_device(), training_forward_propagation, is_training);

            loss->back_propagate(*current_batch, training_forward_propagation, training_back_propagation);

            training_error += training_back_propagation.error;

            if (is_text_classification_model)
                training_accuracy += training_back_propagation.accuracy();

            update_parameters(training_back_propagation, optimization_data, current_learning_rate);

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

            thread validation_worker([&]()
            {
                for(Index iteration = 0; iteration < validation_batches_number; iteration++)
                {
                    BatchCuda* batch = empty_validation_queue.pop();
                    batch->fill_host(training_batches[iteration],
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

        // Elapsed time
        elapsed_time = get_elapsed_time(beginning_time);

        if (display && epoch % display_period == 0)
        {
            cout << "Training error: " << training_error << endl;
            if (is_text_classification_model) cout << "Training accuracy: " << training_accuracy << endl;
            if (has_validation) cout << "Validation error: " << validation_error << endl;
            if (has_validation && is_text_classification_model) cout << "Validation accuracy: " << validation_accuracy << endl;
            cout << "Elapsed time: " << write_time(elapsed_time) << endl;
        }

        // Stopping criteria
        stop_training = check_stopping_condition(results, epoch, elapsed_time,
                                                  results.training_error_history(epoch),
                                                  validation_failures, training_loss_goal,
                                                  maximum_validation_failures);

        if (stop_training)
        {
            results.loss = training_back_propagation.loss_value;
            results.validation_failures = validation_failures;
            results.resize_training_error_history(epoch + 1);
            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);
            results.elapsed_time = write_time(elapsed_time);
            break;
        }

        if (epoch != 0 && epoch % save_period == 0) neural_network->save(neural_network_file_name);
    }

    cudaStreamDestroy(memory_stream);
    cudaEventDestroy(batch_ready_event);

    neural_network->copy_parameters_host();
    set_unscaling();

    if (display) results.print();

    return results;
}


void StochasticGradientDescent::update_parameters(BackPropagationCuda& back_propagation,
                                                  SGDOptimizationDataCuda& optimization_data,
                                                  type current_learning_rate) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_device().size();

    float* parameters_device_data = neural_network->get_parameters_device().data;
    const float* gradients_device = back_propagation.neural_network.gradients.data;

    optimization_data.iteration++;
    const float momentum_f = static_cast<float>(momentum);

    sgd_update_device(
        static_cast<int>(parameters_number),
        parameters_device_data,
        optimization_data.velocity.data,
        gradients_device,
        static_cast<float>(current_learning_rate),
        momentum_f,
        nesterov);
}


SGDOptimizationDataCuda::SGDOptimizationDataCuda(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    set(new_stochastic_gradient_descent);
}


void SGDOptimizationDataCuda::set(StochasticGradientDescent* new_stochastic_gradient_descent)
{
    stochastic_gradient_descent = new_stochastic_gradient_descent;

    NeuralNetwork* neural_network = stochastic_gradient_descent->get_loss()->get_neural_network();

    const Index parameters_number = neural_network->get_parameters_device().size();

    velocity.resize({parameters_number});

    CHECK_CUDA(cudaMemset(velocity.data, 0, parameters_number * sizeof(float)));
}


void SGDOptimizationDataCuda::print() const
{
    cout << "--- SGD Optimization Data (CUDA) ---" << endl;
    NeuralNetwork* neural_network = stochastic_gradient_descent->get_loss()->get_neural_network();

    cout << "------------------------------------" << endl;
}

#endif

REGISTER(Optimizer, StochasticGradientDescent, "StochasticGradientDescent");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
