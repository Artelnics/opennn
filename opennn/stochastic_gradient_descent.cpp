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
#include "profiler.h"
#include "batch.h"
#include "device_backend.h"
#include "stochastic_gradient_descent.h"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

static void update_parameters_cuda(NeuralNetwork* neural_network,
                                   BackPropagation& back_propagation,
                                   OptimizerData& optimizer_data,
                                   float current_learning_rate,
                                   float momentum,
                                   bool nesterov)
{
    const Index parameters_number = neural_network->get_parameters_size();

    float* const velocity_ptr = optimizer_data.views.empty()
        ? nullptr
        : optimizer_data.views[StochasticGradientDescent::Velocity].as<float>();

    PROFILE_SCOPE("optim:sgd_update_cuda");
    sgd_update_cuda(
        parameters_number,
        neural_network->get_parameters_data(),
        velocity_ptr,
        back_propagation.gradient.as<float>(),
        current_learning_rate,
        momentum,
        nesterov,
        neural_network->get_parameters_bf16_mirror_data());
}

#else

static void update_parameters_cuda(NeuralNetwork*,
                                   BackPropagation&,
                                   OptimizerData&,
                                   float,
                                   float,
                                   bool)
{
    throw runtime_error("update_parameters_cuda requires CUDA support.");
}

#endif

StochasticGradientDescent::StochasticGradientDescent(Loss* new_loss)
    : Optimizer(new_loss)
{
    set_default();
}

void StochasticGradientDescent::set_default()
{
    name = "StochasticGradientDescent";


    initial_learning_rate = 0.001f;
    initial_decay = 0.001f;
    momentum = 0.0f;
    nesterov = false;
    batch_size = 0;


    training_loss_goal = 0.0f;
    maximum_time = 3600.0f;
    maximum_epochs = 1000;


    display_period = 100;
}

void StochasticGradientDescent::set_batch_size(const Index new_batch_size)
{
    batch_size = new_batch_size;
}

void StochasticGradientDescent::set_initial_learning_rate(const float new_learning_rate)
{
    initial_learning_rate = new_learning_rate;
}

void StochasticGradientDescent::set_initial_decay(const float new_decay)
{
    initial_decay = new_decay;
}

void StochasticGradientDescent::set_momentum(const float new_momentum)
{
    momentum = new_momentum;
}

void StochasticGradientDescent::set_nesterov(bool new_nesterov_momentum)
{
    nesterov = new_nesterov_momentum;
}

void StochasticGradientDescent::update_parameters(BackPropagation& back_propagation,
                                                  OptimizerData& optimizer_data,
                                                  float current_learning_rate) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    optimizer_data.iteration++;

    if (current_learning_rate == 0.0f)
        return;

    throw_if(momentum > 0.0f && optimizer_data.views.empty(),
             "StochasticGradientDescent::update_parameters: velocity buffer is not initialized.");

    if (neural_network->is_gpu())
    {
        update_parameters_cuda(neural_network, back_propagation, optimizer_data,
                               current_learning_rate, momentum, nesterov);
        return;
    }

    VectorMap parameters(neural_network->get_parameters_data(),
                         neural_network->get_parameters_size());

    VectorMap gradient(back_propagation.gradient.as<float>(),
                       back_propagation.gradient.size_in_floats());

    const Index parameters_size = parameters.size();

    if (momentum <= 0.0f)
    {
        #pragma omp parallel for
        for (Index i = 0; i < parameters_size; ++i)
        {
            parameters(i) -= current_learning_rate * gradient(i);
        }
    }
    else
    {
        VectorMap velocity = optimizer_data.views[Velocity].as_vector();

        #pragma omp parallel for
        for (Index i = 0; i < parameters_size; ++i)
        {
            const float learning_rate_gradient = current_learning_rate * gradient(i);
            const float new_velocity = momentum * velocity(i) - learning_rate_gradient;
            velocity(i) = new_velocity;
            parameters(i) += nesterov ? momentum * new_velocity - learning_rate_gradient : new_velocity;
        }
    }
}

#ifdef OPENNN_HAS_CUDA
void StochasticGradientDescent::update_parameters_capturable(BackPropagation& back_propagation,
                                                             OptimizerData& optimizer_data) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    float* const velocity_ptr = optimizer_data.views.empty()
        ? nullptr
        : optimizer_data.views[Velocity].as<float>();

    sgd_update_capturable_cuda(
        neural_network->get_parameters_size(),
        neural_network->get_parameters_data(),
        velocity_ptr,
        back_propagation.gradient.as<float>(),
        optimizer_data.graph_effective_lr.as<float>(),
        momentum,
        nesterov,
        neural_network->get_parameters_bf16_mirror_data(),
        Backend::get_compute_stream());
}
#else
void StochasticGradientDescent::update_parameters_capturable(BackPropagation&, OptimizerData&) const
{
    throw runtime_error("update_parameters_capturable requires CUDA support.");
}
#endif

TrainingResult StochasticGradientDescent::train()
{
    TrainingResult results(maximum_epochs + 1);

    if (!loss || !loss->get_neural_network() || !loss->get_dataset())
        return results;

    NeuralNetwork* neural_network = loss->get_neural_network();
    const bool on_gpu = neural_network->is_gpu();

    if (display) cout << "Training with stochastic gradient descent (SGD)"
                     << (on_gpu ? " CUDA" : "") << "...\n";


    Dataset* dataset = loss->get_dataset();

    const bool has_validation = dataset->has_validation();

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");
    const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");

    const vector<Index> training_sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> validation_sample_indices = dataset->get_sample_indices("Validation");

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const Index effective_batch_size = batch_size <= 0
        ? get_maximum_batch_size()
        : batch_size;

    const Index training_batch_size = (effective_batch_size <= 0 || effective_batch_size > training_samples_number)
        ? training_samples_number
        : effective_batch_size;
    const Index validation_batch_size = (effective_batch_size <= 0 || effective_batch_size > validation_samples_number)
        ? validation_samples_number
        : effective_batch_size;
    const Index training_batches_number = (training_batch_size > 0)
        ? training_samples_number / training_batch_size
        : 0;

    warn_dropped_samples(training_batch_size, training_samples_number, "training");
    if (has_validation)
        warn_dropped_samples(validation_batch_size, validation_samples_number, "validation");

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> validation_batches;

    set_names();
    set_scaling();

    
    BatchPools batch_pools;
    setup_batch_pools(batch_pools,
                      *dataset,
                      *neural_network,
                      training_batch_size,
                      validation_batch_size,
                      has_validation);

    ForwardPropagation training_forward_propagation(training_batch_size, neural_network);

    loss->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_size, loss);

    // Validation runs only at epoch boundaries, never concurrently with a
    // training step, so its activations overlay the (then-idle) training
    // ForwardPropagation buffer instead of reserving a second full buffer.
    // set() falls back to its own allocation if the training buffer is smaller.
    unique_ptr<ForwardPropagation> validation_forward_propagation;
    if (has_validation)
    {
        validation_forward_propagation = make_unique<ForwardPropagation>();
        validation_forward_propagation->set(validation_batch_size, neural_network,
                                            &training_forward_propagation.data);
    }

    ForwardPropagation* validation_fp = validation_forward_propagation.get();
    mark_validation_propagation(validation_fp);

    setup_device_training();

    const Index parameters_number = neural_network->get_parameters_size();
    OptimizerData optimizer_data;
    if (momentum > 0.0f)
        optimizer_data.set({Shape{parameters_number}}, neural_network->get_device());

    optimizer_data.iteration = 1;

    float training_error = 0.0f;
    float training_accuracy = 0.0f;
    float validation_error = 0.0f;
    float validation_accuracy = 0.0f;
    Index validation_failures = 0;
    reset_best_parameters();

    const bool is_token_cross_entropy = (loss->get_error() == Loss::Error::CrossEntropy3d);

    const bool shuffle = shuffle_samples;

    float current_learning_rate = initial_learning_rate;
    const auto training_update = [&](BackPropagation& back_propagation) {
        update_parameters(back_propagation, optimizer_data, current_learning_rate);
    };

#ifdef OPENNN_HAS_CUDA
    reset_graph_capture();

    if (on_gpu)
    {
        optimizer_data.graph_effective_lr.resize_bytes(Index(sizeof(float)), Device::CUDA);
        optimizer_data.graph_effective_lr.setZero();

        graph_update = [&](BackPropagation& back_propagation) {
            update_parameters_capturable(back_propagation, optimizer_data);
        };
    }
#endif

    const bool needs_cuda_warmup = on_gpu && device::is_cuda_build() && training_batches_number > 0;

    if (needs_cuda_warmup)
    {
        dataset->get_batches(training_sample_indices, training_batch_size, false, training_batches);
        if (has_validation)
            dataset->get_batches(validation_sample_indices, validation_batch_size, false, validation_batches);

        // The warmup steps the real optimizer_data and re-zeroes it below (a
        // dedicated warmup OptimizerData would add a parameters-sized buffer
        // to the peak). current_learning_rate still equals initial_learning_rate here.
        warmup_device_training(training_forward_propagation,
                               training_back_propagation,
                               batch_pools.training_empty_queue,
                               training_batches,
                               input_feature_indices,
                               decoder_feature_indices,
                               target_feature_indices,
                               training_update,
                               validation_fp,
                               has_validation ? &batch_pools.validation_queue() : nullptr,
                               has_validation ? &validation_batches : nullptr,
                               batch_pools.fixed_training_batch.get());

        if (momentum > 0.0f) optimizer_data.data.setZero();
        optimizer_data.iteration = 1;
    }

    time_t beginning_time;
    time(&beginning_time);
    float elapsed_time = 0.0f;

    {
        device::CudaAllocationGrowthGuard steady_state_guard(needs_cuda_warmup);

        for (Index epoch = 0; epoch <= maximum_epochs; ++epoch)
        {
            if (should_display(epoch)) cout << "Epoch: " << epoch << "\n";

            dataset->get_batches(training_sample_indices, training_batch_size, shuffle, training_batches);

            current_learning_rate = initial_learning_rate / (1.0f + float(epoch) * initial_decay);

#ifdef OPENNN_HAS_CUDA
            if (graph_update && on_gpu)
                set_scalar_device_cuda(optimizer_data.graph_effective_lr.as<float>(),
                                       current_learning_rate,
                                       Backend::get_compute_stream());
#endif

            const Loss::EvaluationResult training_evaluation_result = train_epoch(training_forward_propagation,
                                                                                 training_back_propagation,
                                                                                 batch_pools.training_empty_queue,
                                                                                 training_batches,
                                                                                 input_feature_indices,
                                                                                 decoder_feature_indices,
                                                                                 target_feature_indices,
                                                                                 training_update,
                                                                                 batch_pools.fixed_training_batch.get());

            training_error = training_evaluation_result.error;
            training_accuracy = training_evaluation_result.accuracy;
            results.training_error_history(epoch) = training_error;

            if (has_validation)
            {
                dataset->get_batches(validation_sample_indices, validation_batch_size, false, validation_batches);

                const Loss::EvaluationResult validation_evaluation_result = evaluate_epoch(*validation_fp,
                                                                                          batch_pools.validation_queue(),
                                                                                          validation_batches,
                                                                                          input_feature_indices,
                                                                                          decoder_feature_indices,
                                                                                          target_feature_indices);

                validation_error = validation_evaluation_result.error;
                validation_accuracy = validation_evaluation_result.accuracy;
                results.validation_error_history(epoch) = validation_error;

                update_best_parameters(neural_network, validation_error, epoch, validation_failures);
            }

            elapsed_time = get_elapsed_time(beginning_time);

            display_epoch_results(epoch, training_error, training_accuracy,
                                  validation_error, validation_accuracy,
                                  has_validation, is_token_cross_entropy, elapsed_time);

            if (check_stopping_condition(results, epoch, elapsed_time,
                                         results.training_error_history(epoch),
                                         validation_failures,
                                         training_back_propagation.loss,
                                         has_validation))
                break;
        }
    }

    teardown_device_training();

    restore_best_parameters(neural_network, results);

    set_unscaling();

    if (display) results.print();

    return results;
}

void StochasticGradientDescent::to_JSON(JsonWriter& printer) const
{
    printer.open_element("StochasticGradientDescent");

    write_json(printer, {
        {"BatchSize", to_string(batch_size)},
        {"InitialLearningRate", to_string(initial_learning_rate)},
        {"InitialDecay", to_string(initial_decay)},
        {"Momentum", to_string(momentum)},
        {"Nesterov", to_string(nesterov)},
        {"ApplyMomentum", to_string(momentum > 0.0f)}
    });
    write_common_json(printer);

    printer.close_element();
}

void StochasticGradientDescent::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "StochasticGradientDescent");

    set_batch_size(read_json_index(root_element, "BatchSize"));

    if (root_element->has("InitialLearningRate")) set_initial_learning_rate(read_json_float(root_element, "InitialLearningRate"));
    if (root_element->has("InitialDecay"))        set_initial_decay(read_json_float(root_element, "InitialDecay"));
    if (root_element->has("Nesterov"))            set_nesterov(read_json_bool(root_element, "Nesterov"));

    if (root_element->has("Momentum"))
        set_momentum(read_json_float(root_element, "Momentum"));
    else
        set_momentum(read_json_bool(root_element, "ApplyMomentum") ? 0.9f : 0.0f);

    read_common_json(root_element);
}

REGISTER(Optimizer, StochasticGradientDescent, "StochasticGradientDescent");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
