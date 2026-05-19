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

    initial_learning_rate = 0.001f;
    initial_decay = 0.001f;
    momentum = 0.0f;
    nesterov = false;
    batch_size = 0;

    // Stopping criteria

    training_loss_goal = 0.0f;
    maximum_time = 3600.0f;
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
                                                  OptimizerData& optimization_data,
                                                  float current_learning_rate) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    optimization_data.iteration++;

    if (current_learning_rate == 0.0f)
        return;

    if (momentum > 0.0f && optimization_data.views.empty())
        throw runtime_error("StochasticGradientDescent::update_parameters: velocity buffer is not initialized.");

#ifdef OPENNN_HAS_CUDA
    if (is_gpu())
        CHECK_CUDA(cudaStreamSynchronize(Backend::get_compute_stream()));
#endif

#ifdef OPENNN_HAS_CUDA
    if (is_gpu())
    {
        const Index parameters_number = neural_network->get_parameters_size();

        float* const velocity_ptr = optimization_data.views.empty()
            ? nullptr
            : optimization_data.views[Velocity].as<float>();

        // BF16 mirror (if allocated) is refreshed inside the same kernel —
        // saves a separate FP32→BF16 cast pass over the whole parameter set.
        PROFILE_SCOPE("optim:sgd_update_cuda");
        sgd_update_cuda(
            parameters_number,
            neural_network->get_parameters_data(),
            velocity_ptr,
            back_propagation.gradient.as<float>(),
            current_learning_rate,
            momentum,
            nesterov,
            neural_network->get_parameters_bf16_data());

        return;
    }
#endif

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
        VectorMap velocity(optimization_data.views[Velocity].as<float>(),
                           optimization_data.views[Velocity].size());

        #pragma omp parallel for
        for (Index i = 0; i < parameters_size; ++i)
        {
            const float lr_g = current_learning_rate * gradient(i);
            const float v_new = momentum * velocity(i) - lr_g;
            velocity(i) = v_new;
            parameters(i) += nesterov ? momentum * v_new - lr_g : v_new;
        }
    }
}

TrainingResults StochasticGradientDescent::train()
{
    TrainingResults results(maximum_epochs + 1);

    const bool on_gpu = is_gpu();

    if (display) cout << "Training with stochastic gradient descent (SGD)"
                     << (on_gpu ? " CUDA" : "") << "...\n";

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

    // Neural network

    NeuralNetwork* neural_network = loss->get_neural_network();

    set_names();
    set_scaling();

    const int pool_size = on_gpu ? max(num_workers + 1, 3) : 1;

    ThreadSafeQueue<Batch*> empty_training_queue;
    vector<unique_ptr<Batch>> training_batch_pool;

    for (int i = 0; i < pool_size; ++i)
    {
        training_batch_pool.push_back(make_unique<Batch>(training_batch_size, dataset));
        empty_training_queue.push(training_batch_pool.back().get());
    }

    ThreadSafeQueue<Batch*> empty_validation_queue;
    vector<unique_ptr<Batch>> validation_batch_pool;

    const bool share_batch_pool = has_validation && validation_batch_size == training_batch_size;

    if (has_validation && !share_batch_pool)
    {
        for (int i = 0; i < pool_size; ++i)
        {
            validation_batch_pool.push_back(make_unique<Batch>(validation_batch_size, dataset));
            empty_validation_queue.push(validation_batch_pool.back().get());
        }
    }

    ThreadSafeQueue<Batch*>& validation_empty_q = share_batch_pool ? empty_training_queue : empty_validation_queue;
    ForwardPropagation training_forward_propagation(training_batch_size, neural_network);

    loss->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_size, loss);

    unique_ptr<ForwardPropagation> validation_forward_propagation;

    if (has_validation)
        validation_forward_propagation = make_unique<ForwardPropagation>(validation_batch_size, neural_network);

    ForwardPropagation* validation_fp = has_validation
        ? validation_forward_propagation.get()
        : nullptr;

    vector<Batch*> all_batches;
    all_batches.reserve(training_batch_pool.size() + validation_batch_pool.size());
    for (auto& b : training_batch_pool)   all_batches.push_back(b.get());
    for (auto& b : validation_batch_pool) all_batches.push_back(b.get());
    setup_device_training(all_batches);

    // Optimization data

    const Index parameters_number = loss->get_neural_network()->get_parameters_size();
    const Device device = current_device();

    OptimizerData optimization_data;
    if (momentum > 0.0f)
        optimization_data.set({Shape{parameters_number}}, device);

    optimization_data.iteration = 1;

    float training_error = 0.0f;
    float training_accuracy = 0.0f;
    float validation_error = 0.0f;
    float validation_accuracy = 0.0f;
    Index validation_failures = 0;

    // True for sequence/token-level cross-entropy losses (translation, language
    // modelling, chat). Gates the per-token metrics (accuracy + perplexity)
    // and the per-batch token-count plumbing in train_epoch / evaluate_epoch.
    const bool is_token_cross_entropy = (loss->get_error() == Loss::Error::CrossEntropy3d);

    bool stop_training = false;
    const bool shuffle = !neural_network->has(LayerType::Recurrent);

    time_t beginning_time;
    time(&beginning_time);
    float elapsed_time = 0.0f;

    float current_learning_rate = initial_learning_rate;
    const auto training_update = [&](BackPropagation& back_propagation) {
        update_parameters(back_propagation, optimization_data, current_learning_rate);
    };

    // Main loop

    for (Index epoch = 0; epoch <= maximum_epochs; ++epoch)
    {
        if (should_display(epoch)) cout << "Epoch: " << epoch << "\n";

        dataset->get_batches(training_sample_indices, training_batch_size, shuffle, training_batches);

        current_learning_rate = initial_learning_rate / (1.0f + float(epoch) * initial_decay);

        const EpochStats train_stats = train_epoch(is_token_cross_entropy,
                                                   training_forward_propagation,
                                                   training_back_propagation,
                                                   empty_training_queue,
                                                   training_batches,
                                                   input_feature_indices,
                                                   decoder_feature_indices,
                                                   target_feature_indices,
                                                   training_update,
                                                   should_display(epoch));

        training_error = train_stats.error;
        training_accuracy = train_stats.accuracy;
        results.training_error_history(epoch) = training_error;

        if (has_validation)
        {
            dataset->get_batches(validation_sample_indices, validation_batch_size, shuffle, validation_batches);

            const EpochStats val_stats = evaluate_epoch(is_token_cross_entropy,
                                                        *validation_fp,
                                                        validation_empty_q,
                                                        validation_batches,
                                                        input_feature_indices,
                                                        decoder_feature_indices,
                                                        target_feature_indices);

            validation_error = val_stats.error;
            validation_accuracy = val_stats.accuracy;
            results.validation_error_history(epoch) = validation_error;

            if (epoch != 0 && validation_error > results.validation_error_history(epoch - 1))
                ++validation_failures;
        }

        elapsed_time = get_elapsed_time(beginning_time);

        if (should_display(epoch))
        {
            cout << "Training error: " << training_error << "\n";
            if (is_token_cross_entropy) cout << "Training perplexity: " << exp(training_error) << "\n";
            if (is_token_cross_entropy) cout << "Training accuracy: " << training_accuracy << "\n";
            if (has_validation) cout << "Validation error: " << validation_error << "\n";
            if (has_validation && is_token_cross_entropy) cout << "Validation perplexity: " << exp(validation_error) << "\n";
            if (has_validation && is_token_cross_entropy) cout << "Validation accuracy: " << validation_accuracy << "\n";
            cout << "Elapsed time: " << get_time(elapsed_time) << "\n";
        }

        stop_training = check_stopping_condition(results, epoch, elapsed_time,
                                                  results.training_error_history(epoch),
                                                  validation_failures);

        if (stop_training)
        {
            results.loss = training_back_propagation.loss;
            results.validation_failures = validation_failures;
            results.resize_training_error_history(epoch + 1);
            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);
            results.elapsed_time = get_time(elapsed_time);
            break;
        }
    }

    teardown_device_training();

    set_unscaling();

    if (display) results.print();

    return results;
}

void StochasticGradientDescent::to_JSON(JsonWriter& printer) const
{
    printer.open_element("StochasticGradientDescent");

    write_json(printer, {
        {"BatchSize", to_string(batch_size)},
        {"ApplyMomentum", to_string(momentum > 0.0f)}
    });
    write_common_json(printer);

    printer.close_element();
}

void StochasticGradientDescent::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "StochasticGradientDescent");

    set_batch_size(read_json_index(root_element, "BatchSize"));

    const bool apply_momentum = read_json_bool(root_element, "ApplyMomentum");
    set_momentum(apply_momentum ? 0.9f : 0.0f);

    read_common_json(root_element);
}

REGISTER(Optimizer, StochasticGradientDescent, "StochasticGradientDescent");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
