//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstring>
#include "registry.h"
#include "dataset.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "loss.h"
#include "profiler.h"
#include "batch.h"
#include "cuda_dispatch.h"
#include "adaptive_moment_estimation.h"

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#endif

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

void AdaptiveMomentEstimation::set_beta_1(const float new_beta_1)
{
    beta_1 = new_beta_1;
}

void AdaptiveMomentEstimation::set_beta_2(const float new_beta_2)
{
    beta_2 = new_beta_2;
}

void AdaptiveMomentEstimation::set_default()
{
    display_period = 100;
    name = "AdaptiveMomentEstimation";
}

void AdaptiveMomentEstimation::set_learning_rate(const float new_learning_rate)
{
    learning_rate = new_learning_rate;
}

TrainingResults AdaptiveMomentEstimation::train()
{
    TrainingResults results(maximum_epochs + 1);

    const bool on_gpu = is_gpu();

    if (display) cout << "Training with adaptive moment estimation \"Adam\""
                     << (on_gpu ? " CUDA" : "") << " ...\n";

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

    const Index training_batch_size = (batch_size <= 0 || batch_size > training_samples_number)
        ? training_samples_number
        : batch_size;
    const Index validation_batch_size = (batch_size <= 0 || batch_size > validation_samples_number)
        ? validation_samples_number
        : batch_size;
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

    const int pool_size = on_gpu ? 3 : 2;

    ThreadSafeQueue<Batch*> empty_training_queue;
    ThreadSafeQueue<Batch*> ready_training_queue;
    vector<unique_ptr<Batch>> training_batch_pool;

    for (int i = 0; i < pool_size; ++i)
    {
        training_batch_pool.push_back(make_unique<Batch>(training_batch_size, dataset));
        empty_training_queue.push(training_batch_pool.back().get());
    }

    ThreadSafeQueue<Batch*> empty_validation_queue;
    ThreadSafeQueue<Batch*> ready_validation_queue;
    vector<unique_ptr<Batch>> validation_batch_pool;

    const bool share_batch_pool = has_validation && validation_batch_size == training_batch_size;

    if (has_validation && !share_batch_pool)
        for (int i = 0; i < pool_size; ++i)
        {
            validation_batch_pool.push_back(make_unique<Batch>(validation_batch_size, dataset));
            empty_validation_queue.push(validation_batch_pool.back().get());
        }

    ThreadSafeQueue<Batch*>& validation_empty_q = share_batch_pool ? empty_training_queue : empty_validation_queue;
    ThreadSafeQueue<Batch*>& validation_ready_q = share_batch_pool ? ready_training_queue : ready_validation_queue;
    ForwardPropagation training_forward_propagation(training_batch_size, neural_network);

    loss->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_size, loss);

    unique_ptr<ForwardPropagation> validation_forward_propagation;

    if (has_validation && validation_batch_size != training_batch_size)
        validation_forward_propagation = make_unique<ForwardPropagation>(validation_batch_size, neural_network);

    ForwardPropagation* validation_fp = has_validation
        ? (validation_forward_propagation ? validation_forward_propagation.get() : &training_forward_propagation)
        : nullptr;

    setup_device_training();

    // Optimization data

    const Index parameters_number = loss->get_neural_network()->get_parameters_size();

    const Device device = is_gpu() ? Device::CUDA : Device::CPU;

    OptimizerData optimization_data;
    optimization_data.set({Shape{parameters_number}, Shape{parameters_number}}, device);

    optimization_data.iteration = 1;

    float training_error = 0.0f;
    float training_accuracy = 0.0f;
    float validation_error = 0.0f;
    float validation_accuracy = 0.0f;
    Index validation_failures = 0;
    float best_validation_error = numeric_limits<float>::max();

    vector<float> best_parameters;

    const bool is_token_cross_entropy = (loss->get_error() == Loss::Error::CrossEntropy3d);

    bool stop_training = false;
    const bool shuffle = !neural_network->has(LayerType::Recurrent);

    time_t beginning_time;
    time(&beginning_time);
    float elapsed_time = 0.0f;

    // Main loop

    const auto training_update = [&](BackPropagation& back_propagation) {
        update_parameters(back_propagation, optimization_data);
    };

    for (Index epoch = 0; epoch <= maximum_epochs; ++epoch)
    {
        if (should_display(epoch)) cout << "Epoch: " << epoch << "\n";

        dataset->get_batches(training_sample_indices, training_batch_size, shuffle, training_batches);

        const EpochStats train_stats = train_epoch(is_token_cross_entropy,
                                                   training_forward_propagation,
                                                   training_back_propagation,
                                                   empty_training_queue,
                                                   ready_training_queue,
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
                                                        validation_ready_q,
                                                        validation_batches,
                                                        input_feature_indices,
                                                        decoder_feature_indices,
                                                        target_feature_indices);

            validation_error = val_stats.error;
            validation_accuracy = val_stats.accuracy;
            results.validation_error_history(epoch) = validation_error;

            if(validation_error < best_validation_error)
            {
                best_validation_error = validation_error;
                validation_failures = 0;

                const Index psize = neural_network->get_parameters_size();
                if(Index(best_parameters.size()) != psize)
                    best_parameters.resize(psize);

                const float* src = neural_network->get_parameters_data();
                const size_t bytes = size_t(psize) * sizeof(float);
#ifdef OPENNN_HAS_CUDA
                if(Configuration::instance().is_gpu())
                {
                    CHECK_CUDA(cudaStreamSynchronize(Backend::get_compute_stream()));
                    CHECK_CUDA(cudaMemcpy(best_parameters.data(), src, bytes, cudaMemcpyDeviceToHost));
                }
                else
#endif
                    std::memcpy(best_parameters.data(), src, bytes);
            }
            else
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
            results.loss = training_back_propagation.loss_value;
            results.validation_failures = validation_failures;
            results.resize_training_error_history(epoch + 1);
            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);
            results.elapsed_time = get_time(elapsed_time);
            break;
        }
    }

    teardown_device_training();

    if(results.stopping_condition == StoppingCondition::MaximumSelectionErrorIncreases
       && !best_parameters.empty()
       && Index(best_parameters.size()) == neural_network->get_parameters_size())
    {
        if(display)
            cout << "Restoring best parameters (validation error " << best_validation_error << ")\n";

        // Use set_parameters (not memcpy) so the GPU path properly issues
        // cudaMemcpy(H2D) and refreshes the BF16 mirror via
        // cast_parameters_to_bf16. memcpy into a device pointer is UB.
        VectorR best_view(best_parameters.size());
        std::memcpy(best_view.data(), best_parameters.data(),
                    best_parameters.size() * sizeof(float));
        neural_network->set_parameters(best_view);
    }

    set_unscaling();

    if (display) results.print();

    return results;
}

void AdaptiveMomentEstimation::update_parameters(BackPropagation& back_propagation,
                                                 OptimizerData& optimization_data) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    optimization_data.iteration++;

    {
        PROFILE_SCOPE("optim:clip_gradient_norm");
        clip_gradient_norm(back_propagation.gradient, 1.0f);
    }

    const float iteration = static_cast<float>(optimization_data.iteration);

    const float bias_correction_1 = 1.0f - pow(beta_1, iteration);
    const float bias_correction_2 = 1.0f - pow(beta_2, iteration);

    IF_GPU({
        PROFILE_SCOPE("optim:adam_update_cuda");
        const Index parameters_number = neural_network->get_parameters_size();

        adam_update_cuda(
            parameters_number,
            neural_network->get_parameters_data(),
            optimization_data.views[GradientMoment].as<float>(),
            optimization_data.views[SquareGradientMoment].as<float>(),
            back_propagation.gradient.as<float>(),
            beta_1,
            beta_2,
            learning_rate,
            EPSILON,
            bias_correction_1,
            bias_correction_2,
            neural_network->get_parameters_bf16_data());

        return;
    });

    VectorMap parameters(neural_network->get_parameters_data(),
                         neural_network->get_parameters_size());

    VectorMap gradient_exponential_decay(optimization_data.views[GradientMoment].as<float>(),
                                         optimization_data.views[GradientMoment].size());
    VectorMap square_gradient_exponential_decay(optimization_data.views[SquareGradientMoment].as<float>(),
                                                optimization_data.views[SquareGradientMoment].size());

    VectorMap gradient(back_propagation.gradient.as<float>(),
                       back_propagation.gradient.size_in_floats());

    const Index parameters_size = parameters.size();
    const float one_minus_beta_1 = 1.0f - beta_1;
    const float one_minus_beta_2 = 1.0f - beta_2;

    const float sqrt_bias_correction_2 = sqrt(bias_correction_2);
    const float effective_learning_rate = learning_rate * sqrt_bias_correction_2 / bias_correction_1;
    const float effective_epsilon = EPSILON * sqrt_bias_correction_2;

    #pragma omp parallel for if(parameters_size > 4096)
    for (Index i = 0; i < parameters_size; ++i)
    {
        const float gradient_value = gradient(i);

        auto& first_moment = gradient_exponential_decay(i);
        auto& second_moment = square_gradient_exponential_decay(i);

        first_moment = beta_1 * first_moment + one_minus_beta_1 * gradient_value;
        second_moment = beta_2 * second_moment + one_minus_beta_2 * gradient_value * gradient_value;

        parameters(i) -= effective_learning_rate * first_moment / (sqrt(second_moment) + effective_epsilon);
    }
}

void AdaptiveMomentEstimation::to_JSON(JsonWriter& printer) const
{
    printer.open_element("AdaptiveMomentEstimation");

    add_json_field(printer, "BatchSize", to_string(batch_size));
    write_common_json(printer);

    printer.close_element();
}

void AdaptiveMomentEstimation::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "AdaptiveMomentEstimation");

    set_batch_size(read_json_index(root_element, "BatchSize"));
    read_common_json(root_element);
}

REGISTER(Optimizer, AdaptiveMomentEstimation, "AdaptiveMomentEstimation");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
