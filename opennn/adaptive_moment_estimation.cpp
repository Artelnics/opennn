//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstring>
#include <cstdlib>
#include "registry.h"
#include "dataset.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "loss.h"
#include "profiler.h"
#include "batch.h"
#include "configuration.h"
#include "device_backend.h"
#include "adaptive_moment_estimation.h"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

static void update_parameters_cuda(NeuralNetwork* neural_network,
                                   BackPropagation& back_propagation,
                                   OptimizerData& optimization_data,
                                   float beta_1,
                                   float beta_2,
                                   float learning_rate,
                                   float bias_correction_1,
                                   float bias_correction_2)
{
    PROFILE_SCOPE("optim:adam_update_cuda");
    const Index parameters_number = neural_network->get_parameters_size();

    adam_update_cuda(
        parameters_number,
        neural_network->get_parameters_data(),
        optimization_data.views[AdaptiveMomentEstimation::GradientMoment].as<float>(),
        optimization_data.views[AdaptiveMomentEstimation::SquareGradientMoment].as<float>(),
        back_propagation.gradient.as<float>(),
        beta_1,
        beta_2,
        learning_rate,
        EPSILON,
        bias_correction_1,
        bias_correction_2,
        neural_network->get_parameters_bf16_mirror_data());
}

#else

static void update_parameters_cuda(NeuralNetwork*,
                                   BackPropagation&,
                                   OptimizerData&,
                                   float,
                                   float,
                                   float,
                                   float,
                                   float)
{
    throw runtime_error("update_parameters_cuda requires CUDA support.");
}

#endif

AdaptiveMomentEstimation::AdaptiveMomentEstimation(Loss* new_loss)
    : Optimizer(new_loss)
{
    set_default();
}

void AdaptiveMomentEstimation::set_batch_size(const Index new_batch_size)
{
    batch_size = new_batch_size;
}

void AdaptiveMomentEstimation::set_beta_1(const float new_beta_1)
{
    throw_if(new_beta_1 < 0.0f || new_beta_1 >= 1.0f,
             "AdaptiveMomentEstimation::set_beta_1: beta_1 must be in [0, 1).");

    beta_1 = new_beta_1;
}

void AdaptiveMomentEstimation::set_beta_2(const float new_beta_2)
{
    throw_if(new_beta_2 < 0.0f || new_beta_2 >= 1.0f,
             "AdaptiveMomentEstimation::set_beta_2: beta_2 must be in [0, 1).");

    beta_2 = new_beta_2;
}

void AdaptiveMomentEstimation::set_default()
{
    batch_size = 0;
    display_period = 100;
    name = "AdaptiveMomentEstimation";
}

void AdaptiveMomentEstimation::set_learning_rate(const float new_learning_rate)
{
    learning_rate = new_learning_rate;
}

TrainingResult AdaptiveMomentEstimation::train()
{
    TrainingResult results(maximum_epochs + 1);

    if (!loss || !loss->get_neural_network() || !loss->get_dataset())
        return results;

    const bool on_gpu = is_gpu();

    if (display) cout << "Training with adaptive moment estimation \"Adam\""
                     << (on_gpu ? " CUDA" : "") << " ...\n";


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


    NeuralNetwork* neural_network = loss->get_neural_network();

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

    setup_device_training();


    const Index parameters_number = neural_network->get_parameters_size();

    const Device device = current_device();

    OptimizerData optimization_data;
    optimization_data.set({Shape{parameters_number}, Shape{parameters_number}}, device);

    optimization_data.iteration = 0;

    float training_error = 0.0f;
    float training_accuracy = 0.0f;
    float validation_error = 0.0f;
    float validation_accuracy = 0.0f;
    Index validation_failures = 0;
    reset_best_parameters();

    const bool is_token_cross_entropy = (loss->get_error() == Loss::Error::CrossEntropy3d);

    const bool shuffle = shuffle_samples
                      && !neural_network->has(LayerType::Recurrent)
                      && !neural_network->has(LayerType::LongShortTermMemory);

    const auto training_update = [&](BackPropagation& back_propagation) {
        update_parameters(back_propagation, optimization_data);
    };

#ifdef OPENNN_HAS_CUDA
    reset_graph_capture();

    if (on_gpu)
    {
        optimization_data.graph_step.resize_bytes(Index(sizeof(int)), Device::CUDA);
        optimization_data.graph_step.setZero();
        optimization_data.graph_effective_lr.resize_bytes(Index(sizeof(float)), Device::CUDA);
        optimization_data.graph_effective_eps.resize_bytes(Index(sizeof(float)), Device::CUDA);

        graph_update = [&](BackPropagation& back_propagation) {
            update_parameters_capturable(back_propagation, optimization_data);
        };
    }
#endif

    const bool needs_cuda_warmup = on_gpu && device::is_cuda_build() && training_batches_number > 0;

    if (needs_cuda_warmup)
    {
        dataset->get_batches(training_sample_indices, training_batch_size, false, training_batches);
        if (has_validation)
            dataset->get_batches(validation_sample_indices, validation_batch_size, false, validation_batches);

        OptimizerData warmup_optimization_data;
        warmup_optimization_data.set({Shape{parameters_number}, Shape{parameters_number}}, device);
        warmup_optimization_data.iteration = 0;

        const auto warmup_update = [&](BackPropagation& back_propagation) {
            update_parameters(back_propagation, warmup_optimization_data);
        };

        warmup_device_training(training_forward_propagation,
                               training_back_propagation,
                               batch_pools.training_empty_queue,
                               training_batches,
                               input_feature_indices,
                               decoder_feature_indices,
                               target_feature_indices,
                               warmup_update,
                               validation_fp,
                               has_validation ? &batch_pools.validation_queue() : nullptr,
                               has_validation ? &validation_batches : nullptr,
                               batch_pools.fixed_training_batch.get());

#ifdef OPENNN_HAS_CUDA
        // The graph epoch path ignores warmup_update and steps the real
        // optimization_data (the captured graph references its buffers), so the
        // warmup leaves step/moments non-zero while the model state is restored.
        if (graph_update)
        {
            optimization_data.data.setZero();
            optimization_data.graph_step.setZero();
        }
#endif
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
                dataset->get_batches(validation_sample_indices, validation_batch_size, shuffle, validation_batches);

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

void AdaptiveMomentEstimation::update_parameters(BackPropagation& back_propagation,
                                                 OptimizerData& optimization_data) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    optimization_data.iteration++;

    {
        PROFILE_SCOPE("optim:clip_gradient_norm");
        clip_gradient_norm(back_propagation.gradient, gradient_clip_norm);
    }

    const float iteration = static_cast<float>(optimization_data.iteration);

    const float bias_correction_1 = 1.0f - pow(beta_1, iteration);
    const float bias_correction_2 = 1.0f - pow(beta_2, iteration);

    if (neural_network->is_gpu())
    {
        update_parameters_cuda(neural_network, back_propagation, optimization_data,
                               beta_1, beta_2, learning_rate,
                               bias_correction_1, bias_correction_2);
        return;
    }

    VectorMap parameters(neural_network->get_parameters_data(),
                         neural_network->get_parameters_size());

    VectorMap gradient_exponential_decay = optimization_data.views[GradientMoment].as_vector();
    VectorMap square_gradient_exponential_decay = optimization_data.views[SquareGradientMoment].as_vector();

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

void AdaptiveMomentEstimation::update_parameters_capturable(BackPropagation& back_propagation,
                                                            OptimizerData& optimization_data) const
{
#ifdef OPENNN_HAS_CUDA
    NeuralNetwork* neural_network = loss->get_neural_network();

    clip_gradient_norm(back_propagation.gradient, gradient_clip_norm);

    adam_update_capturable_cuda(
        neural_network->get_parameters_size(),
        neural_network->get_parameters_data(),
        optimization_data.views[GradientMoment].as<float>(),
        optimization_data.views[SquareGradientMoment].as<float>(),
        back_propagation.gradient.as<float>(),
        beta_1, beta_2, learning_rate, EPSILON,
        optimization_data.graph_step.as<int>(),
        optimization_data.graph_effective_lr.as<float>(),
        optimization_data.graph_effective_eps.as<float>(),
        neural_network->get_parameters_bf16_mirror_data(),
        Backend::get_compute_stream());
#else
    (void)back_propagation; (void)optimization_data;
    throw runtime_error("update_parameters_capturable requires CUDA support.");
#endif
}

void AdaptiveMomentEstimation::to_JSON(JsonWriter& printer) const
{
    printer.open_element("AdaptiveMomentEstimation");

    add_json_field(printer, "BatchSize", to_string(batch_size));
    add_json_field(printer, "LearningRate", to_string(learning_rate));
    add_json_field(printer, "Beta1", to_string(beta_1));
    add_json_field(printer, "Beta2", to_string(beta_2));
    add_json_field(printer, "GradientClipNorm", to_string(gradient_clip_norm));
    write_common_json(printer);

    printer.close_element();
}

void AdaptiveMomentEstimation::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "AdaptiveMomentEstimation");

    set_batch_size(read_json_index(root_element, "BatchSize"));
    if (root_element->has("LearningRate"))     set_learning_rate(read_json_float(root_element, "LearningRate"));
    if (root_element->has("Beta1"))            set_beta_1(read_json_float(root_element, "Beta1"));
    if (root_element->has("Beta2"))            set_beta_2(read_json_float(root_element, "Beta2"));
    if (root_element->has("GradientClipNorm")) set_gradient_clip_norm(read_json_float(root_element, "GradientClipNorm"));
    read_common_json(root_element);
}

REGISTER(Optimizer, AdaptiveMomentEstimation, "AdaptiveMomentEstimation");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
