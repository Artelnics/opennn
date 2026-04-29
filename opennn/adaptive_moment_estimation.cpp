//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D A P T I V E   M O M E N T   E S T I M A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "loss.h"
#include "profiler.h"
#include "batch.h"
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

    const bool is_gpu = Device::instance().is_gpu();

    if(display) cout << "Training with adaptive moment estimation \"Adam\""
                     << (is_gpu ? " CUDA" : "") << " ...\n";

    // Dataset

    Dataset* dataset = loss->get_dataset();

    if(!dataset)
        throw runtime_error("AdaptiveMomentEstimation error: dataset is not set.");

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

    const Index training_batches_number = (training_batch_size != 0)
        ? training_samples_number / training_batch_size
        : 0;

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> validation_batches;

    // Neural network

    NeuralNetwork* neural_network = loss->get_neural_network();

    set_names();
    set_scaling();

    // Batch pool: minimum 2 for producer/consumer double-buffer (avoids worker-main
    // deadlock on prefetch_before_loop + pop_next). GPU uses 3 for triple-buffer H2D.

    const int pool_size = is_gpu ? 3 : 2;

    ThreadSafeQueue<Batch*> empty_training_queue;
    ThreadSafeQueue<Batch*> ready_training_queue;
    vector<unique_ptr<Batch>> training_batch_pool;

    for(int i = 0; i < pool_size; ++i)
    {
        training_batch_pool.push_back(make_unique<Batch>(training_batch_size, dataset));
        empty_training_queue.push(training_batch_pool.back().get());
    }

    ThreadSafeQueue<Batch*> empty_validation_queue;
    ThreadSafeQueue<Batch*> ready_validation_queue;
    vector<unique_ptr<Batch>> validation_batch_pool;

    if(has_validation)
    {
        for(int i = 0; i < pool_size; ++i)
        {
            validation_batch_pool.push_back(make_unique<Batch>(validation_batch_size, dataset));
            empty_validation_queue.push(validation_batch_pool.back().get());
        }
    }

    // Forward / back propagation

    ForwardPropagation training_forward_propagation(training_batch_size, neural_network);

    loss->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_size, loss);

    unique_ptr<ForwardPropagation> validation_forward_propagation;
    unique_ptr<BackPropagation> validation_back_propagation;

    if(has_validation)
    {
        validation_forward_propagation = make_unique<ForwardPropagation>(validation_batch_size, neural_network);
        validation_back_propagation = make_unique<BackPropagation>(validation_batch_size, loss);
    }

    setup_device_training(training_forward_propagation,
                          training_back_propagation,
                          validation_forward_propagation.get(),
                          validation_back_propagation.get());

    // Optimization data

    const Index parameters_number = loss->get_neural_network()->get_parameters_size();

    OptimizerData optimization_data;
    optimization_data.set({Shape{parameters_number}, Shape{parameters_number}});

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) optimization_data.allocate_device();
#endif

    optimization_data.iteration = 1;

    type training_error = type(0);
    type training_accuracy = type(0);
    type validation_error = type(0);
    type validation_accuracy = type(0);
    Index validation_failures = 0;

    const bool is_classification_model = (loss->get_error() == Loss::Error::CrossEntropy3d);

    bool stop_training = false;
    const bool shuffle = !neural_network->has(LayerType::Recurrent);

    time_t beginning_time;
    time(&beginning_time);
    type elapsed_time = type(0);

    // Main loop

    const auto training_update = [&](BackPropagation& bp) {
        update_parameters(bp, optimization_data);
    };

    for(Index epoch = 0; epoch <= maximum_epochs; ++epoch)
    {
        if(should_display(epoch)) cout << "Epoch: " << epoch << "\n";

        dataset->get_batches(training_sample_indices, training_batch_size, shuffle, training_batches);

        const EpochStats train_stats = run_epoch(Phase::Training,
                                                 is_classification_model,
                                                 training_forward_propagation,
                                                 training_back_propagation,
                                                 empty_training_queue,
                                                 ready_training_queue,
                                                 training_batches,
                                                 input_feature_indices,
                                                 decoder_feature_indices,
                                                 target_feature_indices,
                                                 training_update);

        training_error = train_stats.error;
        training_accuracy = train_stats.accuracy;
        results.training_error_history(epoch) = training_error;

        if(has_validation)
        {
            dataset->get_batches(validation_sample_indices, validation_batch_size, shuffle, validation_batches);

            const EpochStats val_stats = run_epoch(Phase::Validation,
                                                   is_classification_model,
                                                   *validation_forward_propagation,
                                                   *validation_back_propagation,
                                                   empty_validation_queue,
                                                   ready_validation_queue,
                                                   validation_batches,
                                                   input_feature_indices,
                                                   decoder_feature_indices,
                                                   target_feature_indices,
                                                   [](BackPropagation&){});

            validation_error = val_stats.error;
            validation_accuracy = val_stats.accuracy;
            results.validation_error_history(epoch) = validation_error;

            if(epoch != 0 && results.validation_error_history(epoch) > results.validation_error_history(epoch - 1))
                ++validation_failures;
        }

        elapsed_time = get_elapsed_time(beginning_time);

        if(should_display(epoch))
        {
            cout << "Training error: " << training_error << "\n";
            if(is_classification_model) cout << "Training accuracy: " << training_accuracy << "\n";
            if(has_validation) cout << "Validation error: " << validation_error << "\n";
            if(has_validation && is_classification_model) cout << "Validation accuracy: " << validation_accuracy << "\n";
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

    teardown_device_training();

    set_unscaling();

    if(display) results.print();

    return results;
}

void AdaptiveMomentEstimation::update_parameters(BackPropagation& back_propagation,
                                                 OptimizerData& optimization_data) const
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    optimization_data.iteration++;

    {
        PROFILE_SCOPE("optim:clip_gradient_norm");
        clip_gradient_norm(back_propagation.gradient, type(1));
    }

    const type iteration = static_cast<type>(optimization_data.iteration);

    const type bias_correction_1 = type(1) - pow(beta_1, iteration);
    const type bias_correction_2 = type(1) - pow(beta_2, iteration);

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu())
    {
        PROFILE_SCOPE("optim:adam_update_cuda");
        const Index parameters_number = neural_network->get_parameters_size();

        adam_update_cuda(
            parameters_number,
            neural_network->get_parameters_device(),
            optimization_data.views[GradientMoment].as<float>(),
            optimization_data.views[SquareGradientMoment].as<float>(),
            back_propagation.gradient.as<type>(),
            beta_1,
            beta_2,
            learning_rate,
            EPSILON,
            bias_correction_1,
            bias_correction_2);

        // Master FP32 weights just changed in-place. Refresh the BF16 working
        // copy so the next forward pass sees up-to-date weights. No-op when
        // OPENNN_BF16_ACTIVATIONS is off (parameters_bf16 stays empty).
        neural_network->cast_parameters_to_bf16();
        return;
    }
#endif

    VectorMap parameters(neural_network->get_parameters_data(),
                         neural_network->get_parameters_size());

    VectorMap gradient_exponential_decay(optimization_data.views[GradientMoment].as<float>(),
                                         optimization_data.views[GradientMoment].size());
    VectorMap square_gradient_exponential_decay(optimization_data.views[SquareGradientMoment].as<float>(),
                                                optimization_data.views[SquareGradientMoment].size());

    VectorMap gradient(back_propagation.gradient.as<type>(),
                       back_propagation.gradient.size());

    const Index n = parameters.size();
    const type one_minus_beta_1 = type(1) - beta_1;
    const type one_minus_beta_2 = type(1) - beta_2;

    const type s = std::sqrt(bias_correction_2);
    const type effective_learning_rate = learning_rate * s / bias_correction_1;
    const type effective_epsilon = EPSILON * s;

    #pragma omp parallel for
    for (Index i = 0; i < n; ++i)
    {
        const type g = gradient(i);

        const type m_new = beta_1 * gradient_exponential_decay(i) + one_minus_beta_1 * g;
        gradient_exponential_decay(i) = m_new;

        const type v_new = beta_2 * square_gradient_exponential_decay(i) + one_minus_beta_2 * g * g;
        square_gradient_exponential_decay(i) = v_new;

        parameters(i) -= effective_learning_rate * m_new / (std::sqrt(v_new) + effective_epsilon);
    }
}

void AdaptiveMomentEstimation::to_XML(XmlPrinter& printer) const
{
    printer.open_element("AdaptiveMomentEstimation");

    add_xml_element(printer, "BatchSize", to_string(batch_size));
    write_common_xml(printer);

    printer.close_element();
}

void AdaptiveMomentEstimation::from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "AdaptiveMomentEstimation");

    set_batch_size(read_xml_index(root_element, "BatchSize"));
    read_common_xml(root_element);
}

REGISTER(Optimizer, AdaptiveMomentEstimation, "AdaptiveMomentEstimation");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
