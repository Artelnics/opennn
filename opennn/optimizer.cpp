//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_dataset.h"
#include "tabular_dataset.h"
#include "time_series_dataset.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"
#include "loss.h"
#include "optimizer.h"
#include "variable.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "batch.h"
#include "neural_network.h"
#include "profiler.h"
#include <chrono>

namespace opennn
{

Optimizer::Optimizer(Loss* new_loss)
{
    set(new_loss);
}

void Optimizer::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Optimizer");

    add_json_field(printer, "Display", to_string(display));

    printer.close_element();
}

void Optimizer::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "Optimizer");

    set_display(read_json_bool(root_element, "Display"));
}

void Optimizer::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        throw runtime_error("Cannot open file: " + file_name.string());

    JsonWriter printer;
    to_JSON(printer);
    file << printer.c_str();
}

void Optimizer::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
}

float Optimizer::get_elapsed_time(const time_t &beginning_time)
{
    time_t current_time;
    time(&current_time);
    return float(difftime(current_time, beginning_time));
}

void Optimizer::set_names()
{
    const Dataset* dataset = loss->get_dataset();

    const vector<Variable> input_variables = dataset->get_variables("Input");
    const vector<Variable> target_variables = dataset->get_variables("Target");

    NeuralNetwork* neural_network = loss->get_neural_network();

    neural_network->set_input_variables(input_variables);
    neural_network->set_output_variables(target_variables);
}

void Optimizer::set_scaling()
{
    Dataset* dataset = loss->get_dataset();
    NeuralNetwork* neural_network = loss->get_neural_network();

    // Scaling layer

    vector<Descriptives> input_variable_descriptives;
    vector<string> input_variable_scalers;

    // Discriminate by the scaling layer's input rank: 1 = tabular,
    // 2 = time-series ([seq, feat]), 3 = image ([H, W, C]).
    if (auto* scaling_layer = dynamic_cast<Scaling*>(neural_network->get_first(LayerType::Scaling)))
    {
        switch (scaling_layer->get_input_shape().rank)
        {
            case 1:
            {
                auto* tabular_dataset = dynamic_cast<TabularDataset*>(dataset);
                if (!tabular_dataset) throw runtime_error("Expected TabularDataset.");
                input_variable_scalers = tabular_dataset->get_feature_scalers("Input");
                input_variable_descriptives = tabular_dataset->scale_features("Input");
                scaling_layer->set_descriptives(input_variable_descriptives);
                scaling_layer->set_scalers(input_variable_scalers);
                break;
            }

            case 2:
            {
                auto* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);
                if (!time_series_dataset) throw runtime_error("Expected TimeSeriesDataset.");
                input_variable_scalers = time_series_dataset->get_feature_scalers("Input");
                input_variable_descriptives = time_series_dataset->scale_features("Input");
                scaling_layer->set_descriptives(input_variable_descriptives);
                scaling_layer->set_scalers(input_variable_scalers);
                break;
            }

            case 3:
            {
                auto* image_dataset = dynamic_cast<ImageDataset*>(dataset);
                if (!image_dataset) throw runtime_error("Expected ImageDataset.");
                image_dataset->scale_features("Input");
                scaling_layer->set_scalers("ImageMinMax");
                break;
            }

            default:
                throw runtime_error("Unexpected Scaling input rank: "
                                    + to_string(scaling_layer->get_input_shape().rank));
        }
    }

    if (!neural_network->has(LayerType::Unscaling))
        return;

    // Unscaling layer

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    const bool has_pure_targets = any_of(target_feature_indices.begin(), target_feature_indices.end(),
        [&](Index target_index) { return find(input_feature_indices.begin(), input_feature_indices.end(), target_index) == input_feature_indices.end(); });

    vector<Descriptives> target_variable_descriptives;
    vector<string> target_variable_scalers;

    if (has_pure_targets)
    {
        auto* tabular_dataset = dynamic_cast<TabularDataset*>(dataset);
        if (!tabular_dataset) throw runtime_error("Expected TabularDataset for target unscaling.");
        target_variable_descriptives = tabular_dataset->scale_features("Target");
        target_variable_scalers = tabular_dataset->get_feature_scalers("Target");
    }

    vector<Descriptives> unscaling_layer_descriptives;
    vector<string> unscaling_layer_scalers;
    unscaling_layer_descriptives.reserve(target_feature_indices.size());
    unscaling_layer_scalers.reserve(target_feature_indices.size());

    for (size_t i = 0; i < target_feature_indices.size(); ++i)
    {
        const Index target_index = target_feature_indices[i];

        auto it = find(input_feature_indices.begin(), input_feature_indices.end(), target_index);

        if (it != input_feature_indices.end())
        {
            const Index input_pos = distance(input_feature_indices.begin(), it);

            unscaling_layer_descriptives.push_back(input_variable_descriptives[input_pos]);
            unscaling_layer_scalers.push_back(input_variable_scalers[input_pos]);
        }
        else
        {
            unscaling_layer_descriptives.push_back(target_variable_descriptives[i]);
            unscaling_layer_scalers.push_back(target_variable_scalers[i]);
        }
    }

    auto* unscaling_layer = dynamic_cast<Unscaling*>(neural_network->get_first(LayerType::Unscaling));
    if (!unscaling_layer) throw runtime_error("Expected Unscaling layer.");

    if (ssize(unscaling_layer_descriptives) != unscaling_layer->get_outputs_number())
        throw runtime_error("Unscaling setup error: Mismatch between number of target variables and unscaling layer neurons.");

    unscaling_layer->set_descriptives(unscaling_layer_descriptives);
    unscaling_layer->set_scalers(unscaling_layer_scalers);
}

void Optimizer::set_unscaling()
{
    Dataset* dataset = loss->get_dataset();
    NeuralNetwork* neural_network = loss->get_neural_network();

    auto reconstruct_descriptives = [](const VectorR& minimums, const VectorR& maximums,
                                       const VectorR& means, const VectorR& std_devs)
    {
        const Index descriptives_count = minimums.size();
        vector<Descriptives> descriptives(descriptives_count);
        for (Index i = 0; i < descriptives_count; ++i)
        {
            descriptives[i].minimum = minimums[i];
            descriptives[i].maximum = maximums[i];
            descriptives[i].mean = means[i];
            descriptives[i].standard_deviation = std_devs[i];
        }
        return descriptives;
    };

    // Mirror set_scaling's rank dispatch: tabular/time-series invert via descriptives,
    // image inverts at the dataset level (no descriptives kept).
    if (auto* layer = dynamic_cast<Scaling*>(neural_network->get_first(LayerType::Scaling)))
    {
        switch (layer->get_input_shape().rank)
        {
            case 1:
            case 2:
                dataset->unscale_features("Input",
                    reconstruct_descriptives(layer->get_minimums(), layer->get_maximums(),
                                              layer->get_means(), layer->get_standard_deviations()));
                break;

            case 3:
                if (auto* image_dataset = dynamic_cast<ImageDataset*>(dataset))
                    image_dataset->unscale_features("Input");
                break;
        }
    }

    if (!neural_network->has(LayerType::Unscaling))
        return;

    auto* unscaling_layer = dynamic_cast<Unscaling*>(neural_network->get_first(LayerType::Unscaling));
    if (!unscaling_layer) return;

    const vector<Descriptives> all_target_descriptives = reconstruct_descriptives(
        unscaling_layer->get_minimums(),
        unscaling_layer->get_maximums(),
        unscaling_layer->get_means(),
        unscaling_layer->get_standard_deviations());

    const vector<Index> input_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_indices = dataset->get_feature_indices("Target");

    vector<Descriptives> unscaled_targets_descriptives;

    for (size_t i = 0; i < target_indices.size(); ++i)
    {
        const bool is_input = find(input_indices.begin(), input_indices.end(), target_indices[i]) != input_indices.end();

        if (!is_input && i < all_target_descriptives.size())
            unscaled_targets_descriptives.push_back(all_target_descriptives[i]);
    }

    if (!unscaled_targets_descriptives.empty())
        dataset->unscale_features("Target", unscaled_targets_descriptives);
}

bool Optimizer::check_stopping_condition(TrainingResults& results,
                                          const Index epoch,
                                          const float elapsed_time,
                                          const float training_error,
                                          const Index validation_failures) const
{
    if (training_error < training_loss_goal)
    {
        if (display) cout << "Epoch " << epoch << "\nLoss goal reached: " << training_error << "\n";
        results.stopping_condition = StoppingCondition::LossGoal;
    }
    else if (validation_failures >= maximum_validation_failures)
    {
        if (display) cout << "Epoch " << epoch << "\nMaximum selection failures reached: " << validation_failures << "\n";
        results.stopping_condition = StoppingCondition::MaximumSelectionErrorIncreases;
    }
    else if (epoch == maximum_epochs)
    {
        if (display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch << "\n";
        results.stopping_condition = StoppingCondition::MaximumEpochsNumber;
    }
    else if (elapsed_time >= maximum_time)
    {
        if (display) cout << "Epoch " << epoch << "\nMaximum training time reached: " << get_time(elapsed_time) << "\n";
        results.stopping_condition = StoppingCondition::MaximumTime;
    }
    else
        return false;

    return true;
}

void Optimizer::write_common_xml(JsonWriter& printer) const
{
    write_json(printer, {
        {"LossGoal", to_string(training_loss_goal)},
        {"MaximumSelectionFailures", to_string(maximum_validation_failures)},
        {"MaximumEpochsNumber", to_string(maximum_epochs)},
        {"MaximumTime", to_string(maximum_time)}
    });
}

void Optimizer::read_common_xml(const Json* root_element)
{
    set_loss_goal(read_json_type(root_element, "LossGoal"));
    set_maximum_validation_failures(read_json_index(root_element, "MaximumSelectionFailures"));
    set_maximum_epochs(read_json_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_json_type(root_element, "MaximumTime"));
}

TrainingResults::TrainingResults(const Index epochs_number)
{
    training_error_history = VectorR::Constant(1 + epochs_number, -1.0f);
    validation_error_history = VectorR::Constant(1 + epochs_number, -1.0f);
}

string TrainingResults::write_stopping_condition() const
{
    switch (stopping_condition)
    {
    case Optimizer::StoppingCondition::None:
        return "None";

    case Optimizer::StoppingCondition::MinimumLossDecrease:
        return "Minimum loss decrease";

    case Optimizer::StoppingCondition::LossGoal:
        return "Loss goal";

    case Optimizer::StoppingCondition::MaximumSelectionErrorIncreases:
        return "Maximum selection error increases";

    case Optimizer::StoppingCondition::MaximumEpochsNumber:
        return "Maximum epochs number";

    case Optimizer::StoppingCondition::MaximumTime:
        return "Maximum training time";

    default:
        return string();
    }
}

float TrainingResults::get_training_error() const
{
    return training_error_history(training_error_history.size() - 1);
}

float TrainingResults::get_validation_error() const
{
    if (validation_error_history.size() == 0) return 0.0f;

    return validation_error_history(validation_error_history.size() - 1);
}

Index TrainingResults::get_epochs_number() const
{
    return training_error_history.size() - 1;
}

void TrainingResults::resize_training_error_history(const Index new_size)
{
    training_error_history.conservativeResize(new_size);
}

void TrainingResults::resize_validation_error_history(const Index new_size)
{
    validation_error_history.conservativeResize(new_size);
}

void TrainingResults::save(const filesystem::path& file_name) const
{
    const Tensor<string, 2> override_results = write_override_results();

    ofstream file(file_name);

    if (!file) return;

    for (Index i = 0; i < override_results.dimension(0); ++i)
        file << override_results(i,0) << "; " << override_results(i,1) << "\n";

    file.close();
}

void TrainingResults::print(const string &message) const
{
    const Index epochs_number = training_error_history.size();

    cout << message << "\n"
         << "Training results" << "\n"
         << "Epochs number: " << epochs_number - 1 << "\n"
         << "Training error: " << training_error_history(epochs_number - 1) << "\n";
    if (validation_error_history.size() > 0)
        cout << "Validation error: " << validation_error_history(epochs_number - 1) << "\n";
    cout << "Stopping condition: " << write_stopping_condition() << "\n";
}

Tensor<string, 2> TrainingResults::write_override_results(const Index precision) const
{
    Tensor<string, 2> override_results(5, 2);

    override_results(0, 0) = "Epochs number";
    override_results(1, 0) = "Elapsed time";
    override_results(2, 0) = "Stopping criterion";
    override_results(3, 0) = "Training error";
    override_results(4, 0) = "Validation error";

    const Index size = training_error_history.size();

    if (size == 0)
    {
        for (Index i = 0; i < 5; ++i)
            override_results(i, 1) = "NA";

        return override_results;
    }

    override_results(0, 1) = to_string(size - 1);
    override_results(1, 1) = elapsed_time;
    override_results(2, 1) = write_stopping_condition();
    override_results(3, 1) = to_string(training_error_history(size - 1));

    // Final selection error

    ostringstream buffer;

    if (validation_error_history.size() == 0)
        buffer << "NAN";
    else
        buffer << setprecision(precision) << validation_error_history(size - 1);

    override_results(4, 1) = buffer.str();

    return override_results;
}

void OptimizerData::print() const
{
    cout << "Potential parameters:" << "\n"
         << potential_parameters << "\n"
         << "Training direction:" << "\n"
         << training_direction << "\n"
         << "Initial learning rate:" << "\n"
         << initial_learning_rate << "\n";
}

void OptimizerData::set(const vector<Shape>& slot_shapes, Device device)
{
    const Index total_bytes = aligned_total_elements(slot_shapes) * Index(sizeof(float));

    if (total_bytes > 0)
    {
        data.resize_bytes(total_bytes, device);
        data.setZero();
    }

    views.clear();
    views.reserve(slot_shapes.size());

    uint8_t* cursor = (total_bytes > 0) ? data.as<uint8_t>() : nullptr;

    for (const Shape& shape : slot_shapes)
    {
        if (shape.size() > 0 && cursor)
        {
            views.emplace_back(cursor, shape, Type::FP32);
            cursor += get_aligned_bytes(shape.size() * Index(sizeof(float)));
        }
        else
        {
            views.emplace_back();
        }
    }
}

void Optimizer::setup_device_training()
{
#ifdef OPENNN_HAS_CUDA
    if (!is_gpu()) return;

    NeuralNetwork* neural_network = loss->get_neural_network();

    neural_network->copy_parameters_device();
    neural_network->copy_states_device();

    cudaStreamCreateWithFlags(&memory_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&batch_ready_event[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&batch_ready_event[1], cudaEventDisableTiming);
#endif
}

void Optimizer::teardown_device_training()
{
#ifdef OPENNN_HAS_CUDA
    if (!is_gpu()) return;

    cudaStreamDestroy(memory_stream);
    cudaEventDestroy(batch_ready_event[0]);
    cudaEventDestroy(batch_ready_event[1]);
    memory_stream = nullptr;
    batch_ready_event[0] = nullptr;
    batch_ready_event[1] = nullptr;

    NeuralNetwork* neural_network = loss->get_neural_network();
    neural_network->copy_parameters_host();
    neural_network->copy_states_host();
#endif
}

void Optimizer::prefetch_batch(Batch& batch, Index sample_count, int slot)
{
#ifdef OPENNN_HAS_CUDA
    if (!is_gpu()) return;
    batch.copy_device_async(sample_count, memory_stream);
    cudaEventRecord(batch_ready_event[slot], memory_stream);
#else
    (void)batch; (void)sample_count; (void)slot;
#endif
}

void Optimizer::wait_prefetch(int slot)
{
#ifdef OPENNN_HAS_CUDA
    if (!is_gpu()) return;
    cudaStreamWaitEvent(Backend::get_compute_stream(), batch_ready_event[slot], 0);
#else
    (void)slot;
#endif
}

void Optimizer::sync_device()
{
#ifdef OPENNN_HAS_CUDA
    if (is_gpu()) cudaStreamSynchronize(Backend::get_compute_stream());
#endif
}

void Optimizer::clip_gradient_norm(Buffer& gradient, float max_norm)
{
    const Index gradient_size = gradient.size_in_floats();
    if (gradient_size <= 0) return;

#ifdef OPENNN_HAS_CUDA
    if (is_gpu())
    {
        static Buffer squared_norm_device(Device::CUDA);
        squared_norm_device.grow_to(Index(sizeof(float)));
        float* squared_norm_ptr = squared_norm_device.as<float>();

        cublasHandle_t handle = Backend::get_cublas_handle();
        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        CHECK_CUBLAS(cublasSdot(handle,
                                to_int(gradient_size),
                                gradient.as<float>(), 1,
                                gradient.as<float>(), 1,
                                squared_norm_ptr));
        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

        clip_gradient_norm_cuda(gradient_size, gradient.as<float>(), squared_norm_ptr, max_norm, GRADIENT_NORM_EPS);
        return;
    }
#endif

    VectorMap gradient_view(gradient.as<float>(), gradient.size_in_floats());
    const float gradient_norm = gradient_view.norm();
    if (gradient_norm > max_norm)
        gradient_view *= max_norm / (gradient_norm + GRADIENT_NORM_EPS);
}

Optimizer::EpochStats Optimizer::train_epoch(bool is_classification,
                                  ForwardPropagation& forward_propagation,
                                  BackPropagation& back_propagation,
                                  ThreadSafeQueue<Batch*>& empty_queue,
                                  ThreadSafeQueue<Batch*>& ready_queue,
                                  const vector<vector<Index>>& batches,
                                  const vector<Index>& input_feature_indices,
                                  const vector<Index>& decoder_feature_indices,
                                  const vector<Index>& target_feature_indices,
                                  const std::function<void(BackPropagation&)>& update)
{
    EpochStats stats;

    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index batches_number = Index(batches.size());
    if (batches_number == 0) return stats;

    // Profiler disabled by default; flip to true to capture per-section timings.
    const bool profile_this = false;
    if (profile_this)
    {
        ::opennn::profiler::enabled() = true;
        ::opennn::profiler::global_stats().clear();
    }
    const auto epoch_t0 = std::chrono::steady_clock::now();

    std::thread worker([&]()
    {
        for (Index iteration = 0; iteration < batches_number; ++iteration)
        {
            Batch* batch = empty_queue.pop();
            batch->fill(batches[iteration],
                        input_feature_indices,
                        decoder_feature_indices,
                        target_feature_indices,
                        true);
            ready_queue.push(batch);
        }
    });

    Batch* next_batch = ready_queue.pop();
    prefetch_batch(*next_batch, batches[0].size(), 0);

    for (Index iteration = 0; iteration < batches_number; ++iteration)
    {
        Batch* current_batch = next_batch;
        next_batch = nullptr;

        wait_prefetch(iteration % 2);

        if (iteration + 1 < batches_number)
        {
            next_batch = ready_queue.pop();
            prefetch_batch(*next_batch, batches[iteration + 1].size(), (iteration + 1) % 2);
        }

        neural_network->forward_propagate(current_batch->get_inputs(), forward_propagation, true);

        loss->back_propagate(*current_batch, forward_propagation, back_propagation);

        stats.error += back_propagation.error;
        if (is_classification) stats.accuracy += back_propagation.accuracy;

        update(back_propagation);

        sync_device();

        empty_queue.push(current_batch);
    }

    worker.join();

    stats.error /= float(batches_number);
    if (is_classification) stats.accuracy /= float(batches_number);

    if (profile_this)
    {
        const auto epoch_t1 = std::chrono::steady_clock::now();
        const double epoch_ms = std::chrono::duration<double, std::milli>(epoch_t1 - epoch_t0).count();
        ::opennn::profiler::global_stats().print(std::cout, "Epoch breakdown (training)", epoch_ms);
        std::cout << "  Wall-clock epoch time: " << std::fixed << std::setprecision(2) << epoch_ms << " ms\n\n";
        ::opennn::profiler::enabled() = false;
    }

    return stats;
}

Optimizer::EpochStats Optimizer::evaluate_epoch(bool is_classification,
                                     ForwardPropagation& forward_propagation,
                                     ThreadSafeQueue<Batch*>& empty_queue,
                                     ThreadSafeQueue<Batch*>& ready_queue,
                                     const vector<vector<Index>>& batches,
                                     const vector<Index>& input_feature_indices,
                                     const vector<Index>& decoder_feature_indices,
                                     const vector<Index>& target_feature_indices)
{
    EpochStats stats;

    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index batches_number = Index(batches.size());
    if (batches_number == 0) return stats;

    std::thread worker([&]()
    {
        for (Index iteration = 0; iteration < batches_number; ++iteration)
        {
            Batch* batch = empty_queue.pop();
            batch->fill(batches[iteration],
                        input_feature_indices,
                        decoder_feature_indices,
                        target_feature_indices,
                        false);
            ready_queue.push(batch);
        }
    });

    Batch* next_batch = ready_queue.pop();
    prefetch_batch(*next_batch, batches[0].size(), 0);

    for (Index iteration = 0; iteration < batches_number; ++iteration)
    {
        Batch* current_batch = next_batch;
        next_batch = nullptr;

        wait_prefetch(iteration % 2);

        if (iteration + 1 < batches_number)
        {
            next_batch = ready_queue.pop();
            prefetch_batch(*next_batch, batches[iteration + 1].size(), (iteration + 1) % 2);
        }

        neural_network->forward_propagate(current_batch->get_inputs(), forward_propagation, true);
        const Loss::EvaluationResult eval = loss->calculate_error(*current_batch, forward_propagation);

        stats.error += eval.error;
        if (is_classification) stats.accuracy += eval.accuracy;

        sync_device();

        empty_queue.push(current_batch);
    }

    worker.join();

    stats.error /= float(batches_number);
    if (is_classification) stats.accuracy /= float(batches_number);

    return stats;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
