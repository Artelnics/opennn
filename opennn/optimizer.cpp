//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "image_dataset.h"
#include "language_dataset.h"
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
#include "string_utilities.h"
#include <cctype>
#include <chrono>
#include <cstdlib>

#if defined(__linux__) || defined(__unix__)
#include <unistd.h>
#endif
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace opennn
{

namespace
{

bool env_flag_enabled(const char* name)
{
    const char* value = getenv(name);
    if (!value) return false;

    string text(value);
    ranges::transform(text, text.begin(),
                      [](unsigned char c) { return static_cast<char>(tolower(c)); });

    return text == "1" || text == "true" || text == "on" || text == "yes";
}

bool profile_enabled_from_env()
{
    static const bool enabled = env_flag_enabled("OPENNN_PROFILE");
    return enabled;
}

void sync_cuda_for_debug()
{
#ifdef OPENNN_HAS_CUDA
    static const bool enabled = env_flag_enabled("OPENNN_CUDA_DEBUG_SYNC");
    if (is_gpu() && enabled)
        CHECK_CUDA(cudaStreamSynchronize(Backend::get_compute_stream()));
#endif
}

#ifdef OPENNN_HAS_CUDA
bool cuda_sync_each_batch()
{
    static const bool enabled = env_flag_enabled("OPENNN_CUDA_SYNC_EACH_BATCH");
    return enabled;
}
#endif

#ifdef OPENNN_HAS_CUDA
struct DeviceEpochMetrics
{
    Buffer values{Device::CUDA};

    void reset()
    {
        values.grow_to(2 * Index(sizeof(float)));
        CHECK_CUDA(cudaMemsetAsync(values.data, 0, 2 * sizeof(float),
                                   Backend::get_compute_stream()));
    }

    float* error_sum() { return values.as<float>(); }
    float* accuracy_sum() { return values.as<float>() + 1; }

    Optimizer::EpochStats read(Index batches_number, bool include_accuracy)
    {
        float host[2] = {0.0f, 0.0f};
        CHECK_CUDA(cudaMemcpyAsync(host, values.data, sizeof(host),
                                   cudaMemcpyDeviceToHost,
                                   Backend::get_compute_stream()));
        CHECK_CUDA(cudaStreamSynchronize(Backend::get_compute_stream()));

        Optimizer::EpochStats stats;
        if (batches_number > 0)
        {
            stats.error = host[0] / float(batches_number);
            if (include_accuracy) stats.accuracy = host[1] / float(batches_number);
        }
        return stats;
    }
};
#endif

}

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
        throw runtime_error(format("Cannot open file: {}", file_name.string()));

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

void Optimizer::warn_dropped_samples(Index batch_size,
                                     Index samples_number,
                                     const char* context) const
{
    if (!display) return;
    if (batch_size <= 0 || samples_number <= 0) return;
    if (batch_size >= samples_number)           return;
    if (samples_number % batch_size == 0)       return;

    const Index lost = samples_number % batch_size;
    ostringstream pct;
    pct << fixed << setprecision(2)
        << (100.0 * double(lost) / double(samples_number));
    cout << "Warning: " << context << " batch_size " << batch_size
         << " does not divide " << samples_number << " samples. "
         << lost << " sample(s) (" << pct.str()
         << " % of total) dropped per epoch.\n";
}

Index Optimizer::get_maximum_batch_size() const
{
    if (!loss)
        throw runtime_error("Optimizer::get_maximum_batch_size: loss is not set.");

    const Dataset* dataset = loss->get_dataset();
    const NeuralNetwork* neural_network = loss->get_neural_network();

    if (!dataset)
        throw runtime_error("Optimizer::get_maximum_batch_size: dataset is not set.");
    if (!neural_network)
        throw runtime_error("Optimizer::get_maximum_batch_size: neural network is not set.");

    const Index training_samples_number = dataset->get_samples_number("Training");
    if (training_samples_number <= 0) return 0;
    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const bool on_gpu = is_gpu();

    // Available memory

    Index available_bytes = 0;
    if (on_gpu)
    {
#ifdef OPENNN_HAS_CUDA
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
            throw runtime_error("Optimizer::get_maximum_batch_size: cudaMemGetInfo failed.");
        available_bytes = Index(free_bytes);
#else
        throw runtime_error("Optimizer::get_maximum_batch_size: CUDA not compiled in.");
#endif
    }
    else
    {
#if defined(__linux__) || defined(__unix__)
        const long pages = sysconf(_SC_AVPHYS_PAGES);
        const long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages <= 0 || page_size <= 0)
            throw runtime_error("Optimizer::get_maximum_batch_size: sysconf failed to query available RAM.");
        available_bytes = Index(pages) * Index(page_size);
#elif defined(_WIN32)
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status))
            throw runtime_error("Optimizer::get_maximum_batch_size: GlobalMemoryStatusEx failed.");
        available_bytes = Index(status.ullAvailPhys);
#else
        throw runtime_error("Optimizer::get_maximum_batch_size: no portable API to query available RAM on this platform.");
#endif
    }

    const Index budget = Index(double(available_bytes) * 0.8);

    // Fixed (batch-independent) memory — use aligned shapes to match the
    // actual allocations done in NeuralNetwork/BackPropagation/OptimizerData.

    const Index parameters_number       = neural_network->get_parameters_number();
    const Index parameters_aligned_size = get_aligned_size(neural_network->get_parameter_specs());
    const Index slot_aligned_size       = get_aligned_size(parameters_number);
    const bool bf16_train = on_gpu && is_bf16_training();
    const bool bf16_input = bf16_train && dynamic_cast<const LanguageDataset*>(dataset) == nullptr;

    Index fixed_bytes = 0;
    // Parameters (FP32 master)
    fixed_bytes += parameters_aligned_size * Index(sizeof(float));
    // BF16 mirror (same element count, 2 bytes each)
    if (bf16_train) fixed_bytes += parameters_aligned_size * Index(sizeof(bfloat16));
    // States buffer
    fixed_bytes += neural_network->get_states_size() * Index(sizeof(float));
    // Gradient (FP32)
    fixed_bytes += parameters_aligned_size * Index(sizeof(float));
    // Optimizer slots: 2 × aligned(parameters_number) for Adam (worst case; SGD
    // with momentum uses 1, vanilla SGD uses 0).
    fixed_bytes += 2 * slot_aligned_size * Index(sizeof(float));

    if (fixed_bytes >= budget)
        throw runtime_error(format("Optimizer::get_maximum_batch_size: fixed memory ({} MiB) exceeds 80% budget ({} MiB).",
                                   fixed_bytes / (1ull << 20), budget / (1ull << 20)));

    const Index dynamic_budget = budget - fixed_bytes;

    // Per-batch memory (FP, BP, batch buffers in the training pool; plus
    // validation FP + pool when the chosen batch is larger than validation —
    // Adam/SGD allocate a separate FP and pool in that case).

    const int batch_pool_size = max(num_workers + 1, on_gpu ? 3 : 2);
    const Shape input_shape   = dataset->get_shape("Input");
    const Shape target_shape  = dataset->get_shape("Target");
    const Shape decoder_shape = dataset->get_shape("Decoder");

    const Shape output_shape = neural_network->get_output_shape();
    const Type compute_dtype = bf16_train ? Type::BF16 : Type::FP32;

    auto pool_bytes_for_batch = [&](Index b) -> Index {
        Index single_batch = 0;
        if (!input_shape.empty())
            single_batch += b * input_shape.size() * (bf16_input ? Index(sizeof(bfloat16))
                                                                 : Index(sizeof(float)));
        if (!target_shape.empty())
            single_batch += b * target_shape.size() * Index(sizeof(float));
        if (!decoder_shape.empty())
            single_batch += b * decoder_shape.size() * Index(sizeof(float));
        return Index(batch_pool_size) * single_batch;
    };

    auto bytes_for_run = [&](Index b) -> Index {
        if (b <= 0) return 0;

        const auto forward_specs  = neural_network->get_forward_specs(b);
        const auto backward_specs = neural_network->get_backward_specs(b);

        Index total = 0;
        total += get_aligned_bytes(forward_specs);
        total += get_aligned_bytes(backward_specs);

        // Output-delta slot (delta_views[last][0] = Shape{b}.append(output_shape)).
        // Approximation: per-layer extra deltas for multi-consumer branches are
        // ignored — they are rare and rely on graph traversal.
        if (!output_shape.empty())
        {
            const Index out_elems = b * output_shape.size();
            total += get_aligned_bytes(out_elems, compute_dtype);
        }

        total += pool_bytes_for_batch(b);

        // BF16 prefetch staging: a single FP32 buffer of input_elements.
        if (bf16_input && !input_shape.empty())
            total += get_aligned_bytes(b * input_shape.size(), Type::FP32);

        return total;
    };

    auto bytes_for_batch = [&](Index b) -> Index {
        Index total = bytes_for_run(b);

        // Adam/SGD allocate a separate validation FP + pool iff
        // validation_batch_size != training_batch_size. validation_batch_size
        // is min(b, validation_samples_number).
        if (validation_samples_number > 0 && b > validation_samples_number)
            total += bytes_for_run(validation_samples_number);

        return total;
    };

    if (bytes_for_batch(1) > dynamic_budget)
        throw runtime_error(format("Optimizer::get_maximum_batch_size: not enough memory for batch_size=1. "
                                   "Need {} MiB, available {} MiB.",
                                   bytes_for_batch(1) / (1ull << 20), dynamic_budget / (1ull << 20)));

    Index lo = 1;
    Index hi = training_samples_number;
    while (lo < hi)
    {
        const Index mid = lo + (hi - lo + 1) / 2;
        if (bytes_for_batch(mid) <= dynamic_budget) lo = mid;
        else                                        hi = mid - 1;
    }

    return lo;
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
                throw_if(!tabular_dataset, "Expected TabularDataset.");
                input_variable_scalers = tabular_dataset->get_feature_scalers("Input");
                input_variable_descriptives = tabular_dataset->scale_features("Input");
                scaling_layer->set_descriptives(input_variable_descriptives);
                scaling_layer->set_scalers(input_variable_scalers);
                break;
            }

            case 2:
            {
                auto* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);
                throw_if(!time_series_dataset, "Expected TimeSeriesDataset.");
                input_variable_scalers = time_series_dataset->get_feature_scalers("Input");
                input_variable_descriptives = time_series_dataset->scale_features("Input");
                scaling_layer->set_descriptives(input_variable_descriptives);
                scaling_layer->set_scalers(input_variable_scalers);
                break;
            }

            case 3:
            {
                auto* image_dataset = dynamic_cast<ImageDataset*>(dataset);
                throw_if(!image_dataset, "Expected ImageDataset.");
                image_dataset->scale_features("Input");
                scaling_layer->set_scalers("ImageMinMax");
                break;
            }

            default:
                throw runtime_error(format("Unexpected Scaling input rank: {}",
                                           scaling_layer->get_input_shape().rank));
        }
    }

    if (!neural_network->has(LayerType::Unscaling))
        return;

    // Unscaling layer

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    const bool has_pure_targets = ranges::any_of(target_feature_indices,
        [&](Index target_index) { return ranges::find(input_feature_indices, target_index) == input_feature_indices.end(); });

    vector<Descriptives> target_variable_descriptives;
    vector<string> target_variable_scalers;

    if (has_pure_targets)
    {
        auto* tabular_dataset = dynamic_cast<TabularDataset*>(dataset);
        throw_if(!tabular_dataset, "Expected TabularDataset for target unscaling.");
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

        auto it = ranges::find(input_feature_indices, target_index);

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
    throw_if(!unscaling_layer, "Expected Unscaling layer.");

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
        const bool is_input = ranges::find(input_indices, target_indices[i]) != input_indices.end();

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

void Optimizer::write_common_json(JsonWriter& printer) const
{
    write_json(printer, {
        {"LossGoal", to_string(training_loss_goal)},
        {"MaximumSelectionFailures", to_string(maximum_validation_failures)},
        {"MaximumEpochsNumber", to_string(maximum_epochs)},
        {"MaximumTime", to_string(maximum_time)}
    });
}

void Optimizer::read_common_json(const Json* root_element)
{
    set_loss_goal(read_json_type(root_element, "LossGoal"));
    set_maximum_validation_failures(read_json_index(root_element, "MaximumSelectionFailures"));
    set_maximum_epochs(read_json_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_json_type(root_element, "MaximumTime"));
}

TrainingResults::TrainingResults(const Index epochs_number)
{
    training_error_history = VectorR::Constant(epochs_number, -1.0f);
    validation_error_history = VectorR::Constant(epochs_number, -1.0f);
}

string TrainingResults::write_stopping_condition() const
{
    using enum Optimizer::StoppingCondition;
    switch (stopping_condition)
    {
    case None:
        return "None";

    case MinimumLossDecrease:
        return "Minimum loss decrease";

    case LossGoal:
        return "Loss goal";

    case MaximumSelectionErrorIncreases:
        return "Maximum selection error increases";

    case MaximumEpochsNumber:
        return "Maximum epochs number";

    case MaximumTime:
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

    if (!file)
        throw runtime_error(format("TrainingResults::save: cannot open {}", file_name.string()));

    for (Index i = 0; i < override_results.dimension(0); ++i)
        file << override_results(i,0) << "; " << override_results(i,1) << "\n";

    file.close();
}

void TrainingResults::print(const string &message) const
{
    const Index epochs_number = training_error_history.size();

    Index best_epoch = epochs_number - 1;
    if (validation_error_history.size() > 0)
    {
        float best_val = numeric_limits<float>::max();
        for (Index e = 0; e < validation_error_history.size(); ++e)
            if (validation_error_history(e) < best_val)
            {
                best_val = validation_error_history(e);
                best_epoch = e;
            }
    }

    cout << message << "\n"
         << "Training results" << "\n"
         << "Epochs number: " << epochs_number - 1 << "\n"
         << "Training error: " << training_error_history(best_epoch) << "\n";
    if (validation_error_history.size() > 0)
        cout << "Validation error: " << validation_error_history(best_epoch) << "\n";
    if (best_epoch != epochs_number - 1)
        cout << "Best epoch: " << best_epoch
             << " (restored parameters correspond to this epoch)\n";
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
    const Index total_bytes = get_aligned_bytes(slot_shapes, Type::FP32);

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
            cursor += get_aligned_bytes(shape.size(), Type::FP32);
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

    clear_batch_reuse_events();

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

    if (memory_stream) CHECK_CUDA(cudaStreamSynchronize(memory_stream));
    CHECK_CUDA(cudaStreamSynchronize(Backend::get_compute_stream()));

    clear_batch_reuse_events();

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

    cudaEvent_t& reuse_event = batch_reuse_events[&batch];
    if (!reuse_event)
        CHECK_CUDA(cudaEventCreateWithFlags(&reuse_event, cudaEventDisableTiming));

    if (batch_reuse_recorded.contains(&batch))
        CHECK_CUDA(cudaStreamWaitEvent(memory_stream, reuse_event, 0));

    float* fp32_staging = nullptr;
    if (batch.needs_fp32_staging)
    {
        prefetch_fp32_staging.grow_to(batch.get_input_elements() * Index(sizeof(float)));
        fp32_staging = prefetch_fp32_staging.as<float>();
    }

    batch.copy_device_async(sample_count, memory_stream, fp32_staging);
    CHECK_CUDA(cudaEventRecord(batch_ready_event[slot], memory_stream));
#else
    (void)batch; (void)sample_count; (void)slot;
#endif
}

void Optimizer::wait_prefetch(int slot)
{
#ifdef OPENNN_HAS_CUDA
    if (!is_gpu()) return;
    CHECK_CUDA(cudaStreamWaitEvent(Backend::get_compute_stream(), batch_ready_event[slot], 0));
#else
    (void)slot;
#endif
}

void Optimizer::record_batch_reuse(Batch& batch)
{
#ifdef OPENNN_HAS_CUDA
    if (!is_gpu()) return;

    cudaEvent_t& reuse_event = batch_reuse_events[&batch];
    if (!reuse_event)
        CHECK_CUDA(cudaEventCreateWithFlags(&reuse_event, cudaEventDisableTiming));

    CHECK_CUDA(cudaEventRecord(reuse_event, Backend::get_compute_stream()));
    batch_reuse_recorded.insert(&batch);
#else
    (void)batch;
#endif
}

void Optimizer::clear_batch_reuse_events()
{
#ifdef OPENNN_HAS_CUDA
    for (auto& [batch, event] : batch_reuse_events)
    {
        (void)batch;
        if (event) cudaEventDestroy(event);
    }
#endif
    batch_reuse_events.clear();
    batch_reuse_recorded.clear();
}

void Optimizer::sync_device()
{
#ifdef OPENNN_HAS_CUDA
    if (is_gpu() && cuda_sync_each_batch())
        CHECK_CUDA(cudaStreamSynchronize(Backend::get_compute_stream()));
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
        float* const squared_norm_ptr = squared_norm_device.as<float>();

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
                                  const vector<vector<Index>>& batches,
                                  const vector<Index>& input_feature_indices,
                                  const vector<Index>& decoder_feature_indices,
                                  const vector<Index>& target_feature_indices,
                                  const function<void(BackPropagation&)>& update,
                                  bool show_progress)
{
    EpochStats stats;

    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index batches_number = Index(batches.size());
    if (batches_number == 0) return stats;

#ifdef OPENNN_HAS_CUDA
    const bool use_device_metrics = is_gpu() && loss->supports_device_epoch_metrics();
    DeviceEpochMetrics device_metrics;
    if (use_device_metrics) device_metrics.reset();
#else
    const bool use_device_metrics = false;
#endif

    const bool profile_this = profile_enabled_from_env();
    if (profile_this)
    {
        ::opennn::enabled() = true;
        ::opennn::global_stats().clear();
    }
    const auto epoch_t0 = chrono::steady_clock::now();

    // Ordered consumption: each worker stores its filled batch at ready[iter].
    // The main loop polls ready[0], ready[1], ... — so the consumer sees the
    // same iteration order regardless of num_workers, preserving reproducibility.
    auto ready = make_unique<atomic<Batch*>[]>(batches_number);
    for (Index i = 0; i < batches_number; ++i) ready[i].store(nullptr);

    atomic<Index> next_iteration{0};
    // Accumulators for per-worker timing — written atomically so we avoid the
    // race that PROFILE_SCOPE would have on the global stats map.
    atomic<int64_t> worker_pop_us{0};
    atomic<int64_t> worker_fill_us{0};
    atomic<long>    worker_fills{0};

    vector<thread> workers;
    workers.reserve(num_workers);
    // Batch loading is already parallelized across workers. Avoid nested OpenMP
    // teams inside Dataset::fill_*; on MinGW this can corrupt the heap when
    // worker threads tear down.
    const bool parallelize_samples_within_batch = false;
    for (int w = 0; w < num_workers; ++w)
        workers.emplace_back([&]() {
            for (;;) {
                const Index it = next_iteration.fetch_add(1);
                if (it >= batches_number) return;
                const auto t_pop0 = chrono::steady_clock::now();
                Batch* batch = empty_queue.pop();
                const auto t_fill0 = chrono::steady_clock::now();
                batch->fill(batches[it],
                            input_feature_indices,
                            decoder_feature_indices,
                            target_feature_indices,
                            /*is_training=*/true,
                            parallelize_samples_within_batch);
                const auto t_fill1 = std::chrono::steady_clock::now();
                ready[it].store(batch, std::memory_order_release);

                if (profile_enabled_from_env())
                {
                    worker_pop_us.fetch_add(
                        chrono::duration_cast<chrono::microseconds>(t_fill0 - t_pop0).count(),
                        memory_order_relaxed);
                    worker_fill_us.fetch_add(
                        chrono::duration_cast<chrono::microseconds>(t_fill1 - t_fill0).count(),
                        memory_order_relaxed);
                    worker_fills.fetch_add(1, memory_order_relaxed);
                }
            }
        });

    auto wait_for_iteration = [&](Index it) -> Batch* {
        Batch* p = nullptr;
        while (!(p = ready[it].load(memory_order_acquire)))
            this_thread::yield();
        return p;
    };

    Batch* next_batch = nullptr;
    {
        PROFILE_SCOPE_HOST("step:wait_fill");
        next_batch = wait_for_iteration(0);
    }
    {
        PROFILE_SCOPE_HOST("step:prefetch_h2d_issue");
        prefetch_batch(*next_batch, next_batch->current_sample_count, 0);
    }

    // Repaint at most ~200 times across the epoch (one ~ every 0.5%) so the
    // bar advances visibly without spamming the console on long epochs.
    const Index progress_step = max(Index(1), batches_number / 200);
    if (show_progress) display_progress_bar(0, int(batches_number));

    for (Index iteration = 0; iteration < batches_number; ++iteration)
    {
        Batch* current_batch = next_batch;
        next_batch = nullptr;

        {
            PROFILE_SCOPE("step:wait_prefetch");
            wait_prefetch(iteration % 2);
        }

        if (iteration + 1 < batches_number)
        {
            {
                PROFILE_SCOPE_HOST("step:wait_fill");
                next_batch = wait_for_iteration(iteration + 1);
            }
            {
                PROFILE_SCOPE_HOST("step:prefetch_h2d_issue");
                prefetch_batch(*next_batch, next_batch->current_sample_count, (iteration + 1) % 2);
            }
        }

        {
            PROFILE_SCOPE("step:fwd_total");
            neural_network->forward_propagate(current_batch->get_inputs(), forward_propagation, true);
        }
        sync_cuda_for_debug();

        {
            PROFILE_SCOPE("step:bwd_total");
#ifdef OPENNN_HAS_CUDA
            if (use_device_metrics)
            {
                if (!loss->back_propagate_device_metrics(*current_batch,
                                                          forward_propagation,
                                                          back_propagation,
                                                          device_metrics.error_sum(),
                                                          is_classification ? device_metrics.accuracy_sum() : nullptr))
                    throw runtime_error("Device epoch metrics unexpectedly unsupported for this loss.");
            }
            else
#endif
            {
                loss->back_propagate(*current_batch, forward_propagation, back_propagation);
            }
        }
        sync_cuda_for_debug();

        record_batch_reuse(*current_batch);

        if (!use_device_metrics)
        {
            stats.error += back_propagation.error;
            if (is_classification) stats.accuracy += back_propagation.accuracy;
        }

        {
            PROFILE_SCOPE("step:optim_total");
            update(back_propagation);
        }
        sync_cuda_for_debug();

        {
            PROFILE_SCOPE("step:sync_device");
            sync_device();
        }

        empty_queue.push(current_batch);

        if (show_progress
            && ((iteration + 1) % progress_step == 0 || iteration + 1 == batches_number))
            display_progress_bar(int(iteration + 1), int(batches_number));
    }
    if (show_progress) cout << "\n";

    for (auto& w : workers) w.join();

#ifdef OPENNN_HAS_CUDA
    if (use_device_metrics)
    {
        stats = device_metrics.read(batches_number, is_classification);
        back_propagation.error = stats.error;
        back_propagation.accuracy = stats.accuracy;
        back_propagation.loss_value = stats.error;
    }
    else
#endif
    {
        stats.error /= float(batches_number);
        if (is_classification) stats.accuracy /= float(batches_number);
    }

    if (profile_this)
    {
        const auto epoch_t1 = chrono::steady_clock::now();
        const double epoch_ms = chrono::duration<double, milli>(epoch_t1 - epoch_t0).count();

        // Fold worker accumulators into the global stats so they show up in
        // the unified table. Sum across all workers; per-call divides by the
        // count of fill operations seen.
        if (const long w_calls = worker_fills.load(); w_calls > 0)
        {
            auto& fill_entry = ::opennn::global_stats().entries["worker:fill"];
            fill_entry.total_ms = double(worker_fill_us.load()) / 1000.0;
            fill_entry.calls    = w_calls;

            auto& wait_entry = ::opennn::global_stats().entries["worker:queue_wait"];
            wait_entry.total_ms = double(worker_pop_us.load()) / 1000.0;
            wait_entry.calls    = w_calls;
        }

        ::opennn::global_stats().print(cout, "Epoch breakdown (training)", epoch_ms);
        cout << "  Wall-clock epoch time: " << fixed << setprecision(2) << epoch_ms << " ms"
                  << " | num_workers=" << num_workers << "\n\n";
        // Keep the profiler enabled across epochs so the user can see
        // inter-epoch trends (data-loading hiding, cache warm-up, etc.).
        // Stats accumulate; reset them so the per-epoch table stays clean.
        ::opennn::global_stats().clear();
    }

    return stats;
}

Optimizer::EpochStats Optimizer::evaluate_epoch(bool is_classification,
                                     ForwardPropagation& forward_propagation,
                                     ThreadSafeQueue<Batch*>& empty_queue,
                                     const vector<vector<Index>>& batches,
                                     const vector<Index>& input_feature_indices,
                                     const vector<Index>& decoder_feature_indices,
                                     const vector<Index>& target_feature_indices)
{
    EpochStats stats;

    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index batches_number = Index(batches.size());
    if (batches_number == 0) return stats;

#ifdef OPENNN_HAS_CUDA
    const bool use_device_metrics = is_gpu() && loss->supports_device_epoch_metrics();
    DeviceEpochMetrics device_metrics;
    if (use_device_metrics) device_metrics.reset();
#else
    const bool use_device_metrics = false;
#endif

    auto ready = make_unique<atomic<Batch*>[]>(batches_number);
    for (Index i = 0; i < batches_number; ++i) ready[i].store(nullptr);

    atomic<Index> next_iteration{0};
    vector<thread> workers;
    workers.reserve(num_workers);
    const bool parallelize_samples_within_batch = false;
    for (int w = 0; w < num_workers; ++w)
        workers.emplace_back([&]() {
            for (;;) {
                const Index it = next_iteration.fetch_add(1);
                if (it >= batches_number) return;
                Batch* batch = empty_queue.pop();
                batch->fill(batches[it],
                            input_feature_indices,
                            decoder_feature_indices,
                            target_feature_indices,
                            /*is_training=*/false,
                            parallelize_samples_within_batch);
                ready[it].store(batch, std::memory_order_release);
            }
        });

    auto wait_for_iteration = [&](Index it) -> Batch* {
        Batch* p = nullptr;
        while (!(p = ready[it].load(memory_order_acquire)))
            this_thread::yield();
        return p;
    };

    Batch* next_batch = wait_for_iteration(0);
    prefetch_batch(*next_batch, next_batch->current_sample_count, 0);

    for (Index iteration = 0; iteration < batches_number; ++iteration)
    {
        Batch* current_batch = next_batch;
        next_batch = nullptr;

        wait_prefetch(iteration % 2);

        if (iteration + 1 < batches_number)
        {
            next_batch = wait_for_iteration(iteration + 1);
            prefetch_batch(*next_batch, next_batch->current_sample_count, (iteration + 1) % 2);
        }

        // is_training=false in evaluate_epoch: disables dropout and prevents
        // BatchNorm running stats from being updated with validation data.
        neural_network->forward_propagate(current_batch->get_inputs(), forward_propagation, false);
        sync_cuda_for_debug();
        Loss::EvaluationResult eval;
#ifdef OPENNN_HAS_CUDA
        if (use_device_metrics)
        {
            if (!loss->calculate_error_device_metrics(*current_batch,
                                                      forward_propagation,
                                                      device_metrics.error_sum(),
                                                      is_classification ? device_metrics.accuracy_sum() : nullptr))
                throw runtime_error("Device epoch metrics unexpectedly unsupported for this loss.");
        }
        else
#endif
        {
            eval = loss->calculate_error(*current_batch, forward_propagation);
        }
        sync_cuda_for_debug();

        record_batch_reuse(*current_batch);

        if (!use_device_metrics)
        {
            stats.error += eval.error;
            if (is_classification) stats.accuracy += eval.accuracy;
        }

        sync_device();

        empty_queue.push(current_batch);
    }

    for (auto& w : workers) w.join();

#ifdef OPENNN_HAS_CUDA
    if (use_device_metrics)
        stats = device_metrics.read(batches_number, is_classification);
    else
#endif
    {
        stats.error /= float(batches_number);
        if (is_classification) stats.accuracy /= float(batches_number);
    }

    return stats;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
