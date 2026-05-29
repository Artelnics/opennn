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

BatchFillSession::BatchFillSession(Index batches_number)
    : ready(make_unique<atomic<Batch*>[]>(batches_number))
{
    for (Index i = 0; i < batches_number; ++i)
        ready[i].store(nullptr);
}

BatchFillSession::~BatchFillSession()
{
    // If the consumer abandoned the session mid-epoch (e.g. threw out of
    // train_epoch), workers may still be blocked in empty_queue.pop() or about
    // to claim a fresh batch. Signal cancellation and wake any blocked pops,
    // then join all workers before the queue is reopened for the next epoch.
    cancelled.store(true, memory_order_release);
    if (empty_queue) empty_queue->close();

    workers.clear();   // ~jthread joins each worker

    if (empty_queue) empty_queue->reopen();
}

void BatchFillSession::rethrow_if_error()
{
    if (!error_pending.load(memory_order_acquire)) return;

    exception_ptr e;
    {
        lock_guard<mutex> elock(error_mutex);
        swap(e, worker_error);
        error_pending.store(false, memory_order_release);
    }
    if (e) rethrow_exception(e);
}

// --- Optimizer ---------------------------------------------------------------

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

void sync_cuda_for_debug(bool on_gpu)
{
#ifdef OPENNN_HAS_CUDA
    static const bool enabled = env_flag_enabled("OPENNN_CUDA_DEBUG_SYNC");
    if (on_gpu && enabled)
        CHECK_CUDA(cudaStreamSynchronize(Backend::get_compute_stream()));
#else
    (void)on_gpu;
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
    cout << format("Warning: {} batch_size {} does not divide {} samples. "
                   "{} sample(s) ({:.2f} % of total) dropped per epoch.\n",
                   context, batch_size, samples_number,
                   lost, 100.0 * double(lost) / double(samples_number));
}

ThreadSafeQueue<Batch*>& Optimizer::BatchPools::validation_queue()
{
    return validation_uses_training_pool
        ? training_empty_queue
        : validation_empty_queue;
}

vector<Batch*> Optimizer::BatchPools::all_batches() const
{
    vector<Batch*> batches;
    batches.reserve(training_pool.size() + validation_pool.size());

    for (const auto& batch : training_pool)
        batches.push_back(batch.get());
    for (const auto& batch : validation_pool)
        batches.push_back(batch.get());

    return batches;
}

void Optimizer::setup_batch_pools(BatchPools& pools,
                                  Dataset& dataset,
                                  NeuralNetwork& neural_network,
                                  Index training_batch_size,
                                  Index validation_batch_size,
                                  bool has_validation)
{
    apply_effective_num_workers(neural_network);

    const int pool_size = get_batch_pool_size(neural_network);
    const auto& config = neural_network.get_config();

    auto fill_pool = [&](ThreadSafeQueue<Batch*>& queue,
                         vector<unique_ptr<Batch>>& pool,
                         Index batch_size)
    {
        for (int i = 0; i < pool_size; ++i)
        {
            pool.push_back(make_unique<Batch>(batch_size, &dataset, config));
            queue.push(pool.back().get());
        }
    };

    fill_pool(pools.training_empty_queue,
              pools.training_pool,
              training_batch_size);

    pools.validation_uses_training_pool =
        has_validation && validation_batch_size == training_batch_size;

    if (has_validation && !pools.validation_uses_training_pool)
        fill_pool(pools.validation_empty_queue,
                  pools.validation_pool,
                  validation_batch_size);
}

unique_ptr<BatchFillSession> Optimizer::start_batch_workers(
    ThreadSafeQueue<Batch*>& empty_queue,
    const vector<vector<Index>>& batches,
    const vector<Index>& input_feature_indices,
    const vector<Index>& decoder_feature_indices,
    const vector<Index>& target_feature_indices,
    bool is_training,
    WorkerProfileCounters* profile_counters)
{
    const Index batches_number = Index(batches.size());

    auto session = make_unique<BatchFillSession>(batches_number);
    BatchFillSession* const session_ptr = session.get();
    session->empty_queue = &empty_queue;

    auto worker_body = [&, session_ptr, is_training, profile_counters]()
    {
        try
        {
            for (;;)
            {
                const auto t_pop0 = chrono::steady_clock::now();
                Batch* batch = empty_queue.pop();
                const auto t_fill0 = chrono::steady_clock::now();

                // Cancellation: pop returns nullptr when the queue was closed
                // by ~BatchFillSession, or the flag was flipped after we already
                // held a real batch. Either way, stop draining iterations.
                if (!batch || session_ptr->cancelled.load(memory_order_acquire))
                {
                    if (batch) empty_queue.push(batch);
                    return;
                }

                const Index it = session_ptr->next_iteration.fetch_add(1);
                if (it >= batches_number)
                {
                    empty_queue.push(batch);
                    return;
                }

                batch->wait_h2d_complete();
                batch->fill(batches[size_t(it)],
                            input_feature_indices,
                            decoder_feature_indices,
                            target_feature_indices,
                            is_training,
                            /*parallelize_samples_within_batch=*/false);

                const auto t_fill1 = chrono::steady_clock::now();
                session_ptr->ready[it].store(batch, memory_order_release);

                if (profile_counters)
                {
                    profile_counters->pop_us.fetch_add(
                        chrono::duration_cast<chrono::microseconds>(t_fill0 - t_pop0).count(),
                        memory_order_relaxed);
                    profile_counters->fill_us.fetch_add(
                        chrono::duration_cast<chrono::microseconds>(t_fill1 - t_fill0).count(),
                        memory_order_relaxed);
                    profile_counters->fills.fetch_add(1, memory_order_relaxed);
                }
            }
        }
        catch (...)
        {
            lock_guard<mutex> elock(session_ptr->error_mutex);
            if (!session_ptr->worker_error)
                session_ptr->worker_error = current_exception();
            session_ptr->error_pending.store(true, memory_order_release);
        }
    };

    session->workers.reserve(num_workers);
    for (int w = 0; w < num_workers; ++w)
        session->workers.emplace_back(worker_body);

    return session;
}

Batch* Optimizer::wait_for_filled_batch(BatchFillSession& session, Index iteration)
{
    Batch* batch = nullptr;
    while (!(batch = session.ready[iteration].load(memory_order_acquire)))
    {
        session.rethrow_if_error();
        this_thread::yield();
    }
    return batch;
}

int Optimizer::get_effective_num_workers(const NeuralNetwork& neural_network) const
{
    if (neural_network.is_gpu()
        && neural_network.has(LayerType::Recurrent)
        && num_workers > 1)
        return 1;

    return num_workers;
}

int Optimizer::get_batch_pool_size(const NeuralNetwork& neural_network) const
{
    return neural_network.is_gpu()
        ? max(get_effective_num_workers(neural_network) + 1, 3)
        : 1;
}

void Optimizer::apply_effective_num_workers(const NeuralNetwork& neural_network)
{
    num_workers = get_effective_num_workers(neural_network);
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

    const bool on_gpu = neural_network->is_gpu();

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


    const Index parameters_number       = neural_network->get_parameters_number();
    const Index parameters_aligned_size = get_aligned_size(neural_network->get_parameter_specs());
    const Index slot_aligned_size       = get_aligned_size(parameters_number);
    const bool bf16_train = on_gpu && neural_network->get_training_type() == Type::BF16;
    const bool bf16_input = bf16_train && dynamic_cast<const LanguageDataset*>(dataset) == nullptr;

    Index fixed_bytes = 0;

    fixed_bytes += parameters_aligned_size * Index(sizeof(float));
    if (bf16_train) fixed_bytes += parameters_aligned_size * Index(sizeof(bfloat16));
    fixed_bytes += neural_network->get_states_size() * Index(sizeof(float));
    fixed_bytes += parameters_aligned_size * Index(sizeof(float));
    fixed_bytes += 2 * slot_aligned_size * Index(sizeof(float));

    if (fixed_bytes >= budget)
        throw runtime_error(format("Optimizer::get_maximum_batch_size: fixed memory ({} MiB) exceeds 80% budget ({} MiB).",
                                   fixed_bytes / (1ull << 20), budget / (1ull << 20)));

    const Index dynamic_budget = budget - fixed_bytes;

    const int batch_pool_size = get_batch_pool_size(*neural_network);
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

        if (!output_shape.empty())
        {
            const Index out_elems = b * output_shape.size();
            total += get_aligned_bytes(out_elems, compute_dtype);
        }

        total += pool_bytes_for_batch(b);

        if (bf16_input && !input_shape.empty())
            total += get_aligned_bytes(b * input_shape.size(), Type::FP32);

        return total;
    };

    auto bytes_for_batch = [&](Index b) -> Index {
        Index total = bytes_for_run(b);

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

                image_dataset->set_input_scaling(scaling_layer->get_descriptives(),
                                                 scaling_layer->get_scalers(),
                                                 scaling_layer->get_min_range(),
                                                 scaling_layer->get_max_range());
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

    const Index unscaling_outputs = unscaling_layer->get_outputs_number();

    if (auto* ts_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);
        ts_dataset && ts_dataset->get_multi_target()
        && unscaling_outputs > ssize(unscaling_layer_descriptives))
    {
        const Index n_targets = ssize(unscaling_layer_descriptives);
        const Index future_steps = ts_dataset->get_future_time_steps();

        if (n_targets > 0 && unscaling_outputs == n_targets * future_steps)
        {
            vector<Descriptives> expanded_desc;
            vector<string> expanded_scalers;
            expanded_desc.reserve(unscaling_outputs);
            expanded_scalers.reserve(unscaling_outputs);
            for (Index c = 0; c < n_targets; ++c)
                for (Index k = 0; k < future_steps; ++k)
                {
                    expanded_desc.push_back(unscaling_layer_descriptives[c]);
                    expanded_scalers.push_back(unscaling_layer_scalers[c]);
                }
            unscaling_layer_descriptives = move(expanded_desc);
            unscaling_layer_scalers      = move(expanded_scalers);
        }
    }

    if (ssize(unscaling_layer_descriptives) != unscaling_outputs)
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
        if (display) cout << "Epoch " << epoch << "\nMaximum validation failures reached: " << validation_failures << "\n";
        results.stopping_condition = StoppingCondition::MaximumValidationErrorIncreases;
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
        {"MaximumValidationFailures", to_string(maximum_validation_failures)},
        {"MaximumEpochsNumber", to_string(maximum_epochs)},
        {"MaximumTime", to_string(maximum_time)}
    });
}

void Optimizer::read_common_json(const Json* root_element)
{
    set_loss_goal(read_json_type(root_element, "LossGoal"));
    set_maximum_validation_failures(read_json_index(root_element,
        root_element->has("MaximumValidationFailures") ? "MaximumValidationFailures" : "MaximumSelectionFailures"));
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

    case MaximumValidationErrorIncreases:
        return "Maximum validation error increases";

    case MaximumEpochsNumber:
        return "Maximum epochs number";

    case MaximumTime:
        return "Maximum training time";

    default:
        return {};
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
    const Index final_epoch = epochs_number - 1;

    Index best_epoch = final_epoch;
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

    const bool restored_best_epoch = restored_best_parameters
        && restored_epoch >= 0
        && restored_epoch < epochs_number;

    const Index reported_epoch = restored_best_epoch ? restored_epoch : final_epoch;

    cout << message << "\n"
         << "Training results" << "\n"
         << "Epochs number: " << final_epoch << "\n"
         << "Training error: " << training_error_history(reported_epoch) << "\n";
    if (validation_error_history.size() > 0)
        cout << "Validation error: " << validation_error_history(reported_epoch) << "\n";
    if (validation_error_history.size() > 0 && best_epoch != final_epoch)
    {
        if (restored_best_epoch)
            cout << "Best epoch: " << restored_epoch
                 << " (restored parameters and states correspond to this epoch)\n";
        else
            cout << "Best validation epoch: " << best_epoch
                 << " (final parameters correspond to epoch " << final_epoch << ")\n";
    }
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

    // Final validation error

    override_results(4, 1) = validation_error_history.size() == 0
        ? "NAN"
        : format("{:.{}g}", validation_error_history(size - 1), precision);

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
#ifdef OPENNN_HAS_CUDA
        if (device == Device::CUDA)
            CHECK_CUDA(cudaMemsetAsync(data.data, 0, total_bytes, Backend::get_compute_stream()));
        else
#endif
        data.setZero();
    }

    views.clear();
    views.reserve(slot_shapes.size());

    uint8_t* cursor = (total_bytes > 0) ? data.as<uint8_t>() : nullptr;

    for (const Shape& shape : slot_shapes)
    {
        if (shape.size() > 0 && cursor)
        {
            views.emplace_back(cursor, shape, Type::FP32, data.device_type);
            cursor += get_aligned_bytes(shape.size(), Type::FP32);
        }
        else
        {
            views.emplace_back();
        }
    }
}

void Optimizer::setup_device_training(const vector<Batch*>& batches)
{
    (void)batches;
#ifdef OPENNN_HAS_CUDA
    NeuralNetwork* neural_network = loss->get_neural_network();
    if (!neural_network->is_gpu()) return;

    neural_network->copy_parameters_device();
    neural_network->copy_states_device();

    setup_resident_datasets();
#endif
}

#ifdef OPENNN_HAS_CUDA

void Optimizer::setup_resident_datasets()
{
    resident_train_      = {};
    resident_validation_ = {};

    if (const char* e = getenv("OPENNN_DISABLE_RESIDENT"); e && e[0] == '1') return;

    Dataset* dataset = loss->get_dataset();
    NeuralNetwork* neural_network = loss->get_neural_network();
    if (!dataset || !neural_network) return;
    if (!neural_network->is_gpu()) return;

    if (!dataset->supports_device_residency()) return;
    if (neural_network->has(LayerType::Recurrent)
        || neural_network->has(LayerType::LongShortTermMemory)) return;

    const Index n_train = dataset->get_samples_number("Training");
    const Index n_val   = dataset->has_validation() ? dataset->get_samples_number("Validation") : 0;
    const Index in_f    = Index(dataset->get_feature_indices("Input").size());
    const Index tgt_f   = Index(dataset->get_feature_indices("Target").size());
    if (n_train == 0 || in_f == 0) return;

    const Index resident_bytes =
        (n_train + n_val) * (in_f + tgt_f) * Index(sizeof(float))
        + (n_train + n_val) * Index(sizeof(Index));
    size_t free_bytes = 0, total_bytes = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    const Index margin = Index(512) << 20;   // 512 MB headroom
    if (resident_bytes + margin > Index(free_bytes)) return;

    if (dataset->ensure_device_resident("Training"))
    {
        resident_train_.data   = dataset->get_device_resident("Training");
        resident_train_.active = (resident_train_.data != nullptr);
    }
    if (n_val > 0 && dataset->ensure_device_resident("Validation"))
    {
        resident_validation_.data   = dataset->get_device_resident("Validation");
        resident_validation_.active = (resident_validation_.data != nullptr);
    }

    if (display && resident_train_.active)
        cout << "[GPU] training dataset resident on device ("
             << (resident_train_.data->rows * (in_f + tgt_f) * Index(sizeof(float))) / (1ull << 20)
             << " MB); batches gathered on-device.\n";
}

void Optimizer::upload_resident_epoch_indices(ResidentEpochState& state,
                                              const vector<vector<Index>>& batches,
                                              Index batch_size)
{
    const Index num_batches = Index(batches.size());
    if (num_batches == 0 || batch_size == 0 || !state.data) return;

    const vector<Index>& row_of = state.data->row_of;
    vector<Index> all_rows(size_t(num_batches) * size_t(batch_size));
    for (Index b = 0; b < num_batches; ++b)
    {
        const vector<Index>& idx = batches[size_t(b)];
        Index* dst = all_rows.data() + size_t(b) * size_t(batch_size);
        for (Index j = 0; j < Index(idx.size()); ++j)
            dst[j] = row_of[size_t(idx[size_t(j)])];
    }

    state.row_index_buffer.resize_bytes(Index(all_rows.size()) * Index(sizeof(Index)), Device::CUDA);
    cudaStream_t stream = Backend::get_compute_stream();
    CHECK_CUDA(cudaMemcpyAsync(state.row_index_buffer.data, all_rows.data(),
                               all_rows.size() * sizeof(Index), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));   // host all_rows freed on return
}

#endif

void Optimizer::teardown_device_training()
{
#ifdef OPENNN_HAS_CUDA
    NeuralNetwork* neural_network = loss->get_neural_network();
    if (!neural_network->is_gpu()) return;

    CHECK_CUDA(cudaStreamSynchronize(Backend::get_compute_stream()));

    neural_network->copy_parameters_host();
    neural_network->copy_states_host();
#endif
}

void Optimizer::prefetch_batch(Batch& batch)
{
#ifdef OPENNN_HAS_CUDA
    if (!batch.uses_cuda()) return;

    batch.copy_device_async(Backend::get_compute_stream());
#else
    (void)batch;
#endif
}

void Optimizer::sync_device(bool on_gpu)
{
#ifdef OPENNN_HAS_CUDA
    if (on_gpu && (has_recurrent_layers_ || cuda_sync_each_batch()))
        CHECK_CUDA(cudaStreamSynchronize(Backend::get_compute_stream()));
#else
    (void)on_gpu;
#endif
}

void Optimizer::clip_gradient_norm(Buffer& gradient, float max_norm)
{
    const Index gradient_size = gradient.size_in_floats();
    if (gradient_size <= 0) return;

#ifdef OPENNN_HAS_CUDA
    if (gradient.device_type == Device::CUDA)
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

    has_recurrent_layers_ = neural_network->has(LayerType::Recurrent);
    const bool on_gpu = neural_network->is_gpu();

    auto set_epoch_loss = [&]()
    {
        const TensorView parameters(neural_network->get_parameters_data(),
                                    {neural_network->get_parameters_size()},
                                    Type::FP32,
                                    neural_network->get_device());
        back_propagation.regularization = loss->calculate_regularization(parameters);
        back_propagation.loss = stats.error + back_propagation.regularization;
    };

#ifdef OPENNN_HAS_CUDA
    const bool use_device_metrics = on_gpu && loss->supports_device_epoch_metrics();
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

    if (!on_gpu)
    {
        Batch* batch = empty_queue.pop();
        const Index progress_step = max(Index(1), batches_number / 200);
        if (show_progress) display_progress_bar(0, int(batches_number));

        for (Index iteration = 0; iteration < batches_number; ++iteration)
        {
            {
                PROFILE_SCOPE_HOST("step:fill");
                batch->fill(batches[iteration],
                            input_feature_indices,
                            decoder_feature_indices,
                            target_feature_indices,
                            /*is_training=*/true,
                            /*parallelize_samples=*/true);
            }

            {
                PROFILE_SCOPE("step:fwd_total");
                neural_network->forward_propagate(batch->get_inputs(), forward_propagation, true);
            }

            {
                PROFILE_SCOPE("step:bwd_total");
                loss->back_propagate(*batch, forward_propagation, back_propagation);
            }

            stats.error += back_propagation.error;
            if (is_classification) stats.accuracy += back_propagation.accuracy;

            {
                PROFILE_SCOPE("step:optim_total");
                update(back_propagation);
            }

            if (show_progress
                && ((iteration + 1) % progress_step == 0 || iteration + 1 == batches_number))
                display_progress_bar(int(iteration + 1), int(batches_number));
        }

        if (show_progress) cout << "\n";
        empty_queue.push(batch);

        stats.error /= float(batches_number);
        if (is_classification) stats.accuracy /= float(batches_number);
        set_epoch_loss();

        if (profile_this)
        {
            const auto epoch_t1 = chrono::steady_clock::now();
            const double epoch_ms = chrono::duration<double, milli>(epoch_t1 - epoch_t0).count();
            ::opennn::global_stats().print(cout, "Epoch breakdown (training)", epoch_ms);
            cout << "  Wall-clock epoch time: " << fixed << setprecision(2) << epoch_ms << " ms"
                 << " | num_workers=0\n\n";
            ::opennn::global_stats().clear();
        }

        return stats;
    }

#ifdef OPENNN_HAS_CUDA
    if (resident_train_.active)
    {
        ResidentEpochState& state = resident_train_;
        const Index batch_size = Index(batches[0].size());
        upload_resident_epoch_indices(state, batches, batch_size);

        Batch* batch = empty_queue.pop();
        const Index progress_step = max(Index(1), batches_number / 200);
        if (show_progress) display_progress_bar(0, int(batches_number));

        for (Index iteration = 0; iteration < batches_number; ++iteration)
        {
            {
                PROFILE_SCOPE("step:gather");
                batch->gather_device_async(batch_size,
                                           state.row_index_buffer.as<Index>() + iteration * batch_size,
                                           state.data->input.as<float>(),  state.data->input_features,
                                           state.data->target.as<float>(), state.data->target_features);
            }
            {
                PROFILE_SCOPE("step:fwd_total");
                neural_network->forward_propagate(batch->get_inputs(), forward_propagation, true);
            }
            {
                PROFILE_SCOPE("step:bwd_total");
                if (use_device_metrics)
                {
                    if (!loss->back_propagate_device_metrics(*batch, forward_propagation, back_propagation,
                                                             device_metrics.error_sum(),
                                                             is_classification ? device_metrics.accuracy_sum() : nullptr))
                        throw runtime_error("Device epoch metrics unexpectedly unsupported for this loss.");
                }
                else
                    loss->back_propagate(*batch, forward_propagation, back_propagation);
            }

            if (!use_device_metrics)
            {
                stats.error += back_propagation.error;
                if (is_classification) stats.accuracy += back_propagation.accuracy;
            }

            {
                PROFILE_SCOPE("step:optim_total");
                update(back_propagation);
            }
            {
                PROFILE_SCOPE("step:sync_device");
                sync_device(on_gpu);
            }

            if (show_progress
                && ((iteration + 1) % progress_step == 0 || iteration + 1 == batches_number))
                display_progress_bar(int(iteration + 1), int(batches_number));
        }
        if (show_progress) cout << "\n";
        empty_queue.push(batch);

        if (use_device_metrics)
        {
            stats = device_metrics.read(batches_number, is_classification);
            back_propagation.error = stats.error;
            back_propagation.accuracy = stats.accuracy;
            set_epoch_loss();
        }
        else
        {
            stats.error /= float(batches_number);
            if (is_classification) stats.accuracy /= float(batches_number);
            set_epoch_loss();
        }

        if (profile_this)
        {
            const auto epoch_t1 = chrono::steady_clock::now();
            const double epoch_ms = chrono::duration<double, milli>(epoch_t1 - epoch_t0).count();
            ::opennn::global_stats().print(cout, "Epoch breakdown (training, resident)", epoch_ms);
            cout << "  Wall-clock epoch time: " << fixed << setprecision(2) << epoch_ms
                 << " ms | resident=on\n\n";
            ::opennn::global_stats().clear();
        }

        return stats;
    }
#endif

    WorkerProfileCounters worker_profile;
    auto session = start_batch_workers(empty_queue,
                                       batches,
                                       input_feature_indices,
                                       decoder_feature_indices,
                                       target_feature_indices,
                                       /*is_training=*/true,
                                       profile_this ? &worker_profile : nullptr);

    Batch* next_batch = nullptr;
    {
        PROFILE_SCOPE_HOST("step:wait_fill");
        next_batch = wait_for_filled_batch(*session, 0);
    }
    {
        PROFILE_SCOPE_HOST("step:prefetch_h2d_issue");
        prefetch_batch(*next_batch);
    }

    const Index progress_step = max(Index(1), batches_number / 200);
    if (show_progress) display_progress_bar(0, int(batches_number));

    for (Index iteration = 0; iteration < batches_number; ++iteration)
    {
        Batch* current_batch = next_batch;
        next_batch = nullptr;

        if (iteration + 1 < batches_number)
        {
            {
                PROFILE_SCOPE_HOST("step:wait_fill");
                next_batch = wait_for_filled_batch(*session, iteration + 1);
            }
            {
                PROFILE_SCOPE_HOST("step:prefetch_h2d_issue");
                prefetch_batch(*next_batch);
            }
        }

        {
            PROFILE_SCOPE("step:fwd_total");
            neural_network->forward_propagate(current_batch->get_inputs(), forward_propagation, true);
        }
        sync_cuda_for_debug(on_gpu);

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
        sync_cuda_for_debug(on_gpu);

        if (!use_device_metrics)
        {
            stats.error += back_propagation.error;
            if (is_classification) stats.accuracy += back_propagation.accuracy;
        }

        {
            PROFILE_SCOPE("step:optim_total");
            update(back_propagation);
        }
        sync_cuda_for_debug(on_gpu);

        {
            PROFILE_SCOPE("step:sync_device");
            sync_device(on_gpu);
        }

        empty_queue.push(current_batch);

        if (show_progress
            && ((iteration + 1) % progress_step == 0 || iteration + 1 == batches_number))
            display_progress_bar(int(iteration + 1), int(batches_number));
    }
    if (show_progress) cout << "\n";

    session->rethrow_if_error();

#ifdef OPENNN_HAS_CUDA
    if (use_device_metrics)
    {
        stats = device_metrics.read(batches_number, is_classification);
        back_propagation.error = stats.error;
        back_propagation.accuracy = stats.accuracy;
        set_epoch_loss();
    }
    else
#endif
    {
        stats.error /= float(batches_number);
        if (is_classification) stats.accuracy /= float(batches_number);
        set_epoch_loss();
    }

    if (profile_this)
    {
        const auto epoch_t1 = chrono::steady_clock::now();
        const double epoch_ms = chrono::duration<double, milli>(epoch_t1 - epoch_t0).count();

        if (const long w_calls = worker_profile.fills.load(); w_calls > 0)
        {
            auto& fill_entry = ::opennn::global_stats().entries["worker:fill"];
            fill_entry.total_ms = double(worker_profile.fill_us.load()) / 1000.0;
            fill_entry.calls    = w_calls;

            auto& wait_entry = ::opennn::global_stats().entries["worker:queue_wait"];
            wait_entry.total_ms = double(worker_profile.pop_us.load()) / 1000.0;
            wait_entry.calls    = w_calls;
        }

        ::opennn::global_stats().print(cout, "Epoch breakdown (training)", epoch_ms);
        cout << "  Wall-clock epoch time: " << fixed << setprecision(2) << epoch_ms << " ms"
                  << " | num_workers=" << num_workers << "\n\n";
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

    has_recurrent_layers_ = neural_network->has(LayerType::Recurrent);
    const bool on_gpu = neural_network->is_gpu();

#ifdef OPENNN_HAS_CUDA
    const bool use_device_metrics = on_gpu && loss->supports_device_epoch_metrics();
    DeviceEpochMetrics device_metrics;
    if (use_device_metrics) device_metrics.reset();
#else
    const bool use_device_metrics = false;
#endif

    if (!on_gpu)
    {
        Batch* batch = empty_queue.pop();

        for (Index iteration = 0; iteration < batches_number; ++iteration)
        {
            batch->fill(batches[iteration],
                        input_feature_indices,
                        decoder_feature_indices,
                        target_feature_indices,
                        /*is_training=*/false,
                        /*parallelize_samples=*/true);

            neural_network->forward_propagate(batch->get_inputs(), forward_propagation, false);

            const Loss::EvaluationResult eval = loss->calculate_error(*batch, forward_propagation);

            stats.error += eval.error;
            if (is_classification) stats.accuracy += eval.accuracy;
        }

        empty_queue.push(batch);

        stats.error /= float(batches_number);
        if (is_classification) stats.accuracy /= float(batches_number);

        return stats;
    }

#ifdef OPENNN_HAS_CUDA
    if (resident_validation_.active)
    {
        ResidentEpochState& state = resident_validation_;
        const Index batch_size = Index(batches[0].size());
        upload_resident_epoch_indices(state, batches, batch_size);

        Batch* batch = empty_queue.pop();
        for (Index iteration = 0; iteration < batches_number; ++iteration)
        {
            batch->gather_device_async(batch_size,
                                       state.row_index_buffer.as<Index>() + iteration * batch_size,
                                       state.data->input.as<float>(),  state.data->input_features,
                                       state.data->target.as<float>(), state.data->target_features);

            neural_network->forward_propagate(batch->get_inputs(), forward_propagation, false);

            Loss::EvaluationResult eval;
            if (use_device_metrics)
            {
                if (!loss->calculate_error_device_metrics(*batch, forward_propagation,
                                                          device_metrics.error_sum(),
                                                          is_classification ? device_metrics.accuracy_sum() : nullptr))
                    throw runtime_error("Device epoch metrics unexpectedly unsupported for this loss.");
            }
            else
            {
                eval = loss->calculate_error(*batch, forward_propagation);
                stats.error += eval.error;
                if (is_classification) stats.accuracy += eval.accuracy;
            }
            sync_device(on_gpu);
        }
        empty_queue.push(batch);

        if (use_device_metrics)
            stats = device_metrics.read(batches_number, is_classification);
        else
        {
            stats.error /= float(batches_number);
            if (is_classification) stats.accuracy /= float(batches_number);
        }
        return stats;
    }
#endif

    auto session = start_batch_workers(empty_queue,
                                       batches,
                                       input_feature_indices,
                                       decoder_feature_indices,
                                       target_feature_indices,
                                       /*is_training=*/false);

    Batch* next_batch = wait_for_filled_batch(*session, 0);
    prefetch_batch(*next_batch);

    for (Index iteration = 0; iteration < batches_number; ++iteration)
    {
        Batch* current_batch = next_batch;
        next_batch = nullptr;

        if (iteration + 1 < batches_number)
        {
            next_batch = wait_for_filled_batch(*session, iteration + 1);
            prefetch_batch(*next_batch);
        }

        neural_network->forward_propagate(current_batch->get_inputs(), forward_propagation, false);
        sync_cuda_for_debug(on_gpu);
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
        sync_cuda_for_debug(on_gpu);

        if (!use_device_metrics)
        {
            stats.error += eval.error;
            if (is_classification) stats.accuracy += eval.accuracy;
        }

        sync_device(on_gpu);

        empty_queue.push(current_batch);
    }

    session->rethrow_if_error();

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
