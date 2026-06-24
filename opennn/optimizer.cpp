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
#include "device_backend.h"
#include "neural_network.h"
#include "profiler.h"
#include "string_utilities.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <mutex>
#include <stop_token>
#include <thread>

#if defined(__linux__) || defined(__unix__)
#include <unistd.h>
#endif
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

static void clip_gradient_norm_device(Buffer& gradient, Index gradient_size, float max_norm)
{
    thread_local Buffer squared_norm_device(Device::CUDA);
    if (!squared_norm_device.data)
        squared_norm_device.grow_to(Index(sizeof(float)));
    float* const squared_norm_ptr = squared_norm_device.as<float>();

    cublasHandle_t handle = Backend::get_cublas_handle();
    {
        device::CublasPointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_DEVICE);
        CHECK_CUBLAS(cublasSdot(handle,
                                to_int(gradient_size),
                                gradient.as<float>(), 1,
                                gradient.as<float>(), 1,
                                squared_norm_ptr));
    }

    clip_gradient_norm_cuda(gradient_size, gradient.as<float>(), squared_norm_ptr, max_norm, GRADIENT_NORM_EPS);
}

#else

static void clip_gradient_norm_device(Buffer&, Index, float)
{
    throw runtime_error("clip_gradient_norm_device requires CUDA support.");
}

#endif

namespace
{

void sync_cuda_for_debug(bool on_gpu)
{
    static const bool enabled = env_flag_enabled("OPENNN_CUDA_DEBUG_SYNC");
    if (on_gpu && enabled)
        device::synchronize(Backend::get_compute_stream());
}

Loss::EvaluationResult average_epoch_metrics(Loss::EvaluationResult sums,
                                             Index batches_number,
                                             bool include_accuracy)
{
    if (batches_number <= 0) return sums;

    sums.error /= float(batches_number);
    if (include_accuracy) sums.accuracy /= float(batches_number);
    return sums;
}

struct DeviceEpochMetricSums
{
    static Buffer& values()
    {
        thread_local Buffer buffer{Device::CUDA};
        return buffer;
    }

    void reset()
    {
        if (!device::is_cuda_build()) return;

        Buffer& metric_values = values();
        metric_values.grow_to(2 * Index(sizeof(float)));
        device::set_zero_async(metric_values.data, 2 * Index(sizeof(float)),
                               Backend::get_compute_stream());
    }

    float* error_sum() { return values().as<float>(); }
    float* accuracy_sum() { return values().as<float>() + 1; }

    Loss::EvaluationResult read()
    {
        Loss::EvaluationResult sums;
        if (!device::is_cuda_build()) return sums;

        float host[2] = {0.0f, 0.0f};
        cudaStream_t stream = Backend::get_compute_stream();
        device::copy_async(host, values().data, Index(sizeof(host)),
                           device::CopyKind::DeviceToHost,
                           stream);
        device::synchronize(stream);

        sums.error = host[0];
        sums.accuracy = host[1];
        return sums;
    }
};

}

struct Optimizer::WorkerProfileCounters
{
    atomic<int64_t> pop_us{0};
    atomic<int64_t> fill_us{0};
    atomic<long> fills{0};

    void record(chrono::steady_clock::time_point pop_begin,
                chrono::steady_clock::time_point fill_begin,
                chrono::steady_clock::time_point fill_end)
    {
        pop_us.fetch_add(
            chrono::duration_cast<chrono::microseconds>(fill_begin - pop_begin).count(),
            memory_order_relaxed);
        fill_us.fetch_add(
            chrono::duration_cast<chrono::microseconds>(fill_end - fill_begin).count(),
            memory_order_relaxed);
        fills.fetch_add(1, memory_order_relaxed);
    }

    void publish() const
    {
        const long calls = fills.load();
        if (calls <= 0) return;

        auto& fill_entry = ::opennn::global_stats().entries["worker:fill"];
        fill_entry.total_ms = double(fill_us.load()) / 1000.0;
        fill_entry.calls = calls;

        auto& wait_entry = ::opennn::global_stats().entries["worker:queue_wait"];
        wait_entry.total_ms = double(pop_us.load()) / 1000.0;
        wait_entry.calls = calls;
    }
};

Optimizer::Optimizer(Loss* new_loss)
{
    set(new_loss);
}

Optimizer::~Optimizer() = default;

void Optimizer::reset_graph_capture()
{
    for (device::GraphExecHandle& exec : training_graph_execs)
        exec.reset();
    graph_update = nullptr;
}

bool Optimizer::cuda_graph_requested() const
{
    return use_cuda_graph.value_or(env_flag_enabled("OPENNN_CUDA_GRAPH"));
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

    throw_if(!file.is_open(), format("Cannot open file: {}", file_name.string()));

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
    return float(difftime(time(nullptr), beginning_time));
}

void Optimizer::warn_dropped_samples(Index batch_size,
                                      Index samples_number,
                                      const char* context) const
{
    if (!display
        || batch_size <= 0
        || samples_number <= 0
        || batch_size >= samples_number
        || samples_number % batch_size == 0)
        return;

    const Index lost = samples_number % batch_size;
    cout << format("Warning: {} batch_size {} does not divide {} samples. "
                   "{} sample(s) ({:.2f} % of total) dropped per epoch.\n",
                   context, batch_size, samples_number,
                   lost, 100.0 * double(lost) / double(samples_number));
}

void Optimizer::setup_batch_pools(BatchPools& pools,
                                  Dataset& dataset,
                                  NeuralNetwork& neural_network,
                                  Index training_batch_size,
                                  Index validation_batch_size,
                                  bool has_validation)
{
    const int pool_size = get_batch_pool_size(neural_network);
    const auto& config = neural_network.get_config();

    auto fill_pool = [&](ThreadSafeQueue<Batch*>& queue,
                         vector<unique_ptr<Batch>>& pool,
                         Index batch_size,
                         bool prefetch_only)
    {
        for (int i = 0; i < pool_size; ++i)
        {
            pool.push_back(make_unique<Batch>(batch_size, &dataset, config, prefetch_only));
            queue.push(pool.back().get());
        }
    };

    // On GPU the prefetch pool only stages data into the fixed compute batch, so
    // pool slots need no device buffers (saves pool_size device batch copies). This
    // is unsafe only if validation reuses the training pool to compute directly on
    // it (no fixed batch in evaluate_epoch) -- then the pool keeps its buffers.
    const bool validation_reuses_training_pool =
        has_validation && validation_batch_size == training_batch_size;
    const bool training_prefetch_only = neural_network.is_gpu()
                                     && device::is_cuda_build()
                                     && !validation_reuses_training_pool;

    fill_pool(pools.training_empty_queue,
              pools.training_pool,
              training_batch_size,
              training_prefetch_only);

    graph_slots = {};

    if (neural_network.is_gpu() && device::is_cuda_build())
    {
        pools.fixed_training_batch = make_unique<Batch>(training_batch_size, &dataset, config);
        graph_slots[0] = pools.fixed_training_batch.get();

        if (cuda_graph_requested())
            for (int i = 1; i < graph_slots_count; ++i)
            {
                pools.graph_slot_pool.push_back(
                    make_unique<Batch>(training_batch_size, &dataset, config));
                graph_slots[size_t(i)] = pools.graph_slot_pool.back().get();
            }
    }

    pools.validation_uses_training_pool = validation_reuses_training_pool;

    // Validation computes directly on its pool batch (evaluate_epoch sets no fixed
    // device batch), so validation pool slots must keep their device buffers.
    if (has_validation && !pools.validation_uses_training_pool)
        fill_pool(pools.validation_empty_queue,
                  pools.validation_pool,
                  validation_batch_size,
                  /*prefetch_only=*/false);
}

unique_ptr<BatchPrefetchSession> Optimizer::start_batch_prefetch(
    ThreadSafeQueue<Batch*>& empty_queue,
    const vector<vector<Index>>& batches,
    const vector<Index>& input_feature_indices,
    const vector<Index>& decoder_feature_indices,
    const vector<Index>& target_feature_indices,
    bool is_training,
    WorkerProfileCounters* profile_counters)
{
    const Index batches_number = Index(batches.size());

    auto session = make_unique<BatchPrefetchSession>(empty_queue, batches_number);
    BatchPrefetchSession* const session_ptr = session.get();
    const auto* const batches_ptr = &batches;
    const auto* const input_indices = &input_feature_indices;
    const auto* const decoder_indices = &decoder_feature_indices;
    const auto* const target_indices = &target_feature_indices;

    auto worker_body = [batches_ptr,
                        input_indices,
                        decoder_indices,
                        target_indices,
                        session_ptr,
                        batches_number,
                        is_training,
                        profile_counters](stop_token stop)
    {
        try
        {
            for (;;)
            {
                const auto t_pop0 = chrono::steady_clock::now();
                Batch* batch = session_ptr->empty_queue.pop();
                const auto t_fill0 = chrono::steady_clock::now();

                if (!batch || stop.stop_requested())
                {
                    if (batch) session_ptr->empty_queue.push(batch);
                    return;
                }

                const Index it = session_ptr->next_iteration.fetch_add(1);
                if (it >= batches_number)
                {
                    session_ptr->empty_queue.push(batch);
                    return;
                }

                batch->wait_h2d_complete();
                batch->fill((*batches_ptr)[size_t(it)],
                            *input_indices,
                            *decoder_indices,
                            *target_indices,
                            is_training);

                const auto t_fill1 = chrono::steady_clock::now();
                session_ptr->ready_batches[size_t(it)].store(batch, memory_order_release);

                if (profile_counters)
                    profile_counters->record(t_pop0, t_fill0, t_fill1);
            }
        }
        catch (...)
        {
            session_ptr->capture_current_exception();
        }
    };

    NeuralNetwork* neural_network = loss->get_neural_network();
    const int batch_workers_number = get_batch_workers_number(*neural_network);

    session->threads.reserve(size_t(batch_workers_number));
    for (int i = 0; i < batch_workers_number; ++i)
        session->threads.emplace_back(worker_body);

    return session;
}

int Optimizer::get_batch_workers_number(const NeuralNetwork& neural_network) const
{
    return neural_network.is_gpu() && neural_network.has(LayerType::Recurrent)
        ? 1
        : workers_number;
}

int Optimizer::get_batch_pool_size(const NeuralNetwork& neural_network) const
{
    // OPENNN_BATCH_POOL overrides the prefetch-pool depth. Each pooled Batch holds
    // a full input+target copy on the GPU, so lowering this trades a little
    // prefetch overlap for a larger max batch. Default keeps 3.
    if (const char* pool = std::getenv("OPENNN_BATCH_POOL"))
        return max(1, atoi(pool));
    return neural_network.is_gpu()
        ? max(get_batch_workers_number(neural_network) + 1, 3)
        : 1;
}

Index Optimizer::get_maximum_batch_size() const
{
    throw_if(!loss, "Optimizer::get_maximum_batch_size: loss is not set.");

    const Dataset* dataset = loss->get_dataset();
    const NeuralNetwork* neural_network = loss->get_neural_network();

    throw_if(!dataset, "Optimizer::get_maximum_batch_size: dataset is not set.");
    throw_if(!neural_network, "Optimizer::get_maximum_batch_size: neural network is not set.");

    const Index training_samples_number = dataset->get_samples_number("Training");
    if (training_samples_number <= 0) return 0;
    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const bool on_gpu = neural_network->is_gpu();


    Index available_bytes = 0;
    if (on_gpu)
    {
        available_bytes = Index(device::available_memory());
    }
    else
    {
#if defined(__linux__) || defined(__unix__)
        const long pages = sysconf(_SC_AVPHYS_PAGES);
        const long page_size = sysconf(_SC_PAGE_SIZE);
        throw_if(pages <= 0 || page_size <= 0,
                 "Optimizer::get_maximum_batch_size: sysconf failed to query available RAM.");
        available_bytes = Index(pages) * Index(page_size);
#elif defined(_WIN32)
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        throw_if(!GlobalMemoryStatusEx(&status),
                 "Optimizer::get_maximum_batch_size: GlobalMemoryStatusEx failed.");
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
    const bool bf16_input = bf16_train && dataset->supports_bf16_inputs();

    Index fixed_bytes = 0;

    fixed_bytes += parameters_aligned_size * Index(sizeof(float));
    if (bf16_train) fixed_bytes += parameters_aligned_size * Index(sizeof(bfloat16));
    fixed_bytes += neural_network->get_states_size() * Index(sizeof(float));
    fixed_bytes += parameters_aligned_size * Index(sizeof(float));
    fixed_bytes += 2 * slot_aligned_size * Index(sizeof(float));

    throw_if(fixed_bytes >= budget,
             format("Optimizer::get_maximum_batch_size: fixed memory ({} MiB) exceeds 80% budget ({} MiB).",
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
        // On GPU the prefetch pool is host-only and compute runs on a single fixed
        // device batch, so only one device batch copy lives in VRAM; on CPU each of
        // the batch_pool_size slots holds a device (host) copy.
        const Index device_batch_copies = on_gpu ? Index(1) : Index(batch_pool_size);
        return device_batch_copies * single_batch;
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

    throw_if(bytes_for_batch(1) > dynamic_budget,
             format("Optimizer::get_maximum_batch_size: not enough memory for batch_size=1. "
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


    vector<Descriptives> input_variable_descriptives;
    vector<string> input_variable_scalers;

    if (auto* scaling_layer = dynamic_cast<Scaling*>(neural_network->get_first(LayerType::Scaling)))
    {
        switch (scaling_layer->get_input_shape().rank)
        {
            case 1:
            case 2:
            {
                auto* tabular_dataset = dynamic_cast<TabularDataset*>(dataset);
                throw_if(!tabular_dataset, "Expected TabularDataset.");
                input_variable_scalers = tabular_dataset->get_feature_scalers("Input");
                input_variable_descriptives = tabular_dataset->scale_features("Input");
                scaling_layer->set_descriptives(input_variable_descriptives);
                scaling_layer->set_scalers(input_variable_scalers);
                break;
            }

            case 3:
            {
#ifdef OPENNN_NO_VISION
                throw runtime_error("Rank-3 (image) scaling requires the vision build (OPENNN_NO_VISION is set).");
#else
                auto* image_dataset = dynamic_cast<ImageDataset*>(dataset);
                throw_if(!image_dataset, "Expected ImageDataset.");

                image_dataset->set_input_scaling(scaling_layer->get_descriptives(),
                                                 scaling_layer->get_scalers(),
                                                 scaling_layer->get_min_range(),
                                                 scaling_layer->get_max_range());
#endif
                break;
            }

            default:
                throw runtime_error(format("Unexpected Scaling input rank: {}",
                                           scaling_layer->get_input_shape().rank));
        }
    }

    if (!neural_network->has(LayerType::Unscaling))
        return;


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
            for (Index i = 0; i < n_targets; ++i)
                for (Index j = 0; j < future_steps; ++j)
                {
                    expanded_desc.push_back(unscaling_layer_descriptives[i]);
                    expanded_scalers.push_back(unscaling_layer_scalers[i]);
                }
            unscaling_layer_descriptives = move(expanded_desc);
            unscaling_layer_scalers      = move(expanded_scalers);
        }
    }

    throw_if(ssize(unscaling_layer_descriptives) != unscaling_outputs,
             "Unscaling setup error: Mismatch between number of target variables and unscaling layer neurons.");

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
        vector<Descriptives> descriptives;
        descriptives.reserve(minimums.size());
        for (Index i = 0; i < minimums.size(); ++i)
            descriptives.emplace_back(minimums[i], maximums[i], means[i], std_devs[i]);
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

void Optimizer::warmup_device_training(
    ForwardPropagation& training_forward_propagation,
    BackPropagation& training_back_propagation,
    ThreadSafeQueue<Batch*>& training_empty_queue,
    const vector<vector<Index>>& training_batches,
    const vector<Index>& input_feature_indices,
    const vector<Index>& decoder_feature_indices,
    const vector<Index>& target_feature_indices,
    const function<void(BackPropagation&)>& update,
    ForwardPropagation* validation_forward_propagation,
    ThreadSafeQueue<Batch*>* validation_empty_queue,
    const vector<vector<Index>>* validation_batches,
    Batch* fixed_training_batch)
{
    NeuralNetwork* neural_network = loss ? loss->get_neural_network() : nullptr;
    if (!device::is_cuda_build()
        || !neural_network
        || !neural_network->is_gpu()
        || training_batches.empty())
        return;

    cudaStream_t stream = Backend::get_compute_stream();

    const Index parameters_bytes = neural_network->get_parameters_size() * Index(sizeof(float));
    const Index states_bytes = neural_network->get_states_buffer_size() * Index(sizeof(float));

    Buffer parameters_snapshot{Device::CUDA};
    Buffer states_snapshot{Device::CUDA};

    if (parameters_bytes > 0)
    {
        parameters_snapshot.resize_bytes(parameters_bytes, Device::CUDA);
        device::copy_async(parameters_snapshot.data,
                           neural_network->get_parameters_data(),
                           parameters_bytes,
                           device::CopyKind::DeviceToDevice,
                           stream);
    }

    if (states_bytes > 0)
    {
        states_snapshot.resize_bytes(states_bytes, Device::CUDA);
        device::copy_async(states_snapshot.data,
                           neural_network->get_states_data(),
                           states_bytes,
                           device::CopyKind::DeviceToDevice,
                           stream);
    }

    auto restore_model_state = [&]()
    {
        if (parameters_bytes > 0)
        {
            device::copy_async(neural_network->get_parameters_data(),
                               parameters_snapshot.data,
                               parameters_bytes,
                               device::CopyKind::DeviceToDevice,
                               stream);
            neural_network->cast_parameters_to_bf16();
        }

        if (states_bytes > 0)
        {
            device::copy_async(neural_network->get_states_data(),
                               states_snapshot.data,
                               states_bytes,
                               device::CopyKind::DeviceToDevice,
                               stream);
        }

        device::synchronize(stream);
    };

    const bool has_validation_warmup = validation_forward_propagation
                                    && validation_empty_queue
                                    && validation_batches
                                    && !validation_batches->empty();

    try
    {
        train_epoch(training_forward_propagation,
                    training_back_propagation,
                    training_empty_queue,
                    vector<vector<Index>>{training_batches.front()},
                    input_feature_indices,
                    decoder_feature_indices,
                    target_feature_indices,
                    update,
                    fixed_training_batch);

        if (has_validation_warmup)
            evaluate_epoch(*validation_forward_propagation,
                           *validation_empty_queue,
                           vector<vector<Index>>{validation_batches->front()},
                           input_feature_indices,
                           decoder_feature_indices,
                           target_feature_indices);

        restore_model_state();
    }
    catch (...)
    {
        restore_model_state();
        throw;
    }
}

void Optimizer::display_epoch_results(const Index epoch,
                                      const float training_error,
                                      const float training_accuracy,
                                      const float validation_error,
                                      const float validation_accuracy,
                                      const bool has_validation,
                                      const bool is_token_cross_entropy,
                                      const float elapsed_time) const
{
    if (!should_display(epoch)) return;

    cout << "Training error: " << training_error << "\n";
    if (is_token_cross_entropy) cout << "Training perplexity: " << exp(training_error) << "\n";
    if (is_token_cross_entropy) cout << "Training accuracy: " << training_accuracy << "\n";
    if (has_validation) cout << "Validation error: " << validation_error << "\n";
    if (has_validation && is_token_cross_entropy) cout << "Validation perplexity: " << exp(validation_error) << "\n";
    if (has_validation && is_token_cross_entropy) cout << "Validation accuracy: " << validation_accuracy << "\n";
    cout << "Elapsed time: " << get_time(elapsed_time) << "\n";
}

bool Optimizer::check_stopping_condition(TrainingResult& results,
                                          const Index epoch,
                                          const float elapsed_time,
                                          const float training_error,
                                          const Index validation_failures,
                                          const float training_loss,
                                          const bool has_validation) const
{
    if (!results.stopping_condition)
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
    }

    results.loss = training_loss;
    results.validation_failures = validation_failures;
    results.resize_training_error_history(epoch + 1);
    results.resize_validation_error_history(has_validation ? epoch + 1 : 0);
    results.elapsed_time = get_time(elapsed_time);

    return true;
}

void Optimizer::reset_best_parameters()
{
    best_validation_error = numeric_limits<float>::max();
    best_epoch = -1;
    best_parameters.clear();
    best_states.clear();
}

void Optimizer::update_best_parameters(NeuralNetwork* neural_network, float validation_error,
                                       Index epoch, Index& validation_failures)
{
    if (validation_error >= best_validation_error)
    {
        ++validation_failures;
        return;
    }

    best_validation_error = validation_error;
    best_epoch = epoch;
    validation_failures = 0;

    const tuple<vector<float>&, const float*, Index> snapshots[] = {
        {best_parameters, neural_network->get_parameters_data(), neural_network->get_parameters_size()},
        {best_states,     neural_network->get_states_data(),     neural_network->get_states_buffer_size()}
    };

    for (const auto& [destination, source, size] : snapshots)
    {
        if (size == 0) continue;

        if (Index(destination.size()) != size)
            destination.resize(size);

        const size_t bytes = size_t(size) * sizeof(float);
        if (neural_network->is_gpu() && device::is_cuda_build())
        {
            cudaStream_t stream = Backend::get_compute_stream();
            device::copy_async(destination.data(), source, Index(bytes),
                               device::CopyKind::DeviceToHost, stream);
            device::synchronize(stream);
        }
        else
            memcpy(destination.data(), source, bytes);
    }
}

void Optimizer::restore_best_parameters(NeuralNetwork* neural_network, TrainingResult& results)
{
    if (!results.stopping_condition
        || *results.stopping_condition != StoppingCondition::MaximumValidationErrorIncreases
        || best_parameters.empty()
        || Index(best_parameters.size()) != neural_network->get_parameters_size())
        return;

    if (display)
        cout << "Restoring best parameters and states from epoch " << best_epoch
             << " (validation error " << best_validation_error << ")\n";

    neural_network->set_parameters(Map<const VectorR>(best_parameters.data(), Index(best_parameters.size())));

    if (!best_states.empty())
        neural_network->set_states(Map<const VectorR>(best_states.data(), Index(best_states.size())));

    results.restored_best_parameters = true;
    results.restored_epoch = best_epoch;
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
    set_loss_goal(read_json_float(root_element, "LossGoal"));
    set_maximum_validation_failures(read_json_index(root_element,
        root_element->has("MaximumValidationFailures") ? "MaximumValidationFailures" : "MaximumSelectionFailures"));
    set_maximum_epochs(read_json_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_json_float(root_element, "MaximumTime"));
}

void Optimizer::setup_device_training()
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    if (!neural_network->is_gpu()) return;

    neural_network->copy_parameters_device();
    neural_network->copy_states_device();

    if (env_flag_enabled("OPENNN_GPU_RESIDENT_DATA"))
        loss->get_dataset()->enable_device_residency();
}

void Optimizer::teardown_device_training()
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    if (!neural_network->is_gpu()) return;

    device::synchronize(Backend::get_compute_stream());

    neural_network->copy_parameters_host();
    neural_network->copy_states_host();

    if (loss->get_dataset()->is_device_resident())
        loss->get_dataset()->disable_device_residency();

    // The captured graphs, the update closure and the slot pointers reference
    // train()-local resources; they must not outlive them.
    reset_graph_capture();
    graph_slots = {};
}

void Optimizer::prefetch_batch(Batch& batch)
{
    if (!batch.uses_cuda() || !batch.needs_device_copy) return;

    batch.copy_device_async(Backend::get_transfer_stream());
}

void Optimizer::sync_device(bool on_gpu)
{
    static const bool sync_each_batch = env_flag_enabled("OPENNN_CUDA_SYNC_EACH_BATCH");
    if (on_gpu && (has_recurrent_layers_ || sync_each_batch))
        device::synchronize(Backend::get_compute_stream());
}

void Optimizer::clip_gradient_norm(Buffer& gradient, float max_norm)
{
    if (max_norm <= 0.0f) return;

    const Index gradient_size = gradient.size_in_floats();
    if (gradient_size <= 0) return;

    if (gradient.device_type == Device::CUDA)
    {
        clip_gradient_norm_device(gradient, gradient_size, max_norm);
        return;
    }

    VectorMap gradient_view(gradient.as<float>(), gradient_size);
    const float gradient_norm = gradient_view.norm();
    if (gradient_norm > max_norm)
        gradient_view *= max_norm / (gradient_norm + GRADIENT_NORM_EPS);
}

bool Optimizer::graph_epoch_enabled(bool use_device_metrics, Batch* fixed_device_batch) const
{
    return cuda_graph_requested()
        && fixed_device_batch
        && fixed_device_batch->uses_cuda()
        && use_device_metrics
        && bool(graph_update);
}

Loss::EvaluationResult Optimizer::run_graph_epoch(
    ForwardPropagation& forward_propagation,
    BackPropagation& back_propagation,
    ThreadSafeQueue<Batch*>& empty_queue,
    const vector<vector<Index>>& batches,
    const vector<Index>& input_feature_indices,
    const vector<Index>& decoder_feature_indices,
    const vector<Index>& target_feature_indices,
    Batch* fixed_device_batch)
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index batches_number = Index(batches.size());
    const bool tracks_accuracy = loss->get_error() == Loss::Error::CrossEntropy3d;

    DeviceEpochMetricSums device_metrics;
    device_metrics.reset();

    cudaStream_t compute = Backend::get_compute_stream();
    cudaStream_t transfer = Backend::get_transfer_stream();

    // Slot ring: [0] is the shared fixed device batch; the rest come from the
    // dedicated graph slot pool. The staged path needs the full ring (two
    // groups of graph_group_size); the upload path uses at most two slots.
    array<Batch*, graph_slots_count> slots = {};
    slots[0] = fixed_device_batch;
    int usable_slots = 1;
    if (graph_slots[0] == fixed_device_batch)
        while (usable_slots < graph_slots_count && graph_slots[size_t(usable_slots)])
        {
            slots[size_t(usable_slots)] = graph_slots[size_t(usable_slots)];
            ++usable_slots;
        }

    const bool profile_this = env_flag_enabled("OPENNN_PROFILE");
    if (profile_this)
    {
        ::opennn::enabled() = true;
        ::opennn::global_stats().clear();
    }
    const auto epoch_t0 = chrono::steady_clock::now();
    WorkerProfileCounters worker_profile;

    auto session = start_batch_prefetch(empty_queue, batches,
                                        input_feature_indices,
                                        decoder_feature_indices,
                                        target_feature_indices,
                                        /*is_training=*/true,
                                        profile_this ? &worker_profile : nullptr);

    // Host-loaded FP32 data takes the staged path: the H2D copy is captured
    // INSIDE each slot's graph, reading the slot's own pinned host buffer at a
    // fixed address. Per iteration the main thread only does a small host
    // memcpy + one graph launch, instead of several CUDA API calls whose issue
    // latency starves the GPU. Device-resident (gather) and BF16 batches keep
    // the upload-outside-the-graph path.
    const bool staged_h2d = !loss->get_dataset()->is_device_resident()
                         && !fixed_device_batch->input_is_bf16;

    const auto stage_into_slot = [](const Batch& source, Batch& slot)
    {
        const Index samples = source.current_sample_count;
        slot.current_sample_count = samples;
        slot.needs_device_copy = false;

        const auto copy_section = [&](const BatchSlot& from, BatchSlot& to)
        {
            if (!from.host || !to.host || from.features_number <= 0) return;
            memcpy(to.host, from.host,
                   size_t(samples) * size_t(from.features_number) * sizeof(float));
        };

        copy_section(source.input,   slot.input);
        copy_section(source.decoder, slot.decoder);
        copy_section(source.target,  slot.target);
    };

    const auto issue_slot_h2d = [](Batch& slot, cudaStream_t stream)
    {
        const auto copy_section = [&](BatchSlot& section)
        {
            if (!section.host || !section.buffer.data || section.features_number <= 0) return;
            device::copy_async(section.buffer.data, section.host,
                               slot.current_sample_count * section.features_number * Index(sizeof(float)),
                               device::CopyKind::HostToDevice, stream);
        };

        copy_section(slot.input);
        copy_section(slot.decoder);
        copy_section(slot.target);
    };

    // Resident gather: copy the worker batch's per-batch row indices and gather
    // metadata into the slot so the slot gathers from itself; the captured
    // graph reads the slot's fixed index buffer on replay.
    const auto stage_gather_indices = [](const Batch& source, Batch& slot)
    {
        slot.current_sample_count = source.current_sample_count;
        slot.device_gather        = source.device_gather;
        slot.input_col_offset     = source.input_col_offset;
        slot.target_col_offset    = source.target_col_offset;
        slot.gather_row_indices   = source.gather_row_indices;
        slot.needs_device_copy    = false;
    };

    // The update closure is copied so the eager fallback keeps working for the
    // rest of the epoch after a capture failure clears the member.
    const auto run_compute_step = [&, update = graph_update](Batch& slot)
    {
        neural_network->forward_propagate(slot.get_inputs(),
                                          forward_propagation, true);
        if (!loss->back_propagate_device_metrics(slot,
                                                 forward_propagation, back_propagation,
                                                 device_metrics.error_sum(),
                                                 tracks_accuracy ? device_metrics.accuracy_sum() : nullptr))
            throw runtime_error("Device epoch metrics unexpectedly unsupported for this loss.");
        update(back_propagation);
    };

    // Capture-or-run: replay `exec` if it exists; otherwise, when graph_update
    // is set, run `warmup` once eagerly and then capture `record` into a fresh
    // graph; otherwise just run `warmup`. The profiler muting (op-level scopes
    // call device::synchronize, illegal inside a capture window), the
    // capture/instantiate lifecycle, and the failure recovery live here so the
    // three call sites below don't each repeat them. `record` defaults to
    // `warmup`; only the staged mega-graph captures a different body (fork/join
    // H2D on the transfer stream) than its eager warmup.
    const auto capture_or_run = [&](device::GraphExecHandle& exec,
                                    const auto& warmup, const auto& record)
    {
        if (exec)
        {
            PROFILE_SCOPE_HOST("step:graph_launch");
            device::launch_graph(exec, compute);
            return;
        }
        if (!graph_update) { warmup(); return; }

        warmup();
        const bool profiler_enabled = ::opennn::enabled();
        ::opennn::enabled() = false;
        try
        {
            device::synchronize(compute);
            device::StreamCapture capture(compute);
            record();
            const device::GraphHandle graph = capture.end();
            device::instantiate_or_update(exec, graph.get());
        }
        catch (const exception& capture_error)
        {
            reset_graph_capture();
            cout << "CUDA graph capture failed (" << capture_error.what()
                 << "); continuing without graphs.\n";
        }
        ::opennn::enabled() = profiler_enabled;
    };

    const bool resident_gather = loss->get_dataset()->is_device_resident()
                              && !fixed_device_batch->input_is_bf16;

    Batch* host_batch = nullptr;
    try
    {
        if (resident_gather
            && usable_slots == graph_slots_count
            && batches_number >= Index(graph_group_size)
            && (graph_update || training_graph_execs[0]))
        {
            // Resident mega-graph: the M device-side gathers run on the transfer
            // stream OUTSIDE the graph (their index H2D is a pageable copy, which
            // is illegal to capture), each rejoined onto compute by an event; the
            // graph then captures only the M compute steps. One launch per group
            // amortizes the per-step launch the single-step resident path paid.
            constexpr Index M = Index(graph_group_size);
            const Index groups = batches_number / M;

            for (Index group = 0; group < groups; ++group)
            {
                const size_t parity = size_t(group & 1);
                const size_t base = parity * size_t(M);
                Batch& event_slot = *slots[base + size_t(M) - 1];
                device::GraphExecHandle& exec = training_graph_execs[parity];

                {
                    PROFILE_SCOPE_HOST("step:group_sync");
                    // The prior same-parity group's graph read these slots; it
                    // must finish before the gathers overwrite them.
                    if (event_slot.h2d_done_recorded)
                        device::synchronize_event(event_slot.h2d_done_event);
                }

                for (Index m = 0; m < M; ++m)
                {
                    Batch& slot = *slots[base + size_t(m)];
                    {
                        PROFILE_SCOPE_HOST("step:wait_fill");
                        host_batch = session->wait(group * M + m);
                    }
                    PROFILE_SCOPE_HOST("step:gather_issue");
                    stage_gather_indices(*host_batch, slot);
                    empty_queue.push(host_batch);
                    host_batch = nullptr;

                    if (slot.h2d_done_recorded)
                        device::stream_wait_event(transfer, slot.h2d_done_event);
                    slot.upload_to_device_batch_async(slot, transfer);
                    slot.wait_h2d_on_compute_stream();
                }

                const auto run_group = [&] {
                    for (Index m = 0; m < M; ++m)
                        run_compute_step(*slots[base + size_t(m)]);
                };
                capture_or_run(exec, run_group, run_group);

                event_slot.record_h2d_done(compute);
            }

            for (Index iteration = groups * M; iteration < batches_number; ++iteration)
            {
                host_batch = session->wait(iteration);
                Batch& slot = *slots[0];
                device::synchronize(compute);
                stage_gather_indices(*host_batch, slot);
                empty_queue.push(host_batch);
                host_batch = nullptr;
                slot.upload_to_device_batch_async(slot, transfer);
                slot.wait_h2d_on_compute_stream();
                run_compute_step(slot);
            }
        }
        else if (staged_h2d
            && usable_slots == graph_slots_count
            && batches_number >= Index(graph_group_size))
        {
            // Mega-graph path: each captured graph covers graph_group_size
            // iterations (H2D from the slots' pinned staging + compute), and
            // two groups ping-pong over the ring. Per group the host pays one
            // event sync, M small memcpys and one launch — cheap enough on
            // WSL's expensive CUDA API to keep the GPU permanently fed.
            constexpr Index M = Index(graph_group_size);
            const Index groups = batches_number / M;

            for (Index group = 0; group < groups; ++group)
            {
                const size_t parity = size_t(group & 1);
                const size_t base = parity * size_t(M);
                Batch& event_slot = *slots[base + size_t(M) - 1];
                device::GraphExecHandle& exec = training_graph_execs[parity];

                {
                    PROFILE_SCOPE_HOST("step:group_sync");
                    // The previous same-parity group read these slots' staging;
                    // it must be complete before the host overwrites it.
                    if (event_slot.h2d_done_recorded)
                        device::synchronize_event(event_slot.h2d_done_event);
                }

                for (Index m = 0; m < M; ++m)
                {
                    {
                        PROFILE_SCOPE_HOST("step:wait_fill");
                        host_batch = session->wait(group * M + m);
                    }
                    {
                        PROFILE_SCOPE_HOST("step:stage_copy");
                        stage_into_slot(*host_batch, *slots[base + size_t(m)]);
                        // Consumed by the host memcpy: recycle immediately.
                        empty_queue.push(host_batch);
                        host_batch = nullptr;
                    }
                }

                // Warmup/eager body: H2D + compute per slot on the compute
                // stream, plus (one-time) creation of the fork/join events the
                // captured body needs. Event creation is here, not in `record`,
                // because cudaEventCreate is illegal inside a capture window.
                const auto warmup_group = [&] {
                    for (Index m = 0; m < M; ++m)
                    {
                        Batch& slot = *slots[base + size_t(m)];
                        issue_slot_h2d(slot, compute);
                        run_compute_step(slot);
                    }
                    if (!graph_fork_events[parity])
                        graph_fork_events[parity].create();
                    for (Index m = 0; m < M; ++m)
                        if (!graph_copy_done_events[base + size_t(m)])
                            graph_copy_done_events[base + size_t(m)].create();
                };
                // Captured body: fork the H2D chain onto the transfer stream
                // inside the graph so the copy engine overlaps the compute steps.
                const auto record_group = [&] {
                    device::record_event(graph_fork_events[parity], compute);
                    device::stream_wait_event(transfer, graph_fork_events[parity]);
                    for (Index m = 0; m < M; ++m)
                    {
                        issue_slot_h2d(*slots[base + size_t(m)], transfer);
                        device::record_event(graph_copy_done_events[base + size_t(m)], transfer);
                    }
                    for (Index m = 0; m < M; ++m)
                    {
                        device::stream_wait_event(compute, graph_copy_done_events[base + size_t(m)]);
                        run_compute_step(*slots[base + size_t(m)]);
                    }
                };
                capture_or_run(exec, warmup_group, record_group);

                event_slot.record_h2d_done(compute);
            }

            // Tail (< M iterations): eager on slot 0 behind a full drain.
            for (Index iteration = groups * M; iteration < batches_number; ++iteration)
            {
                host_batch = session->wait(iteration);
                Batch& slot = *slots[0];
                device::synchronize(compute);
                stage_into_slot(*host_batch, slot);
                empty_queue.push(host_batch);
                host_batch = nullptr;
                issue_slot_h2d(slot, compute);
                run_compute_step(slot);
            }
        }
        else
        {
            // Per-iteration path: device-resident gather / BF16 batches keep
            // their per-slot graphs; tiny staged epochs (e.g. the warmup's
            // single batch) run eagerly so the mega execs are never mixed with
            // per-slot ones.
            const int slots_count = (usable_slots >= 2 && slots[1]) ? 2 : 1;

            for (Index iteration = 0; iteration < batches_number; ++iteration)
            {
                const size_t slot_index = size_t(iteration % slots_count);
                Batch& slot = *slots[slot_index];
                device::GraphExecHandle& exec = training_graph_execs[slot_index];

                {
                    PROFILE_SCOPE_HOST("step:wait_fill");
                    host_batch = session->wait(iteration);
                }

                if (staged_h2d)
                {
                    PROFILE_SCOPE_HOST("step:stage_copy");
                    // Whatever last read this slot must finish before the host
                    // overwrites its staging.
                    if (slot.h2d_done_recorded)
                        device::synchronize_event(slot.h2d_done_event);

                    stage_into_slot(*host_batch, slot);
                    empty_queue.push(host_batch);
                    host_batch = nullptr;
                }
                else
                {
                    PROFILE_SCOPE_HOST("step:h2d_issue");
                    // The graph that last read this slot must finish before the
                    // upload overwrites it (the slot's event is recorded on
                    // compute below).
                    if (slot.h2d_done_recorded)
                        device::stream_wait_event(transfer, slot.h2d_done_event);

                    host_batch->upload_to_device_batch_async(slot, transfer);
                    host_batch->wait_h2d_on_compute_stream();
                }

                // Staged H2D never uses per-slot graphs here (the mega/hybrid
                // paths above own that); run it eagerly. Otherwise capture or
                // replay the single compute step.
                if (staged_h2d)
                {
                    issue_slot_h2d(slot, compute);
                    run_compute_step(slot);
                }
                else
                {
                    const auto run_slot = [&] { run_compute_step(slot); };
                    capture_or_run(exec, run_slot, run_slot);
                }

                slot.record_h2d_done(compute);

                if (host_batch)
                {
                    empty_queue.push(host_batch);
                    host_batch = nullptr;
                }
            }
        }
        device::synchronize(compute);
    }
    catch (...)
    {
        if (host_batch) empty_queue.push(host_batch);
        throw;
    }
    session->rethrow_if_error();

    Loss::EvaluationResult epoch_result =
        average_epoch_metrics(device_metrics.read(), batches_number, tracks_accuracy);
    back_propagation.error = epoch_result.error;
    back_propagation.accuracy = epoch_result.accuracy;

    if (profile_this)
    {
        worker_profile.publish();
        const auto epoch_t1 = chrono::steady_clock::now();
        const double epoch_ms = chrono::duration<double, milli>(epoch_t1 - epoch_t0).count();
        ::opennn::global_stats().print(cout, "Epoch breakdown (graph training)", epoch_ms);
        cout << "  Wall-clock epoch time: " << fixed << setprecision(2) << epoch_ms << " ms"
             << " | workers_number=" << get_batch_workers_number(*neural_network) << "\n\n";
        ::opennn::global_stats().clear();
    }

    return epoch_result;
}

struct Optimizer::EpochLoopContext
{
    ThreadSafeQueue<Batch*>* empty_queue = nullptr;
    const vector<vector<Index>>* batches = nullptr;
    const vector<Index>* input_feature_indices = nullptr;
    const vector<Index>* decoder_feature_indices = nullptr;
    const vector<Index>* target_feature_indices = nullptr;

    bool is_training = true;
    bool on_gpu = false;
    Batch* fixed_device_batch = nullptr;

    WorkerProfileCounters* worker_profile = nullptr;

    function<void(Batch& compute_batch, Loss::EvaluationResult& host_result)> step;
};

Loss::EvaluationResult Optimizer::run_epoch_loop(EpochLoopContext& context)
{
    Loss::EvaluationResult epoch_result;

    const Index batches_number = Index(context.batches->size());
    const bool on_gpu = context.on_gpu;

    auto session = start_batch_prefetch(*context.empty_queue,
                                        *context.batches,
                                        *context.input_feature_indices,
                                        *context.decoder_feature_indices,
                                        *context.target_feature_indices,
                                        context.is_training,
                                        context.worker_profile);

    Batch* const fixed_device_batch = context.fixed_device_batch;
    const bool use_fixed_device_batch = fixed_device_batch && fixed_device_batch->uses_cuda();
    bool fixed_device_batch_in_use = false;

    Batch* next_batch = nullptr;
    auto issue_device_input = [&](Batch& batch)
    {
        if (use_fixed_device_batch)
        {
            PROFILE_SCOPE_HOST("step:fixed_h2d_issue");
            if (fixed_device_batch_in_use)
                device::stream_wait_event(Backend::get_transfer_stream(), fixed_device_batch->h2d_done_event);

            batch.upload_to_device_batch_async(*fixed_device_batch, Backend::get_transfer_stream());
            return;
        }

        PROFILE_SCOPE_HOST("step:prefetch_h2d_issue");
        prefetch_batch(batch);
    };

    auto fetch_and_issue = [&](Index iteration)
    {
        PROFILE_SCOPE_HOST("step:wait_fill");
        next_batch = session->wait(iteration);
        issue_device_input(*next_batch);
    };

    auto mark_fixed_device_batch_used = [&]()
    {
        if (!use_fixed_device_batch) return;

        device::record_event(fixed_device_batch->h2d_done_event, Backend::get_compute_stream());
        fixed_device_batch_in_use = true;
    };

    fetch_and_issue(0);

    for (Index iteration = 0; iteration < batches_number; ++iteration)
    {
        Batch* current_batch = next_batch;
        next_batch = nullptr;

        if (!use_fixed_device_batch && iteration + 1 < batches_number)
            fetch_and_issue(iteration + 1);

        if (on_gpu) current_batch->wait_h2d_on_compute_stream();
        Batch& compute_batch = use_fixed_device_batch ? *fixed_device_batch : *current_batch;

        context.step(compute_batch, epoch_result);

        mark_fixed_device_batch_used();

        {
            PROFILE_SCOPE("step:sync_device");
            sync_device(on_gpu);
        }

        context.empty_queue->push(current_batch);

        if (use_fixed_device_batch && iteration + 1 < batches_number)
            fetch_and_issue(iteration + 1);
    }

    session->rethrow_if_error();
    return epoch_result;
}

Loss::EvaluationResult Optimizer::train_epoch(
    ForwardPropagation& forward_propagation,
    BackPropagation& back_propagation,
    ThreadSafeQueue<Batch*>& empty_queue,
    const vector<vector<Index>>& batches,
    const vector<Index>& input_feature_indices,
    const vector<Index>& decoder_feature_indices,
    const vector<Index>& target_feature_indices,
    const function<void(BackPropagation&)>& update,
    Batch* fixed_device_batch)
{
    Loss::EvaluationResult epoch_result;

    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index batches_number = Index(batches.size());
    if (batches_number == 0) return epoch_result;
    const bool tracks_accuracy = loss->get_error() == Loss::Error::CrossEntropy3d;

    has_recurrent_layers_ = neural_network->has(LayerType::Recurrent)
                         || neural_network->has(LayerType::LongShortTermMemory);
    const bool on_gpu = neural_network->is_gpu();

    auto set_epoch_loss = [&]()
    {
        const TensorView parameters(neural_network->get_parameters_data(),
                                    {neural_network->get_parameters_size()},
                                    Type::FP32,
                                    neural_network->get_device());
        back_propagation.regularization = loss->calculate_regularization(parameters);
        back_propagation.loss = epoch_result.error + back_propagation.regularization;
    };

    const bool use_device_metrics = on_gpu && loss->supports_device_epoch_metrics();
    DeviceEpochMetricSums device_metrics;
    if (use_device_metrics) device_metrics.reset();

    static const bool profile_this = env_flag_enabled("OPENNN_PROFILE");
    if (profile_this)
    {
        ::opennn::enabled() = true;
        ::opennn::global_stats().clear();
    }
    const auto epoch_t0 = chrono::steady_clock::now();

    if (!on_gpu)
    {
        Batch* batch = empty_queue.pop();

        for (Index iteration = 0; iteration < batches_number; ++iteration)
        {
            {
                PROFILE_SCOPE_HOST("step:fill");
                batch->fill(batches[iteration],
                            input_feature_indices,
                            decoder_feature_indices,
                            target_feature_indices,
                            /*is_training=*/true);
            }

            {
                PROFILE_SCOPE("step:fwd_total");
                neural_network->forward_propagate(batch->get_inputs(), forward_propagation, true);
            }

            {
                PROFILE_SCOPE("step:bwd_total");
                loss->back_propagate(*batch, forward_propagation, back_propagation);
            }

            epoch_result.error += back_propagation.error;
            if (tracks_accuracy) epoch_result.accuracy += back_propagation.accuracy;

            {
                PROFILE_SCOPE("step:optim_total");
                update(back_propagation);
            }
        }

        empty_queue.push(batch);

        epoch_result = average_epoch_metrics(epoch_result, batches_number, tracks_accuracy);
        set_epoch_loss();

        if (profile_this)
        {
            const auto epoch_t1 = chrono::steady_clock::now();
            const double epoch_ms = chrono::duration<double, milli>(epoch_t1 - epoch_t0).count();
            ::opennn::global_stats().print(cout, "Epoch breakdown (training)", epoch_ms);
            cout << "  Wall-clock epoch time: " << fixed << setprecision(2) << epoch_ms << " ms"
                 << " | workers_number=0\n\n";
            ::opennn::global_stats().clear();
        }

        return epoch_result;
    }

    if (graph_epoch_enabled(use_device_metrics, fixed_device_batch))
    {
        epoch_result = run_graph_epoch(forward_propagation, back_propagation,
                                        empty_queue, batches,
                                        input_feature_indices, decoder_feature_indices,
                                        target_feature_indices,
                                        fixed_device_batch);
        set_epoch_loss();
        return epoch_result;
    }

    WorkerProfileCounters worker_profile;

    EpochLoopContext context;
    context.empty_queue = &empty_queue;
    context.batches = &batches;
    context.input_feature_indices = &input_feature_indices;
    context.decoder_feature_indices = &decoder_feature_indices;
    context.target_feature_indices = &target_feature_indices;
    context.is_training = true;
    context.on_gpu = on_gpu;
    context.fixed_device_batch = fixed_device_batch;
    context.worker_profile = profile_this ? &worker_profile : nullptr;
    context.step = [&](Batch& compute_batch, Loss::EvaluationResult& host_result)
    {
        {
            PROFILE_SCOPE("step:fwd_total");
            neural_network->forward_propagate(compute_batch.get_inputs(), forward_propagation, true);
        }
        sync_cuda_for_debug(on_gpu);

        {
            PROFILE_SCOPE("step:bwd_total");
            if (use_device_metrics)
            {
                if (!loss->back_propagate_device_metrics(compute_batch,
                                                          forward_propagation,
                                                          back_propagation,
                                                          device_metrics.error_sum(),
                                                          tracks_accuracy ? device_metrics.accuracy_sum() : nullptr))
                    throw runtime_error("Device epoch metrics unexpectedly unsupported for this loss.");
            }
            else
            {
                loss->back_propagate(compute_batch, forward_propagation, back_propagation);
            }
        }
        sync_cuda_for_debug(on_gpu);

        if (!use_device_metrics)
        {
            host_result.error += back_propagation.error;
            if (tracks_accuracy) host_result.accuracy += back_propagation.accuracy;
        }

        {
            PROFILE_SCOPE("step:optim_total");
            update(back_propagation);
        }
        sync_cuda_for_debug(on_gpu);
    };

    epoch_result = run_epoch_loop(context);

    if (use_device_metrics)
    {
        epoch_result = average_epoch_metrics(device_metrics.read(), batches_number, tracks_accuracy);
        back_propagation.error = epoch_result.error;
        back_propagation.accuracy = epoch_result.accuracy;
    }
    else
    {
        epoch_result = average_epoch_metrics(epoch_result, batches_number, tracks_accuracy);
    }
    set_epoch_loss();

    if (profile_this)
    {
        const auto epoch_t1 = chrono::steady_clock::now();
        const double epoch_ms = chrono::duration<double, milli>(epoch_t1 - epoch_t0).count();

        worker_profile.publish();

        ::opennn::global_stats().print(cout, "Epoch breakdown (training)", epoch_ms);
        cout << "  Wall-clock epoch time: " << fixed << setprecision(2) << epoch_ms << " ms"
                  << " | workers_number=" << get_batch_workers_number(*neural_network) << "\n\n";
        ::opennn::global_stats().clear();
    }

    return epoch_result;
}

Loss::EvaluationResult Optimizer::evaluate_epoch(
    ForwardPropagation& forward_propagation,
    ThreadSafeQueue<Batch*>& empty_queue,
    const vector<vector<Index>>& batches,
    const vector<Index>& input_feature_indices,
    const vector<Index>& decoder_feature_indices,
    const vector<Index>& target_feature_indices)
{
    Loss::EvaluationResult epoch_result;

    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index batches_number = Index(batches.size());
    if (batches_number == 0) return epoch_result;
    const bool tracks_accuracy = loss->get_error() == Loss::Error::CrossEntropy3d;

    has_recurrent_layers_ = neural_network->has(LayerType::Recurrent)
                         || neural_network->has(LayerType::LongShortTermMemory);
    const bool on_gpu = neural_network->is_gpu();

    const bool use_device_metrics = on_gpu && loss->supports_device_epoch_metrics();
    DeviceEpochMetricSums device_metrics;
    if (use_device_metrics) device_metrics.reset();

    if (!on_gpu)
    {
        Batch* batch = empty_queue.pop();

        for (Index iteration = 0; iteration < batches_number; ++iteration)
        {
            batch->fill(batches[iteration],
                        input_feature_indices,
                        decoder_feature_indices,
                        target_feature_indices,
                        /*is_training=*/false);

            neural_network->forward_propagate(batch->get_inputs(), forward_propagation, false);

            const Loss::EvaluationResult evaluation_result = loss->calculate_error(*batch, forward_propagation);

            epoch_result.error += evaluation_result.error;
            if (tracks_accuracy) epoch_result.accuracy += evaluation_result.accuracy;
        }

        empty_queue.push(batch);

        epoch_result = average_epoch_metrics(epoch_result, batches_number, tracks_accuracy);

        return epoch_result;
    }

    EpochLoopContext context;
    context.empty_queue = &empty_queue;
    context.batches = &batches;
    context.input_feature_indices = &input_feature_indices;
    context.decoder_feature_indices = &decoder_feature_indices;
    context.target_feature_indices = &target_feature_indices;
    context.is_training = false;
    context.on_gpu = on_gpu;
    context.step = [&](Batch& compute_batch, Loss::EvaluationResult& host_result)
    {
        neural_network->forward_propagate(compute_batch.get_inputs(), forward_propagation, false);
        sync_cuda_for_debug(on_gpu);

        if (use_device_metrics)
        {
            if (!loss->calculate_error_device_metrics(compute_batch,
                                                      forward_propagation,
                                                      device_metrics.error_sum(),
                                                      tracks_accuracy ? device_metrics.accuracy_sum() : nullptr))
                throw runtime_error("Device epoch metrics unexpectedly unsupported for this loss.");
        }
        else
        {
            const Loss::EvaluationResult evaluation_result = loss->calculate_error(compute_batch, forward_propagation);
            host_result.error += evaluation_result.error;
            if (tracks_accuracy) host_result.accuracy += evaluation_result.accuracy;
        }
        sync_cuda_for_debug(on_gpu);
    };

    epoch_result = run_epoch_loop(context);

    if (use_device_metrics)
        return average_epoch_metrics(device_metrics.read(), batches_number, tracks_accuracy);

    epoch_result = average_epoch_metrics(epoch_result, batches_number, tracks_accuracy);
    return epoch_result;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
