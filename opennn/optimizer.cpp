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

Optimizer::~Optimizer()
{
    device::destroy_graph(training_graph_exec);
}

void Optimizer::reset_graph_capture()
{
    training_graph_captured = false;
    device::destroy_graph(training_graph_exec);
    training_graph_exec = nullptr;
    graph_update = nullptr;
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

    if (neural_network.is_gpu() && device::is_cuda_build())
        pools.fixed_training_batch = make_unique<Batch>(training_batch_size, &dataset, config);

    pools.validation_uses_training_pool =
        has_validation && validation_batch_size == training_batch_size;

    if (has_validation && !pools.validation_uses_training_pool)
        fill_pool(pools.validation_empty_queue,
                  pools.validation_pool,
                  validation_batch_size);
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
    static const bool cuda_graph_requested = env_flag_enabled("OPENNN_CUDA_GRAPH");
    return cuda_graph_requested
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

    cudaStream_t stream = Backend::get_compute_stream();
    Batch* fill_batch = empty_queue.pop();

    const auto run_compute_step = [&]()
    {
        neural_network->forward_propagate(fixed_device_batch->get_inputs(),
                                          forward_propagation, true);
        if (!loss->back_propagate_device_metrics(*fixed_device_batch,
                                                 forward_propagation, back_propagation,
                                                 device_metrics.error_sum(),
                                                 tracks_accuracy ? device_metrics.accuracy_sum() : nullptr))
            throw runtime_error("Device epoch metrics unexpectedly unsupported for this loss.");
        graph_update(back_propagation);
    };

    for (Index iteration = 0; iteration < batches_number; ++iteration)
    {
        fill_batch->fill(batches[iteration],
                         input_feature_indices, decoder_feature_indices,
                         target_feature_indices, /*is_training=*/true);
        fill_batch->upload_to_device_batch_async(*fixed_device_batch, stream);

        if (!training_graph_captured)
        {
            run_compute_step();
            device::synchronize(stream);
            device::begin_graph_capture(stream);
            run_compute_step();
            training_graph_exec = device::end_graph_capture(stream);
            training_graph_captured = true;
        }
        else
        {
            device::launch_graph(training_graph_exec, stream);
        }
    }
    device::synchronize(stream);
    empty_queue.push(fill_batch);

    Loss::EvaluationResult epoch_result =
        average_epoch_metrics(device_metrics.read(), batches_number, tracks_accuracy);
    back_propagation.error = epoch_result.error;
    back_propagation.accuracy = epoch_result.accuracy;
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
