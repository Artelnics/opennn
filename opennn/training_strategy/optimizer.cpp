//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/dataset/image_dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/dataset/time_series_dataset.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/neural_network/layers/unscaling_layer.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/training_strategy/optimizer.h"
#include "opennn/core/variable.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/dataset/batch.h"
#include "opennn/core/device_backend.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"
#include "opennn/training_strategy/kernel_optimizers.cuh"
#include <atomic>
#include <chrono>
#include <future>
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

static void clip_gradient_norm_device(Buffer&, Index, float) OPENNN_CUDA_STUB_BODY(clip_gradient_norm_device)

#endif

namespace
{

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
    explicit DeviceEpochMetricSums(Buffer& new_values) : values(new_values) {}

    void reset()
    {
        if (!device::is_cuda_build()) return;

        values.grow_to(2 * Index(sizeof(float)));
        device::set_zero_async(values.data, 2 * Index(sizeof(float)),
                               Backend::get_compute_stream());
    }

    float* error_sum() { return values.as<float>(); }
    float* accuracy_sum() { return values.as<float>() + 1; }

    Loss::EvaluationResult read()
    {
        Loss::EvaluationResult sums;
        if (!device::is_cuda_build()) return sums;

        float host[2] = {0.0f, 0.0f};
        const cudaStream_t stream = Backend::get_compute_stream();
        device::copy_async(host, values.data, Index(sizeof(host)),
                           device::CopyKind::DeviceToHost,
                           stream);
        device::synchronize(stream);

        sums.error = host[0];
        sums.accuracy = host[1];
        return sums;
    }

    Buffer& values;
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

    void print_epoch(const chrono::steady_clock::time_point& epoch_t0,
                     const char* banner,
                     int workers_number) const
    {
        publish();

        const double epoch_ms =
            chrono::duration<double, milli>(chrono::steady_clock::now() - epoch_t0).count();
        ::opennn::global_stats().print(cout, banner, epoch_ms);
        cout << "  Wall-clock epoch time: " << fixed << setprecision(2) << epoch_ms << " ms"
             << " | workers_number=" << workers_number << "\n\n";
        ::opennn::global_stats().clear();
    }
};

Optimizer::Optimizer(Loss* new_loss)
{
    set(new_loss);
}

Optimizer::~Optimizer() = default;

void Optimizer::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Optimizer");

    add_json_field(printer, "Display", display);

    printer.close_element();
}

void Optimizer::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "Optimizer");

    set_display(read_json_bool(root_element, "Display"));
}

void Optimizer::save(const filesystem::path& file_name) const
{
    save_json_file(file_name, *this);
}

void Optimizer::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
}

void Optimizer::setup_batch_pools(BatchPools& pools,
                                  Dataset& dataset,
                                  NeuralNetwork& neural_network,
                                  Index training_batch_size,
                                  Index validation_batch_size,
                                  bool has_validation,
                                  TrainingSession& training_session)
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

    const bool validation_reuses_training_pool =
        has_validation && validation_batch_size == training_batch_size;
    const bool training_prefetch_only = neural_network.is_gpu()
                                     && device::is_cuda_build()
                                     && !validation_reuses_training_pool;

    fill_pool(pools.training_empty_queue,
              pools.training_pool,
              training_batch_size,
              training_prefetch_only);

    if (neural_network.is_gpu() && device::is_cuda_build())
    {
        training_session.pipelines[0].slots[0] = make_unique<Batch>(training_batch_size, &dataset, config);

        if (can_use_cuda_graph())
        {
            const Index training_batches = training_batch_size > 0
                ? dataset.get_samples_number(SampleRole::Training) / training_batch_size
                : 0;
            const bool grouped_batches =
                training_batches >= TrainingSession::group_size
                && (dataset.uses_device_residency()
                    || training_session.fixed_batch()->input.type != Type::BF16);

            if (training_batches > 0 && !grouped_batches)
                training_session.pipelines[1].slots[0] = make_unique<Batch>(training_batch_size, &dataset, config);

            if (grouped_batches)
                for (int i = 1; i < TrainingSession::group_size; ++i)
                    training_session.pipelines[0].slots[size_t(i)] = make_unique<Batch>(training_batch_size, &dataset, config);

            if (grouped_batches
                && training_batches >= TrainingSession::slots_count)
                for (int i = 0; i < TrainingSession::group_size; ++i)
                    training_session.pipelines[1].slots[size_t(i)] = make_unique<Batch>(training_batch_size, &dataset, config);
        }
    }

    if (has_validation && !validation_reuses_training_pool)
        fill_pool(pools.validation_empty_queue,
                  pools.validation_pool,
                  validation_batch_size,
                   false);
}

unique_ptr<BatchPrefetchSession> Optimizer::start_batch_prefetch(
    ThreadSafeQueue<Batch*>& empty_queue,
    const vector<vector<Index>>& batches,
    const vector<Index>& input_feature_indices,
    const vector<Index>& decoder_feature_indices,
    const vector<Index>& target_feature_indices,
    FillMode mode,
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
                        mode,
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
                            mode);

                const auto t_fill1 = chrono::steady_clock::now();
                session_ptr->ready_batches[size_t(it)].store(batch, memory_order_release);
                session_ptr->ready_batches[size_t(it)].notify_one();

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

int Optimizer::get_batch_pool_size(const NeuralNetwork& neural_network) const
{

    if (batch_pool_size_override > 0)
        return max(1, batch_pool_size_override);
    return neural_network.is_gpu()
        ? max(get_batch_workers_number(neural_network) + 1, 3)
        : 1;
}

Index Optimizer::get_maximum_batch_size() const
{
    throw_if(!loss,
             "Optimizer::get_maximum_batch_size: loss is not set.");

    const Dataset* dataset = loss->get_dataset();
    const NeuralNetwork* neural_network = loss->get_neural_network();

    throw_if(!dataset,
             "Optimizer::get_maximum_batch_size: dataset is not set.");
    throw_if(!neural_network,
             "Optimizer::get_maximum_batch_size: neural network is not set.");

    const Index training_samples_number =
        dataset->get_samples_number(SampleRole::Training);

    if(training_samples_number <= 0) return 0;

    const Index validation_samples_number =
        dataset->get_samples_number(SampleRole::Validation);

    const bool on_gpu = neural_network->is_gpu();

    Index available_bytes;

    if(on_gpu)
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

        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);

        throw_if(!GlobalMemoryStatusEx(&status),
                 "Optimizer::get_maximum_batch_size: GlobalMemoryStatusEx failed.");

        available_bytes = Index(status.ullAvailPhys);

#else

        throw runtime_error(
            "Optimizer::get_maximum_batch_size: no portable API to query available RAM on this platform.");

#endif
    }

    const double memory_fraction =
        neural_network->has_recurrent_layers() ? 0.6 : 0.8;

    const Index budget = Index(double(available_bytes) * memory_fraction);

    const Index parameters_number = neural_network->get_parameters_number();
    const Index parameters_size =
        get_aligned_size(neural_network->get_parameter_specs());
    const Index slot_size = get_aligned_size(parameters_number);

    const bool bf16_train =
        neural_network->get_training_type() == Type::BF16;

    const bool bf16_input =
        bf16_train && dataset->supports_bf16_inputs();

    const bool fp32_input_staging =
        bf16_input
        && !bf16_host_input_cast_enabled()
        && !dataset->uses_device_residency();

    Index fixed_bytes =
        (neural_network->get_states_size()
         + 2 * parameters_size
         + 2 * slot_size) * Index(sizeof(float));

    if(bf16_train)
        fixed_bytes += parameters_size * Index(sizeof(bfloat16));

    throw_if(fixed_bytes >= budget,
             "Fixed memory ({} MiB) exceeds memory budget ({} MiB).",
             fixed_bytes / (1ull << 20),
             budget / (1ull << 20));

    const Index dynamic_budget = budget - fixed_bytes;

    const Shape input_shape = dataset->get_shape(VariableRole::Input);
    const Shape target_shape = dataset->get_shape(VariableRole::Target);
    const Shape decoder_shape = dataset->get_shape(VariableRole::Decoder);
    const Shape output_shape = neural_network->get_output_shape();

    const Type compute_dtype = bf16_train ? Type::BF16 : Type::FP32;
    const Index batch_copies = on_gpu ? 1 : get_batch_pool_size(*neural_network);

    const auto batch_data_bytes = [&](const Index batch)
    {
        Index bytes = 0;

        if(!input_shape.empty())
            bytes += batch * input_shape.size()
                   * (bf16_input ? Index(sizeof(bfloat16))
                                 : Index(sizeof(float)));

        if(!target_shape.empty())
            bytes += batch * target_shape.size() * Index(sizeof(float));

        if(!decoder_shape.empty())
            bytes += batch * decoder_shape.size() * Index(sizeof(float));

        return batch_copies * bytes;
    };

    const auto run_bytes = [&](const Index batch)
    {
        if(batch <= 0) return Index(0);

        Index bytes =
            get_aligned_bytes(neural_network->get_forward_specs(batch))
            + get_aligned_bytes(neural_network->get_backward_specs(batch))
            + batch_data_bytes(batch);

        if(!output_shape.empty())
            bytes += get_aligned_bytes(batch * output_shape.size(),
                                       compute_dtype);

        if(fp32_input_staging && !input_shape.empty())
            bytes += get_aligned_bytes(batch * input_shape.size(),
                                       Type::FP32);

        return bytes;
    };

    const auto batch_bytes = [&](const Index batch)
    {
        Index bytes = run_bytes(batch);

        if(validation_samples_number > 0
           && batch > validation_samples_number)
        {
            bytes += run_bytes(validation_samples_number);
        }

        return bytes;
    };

    const Index minimum_bytes = batch_bytes(1);

    throw_if(minimum_bytes > dynamic_budget,
             "Not enough memory for batch_size=1: need {} MiB, have {} MiB.",
             minimum_bytes / (1ull << 20),
             dynamic_budget / (1ull << 20));

    const auto batches =
        views::iota(Index(1), training_samples_number + 1);

    const auto first_too_large =
        ranges::partition_point(batches, [&](const Index batch)
        {
            return batch_bytes(batch) <= dynamic_budget;
        });

    return *ranges::prev(first_too_large);
}

void Optimizer::set_names()
{
    const Dataset* dataset = loss->get_dataset();

    const vector<Variable> input_variables = dataset->get_variables(VariableRole::Input);
    const vector<Variable> target_variables = dataset->get_variables(VariableRole::Target);

    NeuralNetwork* neural_network = loss->get_neural_network();

    neural_network->set_input_variables(input_variables);
    neural_network->set_output_variables(target_variables);
}

void Optimizer::set_scaling()
{
    Dataset* dataset = loss->get_dataset();
    NeuralNetwork* neural_network = loss->get_neural_network();

    vector<Descriptives> input_descriptives;
    vector<string> input_scalers;

    Scaling* scaling_layer =
        dynamic_cast<Scaling*>(neural_network->get_first(LayerType::Scaling));

    if(scaling_layer)
    {
        const Index rank = scaling_layer->get_input_shape().rank;

        if(rank <= 2)
        {
            TabularDataset* tabular_dataset = dynamic_cast<TabularDataset*>(dataset);
            throw_if(!tabular_dataset, "Expected TabularDataset.");

            input_scalers = tabular_dataset->get_feature_scalers("Input");
            input_descriptives = tabular_dataset->scale_features("Input");

            scaling_layer->set_descriptives(input_descriptives);
            scaling_layer->set_scalers(input_scalers);
        }
        else if(rank == 3)
        {
            ImageDataset* image_dataset = dynamic_cast<ImageDataset*>(dataset);
            throw_if(!image_dataset, "Expected ImageDataset.");

            image_dataset->set_input_scaling(scaling_layer->get_descriptives(),
                                             scaling_layer->get_scalers(),
                                             scaling_layer->get_min_range(),
                                             scaling_layer->get_max_range());
        }
        else
        {
            throw runtime_error(
                format("Unexpected Scaling input rank: {}", rank));
        }
    }

    Unscaling* unscaling_layer =
        dynamic_cast<Unscaling*>(neural_network->get_first(LayerType::Unscaling));

    if(!unscaling_layer) return;

    const vector<Index> input_indices =
        dataset->get_feature_indices(VariableRole::Input);

    const vector<Index> target_indices =
        dataset->get_feature_indices(VariableRole::Target);

    const bool has_pure_targets = ranges::any_of(target_indices, [&](const Index target)
    {
        return ranges::find(input_indices, target) == input_indices.end();
    });

    vector<Descriptives> target_descriptives;
    vector<string> target_scalers;

    if(has_pure_targets)
    {
        TabularDataset* tabular_dataset = dynamic_cast<TabularDataset*>(dataset);

        throw_if(!tabular_dataset,
                 "Expected TabularDataset for target unscaling.");

        target_descriptives = tabular_dataset->scale_features("Target");
        target_scalers = tabular_dataset->get_feature_scalers("Target");
    }

    vector<Descriptives> unscaling_descriptives;
    vector<string> unscaling_scalers;

    unscaling_descriptives.reserve(target_indices.size());
    unscaling_scalers.reserve(target_indices.size());

    for(size_t i = 0; i < target_indices.size(); ++i)
    {
        const vector<Index>::const_iterator input =
            ranges::find(input_indices, target_indices[i]);

        if(input == input_indices.end())
        {
            unscaling_descriptives.push_back(target_descriptives[i]);
            unscaling_scalers.push_back(target_scalers[i]);
            continue;
        }

        const Index index = distance(input_indices.begin(), input);

        unscaling_descriptives.push_back(input_descriptives[size_t(index)]);
        unscaling_scalers.push_back(input_scalers[size_t(index)]);
    }

    const Index outputs_number = unscaling_layer->get_outputs_number();
    const Index targets_number = ssize(unscaling_descriptives);

    TimeSeriesDataset* time_series = dynamic_cast<TimeSeriesDataset*>(dataset);

    if(time_series
       && time_series->get_multi_target()
       && targets_number > 0
       && outputs_number == targets_number * time_series->get_future_time_steps())
    {
        const Index steps = time_series->get_future_time_steps();

        vector<Descriptives> expanded_descriptives;
        vector<string> expanded_scalers;

        expanded_descriptives.reserve(size_t(outputs_number));
        expanded_scalers.reserve(size_t(outputs_number));

        for(Index i = 0; i < targets_number; ++i)
        {
            for(Index step = 0; step < steps; ++step)
            {
                expanded_descriptives.push_back(unscaling_descriptives[size_t(i)]);
                expanded_scalers.push_back(unscaling_scalers[size_t(i)]);
            }
        }

        unscaling_descriptives = move(expanded_descriptives);
        unscaling_scalers = move(expanded_scalers);
    }

    throw_if(ssize(unscaling_descriptives) != outputs_number,
             "Unscaling setup error: Mismatch between number of target variables and unscaling layer neurons.");

    unscaling_layer->set_descriptives(unscaling_descriptives);
    unscaling_layer->set_scalers(unscaling_scalers);
}

void Optimizer::set_unscaling()
{
    Dataset* dataset = loss->get_dataset();
    TabularDataset* tabular_dataset = dynamic_cast<TabularDataset*>(dataset);
    NeuralNetwork* neural_network = loss->get_neural_network();

    const auto reconstruct_descriptives =
        [](const VectorR& minimums,
           const VectorR& maximums,
           const VectorR& means,
           const VectorR& standard_deviations)
    {
        vector<Descriptives> descriptives;
        descriptives.reserve(minimums.size());

        for(Index i = 0; i < minimums.size(); ++i)
            descriptives.emplace_back(minimums[i],
                                      maximums[i],
                                      means[i],
                                      standard_deviations[i]);

        return descriptives;
    };

    Scaling* scaling_layer =
        dynamic_cast<Scaling*>(neural_network->get_first(LayerType::Scaling));

    if(scaling_layer && scaling_layer->get_input_shape().rank <= 2)
    {
        throw_if(!tabular_dataset, "Expected TabularDataset.");

        tabular_dataset->unscale_features(
            "Input",
            reconstruct_descriptives(scaling_layer->get_minimums(),
                                     scaling_layer->get_maximums(),
                                     scaling_layer->get_means(),
                                     scaling_layer->get_standard_deviations()));
    }

    Unscaling* unscaling_layer =
        dynamic_cast<Unscaling*>(neural_network->get_first(LayerType::Unscaling));

    if(!unscaling_layer) return;

    const vector<Descriptives> target_descriptives =
        reconstruct_descriptives(unscaling_layer->get_minimums(),
                                 unscaling_layer->get_maximums(),
                                 unscaling_layer->get_means(),
                                 unscaling_layer->get_standard_deviations());

    const vector<Index> input_indices =
        dataset->get_feature_indices(VariableRole::Input);

    const vector<Index> target_indices =
        dataset->get_feature_indices(VariableRole::Target);

    vector<Descriptives> unscaled_descriptives;
    unscaled_descriptives.reserve(target_indices.size());

    for(size_t i = 0; i < target_indices.size(); ++i)
    {
        if(ranges::find(input_indices, target_indices[i]) != input_indices.end())
            continue;

        if(i < target_descriptives.size())
            unscaled_descriptives.push_back(target_descriptives[i]);
    }

    if(unscaled_descriptives.empty()) return;

    throw_if(!tabular_dataset, "Expected TabularDataset.");

    tabular_dataset->unscale_features("Target", unscaled_descriptives);
}

void Optimizer::warmup_device_training(
    ForwardPropagation& training_forward_propagation,
    BackPropagation& training_back_propagation,
    ThreadSafeQueue<Batch*>& training_empty_queue,
    const vector<vector<Index>>& training_batches,
    const vector<Index>& input_feature_indices,
    const vector<Index>& decoder_feature_indices,
    const vector<Index>& target_feature_indices,
    TrainingSession& training_session,
    OptimizerData& optimizer_data,
    ForwardPropagation* validation_forward_propagation,
    ThreadSafeQueue<Batch*>* validation_empty_queue,
    const vector<vector<Index>>* validation_batches)
{
    NeuralNetwork* neural_network = loss ? loss->get_neural_network() : nullptr;

    if(!device::is_cuda_build()
       || !neural_network
       || !neural_network->is_gpu()
       || training_batches.empty())
        return;

    const cudaStream_t stream = Backend::get_compute_stream();

    const Index parameters_bytes =
        neural_network->get_parameters_buffer_size() * Index(sizeof(float));

    const Index states_bytes =
        neural_network->get_states_buffer_size() * Index(sizeof(float));

    Buffer parameters_snapshot{Device::CPU};
    Buffer states_snapshot{Device::CPU};

    if(parameters_bytes > 0)
    {
        parameters_snapshot.resize_bytes(parameters_bytes, Device::CPU);

        device::copy_async(parameters_snapshot.data,
                           neural_network->get_parameters_data(),
                           parameters_bytes,
                           device::CopyKind::DeviceToHost,
                           stream);
    }

    if(states_bytes > 0)
    {
        states_snapshot.resize_bytes(states_bytes, Device::CPU);

        device::copy_async(states_snapshot.data,
                           neural_network->get_states_data(),
                           states_bytes,
                           device::CopyKind::DeviceToHost,
                           stream);
    }

    // The warmup runs a real training step, so undoing it means the optimizer
    // state as well as the model: train_epoch advances Adam's moments and its
    // bias-correction counter, and SGD's momentum buffer. setup_optimizer_data
    // clears all of them -- it zeroes the slot buffer unconditionally, not only
    // when the buffer is resized. Resetting here rather than after the call
    // keeps the invariant on the throwing path too, and stops a future optimizer
    // from silently opting out.
    const auto restore_pre_warmup_state = [&]()
    {
        if(parameters_bytes > 0)
        {
            device::copy_async(neural_network->get_parameters_data(),
                               parameters_snapshot.data,
                               parameters_bytes,
                               device::CopyKind::HostToDevice,
                               stream);

            neural_network->cast_parameters_to_bf16();
        }

        if(states_bytes > 0)
        {
            device::copy_async(neural_network->get_states_data(),
                               states_snapshot.data,
                               states_bytes,
                               device::CopyKind::HostToDevice,
                               stream);
        }

        device::synchronize(stream);

        setup_optimizer_data(optimizer_data,
                             neural_network->get_parameters_buffer_size(),
                             neural_network->get_device());
    };

    const vector<vector<Index>> training_warmup_batch{training_batches.front()};
    const function<void(NeuralNetwork*)> saved_post_batch_callback = post_batch_callback;
    post_batch_callback = {};

    try
    {
        if(validation_forward_propagation
           && validation_empty_queue
           && validation_batches
           && !validation_batches->empty())
        {
            const vector<vector<Index>> validation_warmup_batch{
                validation_batches->front()
            };

            evaluate_epoch(*validation_forward_propagation,
                           *validation_empty_queue,
                           validation_warmup_batch,
                           input_feature_indices,
                           decoder_feature_indices,
                           target_feature_indices,
                           training_session);
        }

        train_epoch(training_forward_propagation,
                    training_back_propagation,
                    training_empty_queue,
                    training_warmup_batch,
                    input_feature_indices,
                    decoder_feature_indices,
                    target_feature_indices,
                    training_session,
                    optimizer_data);

        restore_pre_warmup_state();
        post_batch_callback = saved_post_batch_callback;
    }
    catch(...)
    {
        restore_pre_warmup_state();
        post_batch_callback = saved_post_batch_callback;
        throw;
    }
}

void Optimizer::display_epoch_results(const Index epoch,
                                      const float training_error,
                                      const float training_accuracy,
                                      const float validation_error,
                                      const float validation_accuracy,
                                      const bool has_validation,
                                      const bool validation_fresh,
                                      const bool is_token_cross_entropy,
                                      const float elapsed_time) const
{
    if (!should_display(epoch)) return;

    cout << "Training error: " << training_error << "\n";
    if (is_token_cross_entropy) {
        cout << "Training perplexity: " << exp(training_error) << "\n";
        cout << "Training accuracy: " << training_accuracy << "\n";
    }
    if (has_validation) {
        if (validation_fresh)
        {
            cout << "Validation error: " << validation_error << "\n";
            if (is_token_cross_entropy) {
                cout << "Validation perplexity: " << exp(validation_error) << "\n";
                cout << "Validation accuracy: " << validation_accuracy << "\n";
            }
        }
        else
            cout << "Validation error: ---\n";
    }
    cout << "Elapsed time: " << get_time(elapsed_time) << "\n";
}

TrainingResult Optimizer::train()
{
    TrainingResult results(maximum_epochs + 1);

    if (!loss || !loss->get_neural_network() || !loss->get_dataset())
        return results;

    NeuralNetwork* neural_network = loss->get_neural_network();
    neural_network->warn_if_stale_configuration();

    const bool on_gpu = neural_network->is_gpu();

    if (display) cout << "Training with " << get_display_name()
                     << (on_gpu ? " CUDA" : "") << "...\n";

    Dataset* dataset = loss->get_dataset();

    const bool has_validation = dataset->has_validation();

    const vector<Index> input_feature_indices = dataset->get_feature_indices(VariableRole::Input);
    const vector<Index> target_feature_indices = dataset->get_feature_indices(VariableRole::Target);
    const vector<Index> decoder_feature_indices = dataset->get_feature_indices(VariableRole::Decoder);

    const vector<Index> training_sample_indices = dataset->get_sample_indices(SampleRole::Training);
    const vector<Index> validation_sample_indices = dataset->get_sample_indices(SampleRole::Validation);

    const Index training_samples_number = dataset->get_samples_number(SampleRole::Training);
    const Index validation_samples_number = dataset->get_samples_number(SampleRole::Validation);

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
        ? (training_samples_number + training_batch_size - 1) / training_batch_size
        : 0;

    vector<vector<Index>> training_batches(training_batches_number);
    vector<vector<Index>> validation_batches;

    set_names();
    set_scaling();

    BatchPools batch_pools;
    OptimizerData optimizer_data;
    TrainingSession training_session;

    setup_batch_pools(batch_pools,
                      *dataset,
                      *neural_network,
                      training_batch_size,
                      validation_batch_size,
                      has_validation,
                      training_session);

    // BackPropagation owns the delta layout; ForwardPropagation only co-plans the
    // lifetimes, so it never needs to see the Loss.
    const vector<MemoryPoolEntry> delta_lifetimes =
        BackPropagation::make_co_planned_lifetimes(*loss, training_batch_size);

    ForwardPropagation training_forward_propagation(
        training_batch_size,
        neural_network,
        ForwardPropagationMode::Training,
        {},
        true,
        delta_lifetimes);

    loss->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_batch_size, *loss,
                                              &training_forward_propagation.arena,
                                              training_forward_propagation.co_planned_offsets);

    unique_ptr<ForwardPropagation> validation_forward_propagation;
    if (has_validation)
    {
        validation_forward_propagation = make_unique<ForwardPropagation>();
        validation_forward_propagation->set(validation_batch_size, neural_network,
                                            &training_forward_propagation.arena,
                                            ForwardPropagationMode::Inference);
    }

    ForwardPropagation* validation_fp = validation_forward_propagation.get();
    mark_validation_propagation(validation_fp);

    setup_device_training();

    const Index parameters_buffer_size = neural_network->get_parameters_buffer_size();
    const Device device = neural_network->get_device();

    float training_error = 0.0f;
    float training_accuracy = 0.0f;
    float validation_error = 0.0f;
    float validation_accuracy = 0.0f;
    Index validation_failures = 0;
    BestModelSnapshot best_model;

    const bool is_token_cross_entropy = (loss->get_error() == Loss::Error::CrossEntropy3d);

    setup_optimizer_data(optimizer_data, parameters_buffer_size, device);

    const bool needs_cuda_warmup = on_gpu && device::is_cuda_build() && training_batches_number > 0;

    if (needs_cuda_warmup)
    {
        dataset->get_batches(training_sample_indices, training_batch_size, false, training_batches);
        if (has_validation)
            dataset->get_batches(validation_sample_indices, validation_batch_size, false, validation_batches);

        warmup_device_training(training_forward_propagation,
                               training_back_propagation,
                               batch_pools.training_empty_queue,
                               training_batches,
                               input_feature_indices,
                               decoder_feature_indices,
                               target_feature_indices,
                               training_session,
                               optimizer_data,
                               validation_fp,
                               has_validation ? &batch_pools.validation_queue() : nullptr,
                               has_validation ? &validation_batches : nullptr);
    }

    const bool has_training_tail = training_batch_size > 0
        && training_samples_number % training_batch_size != 0;
    const bool has_validation_tail = has_validation
        && validation_batch_size > 0
        && validation_samples_number % validation_batch_size != 0;

    training_session.cuda_graph_capture_allowed = training_session.has_graph_batches()
        && !has_training_tail;

    time_t beginning_time;
    time(&beginning_time);
    float elapsed_time = 0.0f;

    {
        device::CudaAllocationGrowthGuard steady_state_guard(
            needs_cuda_warmup && !has_training_tail && !has_validation_tail);

        // Shuffling and slicing 10M+ sample indices costs a visible fraction of a
        // fast GPU epoch, so the next epoch's batches are built on a helper thread
        // while the GPU trains the current one. The shared RNG draw order is
        // unchanged (one shuffle per epoch, same sequence), so batch composition
        // stays identical to the synchronous path.
        vector<vector<Index>> next_training_batches;
        future<void> next_training_batches_ready;

        for (Index epoch = 0; epoch <= maximum_epochs; ++epoch)
        {
            if (should_display(epoch)) cout << "Epoch: " << epoch << "\n";

            if (next_training_batches_ready.valid())
            {
                next_training_batches_ready.get();
                training_batches.swap(next_training_batches);
            }
            else
                dataset->get_batches(training_sample_indices, training_batch_size, shuffle_samples, training_batches);

            if (on_gpu && epoch + 1 < maximum_epochs)
                next_training_batches_ready = async(launch::async, [&]
                {
                    dataset->get_batches(training_sample_indices, training_batch_size,
                                         shuffle_samples, next_training_batches);
                });

            on_epoch_begin(epoch, optimizer_data);

            const Loss::EvaluationResult training_evaluation_result = train_epoch(training_forward_propagation,
                                                                                 training_back_propagation,
                                                                                 batch_pools.training_empty_queue,
                                                                                 training_batches,
                                                                                 input_feature_indices,
                                                                                 decoder_feature_indices,
                                                                                 target_feature_indices,
                                                                                 training_session,
                                                                                 optimizer_data);

            training_error = training_evaluation_result.error;
            training_accuracy = training_evaluation_result.accuracy;
            results.training_error_history(epoch) = training_error;

            // validation_period, not display_period: this gate decides whether validation
            // is *computed*, and everything downstream reads the result. Gating it on the
            // printing cadence left validation_error_history at its -1 sentinel on every
            // epoch that was not a multiple of display_period, which get_validation_error()
            // and the minimal_index() best-epoch search then consumed as real values.
            const bool val_fresh = has_validation && (epoch % validation_period == 0);

            if (val_fresh)
            {
                dataset->get_batches(validation_sample_indices, validation_batch_size, false, validation_batches);

                const Loss::EvaluationResult validation_evaluation_result = evaluate_epoch(*validation_fp,
                                                                                          batch_pools.validation_queue(),
                                                                                          validation_batches,
                                                                                          input_feature_indices,
                                                                                          decoder_feature_indices,
                                                                                          target_feature_indices,
                                                                                          training_session);

                validation_error = validation_evaluation_result.error;
                validation_accuracy = validation_evaluation_result.accuracy;
                results.validation_error_history(epoch) = validation_error;

                update_best_parameters(neural_network, validation_error, epoch,
                                       validation_failures, best_model);
            }

            elapsed_time = get_elapsed_time(beginning_time);

            display_epoch_results(epoch, training_error, training_accuracy,
                                  validation_error, validation_accuracy,
                                  has_validation, val_fresh, is_token_cross_entropy, elapsed_time);

            if (post_epoch_callback)
                post_epoch_callback(epoch, training_error, validation_error, neural_network);

            if (check_stopping_condition(results, epoch, elapsed_time,
                                         results.training_error_history(epoch),
                                         validation_failures,
                                         training_back_propagation.metrics.loss_value,
                                         has_validation))
                break;
        }
    }

    teardown_device_training();

    restore_best_parameters(neural_network, results, best_model);

    set_unscaling();

    if (display) results.print();

    return results;
}

void Optimizer::prepare_full_batch_training(FullBatchContext& context, const char* banner)
{
    if (display) cout << banner << "\n";

    Dataset* dataset = loss->get_dataset();
    NeuralNetwork* neural_network = loss->get_neural_network();

    context.neural_network = neural_network;
    context.training_samples_number = dataset->get_samples_number(SampleRole::Training);
    context.validation_samples_number = dataset->get_samples_number(SampleRole::Validation);

    const vector<Index> training_sample_indices = dataset->get_sample_indices(SampleRole::Training);
    const vector<Index> validation_sample_indices = dataset->get_sample_indices(SampleRole::Validation);

    const vector<Index> input_feature_indices = dataset->get_feature_indices(VariableRole::Input);
    const vector<Index> target_feature_indices = dataset->get_feature_indices(VariableRole::Target);

    set_names();
    set_scaling();

    context.training_batch = make_unique<Batch>(context.training_samples_number,
                                                dataset,
                                                neural_network->get_config());
    context.training_batch->fill(training_sample_indices, input_feature_indices, {},
                                 target_feature_indices, FillMode::Training);

    context.validation_batch = make_unique<Batch>(context.validation_samples_number,
                                                  dataset,
                                                  neural_network->get_config());
    context.validation_batch->fill(validation_sample_indices, input_feature_indices, {},
                                   target_feature_indices, FillMode::Validation);

    context.training_forward_propagation =
        make_unique<ForwardPropagation>(
            context.training_samples_number,
            neural_network,
            ForwardPropagationMode::Training,
            InferenceShapePolicy{},
            true);

    if (context.validation_samples_number > 0)
        context.validation_forward_propagation =
            make_unique<ForwardPropagation>(context.validation_samples_number, neural_network,
                                            ForwardPropagationMode::Inference);

    mark_validation_propagation(context.validation_forward_propagation.get());

    loss->set_normalization_coefficient();
}

TrainingResult Optimizer::train_full_batch(FullBatchContext& context, const FullBatchHooks& hooks)
{
    TrainingResult results(maximum_epochs + 1);

    NeuralNetwork* neural_network = context.neural_network;
    const bool has_validation = context.validation_forward_propagation != nullptr;

    Index validation_failures = 0;
    BestModelSnapshot best_model;

    float old_loss = 0.0f;
    float loss_decrease = MAX;

    time_t beginning_time;
    time(&beginning_time);
    float elapsed_time = 0.0f;

    if (hooks.setup_state) hooks.setup_state();

    for (Index epoch = 0; epoch <= maximum_epochs; ++epoch)
    {
        if (should_display(epoch)) cout << "Epoch: " << epoch << "\n";

        neural_network->forward_propagate(context.training_batch->get_inputs(),
                                          *context.training_forward_propagation,
                                          true);

        const FullBatchStep step = hooks.train_step();

        results.training_error_history(epoch) = step.training_error;

        float validation_error = 0.0f;

        if (has_validation)
        {
            neural_network->forward_propagate(context.validation_batch->get_inputs(),
                                              *context.validation_forward_propagation,
                                              false);

            validation_error = hooks.validation_error();

            results.validation_error_history(epoch) = validation_error;

            update_best_parameters(neural_network, validation_error, epoch,
                                   validation_failures, best_model);
        }

        elapsed_time = get_elapsed_time(beginning_time);

        if (should_display(epoch))
        {
            cout << "Training error: " << step.displayed_error << "\n";
            if (has_validation) cout << "Validation error: " << validation_error << "\n";
            if (hooks.display_extra) hooks.display_extra();
            cout << "Elapsed time: " << get_time(elapsed_time) << "\n";
        }

        if (epoch != 0) loss_decrease = old_loss - step.loss;

        old_loss = step.loss;

        if (loss_decrease < hooks.minimum_loss_decrease)
        {
            if (display) cout << "Epoch " << epoch << "\nMinimum loss decrease reached: " << loss_decrease << "\n";
            results.stopping_condition = StoppingCondition::MinimumLossDecrease;
        }

        if (check_stopping_condition(results, epoch, elapsed_time,
                                     results.training_error_history(epoch),
                                     validation_failures,
                                     step.loss,
                                     has_validation))
            break;

        if (hooks.post_step) hooks.post_step();
    }

    restore_best_parameters(neural_network, results, best_model);

    set_unscaling();

    if (display) results.print();

    return results;
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
        else if (epoch + 1 >= maximum_epochs)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum epochs number reached: " << epoch + 1 << "\n";
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
    results.resize_training_error_history(epoch + 1);
    results.resize_validation_error_history(has_validation ? epoch + 1 : 0);
    results.elapsed_time = get_time(elapsed_time);

    return true;
}

void Optimizer::update_best_parameters(NeuralNetwork* neural_network,
                                       float validation_error,
                                       Index epoch,
                                       Index& validation_failures,
                                       BestModelSnapshot& best_model)
{
    constexpr float MIN_DELTA = 1e-7f;

    if (std::isnan(validation_error))
        return;

    if (validation_error >= best_model.validation_error - MIN_DELTA)
    {
        ++validation_failures;
        return;
    }

    best_model.validation_error = validation_error;
    best_model.epoch = epoch;
    validation_failures = 0;

    if (post_best_callback)
        post_best_callback(epoch, validation_error);

    const tuple<vector<float>&, const float*, Index> snapshots[] = {
        {best_model.parameters, neural_network->get_parameters_data(), neural_network->get_parameters_buffer_size()},
        {best_model.states,     neural_network->get_states_data(),     neural_network->get_states_buffer_size()}
    };

    for (const auto& [destination, source, size] : snapshots)
    {
        if (size == 0) continue;

        if (Index(destination.size()) != size)
            destination.resize(size);

        const size_t bytes = size_t(size) * sizeof(float);
        if (neural_network->is_gpu() && device::is_cuda_build())
        {
            const cudaStream_t stream = Backend::get_compute_stream();
            device::copy_async(destination.data(), source, Index(bytes),
                               device::CopyKind::DeviceToHost, stream);
            device::synchronize(stream);
        }
        else
            memcpy(destination.data(), source, bytes);
    }
}

void Optimizer::restore_best_parameters(NeuralNetwork* neural_network,
                                        TrainingResult& results,
                                        const BestModelSnapshot& best_model)
{
    if (!restore_best
        || best_model.parameters.empty()
        || Index(best_model.parameters.size()) != neural_network->get_parameters_buffer_size())
        return;

    if (display)
        cout << "Restoring best parameters and states from epoch " << best_model.epoch
             << " (validation error " << best_model.validation_error << ")\n";

    neural_network->set_parameters(Map<const VectorR>(best_model.parameters.data(),
                                                       Index(best_model.parameters.size())));

    if (!best_model.states.empty())
        neural_network->set_states(Map<const VectorR>(best_model.states.data(),
                                                      Index(best_model.states.size())));

    results.restored_epoch = best_model.epoch;
}

void Optimizer::write_common_json(JsonWriter& printer) const
{
    write_json(printer, {
        {"LossGoal", training_loss_goal},
        {"MaximumValidationFailures", maximum_validation_failures},
        {"MaximumEpochsNumber", maximum_epochs},
        {"MaximumTime", maximum_time},
        {"GradientClipNorm", gradient_clip_norm},
        {"DisplayPeriod", display_period}
    });
}

void Optimizer::read_common_json(const Json* root_element)
{
    set_loss_goal(read_json_float(root_element, "LossGoal"));
    set_maximum_validation_failures(read_json_index(root_element,
        root_element->has("MaximumValidationFailures") ? "MaximumValidationFailures" : "MaximumSelectionFailures"));
    set_maximum_epochs(read_json_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_json_float(root_element, "MaximumTime"));

    if (root_element->has("GradientClipNorm"))
        set_gradient_clip_norm(read_json_float(root_element, "GradientClipNorm"));

    if (root_element->has("DisplayPeriod"))
        set_display_period(read_json_index(root_element, "DisplayPeriod"));
}

void Optimizer::setup_device_training()
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    if (!neural_network->is_gpu()) return;

    neural_network->copy_parameters_device();
    neural_network->copy_states_device();

    if (loss->get_dataset()->uses_device_residency())
        loss->get_dataset()->enable_device_residency();
}

void Optimizer::teardown_device_training()
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    if (!neural_network->is_gpu()) return;

    device::synchronize(Backend::get_compute_stream());

    if (loss->get_dataset()->is_device_resident())
        loss->get_dataset()->disable_device_residency();

    neural_network->copy_parameters_host();
    neural_network->copy_states_host();
}

void Optimizer::prefetch_batch(Batch& batch)
{
    if (!batch.uses_cuda()) return;

    batch.upload_to_device_batch_async(batch, Backend::get_transfer_stream());
}

void Optimizer::sync_device(const bool on_gpu,
                            const bool has_recurrent_layers,
                            TrainingSession& training_session)
{
    if (!on_gpu) return;

    if (!has_recurrent_layers) return;

    CudaEvent& slot = training_session.throttle_events[training_session.throttle_cursor];
    training_session.throttle_cursor =
        (training_session.throttle_cursor + 1) % training_session.throttle_events.size();

    if (slot.handle)
        device::synchronize_event(slot.handle);
    else
        slot.create();

    device::record_event(slot.handle, Backend::get_compute_stream());
}

void Optimizer::clip_gradient_norm(Buffer& gradient, float max_norm)
{
    const Index gradient_size = gradient.size_in_floats();
    if (max_norm <= 0.0f || gradient_size <= 0) return;

    if (gradient.device_type == Device::CUDA)
        clip_gradient_norm_device(gradient, gradient_size, max_norm);
    else
    {
        VectorMap gradient_view = gradient.as_vector();
        const float gradient_norm = gradient_view.norm();
        if (gradient_norm > max_norm)
            gradient_view *= max_norm / (gradient_norm + GRADIENT_NORM_EPS);
    }
}

Loss::EvaluationResult Optimizer::run_graph_epoch(
    TrainingSession& training_session,
    OptimizerData& optimizer_data,
    ForwardPropagation& forward_propagation,
    BackPropagation& back_propagation,
    ThreadSafeQueue<Batch*>& empty_queue,
    const vector<vector<Index>>& batches,
    const vector<Index>& input_feature_indices,
    const vector<Index>& decoder_feature_indices,
    const vector<Index>& target_feature_indices)
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index batches_number = Index(batches.size());
    const bool tracks_accuracy = loss->get_error() == Loss::Error::CrossEntropy3d;

    DeviceEpochMetricSums device_metrics(training_session.device_metrics);
    device_metrics.reset();

    const cudaStream_t compute = Backend::get_compute_stream();
    const cudaStream_t transfer = Backend::get_transfer_stream();

    auto& pipelines = training_session.pipelines;
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
                                        FillMode::Training,
                                        profile_this ? &worker_profile : nullptr);

    const bool staged_h2d = !loss->get_dataset()->is_device_resident()
                         && training_session.fixed_batch()->input.type != Type::BF16;

    const auto stage_into_slot = [](const Batch& source, Batch& slot)
    {
        const auto copy_section = [&](const BatchSlot& from, BatchSlot& to)
        {
            const Index values_count = from.shape.size();
            if (!from.host || !to.host || values_count <= 0) return;
            memcpy(to.host, from.host, size_t(values_count) * sizeof(float));
        };

        copy_section(source.input,   slot.input);
        copy_section(source.decoder, slot.decoder);
        copy_section(source.target,  slot.target);
    };

    const auto issue_slot_h2d = [](Batch& slot, cudaStream_t stream)
    {
        const auto copy_section = [&](BatchSlot& section)
        {
            const Index values_count = section.shape.size();
            if (!section.host || !section.buffer.data || values_count <= 0) return;
            device::copy_async(section.buffer.data, section.host,
                               values_count * Index(sizeof(float)),
                               device::CopyKind::HostToDevice, stream);
        };

        copy_section(slot.input);
        copy_section(slot.decoder);
        copy_section(slot.target);
    };

    const auto run_compute_step = [&](Batch& slot)
    {
        neural_network->forward_propagate(slot.get_inputs(),
                                          forward_propagation, true);
        if (!loss->back_propagate_device_metrics(slot,
                                                 forward_propagation, back_propagation,
                                                 device_metrics.error_sum(),
                                                 tracks_accuracy ? device_metrics.accuracy_sum() : nullptr))
            throw runtime_error("Device epoch metrics unexpectedly unsupported for this loss.");
        update_parameters(back_propagation, optimizer_data, UpdateMode::Capturable);
    };

    const auto capture_or_run = [&](device::GraphExecHandle& exec,
                                    const auto& operation)
    {
        if (exec)
        {
            PROFILE_SCOPE_HOST("step:graph_launch");
            device::launch_graph(exec, compute);
            return;
        }
        if (!training_session.cuda_graph_capture_allowed)
        {
            operation();
            return;
        }

        const bool profiler_enabled = ::opennn::enabled();
        ::opennn::enabled() = false;
        try
        {
            device::synchronize(compute);
            device::StreamCapture capture(compute);
            operation();
            capture.end(exec);
            device::launch_graph(exec, compute);
        }
        catch (const exception& capture_error)
        {
            training_session.disable_cuda_graph_capture();
            cout << "CUDA graph capture failed (" << capture_error.what()
                 << "); continuing without graphs.\n";
            ::opennn::enabled() = profiler_enabled;
            operation();
            return;
        }
        ::opennn::enabled() = profiler_enabled;
    };

    const bool resident_gather = loss->get_dataset()->is_device_resident();

    constexpr Index M = Index(TrainingSession::group_size);
    Batch* host_batch = nullptr;

    const auto run_grouped_epoch = [&](const auto& stage_slot,
                                       const auto& launch_group,
                                       const auto& stage_tail_slot)
    {
        const Index groups = batches_number / M;

        for (Index group = 0; group < groups; ++group)
        {
            TrainingSession::GraphPipeline& pipeline =
                pipelines[size_t(group) % pipelines.size()];
            Batch& event_slot = *pipeline.slots[size_t(M) - 1];

            {
                PROFILE_SCOPE_HOST("step:group_sync");

                if (event_slot.h2d_done_recorded)
                    device::synchronize_event(event_slot.h2d_done_event);
            }

            for (Index m = 0; m < M; ++m)
            {
                {
                    PROFILE_SCOPE_HOST("step:wait_fill");
                    host_batch = session->wait(group * M + m);
                }
                stage_slot(*pipeline.slots[size_t(m)]);
            }

            launch_group(pipeline);

            event_slot.record_h2d_done(compute);
        }

        for (Index iteration = groups * M; iteration < batches_number; ++iteration)
        {
            host_batch = session->wait(iteration);
            Batch& slot = *training_session.fixed_batch();
            device::synchronize(compute);
            stage_tail_slot(slot);
            run_compute_step(slot);
        }
    };

    try
    {
        if (resident_gather
            && batches_number >= Index(TrainingSession::group_size))
        {
            run_grouped_epoch(
                [&](Batch& slot)
                {
                    PROFILE_SCOPE_HOST("step:gather_issue");
                    slot.device_gather = host_batch->device_gather;
                    empty_queue.push(host_batch);
                    host_batch = nullptr;

                    slot.upload_to_device_batch_async(slot, transfer);
                    slot.wait_h2d_on_compute_stream();
                },
                [&](TrainingSession::GraphPipeline& pipeline)
                {
                    const auto run_group = [&] {
                        for (Index m = 0; m < M; ++m)
                            run_compute_step(*pipeline.slots[size_t(m)]);
                    };
                    capture_or_run(pipeline.exec, run_group);
                },
                [&](Batch& slot)
                {
                    slot.device_gather = host_batch->device_gather;
                    empty_queue.push(host_batch);
                    host_batch = nullptr;
                    slot.upload_to_device_batch_async(slot, transfer);
                    slot.wait_h2d_on_compute_stream();
                });
        }
        else if (staged_h2d
            && batches_number >= Index(TrainingSession::group_size))
        {
            run_grouped_epoch(
                [&](Batch& slot)
                {
                    PROFILE_SCOPE_HOST("step:stage_copy");
                    stage_into_slot(*host_batch, slot);

                    empty_queue.push(host_batch);
                    host_batch = nullptr;
                },
                [&](TrainingSession::GraphPipeline& pipeline)
                {
                    if (!pipeline.fork_event)
                        pipeline.fork_event.create();
                    for (Index m = 0; m < M; ++m)
                        if (!pipeline.copy_done_events[size_t(m)])
                            pipeline.copy_done_events[size_t(m)].create();

                    const auto run_group = [&] {
                        device::record_event(pipeline.fork_event, compute);
                        device::stream_wait_event(transfer, pipeline.fork_event);
                        for (Index m = 0; m < M; ++m)
                        {
                            issue_slot_h2d(*pipeline.slots[size_t(m)], transfer);
                            device::record_event(pipeline.copy_done_events[size_t(m)], transfer);
                        }
                        for (Index m = 0; m < M; ++m)
                        {
                            device::stream_wait_event(compute, pipeline.copy_done_events[size_t(m)]);
                            run_compute_step(*pipeline.slots[size_t(m)]);
                        }
                    };
                    capture_or_run(pipeline.exec, run_group);
                },
                [&](Batch& slot)
                {
                    stage_into_slot(*host_batch, slot);
                    empty_queue.push(host_batch);
                    host_batch = nullptr;
                    issue_slot_h2d(slot, compute);
                });
        }
        else
        {
            for (Index iteration = 0; iteration < batches_number; ++iteration)
            {
                TrainingSession::GraphPipeline& pipeline =
                    pipelines[size_t(iteration) % pipelines.size()];
                Batch& slot = *pipeline.slots[0];

                {
                    PROFILE_SCOPE_HOST("step:wait_fill");
                    host_batch = session->wait(iteration);
                }

                if (staged_h2d)
                {
                    {
                        PROFILE_SCOPE_HOST("step:stage_copy");

                        if (slot.h2d_done_recorded)
                            device::synchronize_event(slot.h2d_done_event);
                        stage_into_slot(*host_batch, slot);
                        empty_queue.push(host_batch);
                        host_batch = nullptr;
                    }
                    const auto run_slot = [&] {
                        issue_slot_h2d(slot, compute);
                        run_compute_step(slot);
                    };
                    capture_or_run(pipeline.exec, run_slot);
                }
                else
                {
                    {
                        PROFILE_SCOPE_HOST("step:h2d_issue");

                        if (slot.h2d_done_recorded)
                            device::stream_wait_event(transfer, slot.h2d_done_event);
                        host_batch->upload_to_device_batch_async(slot, transfer);
                        host_batch->wait_h2d_on_compute_stream();
                    }
                    const auto run_slot = [&] { run_compute_step(slot); };
                    capture_or_run(pipeline.exec, run_slot);
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
    back_propagation.metrics.error = epoch_result.error;
    back_propagation.metrics.accuracy = epoch_result.accuracy;

    if (profile_this)
        worker_profile.print_epoch(epoch_t0, "Epoch breakdown (graph training)",
                                   get_batch_workers_number(*neural_network));

    return epoch_result;
}

struct Optimizer::EpochLoopContext
{
    ThreadSafeQueue<Batch*>* empty_queue = nullptr;
    const vector<vector<Index>>* batches = nullptr;
    const vector<Index>* input_feature_indices = nullptr;
    const vector<Index>* decoder_feature_indices = nullptr;
    const vector<Index>* target_feature_indices = nullptr;

    FillMode fill_mode = FillMode::Training;
    bool on_gpu = false;
    bool has_recurrent_layers = false;
    TrainingSession* training_session = nullptr;
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
                                        context.fill_mode,
                                        context.worker_profile);

    Batch* const fixed_device_batch = context.fixed_device_batch;
    const bool use_fixed_device_batch = fixed_device_batch && fixed_device_batch->uses_cuda();
    bool fixed_device_batch_in_use = false;

    Batch* next_batch = nullptr;
    auto fetch_and_issue = [&](Index iteration)
    {
        PROFILE_SCOPE_HOST("step:wait_fill");
        next_batch = session->wait(iteration);

        if (use_fixed_device_batch)
        {
            PROFILE_SCOPE_HOST("step:fixed_h2d_issue");
            if (fixed_device_batch_in_use)
                device::stream_wait_event(Backend::get_transfer_stream(), fixed_device_batch->h2d_done_event);

            next_batch->upload_to_device_batch_async(*fixed_device_batch, Backend::get_transfer_stream());
            return;
        }

        PROFILE_SCOPE_HOST("step:prefetch_h2d_issue");
        prefetch_batch(*next_batch);
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

        if (use_fixed_device_batch)
        {
            device::record_event(fixed_device_batch->h2d_done_event, Backend::get_compute_stream());
            fixed_device_batch_in_use = true;
        }

        {
            PROFILE_SCOPE("step:sync_device");
            sync_device(on_gpu, context.has_recurrent_layers, *context.training_session);

            if (on_gpu && context.fill_mode != FillMode::Training)
                device::synchronize(Backend::get_compute_stream());
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
    TrainingSession& training_session,
    OptimizerData& optimizer_data)
{
    Loss::EvaluationResult epoch_result;

    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index all_batches_number = Index(batches.size());

    if(all_batches_number == 0) return epoch_result;

    const bool has_tail = Index(batches.back().size()) != forward_propagation.batch_size;
    vector<vector<Index>> complete_batches;
    if(has_tail)
        complete_batches.assign(batches.begin(), batches.end() - 1);

    const vector<vector<Index>>& epoch_batches = has_tail ? complete_batches : batches;
    const Index batches_number = Index(epoch_batches.size());

    const bool tracks_accuracy = loss->get_error() == Loss::Error::CrossEntropy3d;
    const bool on_gpu = neural_network->is_gpu();
    const bool use_graph_batches = training_session.cuda_graph_capture_allowed
        && training_session.has_graph_batches();

    static const bool profile_this = env_flag_enabled("OPENNN_PROFILE");

    if(profile_this)
    {
        ::opennn::enabled() = true;
        ::opennn::global_stats().clear();
    }

    const chrono::steady_clock::time_point epoch_t0 = chrono::steady_clock::now();
    WorkerProfileCounters worker_profile;

    const auto finalize_epoch = [&](Loss::EvaluationResult& result)
    {
        const TensorView parameters(neural_network->get_parameters_data(),
                                    {neural_network->get_parameters_buffer_size()},
                                    Type::FP32,
                                    neural_network->get_device());

        back_propagation.metrics.regularization = loss->calculate_regularization(parameters);
        back_propagation.metrics.loss_value = result.error + back_propagation.metrics.regularization;
    };

    const auto train_tail = [&]
    {
        Loss::EvaluationResult result;
        if(!has_tail) return result;

        const vector<Index>& sample_indices = batches.back();
        const Index tail_size = Index(sample_indices.size());

        Batch batch(tail_size, loss->get_dataset(), neural_network->get_config());
        batch.fill(sample_indices,
                   input_feature_indices,
                   decoder_feature_indices,
                   target_feature_indices,
                   FillMode::Training);

        if(on_gpu)
        {
            prefetch_batch(batch);
            batch.wait_h2d_on_compute_stream();
        }

        const vector<MemoryPoolEntry> delta_lifetimes =
            BackPropagation::make_co_planned_lifetimes(*loss, tail_size);
        ForwardPropagation tail_forward_propagation(
            tail_size,
            neural_network,
            ForwardPropagationMode::Training,
            {},
            false,
            delta_lifetimes);
        BackPropagation tail_back_propagation(
            tail_size,
            *loss,
            &tail_forward_propagation.arena,
            tail_forward_propagation.co_planned_offsets);

        neural_network->forward_propagate(batch.get_inputs(),
                                          tail_forward_propagation,
                                          true);
        loss->back_propagate(batch,
                             tail_forward_propagation,
                             tail_back_propagation);

        if(!std::isnan(tail_back_propagation.metrics.error))
        {
            result.error = tail_back_propagation.metrics.error;
            result.accuracy = tail_back_propagation.metrics.accuracy;
            result.active_tokens_count = tail_back_propagation.metrics.active_tokens_count;
            update_parameters(tail_back_propagation, optimizer_data);
        }

        if(post_batch_callback)
            post_batch_callback(neural_network);

        if(on_gpu)
            device::synchronize(Backend::get_compute_stream());

        // Constructing a BackPropagation links every layer's gradient view to
        // its buffer. Restore those views before the temporary tail context dies.
        back_propagation.set(forward_propagation.batch_size,
                             *loss,
                             &forward_propagation.arena,
                             forward_propagation.co_planned_offsets);

        return result;
    };

    const auto merge_tail = [&](Loss::EvaluationResult& result)
    {
        if(!has_tail) return;

        const Loss::EvaluationResult tail_result = train_tail();
        const Index complete_samples = batches_number * forward_propagation.batch_size;
        const Index tail_samples = Index(batches.back().size());
        const float total_samples = float(complete_samples + tail_samples);

        result.error = (result.error * float(complete_samples)
                        + tail_result.error * float(tail_samples)) / total_samples;
        if(tracks_accuracy)
            result.accuracy = (result.accuracy * float(complete_samples)
                               + tail_result.accuracy * float(tail_samples)) / total_samples;
        result.active_tokens_count += tail_result.active_tokens_count;

        back_propagation.metrics.error = result.error;
        back_propagation.metrics.accuracy = result.accuracy;
        back_propagation.metrics.active_tokens_count = result.active_tokens_count;
    };

    if(!on_gpu)
    {
        Batch* batch = empty_queue.pop();

        for(Index iteration = 0; iteration < batches_number; ++iteration)
        {
            {
                PROFILE_SCOPE_HOST("step:fill");
                batch->fill(epoch_batches[size_t(iteration)],
                            input_feature_indices,
                            decoder_feature_indices,
                            target_feature_indices,
                            FillMode::Training);
            }

            {
                PROFILE_SCOPE("step:fwd_total");
                neural_network->forward_propagate(batch->get_inputs(),
                                                  forward_propagation,
                                                  true);
            }

            {
                PROFILE_SCOPE("step:bwd_total");
                loss->back_propagate(*batch, forward_propagation, back_propagation);
            }

            if (!std::isnan(back_propagation.metrics.error))
            {
                epoch_result.error += back_propagation.metrics.error;

                if(tracks_accuracy)
                    epoch_result.accuracy += back_propagation.metrics.accuracy;

                {
                    PROFILE_SCOPE("step:optim_total");
                    update_parameters(back_propagation, optimizer_data);
                }
            }

            if(post_batch_callback)
                post_batch_callback(neural_network);
        }

        empty_queue.push(batch);

        epoch_result =
            average_epoch_metrics(epoch_result, batches_number, tracks_accuracy);

        merge_tail(epoch_result);
        finalize_epoch(epoch_result);

        if(profile_this)
            worker_profile.print_epoch(epoch_t0, "Epoch breakdown (training)", 0);

        return epoch_result;
    }

    if(use_graph_batches)
    {
        epoch_result = run_graph_epoch(training_session,
                                       optimizer_data,
                                       forward_propagation,
                                       back_propagation,
                                       empty_queue,
                                       epoch_batches,
                                       input_feature_indices,
                                       decoder_feature_indices,
                                       target_feature_indices);

        merge_tail(epoch_result);
        finalize_epoch(epoch_result);
        return epoch_result;
    }

    const bool use_device_metrics = loss->supports_device_epoch_metrics();

    DeviceEpochMetricSums device_metrics(training_session.device_metrics);

    if(use_device_metrics)
        device_metrics.reset();

    EpochLoopContext context{
        &empty_queue,
        &epoch_batches,
        &input_feature_indices,
        &decoder_feature_indices,
        &target_feature_indices,
        FillMode::Training,
        true,
        neural_network->has_recurrent_layers(),
        &training_session,
        training_session.fixed_batch(),
        profile_this ? &worker_profile : nullptr
    };

    context.step = [&](Batch& batch, Loss::EvaluationResult& result)
    {
        {
            PROFILE_SCOPE("step:fwd_total");
            neural_network->forward_propagate(batch.get_inputs(),
                                              forward_propagation,
                                              true);
        }

        {
            PROFILE_SCOPE("step:bwd_total");

            if(use_device_metrics)
            {
                if(!loss->back_propagate_device_metrics(
                       batch,
                       forward_propagation,
                       back_propagation,
                       device_metrics.error_sum(),
                       tracks_accuracy ? device_metrics.accuracy_sum() : nullptr))
                {
                    throw runtime_error(
                        "Device epoch metrics unexpectedly unsupported for this loss.");
                }
            }
            else
            {
                loss->back_propagate(batch,
                                     forward_propagation,
                                     back_propagation);
            }
        }

        const bool batch_ok = use_device_metrics || !std::isnan(back_propagation.metrics.error);

        if (!use_device_metrics && batch_ok)
        {
            result.error += back_propagation.metrics.error;

            if(tracks_accuracy)
                result.accuracy += back_propagation.metrics.accuracy;
        }

        if (batch_ok)
        {
            PROFILE_SCOPE("step:optim_total");
            update_parameters(back_propagation, optimizer_data);
        }

        if(post_batch_callback)
            post_batch_callback(neural_network);
    };

    if(batches_number > 0)
        epoch_result = run_epoch_loop(context);

    if(use_device_metrics)
        epoch_result = device_metrics.read();

    epoch_result =
        average_epoch_metrics(epoch_result, batches_number, tracks_accuracy);

    if(use_device_metrics)
    {
        back_propagation.metrics.error = epoch_result.error;
        back_propagation.metrics.accuracy = epoch_result.accuracy;
    }

    merge_tail(epoch_result);
    finalize_epoch(epoch_result);

    if(profile_this)
    {
        worker_profile.print_epoch(epoch_t0,
                                   "Epoch breakdown (training)",
                                   get_batch_workers_number(*neural_network));
    }

    return epoch_result;
}

Loss::EvaluationResult Optimizer::evaluate_epoch(
    ForwardPropagation& forward_propagation,
    ThreadSafeQueue<Batch*>& empty_queue,
    const vector<vector<Index>>& batches,
    const vector<Index>& input_feature_indices,
    const vector<Index>& decoder_feature_indices,
    const vector<Index>& target_feature_indices,
    TrainingSession& training_session)
{
    Loss::EvaluationResult epoch_result;

    NeuralNetwork* neural_network = loss->get_neural_network();
    const Index all_batches_number = Index(batches.size());

    if(all_batches_number == 0) return epoch_result;

    const bool has_tail = Index(batches.back().size()) != forward_propagation.batch_size;
    vector<vector<Index>> complete_batches;
    if(has_tail)
        complete_batches.assign(batches.begin(), batches.end() - 1);

    const vector<vector<Index>>& epoch_batches = has_tail ? complete_batches : batches;
    const Index batches_number = Index(epoch_batches.size());

    const bool tracks_accuracy = loss->get_error() == Loss::Error::CrossEntropy3d;
    const bool on_gpu = neural_network->is_gpu();

    const auto evaluate_tail = [&]
    {
        Loss::EvaluationResult result;
        if(!has_tail) return result;

        const vector<Index>& sample_indices = batches.back();
        const Index tail_size = Index(sample_indices.size());

        Batch batch(tail_size, loss->get_dataset(), neural_network->get_config());
        batch.fill(sample_indices,
                   input_feature_indices,
                   decoder_feature_indices,
                   target_feature_indices,
                   FillMode::Validation);

        if(on_gpu)
        {
            prefetch_batch(batch);
            batch.wait_h2d_on_compute_stream();
        }

        ForwardPropagation tail_forward_propagation(
            tail_size,
            neural_network,
            ForwardPropagationMode::Inference,
            {},
            true);
        neural_network->forward_propagate(batch.get_inputs(),
                                          tail_forward_propagation,
                                          false);
        result = loss->calculate_error(batch, tail_forward_propagation);

        if(on_gpu)
            device::synchronize(Backend::get_compute_stream());

        return result;
    };

    const auto merge_tail = [&](Loss::EvaluationResult& result)
    {
        if(!has_tail) return;

        const Loss::EvaluationResult tail_result = evaluate_tail();
        const Index complete_samples = batches_number * forward_propagation.batch_size;
        const Index tail_samples = Index(batches.back().size());
        const float total_samples = float(complete_samples + tail_samples);

        result.error = (result.error * float(complete_samples)
                        + tail_result.error * float(tail_samples)) / total_samples;
        if(tracks_accuracy)
            result.accuracy = (result.accuracy * float(complete_samples)
                               + tail_result.accuracy * float(tail_samples)) / total_samples;
        result.active_tokens_count += tail_result.active_tokens_count;
    };

    if(!on_gpu)
    {
        Batch* batch = empty_queue.pop();

        for(Index iteration = 0; iteration < batches_number; ++iteration)
        {
            batch->fill(epoch_batches[size_t(iteration)],
                        input_feature_indices,
                        decoder_feature_indices,
                        target_feature_indices,
                        FillMode::Validation);

            neural_network->forward_propagate(batch->get_inputs(),
                                              forward_propagation,
                                              false);

            const Loss::EvaluationResult result =
                loss->calculate_error(*batch, forward_propagation);

            epoch_result.error += result.error;

            if(tracks_accuracy)
                epoch_result.accuracy += result.accuracy;
        }

        empty_queue.push(batch);

        epoch_result = average_epoch_metrics(epoch_result,
                                             batches_number,
                                             tracks_accuracy);
        merge_tail(epoch_result);
        return epoch_result;
    }

    const bool use_device_metrics = loss->supports_device_epoch_metrics();

    DeviceEpochMetricSums device_metrics(training_session.device_metrics);

    if(use_device_metrics)
        device_metrics.reset();

    EpochLoopContext context{
        &empty_queue,
        &epoch_batches,
        &input_feature_indices,
        &decoder_feature_indices,
        &target_feature_indices,
        FillMode::Validation,
        true,
        neural_network->has_recurrent_layers(),
        &training_session
    };

    context.step = [&](Batch& batch, Loss::EvaluationResult& result)
    {
        neural_network->forward_propagate(batch.get_inputs(),
                                          forward_propagation,
                                          false);

        if(use_device_metrics)
        {
            throw_if(!loss->calculate_error_device_metrics(
                        batch,
                        forward_propagation,
                        device_metrics.error_sum(),
                        tracks_accuracy ? device_metrics.accuracy_sum() : nullptr),
                    "Device epoch metrics unexpectedly unsupported for this loss.");

            return;        
        }

        const Loss::EvaluationResult evaluation =
            loss->calculate_error(batch, forward_propagation);

        result.error += evaluation.error;

        if(tracks_accuracy)
            result.accuracy += evaluation.accuracy;
    };

    if(batches_number > 0)
        epoch_result = run_epoch_loop(context);

    if(use_device_metrics)
        epoch_result = device_metrics.read();

    epoch_result = average_epoch_metrics(epoch_result,
                                         batches_number,
                                         tracks_accuracy);
    merge_tail(epoch_result);
    return epoch_result;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
