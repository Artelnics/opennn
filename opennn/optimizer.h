//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include "json.h"
#include "tensor_utilities.h"
#include "thread_safe_queue.h"

namespace opennn
{

inline constexpr float GRADIENT_NORM_EPS = 1e-6f;

class Loss;
struct Batch;
struct Buffer;
struct ForwardPropagation;
struct BackPropagation;

struct TrainingResults;

// Persistent pool of worker threads. The previous design spawned a fresh
// vector<jthread> per epoch — for long training runs (1000+ epochs) that
// added thousands of unnecessary pthread_create/_join syscalls. Workers
// are created once, then sleep on a condition_variable between epochs.
//
// Usage:
//   pool.submit([&](){ /* worker body: claim atomic iterations, fill batches */ });
//   ...consume ready[] in main thread...
//   pool.wait();   // block until every worker has returned from its job
//
// Exception safety: if the consumer scope throws while workers are still
// running, the caller must call wait() before unwinding (the worker closures
// typically capture stack references). See train_epoch's try/catch.
class WorkerPool
{
public:
    explicit WorkerPool(int num_workers);
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    [[nodiscard]] int size() const { return static_cast<int>(workers_.size()); }

    void submit(function<void()> job);
    void wait();

    // If any worker caught an exception during its job, rethrow it from
    // the calling (consumer) thread and clear the stored error. Call this
    // inside busy-wait loops to avoid hanging when a worker dies.
    [[nodiscard]] bool has_error() const noexcept
    {
        return error_pending_.load(memory_order_acquire);
    }
    void rethrow_if_error();

private:
    vector<jthread> workers_;
    mutex                mutex_;
    condition_variable   cv_start_;
    condition_variable   cv_done_;
    function<void()>     current_job_;
    uint64_t             generation_  = 0;
    int                  outstanding_ = 0;

    mutex               error_mutex_;
    exception_ptr       worker_error_;
    atomic<bool>        error_pending_{false};
};

class Optimizer
{

public:

    struct EpochStats
    {
        float error = 0.0f;
        float accuracy = 0.0f;
    };

    Optimizer(Loss* = nullptr);
    virtual ~Optimizer() = default;

    enum class StoppingCondition{None,
                                 MinimumLossDecrease,
                                 LossGoal,
                                 MaximumValidationErrorIncreases,
                                 MaximumSelectionErrorIncreases = MaximumValidationErrorIncreases,
                                 MaximumEpochsNumber,
                                 MaximumTime};

    const Loss* get_loss() const { return loss; }

    bool get_display() const { return display; }

    void set(Loss* new_loss) { loss = new_loss; }

    virtual void set_loss(Loss* new_loss) { loss = new_loss; }

    virtual void set_display(bool new_display) { display = new_display; }

    void set_display_period(const Index new_display_period) { display_period = new_display_period; }

    void set_num_workers(int n) { num_workers = max(1, n); }
    int  get_num_workers() const { return num_workers; }

    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

    void set_loss_goal(const float new_loss_goal) { training_loss_goal = new_loss_goal; }
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    virtual TrainingResults train() = 0;

    Index get_maximum_batch_size() const;

    const string& get_name() const { return name; }

    virtual void print() const {}

    virtual void from_JSON(const JsonDocument&);

    virtual void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    static float get_elapsed_time(const time_t& beginning_time);

protected:

    void set_names();
    void set_scaling();
    void set_unscaling();

    bool check_stopping_condition(TrainingResults&, Index epoch, float elapsed_time,
                                   float training_error, Index validation_failures) const;

    void write_common_json(JsonWriter&) const;
    void read_common_json(const Json*);

    void setup_device_training(const vector<Batch*>& batches);
    void teardown_device_training();

    // All GPU work (kernels, cuBLAS, H→D copies) runs on Backend's single
    // compute_stream. prefetch_batch enqueues the copy; the kernels that read
    // it land later on the same stream, so ordering is implicit and
    // wait_prefetch / record_batch_reuse have no work to do. They're kept as
    // no-op hooks to leave the train_epoch / evaluate_epoch loop structure
    // untouched and for future use if a real prefetch stream is reintroduced.
    void prefetch_batch(Batch& batch, Index sample_count);
    void wait_prefetch(Batch& batch);
    void record_batch_reuse(Batch& batch);

    void sync_device();

    static void clip_gradient_norm(Buffer& gradient, float max_norm);

    bool should_display(Index epoch) const { return display && epoch % display_period == 0; }

    void warn_dropped_samples(Index batch_size,
                              Index samples_number,
                              const char* context) const;

    EpochStats train_epoch(bool tracks_accuracy,
                           ForwardPropagation& forward_propagation,
                           BackPropagation& back_propagation,
                           ThreadSafeQueue<Batch*>& empty_queue,
                           const vector<vector<Index>>& batches,
                           const vector<Index>& input_feature_indices,
                           const vector<Index>& decoder_feature_indices,
                           const vector<Index>& target_feature_indices,
                           const function<void(BackPropagation&)>& update,
                           bool show_progress = true);

    EpochStats evaluate_epoch(bool tracks_accuracy,
                              ForwardPropagation& forward_propagation,
                              ThreadSafeQueue<Batch*>& empty_queue,
                              const vector<vector<Index>>& batches,
                              const vector<Index>& input_feature_indices,
                              const vector<Index>& decoder_feature_indices,
                              const vector<Index>& target_feature_indices);

    Loss* loss = nullptr;

    float training_loss_goal = 0.0f;

    Index maximum_validation_failures = numeric_limits<Index>::max();

    Index maximum_epochs = 10000;

    float maximum_time = 360000.0f;

    Index display_period = 10;

    bool display = true;

    string name;

    int num_workers = 2;

    // Created lazily on first train() call; sized at num_workers. Re-created
    // if the user changes num_workers between train() invocations.
    unique_ptr<WorkerPool> worker_pool;

    Buffer prefetch_fp32_staging{Device::CUDA};

    // Refreshed at the start of every train_epoch / evaluate_epoch call.
    // Drives sync_device(): recurrent nets need a per-batch stream drain to
    // keep the driver queue from saturating with their many small kernels;
    // non-recurrent nets don't and benefit from the missing sync via
    // CPU/GPU overlap. See sync_device() for the full rationale.
    bool has_recurrent_layers_ = false;
};

struct OptimizerData
{
    OptimizerData() = default;
    virtual ~OptimizerData() = default;

    virtual void print() const;

    void set(const vector<Shape>& slot_shapes, Device device = Device::CPU);

    Buffer data;
    vector<TensorView> views;

    // Shared state across all optimizers
    VectorR potential_parameters;
    VectorR training_direction;
    float initial_learning_rate = 0.0f;
    Index iteration = 0;
};

struct TrainingResults
{
    TrainingResults(const Index = 0);
    virtual ~TrainingResults() = default;

    string write_stopping_condition() const;

    float get_training_error() const;

    float get_validation_error() const;

    Index get_epochs_number() const;

    void save(const filesystem::path&) const;

    void print(const string& message = {}) const;

    Optimizer::StoppingCondition stopping_condition = Optimizer::StoppingCondition::None;

    Tensor<string, 2> write_override_results(const Index = 3) const;

    void resize_training_error_history(const Index);

    void resize_validation_error_history(const Index);

    VectorR training_error_history;

    VectorR validation_error_history;

    string elapsed_time;

    float loss = NAN;

    Index validation_failures = 0;

    bool restored_best_parameters = false;

    Index restored_epoch = -1;

    float loss_decrease = 0.0f;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
