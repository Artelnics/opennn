//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <array>
#include <functional>
#include "batch.h"
#include "device_backend.h"
#include "forward_propagation.h"
#include "json.h"
#include "loss.h"
#include "tensor_types.h"
#include "thread_safe_queue.h"
#include "training_result.h"

namespace opennn
{

inline constexpr float GRADIENT_NORM_EPS = 1e-6f;

class NeuralNetwork;
struct Buffer;
struct BackPropagation;

class Optimizer
{

public:

    explicit Optimizer(Loss* = nullptr);
    virtual ~Optimizer();

    using StoppingCondition = opennn::StoppingCondition;

    const Loss* get_loss() const noexcept { return loss; }

    bool get_display() const noexcept { return display; }

    void set(Loss* new_loss) { loss = new_loss; }

    virtual void set_loss(Loss* new_loss) { set(new_loss); }

    virtual void set_display(bool new_display) { display = new_display; }

    void set_display_period(const Index new_display_period) { display_period = new_display_period; }

    void set_workers_number(int new_workers_number) { workers_number = max(1, new_workers_number); }
    int  get_workers_number() const noexcept { return workers_number; }

    void set_cuda_graph(bool enabled) { use_cuda_graph = enabled; }
    bool get_cuda_graph() const noexcept { return use_cuda_graph; }

    void set_shuffle(bool enabled) { shuffle_samples = enabled; }
    bool get_shuffle() const noexcept { return shuffle_samples; }

    void set_batch_pool_size(int size) { batch_pool_size_override = size; }
    int  get_batch_pool_size_override() const noexcept { return batch_pool_size_override; }

    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

    void set_loss_goal(const float new_loss_goal) { training_loss_goal = new_loss_goal; }
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    virtual TrainingResult train() = 0;

    Index get_maximum_batch_size() const;

    const string& get_name() const noexcept { return name; }

    virtual void print() const {}

    virtual void from_JSON(const JsonDocument&);

    virtual void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    static float get_elapsed_time(const time_t&);

protected:

    void set_names();
    void set_scaling();
    void set_unscaling();

    bool check_stopping_condition(TrainingResult&, Index, float,
                                   float, Index,
                                   float, bool) const;

    // Track the lowest-validation-error parameters/states seen so far. On a new
    // best, snapshot them and reset validation_failures; otherwise count one
    // failure (epochs-since-best). Used by every optimizer's epoch loop.
    void update_best_parameters(NeuralNetwork*, float,
                                Index, Index&);

    void restore_best_parameters(NeuralNetwork*, TrainingResult&);

    void reset_best_parameters();

    // Validation batches come from the dataset that set_scaling() pre-scaled in
    // place, so their forward passes must skip the leading Scaling layers (see
    // ForwardPropagation::inputs_pre_scaled). Every optimizer calls this on the
    // propagation it evaluates validation with.
    static void mark_validation_propagation(ForwardPropagation* validation_propagation)
    {
        if (validation_propagation)
            validation_propagation->inputs_pre_scaled = true;
    }

    void write_common_json(JsonWriter&) const;
    void read_common_json(const Json*);

    void setup_device_training();
    void teardown_device_training();

    void warmup_device_training(ForwardPropagation&,
                                BackPropagation&,
                                ThreadSafeQueue<Batch*>&,
                                const vector<vector<Index>>&,
                                const vector<Index>&,
                                const vector<Index>&,
                                const vector<Index>&,
                                const function<void(BackPropagation&)>&,
                                ForwardPropagation* validation_forward_propagation = nullptr,
                                ThreadSafeQueue<Batch*>* validation_empty_queue = nullptr,
                                const vector<vector<Index>>* validation_batches = nullptr,
                                Batch* fixed_training_batch = nullptr);

    void prefetch_batch(Batch&);

    void sync_device(bool);

    static void clip_gradient_norm(Buffer&, float);

    bool should_display(Index epoch) const noexcept { return display && epoch % display_period == 0; }

    void display_epoch_results(Index, float, float,
                               float, float,
                               bool, bool,
                               float) const;

    void warn_dropped_samples(Index,
                              Index,
                              const char*) const;

    void setup_batch_pools(BatchPools&,
                           Dataset&,
                           NeuralNetwork&,
                           Index,
                           Index,
                           bool);

    struct WorkerProfileCounters;

    unique_ptr<BatchPrefetchSession> start_batch_prefetch(
        ThreadSafeQueue<Batch*>&,
        const vector<vector<Index>>&,
        const vector<Index>&,
        const vector<Index>&,
        const vector<Index>&,
        bool,
        WorkerProfileCounters* profile_counters = nullptr);

    int get_batch_workers_number(const NeuralNetwork&) const;
    int get_batch_pool_size(const NeuralNetwork&) const;

    struct EpochLoopContext;
    Loss::EvaluationResult run_epoch_loop(EpochLoopContext&);

    void reset_graph_capture();

    bool cuda_graph_requested() const;
    bool graph_epoch_enabled(bool, Batch*) const;
    Loss::EvaluationResult run_graph_epoch(ForwardPropagation&,
                                           BackPropagation&,
                                           ThreadSafeQueue<Batch*>&,
                                           const vector<vector<Index>>&,
                                           const vector<Index>&,
                                           const vector<Index>&,
                                           const vector<Index>&,
                                           Batch*);

    Loss::EvaluationResult train_epoch(ForwardPropagation&,
                                       BackPropagation&,
                                       ThreadSafeQueue<Batch*>&,
                                       const vector<vector<Index>>&,
                                       const vector<Index>&,
                                       const vector<Index>&,
                                       const vector<Index>&,
                                       const function<void(BackPropagation&)>&,
                                       Batch* fixed_device_batch = nullptr);

    Loss::EvaluationResult evaluate_epoch(ForwardPropagation&,
                                          ThreadSafeQueue<Batch*>&,
                                          const vector<vector<Index>>&,
                                          const vector<Index>&,
                                          const vector<Index>&,
                                          const vector<Index>&);

    Loss* loss = nullptr;

    float training_loss_goal = 0.0f;

    Index maximum_validation_failures = numeric_limits<Index>::max();

    float best_validation_error = numeric_limits<float>::max();
    Index best_epoch = -1;
    vector<float> best_parameters;
    vector<float> best_states;

    Index maximum_epochs = 10000;

    float maximum_time = 360000.0f;

    Index display_period = 10;

    bool display = true;

    // Shuffle the training samples each epoch. Toggle from code with
    bool shuffle_samples = true;

    // Prefetch-pool depth override (0 = auto). Set from code with
    // set_batch_pool_size(); each pooled Batch holds a full input+target copy
    // on the GPU, so lowering it trades prefetch overlap for a larger max batch.
    int batch_pool_size_override = 0;

    string name;

    int workers_number = 2;

    bool has_recurrent_layers_ = false;

    array<CudaEvent, 4> batch_throttle_events_;
    size_t batch_throttle_cursor_ = 0;

    // Slot ring for the graph epoch. The staged (host FP32) path groups
    // graph_group_size iterations into one mega-graph whose H2D nodes read each
    // slot's pinned host buffer, and ping-pongs two groups over the ring; the
    // upload path (device-resident / BF16) uses slots [0..1] with one graph per
    // slot. Two execs either way (per group parity or per slot).
    static constexpr int graph_group_size = 8;
    static constexpr int graph_slots_count = 2 * graph_group_size;
    array<device::GraphExecHandle, 2> training_graph_execs;
    array<Batch*, graph_slots_count> graph_slots{};
    // Capture-internal fork/join events so the mega-graph's H2D nodes run on a
    // forked stream and overlap compute. Replays re-record them; they must not
    // be used for anything else.
    array<CudaEvent, 2> graph_fork_events;
    array<CudaEvent, graph_slots_count> graph_copy_done_events;
    function<void(BackPropagation&)> graph_update;
    // CUDA graph capture/replay is opt-in. Toggle from code with set_cuda_graph();
    // there is no environment-variable fallback.
    bool use_cuda_graph = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
