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
struct ForwardPropagation;
struct BackPropagation;

class Optimizer
{

public:

    Optimizer(Loss* = nullptr);
    virtual ~Optimizer();

    using StoppingCondition = opennn::StoppingCondition;

    const Loss* get_loss() const { return loss; }

    bool get_display() const { return display; }

    void set(Loss* new_loss) { loss = new_loss; }

    virtual void set_loss(Loss* new_loss) { set(new_loss); }

    virtual void set_display(bool new_display) { display = new_display; }

    void set_display_period(const Index new_display_period) { display_period = new_display_period; }

    void set_workers_number(int new_workers_number) { workers_number = max(1, new_workers_number); }
    int  get_workers_number() const { return workers_number; }

    void set_cuda_graph(bool enabled) { use_cuda_graph = enabled; }
    bool get_cuda_graph() const { return use_cuda_graph; }

    void set_shuffle(bool enabled) { shuffle_samples = enabled; }
    bool get_shuffle() const { return shuffle_samples; }

    void set_batch_pool_size(int size) { batch_pool_size_override = size; }
    int  get_batch_pool_size_override() const { return batch_pool_size_override; }

    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

    void set_loss_goal(const float new_loss_goal) { training_loss_goal = new_loss_goal; }
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    virtual TrainingResult train() = 0;

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

    bool check_stopping_condition(TrainingResult&, Index epoch, float elapsed_time,
                                   float training_error, Index validation_failures,
                                   float training_loss, bool has_validation) const;

    // Track the lowest-validation-error parameters/states seen so far. On a new
    // best, snapshot them and reset validation_failures; otherwise count one
    // failure (epochs-since-best). Used by every optimizer's epoch loop.
    void update_best_parameters(NeuralNetwork* neural_network, float validation_error,
                                Index epoch, Index& validation_failures);

    // If training stopped on MaximumValidationErrorIncreases, restore the
    // snapshot taken by update_best_parameters so the final model is the best
    // one, not the last (possibly worse) epoch's.
    void restore_best_parameters(NeuralNetwork* neural_network, TrainingResult& results);

    void reset_best_parameters();

    void write_common_json(JsonWriter&) const;
    void read_common_json(const Json*);

    void setup_device_training();
    void teardown_device_training();

    void warmup_device_training(ForwardPropagation& training_forward_propagation,
                                BackPropagation& training_back_propagation,
                                ThreadSafeQueue<Batch*>& training_empty_queue,
                                const vector<vector<Index>>& training_batches,
                                const vector<Index>& input_feature_indices,
                                const vector<Index>& decoder_feature_indices,
                                const vector<Index>& target_feature_indices,
                                const function<void(BackPropagation&)>& update,
                                ForwardPropagation* validation_forward_propagation = nullptr,
                                ThreadSafeQueue<Batch*>* validation_empty_queue = nullptr,
                                const vector<vector<Index>>* validation_batches = nullptr,
                                Batch* fixed_training_batch = nullptr);

    void prefetch_batch(Batch& batch);

    void sync_device(bool on_gpu);

    static void clip_gradient_norm(Buffer& gradient, float max_norm);

    bool should_display(Index epoch) const { return display && epoch % display_period == 0; }

    void display_epoch_results(Index epoch, float training_error, float training_accuracy,
                               float validation_error, float validation_accuracy,
                               bool has_validation, bool is_token_cross_entropy,
                               float elapsed_time) const;

    void warn_dropped_samples(Index batch_size,
                              Index samples_number,
                              const char* context) const;

    void setup_batch_pools(BatchPools&,
                           Dataset&,
                           NeuralNetwork&,
                           Index training_batch_size,
                           Index validation_batch_size,
                           bool has_validation);

    struct WorkerProfileCounters;

    unique_ptr<BatchPrefetchSession> start_batch_prefetch(
        ThreadSafeQueue<Batch*>& empty_queue,
        const vector<vector<Index>>& batches,
        const vector<Index>& input_feature_indices,
        const vector<Index>& decoder_feature_indices,
        const vector<Index>& target_feature_indices,
        bool is_training,
        WorkerProfileCounters* profile_counters = nullptr);

    int get_batch_workers_number(const NeuralNetwork&) const;
    int get_batch_pool_size(const NeuralNetwork&) const;

    struct EpochLoopContext;
    Loss::EvaluationResult run_epoch_loop(EpochLoopContext& context);

    void reset_graph_capture();

    bool cuda_graph_requested() const;
    bool graph_epoch_enabled(bool use_device_metrics, Batch* fixed_device_batch) const;
    Loss::EvaluationResult run_graph_epoch(ForwardPropagation& forward_propagation,
                                           BackPropagation& back_propagation,
                                           ThreadSafeQueue<Batch*>& empty_queue,
                                           const vector<vector<Index>>& batches,
                                           const vector<Index>& input_feature_indices,
                                           const vector<Index>& decoder_feature_indices,
                                           const vector<Index>& target_feature_indices,
                                           Batch* fixed_device_batch);

    Loss::EvaluationResult train_epoch(ForwardPropagation& forward_propagation,
                                       BackPropagation& back_propagation,
                                       ThreadSafeQueue<Batch*>& empty_queue,
                                       const vector<vector<Index>>& batches,
                                       const vector<Index>& input_feature_indices,
                                       const vector<Index>& decoder_feature_indices,
                                       const vector<Index>& target_feature_indices,
                                       const function<void(BackPropagation&)>& update,
                                       Batch* fixed_device_batch = nullptr);

    Loss::EvaluationResult evaluate_epoch(ForwardPropagation& forward_propagation,
                                          ThreadSafeQueue<Batch*>& empty_queue,
                                          const vector<vector<Index>>& batches,
                                          const vector<Index>& input_feature_indices,
                                          const vector<Index>& decoder_feature_indices,
                                          const vector<Index>& target_feature_indices);

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
    // set_shuffle(); recurrent/LSTM networks always train in order regardless.
    bool shuffle_samples = true;

    // Prefetch-pool depth override (0 = auto). Set from code with
    // set_batch_pool_size(); each pooled Batch holds a full input+target copy
    // on the GPU, so lowering it trades prefetch overlap for a larger max batch.
    int batch_pool_size_override = 0;

    string name;

    int workers_number = 2;

    bool has_recurrent_layers_ = false;

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
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
