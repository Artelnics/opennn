//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <functional>
#include "batch.h"
#include "json.h"
#include "loss.h"
#include "tensor_utilities.h"
#include "thread_safe_queue.h"

namespace opennn
{

inline constexpr float GRADIENT_NORM_EPS = 1e-6f;

class NeuralNetwork;
struct Buffer;
struct ForwardPropagation;
struct BackPropagation;

struct TrainingResult;
struct BatchFillSession;

class Optimizer
{

public:

    Optimizer(Loss* = nullptr);
    virtual ~Optimizer();

    enum class StoppingCondition{None,
                                 MinimumLossDecrease,
                                 LossGoal,
                                 MaximumValidationErrorIncreases,
                                 MaximumEpochsNumber,
                                 MaximumTime};

    const Loss* get_loss() const { return loss; }

    bool get_display() const { return display; }

    void set(Loss* new_loss) { loss = new_loss; }

    virtual void set_loss(Loss* new_loss) { loss = new_loss; }

    virtual void set_display(bool new_display) { display = new_display; }

    void set_display_period(const Index new_display_period) { display_period = new_display_period; }

    void set_num_workers(int new_num_workers) { num_workers = max(1, new_num_workers); }
    int  get_num_workers() const { return num_workers; }

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
                                   float training_error, Index validation_failures) const;

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

    void warn_dropped_samples(Index batch_size,
                              Index samples_number,
                              const char* context) const;

    struct BatchPools
    {
        ThreadSafeQueue<Batch*> training_empty_queue;
        ThreadSafeQueue<Batch*> validation_empty_queue;

        vector<unique_ptr<Batch>> training_pool;
        vector<unique_ptr<Batch>> validation_pool;
        unique_ptr<Batch> fixed_training_batch;

        bool validation_uses_training_pool = false;

        ThreadSafeQueue<Batch*>& validation_queue();
    };

    void setup_batch_pools(BatchPools&,
                           Dataset&,
                           NeuralNetwork&,
                           Index training_batch_size,
                           Index validation_batch_size,
                           bool has_validation);

    struct WorkerProfileCounters;

    unique_ptr<BatchFillSession> start_batch_workers(
        ThreadSafeQueue<Batch*>& empty_queue,
        const vector<vector<Index>>& batches,
        const vector<Index>& input_feature_indices,
        const vector<Index>& decoder_feature_indices,
        const vector<Index>& target_feature_indices,
        bool is_training,
        WorkerProfileCounters* profile_counters = nullptr);

    int get_effective_num_workers(const NeuralNetwork&) const;
    int get_batch_pool_size(const NeuralNetwork&) const;
    void apply_effective_num_workers(const NeuralNetwork&);

    struct EpochLoopContext;
    Loss::EvaluationResult run_epoch_loop(EpochLoopContext& context);

    void reset_graph_capture();

    bool graph_epoch_enabled(bool use_device_metrics, Batch* fixed_device_batch) const;
    Loss::EvaluationResult run_graph_epoch(ForwardPropagation& forward_propagation,
                                           BackPropagation& back_propagation,
                                           ThreadSafeQueue<Batch*>& empty_queue,
                                           const vector<vector<Index>>& batches,
                                           const vector<Index>& input_feature_indices,
                                           const vector<Index>& decoder_feature_indices,
                                           const vector<Index>& target_feature_indices,
                                           bool show_progress,
                                           Batch* fixed_device_batch);

    Loss::EvaluationResult train_epoch(ForwardPropagation& forward_propagation,
                                       BackPropagation& back_propagation,
                                       ThreadSafeQueue<Batch*>& empty_queue,
                                       const vector<vector<Index>>& batches,
                                       const vector<Index>& input_feature_indices,
                                       const vector<Index>& decoder_feature_indices,
                                       const vector<Index>& target_feature_indices,
                                       const function<void(BackPropagation&)>& update,
                                       bool show_progress = true,
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

    Index maximum_epochs = 10000;

    float maximum_time = 360000.0f;

    Index display_period = 10;

    bool display = true;

    string name;

    int num_workers = 2;

    bool has_recurrent_layers_ = false;

    void* training_graph_exec = nullptr;
    bool  training_graph_captured = false;
    function<void(BackPropagation&)> graph_update;
};

struct OptimizerData
{
    OptimizerData() = default;
    virtual ~OptimizerData() = default;

    virtual void print() const;

    void set(const vector<Shape>& slot_shapes, Device device = Device::CPU);

    Buffer data;
    vector<TensorView> views;

    VectorR potential_parameters;
    VectorR training_direction;
    float initial_learning_rate = 0.0f;
    Index iteration = 0;

    Buffer graph_step{Device::CUDA};
    Buffer graph_effective_lr{Device::CUDA};
    Buffer graph_effective_eps{Device::CUDA};
};

struct TrainingResult
{
    TrainingResult(const Index = 0);
    virtual ~TrainingResult() = default;

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
