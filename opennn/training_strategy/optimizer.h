//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <array>
#include <functional>
#include "opennn/dataset/batch.h"
#include "opennn/core/device_backend.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/core/json.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/thread_safe_queue.h"
#include "opennn/training_strategy/training_result.h"

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

    void set_cuda_graph(bool enabled) { use_cuda_graph = enabled; }

    void set_shuffle(bool enabled) { shuffle_samples = enabled; }

    void set_batch_pool_size(int size) { batch_pool_size_override = size; }

    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    Index get_maximum_epochs() const noexcept { return maximum_epochs; }
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

    void set_loss_goal(const float new_loss_goal) { training_loss_goal = new_loss_goal; }
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    void set_validation_period(const Index n) { validation_period = n; }

    void set_gradient_clip_norm(const float new_clip) { gradient_clip_norm = new_clip; }

    virtual TrainingResult train();

    Index get_maximum_batch_size() const;

    const string& get_name() const noexcept { return name; }

    virtual void print() const {}

    virtual void from_JSON(const JsonDocument&);

    virtual void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    function<void(Index, float, float, NeuralNetwork*)> post_epoch_callback;
    function<void(NeuralNetwork*)> post_batch_callback;
    function<void(Index, float)> post_best_callback;

protected:

    enum class UpdateMode { Standard, Capturable };

    struct TrainingSession
    {
        static constexpr int group_size = 8;
        static constexpr int slots_count = 2 * group_size;
        static constexpr int pipelines_count = 2;

        struct GraphPipeline
        {
            array<unique_ptr<Batch>, group_size> slots;
            device::GraphExecHandle exec;
            CudaEvent fork_event;
            array<CudaEvent, group_size> copy_done_events;
        };

        Batch* fixed_batch() const { return pipelines[0].slots[0].get(); }

        bool has_graph_batches() const
        {
            return pipelines[0].slots[1] || pipelines[1].slots[0];
        }

        void disable_cuda_graph_capture()
        {
            for (GraphPipeline& pipeline : pipelines)
                pipeline.exec.reset();
            cuda_graph_capture_allowed = false;
        }

        // The remainder batch (samples % batch_size) trains eagerly after the
        // whole batches, in contexts of its own size. They are built once - on
        // the warm-up pass, before the steady-state allocation guard arms - and
        // re-linked each epoch, so a tail neither allocates inside the guarded
        // epoch loop nor forces CUDA graph capture off for the whole batches.
        struct TailContext
        {
            unique_ptr<Batch> batch;
            unique_ptr<ForwardPropagation> forward;
            unique_ptr<BackPropagation> backward;
            Index size = 0;
        };

        array<GraphPipeline, pipelines_count> pipelines;
        TailContext tail;
        Buffer device_metrics{Device::CUDA};
        array<CudaEvent, 4> throttle_events;
        size_t throttle_cursor = 0;
        bool cuda_graph_capture_allowed = false;
    };

    void set_names();
    void set_scaling();
    void set_unscaling();

    bool check_stopping_condition(TrainingResult&, 
        Index, float, float, Index, float, bool) const;

    struct BestModelSnapshot
    {
        float validation_error = MAX;
        Index epoch = -1;
        vector<float> parameters;
        vector<float> states;
    };

    void update_best_parameters(NeuralNetwork*,
                                float,
                                Index,
                                Index&,
                                BestModelSnapshot&);

    void restore_best_parameters(NeuralNetwork*,
                                 TrainingResult&,
                                 const BestModelSnapshot&);

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
                                TrainingSession&,
                                OptimizerData&,
                                ForwardPropagation* validation_forward_propagation = nullptr,
                                ThreadSafeQueue<Batch*>* validation_empty_queue = nullptr,
                                const vector<vector<Index>>* validation_batches = nullptr);

    void prefetch_batch(Batch&);

    void sync_device(bool on_gpu, bool has_recurrent_layers, TrainingSession&);

    static void clip_gradient_norm(Buffer&, float);

    bool should_display(Index epoch) const noexcept { return display && epoch % display_period == 0; }

    void display_epoch_results(Index, float, float,
                               float, float,
                               bool, bool, bool,
                               float) const;

    void setup_batch_pools(BatchPools&,
                           Dataset&,
                           NeuralNetwork&,
                           Index,
                           Index,
                           bool,
                           TrainingSession&);

    struct WorkerProfileCounters;

    unique_ptr<BatchPrefetchSession> start_batch_prefetch(
        ThreadSafeQueue<Batch*>&,
        const vector<vector<Index>>&,
        const vector<Index>&,
        const vector<Index>&,
        const vector<Index>&,
        FillMode,
        WorkerProfileCounters* profile_counters = nullptr);

    int get_batch_workers_number(const NeuralNetwork&) const { return workers_number; }
    int get_batch_pool_size(const NeuralNetwork&) const;

    struct EpochLoopContext;
    Loss::EvaluationResult run_epoch_loop(EpochLoopContext&);

    struct FullBatchContext
    {
        NeuralNetwork* neural_network = nullptr;
        Index training_samples_number = 0;
        Index validation_samples_number = 0;
        unique_ptr<Batch> training_batch;
        unique_ptr<Batch> validation_batch;
        unique_ptr<ForwardPropagation> training_forward_propagation;
        unique_ptr<ForwardPropagation> validation_forward_propagation;
    };

    struct FullBatchStep
    {
        float training_error = 0.0f;
        float displayed_error = 0.0f;
        float loss = 0.0f;
    };

    struct FullBatchHooks
    {
        function<void()> setup_state;
        function<FullBatchStep()> train_step;
        function<float()> validation_error;
        function<void()> display_extra;
        function<void()> post_step;
        float minimum_loss_decrease = -MAX;
    };

    void prepare_full_batch_training(FullBatchContext&, const char* banner);
    TrainingResult train_full_batch(FullBatchContext&, const FullBatchHooks&);

    virtual string get_display_name() const { return name; }
    virtual void setup_optimizer_data(OptimizerData&, Index, Device) {}
    virtual void update_parameters(BackPropagation&, OptimizerData&,
                                   UpdateMode = UpdateMode::Standard)
    { 
        throw runtime_error("train() requires a mini-batch optimizer (SGD or Adam)."); 
    }

    virtual bool supports_cuda_graph() const noexcept { return false; }
    bool can_use_cuda_graph() const
    {
        return use_cuda_graph
            && supports_cuda_graph()
            && device::is_cuda_build()
            && loss
            && loss->supports_device_epoch_metrics();
    }
    virtual void on_epoch_begin(Index, OptimizerData&) {}

    Loss::EvaluationResult run_graph_epoch(TrainingSession&,
                                           OptimizerData&,
                                           ForwardPropagation&,
                                           BackPropagation&,
                                           ThreadSafeQueue<Batch*>&,
                                           const vector<vector<Index>>&,
                                           const vector<Index>&,
                                           const vector<Index>&,
                                           const vector<Index>&);

    Loss::EvaluationResult train_epoch(ForwardPropagation&,
                                       BackPropagation&,
                                       ThreadSafeQueue<Batch*>&,
                                       const vector<vector<Index>>&,
                                       const vector<Index>&,
                                       const vector<Index>&,
                                       const vector<Index>&,
                                       TrainingSession&,
                                       OptimizerData&);

    Loss::EvaluationResult evaluate_epoch(ForwardPropagation&,
                                          ThreadSafeQueue<Batch*>&,
                                          const vector<vector<Index>>&,
                                          const vector<Index>&,
                                          const vector<Index>&,
                                          const vector<Index>&,
                                          TrainingSession&);

    Loss* loss = nullptr;

    float training_loss_goal = 0.0f;

    Index maximum_validation_failures = numeric_limits<Index>::max();
    Index validation_period = 1;

    float gradient_clip_norm = 0.0f;

    bool restore_best = true;

    Index maximum_epochs = 10000;

    Index batch_size = 0;

    float maximum_time = 360000.0f;

    Index display_period = 10;

    bool display = true;

    bool shuffle_samples = true;

    int batch_pool_size_override = 0;

    int workers_number = 2;

    bool use_cuda_graph = false;

    string name;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
