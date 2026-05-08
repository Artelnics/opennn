//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file optimizer.h
 * @brief Declares the Optimizer abstract base class and the supporting
 *        EpochStats / OptimizerData / TrainingResults structures.
 *
 * Concrete optimizers (StochasticGradientDescent, AdaptiveMomentEstimation,
 * QuasiNewtonMethod, LevenbergMarquardtAlgorithm, ...) extend Optimizer to
 * implement train(), reusing the base-class infrastructure for batch
 * prefetching, stopping conditions and JSON I/O.
 */

#pragma once

#include <functional>
#include "json.h"
#include "tensor_utilities.h"
#include "thread_safe_queue.h"

namespace opennn
{

class Loss;
struct Batch;
struct Buffer;
struct ForwardPropagation;
struct BackPropagation;

struct TrainingResults;

/**
 * @struct EpochStats
 * @brief Aggregate metrics produced for a single training or evaluation epoch.
 */
struct EpochStats
{
    /** @brief Mean loss over the epoch. */
    float error = 0.0f;
    /** @brief Mean classification accuracy over the epoch (0 if not classification). */
    float accuracy = 0.0f;
};

/**
 * @class Optimizer
 * @brief Abstract base class for every training algorithm in OpenNN.
 *
 * Holds the loss function pointer, the training stopping criteria
 * (epoch budget, time budget, loss goal, validation failure budget) and
 * the device-side machinery used to overlap data transfer with compute
 * (CUDA streams, batch prefetching queues).
 *
 * Subclasses implement train(); the base class provides the helpers
 * train_epoch() and evaluate_epoch() that drive a complete pass over the
 * dataset.
 */
class Optimizer
{

public:

    /**
     * @brief Constructs an optimizer bound to a loss function.
     * @param loss Loss to optimize; may be nullptr if set later via set_loss().
     */
    Optimizer(Loss* loss = nullptr);
    /** @brief Virtual destructor. */
    virtual ~Optimizer() = default;

    /**
     * @enum StoppingCondition
     * @brief Reasons that can terminate a training run.
     */
    enum class StoppingCondition{None,                            ///< No stopping condition was hit (still running).
                                 MinimumLossDecrease,             ///< Loss decrease between epochs fell below the threshold.
                                 LossGoal,                        ///< Training loss reached the configured goal.
                                 MaximumSelectionErrorIncreases,  ///< Validation error increased too many times.
                                 MaximumEpochsNumber,             ///< Configured epoch budget exhausted.
                                 MaximumTime};                    ///< Configured time budget exhausted.

    /** @brief Read-only access to the loss being optimized. */
    const Loss* get_loss() const { return loss; }

    /** @brief Whether progress should be printed to stdout during training. */
    bool get_display() const { return display; }

    /**
     * @brief Re-initializes the optimizer by setting its loss pointer.
     * @param new_loss Loss to optimize.
     */
    void set(Loss* new_loss) { loss = new_loss; }

    /**
     * @brief Updates the loss pointer; subclasses may override to refresh
     *        cached state derived from the loss.
     * @param new_loss Loss to optimize.
     */
    virtual void set_loss(Loss* new_loss) { loss = new_loss; }

    /**
     * @brief Toggles per-epoch progress printing.
     * @param new_display True to print progress to stdout.
     */
    virtual void set_display(bool new_display) { display = new_display; }

    /**
     * @brief Sets how often progress is printed.
     * @param new_display_period Number of epochs between progress prints.
     */
    void set_display_period(const Index new_display_period) { display_period = new_display_period; }

    /**
     * @brief Sets the maximum number of epochs.
     * @param new_maximum_epochs Epoch budget; training stops after that many epochs.
     */
    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    /**
     * @brief Sets the maximum wall-clock training time.
     * @param new_maximum_time Time budget in seconds.
     */
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

    /**
     * @brief Sets the training-loss goal.
     * @param new_loss_goal Training stops when the loss reaches this value.
     */
    void set_loss_goal(const float new_loss_goal) { training_loss_goal = new_loss_goal; }
    /**
     * @brief Sets the maximum number of consecutive validation-error increases tolerated.
     * @param new_maximum_validation_failures Failure budget for early stopping.
     */
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    /**
     * @brief Runs the optimization to completion.
     * @return Per-epoch error history and the stopping condition that fired.
     */
    virtual TrainingResults train() = 0;

    /** @brief Canonical name of the optimizer (set by subclasses). */
    const string& get_name() const { return name; }

    /** @brief Prints a human-readable summary of the optimizer to stdout. */
    virtual void print() const {}

    /**
     * @brief Loads optimizer hyperparameters from a parsed JSON document.
     */
    virtual void from_JSON(const JsonDocument&);

    /**
     * @brief Writes optimizer hyperparameters to a streaming JSON writer.
     */
    virtual void to_JSON(JsonWriter&) const;

    /**
     * @brief Saves the optimizer state to a file.
     *
     * Receives the destination path.
     */
    void save(const filesystem::path&) const;
    /**
     * @brief Loads the optimizer state from a file.
     *
     * Receives the source path.
     */
    void load(const filesystem::path&);

    /**
     * @brief Computes the elapsed wall-clock time since a reference instant.
     * @param beginning_time Reference instant produced by std::time().
     * @return Elapsed time in seconds.
     */
    static float get_elapsed_time(const time_t& beginning_time);

protected:

    /** @brief Subclass hook to refresh layer name caches after a loss change. */
    void set_names();
    /** @brief Subclass hook to install the dataset-derived input scalers. */
    void set_scaling();
    /** @brief Subclass hook to install the dataset-derived output unscalers. */
    void set_unscaling();

    /**
     * @brief Evaluates every stopping criterion and updates the result accordingly.
     * @param results Mutable training results being built incrementally.
     * @param epoch Current epoch index.
     * @param elapsed_time Elapsed time in seconds since training started.
     * @param training_error Current training error.
     * @param validation_failures Consecutive validation-error increases observed.
     * @return True if training should stop.
     */
    bool check_stopping_condition(TrainingResults& results, Index epoch, float elapsed_time,
                                   float training_error, Index validation_failures) const;

    /**
     * @brief Writes the common Optimizer fields to JSON.
     *
     * Receives the streaming JSON writer; called by subclasses' to_JSON().
     */
    void write_common_xml(JsonWriter&) const;
    /**
     * @brief Reads the common Optimizer fields from JSON.
     *
     * Receives the JSON node; called by subclasses' from_JSON().
     */
    void read_common_xml(const Json*);

    /** @brief Allocates the CUDA stream and events used for batch prefetching. */
    void setup_device_training();
    /** @brief Releases the CUDA stream and events allocated by setup_device_training(). */
    void teardown_device_training();

    /**
     * @brief Asynchronously prefetches the next training batch into a slot.
     * @param batch Batch buffer that will be filled.
     * @param sample_count Number of samples in this batch.
     * @param slot Double-buffer slot index (0 or 1).
     */
    void prefetch_batch(Batch& batch, Index sample_count, int slot);

    /**
     * @brief Waits for the prefetch into a given slot to finish.
     * @param slot Double-buffer slot index (0 or 1).
     */
    void wait_prefetch(int slot);

    /** @brief Synchronizes the device on the optimizer's CUDA stream. */
    void sync_device();

    /**
     * @brief In-place gradient norm clipping.
     * @param gradient Gradient buffer to clip.
     * @param max_norm Maximum allowed L2 norm.
     */
    static void clip_gradient_norm(Buffer& gradient, float max_norm);

    /**
     * @brief Whether the current epoch should print progress.
     * @param epoch Current epoch index.
     * @return True if display is enabled and the epoch is on the period.
     */
    bool should_display(Index epoch) const { return display && epoch % display_period == 0; }

    /**
     * @brief Runs a single training epoch over all batches.
     * @param is_classification Whether the model produces class probabilities.
     * @param forward_propagation Forward buffer reused across batches.
     * @param back_propagation Back buffer reused across batches.
     * @param empty_queue Queue feeding the prefetcher with empty batches.
     * @param ready_queue Queue receiving batches that are ready to train on.
     * @param batches Per-batch sample index lists.
     * @param input_feature_indices Indices of dataset input columns.
     * @param decoder_feature_indices Indices of dataset decoder columns
     *                                (empty when the model has no decoder).
     * @param target_feature_indices Indices of dataset target columns.
     * @param update Subclass-supplied parameter update functor.
     * @return Mean error and accuracy over the epoch.
     */
    EpochStats train_epoch(bool is_classification,
                           ForwardPropagation& forward_propagation,
                           BackPropagation& back_propagation,
                           ThreadSafeQueue<Batch*>& empty_queue,
                           ThreadSafeQueue<Batch*>& ready_queue,
                           const vector<vector<Index>>& batches,
                           const vector<Index>& input_feature_indices,
                           const vector<Index>& decoder_feature_indices,
                           const vector<Index>& target_feature_indices,
                           const std::function<void(BackPropagation&)>& update);

    /**
     * @brief Runs a single evaluation pass over all batches without updating
     *        parameters.
     * @param is_classification Whether the model produces class probabilities.
     * @param forward_propagation Forward buffer reused across batches.
     * @param empty_queue Queue feeding the prefetcher with empty batches.
     * @param ready_queue Queue receiving batches that are ready to evaluate.
     * @param batches Per-batch sample index lists.
     * @param input_feature_indices Indices of dataset input columns.
     * @param decoder_feature_indices Indices of dataset decoder columns.
     * @param target_feature_indices Indices of dataset target columns.
     * @return Mean error and accuracy over the epoch.
     */
    EpochStats evaluate_epoch(bool is_classification,
                              ForwardPropagation& forward_propagation,
                              ThreadSafeQueue<Batch*>& empty_queue,
                              ThreadSafeQueue<Batch*>& ready_queue,
                              const vector<vector<Index>>& batches,
                              const vector<Index>& input_feature_indices,
                              const vector<Index>& decoder_feature_indices,
                              const vector<Index>& target_feature_indices);

    /** @brief Loss being optimized; not owned. */
    Loss* loss = nullptr;

    /** @brief Training stops when the training loss reaches this value. */
    float training_loss_goal = 0.0f;

    /** @brief Maximum number of consecutive validation-error increases tolerated. */
    Index maximum_validation_failures = numeric_limits<Index>::max();

    /** @brief Maximum number of training epochs. */
    Index maximum_epochs = 10000;

    /** @brief Maximum wall-clock training time in seconds. */
    float maximum_time = 360000.0f;

    /** @brief Number of epochs between progress prints. */
    Index display_period = 10;

    /** @brief Whether progress should be printed to stdout during training. */
    bool display = true;

    /** @brief Canonical name of the optimizer (set by subclasses). */
    string name;

    /** @brief CUDA stream used to prefetch batches into device memory. */
    cudaStream_t memory_stream = nullptr;
    /** @brief CUDA events signaling when each prefetched batch is ready. */
    cudaEvent_t batch_ready_event[2] = {nullptr, nullptr};
};

/**
 * @struct OptimizerData
 * @brief Per-optimizer scratch state shared across iterations.
 *
 * Holds an owning data buffer plus a vector of TensorViews into it (slot
 * shapes are decided by the subclass), and three small fields used by
 * line-search-based methods.
 */
struct OptimizerData
{
    /** @brief Default constructor; data buffer left empty. */
    OptimizerData() = default;
    /** @brief Virtual destructor. */
    virtual ~OptimizerData() = default;

    /** @brief Prints a human-readable summary of the scratch state. */
    virtual void print() const;

    /**
     * @brief Allocates the scratch buffer and slices it into views.
     * @param slot_shapes Per-slot tensor shapes.
     * @param device CPU or GPU memory placement.
     */
    void set(const vector<Shape>& slot_shapes, Device device = Device::CPU);

    /** @brief Owning storage for the per-slot scratch tensors. */
    Buffer data;
    /** @brief Per-slot non-owning views into @ref data. */
    vector<TensorView> views;

    /** @brief Candidate parameter vector used by line searches. */
    VectorR potential_parameters;
    /** @brief Current search direction (e.g. quasi-Newton step). */
    VectorR training_direction;
    /** @brief Initial learning rate at the start of a line search. */
    float initial_learning_rate = 0.0f;
    /** @brief Iteration counter used by adaptive optimizers. */
    Index iteration = 0;
};

/**
 * @struct TrainingResults
 * @brief Per-epoch error history and final summary produced by Optimizer::train().
 */
struct TrainingResults
{
    /**
     * @brief Constructs a TrainingResults pre-sized for an expected epoch count.
     * @param expected_epochs Initial capacity for the error history vectors.
     */
    TrainingResults(const Index expected_epochs = 0);
    /** @brief Virtual destructor. */
    virtual ~TrainingResults() = default;

    /**
     * @brief Returns the canonical string name of the stopping condition.
     * @return Name (e.g. "MaximumEpochsNumber").
     */
    string write_stopping_condition() const;

    /** @brief Final training error (last entry of training_error_history). */
    float get_training_error() const;

    /** @brief Final validation error (last entry of validation_error_history). */
    float get_validation_error() const;

    /** @brief Number of epochs effectively run. */
    Index get_epochs_number() const;

    /**
     * @brief Saves the training results (history and final summary) to a file.
     *
     * Receives the destination path.
     */
    void save(const filesystem::path&) const;

    /**
     * @brief Prints a human-readable summary to stdout.
     * @param message Optional prefix message.
     */
    void print(const string& message = string()) const;

    /** @brief Stopping condition that ended the training run. */
    Optimizer::StoppingCondition stopping_condition = Optimizer::StoppingCondition::None;

    /**
     * @brief Returns a 2D string table summarizing the training run.
     * @param decimals Number of decimals used when formatting floats.
     * @return Tensor of (rows, columns) strings ready for pretty-printing.
     */
    Tensor<string, 2> write_override_results(const Index decimals = 3) const;

    /**
     * @brief Resizes the training error history.
     *
     * Receives the new size.
     */
    void resize_training_error_history(const Index);

    /**
     * @brief Resizes the validation error history.
     *
     * Receives the new size.
     */
    void resize_validation_error_history(const Index);

    /** @brief Per-epoch training error. */
    VectorR training_error_history;

    /** @brief Per-epoch validation error. */
    VectorR validation_error_history;

    /** @brief Total elapsed wall-clock time, formatted as "hh:mm:ss". */
    string elapsed_time;

    /** @brief Final loss value. */
    float loss = NAN;

    /** @brief Number of consecutive validation-error increases at end of training. */
    Index validation_failures = 0;

    /** @brief Loss decrease observed in the last epoch. */
    float loss_decrease = 0.0f;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
