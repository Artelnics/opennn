//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E R A T O R S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"
#include "enum_map.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class Json;
class JsonWriter;

struct Operator
{
    virtual ~Operator() = default;

    virtual vector<TensorSpec> parameter_specs() const { return {}; }
    virtual vector<TensorSpec> state_specs()     const { return {}; }

    virtual void link_parameters(span<const TensorView>) {}
    virtual void link_gradients (span<const TensorView>) {}
    virtual void link_states    (span<const TensorView>) {}

    virtual void set_parameters_random() {}
    virtual void set_parameters_glorot() {}

    virtual void forward_propagate(ForwardPropagation&, size_t, bool) noexcept {}
    virtual void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept {}

    virtual void to_JSON  (JsonWriter&) const {}
    virtual void from_JSON(const Json*)       {}
    virtual void load_state_from_JSON(const Json*) {}

    virtual void destroy_cuda() {}

    vector<size_t> input_slots = {0};
    vector<size_t> output_slots = {1};

    vector<size_t> input_delta_slots = {1};
    vector<size_t> output_delta_slots = {0};

    TensorView& get_input(ForwardPropagation& fp, size_t layer, size_t i = 0) const noexcept
    {
        return fp.views[layer][input_slots[i]][0];
    }

    vector<TensorView>& get_inputs(ForwardPropagation& fp, size_t layer, size_t i = 0) const noexcept
    {
        return fp.views[layer][input_slots[i]];
    }

    TensorView& get_output(ForwardPropagation& fp, size_t layer, size_t i = 0) const noexcept
    {
        return fp.views[layer][output_slots[i]][0];
    }

    TensorView& get_output_delta(BackPropagation& bp, size_t layer, size_t i = 0) const noexcept
    {
        return bp.delta_views[layer][output_delta_slots[i]];
    }

    TensorView& get_input_delta(BackPropagation& bp, size_t layer, size_t i = 0) const noexcept
    {
        return bp.delta_views[layer][input_delta_slots[i]];
    }
};

struct AddOp : Operator
{
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
    void check(const vector<TensorView>& inputs, const TensorView& output) const;
};

struct DropoutOp : Operator
{
    float rate = 0.0f;

    Buffer mask;

    vector<size_t> save_slots;

    bool active() const { return rate > 0.0f; }

    void set_rate(float new_rate);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

    void destroy_cuda() override;

    ~DropoutOp() override { destroy_cuda(); }

    DropoutOp() = default;
    DropoutOp(DropoutOp&&) noexcept = default;
    DropoutOp& operator=(DropoutOp&&) noexcept = default;
};

struct ActivationOp : Operator
{
    // Enum lives at namespace scope in tensor_utilities.h; this alias keeps
    // existing call sites (`ActivationOp::Function::Tanh`) working unchanged.
    using Function = ActivationFunction;

    static const EnumMap<Function>& map() { return activation_function_map(); }
    static Function from_string(const string& name) { return activation_function_from_string(name); }
    static const string& to_string(Function function) { return activation_function_to_string(function); }
    static cudnnActivationMode_t to_cudnn_mode(Function function);

    Function function = Function::Identity;

    cudnnActivationDescriptor_t descriptor = nullptr;

    // Backward override: when non-empty, back_propagate reads the activation's
    // output from this slot instead of output_slots[0]. Used when a downstream
    // operator (e.g. DropoutOp) overwrites the activation's output in place.
    vector<size_t> output_slots_backward;

    void set_function(Function new_function);
    void set_function(const string& name);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    void apply_delta(const TensorView& outputs, TensorView& delta) const;

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

    void destroy_cuda() override;

    ~ActivationOp() override { destroy_cuda(); }

    ActivationOp() = default;
    ActivationOp(const ActivationOp&) = delete;
    ActivationOp& operator=(const ActivationOp&) = delete;
};

struct CombinationOp : Operator
{
    Index input_features  = 0;
    Index output_features = 0;
    Type  weight_type     = Type::FP32;

    TensorView weights;
    TensorView bias;

    TensorView weight_gradient;
    TensorView bias_gradient;

    void set(Index new_input_features, Index new_output_features, Type new_weight_type = Type::FP32);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    void apply(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS);

    void apply_delta(const TensorView& output_delta,
                     const TensorView& input,
                     TensorView& input_delta,
                     bool accumulate_input_delta = false) const;

private:
};

struct CombinationReluOp : Operator
{
    CombinationOp combination;
    ActivationOp  activation;

    void set(Index input_features, Index output_features, Type weight_type = Type::FP32);

    vector<TensorSpec> parameter_specs() const override { return combination.parameter_specs(); }
    void link_parameters(span<const TensorView> views) override { combination.link_parameters(views); }
    void link_gradients (span<const TensorView> views) override { combination.link_gradients(views); }

    void set_parameters_random() override { combination.set_parameters_random(); }
    void set_parameters_glorot() override { combination.set_parameters_glorot(); }

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

// Elman-style recurrent op (simple RNN with tied weights over the time axis).
// Forward (per step t): z[t] = X[t]·W_in + h[t-1]·W_rec + b ;  h[t] = σ(z[t])
// Output of the op = h[T-1]. Hidden states and σ'(z) are stored across steps
// so the backward pass can do BPTT without recomputing the activations.
//
// Slot convention (set by hosting layer in configure_operators()):
//   input_slots  = {Input}                                       // 3D (batch, time, features)
//   output_slots = {Output, HiddenStates, ActivationDerivatives} // first is principal output (2D), the other two are 3D internal buffers
//
// Activation: Tanh by default (sane RNN choice). Sigmoid / Identity / ReLU also accepted.
struct RecurrentOp : Operator
{
    enum BackwardSlot
    {
        OutputDeltaSlot = 0,
        InputDeltaSlot,
        StepInputScratchSlot,
        StepPrevHScratchSlot,
        DeltaScratchSlot,
        NextCarryScratchSlot,
        StepInDeltaScratchSlot
    };

    Index input_features  = 0;
    Index time_steps      = 0;
    Index output_features = 0;
    Type  weight_type     = Type::FP32;
    ActivationOp::Function activation = ActivationOp::Function::Tanh;

    TensorView bias;
    TensorView input_weights;
    TensorView recurrent_weights;

    TensorView bias_gradient;
    TensorView input_weight_gradient;
    TensorView recurrent_weight_gradient;

    void set(Index new_input_features,
             Index new_time_steps,
             Index new_output_features,
             ActivationOp::Function = ActivationOp::Function::Tanh,
             Type = Type::FP32);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
    void apply(const TensorView& input,
               TensorView& hidden_states,
               TensorView& activation_derivatives,
               TensorView& output,
               bool is_training);
    void apply_gpu(const TensorView& input,
                   TensorView& hidden_states,
                   TensorView& activation_derivatives,
                   TensorView& output,
                   bool is_training);

    void apply_delta(const TensorView& input,
                     const TensorView& hidden_states,
                     const TensorView& activation_derivatives,
                     const TensorView& output_delta,
                     TensorView& input_delta) const;
    void apply_delta_gpu(const TensorView& input,
                         const TensorView& hidden_states,
                         const TensorView& activation_derivatives,
                         const TensorView& output_delta,
                         TensorView& input_delta,
                         TensorView& step_input_scratch,
                         TensorView& step_prev_h_scratch,
                         TensorView& delta_scratch,
                         TensorView& next_carry_scratch,
                         TensorView& step_in_delta_scratch) const;

    // Forward-only device-side scratch. Forward state must persist across
    // timesteps within a single forward pass (notably the swap between
    // step_hidden_buf and prev_hidden_buf carries h[t-1] into step t), so
    // these stay as per-instance buffers rather than slots in the framework
    // forward pool (which has no lifetime reuse).
    //
    // Backward-only scratch lives in the per-layer delta_views pool, declared
    // by Recurrent::get_backward_specs and consumed via the BackwardSlot
    // indices above.
    mutable Buffer step_input_buf      {Device::CUDA};   // (batch, in_features)
    mutable Buffer step_hidden_buf     {Device::CUDA};   // (batch, out_features)
    mutable Buffer prev_hidden_buf     {Device::CUDA};   // (batch, out_features)
    mutable Buffer step_derivs_buf     {Device::CUDA};   // (batch, out_features)
};

struct BatchNormOp : Operator
{
    Index features = 0;
    float momentum = 0.1f;

    TensorView gamma;
    TensorView beta;
    TensorView running_mean;
    TensorView running_variance;

    TensorView gamma_gradient;
    TensorView beta_gradient;

    bool active() const { return features > 0; }

    void set(Index new_features, float new_momentum = 0.1f);

    vector<TensorSpec> parameter_specs() const override;
    vector<TensorSpec> state_specs()     const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;
    void link_states    (span<const TensorView> views) override;

    void set_parameters_random() override { init_defaults(); }
    void set_parameters_glorot() override { init_defaults(); }

    void init_defaults();

    // Slot convention (set by hosting layer):
    //   input_slots  = {input}
    //   output_slots = {output, mean, inverse_variance}
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    void apply_delta(const TensorView& input,
                     const TensorView& mean,
                     const TensorView& inverse_variance,
                     TensorView& delta) const;

    void update_inference_cache();
    void invalidate_inference_cache() { inference_cache_dirty = true; }

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;
    void load_state_from_JSON(const Json* parent) override;

private:
    VectorR inference_scale;
    VectorR inference_shift;
    bool    inference_cache_dirty = true;

    mutable VectorR delta_scale_scratch;

    void apply_inference_cpu(const TensorView& input, TensorView& output);
    void apply_inference_gpu(const TensorView& input, TensorView& output);

    void apply_training_cpu (const TensorView& input,
                             TensorView& mean, TensorView& inverse_variance,
                             TensorView& output);
    void apply_training_gpu (const TensorView& input,
                             TensorView& mean, TensorView& inverse_variance,
                             TensorView& output);

    void apply_delta_cpu(const TensorView& input,
                         const TensorView& mean,
                         const TensorView& inverse_variance,
                         TensorView& delta) const;
    void apply_delta_gpu(const TensorView& input,
                         const TensorView& mean,
                         const TensorView& inverse_variance,
                         TensorView& delta) const;
};

struct ConvolutionOp : Operator
{
    Index input_height = 0;
    Index input_width = 0;

    Index kernels_number = 0;
    Index kernel_height = 0;
    Index kernel_width = 0;
    Index kernel_channels = 0;

    Index padding_height = 0;
    Index padding_width = 0;

    Type compute_dtype = Type::FP32;

    TensorView weights;
    TensorView bias;

    TensorView weight_gradient;
    TensorView bias_gradient;

#ifdef OPENNN_HAS_CUDA
    cudnnFilterDescriptor_t      kernel_descriptor      = nullptr;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

    cudnnConvolutionFwdAlgo_t       algorithm_forward = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdDataAlgo_t   algorithm_data    = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdFilterAlgo_t algorithm_filter  = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

    size_t cudnn_workspace_size_ = 0;

    // High-water-mark of the batch size for which the cuDNN plan
    // (algorithms + workspaces) is currently valid. Lazy-initialized on the
    // first apply_gpu and re-tuned only if a larger batch arrives (e.g. test
    // batch larger than training).
    Index planned_batch_size = 0;
#endif

    void set(Index input_h, Index input_w,
             Index kernels_n, Index kernel_h, Index kernel_w, Index kernel_c,
             Index row_stride, Index column_stride,
             Index padding_h, Index padding_w,
             Type compute_dtype);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void destroy_cuda() override;

    ~ConvolutionOp() override { destroy_cuda(); }

    ConvolutionOp() = default;
    ConvolutionOp(const ConvolutionOp&) = delete;
    ConvolutionOp& operator=(const ConvolutionOp&) = delete;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    void apply(const TensorView& input, TensorView& output, cudnnActivationDescriptor_t fused_activation = nullptr);

    void apply_delta(const TensorView& input,
                     const TensorView& output_delta,
                     TensorView& input_delta) const;

private:
    void apply_cpu(const TensorView& input, TensorView& output);
    void apply_gpu(const TensorView& input, TensorView& output, cudnnActivationDescriptor_t fused_activation = nullptr);

    void apply_delta_cpu(const TensorView& input, const TensorView& output_delta,
                         TensorView& input_delta) const;
    void apply_delta_gpu(const TensorView& input, const TensorView& output_delta,
                         TensorView& input_delta) const;

    void plan_convolution_algorithms(const TensorView& input, const TensorView& output);

    array<pair<Index, Index>, 4> nhwc_padding() const;
};

struct ConvolutionReluOp : Operator
{
    ConvolutionOp convolution;
    ActivationOp  activation;

    void set(Index input_h, Index input_w,
             Index kernels_n, Index kernel_h, Index kernel_w, Index kernel_c,
             Index row_stride, Index column_stride,
             Index padding_h, Index padding_w,
             Type compute_dtype);

    vector<TensorSpec> parameter_specs() const override { return convolution.parameter_specs(); }
    void link_parameters(span<const TensorView> views) override { convolution.link_parameters(views); }
    void link_gradients (span<const TensorView> views) override { convolution.link_gradients(views); }

    void set_parameters_random() override { convolution.set_parameters_random(); }
    void set_parameters_glorot() override { convolution.set_parameters_glorot(); }

    void destroy_cuda() override { convolution.destroy_cuda(); activation.destroy_cuda(); }
    ~ConvolutionReluOp() override { destroy_cuda(); }

    ConvolutionReluOp() = default;
    ConvolutionReluOp(const ConvolutionReluOp&) = delete;
    ConvolutionReluOp& operator=(const ConvolutionReluOp&) = delete;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

struct LayerNormOp : Operator
{
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    TensorView gamma;
    TensorView beta;

    TensorView gamma_gradient;
    TensorView beta_gradient;

    void set(Index sequence_length, Index embedding_dimension);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;

    void set_parameters_random() override { init_defaults(); }
    void set_parameters_glorot() override { init_defaults(); }

    void init_defaults();

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
};

struct MultiHeadProjectionOp : Operator
{
    CombinationOp combination;

    Index input_features = 0;
    Type  compute_dtype  = Type::FP32;

    // Which view inside views[input_slots[0]] to read. 0 for query path, 1 for
    // source path; clamped to size()-1 so self-attention (single input view)
    // works regardless.
    size_t input_view_index = 0;

    // Slot holding the shared transpose-scratch buffer.
    vector<size_t> scratch_slots;

    // Backward configuration. Self vs cross-attention is detected per-call from
    // forward_views[input_slots[0]].size() (1 = self, 2 = cross). The two pairs
    // below select destination slot and accumulate flag for each mode.
    vector<size_t> input_delta_slots_self;
    vector<size_t> input_delta_slots_cross;
    bool accumulate_input_delta_self  = false;
    bool accumulate_input_delta_cross = false;

    void set(Index input_features, Index heads_number, Index head_dimension, Type compute_dtype);

    vector<TensorSpec> parameter_specs() const override { return combination.parameter_specs(); }
    void link_parameters(span<const TensorView> views) override { combination.link_parameters(views); }
    void link_gradients (span<const TensorView> views) override { combination.link_gradients(views); }

    void set_parameters_random() override { combination.set_parameters_random(); }
    void set_parameters_glorot() override { combination.set_parameters_glorot(); }

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

struct AttentionOp : Operator
{
    Index heads_number = 0;
    Index head_dimension = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;
    bool  use_causal_mask = false;
    Type  compute_dtype = Type::FP32;

    bool use_sdpa = false;

    MatrixR causal_mask;

    DropoutOp dropout;

    void set(Index heads_number, Index head_dimension,
             Index query_sequence_length, Index source_sequence_length,
             bool use_causal_mask, Type compute_dtype);

    static bool sdpa_supported(Type dtype);

    void set_dropout_rate(float rate) { dropout.set_rate(rate); }

    vector<TensorSpec> forward_scratch_specs(Index batch_size) const;

    TensorSpec backward_scratch_spec(Index batch_size) const;

    size_t source_view_index = 1;  // 1 = source path; clamped to size()-1 for self-attention

    vector<size_t> scratch_slots;
    vector<size_t> attention_output_slots;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

    void destroy_cuda() override;

    AttentionOp();
    ~AttentionOp() override;
    AttentionOp(AttentionOp&&) noexcept;
    AttentionOp& operator=(AttentionOp&&) noexcept;
    AttentionOp(const AttentionOp&) = delete;
    AttentionOp& operator=(const AttentionOp&) = delete;

    struct SDPACache;

private:
    float scaling_factor() const;

    void apply_cpu(const TensorView& query,
                   const TensorView& key,
                   const TensorView& value,
                   const TensorView& source_input,
                   TensorView& attention_weights,
                   TensorView& attention_weights_dropped,
                   TensorView& output,
                   void* scratch,
                   bool is_training);

    void apply_gpu(const TensorView& query,
                   const TensorView& key,
                   const TensorView& value,
                   const TensorView& source_input,
                   TensorView& attention_weights,
                   TensorView& attention_weights_dropped,
                   TensorView& output,
                   void* scratch,
                   bool is_training);

    void apply_delta_cpu(const TensorView& query,
                         const TensorView& key,
                         const TensorView& value,
                         const TensorView& attention_output,
                         const TensorView& attention_weights,
                         const TensorView& attention_weights_dropped,
                         const TensorView& output_delta,
                         TensorView& attention_weight_delta,
                         TensorView& query_delta,
                         TensorView& key_delta,
                         TensorView& value_delta) const;

    void apply_delta_gpu(const TensorView& query,
                         const TensorView& key,
                         const TensorView& value,
                         const TensorView& attention_output,
                         const TensorView& attention_weights,
                         const TensorView& attention_weights_dropped,
                         const TensorView& output_delta,
                         TensorView& attention_weight_delta,
                         TensorView& query_delta,
                         TensorView& key_delta,
                         TensorView& value_delta) const;

    void apply_delta_gpu_unfused(const TensorView& query,
                                 const TensorView& key,
                                 const TensorView& value,
                                 const TensorView& attention_weights,
                                 const TensorView& attention_weights_dropped,
                                 const TensorView& output_delta,
                                 TensorView& attention_weight_delta,
                                 TensorView& query_delta,
                                 TensorView& key_delta,
                                 TensorView& value_delta) const;

    static bool get_contiguous_source_lengths(const TensorView& source_input,
                                              vector<Index>& lengths,
                                              bool& has_padding);
    static void softmax_rows_prefix(float* matrix, Index rows, Index cols, Index length);
    static Index infer_attention_prefix_length(const TensorView& attention_weights,
                                               Index batch_index);

    template<typename SoftmaxBwd>
    void apply_delta_unfused(const TensorView& query,
                              const TensorView& key,
                              const TensorView& value,
                              const TensorView& attention_weights,
                              const TensorView& attention_weights_dropped,
                              const TensorView& output_delta,
                              TensorView& attention_weight_delta,
                              TensorView& query_delta,
                              TensorView& key_delta,
                              TensorView& value_delta,
                              SoftmaxBwd&& softmax_bwd) const;

    mutable unique_ptr<SDPACache> sdpa_cache;

    uint64_t sdpa_dropout_seed   = 0x9E3779B97F4A7C15ULL;
    uint64_t sdpa_dropout_offset = 0;
    mutable uint64_t sdpa_last_used_offset = 0;
};

struct MergeOp : Operator
{
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index head_dimension = 0;
    Type  compute_dtype = Type::FP32;

    void set(Index heads_number, Index query_sequence_length, Index head_dimension, Type compute_dtype);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

struct PoolOp : Operator
{
    enum Method { Max, Average };

    Index input_height = 0;
    Index input_width = 0;
    Index input_channels = 0;

    Index pool_height = 1;
    Index pool_width = 1;
    Index row_stride = 1;
    Index column_stride = 1;
    Index padding_height = 0;
    Index padding_width = 0;

    Method method = Max;

#ifdef OPENNN_HAS_CUDA
    cudnnPoolingDescriptor_t pooling_descriptor = nullptr;
#endif

    void set(Index input_h, Index input_w, Index input_c,
             Index pool_h, Index pool_w,
             Index row_stride, Index column_stride,
             Index padding_h, Index padding_w,
             Method method);

    void destroy_cuda() override;

    ~PoolOp() override { destroy_cuda(); }

    PoolOp() = default;
    PoolOp(const PoolOp&) = delete;
    PoolOp& operator=(const PoolOp&) = delete;

    // Slot convention (set by Pooling layer in update_pool_operator):
    //   input_slots  = {Input}
    //   output_slots = {Output, MaximalIndices} for MaxPooling
    //   output_slots = {Output}                  for AveragePooling
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
    void apply_cpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
    void apply_gpu(const TensorView& input, TensorView& output);

    void apply_delta_cpu(const TensorView& output_delta,
                         const TensorView& maximal_indices,
                         TensorView& input_delta) const;
    void apply_delta_gpu(const TensorView& input,
                         const TensorView& output,
                         const TensorView& output_delta,
                         TensorView& input_delta) const;
};

struct Pool3dOp : Operator
{
    enum Method { Max, Average };
    Method method = Average;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

struct EmbeddingLookupOp : Operator
{
    Index vocabulary_size     = 0;
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    bool  scale_embedding         = false;
    bool  add_positional_encoding = false;

    TensorView weights;
    TensorView positional_encoding;

    TensorView weight_gradient;

    void set(Index new_vocabulary_size, Index new_sequence_length, Index new_embedding_dimension);

    vector<TensorSpec> parameter_specs() const override;
    vector<TensorSpec> state_specs()     const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;
    void link_states    (span<const TensorView> views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void init_positional_encoding();

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
};

struct FlatOp : Operator
{
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

struct BoundOp : Operator
{
    enum class Method { NoBounding, Bounding };

    Method method = Method::Bounding;

    TensorView lower;
    TensorView upper;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
};

struct ScaleOp : Operator
{
    float min_range = -1.0f;
    float max_range = 1.0f;

    TensorView minimums;
    TensorView maximums;
    TensorView means;
    TensorView standard_deviations;
    TensorView scalers;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
};

struct UnscaleOp : Operator
{
    float min_range = -1.0f;
    float max_range = 1.0f;

    TensorView minimums;
    TensorView maximums;
    TensorView means;
    TensorView standard_deviations;
    TensorView scalers;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
};

struct DetectionOp : Operator
{
    Index grid_size = 0;
    Index boxes_per_cell = 0;
    Index classes_number = 0;

    vector<array<float, 2>> anchors;

    void set(const Shape& input_shape, const vector<array<float, 2>>& new_anchors);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
    void apply(const TensorView& input, TensorView& output) const;
    void apply_delta(const TensorView& output, const TensorView& output_delta, TensorView& input_delta) const;
};

struct NonMaxSuppressionOp : Operator
{
    Index grid_size = 0;
    Index boxes_per_cell = 0;
    Index classes_number = 0;
    float confidence_threshold = 0.5f;
    float iou_threshold = 0.4f;

    void set(const Shape& input_shape,
             Index new_boxes_per_cell,
             float new_confidence_threshold,
             float new_iou_threshold);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;

private:
    void apply(const TensorView& input, TensorView& output) const;
};

struct LongShortTermMemoryOp : Operator
{
    enum ForwardSlot
    {
        InputSlot = 0,
        OutputSlot,
        ForgetGateSlot,
        InputGateSlot,
        CandidateGateSlot,
        OutputGateSlot,
        CellStateSlot,
        HiddenStateSlot,
        CellActivationSlot
    };

    enum BackwardSlot
    {
        OutputDeltaSlot = 0,
        InputDeltaSlot,
        HiddenDeltaScratchSlot,
        CellDeltaScratchSlot,
        ForgetDeltaScratchSlot,
        InputDeltaScratchSlot,
        CandidateDeltaScratchSlot,
        OutputDeltaScratchSlot
    };

    Index input_features  = 0;
    Index output_features = 0;
    Index time_steps      = 0;

    ActivationOp::Function activation_function = ActivationOp::Function::Tanh;
    ActivationOp::Function recurrent_activation_function = ActivationOp::Function::Sigmoid;

    TensorView forget_bias;
    TensorView input_bias;
    TensorView candidate_bias;
    TensorView output_bias;

    TensorView forget_weights;
    TensorView input_weights;
    TensorView candidate_weights;
    TensorView output_weights;

    TensorView forget_recurrent_weights;
    TensorView input_recurrent_weights;
    TensorView candidate_recurrent_weights;
    TensorView output_recurrent_weights;

    TensorView forget_bias_gradient;
    TensorView input_bias_gradient;
    TensorView candidate_bias_gradient;
    TensorView output_bias_gradient;

    TensorView forget_weight_gradient;
    TensorView input_weight_gradient;
    TensorView candidate_weight_gradient;
    TensorView output_weight_gradient;

    TensorView forget_recurrent_weight_gradient;
    TensorView input_recurrent_weight_gradient;
    TensorView candidate_recurrent_weight_gradient;
    TensorView output_recurrent_weight_gradient;

    void set(Index new_input_features,
             Index new_output_features,
             Index new_time_steps,
             ActivationOp::Function new_activation_function = ActivationOp::Function::Tanh,
             ActivationOp::Function new_recurrent_activation_function = ActivationOp::Function::Sigmoid);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
    void apply(const TensorView& input,
               TensorView& output,
               TensorView& forget_gate,
               TensorView& input_gate,
               TensorView& candidate_gate,
               TensorView& output_gate,
               TensorView& cell_state,
               TensorView& hidden_state,
               TensorView& cell_activation) const;

    void apply_delta(const TensorView& input,
                     const TensorView& output_delta,
                     TensorView& input_delta,
                     TensorView& hidden_delta_scratch,
                     TensorView& cell_delta_scratch,
                     TensorView& forget_delta_scratch,
                     TensorView& input_delta_scratch,
                     TensorView& candidate_delta_scratch,
                     TensorView& output_delta_scratch,
                     const TensorView& forget_gate,
                     const TensorView& input_gate,
                     const TensorView& candidate_gate,
                     const TensorView& output_gate,
                     const TensorView& cell_state,
                     const TensorView& hidden_state,
                     const TensorView& cell_activation) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
