//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E R A T O R S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_types.h"
#include "tensor_operations.h"
#include "enum_map.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class Json;
class JsonWriter;

template<typename Handle>
struct CudnnDescriptor
{
    Handle handle = nullptr;
#ifdef OPENNN_HAS_CUDA
    cudnnStatus_t (*deleter)(Handle) = nullptr;
#else
    void (*deleter)(Handle) = nullptr;
#endif

    CudnnDescriptor() = default;

    CudnnDescriptor(CudnnDescriptor&& other) noexcept
        : handle(other.handle), deleter(other.deleter)
    {
        other.handle = nullptr;
        other.deleter = nullptr;
    }

    CudnnDescriptor& operator=(CudnnDescriptor&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            handle = other.handle;
            deleter = other.deleter;
            other.handle = nullptr;
            other.deleter = nullptr;
        }
        return *this;
    }

    CudnnDescriptor(const CudnnDescriptor&) = delete;
    CudnnDescriptor& operator=(const CudnnDescriptor&) = delete;

    ~CudnnDescriptor() { reset(); }

    void reset()
    {
        if (handle && deleter) deleter(handle);
        handle = nullptr;
        deleter = nullptr;
    }

    operator Handle() const { return handle; }
    explicit operator bool() const { return handle != nullptr; }
};

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

    virtual void forward_propagate(ForwardPropagation&, size_t, bool) {}
    virtual void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const {}

    virtual void to_JSON  (JsonWriter&) const {}
    virtual void from_JSON(const Json*)       {}
    virtual void load_state_from_JSON(const Json*) {}

    virtual void destroy_cuda() {}

    vector<size_t> input_slots = {0};
    vector<size_t> output_slots = {1};

    vector<size_t> input_delta_slots = {1};
    vector<size_t> output_delta_slots = {0};

    TensorView& get_input(ForwardPropagation& fp, size_t layer, size_t slot_index = 0) const noexcept
    {
        const size_t slot = input_slots[slot_index];
        return slot == 0 ? fp.input_views[layer][0] : fp.forward_slots[layer][slot];
    }

    vector<TensorView>& get_inputs(ForwardPropagation& fp, size_t layer, size_t = 0) const noexcept
    {
        return fp.input_views[layer];
    }

    TensorView& get_output(ForwardPropagation& fp, size_t layer, size_t slot_index = 0) const noexcept
    {
        return fp.forward_slots[layer][output_slots[slot_index]];
    }

    TensorView& get_output_delta(BackPropagation& bp, size_t layer, size_t slot_index = 0) const noexcept
    {
        const size_t slot = output_delta_slots[slot_index];
        return slot == 0 ? bp.layer_output_deltas[layer] : bp.backward_slots[layer][slot];
    }

    TensorView& get_input_delta(BackPropagation& bp, size_t layer, size_t slot_index = 0) const noexcept
    {
        return bp.backward_slots[layer][input_delta_slots[slot_index]];
    }
};

struct AddOp : Operator
{
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

private:
    void check(const vector<TensorView>& inputs, const TensorView& output) const;
};

struct UpsampleOp : Operator
{
    Index input_height = 0;
    Index input_width = 0;
    Index channels = 0;
    Index scale_factor = 2;

    void set(Index in_h, Index in_w, Index ch, Index scale);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

private:
    void apply(const TensorView& input, TensorView& output) const;
    void apply_delta(const TensorView& output_delta, TensorView& input_delta) const;
};

struct ConcatenationOp : Operator
{
    Index height = 0;
    Index width = 0;
    vector<Index> input_channels;

    void set(Index h, Index w, const vector<Index>& per_input_channels);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;
};

struct DropoutOp : Operator
{
    float rate = 0.0f;

    Buffer mask;

    vector<size_t> save_slots;

    bool active() const { return rate > 0.0f; }

    void set_rate(float new_rate);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

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
    using Function = ActivationFunction;

    static const EnumMap<Function>& map() { return activation_function_map(); }
    static Function from_string(const string& name) { return activation_function_from_string(name); }
    static const string& to_string(Function function) { return activation_function_to_string(function); }
    static cudnnActivationMode_t to_cudnn_mode(Function function);

    Function function = Function::Identity;

    cudnnActivationDescriptor_t descriptor = nullptr;

    vector<size_t> output_slots_backward;

    bool forward_fused = false;

    void set_function(Function new_function);
    void set_function(const string& name);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

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

    bool  fuse_relu       = false;

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

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

    void apply(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS);

    void apply_delta(const TensorView& output_delta,
                     const TensorView& input,
                     TensorView& input_delta,
                     bool accumulate_input_delta = false) const;
};

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

    bool return_sequences = false;

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

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

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
    mutable Buffer step_input_buf      {Device::CUDA};
    mutable Buffer step_hidden_buf     {Device::CUDA};
    mutable Buffer prev_hidden_buf     {Device::CUDA};
    mutable Buffer step_derivs_buf     {Device::CUDA};
    mutable Buffer step_seq_delta_buf  {Device::CUDA};
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

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

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
    Index row_stride = 1;
    Index column_stride = 1;

    Index padding_height = 0;
    Index padding_width = 0;

    Type compute_dtype = Type::FP32;

    TensorView weights;
    TensorView bias;

    TensorView weight_gradient;
    TensorView bias_gradient;

    cudnnActivationDescriptor_t fused_activation = nullptr;

    cudnnFilterDescriptor_t      kernel_descriptor      = nullptr;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

    cudnnConvolutionFwdAlgo_t       algorithm_forward = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdDataAlgo_t   algorithm_data    = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdFilterAlgo_t algorithm_filter  = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

    size_t cudnn_workspace_size_ = 0;

    Index planned_batch_size = 0;

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

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

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

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;
};

struct MultiHeadProjectionOp : Operator
{
    CombinationOp combination;

    Index input_features = 0;
    Type  compute_dtype  = Type::FP32;

    size_t input_view_index = 0;

    vector<size_t> scratch_slots;

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

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;
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

    static bool sdpa_supported(Type dtype, Device device);

    void set_dropout_rate(float rate) { dropout.set_rate(rate); }

    vector<TensorSpec> forward_scratch_specs(Index batch_size) const;

    TensorSpec backward_scratch_spec(Index batch_size) const;

    size_t source_view_index = 1;

    vector<size_t> scratch_slots;
    vector<size_t> attention_output_slots;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

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

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;
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

    cudnnPoolingDescriptor_t pooling_descriptor = nullptr;

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

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

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

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;
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

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;
};

struct FlatOp : Operator
{
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;
};

struct BoundOp : Operator
{
    enum class Method { NoBounding, Bounding };

    Method method = Method::Bounding;

    TensorView lower;
    TensorView upper;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
};

struct ScaleOp : Operator
{
    bool invert = false;

    float min_range = -1.0f;
    float max_range = 1.0f;

    TensorView minimums;
    TensorView maximums;
    TensorView means;
    TensorView standard_deviations;
    TensorView scalers;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
};

struct DetectionOp : Operator
{
    enum class ClassActivation { Softmax, Sigmoid };

    Index grid_size = 0;
    Index grid_width = 0;
    Index boxes_per_cell = 0;
    Index classes_number = 0;
    ClassActivation class_activation = ClassActivation::Softmax;

    vector<array<float, 2>> anchors;

    void set(const Shape& input_shape, const vector<array<float, 2>>& new_anchors);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

private:
    void apply(const TensorView& input, TensorView& output) const;
    void apply_delta(const TensorView& output, const TensorView& output_delta, TensorView& input_delta) const;
};

struct NonMaxSuppressionOp : Operator
{
    Index grid_size = 0;
    Index grid_width = 0;
    Index boxes_per_cell = 0;
    Index classes_number = 0;
    float confidence_threshold = 0.5f;
    float iou_threshold = 0.4f;

    void set(const Shape& input_shape,
             Index new_boxes_per_cell,
             float new_confidence_threshold,
             float new_iou_threshold);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;

private:
    void apply(const TensorView& input, TensorView& output) const;
};

struct LongShortTermMemoryOp : Operator
{
    enum ForwardSlot
    {
        InputSlot = 0,
        ForgetGateSlot,
        InputGateSlot,
        CandidateGateSlot,
        OutputGateSlot,
        CellStateSlot,
        HiddenStateSlot,
        CellActivationSlot,
        OutputSlot
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

    bool return_sequences = false;

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

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

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

    void apply_gpu(const TensorView& input,
                   TensorView& output,
                   bool return_seq) const;

    void apply_delta_gpu(const TensorView& input,
                         const TensorView& output_delta,
                         TensorView& input_delta,
                         bool return_seq) const;

    void ensure_cudnn_setup_(Index batch_size) const;
    void pack_weights_to_cudnn_() const;
    void unpack_gradients_from_cudnn_() const;

    mutable Buffer weight_space_buf    {Device::CUDA};
    mutable Buffer dweight_space_buf   {Device::CUDA};
    mutable Buffer workspace_buf       {Device::CUDA};
    mutable Buffer reserve_space_buf   {Device::CUDA};
    mutable Buffer y_buf               {Device::CUDA};   // (B, T, H) rank-3 y from cuDNN
    mutable Buffer dy_buf              {Device::CUDA};   // (B, T, H) rank-3 dy for cuDNN
    mutable Buffer dx_scratch_buf      {Device::CUDA};   // (B, T, F) dx sink when input_delta is unused
    mutable Buffer seq_lengths_host_buf{Device::CPU};    // int32[batch], all equal to T
    mutable Buffer seq_lengths_dev_buf {Device::CUDA};

    mutable CudnnDescriptor<cudnnRNNDescriptor_t>     rnn_desc;
    mutable CudnnDescriptor<cudnnRNNDataDescriptor_t> x_data_desc;
    mutable CudnnDescriptor<cudnnRNNDataDescriptor_t> y_data_desc;
    mutable CudnnDescriptor<cudnnTensorDescriptor_t>  h_desc;
    mutable CudnnDescriptor<cudnnTensorDescriptor_t>  c_desc;
    mutable CudnnDescriptor<cudnnDropoutDescriptor_t> dropout_desc;
    mutable Buffer dropout_states_buf{Device::CUDA};

    mutable Index cached_batch_size = -1;
    mutable Index cached_time_steps = -1;
    mutable Index cached_input_features  = -1;
    mutable Index cached_output_features = -1;

    mutable vector<float> grad_tls_buf_;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
