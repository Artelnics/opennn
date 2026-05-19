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

#ifdef OPENNN_HAS_CUDA
struct LtMatmulPlan;
#endif

/// @brief Base class for compute building blocks composed by layers (matmul, activation, dropout, etc.).
struct Operator
{
    virtual ~Operator() = default;

    /// @brief Returns the tensor specs of trainable parameters owned by this operator.
    virtual vector<TensorSpec> parameter_specs() const { return {}; }

    /// @brief Returns the tensor specs of persistent state owned by this operator.
    virtual vector<TensorSpec> state_specs()     const { return {}; }

    /// @brief Binds parameter views provided by the hosting layer.
    virtual void link_parameters(span<const TensorView>) {}

    /// @brief Binds gradient views provided by the hosting layer.
    virtual void link_gradients (span<const TensorView>) {}

    /// @brief Binds state views provided by the hosting layer.
    virtual void link_states    (span<const TensorView>) {}

    /// @brief Initializes parameters with random values.
    virtual void set_parameters_random() {}

    /// @brief Initializes parameters using Glorot (Xavier) initialization.
    virtual void set_parameters_glorot() {}

    /// @brief Runs the operator's forward computation.
    /// @param fp Forward propagation workspace.
    /// @param layer Index of the hosting layer in the workspace.
    /// @param is_training If true, enables training-only behavior (e.g. dropout sampling).
    virtual void forward_propagate(ForwardPropagation&, size_t, bool) noexcept {}

    /// @brief Runs the operator's backward computation, accumulating into gradient/delta buffers.
    /// @param fp Forward propagation workspace (read-only).
    /// @param bp Back propagation workspace receiving gradients and deltas.
    /// @param layer Index of the hosting layer in the workspace.
    virtual void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept {}

    /// @brief Serializes the operator configuration to a JSON writer.
    virtual void to_JSON  (JsonWriter&) const {}

    /// @brief Restores the operator configuration from a JSON node.
    virtual void from_JSON(const Json*)       {}

    /// @brief Restores persistent state (e.g. running statistics) from a JSON node.
    virtual void load_state_from_JSON(const Json*) {}

    /// @brief Releases CUDA resources owned by the operator; called from destructors.
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

/// @brief Element-wise sum of several input tensors (used by residual connections).
struct AddOp : Operator
{
    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
    void check(const vector<TensorView>& inputs, const TensorView& output) const;
};

/// @brief Inverted dropout: at training time zeros activations with probability rate and rescales survivors.
struct DropoutOp : Operator
{
    float rate = 0.0f;

    Buffer mask;

    vector<size_t> save_slots;

    /// @brief Returns true when the dropout rate is non-zero.
    bool active() const { return rate > 0.0f; }

    /// @brief Sets the drop probability (0 disables dropout).
    void set_rate(float new_rate);

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    /// @brief CPU forward implementation; samples the mask and rescales survivors in place.
    void apply_cpu(TensorView& output);

    /// @brief GPU forward implementation; samples the mask and rescales survivors in place.
    void apply_gpu(TensorView& output);

    /// @brief Applies the cached mask to a gradient tensor during the backward pass.
    void apply_delta(TensorView& delta) const;

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

    void destroy_cuda() override;

    ~DropoutOp() override { destroy_cuda(); }

    DropoutOp() = default;
    DropoutOp(DropoutOp&&) noexcept = default;
    DropoutOp& operator=(DropoutOp&&) noexcept = default;

private:
    void apply_delta_cpu(TensorView& delta) const;
    void apply_delta_gpu(TensorView& delta) const;

    void ensure_mask(Index n);
};

/// @brief Element-wise non-linear activation (Identity, Sigmoid, Tanh, ReLU, Softmax).
struct ActivationOp : Operator
{
    /// @brief Supported activation functions.
    enum class Function { Identity, Sigmoid, Tanh, ReLU, Softmax };

    /// @brief Returns the bidirectional mapping between Function values and their string names.
    static const EnumMap<Function>& map();

    /// @brief Returns the Function corresponding to a string name.
    static Function from_string(const string& name);

    /// @brief Returns the string name of a Function.
    static const string& to_string(Function function);

    /// @brief Returns the cuDNN activation mode corresponding to a Function.
    static cudnnActivationMode_t to_cudnn_mode(Function function);

    Function function = Function::Identity;

    cudnnActivationDescriptor_t descriptor = nullptr;

    // Backward override: when non-empty, back_propagate reads the activation's
    // output from this slot instead of output_slots[0]. Used when a downstream
    // operator (e.g. DropoutOp) overwrites the activation's output in place.
    vector<size_t> output_slots_backward;

    /// @brief Selects the activation function and configures cuDNN descriptors.
    void set_function(Function new_function);

    /// @brief Selects the activation function by name (delegates to set_function(Function)).
    void set_function(const string& name);

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    /// @brief CPU forward implementation; applies the activation in place on @p output.
    void apply_cpu(TensorView& output);

    /// @brief GPU forward implementation; applies the activation in place on @p output.
    void apply_gpu(TensorView& output);

    /// @brief Multiplies @p delta by the derivative of the activation evaluated at @p outputs.
    void apply_delta(const TensorView& outputs, TensorView& delta) const;

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

    void destroy_cuda() override;

    ~ActivationOp() override { destroy_cuda(); }

    ActivationOp() = default;
    ActivationOp(const ActivationOp&) = delete;
    ActivationOp& operator=(const ActivationOp&) = delete;

private:
    void apply_delta_cpu(const TensorView& outputs, TensorView& delta) const;
    void apply_delta_gpu(const TensorView& outputs, TensorView& delta) const;
};

/// @brief Affine combination output = input * weights + bias (the dense matmul building block).
struct CombinationOp : Operator
{
    Index input_features  = 0;
    Index output_features = 0;
    Type  weight_type     = Type::FP32;

    TensorView weights;
    TensorView bias;

    TensorView weight_gradient;
    TensorView bias_gradient;

    /// @brief Configures input/output dimensions and the weight storage dtype.
    /// @param new_input_features Number of input features.
    /// @param new_output_features Number of output features.
    /// @param new_weight_type Storage dtype of the weight matrix.
    void set(Index new_input_features, Index new_output_features, Type new_weight_type = Type::FP32);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    /// @brief Computes output = input * weights + bias with an optional fused epilogue (ReLU, bias, etc.).
    void apply(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS);

    /// @brief Computes input_delta from output_delta and updates weight/bias gradients.
    /// @param output_delta Gradient w.r.t. the operator's output.
    /// @param input Forward-pass input (needed for the weight gradient).
    /// @param input_delta Output: gradient w.r.t. the operator's input.
    /// @param accumulate_input_delta If true, accumulates into @p input_delta instead of overwriting.
    void apply_delta(const TensorView& output_delta,
                     const TensorView& input,
                     TensorView& input_delta,
                     bool accumulate_input_delta = false) const;

private:
    void apply_cpu(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue);
    void apply_gpu(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue);

    void apply_delta_cpu(const TensorView& output_delta, const TensorView& input,
                         TensorView& input_delta, bool accumulate_input_delta) const;
    void apply_delta_gpu(const TensorView& output_delta, const TensorView& input,
                         TensorView& input_delta, bool accumulate_input_delta) const;
};

/// @brief Fused affine + ReLU activation (uses cuBLASLt epilogue on GPU when available).
struct CombinationReluOp : Operator
{
    CombinationOp combination;
    ActivationOp  activation;

    /// @brief Configures the underlying CombinationOp; ReLU is fixed.
    void set(Index input_features, Index output_features, Type weight_type = Type::FP32);

    vector<TensorSpec> parameter_specs() const override { return combination.parameter_specs(); }
    void link_parameters(span<const TensorView> views) override { combination.link_parameters(views); }
    void link_gradients (span<const TensorView> views) override { combination.link_gradients(views); }

    void set_parameters_random() override { combination.set_parameters_random(); }
    void set_parameters_glorot() override { combination.set_parameters_glorot(); }

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

/// @brief Batch normalization with learnable scale/shift and running statistics for inference.
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

    /// @brief Returns true when the operator has been configured (features > 0).
    bool active() const { return features > 0; }

    /// @brief Configures the per-feature normalization.
    /// @param new_features Number of channels / features to normalize independently.
    /// @param new_momentum Exponential-moving-average momentum for running statistics.
    void set(Index new_features, float new_momentum = 0.1f);

    vector<TensorSpec> parameter_specs() const override;
    vector<TensorSpec> state_specs()     const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;
    void link_states    (span<const TensorView> views) override;

    void set_parameters_random() override { init_defaults(); }
    void set_parameters_glorot() override { init_defaults(); }

    /// @brief Resets gamma to one, beta to zero, and running stats to identity values.
    void init_defaults();

    // Slot convention (set by hosting layer):
    //   input_slots  = {input}
    //   output_slots = {output, mean, inverse_variance}
    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    /// @brief Computes the input gradient given the cached normalization statistics from forward.
    /// @param input Forward-pass input tensor.
    /// @param mean Per-feature mean cached during forward.
    /// @param inverse_variance Per-feature inverse standard deviation cached during forward.
    /// @param delta In/out gradient: output_delta on input, input_delta on output.
    void apply_delta(const TensorView& input,
                     const TensorView& mean,
                     const TensorView& inverse_variance,
                     TensorView& delta) const;

    /// @brief Rebuilds the fused scale/shift cache used by the inference path from running stats.
    void update_inference_cache();

    /// @brief Marks the inference cache as stale so it is rebuilt on the next inference call.
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

/// @brief 2D convolution operator (NHWC layout) backed by Eigen on CPU and cuDNN on GPU.
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

    /// @brief Configures the convolution geometry and compute precision.
    /// @param input_h Input height in pixels.
    /// @param input_w Input width in pixels.
    /// @param kernels_n Number of output channels (kernels).
    /// @param kernel_h Kernel height.
    /// @param kernel_w Kernel width.
    /// @param kernel_c Kernel channel count (matches input channels).
    /// @param row_stride Vertical stride.
    /// @param column_stride Horizontal stride.
    /// @param padding_h Vertical zero-padding on each side.
    /// @param padding_w Horizontal zero-padding on each side.
    /// @param compute_dtype Dtype used for the matmul computation.
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

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    /// @brief CPU forward path; im2col + GEMM.
    void apply_cpu(const TensorView& input, TensorView& output);

    /// @brief GPU forward path; runs cuDNN convolution with an optional fused activation.
    void apply_gpu(const TensorView& input, TensorView& output, cudnnActivationDescriptor_t fused_activation = nullptr);

    /// @brief Computes input_delta from output_delta and updates weight/bias gradients.
    void apply_delta(const TensorView& input,
                     const TensorView& output_delta,
                     TensorView& input_delta) const;

private:

    void apply_delta_cpu(const TensorView& input, const TensorView& output_delta,
                         TensorView& input_delta) const;
    void apply_delta_gpu(const TensorView& input, const TensorView& output_delta,
                         TensorView& input_delta) const;

    void plan_convolution_algorithms(const TensorView& input, const TensorView& output);

    array<pair<Index, Index>, 4> nhwc_padding() const;
};

/// @brief Fused 2D convolution + ReLU activation (uses cuDNN fused epilogue on GPU).
struct ConvolutionReluOp : Operator
{
    ConvolutionOp convolution;
    ActivationOp  activation;

    /// @brief Configures the underlying ConvolutionOp; ReLU is fixed.
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

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

/// @brief Layer normalization with learnable scale/shift, applied across the embedding dimension.
struct LayerNormOp : Operator
{
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    TensorView gamma;
    TensorView beta;

    TensorView gamma_gradient;
    TensorView beta_gradient;

    /// @brief Configures the operator for a (sequence_length, embedding_dimension) input.
    void set(Index sequence_length, Index embedding_dimension);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;

    void set_parameters_random() override { init_defaults(); }
    void set_parameters_glorot() override { init_defaults(); }

    /// @brief Resets gamma to one and beta to zero.
    void init_defaults();

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
    void apply_cpu(const TensorView& input,
                   TensorView& means, TensorView& standard_deviations, TensorView& normalized,
                   TensorView& output);
                   
    void apply_gpu(const TensorView& input,
                   TensorView& means, TensorView& standard_deviations,
                   TensorView& output);

    void apply_delta_cpu(const TensorView& output_delta,
                         const TensorView& standard_deviations,
                         const TensorView& normalized,
                         TensorView& input_delta) const;
    void apply_delta_gpu(const TensorView& input,
                         const TensorView& output_delta,
                         const TensorView& means, const TensorView& standard_deviations,
                         TensorView& input_delta) const;
};

/// @brief Projects (input_features) into (heads * head_dim) and reshapes for multi-head attention.
struct MultiHeadProjectionOp : Operator
{
    CombinationOp combination;
    Index input_features = 0;
    Index heads_number = 0;
    Index head_dimension = 0;
    Type compute_dtype = Type::FP32;

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

    /// @brief Configures the projection geometry.
    /// @param input_features Embedding dimension of the input tokens.
    /// @param heads_number Number of attention heads.
    /// @param head_dimension Per-head feature size.
    /// @param compute_dtype Dtype used for the projection matmul.
    void set(Index input_features, Index heads_number, Index head_dimension, Type compute_dtype);

    vector<TensorSpec> parameter_specs() const override { return combination.parameter_specs(); }
    void link_parameters(span<const TensorView> views) override { combination.link_parameters(views); }
    void link_gradients (span<const TensorView> views) override { combination.link_gradients(views); }

    void set_parameters_random() override { combination.set_parameters_random(); }
    void set_parameters_glorot() override { combination.set_parameters_glorot(); }

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    /// @brief Projects @p input and reshapes the result into per-head form in @p head_output.
    /// @param input Input tokens (batch, seq, embed).
    /// @param head_output Output tensor (batch, heads, seq, head_dim).
    /// @param scratch Shared transpose-scratch buffer used during the reshape.
    void apply(const TensorView& input, TensorView& head_output, float* scratch);

    /// @brief Computes input_delta from per-head gradients and updates the projection weight gradient.
    /// @param head_delta Gradient w.r.t. the per-head output.
    /// @param input Forward-pass input tokens.
    /// @param input_delta Output gradient w.r.t. the input.
    /// @param accumulate If true, accumulates into @p input_delta instead of overwriting.
    /// @param scratch Shared transpose-scratch buffer.
    void apply_delta(const TensorView& head_delta,
                     const TensorView& input,
                     TensorView& input_delta,
                     bool accumulate,
                     float* scratch) const;
};

/// @brief Scaled dot-product attention with optional causal mask and dropout.
struct AttentionOp : Operator
{
    Index heads_number = 0;
    Index head_dimension = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;
    bool  use_causal_mask = false;
    Type  compute_dtype = Type::FP32;

    MatrixR causal_mask;

    DropoutOp dropout;

    /// @brief Configures the attention geometry and compute precision.
    /// @param heads_number Number of attention heads.
    /// @param head_dimension Per-head feature size.
    /// @param query_sequence_length Length of the query sequence.
    /// @param source_sequence_length Length of the key/value sequence.
    /// @param use_causal_mask If true, applies an upper-triangular causal mask.
    /// @param compute_dtype Dtype used for the QK and softmax-V matmuls.
    void set(Index heads_number, Index head_dimension,
             Index query_sequence_length, Index source_sequence_length,
             bool use_causal_mask, Type compute_dtype);

    /// @brief Sets the post-softmax dropout rate (0 disables dropout).
    void set_dropout_rate(float rate) { dropout.set_rate(rate); }

    /// @brief Returns the tensor specs of the forward-pass scratch buffers used by the operator.
    vector<TensorSpec> forward_scratch_specs(Index batch_size) const;

    // Slot convention (set by hosting layer):
    //   input_slots  = {Query, Key, Value, Input}     (Input read via source_view_index)
    //   output_slots = {AttentionWeights, AttentionWeightsDropped}
    //   scratch_slots = {TransposeScratch}             (used as attention_out + mask_scratch)
    //   attention_output_slots = {ConcatenatedAttentionOutputs}  (backward-only: merged output for SDPA)
    //   output_delta_slots = {AttentionWeightDelta, InputQueryDelta, InputSourceDelta, ValueDelta}
    size_t source_view_index = 1;  // 1 = source path; clamped to size()-1 for self-attention

    vector<size_t> scratch_slots;
    vector<size_t> attention_output_slots;

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    /// @brief Computes attention output from Q, K, V; applies softmax, mask and dropout in place.
    /// @param query Query tensor (batch, heads, Q_seq, head_dim).
    /// @param key Key tensor (batch, heads, S_seq, head_dim).
    /// @param value Value tensor (batch, heads, S_seq, head_dim).
    /// @param source_input Source-side input used to build the padding mask.
    /// @param attention_weights Output attention scores (CPU only; empty on GPU).
    /// @param attention_weights_dropped Optional masked-out scores (CPU only).
    /// @param output Output tensor (batch, heads, Q_seq, head_dim).
    /// @param mask_scratch Scratch buffer for the temporary mask.
    /// @param is_training Enables dropout when true.
    void apply(const TensorView& query,                    // {B, H, Q_seq, D}
               const TensorView& key,                      // {B, H, S_seq, D}
               const TensorView& value,                    // {B, H, S_seq, D}
               const TensorView& source_input,             // {B, S_seq, embed} for padding mask
               TensorView& attention_weights,              // {B, H, Q_seq, S_seq} on CPU; empty on GPU
               TensorView& attention_weights_dropped,      // CPU-only, optional
               TensorView& output,                         // {B, H, Q_seq, D}
               float* mask_scratch,
               bool is_training);

    /// @brief Computes Q/K/V gradients from the output gradient and cached forward activations.
    /// @param query Forward Q tensor.
    /// @param key Forward K tensor.
    /// @param value Forward V tensor.
    /// @param attention_output Forward output tensor (read by GPU SDPA only).
    /// @param attention_weights Forward attention scores (CPU only).
    /// @param attention_weights_dropped Forward masked-out scores (CPU only).
    /// @param output_delta Gradient w.r.t. the attention output.
    /// @param attention_weight_delta Scratch tensor (CPU only).
    /// @param query_delta Output gradient w.r.t. Q.
    /// @param key_delta Output gradient w.r.t. K.
    /// @param value_delta Output gradient w.r.t. V.
    void apply_delta(const TensorView& query,
                     const TensorView& key,
                     const TensorView& value,
                     const TensorView& attention_output,   // forward output O — only read by GPU SDPA
                     const TensorView& attention_weights,
                     const TensorView& attention_weights_dropped,
                     const TensorView& output_delta,        // {B, H, Q_seq, D}
                     TensorView& attention_weight_delta,    // CPU-only scratch; empty on GPU
                     TensorView& query_delta,
                     TensorView& key_delta,
                     TensorView& value_delta) const;

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
                   float* mask_scratch,
                   bool is_training);

    void apply_gpu(const TensorView& query,
                   const TensorView& key,
                   const TensorView& value,
                   const TensorView& source_input,
                   TensorView& attention_weights,
                   TensorView& attention_weights_dropped,
                   TensorView& output,
                   float* mask_scratch,
                   bool is_training);

    // Same signature as apply_delta(); CPU path ignores attention_output.
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

    // Common backbone for the unfused CPU and GPU paths. The softmax-backward
    // step differs (Eigen vs cuDNN) and is supplied as a callable.
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

    // SDPA dropout RNG state. Forward advances `sdpa_dropout_offset`; backward
    // replays the previous step's offset via `sdpa_last_used_offset` so the
    // dropout mask is reproduced. Seed is fixed per AttentionOp instance.
    uint64_t sdpa_dropout_seed   = 0x9E3779B97F4A7C15ULL;
    uint64_t sdpa_dropout_offset = 0;
    mutable uint64_t sdpa_last_used_offset = 0;
};

/// @brief Reshapes (batch, heads, seq, head_dim) tensors back into (batch, seq, embed); no parameters.
// Forward = merge_heads; the layer hosts the shape configuration via set().
struct MergeOp : Operator
{
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index head_dimension = 0;
    Type  compute_dtype = Type::FP32;

    /// @brief Configures the merge geometry.
    void set(Index heads_number, Index query_sequence_length, Index head_dimension, Type compute_dtype);

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;

    // Note: writes the heads gradient back to the SAME forward slot it reads from in
    // forward (input_slots[0]). Buffer reuse for memory efficiency — the next backward
    // op (AttentionOp) consumes the heads gradient from that scratch slot.
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

/// @brief 2D pooling operator supporting max and average reductions.
struct PoolOp : Operator
{
    /// @brief Supported pooling reductions.
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

    /// @brief Configures the pooling geometry.
    /// @param input_h Input height in pixels.
    /// @param input_w Input width in pixels.
    /// @param input_c Number of input channels.
    /// @param pool_h Pooling window height.
    /// @param pool_w Pooling window width.
    /// @param row_stride Vertical stride.
    /// @param column_stride Horizontal stride.
    /// @param padding_h Vertical padding.
    /// @param padding_w Horizontal padding.
    /// @param method Reduction method (Max or Average).
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
    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
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

/// @brief Sequence-wide 1D pooling over the embedding dimension (mean or max).
struct Pool3dOp : Operator
{
    /// @brief Supported pooling reductions.
    enum Method { Max, Average };
    Method method = Max;

    // Slot convention (set by Pooling3d layer):
    //   input_slots  = {Input}
    //   output_slots = {Output, MaximalIndices}
    //   For AveragePooling, MaximalIndices is allocated empty (Shape{}).
    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

/// @brief Token embedding lookup with optional scaling and additive positional encoding.
struct EmbeddingLookupOp : Operator
{
    Index vocabulary_size     = 0;
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    bool  scale_embedding         = false;
    bool  add_positional_encoding = false;

    float embedding_scale = 1.0f;

    TensorView weights;
    TensorView positional_encoding;

    TensorView weight_gradient;

    /// @brief Configures the lookup table dimensions.
    /// @param new_vocabulary_size Number of unique tokens in the vocabulary.
    /// @param new_sequence_length Length of the input token sequence.
    /// @param new_embedding_dimension Size of each embedding vector.
    void set(Index new_vocabulary_size, Index new_sequence_length, Index new_embedding_dimension);

    vector<TensorSpec> parameter_specs() const override;
    vector<TensorSpec> state_specs()     const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;
    void link_states    (span<const TensorView> views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    /// @brief Fills the positional-encoding state tensor with the standard sinusoidal pattern.
    void init_positional_encoding();

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
    void apply_cpu(const TensorView& indices, TensorView& output);
    void apply_gpu(const TensorView& indices, TensorView& output);

    void apply_delta_cpu(const TensorView& indices, const TensorView& output_delta) const;
    void apply_delta_gpu(const TensorView& indices, const TensorView& output_delta) const;
};

/// @brief Flattens a multi-dimensional tensor into a 2D (batch, features) tensor.
struct FlatOp : Operator
{
    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    /// @copydoc Operator::back_propagate
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

/// @brief Clamps each output channel to a configurable lower/upper interval.
struct BoundOp : Operator
{
    /// @brief Disables bounding or enables per-channel clamping.
    enum class Method { NoBounding, Bounding };

    Method method = Method::Bounding;

    TensorView lower;
    TensorView upper;

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
};

/// @brief Scales inputs to a target range using per-feature minimum/maximum or mean/std statistics.
struct ScaleOp : Operator
{
    float min_range = -1.0f;
    float max_range = 1.0f;

    TensorView minimums;
    TensorView maximums;
    TensorView means;
    TensorView standard_deviations;
    TensorView scalers;

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
};

/// @brief Inverse of ScaleOp: maps normalized outputs back to the original feature range.
struct UnscaleOp : Operator
{
    float min_range = -1.0f;
    float max_range = 1.0f;

    TensorView minimums;
    TensorView maximums;
    TensorView means;
    TensorView standard_deviations;
    TensorView scalers;

    /// @copydoc Operator::forward_propagate
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
