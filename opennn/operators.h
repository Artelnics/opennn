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

namespace opennn
{

class Json;
class JsonWriter;

#ifdef OPENNN_HAS_CUDA
struct LtMatmulPlan;
#endif

struct ForwardPropagation;
struct BackPropagation;

struct Operator
{
    virtual ~Operator() = default;

    virtual vector<pair<Shape, Type>> parameter_specs() const { return {}; }
    virtual vector<pair<Shape, Type>> state_specs()     const { return {}; }

    virtual size_t parameter_count() const { return 0; }
    virtual size_t state_count()     const { return 0; }

    virtual void link_parameters(const vector<TensorView>&) {}
    virtual void link_gradients (const vector<TensorView>&) {}
    virtual void link_states    (const vector<TensorView>&) {}

    virtual void set_parameters_random() {}
    virtual void set_parameters_glorot() {}

    virtual void forward_propagate(ForwardPropagation&, size_t, bool) noexcept {}
    virtual void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept {}

    virtual void to_JSON  (JsonWriter&) const {}
    virtual void from_JSON(const Json*)       {}
    virtual void load_state_from_JSON(const Json*) {}

    virtual void destroy_cuda() {}

    virtual Index get_delta_bytes() const;

    vector<size_t> input_slots;
    vector<size_t> output_slots;

    vector<size_t> input_delta_slots;
    vector<size_t> output_delta_slots;
};

struct Add : Operator
{
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
    void check(const vector<TensorView>& inputs, const TensorView& output) const;
};

struct Dropout : Operator
{
    float rate = 0.0f;

    Buffer mask;

    vector<size_t> save_slots;

    bool active() const { return rate > 0.0f; }

    void set_rate(float new_rate);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    void apply_cpu(TensorView& output);
    void apply_gpu(TensorView& output);

    void apply_delta(TensorView& delta) const;

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

    void destroy_cuda() override;

    ~Dropout() override { destroy_cuda(); }

    Dropout() = default;
    Dropout(Dropout&&) noexcept = default;
    Dropout& operator=(Dropout&&) noexcept = default;

private:
    void apply_delta_cpu(TensorView& delta) const;
    void apply_delta_gpu(TensorView& delta) const;

    void ensure_mask(Index n);
};

struct Activation : Operator
{
    enum class Function { Identity, Sigmoid, Tanh, ReLU, Softmax };

    static const EnumMap<Function>& map();
    static Function from_string(const string& name);
    static const string& to_string(Function function);
    static cudnnActivationMode_t to_cudnn_mode(Function function);

    Function function = Function::Identity;

    cudnnActivationDescriptor_t descriptor = nullptr;

    // Backward override: when non-empty, back_propagate reads the activation's
    // output from this slot instead of output_slots[0]. Used when a downstream
    // operator (e.g. Dropout) overwrites the activation's output in place.
    vector<size_t> output_slots_backward;

    void set_function(Function new_function);
    void set_function(const string& name);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    void apply_cpu(TensorView& output);
    void apply_gpu(TensorView& output);

    void apply_delta(const TensorView& outputs, TensorView& delta) const;

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

    void destroy_cuda() override;

    ~Activation() override { destroy_cuda(); }

    Activation() = default;
    Activation(const Activation&) = delete;
    Activation& operator=(const Activation&) = delete;

private:
    void apply_delta_cpu(const TensorView& outputs, TensorView& delta) const;
    void apply_delta_gpu(const TensorView& outputs, TensorView& delta) const;
};

struct Combination : Operator
{
    Index input_features  = 0;
    Index output_features = 0;
    Type  weight_type     = Type::FP32;

    TensorView weights;
    TensorView bias;

    TensorView weight_gradient;
    TensorView bias_gradient;

    void set(Index new_input_features, Index new_output_features, Type new_weight_type = Type::FP32);

    vector<pair<Shape, Type>> parameter_specs() const override;
    size_t parameter_count() const override { return 2; }
    void link_parameters         (const vector<TensorView>& views) override;
    void link_gradients(const vector<TensorView>& views) override;

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
    void apply_cpu(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue);
    void apply_gpu(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue);

    void apply_delta_cpu(const TensorView& output_delta, const TensorView& input,
                         TensorView& input_delta, bool accumulate_input_delta) const;
    void apply_delta_gpu(const TensorView& output_delta, const TensorView& input,
                         TensorView& input_delta, bool accumulate_input_delta) const;
};

struct CombinationRelu : Operator
{
    Combination combination;
    Activation  activation;

    void set(Index input_features, Index output_features, Type weight_type = Type::FP32);

    vector<pair<Shape, Type>> parameter_specs() const override { return combination.parameter_specs(); }
    size_t parameter_count() const override { return combination.parameter_count(); }
    void link_parameters(const vector<TensorView>& views) override { combination.link_parameters(views); }
    void link_gradients (const vector<TensorView>& views) override { combination.link_gradients(views); }

    void set_parameters_random() override { combination.set_parameters_random(); }
    void set_parameters_glorot() override { combination.set_parameters_glorot(); }

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

struct BatchNorm : Operator
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

    vector<pair<Shape, Type>> parameter_specs() const override;
    vector<pair<Shape, Type>> state_specs()     const override;
    size_t parameter_count() const override { return active() ? 2 : 0; }
    size_t state_count()     const override { return active() ? 2 : 0; }
    void link_parameters         (const vector<TensorView>& views) override;
    void link_gradients(const vector<TensorView>& views) override;
    void link_states             (const vector<TensorView>& views) override;

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

struct Convolution : Operator
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

    Buffer workspace{Device::CUDA};
    Buffer backward_filter_workspace{Device::CUDA};

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

    vector<pair<Shape, Type>> parameter_specs() const override;
    size_t parameter_count() const override { return 2; }
    void link_parameters         (const vector<TensorView>& views) override;
    void link_gradients(const vector<TensorView>& views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void destroy_cuda() override;

    ~Convolution() override { destroy_cuda(); }

    Convolution() = default;
    Convolution(const Convolution&) = delete;
    Convolution& operator=(const Convolution&) = delete;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    void apply_cpu(const TensorView& input, TensorView& output);
    void apply_gpu(const TensorView& input, TensorView& output, cudnnActivationDescriptor_t fused_activation = nullptr);

    void apply_delta(const TensorView& input,
                     const TensorView& output_delta,
                     TensorView& input_delta) const;

private:

    void apply_delta_cpu(const TensorView& input, const TensorView& output_delta,
                         TensorView& input_delta) const;
    void apply_delta_gpu(const TensorView& input, const TensorView& output_delta,
                         TensorView& input_delta) const;

    void plan_convolution_algorithms(const TensorView& input, const TensorView& output);
};

struct ConvolutionRelu : Operator
{
    Convolution convolution;
    Activation  activation;

    void set(Index input_h, Index input_w,
             Index kernels_n, Index kernel_h, Index kernel_w, Index kernel_c,
             Index row_stride, Index column_stride,
             Index padding_h, Index padding_w,
             Type compute_dtype);

    vector<pair<Shape, Type>> parameter_specs() const override { return convolution.parameter_specs(); }
    size_t parameter_count() const override { return convolution.parameter_count(); }
    void link_parameters(const vector<TensorView>& views) override { convolution.link_parameters(views); }
    void link_gradients (const vector<TensorView>& views) override { convolution.link_gradients(views); }

    void set_parameters_random() override { convolution.set_parameters_random(); }
    void set_parameters_glorot() override { convolution.set_parameters_glorot(); }

    void destroy_cuda() override { convolution.destroy_cuda(); activation.destroy_cuda(); }
    ~ConvolutionRelu() override { destroy_cuda(); }

    ConvolutionRelu() = default;
    ConvolutionRelu(const ConvolutionRelu&) = delete;
    ConvolutionRelu& operator=(const ConvolutionRelu&) = delete;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

struct LayerNorm : Operator
{
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    TensorView gamma;
    TensorView beta;

    TensorView gamma_gradient;
    TensorView beta_gradient;

    void set(Index sequence_length, Index embedding_dimension);

    vector<pair<Shape, Type>> parameter_specs() const override;
    size_t parameter_count() const override { return 2; }
    void link_parameters         (const vector<TensorView>& views) override;
    void link_gradients(const vector<TensorView>& views) override;

    void set_parameters_random() override { init_defaults(); }
    void set_parameters_glorot() override { init_defaults(); }

    void init_defaults();

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
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

struct MultiHeadProjection : Operator
{
    Combination combination;
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

    void set(Index input_features, Index heads_number, Index head_dimension, Type compute_dtype);

    vector<pair<Shape, Type>> parameter_specs() const override { return combination.parameter_specs(); }
    size_t parameter_count() const override { return combination.parameter_count(); }
    void link_parameters         (const vector<TensorView>& views) override { combination.link_parameters(views); }
    void link_gradients(const vector<TensorView>& views) override { combination.link_gradients(views); }

    void set_parameters_random() override { combination.set_parameters_random(); }
    void set_parameters_glorot() override { combination.set_parameters_glorot(); }

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    void apply(const TensorView& input, TensorView& head_output, float* scratch);

    void apply_delta(const TensorView& head_delta,
                     const TensorView& input,
                     TensorView& input_delta,
                     bool accumulate,
                     float* scratch) const;
};

struct Attention : Operator
{
    Index heads_number = 0;
    Index head_dimension = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;
    bool  use_causal_mask = false;
    Type  compute_dtype = Type::FP32;

    MatrixR causal_mask;

    Dropout dropout;

    void set(Index heads_number, Index head_dimension,
             Index query_sequence_length, Index source_sequence_length,
             bool use_causal_mask, Type compute_dtype);

    void set_dropout_rate(float rate) { dropout.set_rate(rate); }

    vector<pair<Shape, Type>> forward_scratch_specs(Index batch_size) const;

    // Slot convention (set by hosting layer):
    //   input_slots  = {Query, Key, Value, Input}     (Input read via source_view_index)
    //   output_slots = {AttentionWeights, AttentionWeightsDropped}
    //   scratch_slots = {TransposeScratch}             (used as attention_out + mask_scratch)
    //   attention_output_slots = {ConcatenatedAttentionOutputs}  (backward-only: merged output for SDPA)
    //   output_delta_slots = {AttentionWeightDelta, InputQueryDelta, InputSourceDelta, ValueDelta}
    size_t source_view_index = 1;  // 1 = source path; clamped to size()-1 for self-attention

    vector<size_t> scratch_slots;
    vector<size_t> attention_output_slots;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    void apply(const TensorView& query,                    // {B, H, Q_seq, D}
               const TensorView& key,                      // {B, H, S_seq, D}
               const TensorView& value,                    // {B, H, S_seq, D}
               const TensorView& source_input,             // {B, S_seq, embed} for padding mask
               TensorView& attention_weights,              // {B, H, Q_seq, S_seq} on CPU; empty on GPU
               TensorView& attention_weights_dropped,      // CPU-only, optional
               TensorView& output,                         // {B, H, Q_seq, D}
               float* mask_scratch,
               bool is_training);

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

    Attention();
    ~Attention() override;
    Attention(Attention&&) noexcept;
    Attention& operator=(Attention&&) noexcept;
    Attention(const Attention&) = delete;
    Attention& operator=(const Attention&) = delete;

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
};

// Reshapes a (batch, heads, seq, head_dim) tensor into (batch, seq, embed)
// by interleaving heads back into the embedding dimension. Forward = merge_heads;
// no parameters; the layer hosts the shape configuration via set().
struct Merge : Operator
{
    Index heads_number = 0;
    Index query_sequence_length = 0;
    Index head_dimension = 0;
    Type  compute_dtype = Type::FP32;

    void set(Index heads_number, Index query_sequence_length, Index head_dimension, Type compute_dtype);

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;

    // Note: writes the heads gradient back to the SAME forward slot it reads from in
    // forward (input_slots[0]). Buffer reuse for memory efficiency — the next backward
    // op (Attention) consumes the heads gradient from that scratch slot.
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

struct Pool : Operator
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

    ~Pool() override { destroy_cuda(); }

    Pool() = default;
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

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

struct Pool3d : Operator
{
    enum Method { Max, Average };
    Method method = Max;

    // Slot convention (set by Pooling3d layer):
    //   input_slots  = {Input}
    //   output_slots = {Output, MaximalIndices}
    //   For AveragePooling, MaximalIndices is allocated empty (Shape{}).
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

struct EmbeddingLookup : Operator
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

    void set(Index new_vocabulary_size, Index new_sequence_length, Index new_embedding_dimension);

    vector<pair<Shape, Type>> parameter_specs() const override;
    vector<pair<Shape, Type>> state_specs()     const override;
    size_t parameter_count() const override { return 1; }
    size_t state_count()     const override;
    void link_parameters         (const vector<TensorView>& views) override;
    void link_gradients(const vector<TensorView>& views) override;
    void link_states             (const vector<TensorView>& views) override;

    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void init_positional_encoding();

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

private:
    void apply_cpu(const TensorView& indices, TensorView& output);
    void apply_gpu(const TensorView& indices, TensorView& output);

    void apply_delta_cpu(const TensorView& indices, const TensorView& output_delta) const;
    void apply_delta_gpu(const TensorView& indices, const TensorView& output_delta) const;
};

struct Flat : Operator
{
    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;
};

struct Bound : Operator
{
    enum class Method { NoBounding, Bounding };

    Method method = Method::Bounding;
    Index features = 0;

    TensorView lower;
    TensorView upper;

    void set(Method new_method, Index new_features) { method = new_method; features = new_features; }

    vector<pair<Shape, Type>> state_specs() const override;
    size_t state_count() const override;
    void link_states(const vector<TensorView>& views) override;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;

    void load_state_from_JSON(const Json* parent) override;
};

struct Scale : Operator
{
    Index features = 0;
    float min_range = -1.0f;
    float max_range = 1.0f;

    TensorView minimums;
    TensorView maximums;
    TensorView means;
    TensorView standard_deviations;
    TensorView scalers;

    void set(Index new_features) { features = new_features; }

    vector<pair<Shape, Type>> state_specs() const override;
    size_t state_count() const override;
    void link_states(const vector<TensorView>& views) override;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;

    void load_state_from_JSON(const Json* parent) override;
};

struct Unscale : Operator
{
    Index features = 0;
    float min_range = -1.0f;
    float max_range = 1.0f;

    TensorView minimums;
    TensorView maximums;
    TensorView means;
    TensorView standard_deviations;
    TensorView scalers;

    void set(Index new_features) { features = new_features; }

    vector<pair<Shape, Type>> state_specs() const override;
    size_t state_count() const override;
    void link_states(const vector<TensorView>& views) override;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept override;

    void load_state_from_JSON(const Json* parent) override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
