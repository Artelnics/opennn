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

struct Operator
{
    virtual ~Operator() = default;

    virtual vector<pair<Shape, Type>> parameter_specs() const { return {}; }
    virtual vector<pair<Shape, Type>> state_specs()     const { return {}; }

    virtual void link_parameters(const vector<TensorView>&) {}
    virtual void link_states    (const vector<TensorView>&) {}

    virtual void to_JSON  (JsonWriter&)        const {}
    virtual void from_JSON(const Json*)              {}

    virtual void destroy_cuda() {}
};

struct Dropout : Operator
{
    float rate = 0.0f;

    VectorR mask_cpu;

    uint8_t* mask        = nullptr;
    size_t   mask_bytes  = 0;

    bool active() const { return rate > 0.0f; }

    void set_rate(float new_rate);

    void apply(TensorView& output);
    void apply_delta(TensorView& delta) const;

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

    void destroy_cuda() override;

    ~Dropout() override { destroy_cuda(); }

private:
    void apply_cpu(TensorView& output);
    void apply_gpu(TensorView& output);

    void apply_delta_cpu(TensorView& delta) const;
    void apply_delta_gpu(TensorView& delta) const;

    void ensure_mask(Index n);
};

struct Activation : Operator
{
    enum class Function { Identity, Sigmoid, Tanh, ReLU, SELU, Softmax };

    static const EnumMap<Function>& map();
    static Function from_string(const string& name);
    static const string& to_string(Function function);
    static cudnnActivationMode_t to_cudnn_mode(Function function);

    Function function = Function::Identity;

    cudnnActivationDescriptor_t descriptor = nullptr;

    void set_function(Function new_function);
    void set_function(const string& name);

    void apply(TensorView& output);
    void apply_delta(const TensorView& outputs, TensorView& delta) const;

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

    void destroy_cuda() override;

    ~Activation() override { destroy_cuda(); }

private:
    void apply_cpu(TensorView& output);
    void apply_gpu(TensorView& output);

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

    void set(Index new_input_features, Index new_output_features, Type new_weight_type = Type::FP32);

    vector<pair<Shape, Type>> parameter_specs() const override;
    void link_parameters(const vector<TensorView>& views) override;

    void apply(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS);

    void apply_delta(const TensorView& output_delta,
                     const TensorView& input,
                     TensorView& input_delta,
                     TensorView& weight_gradient,
                     TensorView& bias_gradient,
                     bool accumulate_input_delta = false) const;

private:
    void apply_cpu(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue);
    void apply_gpu(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue);

    void apply_delta_cpu(const TensorView& output_delta, const TensorView& input,
                         TensorView& input_delta, TensorView& weight_gradient, TensorView& bias_gradient,
                         bool accumulate_input_delta) const;
    void apply_delta_gpu(const TensorView& output_delta, const TensorView& input,
                         TensorView& input_delta, TensorView& weight_gradient, TensorView& bias_gradient,
                         bool accumulate_input_delta) const;
};

struct BatchNorm : Operator
{
    Index features = 0;
    float momentum = 0.1f;

    TensorView gamma;
    TensorView beta;
    TensorView running_mean;
    TensorView running_variance;

    void set(Index new_features, float new_momentum = 0.1f);

    vector<pair<Shape, Type>> parameter_specs() const override;
    vector<pair<Shape, Type>> state_specs()     const override;
    void link_parameters(const vector<TensorView>& views) override;
    void link_states    (const vector<TensorView>& views) override;

    void init_defaults();

    void apply_inference(const TensorView& input, TensorView& output);
    void apply_training (const TensorView& input,
                         TensorView& mean, TensorView& inverse_variance,
                         TensorView& output);
    void apply_delta    (const TensorView& input,
                         const TensorView& mean,
                         const TensorView& inverse_variance,
                         TensorView& gamma_gradient,
                         TensorView& beta_gradient,
                         TensorView& delta) const;

    void update_inference_cache();
    void invalidate_inference_cache() { inference_cache_dirty = true; }

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;

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
                         TensorView& gamma_gradient,
                         TensorView& beta_gradient,
                         TensorView& delta) const;
    void apply_delta_gpu(const TensorView& input,
                         const TensorView& mean,
                         const TensorView& inverse_variance,
                         TensorView& gamma_gradient,
                         TensorView& beta_gradient,
                         TensorView& delta) const;
};

struct Convolution : Operator
{
    Index input_height = 0;
    Index input_width = 0;
    Index input_channels = 0;

    Index kernels_number = 0;
    Index kernel_height = 0;
    Index kernel_width = 0;
    Index kernel_channels = 0;

    Index row_stride = 1;
    Index column_stride = 1;
    Index padding_height = 0;
    Index padding_width = 0;

    Type activation_dtype = Type::FP32;

    TensorView weights;
    TensorView bias;

    cudnnFilterDescriptor_t      kernel_descriptor      = nullptr;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;

    cudnnConvolutionFwdAlgo_t       algorithm_forward = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdDataAlgo_t   algorithm_data    = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdFilterAlgo_t algorithm_filter  = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

    void*  workspace                       = nullptr;
    size_t workspace_size                  = 0;
    void*  backward_filter_workspace       = nullptr;
    size_t backward_filter_workspace_size  = 0;

    void set(Index input_h, Index input_w, Index input_c,
             Index kernels_n, Index kernel_h, Index kernel_w, Index kernel_c,
             Index row_stride, Index column_stride,
             Index padding_h, Index padding_w,
             Type activation_dtype);

    Index get_output_height() const;
    Index get_output_width() const;

    vector<pair<Shape, Type>> parameter_specs() const override;
    void link_parameters(const vector<TensorView>& views) override;

    void init_cuda(Index batch_size, bool prefer_relu_algo);
    void destroy_cuda() override;

    ~Convolution() override { destroy_cuda(); }

    void apply(const TensorView& input, TensorView& output);
    void apply_delta(const TensorView& input,
                     const TensorView& output_delta,
                     TensorView& weight_gradient,
                     TensorView& bias_gradient,
                     TensorView& input_delta) const;

private:
    void apply_cpu(const TensorView& input, TensorView& output);
    void apply_gpu(const TensorView& input, TensorView& output);

    void apply_delta_cpu(const TensorView& input, const TensorView& output_delta,
                         TensorView& weight_gradient, TensorView& bias_gradient,
                         TensorView& input_delta) const;
    void apply_delta_gpu(const TensorView& input, const TensorView& output_delta,
                         TensorView& weight_gradient, TensorView& bias_gradient,
                         TensorView& input_delta) const;
};

struct LayerNorm : Operator
{
    Index sequence_length     = 0;
    Index embedding_dimension = 0;

    TensorView gamma;
    TensorView beta;

    void set(Index sequence_length, Index embedding_dimension);

    vector<pair<Shape, Type>> parameter_specs() const override;
    void link_parameters(const vector<TensorView>& views) override;

    void init_defaults();

    void apply(const TensorView& input,
               TensorView& means, TensorView& standard_deviations, TensorView& normalized,
               TensorView& output, Index batch_size);

    void apply_delta(const TensorView& input,
                     const TensorView& output_delta,
                     const TensorView& means, const TensorView& standard_deviations,
                     const TensorView& normalized,
                     TensorView& gamma_gradient, TensorView& beta_gradient,
                     TensorView& input_delta, Index batch_size) const;

private:
    void apply_cpu(const TensorView& input,
                   TensorView& means, TensorView& standard_deviations, TensorView& normalized,
                   TensorView& output, Index batch_size);
    void apply_gpu(const TensorView& input,
                   TensorView& means, TensorView& standard_deviations,
                   TensorView& output, Index batch_size);

    void apply_delta_cpu(const TensorView& output_delta,
                         const TensorView& standard_deviations,
                         const TensorView& normalized,
                         TensorView& gamma_gradient, TensorView& beta_gradient,
                         TensorView& input_delta, Index batch_size) const;
    void apply_delta_gpu(const TensorView& input,
                         const TensorView& output_delta,
                         const TensorView& means, const TensorView& standard_deviations,
                         TensorView& gamma_gradient, TensorView& beta_gradient,
                         TensorView& input_delta, Index batch_size) const;
};

struct Pool : Operator
{
    Index input_height = 0;
    Index input_width = 0;
    Index input_channels = 0;

    Index pool_height = 1;
    Index pool_width = 1;
    Index row_stride = 1;
    Index column_stride = 1;
    Index padding_height = 0;
    Index padding_width = 0;

    int method = 0;  // 0 = Max, 1 = Average (avoids forward-decl-only enum at this point)

    cudnnPoolingDescriptor_t pooling_descriptor = nullptr;

    void set(Index input_h, Index input_w, Index input_c,
             Index pool_h, Index pool_w,
             Index row_stride, Index column_stride,
             Index padding_h, Index padding_w,
             int method);

    Index get_output_height() const;
    Index get_output_width() const;

    void init_cuda();
    void destroy_cuda() override;

    ~Pool() override { destroy_cuda(); }

    void apply(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
    void apply_delta(const TensorView& input,
                     const TensorView& output,
                     const TensorView& output_delta,
                     const TensorView& maximal_indices,
                     TensorView& input_delta) const;

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

    void set(Index new_vocabulary_size, Index new_sequence_length, Index new_embedding_dimension);

    vector<pair<Shape, Type>> parameter_specs() const override;
    vector<pair<Shape, Type>> state_specs()     const override;
    void link_parameters(const vector<TensorView>& views) override;
    void link_states    (const vector<TensorView>& views) override;

    void init_positional_encoding();

    void apply(const TensorView& indices, TensorView& output);
    void apply_delta(const TensorView& indices,
                     const TensorView& output_delta,
                     TensorView& weight_gradient) const;

private:
    void apply_cpu(const TensorView& indices, TensorView& output);
    void apply_gpu(const TensorView& indices, TensorView& output);

    void apply_delta_cpu(const TensorView& indices, const TensorView& output_delta, TensorView& weight_gradient) const;
    void apply_delta_gpu(const TensorView& indices, const TensorView& output_delta, TensorView& weight_gradient) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
