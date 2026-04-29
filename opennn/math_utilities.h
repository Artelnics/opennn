//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T H   U T I L I T I E S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "tensor_utilities.h"

namespace opennn
{

enum class ActivationFunction{
    Linear, Sigmoid, HyperbolicTangent, RectifiedLinear, ScaledExponentialLinear, Softmax, Logistic
};

inline const EnumMap<ActivationFunction>& activation_function_map()
{
    static const vector<pair<ActivationFunction, string>> entries = {
        {ActivationFunction::Linear,                  "Linear"},
        {ActivationFunction::Sigmoid,                 "Sigmoid"},
        {ActivationFunction::HyperbolicTangent,       "HyperbolicTangent"},
        {ActivationFunction::RectifiedLinear,         "RectifiedLinear"},
        {ActivationFunction::ScaledExponentialLinear, "ScaledExponentialLinear"},
        {ActivationFunction::Softmax,                 "Softmax"},
        {ActivationFunction::Logistic,                "Logistic"}
    };
    static const EnumMap<ActivationFunction> map{entries};
    return map;
}

inline ActivationFunction string_to_activation(const string& name)
{
    return activation_function_map().from_string(name, ActivationFunction::Linear);
}

inline const string& activation_to_string(ActivationFunction function)
{
    return activation_function_map().to_string(function);
}

inline cudnnActivationMode_t to_cudnn_activation_mode(ActivationFunction function)
{
    switch(function)
    {
    case ActivationFunction::Sigmoid:                 return CUDNN_ACTIVATION_SIGMOID;
    case ActivationFunction::HyperbolicTangent:       return CUDNN_ACTIVATION_TANH;
    case ActivationFunction::RectifiedLinear:         return CUDNN_ACTIVATION_RELU;
    case ActivationFunction::ScaledExponentialLinear: return CUDNN_ACTIVATION_ELU;
    default:                                          return CUDNN_ACTIVATION_IDENTITY;
    }
}

struct ActivationArguments
{
    ActivationFunction activation_function = ActivationFunction::Linear;
    cudnnActivationDescriptor_t activation_descriptor = nullptr;
};

struct ConvolutionArguments
{
    Shape stride_shape;
    Shape padding_shape;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;
    cudnnFilterDescriptor_t kernel_descriptor = nullptr;
    cudnnConvolutionFwdAlgo_t algorithm_forward = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdDataAlgo_t algorithm_data = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdFilterAlgo_t algorithm_filter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    void* backward_filter_workspace = nullptr;
    size_t backward_filter_workspace_size = 0;
};

struct PoolingArguments
{
    Shape pool_dimensions;
    Shape stride_shape;
    Shape padding_shape;
    cudnnPoolingDescriptor_t pooling_descriptor = nullptr;
};

struct BatchNormalizationArguments
{
    type momentum;
    cudnnBatchNormMode_t batch_normalization_mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    cudnnTensorDescriptor_t per_activation_descriptor = nullptr;
};

struct DropoutArguments
{
    type rate = type(0);
    VectorR mask_cpu;
    cudnnDropoutDescriptor_t descriptor = nullptr;
    void* states = nullptr;
    size_t states_size = 0;
    void* reserve_space = nullptr;
    size_t reserve_size = 0;
};


// Generic

void padding(const TensorView& input, TensorView& output);
void bounding(const TensorView& input, const TensorView& lower_bounds, const TensorView& upper_bounds, TensorView& output);
void scale(const TensorView& input,
           const TensorView& minimums, const TensorView& maximums,
           const TensorView& means, const TensorView& standard_deviations,
           const TensorView& scalers,
           type min_range, type max_range,
           TensorView& output);
void unscale(const TensorView& input,
             const TensorView& minimums, const TensorView& maximums,
             const TensorView& means, const TensorView& standard_deviations,
             const TensorView& scalers,
             type min_range, type max_range,
             TensorView& output);
void copy(const TensorView& source, TensorView& destination);
void addition(const TensorView& input_1, const TensorView& input_2, TensorView& output);
void multiply(const TensorView& input_a, bool transpose_a, const TensorView& input_b, bool transpose_b, TensorView& output, type alpha = 1.0f, type beta = 0.0f);
void multiply_elementwise(const TensorView& input_a, const TensorView& input_b, TensorView& output);
void reduce_sum(const TensorView& input, TensorView& output, type alpha = 1.0f, type beta = 0.0f);
void softmax(TensorView& output);
void softmax_backward(const TensorView& softmax_out, TensorView& output_delta);

// Dense layer

void combination(const TensorView& input, const TensorView& weights, const TensorView& biases, TensorView& output);
void combination_gradient(const TensorView& output_delta, const TensorView& input, const TensorView& weights, TensorView& input_delta, TensorView& weight_gradient, TensorView& bias_gradient, bool accumulate_input_delta);
void activation(TensorView& output, ActivationArguments arguments);

// Equivalent to combination(...) followed by activation(output, args), but on
// GPU fuses both into one cuBLASLt matmul when the activation is supported by
// the cuBLASLt epilogue (currently RectifiedLinear). Other activations fall
// back to the unfused pair.
void combination_activation(const TensorView& input,
                            const TensorView& weights,
                            const TensorView& biases,
                            const ActivationArguments& activation_arguments,
                            TensorView& output);
void activation_delta(const TensorView& outputs, const TensorView& output_delta, TensorView& input_delta, const ActivationArguments& arguments);
void dropout(TensorView& output, DropoutArguments& args);
void dropout_delta(const TensorView& output_delta, TensorView& input_delta, const DropoutArguments& args);

// Batch normalization

void batch_normalization_inference(const TensorView& input, const TensorView& gamma, const TensorView& beta, const TensorView& running_mean, const TensorView& running_variance, TensorView& output);
void batch_normalization_training(const TensorView& input, const TensorView& gamma, const TensorView& beta, TensorView& running_mean, TensorView& running_variance, TensorView& mean, TensorView& inverse_variance, TensorView& output, type momentum = type(0.9));
void batch_normalization_backward(const TensorView& input, const TensorView& output, const TensorView& output_delta, const TensorView& mean, const TensorView& inverse_variance, const TensorView& gamma, TensorView& gamma_gradient, TensorView& beta_gradient, TensorView& input_delta);

// Layer normalization (3D)

void layernorm_forward(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                       TensorView& means, TensorView& standard_deviations, TensorView& normalized,
                       TensorView& output,
                       Index batch_size, Index sequence_length, Index embedding_dimension);

void layernorm_backward(const TensorView& input, const TensorView& output_delta,
                        const TensorView& means, const TensorView& standard_deviations,
                        const TensorView& normalized, const TensorView& gamma,
                        TensorView& gamma_gradient, TensorView& beta_gradient, TensorView& input_delta,
                        Index batch_size, Index sequence_length, Index embedding_dimension);

// Convolution

void convolution(const TensorView& input, const TensorView& kernel, const TensorView& bias, TensorView& output, const ConvolutionArguments& args = {});
void convolution_activation(const TensorView& input, const TensorView& weight, const TensorView& bias, TensorView& output, const ConvolutionArguments& conv_args = {}, const ActivationArguments& activation_arguments = {});
void convolution_backward_weights(const TensorView& input, const TensorView& output_delta, TensorView& weight_grad, TensorView& bias_grad, const ConvolutionArguments& args = {});
void convolution_backward_data(const TensorView& output_delta, const TensorView& kernel, TensorView& input_grad, const ConvolutionArguments& args = {});

// Pooling 4D

void max_pooling(const TensorView& input, TensorView& output, TensorView& maximal_indices, const PoolingArguments& arguments, bool is_training = false);
void average_pooling(const TensorView& input, TensorView& output, const PoolingArguments& arguments);
void max_pooling_backward(const TensorView& input, const TensorView& output, const TensorView& output_delta, const TensorView& maximal_indices, TensorView& input_delta, const PoolingArguments& args);
void average_pooling_backward(const TensorView& input, const TensorView& output, const TensorView& output_delta, TensorView& input_delta, const PoolingArguments& args);

// Pooling 3D

void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training);
void average_pooling_3d_forward(const TensorView& input, TensorView& output);
void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta);
void average_pooling_3d_backward(const TensorView& input, const TensorView& output_delta, TensorView& input_delta);

// Embedding

void embedding_backward(const TensorView& input_indices, const TensorView& output_delta, TensorView& weight_gradient, Index embedding_dimension, bool scale_embedding);

// Multi-head attention

void projection(const TensorView& input, const TensorView& weights, const TensorView& biases, TensorView& output, float* transpose_scratch);

void split_heads(const TensorView& source, TensorView& destination);
void merge_heads(const TensorView& source, TensorView& destination);

void projection_gradient(const TensorView& head_gradient,
                         const TensorView& input,
                         const TensorView& weights,
                         TensorView& bias_gradient,
                         TensorView& weight_gradient,
                         TensorView& input_delta,
                         float* transpose_scratch,
                         bool accumulate);

void attention_masks(const TensorView& source_input, TensorView& attention_weights, const MatrixR& causal_mask, bool use_causal_mask, float* padding_mask_scratch);

#ifdef OPENNN_WITH_CUDA

// Host-side scalar constants used as alpha/beta arguments by cuBLAS/cuDNN
// helpers below and by ops in math_utilities.cpp / layer.cpp. Defined here
// because cuBLAS APIs take them by pointer and they need a stable address.
inline const float one       =  1.0f;
inline const float zero      =  0.0f;
inline const float minus_one = -1.0f;

// ---------------------------------------------------------------------------
// GEMM wrappers around cuBLAS / cuBLASLt. Math-level operations consumed by
// combination / multiply / projection_gradient etc., kept alongside them.
// ---------------------------------------------------------------------------

inline void gemm_cuda(cublasOperation_t transa, cublasOperation_t transb,
                      int m, int n, int k,
                      const void* A, cudaDataType_t Atype, int lda,
                      const void* B, cudaDataType_t Btype, int ldb,
                      void* C, cudaDataType_t Ctype, int ldc,
                      float alpha = 1.0f, float beta = 0.0f)
{
    // CUBLAS_COMPUTE_32F_FAST_TF32 only triggers TF32 rounding for FP32 inputs;
    // for BF16 inputs cuBLAS rejects it (NOT_SUPPORTED) and we want
    // CUBLAS_COMPUTE_32F (FP32 accumulator over BF16 Tensor Cores).
    const cublasComputeType_t compute = (Atype == CUDA_R_16BF || Btype == CUDA_R_16BF)
                                            ? CUBLAS_COMPUTE_32F
                                            : CUBLAS_COMPUTE_DTYPE;
    CHECK_CUBLAS(cublasGemmEx(Device::get_cublas_handle(),
                              transa, transb,
                              m, n, k,
                              &alpha,
                              A, Atype, lda,
                              B, Btype, ldb,
                              &beta,
                              C, Ctype, ldc,
                              compute,
                              CUBLAS_GEMM_DEFAULT));
}

// Strided-batched GEMM (used by MHA's per-head batched matmul). Operands all
// share the same dtype, passed in by the caller (FP32 or BF16 depending on the
// network's activation_dtype).
inline void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                                      int m, int n, int k,
                                      const void* A, int lda, long long stride_a,
                                      const void* B, int ldb, long long stride_b,
                                      void* C, int ldc, long long stride_c,
                                      int batch_count,
                                      cudaDataType_t io_dtype = CUDA_R_32F,
                                      float alpha = 1.0f, float beta = 0.0f)
{
    const cublasComputeType_t compute = (io_dtype == CUDA_R_16BF)
                                            ? CUBLAS_COMPUTE_32F
                                            : CUBLAS_COMPUTE_DTYPE;
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(Device::get_cublas_handle(),
                                            transa, transb,
                                            m, n, k,
                                            &alpha,
                                            A, io_dtype, lda, stride_a,
                                            B, io_dtype, ldb, stride_b,
                                            &beta,
                                            C, io_dtype, ldc, stride_c,
                                            batch_count,
                                            compute,
                                            CUBLAS_GEMM_DEFAULT));
}

// Fused GEMM + (optionally activation +) bias via cuBLASLt epilogue: one launch
// for the whole gemm-bias[-relu] sequence, no intermediate write-then-read of
// the output tensor between stages.
//
// `epilogue` selects the post-matmul fusion. Currently supported:
//   - CUBLASLT_EPILOGUE_BIAS       : D = α(A·B) + bias                (default)
//   - CUBLASLT_EPILOGUE_RELU_BIAS  : D = max(0, α(A·B) + bias)
//
// Layout assumptions (encoded in the cached plan):
//   - All operands use the activation dtype declared on the cached LtMatmulPlan
//     (FP32 or BF16 — the caller chose when get_lt_gemm_plan was first invoked).
//   - Bias is FP32, length = m, broadcast along D's columns
//     (i.e. one bias element per output feature row).
//   - Tightly packed: lda = (transa==N ? m : k), ldb = (transb==N ? k : n), ldc = ldd = m.
//     If a caller needs different strides, it should not use this wrapper.
//
// Not thread-safe: the bias pointer is set on the cached op_desc per call, so concurrent
// invocations on the same shape would race. Matches the rest of this codebase's
// single-stream GPU usage assumption.
inline void gemm_bias_cuda(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const void* A,
                           const void* B,
                           void* C,
                           const float* bias,
                           cudaDataType_t io_dtype = CUDA_R_32F,
                           cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS,
                           float alpha = 1.0f, float beta = 0.0f)
{
    const LtMatmulPlan& plan = Device::get_lt_gemm_plan(m, n, k, transa, transb,
                                                        epilogue, io_dtype, io_dtype);

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    void* workspace = Device::get_cublas_lt_workspace();
    const size_t workspace_size = Device::cublas_lt_workspace_bytes();

    CHECK_CUBLAS(cublasLtMatmul(Device::get_cublas_lt_handle(),
                                plan.op_desc,
                                &alpha,
                                A, plan.a_desc,
                                B, plan.b_desc,
                                &beta,
                                C, plan.c_desc,   // C (read when beta != 0)
                                C, plan.d_desc,   // D (write); aliasing C is supported
                                plan.algo_valid ? &plan.algo : nullptr,
                                workspace, workspace_size,
                                Device::get_compute_stream()));
}

// Fused matmul + bias-gradient via cuBLASLt's BGRADA epilogue. Computes the
// matmul C = α(A·B) + βC and, as a side product, writes the row-wise reduction
// of A (in cuBLAS column-major view) into `bias_grad`. For Dense backward,
// pass A = output_delta with transA = N to get bias_grad = sum_rows(dY) — i.e.
// the bias gradient — for free, replacing a separate sum() reduction kernel.
//
// `bias_grad` length must be m (the matmul's M dim, == D rows in column-major).
// Same threading caveat as gemm_bias_cuda.
inline void gemm_bgrad_cuda(cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const void* A,
                            const void* B,
                            void* C,
                            float* bias_grad,
                            cudaDataType_t io_dtype = CUDA_R_32F,
                            float alpha = 1.0f, float beta = 0.0f)
{
    // Inputs (output_delta, input) follow the activation dtype passed by the
    // caller; the weight gradient (D / C) is always FP32 so Adam can accumulate
    // without precision loss. cuBLASLt mixed-dtype matmul handles bf16 in × fp32
    // out natively.
    const LtMatmulPlan& plan = Device::get_lt_gemm_plan(m, n, k, transa, transb,
                                                        CUBLASLT_EPILOGUE_BGRADA,
                                                        io_dtype,
                                                        CUDA_R_32F);

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_grad, sizeof(bias_grad)));

    void* workspace = Device::get_cublas_lt_workspace();
    const size_t workspace_size = Device::cublas_lt_workspace_bytes();

    CHECK_CUBLAS(cublasLtMatmul(Device::get_cublas_lt_handle(),
                                plan.op_desc,
                                &alpha,
                                A, plan.a_desc,
                                B, plan.b_desc,
                                &beta,
                                C, plan.c_desc,
                                C, plan.d_desc,
                                plan.algo_valid ? &plan.algo : nullptr,
                                workspace, workspace_size,
                                Device::get_compute_stream()));
}

#endif // OPENNN_WITH_CUDA

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
