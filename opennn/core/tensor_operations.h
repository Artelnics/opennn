//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   O P E R A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/enum_map.h"

namespace opennn
{

inline constexpr float INV_SQRT_2      = 0.70710678118654752440f;
inline constexpr float INV_SQRT_2_PI   = 0.39894228040143267794f;
inline constexpr float SQRT_2_OVER_PI  = 0.7978845608028654f;
inline constexpr float GELU_TANH_CUBIC = 0.044715f;

inline float gelu_value(float x)
{
    return 0.5f * x * (1.0f + erff(x * INV_SQRT_2));
}

inline float gelu_tanh_value(float x)
{
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * (x + GELU_TANH_CUBIC * x * x * x)));
}

inline float silu_value(float x)
{
    return x / (1.0f + exp(-x));
}

inline float gelu_derivative(float x)
{
    const float cdf = 0.5f * (1.0f + erff(x * INV_SQRT_2));
    const float pdf = INV_SQRT_2_PI * expf(-0.5f * x * x);
    return cdf + x * pdf;
}

inline float gelu_tanh_derivative(float x)
{
    const float x2 = x * x;
    const float u = SQRT_2_OVER_PI * (x + GELU_TANH_CUBIC * x * x2);
    const float t = tanhf(u);
    const float du = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_TANH_CUBIC * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
}

inline float silu_derivative(float x)
{
    const float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f + x * (1.0f - s));
}

const EnumMap<ActivationFunction>& activation_function_map();
const string& activation_function_to_string(ActivationFunction);
ActivationFunction activation_function_from_string(const string&);

bool activation_needs_input(ActivationFunction function);

inline float activation_forward_value(ActivationFunction function, float x)
{
    using enum ActivationFunction;
    switch (function)
    {
    case Identity:  return x;
    case Sigmoid:   return 1.0f / (1.0f + exp(-x));
    case Tanh:      return tanh(x);
    case ReLU:      return max(0.0f, x);
    case LeakyReLU: return x >= 0.0f ? x : x * LEAKY_RELU_SLOPE;
    case GELU:      return gelu_value(x);
    case GELUTanh:  return gelu_tanh_value(x);
    case SiLU:      return silu_value(x);
    case Softmax:   break;
    }

    throw runtime_error("activation_forward_value: Softmax must be handled separately.");
}

inline float activation_derivative_from_output_value(ActivationFunction function, float y)
{
    using enum ActivationFunction;
    switch (function)
    {
    case Identity:  return 1.0f;
    case Sigmoid:   return y * (1.0f - y);
    case Tanh:      return 1.0f - y * y;
    case ReLU:      return y > 0.0f ? 1.0f : 0.0f;
    case LeakyReLU: return y >= 0.0f ? 1.0f : LEAKY_RELU_SLOPE;
    case Softmax:   break;
    case GELU:
    case GELUTanh:
    case SiLU:      break;
    }

    throw runtime_error("activation_derivative_from_output_value: Softmax/GELU/GELUTanh/SiLU must be handled separately.");
}

template<typename TensorType>
typename TensorType::PlainObject activation_forward_values(ActivationFunction function, const TensorType& values)
{
    return values.unaryExpr([function](float value) { return activation_forward_value(function, value); });
}

template<typename TensorType>
typename TensorType::PlainObject activation_derivative_from_output_values(ActivationFunction function, const TensorType& values)
{
    return values.unaryExpr([function](float value) { return activation_derivative_from_output_value(function, value); });
}

void copy(const TensorView&, TensorView&);

void add(const TensorView&, const TensorView&, TensorView&);

void multiply(const TensorView&, bool, const TensorView&, bool, TensorView&, float alpha = 1.0f, float beta = 0.0f);

void softmax(TensorView&);

void activation_forward(TensorView&, ActivationFunction);
void activation_backward(const TensorView&, TensorView&, ActivationFunction);

// `fused_activation` is applied to the output after the bias. It exists for the
// activations cuBLASLt has no epilogue for: the single-output forward folds it
// into its own store, and every other path runs it as the separate pass it
// would have been anyway - so a caller may always ask, and asking is only ever
// a performance decision, never a correctness one.
void linear_forward(const TensorView&, const TensorView&, const TensorView&,
                    TensorView&, cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS,
                    TensorView* pre_activation = nullptr,
                    const TensorView& weight_scale = {},
                    ActivationFunction fused_activation = ActivationFunction::Identity);
// What a caller can ask the backward to fold into the input-delta GEMM instead
// of paying for it in a pass of its own.
struct LinearBackwardOptions
{
    // The ReLU derivative as a bitmask from a cuBLASLt auxiliary epilogue.
    const TensorView* drelu_mask = nullptr;

    // Another consumer's delta for the same input, summed by the GEMM:
    // input_delta = output_delta * W^T + addend.
    const TensorView* addend = nullptr;

    // Ask for the input delta to be masked by the derivative of the ReLU that
    // produced the input, and learn whether it happened: only the single-output
    // path can, so a caller that reads back false must run the activation
    // backward itself.
    bool* fused_input_relu = nullptr;
};

void linear_backward(const TensorView&, const TensorView&, const TensorView&,
                     const TensorView&, const TensorView&,
                     TensorView&, bool accumulate_input_delta = false,
                     const LinearBackwardOptions& options = {});








void linear_forward_transposed(const TensorView& input, const TensorView& embed_weight, TensorView& output,
                          const TensorView& weight_scale = {});


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
