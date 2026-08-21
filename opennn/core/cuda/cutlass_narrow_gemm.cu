//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U T L A S S   N A R R O W   G E M M   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// The first layer of a tabular classifier contracts a handful of features into
// a wide hidden layer, and cuBLASLt is poor at it. On the HIGGS 28 -> 1024
// layer it reaches 23.8 TFLOP/s: the input's leading dimension is 28 elements,
// so cuBLASLt can only promise two-element alignment and picks an `align2`
// kernel out of a catalogue compiled for sm_80.
//
// The kernels in that catalogue are CUTLASS kernels - the profiles name them
// `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_...` - so the fix is not a
// different library but the same one instantiated for this shape: alignment 4
// on the input, which 28 does divide, and a threadblock tile picked by row
// count rather than by heuristic. Milliseconds against cuBLASLt's best of eight
// heuristics, bf16, and bit-identical output at every point:
//
//     rows        256    1,024    4,096    8,192   16,384   65,536
//     cuBLASLt  0.0022   0.0042   0.0114   0.0216   0.0415   0.1831
//     CUTLASS   0.0022   0.0034   0.0090   0.0156   0.0281   0.1730
//                1.03x    1.25x    1.27x    1.39x    1.48x    1.06x
//
// 16,384 is the one that matters most, because the forward chunks its rows at
// 16,384 above that count, so every large batch runs this layer at that shape.
//
// The bias arrives as a C operand whose row stride is zero, which broadcasts one
// vector down the whole output and lets the ReLU ride in the same epilogue -
// the same fusion the cuBLASLt path gets from CUBLASLT_EPILOGUE_RELU_BIAS.

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/cutlass_narrow_gemm.cuh"
#include "opennn/core/string_utilities.h"

#ifdef OPENNN_HAS_CUTLASS

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/numeric_types.h"

namespace
{

using Element = cutlass::bfloat16_t;
using Accumulator = float;

// A is the input (rows x k, row-major), B the weight panel (k x out, row-major),
// C the broadcast bias and D the output. Alignment 4 on A is what 28 allows;
// 8 on B and D is what an output width that is a multiple of 8 allows.
template<typename ThreadblockShape, typename WarpShape, int Stages, bool Relu>
struct NarrowGemm
{
    using Epilogue = std::conditional_t<
        Relu,
        cutlass::epilogue::thread::LinearCombinationRelu<Element, 8, Accumulator, Accumulator>,
        cutlass::epilogue::thread::LinearCombination<Element, 8, Accumulator, Accumulator>>;

    using Gemm = cutlass::gemm::device::Gemm<
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::RowMajor,
        Accumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        cutlass::gemm::GemmShape<16, 8, 16>,
        Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages,
        4,
        8>;

    static bool run(int rows, int contraction, int out_features,
                    const Element* input, const Element* weights, const Element* bias,
                    Element* output, cudaStream_t stream)
    {
        typename Gemm::Arguments arguments(
            {rows, out_features, contraction},
            {const_cast<Element*>(input), contraction},
            {const_cast<Element*>(weights), out_features},
            {const_cast<Element*>(bias), 0},
            {output, out_features},
            {Accumulator(1), Accumulator(bias ? 1 : 0)});

        Gemm gemm;
        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;

        // Split-K is the only thing that needs a workspace and this never asks
        // for it, which is what makes the call safe to capture in a CUDA graph:
        // a steady-state forward must not allocate.
        if (Gemm::get_workspace_size(arguments) != 0) return false;
        if (gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess) return false;

        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

// The tile that wins moves with the row count, so it is chosen rather than
// fixed: measured, 64x64 below 512 rows, 64x128 to 2,048 and above 32,768, and
// 128x128 between - which is where the chunked forward spends every large batch.
template<bool Relu>
bool dispatch(int rows, int contraction, int out_features,
              const Element* input, const Element* weights, const Element* bias,
              Element* output, cudaStream_t stream)
{
    using Small = cutlass::gemm::GemmShape<64, 64, 32>;
    using Medium = cutlass::gemm::GemmShape<64, 128, 32>;
    using Large = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpSmall = cutlass::gemm::GemmShape<32, 64, 32>;
    using WarpLarge = cutlass::gemm::GemmShape<64, 64, 32>;

    if (rows <= 512)
        return NarrowGemm<Small, WarpSmall, 6, Relu>::run(
            rows, contraction, out_features, input, weights, bias, output, stream);

    if (rows <= 2048 || rows > 32768)
        return NarrowGemm<Medium, WarpSmall, 4, Relu>::run(
            rows, contraction, out_features, input, weights, bias, output, stream);

    return NarrowGemm<Large, WarpLarge, 3, Relu>::run(
        rows, contraction, out_features, input, weights, bias, output, stream);
}

}

bool narrow_k_linear_forward_cutlass(Index rows, Index contraction, Index out_features,
                                     const void* input, const void* weights, const void* bias,
                                     void* output, bool relu, cudaStream_t stream)
{
    static const bool enabled = opennn::env_flag_enabled("OPENNN_CUTLASS_NARROW_K", true);
    if (!enabled) return false;

    if (contraction <= 0 || contraction > 32 || contraction % 4 != 0) return false;
    if (out_features <= 0 || out_features % 8 != 0) return false;
    if (rows <= 0 || rows > Index(std::numeric_limits<int>::max())) return false;
    if (!bias) return false;                    // the broadcast C operand needs a vector

    const auto aligned = [](const void* pointer) {
        return reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
    };
    if (!aligned(input) || !aligned(weights) || !aligned(bias) || !aligned(output)) return false;

    const int rows_int = int(rows);
    const int contraction_int = int(contraction);
    const int out_int = int(out_features);

    const Element* const a = static_cast<const Element*>(input);
    const Element* const b = static_cast<const Element*>(weights);
    const Element* const c = static_cast<const Element*>(bias);
    Element* const d = static_cast<Element*>(output);

    return relu
        ? dispatch<true>(rows_int, contraction_int, out_int, a, b, c, d, stream)
        : dispatch<false>(rows_int, contraction_int, out_int, a, b, c, d, stream);
}

#else

bool narrow_k_linear_forward_cutlass(Index, Index, Index,
                                     const void*, const void*, const void*,
                                     void*, bool, cudaStream_t)
{
    return false;
}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
