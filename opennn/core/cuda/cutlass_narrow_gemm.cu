//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U T L A S S   N A R R O W   G E M M   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/cutlass_narrow_gemm.cuh"

#ifdef OPENNN_HAS_CUTLASS

#include "opennn/core/string_utilities.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/numeric_types.h"

namespace
{

using Element = cutlass::bfloat16_t;
using Accumulator = float;

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

        if (Gemm::get_workspace_size(arguments) != 0) return false;
        if (gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess) return false;

        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

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
    if (!bias) return false;

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
