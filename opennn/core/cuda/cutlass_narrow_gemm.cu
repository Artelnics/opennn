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

// The three tile shapes the path has always carried. They used to be picked by
// a row ladder with boundaries at 512, 2,048 and 32,768, the last of which sent
// the widest shapes back to the middle tile -- a non-monotone step no
// measurement backed. The tiles are unchanged; only the choice between them is
// now timed. All three share the epilogue, the alpha and beta, and -- because
// contraction <= 32 leaves the mainloop a single 32-wide K tile, walked by the
// same two 16-deep mma steps whatever the warp shape -- the accumulation order
// too, so they compute the same bytes and only the cost separates them.
using Small = cutlass::gemm::GemmShape<64, 64, 32>;
using Medium = cutlass::gemm::GemmShape<64, 128, 32>;
using Large = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpSmall = cutlass::gemm::GemmShape<32, 64, 32>;
using WarpLarge = cutlass::gemm::GemmShape<64, 64, 32>;

constexpr int variant_count = 3;

// What runs when a shape cannot be timed: inside a graph capture, or once the
// measurement cache is full. Medium is the old ladder's answer both just above
// 512 rows and again above 32,768, so it is the least surprising untimed pick;
// the one band it loses is 2,048-32,768, which the ladder gave to Large. That
// band is where the 16,384-row chunk lands, and capture does not lose it:
// optimizer.cpp warms training up once with capture disallowed and only records
// on the second pass, so the shape is already measured when the graph is taken.
constexpr int default_variant = 1;

// Cached in place of a variant index for a shape none of the three can serve.
constexpr int declined_variant = -1;

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
bool run_variant(int variant, int rows, int contraction, int out_features,
                 const Element* input, const Element* weights, const Element* bias,
                 Element* output, cudaStream_t stream)
{
    switch (variant)
    {
    case 0:
        return NarrowGemm<Small, WarpSmall, 6, Relu>::run(
            rows, contraction, out_features, input, weights, bias, output, stream);
    case 1:
        return NarrowGemm<Medium, WarpSmall, 4, Relu>::run(
            rows, contraction, out_features, input, weights, bias, output, stream);
    default:
        return NarrowGemm<Large, WarpLarge, 3, Relu>::run(
            rows, contraction, out_features, input, weights, bias, output, stream);
    }
}

// Runs the preferred tile, and on a refusal the others in index order.
// can_implement is a host-side check that launches nothing, so the fallback is
// free whenever the preferred tile serves the shape. It exists so that a call
// that could not be timed -- inside a graph capture, or past the cache cap --
// still serves everything any of the three can serve: a decline from this path
// has to keep meaning that no variant could implement the shape.
template<bool Relu>
bool run_preferred(int preferred, int rows, int contraction, int out_features,
                   const Element* input, const Element* weights, const Element* bias,
                   Element* output, cudaStream_t stream)
{
    if (run_variant<Relu>(preferred, rows, contraction, out_features,
                          input, weights, bias, output, stream))
        return true;

    for (int variant = 0; variant < variant_count; ++variant)
        if (variant != preferred
            && run_variant<Relu>(variant, rows, contraction, out_features,
                                 input, weights, bias, output, stream))
            return true;

    return false;
}

// OPENNN_CUTLASS_NARROW_K_VARIANT pins one tile for A/B against the measured
// choice; anything outside [0, variant_count) leaves the choice measured. A
// pinned tile that cannot implement the shape declines it, exactly as the old
// ladder's single answer did.
int pinned_variant()
{
    static const int pinned = [] {
        const long long value = opennn::env_int_or("OPENNN_CUTLASS_NARROW_K_VARIANT", -1);
        return value >= 0 && value < variant_count ? int(value) : -1;
    }();
    return pinned;
}

// One entry per shape actually seen. The path is reached with a handful of row
// counts -- the batch size, the last partial batch, the 16,384-row chunk -- for
// one layer geometry, so a linear scan over a small fixed array beats a hash
// and cannot grow without bound. A shape none of the three can serve takes a
// slot as well, as a cached decline. Shapes past the cap run the default tile
// untimed rather than evicting a measurement, which keeps the cost of an
// unexpected shape at zero extra launches instead of four per call.
constexpr int max_measured_shapes = 32;

struct VariantCache
{
    struct Entry
    {
        int rows = 0;
        int contraction = 0;
        int out_features = 0;
        bool relu = false;
        int variant = default_variant;
    };

    Entry entries[max_measured_shapes];
    int count = 0;

    const Entry* find(int rows, int contraction, int out_features, bool relu) const
    {
        for (int index = 0; index < count; ++index)
        {
            const Entry& entry = entries[index];
            if (entry.rows == rows && entry.contraction == contraction
                && entry.out_features == out_features && entry.relu == relu)
                return &entry;
        }
        return nullptr;
    }

    void insert(int rows, int contraction, int out_features, bool relu, int variant)
    {
        if (count >= max_measured_shapes) return;
        entries[count] = {rows, contraction, out_features, relu, variant};
        ++count;
    }
};

// Thread-local for the same reason the cuBLASLt plan cache is: there is one GPU
// thread and lanes are streams within it, so a lock would guard nothing while
// being held across a device synchronize.
VariantCache& variant_cache()
{
    static thread_local VariantCache cache;
    return cache;
}

// Timing needs an event synchronize, which a stream under capture cannot take.
// A non-success return is treated as capturing too: cudaStreamIsCapturing on
// the legacy stream reports an error rather than a status while a capture is
// live in the same thread.
bool timing_allowed(cudaStream_t stream)
{
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture_status) != cudaSuccess
        || capture_status != cudaStreamCaptureStatusNone)
    {
        cudaGetLastError();
        return false;
    }
    return true;
}

struct TimingEvent
{
    TimingEvent()
    {
        if (cudaEventCreate(&event) != cudaSuccess)
        {
            cudaGetLastError();
            event = nullptr;
        }
    }

    ~TimingEvent() { if (event) cudaEventDestroy(event); }

    TimingEvent(const TimingEvent&) = delete;
    TimingEvent& operator=(const TimingEvent&) = delete;

    cudaEvent_t event = nullptr;
};

struct Measurement
{
    int variant = default_variant;
    bool served = false;
};

// Times every variant that can implement the shape and returns the fastest.
//
// The launches go on the caller's own pointers with no scratch, because the
// product is idempotent: alpha is 1, beta multiplies the bias operand held at
// stride 0, and the destination is never also a source, so each launch leaves
// the same bytes behind. A variant that cannot implement the shape is skipped
// rather than counted, and `served` stays false only when none of the three
// ran, which is the one case that must reach the caller as a decline.
//
// Concurrent lanes are not fenced off first, as the cuBLASLt tuner does before
// its timed runs: device::lanes_available() and device::synchronize() are not
// among the handful of device entry points kernel_common.cuh declares, and this
// file includes no more than that. A perturbed measurement picks a slower tile,
// never a different result, so the gap is one of measurement quality.
template<bool Relu>
Measurement measure(int rows, int contraction, int out_features,
                    const Element* input, const Element* weights, const Element* bias,
                    Element* output, cudaStream_t stream)
{
    constexpr int timed_runs = 3;

    Measurement measurement;
    const TimingEvent start, stop;

    float best_ms = std::numeric_limits<float>::infinity();
    for (int variant = 0; variant < variant_count; ++variant)
    {
        const auto run = [&] {
            return run_variant<Relu>(variant, rows, contraction, out_features,
                                     input, weights, bias, output, stream);
        };

        if (!run()) continue;

        if (!measurement.served)
        {
            measurement.variant = variant;
            measurement.served = true;
        }

        if (!start.event || !stop.event) continue;

        bool ok = cudaEventRecord(start.event, stream) == cudaSuccess;
        for (int run_index = 0; run_index < timed_runs && ok; ++run_index) ok = run();
        ok = ok && cudaEventRecord(stop.event, stream) == cudaSuccess
                && cudaEventSynchronize(stop.event) == cudaSuccess;

        float milliseconds = 0.0f;
        ok = ok && cudaEventElapsedTime(&milliseconds, start.event, stop.event) == cudaSuccess;
        if (!ok) { cudaGetLastError(); continue; }

        if (milliseconds < best_ms)
        {
            best_ms = milliseconds;
            measurement.variant = variant;
        }
    }

    return measurement;
}

template<bool Relu>
bool dispatch(int rows, int contraction, int out_features,
              const Element* input, const Element* weights, const Element* bias,
              Element* output, cudaStream_t stream)
{
    const int pinned = pinned_variant();
    if (pinned >= 0)
        return run_variant<Relu>(pinned, rows, contraction, out_features,
                                 input, weights, bias, output, stream);

    int variant = default_variant;

    VariantCache& cache = variant_cache();
    if (const VariantCache::Entry* const entry = cache.find(rows, contraction, out_features, Relu))
    {
        if (entry->variant == declined_variant) return false;
        variant = entry->variant;
    }
    else if (cache.count < max_measured_shapes && timing_allowed(stream))
    {
        const Measurement measurement = measure<Relu>(rows, contraction, out_features,
                                                      input, weights, bias, output, stream);

        // The decline is cached like a pick. can_implement is a pure function of
        // the problem size and the alignments, and the three variants carry the
        // same AlignmentA, AlignmentB and epilogue vector width while the caller
        // has already fixed all four pointers at 16 bytes, so the answer cannot
        // change between calls for one key. Leaving it uncached would make a
        // declined shape re-enter measure() on every forward, paying a
        // cudaStreamIsCapturing and three can_implement checks where the old
        // ladder paid one.
        //
        // A pick is cached even when the events failed and it is only the first
        // variant that ran, so a shape is probed once rather than on every call.
        cache.insert(rows, contraction, out_features, Relu,
                     measurement.served ? measurement.variant : declined_variant);

        if (!measurement.served) return false;

        variant = measurement.variant;
    }

    return run_preferred<Relu>(variant, rows, contraction, out_features,
                               input, weights, bias, output, stream);
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
