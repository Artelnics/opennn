// Stand-in for the two ATen entities FlashAttention-2 touches, so its kernels
// build without PyTorch. Only the non-captured path is provided, which is all a
// dropout-free call needs: the kernels are compiled with
// FLASHATTENTION_DISABLE_DROPOUT, so nothing ever unpacks a live philox state.

#pragma once

#include <cstdint>
#include <tuple>
#include <cuda_runtime.h>

namespace at {

struct PhiloxCudaState {
    PhiloxCudaState() = default;
    PhiloxCudaState(uint64_t seed, uint64_t offset) { seed_.val = seed; offset_.val = offset; }

    struct SeedOrPointer { uint64_t val = 0; const int64_t* ptr = nullptr; };

    SeedOrPointer seed_{};
    SeedOrPointer offset_{};
    uint32_t offset_intragraph_ = 0;
    bool captured_ = false;
};

namespace cuda { namespace philox {

__device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(const PhiloxCudaState& arg)
{
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
}

}}  // namespace cuda::philox
}   // namespace at
