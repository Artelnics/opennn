// Stand-in for ATen's unpack of a philox state; the whole of it lives in
// ../CUDAGeneratorImpl.h, which some FlashAttention-2 sources reach through
// this path instead.

#pragma once

#include <ATen/cuda/CUDAGeneratorImpl.h>
