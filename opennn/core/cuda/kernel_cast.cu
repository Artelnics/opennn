//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D T Y P E   C A S T   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_cast.cuh"

// Widening or narrowing cast, four elements per thread (one 16-byte or 8-byte
// load, one 8-byte or 16-byte store) over the aligned prefix and scalar for
// the tail. In BF16 training this is the per-step widening of every
// convolution weight gradient cuDNN stores in BF16.
template<typename Src, typename Dst>
__global__ void cast_kernel(const int n_vec, const int n,
                            const Src* __restrict__ src, Dst* __restrict__ dst)
{
    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;

    float v[4];
    for (Index i = tid; i < n_vec; i += stride)
    {
        VecIO<Src, 4>::load_float(src + i * 4, v);
        VecIO<Dst, 4>::store_float(dst + i * 4, v);
    }
    for (Index i = Index(n_vec) * 4 + tid; i < n; i += stride)
        element_from_float(element_to_float(src[i]), dst[i]);
}

template<typename Src, typename Dst>
void cast_cuda(const Index n, const Src* src, Dst* dst, cudaStream_t stream)
{
    if (stream == nullptr) stream = opennn::device::get_compute_stream();
    launch_vec_on<4>(stream, n, is_aligned<4 * sizeof(Src)>(src) && is_aligned<4 * sizeof(Dst)>(dst),
                   cast_kernel<Src, Dst>, src, dst);
}

void cast_fp32_to_bf16(const Index n, const float* src, __nv_bfloat16* dst, cudaStream_t stream)
{
    cast_cuda(n, src, dst, stream);
}

void cast_bf16_to_fp32(const Index n, const __nv_bfloat16* src, float* dst)
{
    cast_cuda(n, src, dst, nullptr);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
