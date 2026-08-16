#ifndef KERNEL_POOLING_CUH
#define KERNEL_POOLING_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

// Max pooling on NHWC activations with an argmax mask: the forward writes,
// beside Y, one byte per output element holding the window position of the
// maximum (row-major, pool_height * pool_width <= 255). The backward is then a
// gather: each input element visits the few outputs whose window covers it and
// sums the dY of those whose argmax is this element - one read of the small dY
// and mask tensors, one write of dX, no atomics and no zero fill, and neither
// X nor Y is read again. cuDNN's pooling backward re-derives the argmax from X
// and Y and measured 3-5x slower than that on an RTX 3060 (pooling_probe).
// Padded positions are excluded from the maximum, as in cuDNN and the CPU
// path; ties go to the first window position, as in the CPU path.
// `mask` may be null in the forward (inference: nothing to save).

struct MaxPoolGeometry
{
    Index batch;
    int height, width, channels;
    int out_height, out_width;
    int pool_height, pool_width;
    int stride_h, stride_w;
    int pad_h, pad_w;

    // Thread index -> (n, row, column, first channel of a VEC group); channel
    // groups are the fastest index, so a warp touches consecutive channels of
    // one pixel.
    __host__ __device__ void decompose(Index gi, int rows, int columns, int vec,
                                       Index& n, int& row, int& column, int& c0) const
    {
        const int channel_groups = channels / vec;
        c0 = int(gi % channel_groups) * vec;
        Index rest = gi / channel_groups;
        column = int(rest % columns); rest /= columns;
        row = int(rest % rows);
        n = rest / rows;
    }
    __host__ __device__ Index input_offset(Index n, int row, int column, int c0) const
    {
        return ((n * height + row) * Index(width) + column) * channels + c0;
    }
    __host__ __device__ Index output_offset(Index n, int row, int column, int c0) const
    {
        return ((n * out_height + row) * Index(out_width) + column) * channels + c0;
    }
};

template<typename T>
void max_pooling_forward_cuda(const T* x, T* y, uint8_t* mask, const MaxPoolGeometry&);

template<typename T>
void max_pooling_backward_cuda(const T* dy, const uint8_t* mask, T* dx, const MaxPoolGeometry&);

#endif

#endif
