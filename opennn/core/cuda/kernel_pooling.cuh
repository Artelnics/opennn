#ifndef KERNEL_POOLING_CUH
#define KERNEL_POOLING_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

struct MaxPoolGeometry
{
    Index batch;
    int height, width, channels;
    int out_height, out_width;
    int pool_height, pool_width;
    int stride_h, stride_w;
    int pad_h, pad_w;

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
