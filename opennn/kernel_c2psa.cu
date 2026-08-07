#include "kernel_common.cuh"
#include "device_backend.h"




template<typename T>
__global__ void c2psa_split_kernel(
    const int n,
    const T* __restrict__ x,
    T* __restrict__ xa,
    T* __restrict__ cat,
    int C, int H)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int row = i / H;
        const int col = i % H;
        xa[i]                  = x[row * C + col];
        cat[row * C + H + col] = x[row * C + H + col];
    }
}


template<typename T>
__global__ void c2psa_fill_cat_left_kernel(
    const int n,
    const T* __restrict__ attn_v,
    T* __restrict__ cat,
    int C, int H)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int row = i / H;
        const int col = i % H;
        cat[row * C + col] = attn_v[i];
    }
}



template<typename T>
__global__ void c2psa_row_softmax_kernel(const int rows, T* __restrict__ A, int T_sz)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    T* p = A + row * T_sz;

    float maxv = static_cast<float>(p[0]);
    for (int j = 1; j < T_sz; ++j) maxv = fmaxf(maxv, static_cast<float>(p[j]));

    float sum = 0.f;
    for (int j = 0; j < T_sz; ++j)
    {
        const float v = expf(static_cast<float>(p[j]) - maxv);
        p[j] = static_cast<T>(v);
        sum += v;
    }
    const float inv = 1.f / sum;
    for (int j = 0; j < T_sz; ++j) p[j] = static_cast<T>(static_cast<float>(p[j]) * inv);
}



template<typename T>
__global__ void c2psa_softmax_bwd_kernel(
    const int rows,
    const T* __restrict__ A,
    T* __restrict__ dA,
    float scale,
    int T_sz)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const T* Ap  = A  + row * T_sz;
    T*       dAp = dA + row * T_sz;

    float dot = 0.f;
    for (int j = 0; j < T_sz; ++j) dot += static_cast<float>(Ap[j]) * static_cast<float>(dAp[j]);
    for (int j = 0; j < T_sz; ++j)
        dAp[j] = static_cast<T>(static_cast<float>(Ap[j]) * (static_cast<float>(dAp[j]) - dot) * scale);
}




template<typename T>
__global__ void c2psa_scatter_dx_kernel(
    const int n,
    const T* __restrict__ d_xa,
    const T* __restrict__ d_cat,
    T* __restrict__ din,
    int C, int H)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int row = i / C;
        const int col = i % C;
        din[i] = (col < H) ? d_xa[row * H + col] : d_cat[i];
    }
}

template<typename F>
static void c2psa_dispatch(cudaDataType_t dtype, F&& f)
{
    if (dtype == CUDA_R_32F) f(float{});
    else                     f(__nv_bfloat16{});
}

void c2psa_split_cuda(
    const void* x, void* xa, void* cat,
    int BT, int C, int H, cudaDataType_t dtype)
{
    c2psa_dispatch(dtype, [&](auto tag) {
        using T = decltype(tag);
        launch_elementwise_strided(Index(BT) * H, c2psa_split_kernel<T>,
            (const T*)x, (T*)xa, (T*)cat, C, H);
    });
}

void c2psa_fill_cat_left_cuda(
    const void* attn_v, void* cat,
    int BT, int C, int H, cudaDataType_t dtype)
{
    c2psa_dispatch(dtype, [&](auto tag) {
        using T = decltype(tag);
        launch_elementwise_strided(Index(BT) * H, c2psa_fill_cat_left_kernel<T>,
            (const T*)attn_v, (T*)cat, C, H);
    });
}

void c2psa_row_softmax_cuda(void* A, int rows, int T_sz, cudaDataType_t dtype)
{
    c2psa_dispatch(dtype, [&](auto tag) {
        using T = decltype(tag);
        launch_elementwise(Index(rows), c2psa_row_softmax_kernel<T>, (T*)A, T_sz);
    });
}

void c2psa_softmax_bwd_cuda(const void* A, void* dA, float scale, int rows, int T_sz, cudaDataType_t dtype)
{
    c2psa_dispatch(dtype, [&](auto tag) {
        using T = decltype(tag);
        launch_elementwise(Index(rows), c2psa_softmax_bwd_kernel<T>, (const T*)A, (T*)dA, scale, T_sz);
    });
}

void c2psa_scatter_dx_cuda(
    const void* d_xa, const void* d_cat, void* din,
    int BT, int C, int H, cudaDataType_t dtype)
{
    c2psa_dispatch(dtype, [&](auto tag) {
        using T = decltype(tag);
        launch_elementwise_strided(Index(BT) * C, c2psa_scatter_dx_kernel<T>,
            (const T*)d_xa, (const T*)d_cat, (T*)din, C, H);
    });
}
