// Pooling probe: how much of the ResNet-50 step the max-pool costs, and how far
// cuDNN's pooling kernels sit from the memory roofline on this GPU - the two
// numbers that decide whether an own pooling backward (argmax mask from the
// forward, one read + one scatter) is worth writing.
//
// Times cudnnPoolingForward / cudnnPoolingBackward (NHWC, 3x3 stride 2 pad 1,
// the ResNet stem pool) at the CIFAR and ImageNet stem shapes for BF16 and
// FP32, against a device-to-device copy of the same byte count as the
// bandwidth reference. Standalone: raw cuDNN, no library code.
//
//   pooling_probe [batch]      (default 2048 for CIFAR, batch/8 for ImageNet)

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define CHECK_CUDNN(call) do { cudnnStatus_t s = (call); if (s != CUDNN_STATUS_SUCCESS) { \
    fprintf(stderr, "cuDNN %s at %s:%d\n", cudnnGetErrorString(s), __FILE__, __LINE__); exit(1); } } while (0)

namespace
{

struct Shape4 { int n, h, w, c; size_t elements() const { return size_t(n) * h * w * c; } };

float time_ms(cudaStream_t stream, int iterations, const auto& fn)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    for (int i = 0; i < 3; ++i) fn();               // warm-up
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) fn();
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / iterations;
}

void probe(cudnnHandle_t handle, cudaStream_t stream, const char* label,
           const Shape4& in, cudnnDataType_t dtype, size_t element_bytes)
{
    const Shape4 out{in.n, (in.h + 2 - 3) / 2 + 1, (in.w + 2 - 3) / 2 + 1, in.c};

    cudnnTensorDescriptor_t x_desc, y_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NHWC, dtype, in.n, in.c, in.h, in.w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NHWC, dtype, out.n, out.c, out.h, out.w));

    cudnnPoolingDescriptor_t pool;
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pool));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pool, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 3, 3, 1, 1, 2, 2));

    const size_t x_bytes = in.elements() * element_bytes;
    const size_t y_bytes = out.elements() * element_bytes;
    void *x, *y, *dx, *dy, *scratch;
    CHECK_CUDA(cudaMalloc(&x, x_bytes));
    CHECK_CUDA(cudaMalloc(&y, y_bytes));
    CHECK_CUDA(cudaMalloc(&dx, x_bytes));
    CHECK_CUDA(cudaMalloc(&dy, y_bytes));
    CHECK_CUDA(cudaMalloc(&scratch, x_bytes));
    CHECK_CUDA(cudaMemset(x, 0x3c, x_bytes));       // some positive pattern
    CHECK_CUDA(cudaMemset(dy, 0x3c, y_bytes));

    const float one = 1.0f, zero = 0.0f;
    const int iterations = 20;

    const float fwd_ms = time_ms(stream, iterations, [&]
    {
        CHECK_CUDNN(cudnnPoolingForward(handle, pool, &one, x_desc, x, &zero, y_desc, y));
    });
    const float bwd_ms = time_ms(stream, iterations, [&]
    {
        CHECK_CUDNN(cudnnPoolingBackward(handle, pool, &one, y_desc, y, y_desc, dy, x_desc, x, &zero, x_desc, dx));
    });
    // Bandwidth reference: a copy moving x_bytes each way (read + write).
    const float copy_ms = time_ms(stream, iterations, [&]
    {
        CHECK_CUDA(cudaMemcpyAsync(scratch, x, x_bytes, cudaMemcpyDeviceToDevice, stream));
    });

    // Minimum traffic: forward reads X once and writes Y; backward reads dY and
    // an argmax mask (1 byte per output) and writes dX. cuDNN's backward also
    // reads X and Y to recompute the argmax.
    const double fwd_min = double(x_bytes + y_bytes);
    const double bwd_min = double(y_bytes + out.elements() + x_bytes);
    const double bwd_cudnn = double(x_bytes + 2 * y_bytes + x_bytes);
    const double copy_gbs = 2.0 * x_bytes / (copy_ms * 1e6);

    printf("%-22s N=%d %dx%dx%d -> %dx%dx%d  fwd %.3f ms (%.0f GB/s of min traffic; copy %.0f GB/s)  "
           "bwd %.3f ms (%.0f GB/s over cuDNN's traffic; %.0f GB/s over mask-path traffic; mask-path at copy speed %.3f ms)\n",
           label, in.n, in.h, in.w, in.c, out.h, out.w, out.c,
           fwd_ms, fwd_min / (fwd_ms * 1e6), copy_gbs,
           bwd_ms, bwd_cudnn / (bwd_ms * 1e6), bwd_min / (bwd_ms * 1e6), bwd_min / (copy_gbs * 1e6));

    CHECK_CUDA(cudaFree(x)); CHECK_CUDA(cudaFree(y)); CHECK_CUDA(cudaFree(dx)); CHECK_CUDA(cudaFree(dy)); CHECK_CUDA(cudaFree(scratch));
    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(pool));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
}

}

int main(int argc, char** argv)
{
    const int batch = argc > 1 ? atoi(argv[1]) : 2048;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    printf("device=%s cudnn=%zu\n", properties.name, cudnnGetVersion());

    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDNN(cudnnSetStream(handle, stream));

    // ResNet-50 stem pool: 3x3 stride 2 pad 1 on the stem conv output.
    probe(handle, stream, "cifar  bf16", {batch, 16, 16, 64}, CUDNN_DATA_BFLOAT16, 2);
    probe(handle, stream, "cifar  fp32", {batch, 16, 16, 64}, CUDNN_DATA_FLOAT, 4);
    probe(handle, stream, "imagenet bf16", {batch / 8, 112, 112, 64}, CUDNN_DATA_BFLOAT16, 2);
    probe(handle, stream, "imagenet fp32", {batch / 8, 112, 112, 64}, CUDNN_DATA_FLOAT, 4);

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDNN(cudnnDestroy(handle));
    return 0;
}
