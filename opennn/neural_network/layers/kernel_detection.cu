//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/neural_network/layers/kernel_detection.cuh"

static constexpr int class_activation_sigmoid = 1;

__device__ __forceinline__ int detection_box_base(const Index idx, const int grid_size, const int boxes_per_cell,
                                                  const int values_per_box, int& box)
{
    box = idx % boxes_per_cell;
    const int t   = idx / boxes_per_cell;
    const int col = t % grid_size;
    const int t2  = t / grid_size;
    const int row = t2 % grid_size;
    const int b   = t2 / grid_size;

    const int channels = boxes_per_cell * values_per_box;
    const int cell = ((b * grid_size + row) * grid_size + col) * channels;
    return cell + box * values_per_box;
}

__device__ __forceinline__ int detection_v8_cell_base(const Index idx, const int grid_size, const int grid_width,
                                                      const int channels)
{
    const int col = idx % grid_width;
    const int t   = idx / grid_width;
    const int row = t % grid_size;
    const int b   = t / grid_size;

    return ((b * grid_size + row) * grid_width + col) * channels;
}

__global__ void detection_forward_kernel(const int n,
                                         const int grid_size,
                                         const int boxes_per_cell,
                                         const int classes_number,
                                         const int class_activation,
                                         const float* __restrict__ anchors,
                                         const float* __restrict__ src,
                                         float* __restrict__ dst)
{
    const int values_per_box = 5 + classes_number;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n;
         idx += Index(blockDim.x) * gridDim.x)
    {
        int box;
        const int base = detection_box_base(idx, grid_size, boxes_per_cell, values_per_box, box);

        const float aw = anchors[box * 2 + 0];
        const float ah = anchors[box * 2 + 1];

        dst[base + 0] = sigmoid_f(src[base + 0]);
        dst[base + 1] = sigmoid_f(src[base + 1]);
        dst[base + 2] = __expf(fminf(fmaxf(src[base + 2], -4.0f), 4.0f)) * aw;
        dst[base + 3] = __expf(fminf(fmaxf(src[base + 3], -4.0f), 4.0f)) * ah;
        dst[base + 4] = sigmoid_f(src[base + 4]);

        if (class_activation == class_activation_sigmoid)
        {
            for (int c = 0; c < classes_number; ++c)
                dst[base + 5 + c] = sigmoid_f(src[base + 5 + c]);
        }
        else
            row_softmax<float, true>(src + base + 5, dst + base + 5, classes_number, 1e-7f);
    }
}

void detection_forward_cuda(const Index batch_size,
                            const Index grid_size,
                            const Index boxes_per_cell,
                            const Index classes_number,
                            const int class_activation,
                            const float* anchors,
                            const float* input,
                            float* output)
{
    launch_elementwise_strided(batch_size * grid_size * grid_size * boxes_per_cell, detection_forward_kernel,
                               checked_int(grid_size), checked_int(boxes_per_cell), checked_int(classes_number),
                               class_activation, anchors, input, output);
}

__global__ void detection_backward_kernel(const int n,
                                          const int grid_size,
                                          const int boxes_per_cell,
                                          const int classes_number,
                                          const int class_activation,
                                          const float* __restrict__ out,
                                          const float* __restrict__ delta,
                                          float* __restrict__ in_delta)
{
    const int values_per_box = 5 + classes_number;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n;
         idx += Index(blockDim.x) * gridDim.x)
    {
        int box;
        const int base = detection_box_base(idx, grid_size, boxes_per_cell, values_per_box, box);

        const float ox = out[base + 0];
        const float oy = out[base + 1];
        const float oo = out[base + 4];

        in_delta[base + 0] = delta[base + 0] * ox * (1.0f - ox);
        in_delta[base + 1] = delta[base + 1] * oy * (1.0f - oy);

        in_delta[base + 2] = delta[base + 2] * out[base + 2];
        in_delta[base + 3] = delta[base + 3] * out[base + 3];
        in_delta[base + 4] = delta[base + 4] * oo * (1.0f - oo);

        if (class_activation == class_activation_sigmoid)
        {
            for (int c = 0; c < classes_number; ++c)
            {
                const float s = out[base + 5 + c];
                in_delta[base + 5 + c] = delta[base + 5 + c] * s * (1.0f - s);
            }
        }
        else
            row_softmax_backward<float>(out + base + 5, delta + base + 5, in_delta + base + 5, classes_number, 1.0f);
    }
}

void detection_backward_cuda(const Index batch_size,
                             const Index grid_size,
                             const Index boxes_per_cell,
                             const Index classes_number,
                             const int class_activation,
                             const float* output,
                             const float* output_delta,
                             float* input_delta)
{
    launch_elementwise_strided(batch_size * grid_size * grid_size * boxes_per_cell, detection_backward_kernel,
                               checked_int(grid_size), checked_int(boxes_per_cell), checked_int(classes_number),
                               class_activation, output, output_delta, input_delta);
}

static void detection_v8_channels(const Index classes_number, const Index reg_max, int& box_ch, int& channels)
{
    box_ch   = checked_int(4 * max(reg_max, Index(1)));
    channels = checked_int(box_ch + classes_number);
}

__global__ void detection_v8_forward_kernel(const int n,
                                            const int grid_size,
                                            const int grid_width,
                                            const int channels,
                                            const int box_ch,
                                            const float* __restrict__ src,
                                            float* __restrict__ dst)
{
    const bool sigmoid_box = box_ch == 4;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n;
         idx += Index(blockDim.x) * gridDim.x)
    {
        const int base = detection_v8_cell_base(idx, grid_size, grid_width, channels);

        if (sigmoid_box)
            for (int ch = 0; ch < box_ch; ++ch)
                dst[base + ch] = sigmoid_f(src[base + ch]);
        else
            for (int ch = 0; ch < box_ch; ++ch)
                dst[base + ch] = src[base + ch];
        for (int ch = box_ch; ch < channels; ++ch)
            dst[base + ch] = sigmoid_f(src[base + ch]);
    }
}

void detection_v8_forward_cuda(const Index batch_size,
                               const Index grid_size,
                               const Index grid_width,
                               const Index classes_number,
                               const Index reg_max,
                               const float* input,
                               float* output)
{
    int box_ch, channels;
    detection_v8_channels(classes_number, reg_max, box_ch, channels);
    launch_elementwise_strided(batch_size * grid_size * grid_width, detection_v8_forward_kernel,
                               checked_int(grid_size), checked_int(grid_width), channels, box_ch, input, output);
}

__global__ void detection_v8_backward_kernel(const int n,
                                             const int grid_size,
                                             const int grid_width,
                                             const int channels,
                                             const int box_ch,
                                             const float* __restrict__ out,
                                             const float* __restrict__ delta,
                                             float* __restrict__ in_delta)
{
    const bool sigmoid_box = box_ch == 4;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n;
         idx += Index(blockDim.x) * gridDim.x)
    {
        const int base = detection_v8_cell_base(idx, grid_size, grid_width, channels);

        if (sigmoid_box)
            for (int ch = 0; ch < box_ch; ++ch)
            {
                const float s = out[base + ch];
                in_delta[base + ch] = delta[base + ch] * s * (1.0f - s);
            }
        else
            for (int ch = 0; ch < box_ch; ++ch)
                in_delta[base + ch] = delta[base + ch];
        for (int ch = box_ch; ch < channels; ++ch)
        {
            const float s = out[base + ch];
            in_delta[base + ch] = delta[base + ch] * s * (1.0f - s);
        }
    }
}

void detection_v8_backward_cuda(const Index batch_size,
                                const Index grid_size,
                                const Index grid_width,
                                const Index classes_number,
                                const Index reg_max,
                                const float* output,
                                const float* output_delta,
                                float* input_delta)
{
    int box_ch, channels;
    detection_v8_channels(classes_number, reg_max, box_ch, channels);
    launch_elementwise_strided(batch_size * grid_size * grid_width, detection_v8_backward_kernel,
                               checked_int(grid_size), checked_int(grid_width), channels, box_ch,
                               output, output_delta, input_delta);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
