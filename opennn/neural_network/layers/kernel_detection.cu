//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// YOLO detection heads, anchor-based and anchor-free

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/neural_network/layers/kernel_detection.cuh"

__global__ void detection_forward_kernel(const int batch_size,
                                         const int grid_size,
                                         const int boxes_per_cell,
                                         const int classes_number,
                                         const int channels,
                                         const int class_activation,
                                         const float* __restrict__ anchors,
                                         const float* __restrict__ src,
                                         float* __restrict__ dst)
{
    const int values_per_box = 5 + classes_number;
    const int total = batch_size * grid_size * grid_size * boxes_per_cell;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += Index(blockDim.x) * gridDim.x)
    {
        const int box = idx % boxes_per_cell;
        const int t   = idx / boxes_per_cell;
        const int col = t % grid_size;
        const int t2  = t / grid_size;
        const int row = t2 % grid_size;
        const int b   = t2 / grid_size;

        const int cell = ((b * grid_size + row) * grid_size + col) * channels;
        const int base = cell + box * values_per_box;

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
        {
            float max_logit = src[base + 5];
            for (int c = 1; c < classes_number; ++c)
            {
                const float v = src[base + 5 + c];
                if (v > max_logit) max_logit = v;
            }
            float sum = 0.0f;
            for (int c = 0; c < classes_number; ++c)
            {
                const float e = __expf(src[base + 5 + c] - max_logit);
                dst[base + 5 + c] = e;
                sum += e;
            }
            const float inv_sum = 1.0f / (sum + 1e-7f);
            for (int c = 0; c < classes_number; ++c)
                dst[base + 5 + c] *= inv_sum;
        }
    }
}

void detection_forward_cuda(const Index batch_size,
                            const Index grid_size,
                            const Index boxes_per_cell,
                            const Index classes_number,
                            const Index channels,
                            const int class_activation,
                            const float* anchors,
                            const float* input,
                            float* output)
{
    if (batch_size == 0 || grid_size == 0 || boxes_per_cell == 0) return;

    const int total = checked_int(batch_size * grid_size * grid_size * boxes_per_cell);
    OPENNN_CUDA_LAUNCH(detection_forward_kernel<<<grid_size_strided_for(total), block_size, 0,
                               opennn::device::get_compute_stream()>>>(
        checked_int(batch_size),
        checked_int(grid_size),
        checked_int(boxes_per_cell),
        checked_int(classes_number),
        checked_int(channels),
        class_activation,
        anchors, input, output));
}

__global__ void detection_backward_kernel(const int batch_size,
                                          const int grid_size,
                                          const int boxes_per_cell,
                                          const int classes_number,
                                          const int channels,
                                          const int class_activation,
                                          const float* __restrict__ out,
                                          const float* __restrict__ delta,
                                          float* __restrict__ in_delta)
{
    const int values_per_box = 5 + classes_number;
    const int total = batch_size * grid_size * grid_size * boxes_per_cell;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += Index(blockDim.x) * gridDim.x)
    {
        const int box = idx % boxes_per_cell;
        const int t   = idx / boxes_per_cell;
        const int col = t % grid_size;
        const int t2  = t / grid_size;
        const int row = t2 % grid_size;
        const int b   = t2 / grid_size;

        const int cell = ((b * grid_size + row) * grid_size + col) * channels;
        const int base = cell + box * values_per_box;

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
        {
            float dot = 0.0f;
            for (int c = 0; c < classes_number; ++c)
                dot += delta[base + 5 + c] * out[base + 5 + c];

            for (int c = 0; c < classes_number; ++c)
            {
                const float s = out[base + 5 + c];
                in_delta[base + 5 + c] = s * (delta[base + 5 + c] - dot);
            }
        }
    }
}

void detection_backward_cuda(const Index batch_size,
                             const Index grid_size,
                             const Index boxes_per_cell,
                             const Index classes_number,
                             const Index channels,
                             const int class_activation,
                             const float* output,
                             const float* output_delta,
                             float* input_delta)
{
    if (batch_size == 0 || grid_size == 0 || boxes_per_cell == 0) return;

    const int total = checked_int(batch_size * grid_size * grid_size * boxes_per_cell);
    OPENNN_CUDA_LAUNCH(detection_backward_kernel<<<grid_size_strided_for(total), block_size, 0,
                                opennn::device::get_compute_stream()>>>(
        checked_int(batch_size),
        checked_int(grid_size),
        checked_int(boxes_per_cell),
        checked_int(classes_number),
        checked_int(channels),
        class_activation,
        output, output_delta, input_delta));
}

__global__ void detection_v8_forward_kernel(const int batch_size,
                                            const int grid_size,
                                            const int grid_width,
                                            const int channels,
                                            const int box_ch,
                                            const float* __restrict__ src,
                                            float* __restrict__ dst)
{
    const int total = batch_size * grid_size * grid_width;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += Index(blockDim.x) * gridDim.x)
    {
        const int col = idx % grid_width;
        const int t   = idx / grid_width;
        const int row = t % grid_size;
        const int b   = t / grid_size;

        const int base = ((b * grid_size + row) * grid_width + col) * channels;

        for (int ch = 0; ch < box_ch; ++ch)
            dst[base + ch] = (box_ch == 4) ? sigmoid_f(src[base + ch]) : src[base + ch];
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
    if (batch_size == 0 || grid_size == 0) return;

    const int box_ch   = checked_int(4 * max(reg_max, Index(1)));
    const int total    = checked_int(batch_size * grid_size * grid_width);
    const int channels = checked_int(box_ch + classes_number);
    OPENNN_CUDA_LAUNCH(detection_v8_forward_kernel<<<grid_size_for(total), block_size, 0,
                               opennn::device::get_compute_stream()>>>(
        checked_int(batch_size), checked_int(grid_size), checked_int(grid_width),
        channels, box_ch, input, output));
}

__global__ void detection_v8_backward_kernel(const int batch_size,
                                             const int grid_size,
                                             const int grid_width,
                                             const int channels,
                                             const int box_ch,
                                             const float* __restrict__ out,
                                             const float* __restrict__ delta,
                                             float* __restrict__ in_delta)
{
    const int total = batch_size * grid_size * grid_width;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += Index(blockDim.x) * gridDim.x)
    {
        const int col = idx % grid_width;
        const int t   = idx / grid_width;
        const int row = t % grid_size;
        const int b   = t / grid_size;

        const int base = ((b * grid_size + row) * grid_width + col) * channels;

        for (int ch = 0; ch < box_ch; ++ch)
        {
            if (box_ch == 4)
            {
                const float s = out[base + ch];
                in_delta[base + ch] = delta[base + ch] * s * (1.0f - s);
            }
            else
            {
                in_delta[base + ch] = delta[base + ch];
            }
        }
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
    if (batch_size == 0 || grid_size == 0) return;

    const int box_ch   = checked_int(4 * max(reg_max, Index(1)));
    const int total    = checked_int(batch_size * grid_size * grid_width);
    const int channels = checked_int(box_ch + classes_number);
    OPENNN_CUDA_LAUNCH(detection_v8_backward_kernel<<<grid_size_for(total), block_size, 0,
                                opennn::device::get_compute_stream()>>>(
        checked_int(batch_size), checked_int(grid_size), checked_int(grid_width),
        channels, box_ch, output, output_delta, input_delta));
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
