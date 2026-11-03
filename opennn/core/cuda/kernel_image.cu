// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_image.cuh"

namespace opennn
{
namespace
{

__device__ float byte_value(float value)
{
    return float(uint8_t(fminf(255.0f, fmaxf(0.0f, value + 0.5f))));
}

__device__ float blend(float a, float b, float c, float d, float x, float y)
{
    const float top = __fadd_rn(__fmul_rn(a, 1.0f - x), __fmul_rn(b, x));
    const float bottom = __fadd_rn(__fmul_rn(c, 1.0f - x), __fmul_rn(d, x));
    return __fadd_rn(__fmul_rn(top, 1.0f - y), __fmul_rn(bottom, y));
}

__device__ float hue_channel(float hue, float saturation, float value, float phase)
{
    const float wave = fabsf(fmodf(hue * 6.0f + phase, 6.0f) - 3.0f) - 1.0f;
    return byte_value(value * (1.0f - saturation * (1.0f - fminf(1.0f, fmaxf(0.0f, wave)))) * 255.0f);
}

__device__ float3 color_pixel(float3 pixel, ImageColorTransform color)
{
    const float r = pixel.x / 255.0f, g = pixel.y / 255.0f, b = pixel.z / 255.0f;
    const float value = fmaxf(r, fmaxf(g, b));
    const float difference = value - fminf(r, fminf(g, b));
    float hue = 0.0f;
    if (difference > 1e-6f)
    {
        hue = value == r ? (g - b) / difference
            : value == g ? (b - r) / difference + 2.0f : (r - g) / difference + 4.0f;
        hue /= 6.0f;
        if (hue < 0.0f) hue += 1.0f;
    }
    hue += color.hue;
    hue -= floorf(hue);
    const float saturation = fminf(1.0f, fmaxf(0.0f, (value > 0.0f ? difference / value : 0.0f) * color.saturation));
    const float exposure = fminf(1.0f, fmaxf(0.0f, value * color.exposure));
    return {hue_channel(hue, saturation, exposure, 0.0f),
            hue_channel(hue, saturation, exposure, 4.0f),
            hue_channel(hue, saturation, exposure, 2.0f)};
}

__device__ float3 read_color(const uint8_t* input, Index pixel, Index channels, ImageColorTransform color)
{
    const uint8_t* source = input + pixel * channels;
    return color_pixel({float(source[0]), float(source[1]), float(source[2])}, color);
}

template<typename Input, typename Output>
__global__ void affine_kernel(const Input* input, Output* output, const ImageTransformParameters* parameters,
                               const ImageChannelScale* scales, ImageTransformBatch batch, Index count)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += Index(blockDim.x) * gridDim.x)
    {
        const Index c = i % batch.channels;
        const Index pixel = i / batch.channels;
        const Index sample = pixel / (batch.height * batch.width);
        const auto& transform = parameters[sample].affine;
        Index x = pixel % batch.width - transform.shift_x;
        Index y = (pixel / batch.width) % batch.height - transform.shift_y;
        float value = 0.0f;
        if (x >= 0 && x < batch.width && y >= 0 && y < batch.height)
        {
            const float cx = float(batch.width) / 2.0f, cy = float(batch.height) / 2.0f;
            const float sx = __fadd_rn(__fsub_rn(__fmul_rn(transform.cosine, float(x) - cx),
                                                __fmul_rn(transform.sine, float(y) - cy)), cx);
            const float sy = __fadd_rn(__fadd_rn(__fmul_rn(transform.sine, float(x) - cx),
                                                __fmul_rn(transform.cosine, float(y) - cy)), cy);
            if (sx >= 0.0f && sx < float(batch.width) && sy >= 0.0f && sy < float(batch.height))
            {
                x = transform.flip_x ? batch.width - 1 - Index(sx) : Index(sx);
                y = transform.flip_y ? batch.height - 1 - Index(sy) : Index(sy);
                value = float(input[((sample * batch.height + y) * batch.width + x) * batch.channels + c]);
            }
        }
        output[i] = static_cast<Output>(__fadd_rn(__fmul_rn(value, scales[c].scale), scales[c].offset));
    }
}

struct ResizeSample
{
    Index a, b, c, d;
    float dx, dy;
};

struct ImagePixel
{
    Index sample, x, y;
};

__device__ ImagePixel image_pixel(Index pixel, Index width, Index height)
{
    return {pixel / (height * width), pixel % width, (pixel / width) % height};
}

__device__ ResizeSample resize_sample(Index x, Index y, Index width, Index height, Index out_w, Index out_h)
{
    const float sx = __fsub_rn(__fmul_rn(float(x) + 0.5f, float(width)) / float(out_w), 0.5f);
    const float sy = __fsub_rn(__fmul_rn(float(y) + 0.5f, float(height)) / float(out_h), 0.5f);
    const Index x0 = min(width - 1, max(Index(0), Index(floorf(sx))));
    const Index y0 = min(height - 1, max(Index(0), Index(floorf(sy))));
    const Index x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
    return {y0 * width + x0, y0 * width + x1, y1 * width + x0, y1 * width + x1,
            sx - float(x0), sy - float(y0)};
}

__global__ void mosaic_kernel(const uint8_t* input, uint8_t* output,
                               const ImageTransformParameters* parameters, ImageTransformBatch batch, Index count)
{
    const Index pixel = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= count) return;
    const auto point = image_pixel(pixel, batch.width, batch.height);
    const auto& parameter = parameters[point.sample];
    const bool right = point.x >= parameter.cut_x, bottom = point.y >= parameter.cut_y;
    const int q = int(bottom) * 2 + int(right);
    const Index x0 = right ? parameter.cut_x : 0, y0 = bottom ? parameter.cut_y : 0;
    const Index width = right ? batch.width - x0 : parameter.cut_x;
    const Index height = bottom ? batch.height - y0 : parameter.cut_y;
    const auto position = resize_sample(point.x - x0, point.y - y0, batch.width, batch.height, width, height);
    const uint8_t* source = input + parameter.sources[q] * batch.height * batch.width * batch.channels;
    uint8_t* destination = output + pixel * batch.channels;
    Index first_channel = 0;
    if (batch.channels >= 3)
    {
        const auto color = parameter.colors[q];
        const float3 a = read_color(source, position.a, batch.channels, color);
        const float3 b = read_color(source, position.b, batch.channels, color);
        const float3 c = read_color(source, position.c, batch.channels, color);
        const float3 d = read_color(source, position.d, batch.channels, color);
        destination[0] = uint8_t(byte_value(blend(a.x, b.x, c.x, d.x, position.dx, position.dy)));
        destination[1] = uint8_t(byte_value(blend(a.y, b.y, c.y, d.y, position.dx, position.dy)));
        destination[2] = uint8_t(byte_value(blend(a.z, b.z, c.z, d.z, position.dx, position.dy)));
        first_channel = 3;
    }
    for (Index c = first_channel; c < batch.channels; ++c)
        destination[c] = uint8_t(byte_value(blend(source[position.a * batch.channels + c],
            source[position.b * batch.channels + c], source[position.c * batch.channels + c],
            source[position.d * batch.channels + c], position.dx, position.dy)));
}

__device__ float padded_pixel(const uint8_t* source, Index x, Index y, Index c, const ImageTransformBatch& batch)
{
    return x < 0 || x >= batch.width || y < 0 || y >= batch.height
        ? 128.0f : float(source[(y * batch.width + x) * batch.channels + c]);
}

__device__ float cropped_channel(const uint8_t* source, float x, float y, Index c, const ImageTransformBatch& batch)
{
    const Index x0 = Index(floorf(x)), y0 = Index(floorf(y));
    return byte_value(blend(padded_pixel(source, x0, y0, c, batch), padded_pixel(source, x0 + 1, y0, c, batch),
                            padded_pixel(source, x0, y0 + 1, c, batch), padded_pixel(source, x0 + 1, y0 + 1, c, batch),
                            x - float(x0), y - float(y0)));
}

__device__ bool is_cutout(Index x, Index y, const ImageRegion* regions, Index count)
{
    for (Index i = 0; i < count; ++i)
        if (x >= regions[i].x && x < regions[i].x + regions[i].width
            && y >= regions[i].y && y < regions[i].y + regions[i].height) return true;
    return false;
}

template<typename Output>
__device__ void store_channel(Output* destination, float value, const ImageChannelScale& scale)
{
    if constexpr (std::is_same_v<Output, uint8_t>) *destination = uint8_t(value);
    else *destination = static_cast<Output>(__fadd_rn(__fmul_rn(value, scale.scale), scale.offset));
}

template<typename Output>
__global__ void crop_kernel(const uint8_t* input, Output* output, const ImageTransformParameters* parameters,
                             const ImageChannelScale* scales, const ImageRegion* regions,
                             ImageTransformBatch batch, Index count)
{
    for (Index pixel = Index(blockIdx.x) * blockDim.x + threadIdx.x; pixel < count;
         pixel += Index(blockDim.x) * gridDim.x)
    {
        const auto point = image_pixel(pixel, batch.width, batch.height);
        const auto& parameter = parameters[point.sample];
        const auto& transform = parameter.crop;
        const Index source_index = batch.mosaic ? point.sample : parameter.sources[0];
        const uint8_t* source = input + source_index * batch.height * batch.width * batch.channels;
        const Index dx = transform.flip ? batch.width - 1 - point.x : point.x;
        const float width = __fmul_rn(1.0f - transform.crop_left - transform.crop_right, float(batch.width));
        const float height = __fmul_rn(1.0f - transform.crop_top - transform.crop_bottom, float(batch.height));
        const float sx = __fsub_rn(__fadd_rn(__fmul_rn(transform.crop_left, float(batch.width)),
                                           __fmul_rn(float(dx) + 0.5f, width) / float(batch.width)), 0.5f);
        const float sy = __fsub_rn(__fadd_rn(__fmul_rn(transform.crop_top, float(batch.height)),
                                           __fmul_rn(float(point.y) + 0.5f, height) / float(batch.height)), 0.5f);
        Output* destination = output + pixel * batch.channels;
        if (is_cutout(point.x, point.y, regions + point.sample * batch.cutouts, batch.cutouts))
        {
            for (Index c = 0; c < batch.channels; ++c) store_channel(destination + c, 0.0f, scales[c]);
            continue;
        }
        Index first_channel = 0;
        if (batch.enabled && !batch.mosaic && batch.channels >= 3)
        {
            const float3 rgb = color_pixel({cropped_channel(source, sx, sy, 0, batch),
                                           cropped_channel(source, sx, sy, 1, batch),
                                           cropped_channel(source, sx, sy, 2, batch)},
                                          {transform.exposure_mul, transform.saturation_mul, transform.hue_shift});
            store_channel(destination, rgb.x, scales[0]);
            store_channel(destination + 1, rgb.y, scales[1]);
            store_channel(destination + 2, rgb.z, scales[2]);
            first_channel = 3;
        }
        for (Index c = first_channel; c < batch.channels; ++c)
        {
            const float value = batch.enabled ? cropped_channel(source, sx, sy, c, batch)
                                              : float(source[(point.y * batch.width + point.x) * batch.channels + c]);
            store_channel(destination + c, value, scales[c]);
        }
    }
}

template<typename Output>
__global__ void resize_kernel(const uint8_t* input, Output* output, const ImageChannelScale* scales,
                               ImageTransformBatch batch, Index count)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += Index(blockDim.x) * gridDim.x)
    {
        const Index c = i % batch.channels, pixel = i / batch.channels;
        const Index sample = pixel / (batch.output_height * batch.output_width);
        const auto position = resize_sample(pixel % batch.output_width, (pixel / batch.output_width) % batch.output_height,
                                             batch.width, batch.height, batch.output_width, batch.output_height);
        const uint8_t* source = input + sample * batch.height * batch.width * batch.channels;
        const float value = byte_value(blend(source[position.a * batch.channels + c], source[position.b * batch.channels + c],
                                            source[position.c * batch.channels + c], source[position.d * batch.channels + c],
                                            position.dx, position.dy));
        store_channel(output + i, value, scales[c]);
    }
}

template<typename Output>
void transform_images(const void* input, Output* output, const void* metadata, uint8_t* scratch,
                        const ImageTransformBatch& batch, cudaStream_t stream)
{
    const auto* parameters = static_cast<const ImageTransformParameters*>(metadata);
    const auto* scales = reinterpret_cast<const ImageChannelScale*>(static_cast<const uint8_t*>(metadata) + batch.scales_offset());
    const auto* regions = reinterpret_cast<const ImageRegion*>(static_cast<const uint8_t*>(metadata) + batch.cutouts_offset());
    const Index pixels = batch.samples * batch.height * batch.width;
    const int blocks = grid_size_strided_for(checked_int(pixels));
    if (batch.kind == ImageTransformBatch::Kind::Affine)
    {
        if (batch.float_input)
            OPENNN_CUDA_LAUNCH((affine_kernel<<<blocks, block_size, 0, stream>>>(static_cast<const float*>(input), output,
                parameters, scales, batch, pixels * batch.channels)));
        else
            OPENNN_CUDA_LAUNCH((affine_kernel<<<blocks, block_size, 0, stream>>>(static_cast<const uint8_t*>(input), output,
                parameters, scales, batch, pixels * batch.channels)));
        return;
    }
    const auto* source = static_cast<const uint8_t*>(input);
    if (batch.mosaic)
    {
        OPENNN_CUDA_LAUNCH((mosaic_kernel<<<grid_size_for(checked_int(pixels)), block_size, 0, stream>>>(
            source, scratch, parameters, batch, pixels)));
        source = scratch;
        scratch += pixels * batch.channels;
    }
    if (batch.resize())
    {
        OPENNN_CUDA_LAUNCH((crop_kernel<<<blocks, block_size, 0, stream>>>(source, scratch, parameters, scales, regions, batch, pixels)));
        const Index count = batch.samples * batch.output_height * batch.output_width * batch.channels;
        OPENNN_CUDA_LAUNCH((resize_kernel<<<grid_size_strided_for(checked_int(count)), block_size, 0, stream>>>(
            scratch, output, scales, batch, count)));
    }
    else
        OPENNN_CUDA_LAUNCH((crop_kernel<<<blocks, block_size, 0, stream>>>(source, output, parameters, scales, regions, batch, pixels)));
}

}

void transform_images_cuda(const void* input, void* output, bool bf16, const void* metadata, uint8_t* scratch,
                           const ImageTransformBatch& batch, cudaStream_t stream)
{
    if (batch.samples == 0) return;
    if (bf16) transform_images(input, static_cast<__nv_bfloat16*>(output), metadata, scratch, batch, stream);
    else transform_images(input, static_cast<float*>(output), metadata, scratch, batch, stream);
}

}
