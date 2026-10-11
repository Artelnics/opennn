// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/training/kernel_losses.cuh"

template<typename T>
__global__ void minkowski_error_kernel(const int n, float* __restrict__ term_results, const float* __restrict__ targets, const T* __restrict__ outputs, const float power)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
        term_results[i] = powf(fabsf(static_cast<float>(outputs[i]) - targets[i]), power);
}

template<typename T>
void minkowski_error_cuda(const Index n, float* term_results, const float* targets, const T* outputs, const float power)
{
    launch_elementwise(n, minkowski_error_kernel<T>, term_results, targets, outputs, power);
}

// Derivative of |d|^p / p: sign(d) |d|^(p-1), and zero at a zero residual.
__device__ __forceinline__ float minkowski_slope(const float difference, const float power)
{
    const float sign = difference > 0.0f ? 1.0f : (difference < 0.0f ? -1.0f : 0.0f);
    return sign * powf(fabsf(difference), power - 1.0f);
}

template<typename T>
__global__ void minkowski_error_gradient_kernel(const int n, T* __restrict__ deltas, const float* __restrict__ targets, const T* __restrict__ outputs, const float power, const float scale)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
        deltas[i] = static_cast<T>(scale * minkowski_slope(static_cast<float>(outputs[i]) - targets[i], power));
}

template<typename T>
void minkowski_error_gradient_cuda(const Index n,
                                   T* deltas,
                                   const float* targets,
                                   const T* outputs,
                                   const float power,
                                   const float scale)
{
    launch_elementwise(n, minkowski_error_gradient_kernel<T>, deltas, targets, outputs, power, scale);
}

template<typename T>
__global__ void binary_cross_entropy_kernel(const int n, float* __restrict__ term_results, const float* __restrict__ targets, const T* __restrict__ outputs, const float epsilon)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const float out = static_cast<float>(outputs[i]);
        const float tgt = targets[i];

        const float log_pos = logf(out + epsilon);
        const float log_neg = logf(1.0f - out + epsilon);

        term_results[i] = fmaf(tgt, log_pos - log_neg, log_neg);
    }
}

template<typename T>
void binary_cross_entropy_cuda(const Index n, float* term_results, const float* targets, const T* outputs, const float epsilon)
{
    launch_elementwise(n, binary_cross_entropy_kernel<T>, term_results, targets, outputs, epsilon);
}

template<typename T>
__global__ void binary_cross_entropy_logits_kernel(const int n, float* __restrict__ terms,
                                                   const float* __restrict__ targets,
                                                   const T* __restrict__ logits)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const float x = static_cast<float>(logits[i]);
        terms[i] = fmaxf(x, 0.0f) - x * targets[i] + log1pf(expf(-fabsf(x)));
    }
}

template<typename T>
void binary_cross_entropy_logits_cuda(const Index n, float* terms, const float* targets, const T* logits)
{
    launch_elementwise(n, binary_cross_entropy_logits_kernel<T>, terms, targets, logits);
}

template<typename T>
__global__ void binary_cross_entropy_gradient_kernel(
    const int n,
    T* __restrict__ deltas,
    const float* __restrict__ targets,
    const T* __restrict__ outputs,
    const float epsilon, const float scaling_factor)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const float out = static_cast<float>(outputs[i]);
        const float tgt = targets[i];
        deltas[i] = static_cast<T>(((1.0f - tgt) / (1.0f - out + epsilon) - tgt / (out + epsilon)) * scaling_factor);
    }
}

template<typename T>
void binary_cross_entropy_gradient_cuda(const Index n, T* deltas, const float* targets, const T* outputs, const float epsilon, const float scaling_factor)
{
    launch_elementwise(n, binary_cross_entropy_gradient_kernel<T>, deltas, targets, outputs, epsilon, scaling_factor);
}

template<typename T>
__global__ void categorical_cross_entropy_kernel(const int n, float* __restrict__ term_results, const float* __restrict__ targets, const T* __restrict__ outputs, const float epsilon)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const float tgt = targets[i];
        term_results[i] = (tgt > 0.0f) ? tgt * logf(static_cast<float>(outputs[i]) + epsilon) : 0.0f;
    }
}

template<typename T>
void categorical_cross_entropy_cuda(const Index n, float* term_results, const float* targets, const T* outputs, const float epsilon)
{
    launch_elementwise(n, categorical_cross_entropy_kernel<T>, term_results, targets, outputs, epsilon);
}

template<typename T>
__global__ void categorical_cross_entropy_gradient_kernel(
    const int n,
    T* __restrict__ deltas,
    const float* __restrict__ targets,
    const T* __restrict__ outputs,
    const float scaling_factor)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
        deltas[i] = static_cast<T>((static_cast<float>(outputs[i]) - targets[i]) * scaling_factor);
}

template<typename T>
void categorical_cross_entropy_gradient_cuda(const Index n, T* deltas, const float* targets, const T* outputs, const float scaling_factor)
{
    launch_elementwise(n, categorical_cross_entropy_gradient_kernel<T>, deltas, targets, outputs, scaling_factor);
}

template<typename T>
__global__ void weighted_squared_error_kernel(const int n, float* __restrict__ term_results, const float* __restrict__ targets, const T* __restrict__ outputs, const float positives_weight, const float negatives_weight)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const float tgt = targets[i];
        const float diff = static_cast<float>(outputs[i]) - tgt;
        const float weight = (tgt >= 0.5f) ? positives_weight : negatives_weight;

        term_results[i] = diff * diff * weight;
    }
}

template<typename T>
void weighted_squared_error_cuda(const Index n, float* term_results, const float* targets, const T* outputs, const float positives_weight, const float negatives_weight)
{
    launch_elementwise(n, weighted_squared_error_kernel<T>, term_results, targets, outputs, positives_weight, negatives_weight);
}

template<typename T>
__global__ void weighted_squared_error_gradient_kernel(
    const int n,
    T* __restrict__ deltas,
    const float* __restrict__ targets,
    const T* __restrict__ outputs,
    const float positives_weight,
    const float negatives_weight,
    const float scaling_factor)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const float tgt = targets[i];
        const float diff = static_cast<float>(outputs[i]) - tgt;
        const float weight = (tgt >= 0.5f) ? positives_weight : negatives_weight;
        deltas[i] = static_cast<T>(diff * weight * scaling_factor);
    }
}

template<typename T>
void weighted_squared_error_gradient_cuda(const Index n, T* deltas, const float* targets, const T* outputs, const float positives_weight, const float negatives_weight, const float scaling_factor)
{
    launch_elementwise(n, weighted_squared_error_gradient_kernel<T>, deltas, targets, outputs, positives_weight, negatives_weight, scaling_factor);
}

template<typename T>
__global__ void cross_entropy_3d_multiple_forward_kernel(const int total_tokens,
                                                         const int vocab_size,
                                                         const T* __restrict__ outputs,
                                                         const float* __restrict__ targets,
                                                         float* __restrict__ errors,
                                                         float* __restrict__ valid_mask,
                                                         float* __restrict__ correct_mask)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < total_tokens; idx += Index(blockDim.x) * gridDim.x)
    {
        const int target_class = static_cast<int>(targets[idx]);
        const bool valid = target_class > 0 && target_class < vocab_size;

        // Logits in, so the loss is logsumexp(row) - row[target]. One pass finds the max
        // (which the accuracy count needs anyway), a second sums the exponentials.
        float error = 0.0f;
        float best_match = 0.0f;

        if (valid)
        {
            const T* row = outputs + idx * vocab_size;

            float best_value = static_cast<float>(row[0]);
            int best_index = 0;
            for (int k = 1; k < vocab_size; ++k)
            {
                const float value = static_cast<float>(row[k]);
                if (value > best_value) { best_value = value; best_index = k; }
            }

            float sum_exp = 0.0f;
            for (int k = 0; k < vocab_size; ++k)
                sum_exp += __expf(static_cast<float>(row[k]) - best_value);

            error = logf(sum_exp) + best_value - static_cast<float>(row[target_class]);
            best_match = (best_index == target_class) ? 1.0f : 0.0f;
        }

        errors[idx] = error;
        if (valid_mask) valid_mask[idx] = valid ? 1.0f : 0.0f;
        if (correct_mask) correct_mask[idx] = best_match;
    }
}

template<typename T>
void cross_entropy_3d_multiple_forward_cuda(const Index n,
                                            const int vocab_size,
                                            const T* outputs,
                                            const float* targets,
                                            float* errors,
                                            float* valid_mask,
                                            float* correct_mask)
{
    launch_elementwise(n, cross_entropy_3d_multiple_forward_kernel<T>,
                       vocab_size, outputs, targets, errors, valid_mask, correct_mask);
}

template<typename T>
__global__ void cross_entropy_3d_metrics_kernel(const int total_tokens, const int vocab_size,
                                                const T* __restrict__ outputs,
                                                const float* __restrict__ targets,
                                                float* __restrict__ sums)
{
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    float loss = 0.0f, active = 0.0f, correct = 0.0f;

    for (int token = blockIdx.x * warps + warp; token < total_tokens; token += gridDim.x * warps)
    {
        const int target_class = static_cast<int>(targets[token]);
        if (target_class <= 0 || target_class >= vocab_size) continue;

        const T* row = outputs + Index(token) * vocab_size;
        float best_value = -INFINITY;
        int best_index = 0;
        for (int k = lane; k < vocab_size; k += 32)
        {
            const float value = static_cast<float>(row[k]);
            if (value > best_value) { best_value = value; best_index = k; }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            const float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
            const int   other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
            if (other_value > best_value || (other_value == best_value && other_index < best_index))
            {
                best_value = other_value;
                best_index = other_index;
            }
        }
        // The argmax reduction leaves the winner in lane 0 only, so broadcast before use.
        const float row_max = __shfl_sync(0xffffffffu, best_value, 0);
        const int   arg_max = __shfl_sync(0xffffffffu, best_index, 0);

        float sum_exp = 0.0f;
        for (int k = lane; k < vocab_size; k += 32)
            sum_exp += __expf(static_cast<float>(row[k]) - row_max);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xffffffffu, sum_exp, offset);

        if (lane == 0)
        {
            loss += logf(sum_exp) + row_max - static_cast<float>(row[target_class]);
            active += 1.0f;
            correct += arg_max == target_class ? 1.0f : 0.0f;
        }
    }

    __shared__ float block_loss[32], block_active[32], block_correct[32];
    if (lane == 0)
    {
        block_loss[warp] = loss;
        block_active[warp] = active;
        block_correct[warp] = correct;
    }
    __syncthreads();
    if (warp == 0)
    {
        loss    = lane < warps ? block_loss[lane]    : 0.0f;
        active  = lane < warps ? block_active[lane]  : 0.0f;
        correct = lane < warps ? block_correct[lane] : 0.0f;
        loss = warp_reduce_sum(loss);
        active = warp_reduce_sum(active);
        correct = warp_reduce_sum(correct);
        if (lane == 0)
        {
            atomicAdd(sums + 0, loss);
            atomicAdd(sums + 1, active);
            atomicAdd(sums + 2, correct);
        }
    }
}

template<typename T>
void cross_entropy_3d_metrics_cuda(const Index total_tokens, const int vocab_size,
                                   const T* outputs, const float* targets, float* sums)
{
    if (total_tokens <= 0) return;
    const int warps_per_block = block_size / 32;
    const Index needed = (total_tokens + warps_per_block - 1) / warps_per_block;
    const int blocks = int(needed < 4096 ? needed : 4096);
    OPENNN_CUDA_LAUNCH(cross_entropy_3d_metrics_kernel<T><<<blocks, block_size, 0, opennn::device::get_compute_stream()>>>(
        int(total_tokens), vocab_size, outputs, targets, sums));
}

// One warp per token. The inputs are logits, so the softmax denominator is a per-row
// reduction and this can no longer be a flat elementwise kernel.
//
// Safe when output_deltas aliases outputs, which it does whenever
// Loss::output_delta_overwrites_outputs() is on: the max and sum reductions are complete
// (and broadcast) before any store, and in the final loop each lane writes only the
// element it just read.
template<typename T>
__global__ void cross_entropy_3d_multiple_backward_kernel(const int total_tokens,
                                                          const int vocab_size,
                                                          const T* outputs,
                                                          const float* __restrict__ targets,
                                                          T* output_deltas,
                                                          float scale_factor,
                                                          const float* __restrict__ active_count_device)
{
    if (active_count_device)
    {
        const float active_count = active_count_device[0];
        scale_factor = active_count > 0.0f ? 1.0f / active_count : 0.0f;
    }

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    for (int token = blockIdx.x * warps + warp; token < total_tokens; token += gridDim.x * warps)
    {
        const T* row = outputs + Index(token) * vocab_size;
        T* delta_row = output_deltas + Index(token) * vocab_size;

        const int target_class = static_cast<int>(targets[token]);

        if (target_class <= 0 || target_class >= vocab_size)
        {
            for (int k = lane; k < vocab_size; k += 32)
                delta_row[k] = static_cast<T>(0.0f);
            continue;
        }

        float row_max = -INFINITY;
        for (int k = lane; k < vocab_size; k += 32)
            row_max = fmaxf(row_max, static_cast<float>(row[k]));
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            row_max = fmaxf(row_max, __shfl_down_sync(0xffffffffu, row_max, offset));
        row_max = __shfl_sync(0xffffffffu, row_max, 0);

        float sum_exp = 0.0f;
        for (int k = lane; k < vocab_size; k += 32)
            sum_exp += __expf(static_cast<float>(row[k]) - row_max);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xffffffffu, sum_exp, offset);
        sum_exp = __shfl_sync(0xffffffffu, sum_exp, 0);

        const float normalization = scale_factor / sum_exp;

        for (int k = lane; k < vocab_size; k += 32)
        {
            const float scaled_probability = __expf(static_cast<float>(row[k]) - row_max) * normalization;
            delta_row[k] = static_cast<T>(scaled_probability
                                          - (k == target_class ? scale_factor : 0.0f));
        }
    }
}

template<typename T>
void cross_entropy_3d_multiple_backward_cuda(const Index n,
                                             const int vocab_size,
                                             const T* outputs,
                                             const float* targets,
                                             T* output_deltas,
                                             const float scale_factor,
                                             const float* active_count_device)
{
    if (n <= 0 || vocab_size <= 0) return;

    const Index total_tokens = n / vocab_size;
    const int warps_per_block = block_size / 32;
    const Index needed = (total_tokens + warps_per_block - 1) / warps_per_block;
    const int blocks = int(needed < 4096 ? needed : 4096);

    OPENNN_CUDA_LAUNCH(cross_entropy_3d_multiple_backward_kernel<T><<<blocks, block_size, 0,
        opennn::device::get_compute_stream()>>>(
            int(total_tokens), vocab_size, outputs, targets, output_deltas,
            scale_factor, active_count_device));
}

__global__ void accumulate_scaled_metric_kernel(const float* __restrict__ value,
                                                const float scale,
                                                float* __restrict__ error_sum)
{
    error_sum[0] += value[0] * scale;
}

void accumulate_scaled_metric_cuda(const float* value, float scale, float* error_sum)
{
    launch_single(nullptr, accumulate_scaled_metric_kernel, value, scale, error_sum);
}

__global__ void accumulate_cross_entropy_3d_metrics_kernel(const float* __restrict__ values,
                                                           float* __restrict__ error_sum,
                                                           float* __restrict__ accuracy_sum)
{
    const float loss_sum = values[0];
    const float active_count = values[1];
    const float correct_count = values[2];

    if (active_count > 0.0f)
    {
        error_sum[0] += loss_sum / active_count;
        if (accuracy_sum) accuracy_sum[0] += correct_count / active_count;
    }
}

void accumulate_cross_entropy_3d_metrics_cuda(const float* values,
                                              float* error_sum,
                                              float* accuracy_sum)
{
    launch_single(nullptr, accumulate_cross_entropy_3d_metrics_kernel, values, error_sum, accuracy_sum);
}

template<typename T>
__device__ __forceinline__ void l1_gradient_one(T& d, T p, float weight)
{
    const float pf = static_cast<float>(p);
    const float s = (pf > 0.0f) ? 1.0f : ((pf < 0.0f) ? -1.0f : 0.0f);
    d = static_cast<T>(static_cast<float>(d) + weight * s);
}

template<typename T>
__global__ void l1_gradient_kernel(
    const int n_vec, const int n,
    T* __restrict__ deltas,
    const T* __restrict__ parameters,
    const float weight)
{
    constexpr int vec_width = vec16<T>;

    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;

    float4* const       d_v = reinterpret_cast<float4*>(deltas);
    const float4* const p_v = reinterpret_cast<const float4*>(parameters);

    for (Index i = tid; i < n_vec; i += stride)
    {
        float4 d_chunk = d_v[i];
        float4 p_chunk = p_v[i];
        T* d_lanes = reinterpret_cast<T*>(&d_chunk);
        T* p_lanes = reinterpret_cast<T*>(&p_chunk);

        #pragma unroll
        for (int k = 0; k < vec_width; ++k)
            l1_gradient_one(d_lanes[k], p_lanes[k], weight);

        d_v[i] = d_chunk;
    }

    const int tail_start = n_vec * vec_width;
    for (Index i = tail_start + tid; i < n; i += stride)
        l1_gradient_one(deltas[i], parameters[i], weight);
}

template<typename T>
void l1_gradient_cuda(const Index n, T* deltas, const T* parameters, const float weight)
{
    launch_vec_on<vec16<T>>(opennn::device::get_compute_stream(), n, are_aligned<16>(deltas, parameters),
                            l1_gradient_kernel<T>, deltas, parameters, weight);
}

static constexpr float YOLO_EPSILON       = 1e-7f;
static constexpr float YOLO_CORNER_EPS    = 1e-6f;
static constexpr float YOLO_GRAD_CLIP     = 10.0f;

static constexpr float YOLO_INV_PI2       = 4.0f / (3.14159265f * 3.14159265f);

struct CiouTerms
{
    float pl, pr, pt, pb;
    float gl, gr, gt_, gb;
    float iw_raw, ih_raw;
    float iw, ih, inter;
    float uni, iou;
    float ew, eh, enc, giou;
    float dx, dy, rho2, c2;
    float v_diff, v, alpha;
};

__device__ __forceinline__ CiouTerms ciou_terms(const float* pred, const float* gt)
{
    const float pred_x = pred[0], pred_y = pred[1], pred_w = pred[2], pred_h = pred[3];
    const float gt_x   = gt[0],   gt_y   = gt[1],   gt_w   = gt[2],   gt_h   = gt[3];

    CiouTerms t;

    // Edges from center and size; y grows downward, so top is the smaller value.
    t.pl  = pred_x - 0.5f * pred_w;
    t.pr  = pred_x + 0.5f * pred_w;
    t.pt  = pred_y - 0.5f * pred_h;
    t.pb  = pred_y + 0.5f * pred_h;
    t.gl  = gt_x - 0.5f * gt_w;
    t.gr  = gt_x + 0.5f * gt_w;
    t.gt_ = gt_y - 0.5f * gt_h;
    t.gb  = gt_y + 0.5f * gt_h;

    // Signed overlap (negative when disjoint), its clamp, and the enclosing box.
    t.iw_raw = fminf(t.pr, t.gr) - fmaxf(t.pl, t.gl);
    t.ih_raw = fminf(t.pb, t.gb) - fmaxf(t.pt, t.gt_);
    t.iw = fmaxf(t.iw_raw, 0.0f);
    t.ih = fmaxf(t.ih_raw, 0.0f);
    t.ew = fmaxf(t.pr, t.gr) - fminf(t.pl, t.gl);
    t.eh = fmaxf(t.pb, t.gb) - fminf(t.pt, t.gt_);

    t.inter = t.iw * t.ih;
    t.enc   = t.ew * t.eh;
    t.uni   = pred_w * pred_h + gt_w * gt_h - t.inter;

    t.iou  = t.uni > 0.0f ? t.inter / t.uni : 0.0f;
    t.giou = t.enc > 0.0f ? t.iou - (t.enc - t.uni) / t.enc : t.iou;

    // Normalized center distance.
    t.dx   = pred_x - gt_x;
    t.dy   = pred_y - gt_y;
    t.rho2 = t.dx * t.dx + t.dy * t.dy;
    t.c2   = t.ew * t.ew + t.eh * t.eh + YOLO_EPSILON;

    // Aspect-ratio consistency; atan2f(w, h) stays defined for h = 0.
    t.v_diff = atan2f(gt_w, gt_h) - atan2f(pred_w, pred_h);
    t.v      = YOLO_INV_PI2 * t.v_diff * t.v_diff;
    t.alpha  = t.uni > 0.0f ? t.v / (1.0f - t.iou + t.v + YOLO_EPSILON) : 0.0f;

    return t;
}

__device__ __forceinline__ float yolo_ciou_forward(const float* pred, const float* gt, float* out_iou)
{
    const CiouTerms t = ciou_terms(pred, gt);

    *out_iou = t.iou;

    // CIoU penalties on top of the GIoU base term.
    return t.giou - t.rho2 / t.c2 - t.alpha * t.v;
}

__device__ __forceinline__ float corner_max_grad(float a, float b)
{
    // Strict comparisons: the closed tie band and NaN inputs fall through to 1/2.
    if (a > b + YOLO_CORNER_EPS) return 1.0f;
    if (a < b - YOLO_CORNER_EPS) return 0.0f;
    return 0.5f;
}

__device__ __forceinline__ float corner_min_grad(float a, float b)
{
    // Complement by construction; exact because the max subgradient is 0, 1/2 or 1.
    return 1.0f - corner_max_grad(a, b);
}

__device__ __forceinline__ void yolo_ciou_grad(
    const float* pred, const float* gt,
    float* out_iou, float* out_giou,
    float& cx_grad, float& cy_grad, float& w_grad, float& h_grad)
{
    const CiouTerms t = ciou_terms(pred, gt);

    *out_iou = t.iou;
    *out_giou = t.giou;

    const float width  = pred[2];
    const float height = pred[3];

    const float left_max   = corner_max_grad(t.pl, t.gl);
    const float left_min   = corner_min_grad(t.pl, t.gl);
    const float right_max  = corner_max_grad(t.pr, t.gr);
    const float right_min  = corner_min_grad(t.pr, t.gr);
    const float top_max    = corner_max_grad(t.pt, t.gt_);
    const float top_min    = corner_min_grad(t.pt, t.gt_);
    const float bottom_max = corner_max_grad(t.pb, t.gb);
    const float bottom_min = corner_min_grad(t.pb, t.gb);

    // The intersection follows the edges only while the boxes overlap on both axes.
    const float overlapping = (t.iw_raw > 0.0f && t.ih_raw > 0.0f) ? 1.0f : 0.0f;

    // d(1 - GIoU)/d(edge) from the edge derivatives of the intersection, the predicted
    // area and the enclosure. The union moves with the area minus the intersection.
    const auto edge_gradient = [&t](const float d_inter, const float d_area, const float d_enc)
    {
        const float d_union = d_area - d_inter;
        const float d_iou = t.uni > 0.0f
            ? (d_inter * t.uni - t.inter * d_union) / (t.uni * t.uni)
            : 0.0f;
        const float d_penalty = t.enc > 0.0f
            ? (t.uni * d_enc - t.enc * d_union) / (t.enc * t.enc)
            : 0.0f;
        return d_penalty - d_iou;
    };

    const float left_gradient   = edge_gradient(-overlapping * left_max * t.ih,   -height, -left_min * t.eh);
    const float right_gradient  = edge_gradient( overlapping * right_min * t.ih,   height,  right_max * t.eh);
    const float top_gradient    = edge_gradient(-overlapping * top_max * t.iw,    -width,  -top_min * t.ew);
    const float bottom_gradient = edge_gradient( overlapping * bottom_min * t.iw,  width,   bottom_max * t.ew);

    // Edges to center and size: left = cx - w/2, right = cx + w/2, likewise in y.
    float x_total = left_gradient + right_gradient;
    float y_total = top_gradient + bottom_gradient;
    float w_total = 0.5f * (right_gradient - left_gradient);
    float h_total = 0.5f * (bottom_gradient - top_gradient);

    // Distance penalty rho^2 / c^2, where c^2 depends on the box through the enclosure.
    const float dew_dcx = right_max - left_min;
    const float dew_dw  = 0.5f * (right_max + left_min);
    const float deh_dcy = bottom_max - top_min;
    const float deh_dh  = 0.5f * (bottom_max + top_min);
    const float inv_c2_squared = 1.0f / (t.c2 * t.c2);

    x_total += (2.0f * t.dx * t.c2 - t.rho2 * 2.0f * t.ew * dew_dcx) * inv_c2_squared;
    y_total += (2.0f * t.dy * t.c2 - t.rho2 * 2.0f * t.eh * deh_dcy) * inv_c2_squared;
    w_total -= t.rho2 * 2.0f * t.ew * dew_dw * inv_c2_squared;
    h_total -= t.rho2 * 2.0f * t.eh * deh_dh * inv_c2_squared;

    // Aspect penalty alpha * v, with alpha held constant as in the CIoU paper.
    const float diagonal_squared = width * width + height * height + YOLO_EPSILON;
    const float aspect_factor = t.alpha * YOLO_INV_PI2 * 2.0f * t.v_diff / diagonal_squared;

    w_total -= aspect_factor * height;
    h_total += aspect_factor * width;

    cx_grad = x_total;
    cy_grad = y_total;
    w_grad  = w_total;
    h_grad  = h_total;
}

__device__ __forceinline__ float yolo_decode_cell_boxes(const int i, const int boxes_per_cell, const int grid_size,
                                                        const float* pred_raw, const float* gt_raw,
                                                        float* pred, float* gt)
{
    // Slots are [sample][row][column][box]; the sample drops out through the modulo.
    const int cell   = i / boxes_per_cell;
    const int column = cell % grid_size;
    const int row    = (cell / grid_size) % grid_size;

    const float cells = static_cast<float>(grid_size);

    const auto decode = [&](const float* raw, float* box)
    {
        box[0] = (raw[0] + static_cast<float>(column)) / cells;
        box[1] = (raw[1] + static_cast<float>(row)) / cells;
        box[2] = raw[2];
        box[3] = raw[3];
    };

    decode(pred_raw, pred);
    decode(gt_raw, gt);

    return 1.0f / cells;
}

__global__ void yolo_loss_forward_kernel(
    const int n_boxes,
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ error_accum,
    const int values_per_box,
    const int classes_number,
    const int sigmoid_classes,
    const int grid_size,
    const int boxes_per_cell,
    const float lambda_giou,
    const float lambda_noobj,
    const float lambda_class,
    const float focal_gamma,
    const float obj_focal_gamma)
{
    float thread_loss = 0.0f;

    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n_boxes; i += Index(blockDim.x) * gridDim.x)
    {
        const float* out = output + i * values_per_box;
        const float* tgt = target + i * values_per_box;

        // Role from the objectness target: a NaN tau fails both tests and is ignored.
        const float tau = tgt[4];

        if (tau >= 0.5f)
        {
            float pred[4], gt[4];
            yolo_decode_cell_boxes(int(i), boxes_per_cell, grid_size, out, tgt, pred, gt);

            float iou;
            const float similarity = yolo_ciou_forward(pred, gt, &iou);

            const float objectness = out[4];
            const float objectness_loss = -(tau * logf(objectness + YOLO_EPSILON)
                                          + (1.0f - tau) * logf(1.0f - objectness + YOLO_EPSILON));

            float class_loss = 0.0f;

            for (int c = 0; c < classes_number; ++c)
            {
                const float probability = out[5 + c];
                const float label = tgt[5 + c];

                if (sigmoid_classes != 0)
                {
                    const float correct_probability = label > 0.5f ? probability : 1.0f - probability;
                    const float focal = __powf(1.0f - correct_probability, focal_gamma);
                    class_loss += focal * -(label * logf(probability + YOLO_EPSILON)
                                          + (1.0f - label) * logf(1.0f - probability + YOLO_EPSILON));
                }
                else if (label > 0.0f)
                    class_loss -= logf(probability + YOLO_EPSILON);
            }

            thread_loss += lambda_giou * (1.0f - similarity) + objectness_loss + lambda_class * class_loss;
        }
        else if (tau > -0.5f)
        {
            const float objectness = out[4];
            const float weight = obj_focal_gamma > 0.0f ? __powf(objectness, obj_focal_gamma) : 1.0f;
            thread_loss -= lambda_noobj * weight * logf(1.0f - objectness + YOLO_EPSILON);
        }
    }

    // blockDim.x is uniform across the block, so either every thread reaches the
    // barriers of the warp-shuffle reduction or none does. Partial warps cannot use
    // full-mask shuffles, so such launches fall back to one atomic per thread.
    if (blockDim.x % 32 == 0)
    {
        if (block_reduce_sum(thread_loss))
            atomicAdd(error_accum, thread_loss);
    }
    else
        atomicAdd(error_accum, thread_loss);
}

__global__ void yolo_loss_gradient_kernel(
    const int n_boxes,
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ delta,
    const int values_per_box,
    const int classes_number,
    const int sigmoid_classes,
    const float inv_batch,
    const int grid_size,
    const int boxes_per_cell,
    const float lambda_giou,
    const float lambda_noobj,
    const float lambda_class,
    const float focal_gamma,
    const float obj_focal_gamma)
{
    // Clamp to [-clip, clip]; fminf maps a NaN to +clip before the lower bound applies.
    const auto clip = [](const float z) { return fmaxf(fminf(z, YOLO_GRAD_CLIP), -YOLO_GRAD_CLIP); };

    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n_boxes; i += Index(blockDim.x) * gridDim.x)
    {
        const float* out = output + i * values_per_box;
        const float* tgt = target + i * values_per_box;
        float* slot_delta = delta + i * values_per_box;

        // Same roles as the forward pass; ignored slots (NaN tau included) are not written.
        const float tau = tgt[4];

        if (tau >= 0.5f)
        {
            float pred[4], gt[4];
            const float center_scale = yolo_decode_cell_boxes(int(i), boxes_per_cell, grid_size, out, tgt, pred, gt);

            float iou, giou, cx_grad, cy_grad, w_grad, h_grad;
            yolo_ciou_grad(pred, gt, &iou, &giou, cx_grad, cy_grad, w_grad, h_grad);

            const float box_scale = lambda_giou * inv_batch;

            slot_delta[0] = box_scale * center_scale * clip(cx_grad);
            slot_delta[1] = box_scale * center_scale * clip(cy_grad);
            slot_delta[2] = box_scale * clip(w_grad);
            slot_delta[3] = box_scale * clip(h_grad);

            // The 1/(p(1-p)+eps) factors cancel the sigmoid derivative applied by the Detection layer.
            const float objectness = out[4];
            slot_delta[4] = (objectness - tau) / (objectness * (1.0f - objectness) + YOLO_EPSILON) * inv_batch;

            for (int c = 0; c < classes_number; ++c)
            {
                const float probability = out[5 + c];
                const float label = tgt[5 + c];

                if (sigmoid_classes != 0)
                {
                    // Focal factor held constant.
                    const float correct_probability = label > 0.5f ? probability : 1.0f - probability;
                    const float focal = __powf(1.0f - correct_probability, focal_gamma);
                    slot_delta[5 + c] = lambda_class * focal * (probability - label)
                                      / (probability * (1.0f - probability) + YOLO_EPSILON) * inv_batch;
                }
                else if (label > 0.0f)
                    slot_delta[5 + c] = -lambda_class * label / (probability + YOLO_EPSILON) * inv_batch;
            }
        }
        else if (tau > -0.5f)
        {
            const float objectness = out[4];
            float objectness_gradient;

            if (obj_focal_gamma == 0.0f)
                objectness_gradient = lambda_noobj * objectness / (objectness * (1.0f - objectness) + YOLO_EPSILON);
            else
            {
                // d/do of -o^gamma * ln(1 - o), both arguments floored at eps.
                const float complement = fmaxf(1.0f - objectness, YOLO_EPSILON);
                const float floored = fmaxf(objectness, YOLO_EPSILON);
                objectness_gradient = lambda_noobj * __powf(floored, obj_focal_gamma - 1.0f)
                                    * (-obj_focal_gamma * logf(complement) + objectness / complement);
            }

            slot_delta[4] = objectness_gradient * inv_batch;
        }
    }
}

__global__ void yolo_assemble_head_target_kernel(
    const int n,
    const float* __restrict__ target_flat,
    float* __restrict__ head_target,
    const int per_sample_floats,
    const int head_offset,
    const int head_floats)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const Index sample = i / head_floats;
        const Index j = i - sample * head_floats;
        head_target[i] = target_flat[sample * Index(per_sample_floats) + head_offset + j];
    }
}

void yolo_assemble_head_target_cuda(const float* target_flat, float* head_target,
                                    Index batch, Index per_sample_floats,
                                    Index head_offset, Index head_floats)
{
    launch_elementwise(batch * head_floats, yolo_assemble_head_target_kernel,
                       target_flat, head_target,
                       checked_int(per_sample_floats), checked_int(head_offset),
                       checked_int(head_floats));
}

void yolo_error_cuda(const float* output, const float* target, float* error_accumulator,
                     int batch, int grid, int boxes_per_cell, int values_per_box,
                     int classes_number, int sigmoid_classes,
                     float lambda_giou, float lambda_noobj, float lambda_class,
                     float focal_gamma, float obj_focal_gamma)
{
    launch_elementwise(Index(batch) * grid * grid * boxes_per_cell, yolo_loss_forward_kernel,
                       output, target, error_accumulator, values_per_box, classes_number, sigmoid_classes,
                       grid, boxes_per_cell, lambda_giou, lambda_noobj, lambda_class, focal_gamma, obj_focal_gamma);
}

void yolo_gradient_cuda(const float* output, const float* target, float* delta,
                        int batch, int grid, int boxes_per_cell, int values_per_box,
                        int classes_number, int sigmoid_classes, float inv_batch,
                        float lambda_giou, float lambda_noobj, float lambda_class,
                        float focal_gamma, float obj_focal_gamma)
{
    const Index n_boxes = Index(batch) * grid * grid * boxes_per_cell;
    if (n_boxes <= 0) return;

    // The kernel writes only the entries its slot roles define, so the rest must read as zero.
    opennn::device::set_zero_async(delta, n_boxes * values_per_box * Index(sizeof(float)),
                                   opennn::device::get_compute_stream());

    launch_elementwise(n_boxes, yolo_loss_gradient_kernel,
                       output, target, delta, values_per_box, classes_number, sigmoid_classes, inv_batch,
                       grid, boxes_per_cell, lambda_giou, lambda_noobj, lambda_class, focal_gamma, obj_focal_gamma);
}

#define INSTANTIATE(T) \
    template void binary_cross_entropy_cuda<T>(const Index, float*, const float*, const T*, const float); \
    template void binary_cross_entropy_logits_cuda<T>(const Index, float*, const float*, const T*); \
    template void binary_cross_entropy_gradient_cuda<T>(const Index, T*, const float*, const T*, const float, const float); \
    template void categorical_cross_entropy_cuda<T>(const Index, float*, const float*, const T*, const float); \
    template void categorical_cross_entropy_gradient_cuda<T>(const Index, T*, const float*, const T*, const float); \
    template void weighted_squared_error_cuda<T>(const Index, float*, const float*, const T*, const float, const float); \
    template void weighted_squared_error_gradient_cuda<T>(const Index, T*, const float*, const T*, const float, const float, const float); \
    template void cross_entropy_3d_multiple_forward_cuda<T>(const Index, const int, const T*, const float*, float*, float*, float*); \
    template void cross_entropy_3d_metrics_cuda<T>(const Index, const int, const T*, const float*, float*); \
    template void cross_entropy_3d_multiple_backward_cuda<T>(const Index, const int, const T*, const float*, T*, const float, const float*); \
    template void minkowski_error_cuda<T>(const Index, float*, const float*, const T*, const float); \
    template void minkowski_error_gradient_cuda<T>(const Index, T*, const float*, const T*, const float, const float);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

template void l1_gradient_cuda<float>(const Index, float*, const float*, const float);

template<typename TIn, typename TOut>
__global__ void scaled_diff_kernel(const int n,
                                   const TIn* __restrict__ input,
                                   const float* __restrict__ target,
                                   const float scale,
                                   TOut* __restrict__ output)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const float d = static_cast<float>(input[i]) - target[i];
        output[i] = static_cast<TOut>(scale * d);
    }
}

template<typename TIn, typename TOut>
__global__ void binary_cross_entropy_logits_gradient_kernel(const int n,
                                                            const TIn* __restrict__ logits,
                                                            const float* __restrict__ targets,
                                                            const float scale,
                                                            TOut* __restrict__ delta)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const float x = static_cast<float>(logits[i]);
        const float exponential = expf(-fabsf(x));
        const float y = targets[i];
        // Preserve small derivatives even when the sigmoid rounds to one.
        const float residual = x >= 0.0f
            ? ((1.0f - y) - y * exponential) / (1.0f + exponential)
            : ((1.0f - y) * exponential - y) / (1.0f + exponential);
        delta[i] = static_cast<TOut>(residual * scale);
    }
}

template<typename TIn, typename TOut>
void binary_cross_entropy_logits_gradient_cuda(const Index n, const TIn* logits,
                                                const float* targets, float scale, TOut* delta)
{
    launch_elementwise_strided(n, binary_cross_entropy_logits_gradient_kernel<TIn, TOut>,
                              logits, targets, scale, delta);
}

template<typename TIn, typename TOut>
void scaled_diff_cuda_typed(const Index n, const TIn* input, const float* target,
                            const float scale, TOut* output)
{
    launch_elementwise_strided(n, scaled_diff_kernel<TIn, TOut>, input, target, scale, output);
}

template<typename TIn, typename TOut>
__global__ void mean_squared_error_metrics_gradient_kernel(
    const int n, const int batch,
    const TIn* __restrict__ input,
    const float* __restrict__ target,
    TOut* __restrict__ delta,
    float* __restrict__ error_sum)
{
    const float inverse_batch = 1.0f / float(batch);
    float squared_sum = 0.0f;

    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += Index(blockDim.x) * gridDim.x)
    {
        const float difference = static_cast<float>(input[i]) - target[i];
        delta[i] = static_cast<TOut>(difference * inverse_batch);
        squared_sum = fmaf(difference, difference, squared_sum);
    }

    if (block_reduce_sum(squared_sum))
        atomicAdd(error_sum, squared_sum * (0.5f * inverse_batch));
}

template<typename TIn, typename TOut>
void mean_squared_error_metrics_gradient_cuda(
    const Index n, const Index batch,
    const TIn* input, const float* target,
    TOut* delta, float* error_sum)
{
    if (n == 0 || batch == 0) return;
    const int total = checked_int(n);
    OPENNN_CUDA_LAUNCH((mean_squared_error_metrics_gradient_kernel<TIn, TOut>
        <<<grid_size_strided_for(total), block_size, 0,
           opennn::device::get_compute_stream()>>>(
            total, checked_int(batch), input, target, delta, error_sum)));
}

#define INSTANTIATE2(TIn, TOut) \
    template void binary_cross_entropy_logits_gradient_cuda<TIn, TOut>(const Index, const TIn*, const float*, float, TOut*); \
    template void scaled_diff_cuda_typed<TIn, TOut>(const Index, const TIn*, const float*, float, TOut*); \
    template void mean_squared_error_metrics_gradient_cuda<TIn, TOut>(const Index, const Index, const TIn*, const float*, TOut*, float*);

OPENNN_INSTANTIATE_FLOAT_BF16_2(INSTANTIATE2)
#undef INSTANTIATE2
