#include "kernel_common.cuh"

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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(binary_cross_entropy_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, term_results, targets, outputs, epsilon));
}

template void binary_cross_entropy_cuda<float>        (const Index, float*, const float*, const float*,         const float);
template void binary_cross_entropy_cuda<__nv_bfloat16>(const Index, float*, const float*, const __nv_bfloat16*, const float);

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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(binary_cross_entropy_gradient_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, deltas, targets, outputs, epsilon, scaling_factor));
}

template void binary_cross_entropy_gradient_cuda<float>        (const Index, float*,         const float*, const float*,         const float, const float);
template void binary_cross_entropy_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const float*, const __nv_bfloat16*, const float, const float);

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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(categorical_cross_entropy_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, term_results, targets, outputs, epsilon));
}

template void categorical_cross_entropy_cuda<float>        (const Index, float*, const float*, const float*,         const float);
template void categorical_cross_entropy_cuda<__nv_bfloat16>(const Index, float*, const float*, const __nv_bfloat16*, const float);

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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(categorical_cross_entropy_gradient_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, deltas, targets, outputs, scaling_factor));
}

template void categorical_cross_entropy_gradient_cuda<float>        (const Index, float*,         const float*, const float*,         const float);
template void categorical_cross_entropy_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const float*, const __nv_bfloat16*, const float);

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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(weighted_squared_error_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, term_results, targets, outputs, positives_weight, negatives_weight));
}

template void weighted_squared_error_cuda<float>        (const Index, float*, const float*, const float*,         const float, const float);
template void weighted_squared_error_cuda<__nv_bfloat16>(const Index, float*, const float*, const __nv_bfloat16*, const float, const float);

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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(weighted_squared_error_gradient_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, deltas, targets, outputs, positives_weight, negatives_weight, scaling_factor));
}

template void weighted_squared_error_gradient_cuda<float>        (const Index, float*,         const float*, const float*,         const float, const float, const float);
template void weighted_squared_error_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const float*, const __nv_bfloat16*, const float, const float, const float);

template<typename T>
__global__ void cross_entropy_3d_multiple_forward_kernel(const int total_tokens,
                                                         const int vocab_size,
                                                         const T* __restrict__ outputs,
                                                         const float* __restrict__ targets,
                                                         float* __restrict__ errors,
                                                         float* __restrict__ valid_mask,
                                                         float* __restrict__ correct_mask,
                                                         const float epsilon)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < total_tokens; idx += Index(blockDim.x) * gridDim.x)
    {
        const int target_class = static_cast<int>(targets[idx]);
        const bool valid = target_class > 0 && target_class < vocab_size;

        errors[idx] = valid ? -logf(static_cast<float>(outputs[idx * vocab_size + target_class]) + epsilon) : 0.0f;
        if (valid_mask) valid_mask[idx] = valid ? 1.0f : 0.0f;

        if (correct_mask)
        {
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
                best_match = (best_index == target_class) ? 1.0f : 0.0f;
            }
            correct_mask[idx] = best_match;
        }
    }
}

template<typename T>
void cross_entropy_3d_multiple_forward_cuda(const Index n,
                                            const int vocab_size,
                                            const T* outputs,
                                            const float* targets,
                                            float* errors,
                                            float* valid_mask,
                                            float* correct_mask,
                                            const float epsilon)
{
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(cross_entropy_3d_multiple_forward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, vocab_size, outputs, targets, errors, valid_mask, correct_mask, epsilon));
}

template void cross_entropy_3d_multiple_forward_cuda<float>        (const Index, const int, const float*,         const float*, float*, float*, float*, const float);
template void cross_entropy_3d_multiple_forward_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, float*, float*, float*, const float);

template<typename T>
__global__ void cross_entropy_3d_multiple_backward_kernel(const int n,
                                                          const int vocab_size,
                                                          const T* __restrict__ outputs,
                                                          const float* __restrict__ targets,
                                                          T* __restrict__ output_deltas,
                                                          float scale_factor,
                                                          const float* __restrict__ active_count_device)
{
    if (active_count_device)
    {
        const float active_count = active_count_device[0];
        scale_factor = active_count > 0.0f ? 1.0f / active_count : 0.0f;
    }

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
    {
        const int token_index = idx / vocab_size;
        const int class_index = idx % vocab_size;
        const int target_class = static_cast<int>(targets[token_index]);

        if (target_class <= 0 || target_class >= vocab_size)
        {
            output_deltas[idx] = static_cast<T>(0.0f);
            continue;
        }

        output_deltas[idx] = static_cast<T>((static_cast<float>(outputs[idx]) - (class_index == target_class ? 1.0f : 0.0f)) * scale_factor);
    }
}

template<typename T>
void cross_entropy_3d_multiple_backward_cuda(const Index n,
                                             const int vocab_size,
                                             const T* outputs,
                                             const float* targets,
                                             T* output_deltas,
                                             const float scale_factor)
{
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(cross_entropy_3d_multiple_backward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, vocab_size, outputs, targets, output_deltas, scale_factor, nullptr));
}

template void cross_entropy_3d_multiple_backward_cuda<float>        (const Index, const int, const float*,         const float*, float*,         const float);
template void cross_entropy_3d_multiple_backward_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, __nv_bfloat16*, const float);

template<typename T>
void cross_entropy_3d_multiple_backward_device_count_cuda(const Index n,
                                                          const int vocab_size,
                                                          const T* outputs,
                                                          const float* targets,
                                                          T* output_deltas,
                                                          const float* active_count_device)
{
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(cross_entropy_3d_multiple_backward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, vocab_size, outputs, targets, output_deltas, 0.0f, active_count_device));
}

template void cross_entropy_3d_multiple_backward_device_count_cuda<float>        (const Index, const int, const float*,         const float*, float*,         const float*);
template void cross_entropy_3d_multiple_backward_device_count_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, __nv_bfloat16*, const float*);

__global__ void accumulate_scaled_metric_kernel(const float* __restrict__ value,
                                                const float scale,
                                                float* __restrict__ error_sum)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        error_sum[0] += value[0] * scale;
}

void accumulate_scaled_metric_cuda(const float* value, float scale, float* error_sum)
{
    OPENNN_CUDA_LAUNCH(accumulate_scaled_metric_kernel<<<1, 1, 0, opennn::device::get_compute_stream()>>>(value, scale, error_sum));
}

__global__ void accumulate_cross_entropy_3d_metrics_kernel(const float* __restrict__ values,
                                                           float* __restrict__ error_sum,
                                                           float* __restrict__ accuracy_sum)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

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
    OPENNN_CUDA_LAUNCH(accumulate_cross_entropy_3d_metrics_kernel<<<1, 1, 0, opennn::device::get_compute_stream()>>>(
        values, error_sum, accuracy_sum));
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
    constexpr int vec_width = 16 / sizeof(T);

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
    if (n == 0) return;

    constexpr int vec_width = 16 / sizeof(T);
    const int total = checked_int(n);

    const bool aligned = are_float4_aligned(deltas, parameters);

    const int n_vec = aligned ? (total / vec_width) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, vec_width));

    OPENNN_CUDA_LAUNCH(l1_gradient_kernel<T><<<grid_size, block_size, 0, opennn::device::get_compute_stream()>>>(n_vec, total, deltas, parameters, weight));
}

template void l1_gradient_cuda<float>        (const Index, float*,         const float*,         const float);
template void l1_gradient_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*, const float);

// ──────────────────────────────────────────────────────────────────────────────
// YOLO GIoU loss kernels
// ──────────────────────────────────────────────────────────────────────────────

static constexpr float YOLO_EPSILON       = 1e-7f;
static constexpr float YOLO_CORNER_EPS    = 1e-6f;
static constexpr float YOLO_GRAD_CLIP     = 10.0f;

__device__ __forceinline__ float yolo_giou_forward(const float* pred, const float* gt, float* out_iou)
{
    const float pl = pred[0] - 0.5f * pred[2];
    const float pr = pred[0] + 0.5f * pred[2];
    const float pt = pred[1] - 0.5f * pred[3];
    const float pb = pred[1] + 0.5f * pred[3];

    const float gl = gt[0] - 0.5f * gt[2];
    const float gr = gt[0] + 0.5f * gt[2];
    const float gt_ = gt[1] - 0.5f * gt[3];
    const float gb = gt[1] + 0.5f * gt[3];

    const float iw = fmaxf(0.0f, fminf(pr, gr) - fmaxf(pl, gl));
    const float ih = fmaxf(0.0f, fminf(pb, gb) - fmaxf(pt, gt_));
    const float inter = iw * ih;

    const float pa = pred[2] * pred[3];
    const float ga = gt[2] * gt[3];
    const float uni = pa + ga - inter;

    const float iou = (uni > 0.0f) ? inter / uni : 0.0f;
    *out_iou = iou;

    const float ew = fmaxf(pr, gr) - fminf(pl, gl);
    const float eh = fmaxf(pb, gb) - fminf(pt, gt_);
    const float enc = ew * eh;

    const float giou = (enc > 0.0f) ? (iou - (enc - uni) / enc) : iou;

    // CIoU: add center-distance and aspect-ratio penalty
    const float dx   = pred[0] - gt[0];
    const float dy   = pred[1] - gt[1];
    const float rho2 = dx*dx + dy*dy;
    const float c2   = ew*ew + eh*eh + YOLO_EPSILON;

    const float v_diff = atan2f(gt[2], gt[3]) - atan2f(pred[2], pred[3]);
    constexpr float INV_PI2 = 4.0f / (3.14159265f * 3.14159265f);
    const float v     = INV_PI2 * v_diff * v_diff;
    const float alpha = (iou > 0.0f) ? v / (1.0f - iou + v + YOLO_EPSILON) : 0.0f;

    return giou - rho2/c2 - alpha*v;
}

__device__ __forceinline__ float corner_max_grad(float a, float b)
{
    if (a > b + YOLO_CORNER_EPS) return 1.0f;
    if (a < b - YOLO_CORNER_EPS) return 0.0f;
    return 0.5f;
}

__device__ __forceinline__ float corner_min_grad(float a, float b)
{
    if (a < b - YOLO_CORNER_EPS) return 1.0f;
    if (a > b + YOLO_CORNER_EPS) return 0.0f;
    return 0.5f;
}

__device__ __forceinline__ void yolo_giou_grad(
    const float* pred, const float* gt,
    float* out_iou, float* out_giou,
    float& cx_grad, float& cy_grad, float& w_grad, float& h_grad)
{
    const float pw = pred[2], ph = pred[3];
    const float pl = pred[0] - 0.5f * pw;
    const float pr = pred[0] + 0.5f * pw;
    const float pt = pred[1] - 0.5f * ph;
    const float pb = pred[1] + 0.5f * ph;

    const float gl = gt[0] - 0.5f * gt[2];
    const float gr = gt[0] + 0.5f * gt[2];
    const float gt_ = gt[1] - 0.5f * gt[3];
    const float gb = gt[1] + 0.5f * gt[3];

    const float iw_raw = fminf(pr, gr) - fmaxf(pl, gl);
    const float ih_raw = fminf(pb, gb) - fmaxf(pt, gt_);
    const float iw = fmaxf(0.0f, iw_raw);
    const float ih = fmaxf(0.0f, ih_raw);
    const float inter = iw * ih;

    const float pa = pw * ph;
    const float uni = pa + gt[2] * gt[3] - inter;

    const float iou = (uni > 0.0f) ? inter / uni : 0.0f;
    *out_iou = iou;

    const float ew = fmaxf(pr, gr) - fminf(pl, gl);
    const float eh = fmaxf(pb, gb) - fminf(pt, gt_);
    const float enc = ew * eh;

    *out_giou = (enc > 0.0f) ? (iou - (enc - uni) / enc) : iou;

    const float alive = (iw_raw > 0.0f && ih_raw > 0.0f) ? 1.0f : 0.0f;

    const float d_il = alive * -corner_max_grad(pl, gl) * ih;
    const float d_ir = alive *  corner_min_grad(pr, gr) * ih;
    const float d_it = alive * -corner_max_grad(pt, gt_) * iw;
    const float d_ib = alive *  corner_min_grad(pb, gb) * iw;

    const float d_el = -corner_min_grad(pl, gl) * eh;
    const float d_er =  corner_max_grad(pr, gr) * eh;
    const float d_et = -corner_min_grad(pt, gt_) * ew;
    const float d_eb =  corner_max_grad(pb, gb) * ew;

    auto loss_grad_corner = [&](float d_inter, float d_area, float d_enc) -> float
    {
        const float d_uni  = d_area - d_inter;
        const float d_iou  = (uni > 0.0f) ? ((d_inter * uni - inter * d_uni) / (uni * uni)) : 0.0f;
        const float d_pen  = (enc > 0.0f) ? ((uni * d_enc - enc * d_uni) / (enc * enc)) : 0.0f;
        return -d_iou + d_pen;
    };

    const float d_loss_l = loss_grad_corner(d_il, -ph, d_el);
    const float d_loss_r = loss_grad_corner(d_ir,  ph, d_er);
    const float d_loss_t = loss_grad_corner(d_it, -pw, d_et);
    const float d_loss_b = loss_grad_corner(d_ib,  pw, d_eb);

    cx_grad = d_loss_l + d_loss_r;
    cy_grad = d_loss_t + d_loss_b;
    w_grad  = 0.5f * (d_loss_r - d_loss_l);
    h_grad  = 0.5f * (d_loss_b - d_loss_t);

    // CIoU extra gradient terms
    const float dx   = pred[0] - gt[0];
    const float dy   = pred[1] - gt[1];
    const float rho2 = dx*dx + dy*dy;
    const float c2   = ew*ew + eh*eh + YOLO_EPSILON;
    const float ic4  = 1.0f / (c2 * c2);

    const float dew_dcx = corner_max_grad(pr, gr) - corner_min_grad(pl, gl);
    const float deh_dcy = corner_max_grad(pb, gb) - corner_min_grad(pt, gt_);
    const float dew_dw  = 0.5f * (corner_max_grad(pr, gr) + corner_min_grad(pl, gl));
    const float deh_dh  = 0.5f * (corner_max_grad(pb, gb) + corner_min_grad(pt, gt_));
    cx_grad += (2.0f*dx*c2 - rho2*2.0f*ew*dew_dcx) * ic4;
    cy_grad += (2.0f*dy*c2 - rho2*2.0f*eh*deh_dcy) * ic4;
    w_grad  += -rho2 * 2.0f*ew*dew_dw * ic4;
    h_grad  += -rho2 * 2.0f*eh*deh_dh * ic4;

    const float v_diff = atan2f(gt[2], gt[3]) - atan2f(pred[2], pred[3]);
    constexpr float INV_PI2 = 4.0f / (3.14159265f * 3.14159265f);
    const float v     = INV_PI2 * v_diff * v_diff;
    const float alpha = (uni > 0.0f) ? v / (1.0f - iou + v + YOLO_EPSILON) : 0.0f;
    const float wh2   = pw*pw + ph*ph + YOLO_EPSILON;
    const float coeff = alpha * INV_PI2 * 2.0f * v_diff;
    w_grad += coeff * (-ph / wh2);
    h_grad += coeff * (pw / wh2);
}

// ── forward kernel ────────────────────────────────────────────────────────────

__global__ void yolo_loss_forward_kernel(
    const int n_boxes,            // batch * grid * grid * boxes_per_cell
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ error_accum,  // single float, pre-zeroed
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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_boxes; i += blockDim.x * gridDim.x)
    {
        const int base = i * values_per_box;

        if (target[base + 4] >= 0.5f)  // positive cell (anchor-IoU soft target)
        {
            const float* pred_raw = output + base;
            const float* gt_raw   = target + base;

            const int cell_idx = i / boxes_per_cell;
            const int col = cell_idx % grid_size;
            const int row = (cell_idx / grid_size) % grid_size;
            const float inv_grid = 1.0f / float(grid_size);
            const float pred[4] = {(pred_raw[0] + float(col)) * inv_grid, (pred_raw[1] + float(row)) * inv_grid, pred_raw[2], pred_raw[3]};
            const float gt[4]   = {(gt_raw[0]   + float(col)) * inv_grid, (gt_raw[1]   + float(row)) * inv_grid, gt_raw[2],   gt_raw[3]};

            float iou_unused;
            const float ciou = yolo_giou_forward(pred, gt, &iou_unused);

            float contrib = lambda_giou * (1.0f - ciou);
            // Soft BCE objectness: target = anchor-IoU (in [0.5,1.0]), trains calibrated confidence
            const float iou_t = target[base + 4];
            contrib -= iou_t * logf(pred_raw[4] + YOLO_EPSILON) + (1.0f - iou_t) * logf(1.0f - pred_raw[4] + YOLO_EPSILON);

            float class_contrib = 0.0f;
            if (sigmoid_classes)
            {
                for (int c = 0; c < classes_number; ++c)
                {
                    const float p = pred_raw[5 + c];
                    const float t = gt_raw[5 + c];
                    const float p_t   = (t > 0.5f) ? p : (1.0f - p);
                    const float focal = __powf(1.0f - p_t, focal_gamma);
                    class_contrib -= focal * (t * logf(p + YOLO_EPSILON) + (1.0f - t) * logf(1.0f - p + YOLO_EPSILON));
                }
            }
            else
            {
                for (int c = 0; c < classes_number; ++c)
                    if (gt_raw[5 + c] > 0.0f)
                        class_contrib -= logf(pred_raw[5 + c] + YOLO_EPSILON);
            }

            atomicAdd(error_accum, contrib + lambda_class * class_contrib);
        }
        else if (target[base + 4] > -0.5f)  // skip ignore slots (sentinel -1.0)
        {
            const float conf  = output[base + 4];
            const float w_bg  = (obj_focal_gamma > 0.0f) ? __powf(conf, obj_focal_gamma) : 1.0f;
            atomicAdd(error_accum, -lambda_noobj * w_bg * logf(1.0f - conf + YOLO_EPSILON));
        }
    }
}

// ── gradient kernel ───────────────────────────────────────────────────────────

__global__ void yolo_loss_gradient_kernel(
    const int n_boxes,
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ delta,        // pre-zeroed by wrapper
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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_boxes; i += blockDim.x * gridDim.x)
    {
        const int base = i * values_per_box;

        if (target[base + 4] >= 0.5f)  // positive cell (anchor-IoU soft target)
        {
            const float* pred_raw = output + base;
            const float* gt_raw   = target + base;

            const int cell_idx = i / boxes_per_cell;
            const int col = cell_idx % grid_size;
            const int row = (cell_idx / grid_size) % grid_size;
            const float inv_grid = 1.0f / float(grid_size);
            const float pred[4] = {(pred_raw[0] + float(col)) * inv_grid, (pred_raw[1] + float(row)) * inv_grid, pred_raw[2], pred_raw[3]};
            const float gt[4]   = {(gt_raw[0]   + float(col)) * inv_grid, (gt_raw[1]   + float(row)) * inv_grid, gt_raw[2],   gt_raw[3]};

            float iou, giou;
            float cx_g, cy_g, w_g, h_g;
            yolo_giou_grad(pred, gt, &iou, &giou, cx_g, cy_g, w_g, h_g);

            const float scale = lambda_giou * inv_batch;
            delta[base + 0] = scale * inv_grid * fmaxf(-YOLO_GRAD_CLIP, fminf(YOLO_GRAD_CLIP, cx_g));
            delta[base + 1] = scale * inv_grid * fmaxf(-YOLO_GRAD_CLIP, fminf(YOLO_GRAD_CLIP, cy_g));
            delta[base + 2] = scale * fmaxf(-YOLO_GRAD_CLIP, fminf(YOLO_GRAD_CLIP, w_g));
            delta[base + 3] = scale * fmaxf(-YOLO_GRAD_CLIP, fminf(YOLO_GRAD_CLIP, h_g));
            {
                const float c4    = pred_raw[4];
                const float iou_t = target[base + 4];  // soft anchor-IoU target in [0.5,1.0]
                delta[base + 4] = (c4 - iou_t) / (c4 * (1.0f - c4) + YOLO_EPSILON) * inv_batch;
            }

            if (sigmoid_classes)
            {
                for (int c = 0; c < classes_number; ++c)
                {
                    const float p = pred_raw[5 + c];
                    const float t = gt_raw[5 + c];
                    const float p_t   = (t > 0.5f) ? p : (1.0f - p);
                    const float focal = __powf(1.0f - p_t, focal_gamma);
                    delta[base + 5 + c] = lambda_class * focal * (p - t) / (p * (1.0f - p) + YOLO_EPSILON) * inv_batch;
                }
            }
            else
            {
                for (int c = 0; c < classes_number; ++c)
                    if (gt_raw[5 + c] > 0.0f)
                        delta[base + 5 + c] = lambda_class * (-gt_raw[5 + c] / (pred_raw[5 + c] + YOLO_EPSILON)) * inv_batch;
            }
        }
        else if (target[base + 4] > -0.5f)  // skip ignore slots (sentinel -1.0)
        {
            const float c4 = output[base + 4];
            float d4;
            if (obj_focal_gamma == 0.0f) {
                d4 = lambda_noobj * c4 / (c4 * (1.0f - c4) + YOLO_EPSILON);
            } else {
                // dL/dp for L=-p^γ*log(1-p): p^(γ-1)*(-γ*log(1-p)+p/(1-p))
                const float omc = fmaxf(1.0f - c4, YOLO_EPSILON);
                d4 = lambda_noobj * __powf(fmaxf(c4, YOLO_EPSILON), obj_focal_gamma - 1.0f)
                     * (-obj_focal_gamma * logf(omc) + c4 / omc);
            }
            delta[base + 4] = d4 * inv_batch;
        }
    }
}

// ── wrappers ──────────────────────────────────────────────────────────────────

void yolo_error_cuda(const float* output, const float* target, float* error_accumulator,
                     int batch, int grid, int boxes_per_cell, int values_per_box,
                     int classes_number, int sigmoid_classes,
                     float lambda_giou, float lambda_noobj, float lambda_class,
                     float focal_gamma, float obj_focal_gamma)
{
    const int n_boxes = batch * grid * grid * boxes_per_cell;
    if (n_boxes == 0) return;
    OPENNN_CUDA_LAUNCH(yolo_loss_forward_kernel<<<grid_size_for(n_boxes), block_size, 0,
        opennn::device::get_compute_stream()>>>(
        n_boxes, output, target, error_accumulator, values_per_box, classes_number, sigmoid_classes,
        grid, boxes_per_cell, lambda_giou, lambda_noobj, lambda_class, focal_gamma, obj_focal_gamma));
}

void yolo_gradient_cuda(const float* output, const float* target, float* delta,
                        int batch, int grid, int boxes_per_cell, int values_per_box,
                        int classes_number, int sigmoid_classes, float inv_batch,
                        float lambda_giou, float lambda_noobj, float lambda_class,
                        float focal_gamma, float obj_focal_gamma)
{
    const int n_boxes = batch * grid * grid * boxes_per_cell;
    if (n_boxes == 0) return;
    const int n_floats = n_boxes * values_per_box;
    cudaMemsetAsync(delta, 0, size_t(n_floats) * sizeof(float), opennn::device::get_compute_stream());
    OPENNN_CUDA_LAUNCH(yolo_loss_gradient_kernel<<<grid_size_for(n_boxes), block_size, 0,
        opennn::device::get_compute_stream()>>>(
        n_boxes, output, target, delta, values_per_box, classes_number, sigmoid_classes, inv_batch,
        grid, boxes_per_cell, lambda_giou, lambda_noobj, lambda_class, focal_gamma, obj_focal_gamma));
}
