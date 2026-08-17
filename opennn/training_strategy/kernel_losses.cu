//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/training_strategy/kernel_losses.cuh"

template<typename T>
__global__ void mean_absolute_error_gradient_kernel(
    const int n,
    T* __restrict__ deltas,
    const float* __restrict__ targets,
    const T* __restrict__ outputs,
    const float scale)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const float difference = static_cast<float>(outputs[i]) - targets[i];
        const float sign = difference > 0.0f ? 1.0f : (difference < 0.0f ? -1.0f : 0.0f);
        deltas[i] = static_cast<T>(scale * sign);
    }
}

template<typename T>
void mean_absolute_error_gradient_cuda(const Index n,
                                       T* deltas,
                                       const float* targets,
                                       const T* outputs,
                                       const float scale)
{
    launch_elementwise(n, mean_absolute_error_gradient_kernel<T>, deltas, targets, outputs, scale);
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
    launch_elementwise(n, cross_entropy_3d_multiple_forward_kernel<T>,
                       vocab_size, outputs, targets, errors, valid_mask, correct_mask, epsilon);
}


// The device-metrics form of the forward above: one warp per token (lanes
// stride over the vocabulary, coalesced), the per-token loss, validity and
// argmax-hit block-reduced and added into sums[0..2] (loss, active, correct).
// Replaces the per-token error/mask arrays and their three cublasSasum passes;
// sums must be zero on entry. Ties in the argmax go to the lowest class, as in
// the serial scan.
template<typename T>
__global__ void cross_entropy_3d_metrics_kernel(const int total_tokens, const int vocab_size,
                                                const T* __restrict__ outputs,
                                                const float* __restrict__ targets,
                                                const float epsilon,
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
        if (lane == 0)
        {
            loss += -logf(static_cast<float>(row[target_class]) + epsilon);
            active += 1.0f;
            correct += best_index == target_class ? 1.0f : 0.0f;
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
                                   const T* outputs, const float* targets,
                                   const float epsilon, float* sums)
{
    if (total_tokens <= 0) return;
    const int warps_per_block = block_size / 32;
    const Index needed = (total_tokens + warps_per_block - 1) / warps_per_block;
    const int blocks = int(needed < 4096 ? needed : 4096);
    OPENNN_CUDA_LAUNCH(cross_entropy_3d_metrics_kernel<T><<<blocks, block_size, 0, opennn::device::get_compute_stream()>>>(
        int(total_tokens), vocab_size, outputs, targets, epsilon, sums));
}

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
                                             const float scale_factor,
                                             const float* active_count_device)
{
    launch_elementwise(n, cross_entropy_3d_multiple_backward_kernel<T>,
                       vocab_size, outputs, targets, output_deltas, scale_factor, active_count_device);
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

// Everything the CIoU loss and its gradient share for one (pred, gt) pair of
// (cx, cy, w, h) boxes: corners, intersection, union, enclosure and the centre
// distance / aspect-ratio penalties.
struct CiouTerms
{
    float pl, pr, pt, pb;      // predicted box corners
    float gl, gr, gt_, gb;     // ground-truth box corners
    float iw_raw, ih_raw;      // signed intersection extents
    float iw, ih, inter;       // clamped intersection
    float uni, iou;
    float ew, eh, enc, giou;   // enclosing box and GIoU
    float dx, dy, rho2, c2;    // centre offset and enclosing diagonal
    float v_diff, v, alpha;    // aspect-ratio penalty
};

__device__ __forceinline__ CiouTerms ciou_terms(const float* pred, const float* gt)
{
    CiouTerms t;

    t.pl = pred[0] - 0.5f * pred[2];
    t.pr = pred[0] + 0.5f * pred[2];
    t.pt = pred[1] - 0.5f * pred[3];
    t.pb = pred[1] + 0.5f * pred[3];

    t.gl  = gt[0] - 0.5f * gt[2];
    t.gr  = gt[0] + 0.5f * gt[2];
    t.gt_ = gt[1] - 0.5f * gt[3];
    t.gb  = gt[1] + 0.5f * gt[3];

    t.iw_raw = fminf(t.pr, t.gr) - fmaxf(t.pl, t.gl);
    t.ih_raw = fminf(t.pb, t.gb) - fmaxf(t.pt, t.gt_);
    t.iw = fmaxf(0.0f, t.iw_raw);
    t.ih = fmaxf(0.0f, t.ih_raw);
    t.inter = t.iw * t.ih;

    const float pa = pred[2] * pred[3];
    const float ga = gt[2] * gt[3];
    t.uni = pa + ga - t.inter;

    t.iou = (t.uni > 0.0f) ? t.inter / t.uni : 0.0f;

    t.ew = fmaxf(t.pr, t.gr) - fminf(t.pl, t.gl);
    t.eh = fmaxf(t.pb, t.gb) - fminf(t.pt, t.gt_);
    t.enc = t.ew * t.eh;

    t.giou = (t.enc > 0.0f) ? (t.iou - (t.enc - t.uni) / t.enc) : t.iou;

    t.dx   = pred[0] - gt[0];
    t.dy   = pred[1] - gt[1];
    t.rho2 = t.dx*t.dx + t.dy*t.dy;
    t.c2   = t.ew*t.ew + t.eh*t.eh + YOLO_EPSILON;

    t.v_diff = atan2f(gt[2], gt[3]) - atan2f(pred[2], pred[3]);
    t.v      = YOLO_INV_PI2 * t.v_diff * t.v_diff;
    // Guard on uni (not iou) so the loss and its gradient agree for disjoint boxes.
    t.alpha  = (t.uni > 0.0f) ? t.v / (1.0f - t.iou + t.v + YOLO_EPSILON) : 0.0f;

    return t;
}

__device__ __forceinline__ float yolo_ciou_forward(const float* pred, const float* gt, float* out_iou)
{
    const CiouTerms t = ciou_terms(pred, gt);
    *out_iou = t.iou;

    return t.giou - t.rho2/t.c2 - t.alpha*t.v;
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

__device__ __forceinline__ void yolo_ciou_grad(
    const float* pred, const float* gt,
    float* out_iou, float* out_giou,
    float& cx_grad, float& cy_grad, float& w_grad, float& h_grad)
{
    const CiouTerms t = ciou_terms(pred, gt);
    const float pw = pred[2], ph = pred[3];
    const float pl = t.pl, pr = t.pr, pt = t.pt, pb = t.pb;
    const float gl = t.gl, gr = t.gr, gt_ = t.gt_, gb = t.gb;
    const float iw = t.iw, ih = t.ih, ew = t.ew, eh = t.eh;
    const float inter = t.inter, uni = t.uni, enc = t.enc;

    *out_iou  = t.iou;
    *out_giou = t.giou;

    const float alive = (t.iw_raw > 0.0f && t.ih_raw > 0.0f) ? 1.0f : 0.0f;

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

    const float dx = t.dx, dy = t.dy, rho2 = t.rho2, c2 = t.c2;
    const float ic4  = 1.0f / (c2 * c2);

    const float dew_dcx = corner_max_grad(pr, gr) - corner_min_grad(pl, gl);
    const float deh_dcy = corner_max_grad(pb, gb) - corner_min_grad(pt, gt_);
    const float dew_dw  = 0.5f * (corner_max_grad(pr, gr) + corner_min_grad(pl, gl));
    const float deh_dh  = 0.5f * (corner_max_grad(pb, gb) + corner_min_grad(pt, gt_));
    cx_grad += (2.0f*dx*c2 - rho2*2.0f*ew*dew_dcx) * ic4;
    cy_grad += (2.0f*dy*c2 - rho2*2.0f*eh*deh_dcy) * ic4;
    w_grad  += -rho2 * 2.0f*ew*dew_dw * ic4;
    h_grad  += -rho2 * 2.0f*eh*deh_dh * ic4;

    const float wh2   = pw*pw + ph*ph + YOLO_EPSILON;
    const float coeff = t.alpha * YOLO_INV_PI2 * 2.0f * t.v_diff;
    w_grad += coeff * (-ph / wh2);
    h_grad += coeff * (pw / wh2);
}

// Box i's cell-relative (x, y) offsets to grid-normalized centres, for both
// the prediction and its ground truth; returns 1 / grid_size.
__device__ __forceinline__ float yolo_decode_cell_boxes(const int i, const int boxes_per_cell, const int grid_size,
                                                        const float* pred_raw, const float* gt_raw,
                                                        float* pred, float* gt)
{
    const int cell_idx = i / boxes_per_cell;
    const int col = cell_idx % grid_size;
    const int row = (cell_idx / grid_size) % grid_size;
    const float inv_grid = 1.0f / float(grid_size);

    pred[0] = (pred_raw[0] + float(col)) * inv_grid;
    pred[1] = (pred_raw[1] + float(row)) * inv_grid;
    pred[2] = pred_raw[2];
    pred[3] = pred_raw[3];

    gt[0] = (gt_raw[0] + float(col)) * inv_grid;
    gt[1] = (gt_raw[1] + float(row)) * inv_grid;
    gt[2] = gt_raw[2];
    gt[3] = gt_raw[3];

    return inv_grid;
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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_boxes; i += blockDim.x * gridDim.x)
    {
        const int base = i * values_per_box;

        if (target[base + 4] >= 0.5f)
        {
            const float* pred_raw = output + base;
            const float* gt_raw   = target + base;

            float pred[4], gt[4];
            yolo_decode_cell_boxes(i, boxes_per_cell, grid_size, pred_raw, gt_raw, pred, gt);

            float iou_unused;
            const float ciou = yolo_ciou_forward(pred, gt, &iou_unused);

            float contrib = lambda_giou * (1.0f - ciou);

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
        else if (target[base + 4] > -0.5f)
        {
            const float conf  = output[base + 4];
            const float w_bg  = (obj_focal_gamma > 0.0f) ? __powf(conf, obj_focal_gamma) : 1.0f;
            atomicAdd(error_accum, -lambda_noobj * w_bg * logf(1.0f - conf + YOLO_EPSILON));
        }
    }
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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_boxes; i += blockDim.x * gridDim.x)
    {
        const int base = i * values_per_box;

        if (target[base + 4] >= 0.5f)
        {
            const float* pred_raw = output + base;
            const float* gt_raw   = target + base;

            float pred[4], gt[4];
            const float inv_grid = yolo_decode_cell_boxes(i, boxes_per_cell, grid_size, pred_raw, gt_raw, pred, gt);

            float iou, giou;
            float cx_g, cy_g, w_g, h_g;
            yolo_ciou_grad(pred, gt, &iou, &giou, cx_g, cy_g, w_g, h_g);

            const float scale = lambda_giou * inv_batch;
            delta[base + 0] = scale * inv_grid * fmaxf(-YOLO_GRAD_CLIP, fminf(YOLO_GRAD_CLIP, cx_g));
            delta[base + 1] = scale * inv_grid * fmaxf(-YOLO_GRAD_CLIP, fminf(YOLO_GRAD_CLIP, cy_g));
            delta[base + 2] = scale * fmaxf(-YOLO_GRAD_CLIP, fminf(YOLO_GRAD_CLIP, w_g));
            delta[base + 3] = scale * fmaxf(-YOLO_GRAD_CLIP, fminf(YOLO_GRAD_CLIP, h_g));
            {
                const float c4    = pred_raw[4];
                const float iou_t = target[base + 4];
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
        else if (target[base + 4] > -0.5f)
        {
            const float c4 = output[base + 4];
            float d4;
            if (obj_focal_gamma == 0.0f) {
                d4 = lambda_noobj * c4 / (c4 * (1.0f - c4) + YOLO_EPSILON);
            } else {

                const float omc = fmaxf(1.0f - c4, YOLO_EPSILON);
                d4 = lambda_noobj * __powf(fmaxf(c4, YOLO_EPSILON), obj_focal_gamma - 1.0f)
                     * (-obj_focal_gamma * logf(omc) + c4 / omc);
            }
            delta[base + 4] = d4 * inv_batch;
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
    const int n_boxes = batch * grid * grid * boxes_per_cell;
    if (n_boxes == 0) return;
    opennn::device::set_zero_async(delta, Index(n_boxes) * values_per_box * Index(sizeof(float)),
                                   opennn::device::get_compute_stream());
    launch_elementwise(n_boxes, yolo_loss_gradient_kernel,
                       output, target, delta, values_per_box, classes_number, sigmoid_classes, inv_batch,
                       grid, boxes_per_cell, lambda_giou, lambda_noobj, lambda_class, focal_gamma, obj_focal_gamma);
}

#define INSTANTIATE(T) \
    template void binary_cross_entropy_cuda<T>(const Index, float*, const float*, const T*, const float); \
    template void binary_cross_entropy_gradient_cuda<T>(const Index, T*, const float*, const T*, const float, const float); \
    template void categorical_cross_entropy_cuda<T>(const Index, float*, const float*, const T*, const float); \
    template void categorical_cross_entropy_gradient_cuda<T>(const Index, T*, const float*, const T*, const float); \
    template void weighted_squared_error_cuda<T>(const Index, float*, const float*, const T*, const float, const float); \
    template void weighted_squared_error_gradient_cuda<T>(const Index, T*, const float*, const T*, const float, const float, const float); \
    template void cross_entropy_3d_multiple_forward_cuda<T>(const Index, const int, const T*, const float*, float*, float*, float*, const float); \
    template void cross_entropy_3d_metrics_cuda<T>(const Index, const int, const T*, const float*, const float, float*); \
    template void cross_entropy_3d_multiple_backward_cuda<T>(const Index, const int, const T*, const float*, T*, const float, const float*); \
    template void mean_absolute_error_gradient_cuda<T>(const Index, T*, const float*, const T*, float);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

template void l1_gradient_cuda<float>(const Index, float*, const float*, const float);

// Scaled elementwise difference, the shared front half of the squared-error family.
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
void scaled_diff_cuda_typed(const Index n, const TIn* input, const float* target,
                            const float scale, TOut* output)
{
    launch_elementwise_strided(n, scaled_diff_kernel<TIn, TOut>, input, target, scale, output);
}

#define INSTANTIATE2(TIn, TOut) \
    template void scaled_diff_cuda_typed<TIn, TOut>(const Index, const TIn*, const float*, float, TOut*);

OPENNN_INSTANTIATE_FLOAT_BF16_2(INSTANTIATE2)
#undef INSTANTIATE2

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
