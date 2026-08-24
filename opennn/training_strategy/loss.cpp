//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/training_strategy/loss.h"

#include <Eigen/LU>

#include "opennn/core/memory_debug.h"
#include "opennn/core/profiler.h"
#include "opennn/core/statistics.h"
#include "opennn/core/tensor_types.h"
#include "opennn/dataset/batch.h"
#include "opennn/dataset/dataset.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/detection_head.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/error_functions.h"
#include "opennn/training_strategy/kernel_losses.cuh"

namespace opennn
{

#ifndef OPENNN_NO_VISION
namespace
{

struct GIoUResult
{
    float giou = 0.0f;
    float iou  = 0.0f;
    float cx_gradient = 0.0f;
    float cy_gradient = 0.0f;
    float w_gradient  = 0.0f;
    float h_gradient  = 0.0f;
};

constexpr float INV_PI2 = 4.0f / (numbers::pi_v<float> * numbers::pi_v<float>);

GIoUResult yolo_loss_giou_forward(const float* pred, const float* gt)
{
    const float predicted_left = pred[0] - 0.5f * pred[2];
    const float predicted_right = pred[0] + 0.5f * pred[2];
    const float predicted_top = pred[1] - 0.5f * pred[3];
    const float predicted_bottom = pred[1] + 0.5f * pred[3];
    const float ground_left = gt[0] - 0.5f * gt[2];
    const float ground_right = gt[0] + 0.5f * gt[2];
    const float ground_top = gt[1] - 0.5f * gt[3];
    const float ground_bottom = gt[1] + 0.5f * gt[3];

    const float intersection_width = max(0.0f, min(predicted_right, ground_right) - max(predicted_left, ground_left));
    const float intersection_height = max(0.0f, min(predicted_bottom, ground_bottom) - max(predicted_top, ground_top));
    const float intersection_area = intersection_width * intersection_height;

    const float predicted_area = pred[2] * pred[3];
    const float ground_area = gt[2] * gt[3];
    const float union_area = predicted_area + ground_area - intersection_area;

    const float enclosing_width = max(predicted_right, ground_right) - min(predicted_left, ground_left);
    const float enclosing_height = max(predicted_bottom, ground_bottom) - min(predicted_top, ground_top);
    const float enclosing_area = enclosing_width * enclosing_height;

    GIoUResult r;
    r.iou  = (union_area > 0.0f) ? (intersection_area / union_area) : 0.0f;
    r.giou = (enclosing_area > 0.0f) ? (r.iou - (enclosing_area - union_area) / enclosing_area) : r.iou;

    const float dx   = pred[0] - gt[0];
    const float dy   = pred[1] - gt[1];
    const float rho2 = dx*dx + dy*dy;
    const float c2   = enclosing_width*enclosing_width + enclosing_height*enclosing_height + EPSILON;

    const float v_diff = atan2f(gt[2], gt[3]) - atan2f(pred[2], pred[3]);
    const float v     = INV_PI2 * v_diff * v_diff;
    const float alpha = (union_area > 0.0f) ? v / (1.0f - r.iou + v + EPSILON) : 0.0f;

    r.giou -= rho2/c2 + alpha*v;
    return r;
}

GIoUResult yolo_loss_giou_grad(const float* pred, const float* gt)
{
    const float predicted_width = pred[2];
    const float predicted_height = pred[3];

    const float predicted_left = pred[0] - 0.5f * predicted_width;
    const float predicted_right = pred[0] + 0.5f * predicted_width;
    const float predicted_top = pred[1] - 0.5f * predicted_height;
    const float predicted_bottom = pred[1] + 0.5f * predicted_height;
    const float ground_left = gt[0] - 0.5f * gt[2];
    const float ground_right = gt[0] + 0.5f * gt[2];
    const float ground_top = gt[1] - 0.5f * gt[3];
    const float ground_bottom = gt[1] + 0.5f * gt[3];

    const float intersection_width_raw = min(predicted_right, ground_right) - max(predicted_left, ground_left);
    const float intersection_height_raw = min(predicted_bottom, ground_bottom) - max(predicted_top, ground_top);
    const float intersection_width = max(0.0f, intersection_width_raw);
    const float intersection_height = max(0.0f, intersection_height_raw);
    const float intersection_area = intersection_width * intersection_height;

    const float predicted_area = predicted_width * predicted_height;
    const float ground_area = gt[2] * gt[3];
    const float union_area = predicted_area + ground_area - intersection_area;

    const float enclosing_width = max(predicted_right, ground_right) - min(predicted_left, ground_left);
    const float enclosing_height = max(predicted_bottom, ground_bottom) - min(predicted_top, ground_top);
    const float enclosing_area = enclosing_width * enclosing_height;

    GIoUResult r;
    r.iou  = (union_area > 0.0f) ? (intersection_area / union_area) : 0.0f;
    r.giou = (enclosing_area > 0.0f) ? (r.iou - (enclosing_area - union_area) / enclosing_area) : r.iou;

    constexpr float corner_eps = 1e-6f;
    auto max_grad = [&](float a, float b) -> float {
        if (a > b + corner_eps) return 1.0f;
        if (a < b - corner_eps) return 0.0f;
        return 0.5f;
    };
    auto min_grad = [&](float a, float b) -> float {
        if (a < b - corner_eps) return 1.0f;
        if (a > b + corner_eps) return 0.0f;
        return 0.5f;
    };

    const float intersection_alive = (intersection_width_raw > 0.0f && intersection_height_raw > 0.0f) ? 1.0f : 0.0f;
    const float d_intersection_left = intersection_alive * -max_grad(predicted_left, ground_left) * intersection_height;
    const float d_intersection_right = intersection_alive *  min_grad(predicted_right, ground_right) * intersection_height;
    const float d_intersection_top = intersection_alive * -max_grad(predicted_top, ground_top) * intersection_width;
    const float d_intersection_bottom = intersection_alive *  min_grad(predicted_bottom, ground_bottom) * intersection_width;

    const float d_enclosing_left = -min_grad(predicted_left, ground_left) * enclosing_height;
    const float d_enclosing_right =  max_grad(predicted_right, ground_right) * enclosing_height;
    const float d_enclosing_top = -min_grad(predicted_top, ground_top) * enclosing_width;
    const float d_enclosing_bottom =  max_grad(predicted_bottom, ground_bottom) * enclosing_width;

    const float d_area_left = -predicted_height;
    const float d_area_right =  predicted_height;
    const float d_area_top = -predicted_width;
    const float d_area_bottom =  predicted_width;

    auto loss_grad_corner = [&](float d_intersection, float d_area, float d_enclosing) -> float
    {
        const float d_union = d_area - d_intersection;
        const float d_iou = (union_area > 0.0f) ? ((d_intersection * union_area - intersection_area * d_union) / (union_area * union_area)) : 0.0f;
        const float d_penalty = (enclosing_area > 0.0f) ? ((union_area * d_enclosing - enclosing_area * d_union) / (enclosing_area * enclosing_area)) : 0.0f;
        return -d_iou + d_penalty;
    };

    const float d_loss_left = loss_grad_corner(d_intersection_left, d_area_left, d_enclosing_left);
    const float d_loss_right = loss_grad_corner(d_intersection_right, d_area_right, d_enclosing_right);
    const float d_loss_top = loss_grad_corner(d_intersection_top, d_area_top, d_enclosing_top);
    const float d_loss_bottom = loss_grad_corner(d_intersection_bottom, d_area_bottom, d_enclosing_bottom);

    r.cx_gradient = d_loss_left + d_loss_right;
    r.cy_gradient = d_loss_top + d_loss_bottom;
    r.w_gradient  = 0.5f * (d_loss_right - d_loss_left);
    r.h_gradient  = 0.5f * (d_loss_bottom - d_loss_top);

    const float dx   = pred[0] - gt[0];
    const float dy   = pred[1] - gt[1];
    const float rho2 = dx*dx + dy*dy;
    const float c2   = enclosing_width*enclosing_width + enclosing_height*enclosing_height + EPSILON;
    const float ic4  = 1.0f / (c2 * c2);

    const float dew_dcx = max_grad(predicted_right, ground_right) - min_grad(predicted_left, ground_left);
    const float deh_dcy = max_grad(predicted_bottom, ground_bottom) - min_grad(predicted_top, ground_top);
    const float dew_dw  = 0.5f * (max_grad(predicted_right, ground_right) + min_grad(predicted_left, ground_left));
    const float deh_dh  = 0.5f * (max_grad(predicted_bottom, ground_bottom) + min_grad(predicted_top, ground_top));
    r.cx_gradient += (2.0f*dx*c2 - rho2*2.0f*enclosing_width*dew_dcx) * ic4;
    r.cy_gradient += (2.0f*dy*c2 - rho2*2.0f*enclosing_height*deh_dcy) * ic4;
    r.w_gradient  += -rho2 * 2.0f*enclosing_width*dew_dw * ic4;
    r.h_gradient  += -rho2 * 2.0f*enclosing_height*deh_dh * ic4;

    const float v_diff = atan2f(gt[2], gt[3]) - atan2f(pred[2], pred[3]);
    const float v     = INV_PI2 * v_diff * v_diff;
    const float alpha = (union_area > 0.0f) ? v / (1.0f - r.iou + v + EPSILON) : 0.0f;
    const float wh2   = predicted_width*predicted_width + predicted_height*predicted_height + EPSILON;
    const float coeff = alpha * INV_PI2 * 2.0f * v_diff;
    r.w_gradient  += coeff * (-predicted_height / wh2);
    r.h_gradient  += coeff * (predicted_width  / wh2);

    return r;
}

DetectionHeadMetadata get_detection_head_metadata(const NeuralNetwork& neural_network,
                                                  Index layer_index)
{
    const auto* const head = dynamic_cast<const DetectionHeadEndpoint*>(
        neural_network.get_layer(layer_index).get());
    throw_if(!head, "YOLO loss requires detection-head endpoints.");

    const DetectionHeadMetadata metadata = head->get_detection_head_metadata();
    throw_if(metadata.classes_number <= 0,
             "YOLO detection head must declare at least one class.");
    throw_if(metadata.boxes_per_cell <= 0,
             "YOLO detection head must declare at least one box per cell.");
    throw_if(metadata.regression_bins <= 0,
             "YOLO detection head must declare at least one regression bin.");
    return metadata;
}

DetectionHeadMetadata get_loss_head_metadata(const NeuralNetwork& neural_network,
                                             const vector<Index>& detection_indices)
{
    throw_if(detection_indices.empty(),
             "YOLO loss requires at least one detection-head endpoint.");

    const DetectionHeadMetadata first =
        get_detection_head_metadata(neural_network, detection_indices.front());

    for (size_t i = 1; i < detection_indices.size(); ++i)
    {
        const DetectionHeadMetadata current =
            get_detection_head_metadata(neural_network, detection_indices[i]);
        throw_if(current.kind != first.kind
                 || current.classes_number != first.classes_number,
                 "YOLO detection heads must use the same kind and class count.");
        throw_if(first.kind == DetectionHeadKind::AnchorBased
                 && (current.boxes_per_cell != first.boxes_per_cell
                     || current.class_activation != first.class_activation),
                 "Anchor-based YOLO heads must use the same box count and class activation.");
    }

    return first;
}

}

float yolo_error_kernel(const TensorView& output,
                        const TensorView& target,
                        Index boxes_per_cell,
                        Index classes_number,
                        bool sigmoid_classes,
                        YoloLambdas lam)
{
    const float lambda_giou      = lam.giou;
    const float lambda_noobject  = lam.noobj;
    const float lambda_class     = lam.cls;

    const Index values_per_box = 5 + classes_number;
    const Index batch_size = output.get_shape()[0];
    const Index grid_size  = output.get_shape()[1];
    const Index grid_width = output.get_shape()[2];
    const Index channels   = output.get_shape()[3];

    const float* out = output.as<float>();
    const float* tgt = target.as<float>();

    float coordinate_loss = 0.0f;
    float object_loss = 0.0f;
    float noobject_loss = 0.0f, noobject_comp = 0.0f;
    float class_loss = 0.0f;

    for (Index n = 0; n < batch_size; ++n)
        for (Index row = 0; row < grid_size; ++row)
            for (Index col = 0; col < grid_width; ++col)
            {
                const Index cell = ((n * grid_size + row) * grid_width + col) * channels;

                for (Index box = 0; box < boxes_per_cell; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    if (tgt[base + 4] >= 0.5f)
                    {
                        const float inv_grid = 1.0f / float(grid_size);
                        const float output_box[4] = {(out[base + 0] + float(col)) * inv_grid, (out[base + 1] + float(row)) * inv_grid, out[base + 2], out[base + 3]};
                        const float target_box[4] = {(tgt[base + 0] + float(col)) * inv_grid, (tgt[base + 1] + float(row)) * inv_grid, tgt[base + 2], tgt[base + 3]};
                        const GIoUResult g = yolo_loss_giou_forward(output_box, target_box);

                        coordinate_loss += 1.0f - g.giou;

                        const float iou_t = tgt[base + 4];
                        object_loss -= iou_t * log(out[base + 4] + EPSILON) + (1.0f - iou_t) * log(1.0f - out[base + 4] + EPSILON);

                        if (sigmoid_classes)
                        {
                            for (Index c = 0; c < classes_number; ++c)
                            {
                                const float p = out[base + 5 + c];
                                const float t = tgt[base + 5 + c];
                                const float p_t   = (t > 0.5f) ? p : (1.0f - p);
                                const float focal = pow(1.0f - p_t, lam.focal_gamma);
                                class_loss -= focal * (t * log(p + EPSILON) + (1.0f - t) * log(1.0f - p + EPSILON));
                            }
                        }
                        else
                        {
                            for (Index c = 0; c < classes_number; ++c)
                                if (tgt[base + 5 + c] > 0.0f)
                                    class_loss -= log(out[base + 5 + c] + EPSILON);
                        }
                    }
                    else if (tgt[base + 4] > -0.5f)
                    {
                        const float c4 = out[base + 4];
                        const float w_bg = (lam.obj_focal_gamma > 0.0f)
                                           ? pow(c4, lam.obj_focal_gamma) : 1.0f;

                        const float term = w_bg * log(1.0f - c4 + EPSILON);
                        const float kahan_y = -term - noobject_comp;
                        const float kahan_t = noobject_loss + kahan_y;
                        noobject_comp = (kahan_t - noobject_loss) - kahan_y;
                        noobject_loss = kahan_t;
                    }
                }
            }

    return lambda_giou * coordinate_loss + object_loss
         + lambda_noobject * noobject_loss + lambda_class * class_loss;
}

namespace
{

}

#ifdef _MSC_VER
#pragma optimize("", off)
#endif
void yolo_gradient_kernel(const TensorView& output,
                          const TensorView& target,
                          const TensorView& output_delta,
                          Index boxes_per_cell,
                          Index classes_number,
                          bool sigmoid_classes,
                          float inv_batch,
                          YoloLambdas lam)
{
    const float lambda_giou      = lam.giou;
    const float lambda_noobject  = lam.noobj;
    const float lambda_class     = lam.cls;
    constexpr float grad_clip = 10.0f;

    const Index values_per_box = 5 + classes_number;
    const Index batch_size = output.get_shape()[0];
    const Index grid_size  = output.get_shape()[1];
    const Index grid_width = output.get_shape()[2];
    const Index channels   = output.get_shape()[3];

    const float* out = output.as<float>();
    const float* tgt = target.as<float>();
    float* delta = output_delta.as<float>();

    fill_n(delta, output_delta.size(), 0.0f);

    for (Index n = 0; n < batch_size; ++n)
        for (Index row = 0; row < grid_size; ++row)
            for (Index col = 0; col < grid_width; ++col)
            {
                const Index cell = ((n * grid_size + row) * grid_width + col) * channels;

                for (Index box = 0; box < boxes_per_cell; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    if (tgt[base + 4] >= 0.5f)
                    {
                        const float inv_grid = 1.0f / float(grid_size);
                        const float output_box[4] = {(out[base + 0] + float(col)) * inv_grid, (out[base + 1] + float(row)) * inv_grid, out[base + 2], out[base + 3]};
                        const float target_box[4] = {(tgt[base + 0] + float(col)) * inv_grid, (tgt[base + 1] + float(row)) * inv_grid, tgt[base + 2], tgt[base + 3]};
                        const GIoUResult g = yolo_loss_giou_grad(output_box, target_box);

                        const float scale = lambda_giou * inv_batch;
                        delta[base + 0] = scale * inv_grid * clamp(g.cx_gradient, -grad_clip, grad_clip);
                        delta[base + 1] = scale * inv_grid * clamp(g.cy_gradient, -grad_clip, grad_clip);
                        delta[base + 2] = scale * clamp(g.w_gradient,  -grad_clip, grad_clip);
                        delta[base + 3] = scale * clamp(g.h_gradient,  -grad_clip, grad_clip);
                        {
                            const float c4 = out[base + 4];
                            const float iou_t = tgt[base + 4];
                            delta[base + 4] = (c4 - iou_t) / (c4 * (1.0f - c4) + EPSILON) * inv_batch;
                        }

                        if (sigmoid_classes)
                        {
                            for (Index c = 0; c < classes_number; ++c)
                            {
                                const float p = out[base + 5 + c];
                                const float t = tgt[base + 5 + c];
                                const float p_t   = (t > 0.5f) ? p : (1.0f - p);
                                const float focal = pow(1.0f - p_t, lam.focal_gamma);
                                delta[base + 5 + c] = lambda_class * focal * (p - t) / (p * (1.0f - p) + EPSILON) * inv_batch;
                            }
                        }
                        else
                        {
                            for (Index c = 0; c < classes_number; ++c)
                                if (tgt[base + 5 + c] > 0.0f)
                                    delta[base + 5 + c] = lambda_class * (-tgt[base + 5 + c] / (out[base + 5 + c] + EPSILON)) * inv_batch;
                        }
                    }
                    else if (tgt[base + 4] > -0.5f)
                    {
                        const float c4 = out[base + 4];
                        float d4;
                        if (lam.obj_focal_gamma == 0.0f) {
                            d4 = lambda_noobject * c4 / (c4 * (1.0f - c4) + EPSILON);
                        } else {

                            const float g   = lam.obj_focal_gamma;
                            const float omc = max(1.0f - c4, EPSILON);
                            d4 = lambda_noobject * pow(max(c4, EPSILON), g - 1.0f)
                                 * (-g * log(omc) + c4 / omc);
                        }
                        delta[base + 4] = d4 * inv_batch;
                    }
                }
            }
}
#ifdef _MSC_VER
#pragma optimize("", on)
#endif

namespace
{

vector<float> assemble_head_target(const float* tgt,
                                   Index batch_size,
                                   Index per_sample_floats,
                                   Index head_offset,
                                   Index head_floats)
{
    vector<float> head_target(size_t(batch_size) * size_t(head_floats));
    for (Index n = 0; n < batch_size; ++n)
        copy_n(tgt + n * per_sample_floats + head_offset,
               head_floats,
               head_target.data() + n * head_floats);
    return head_target;
}

template <typename LayoutFn>
void for_each_yolo_head_layout(const NeuralNetwork* nn,
                               const vector<Index>& detection_indices,
                               LayoutFn&& fn)
{
    Index per_sample_floats = 0;
    for (Index idx : detection_indices)
    {
        const Shape head_shape = nn->get_layer(idx)->get_output_shape();
        per_sample_floats += head_shape[0] * head_shape[1] * head_shape[2];
    }

    Index head_offset = 0;
    for (Index detection_idx : detection_indices)
    {
        const Shape head_shape = nn->get_layer(detection_idx)->get_output_shape();
        const Index channels = head_shape[2];
        const Index head_floats = head_shape[0] * head_shape[1] * channels;

        fn(detection_idx, head_shape, channels, per_sample_floats, head_offset, head_floats);

        head_offset += head_floats;
    }
}

template <typename HeadFn>
void for_each_yolo_head(const ForwardPropagation& forward_propagation,
                        const NeuralNetwork* nn,
                        const vector<Index>& detection_indices,
                        const float* tgt,
                        Index batch_size,
                        HeadFn&& fn)
{
    for_each_yolo_head_layout(nn, detection_indices,
        [&](Index detection_idx, const Shape& head_shape, Index channels,
            Index per_sample_floats, Index head_offset, Index head_floats)
        {
            const TensorView head_output = forward_propagation.slots[size_t(detection_idx)].back();

            const Shape target_shape({batch_size, head_shape[0], head_shape[1], channels});

            if (head_floats == per_sample_floats)
            {
                const TensorView whole(const_cast<float*>(tgt), target_shape, Type::FP32);

                fn(detection_idx, head_output, whole);
                return;
            }

            vector<float> head_target = assemble_head_target(tgt, batch_size, per_sample_floats,
                                                             head_offset, head_floats);
            const TensorView head_target_view(head_target.data(), target_shape, Type::FP32);

            fn(detection_idx, head_output, head_target_view);
        });
}

Loss::EvaluationResult yolo_error_cpu_multi(const ForwardPropagation& forward_propagation,
                                            const TensorView& target_flat,
                                            const NeuralNetwork* nn,
                                            const vector<Index>& detection_indices,
                                            const DetectionHeadMetadata& head,
                                            YoloLambdas lam)
{
    const Index batch_size = target_flat.get_shape()[0];
    const bool sigmoid_classes = head.uses_sigmoid_classes();

    float total_error = 0.0f;
    for_each_yolo_head(forward_propagation, nn, detection_indices,
                       target_flat.as<float>(), batch_size,
        [&](Index, const TensorView& head_output, const TensorView& head_target)
        {
            total_error += yolo_error_kernel(head_output, head_target,
                                             head.boxes_per_cell,
                                             head.classes_number,
                                             sigmoid_classes,
                                             lam);
        });

    return {.error = total_error / float(batch_size)};
}

void yolo_gradient_cpu_multi(const ForwardPropagation& forward_propagation,
                             const TensorView& target_flat,
                             BackPropagation& back_propagation,
                             const NeuralNetwork* nn,
                             const vector<Index>& detection_indices,
                             const DetectionHeadMetadata& head,
                             YoloLambdas lam)
{
    const Index batch_size = target_flat.get_shape()[0];
    const float inv_batch = 1.0f / float(batch_size);
    const bool sigmoid_classes = head.uses_sigmoid_classes();

    for_each_yolo_head(forward_propagation, nn, detection_indices,
                       target_flat.as<float>(), batch_size,
        [&](Index detection_idx, const TensorView& head_output, const TensorView& head_target)
        {
            TensorView& head_delta = back_propagation.output_deltas[size_t(detection_idx)];
            yolo_gradient_kernel(head_output, head_target, head_delta,
                                 head.boxes_per_cell,
                                 head.classes_number,
                                 sigmoid_classes,
                                 inv_batch,
                                 lam);
        });
}

static constexpr float TAL_ALPHA = 0.0f;
static constexpr float TAL_BETA  = 6.0f;
static constexpr Index TAL_TOP_K = 10;

struct TalResult {
    vector<Index> assign;
    vector<float> iou_map;
};

static float iou_cxcywh(float cx1, float cy1, float w1, float h1,
                         float cx2, float cy2, float w2, float h2)
{
    const float iw = max(0.0f, min(cx1+w1*0.5f, cx2+w2*0.5f) - max(cx1-w1*0.5f, cx2-w2*0.5f));
    const float ih = max(0.0f, min(cy1+h1*0.5f, cy2+h2*0.5f) - max(cy1-h1*0.5f, cy2-h2*0.5f));
    const float inter = iw * ih;
    return inter / (w1*h1 + w2*h2 - inter + 1e-7f);
}

static void dfl_decode_box(const float* box_logits, Index reg_max, Index col, Index row, Index G,
                            float& pred_cx, float& pred_cy, float& pred_w, float& pred_h)
{
    const float inv_g   = 1.0f / float(G);

    if (reg_max <= 1)
    {
        pred_cx = (float(col) + box_logits[0]) * inv_g;
        pred_cy = (float(row) + box_logits[1]) * inv_g;
        pred_w  = box_logits[2];
        pred_h  = box_logits[3];
        return;
    }

    const float cell_cx = (float(col) + 0.5f) * inv_g;
    const float cell_cy = (float(row) + 0.5f) * inv_g;
    const float d_l = dfl_decode(box_logits + 0          , reg_max);
    const float d_t = dfl_decode(box_logits + reg_max     , reg_max);
    const float d_r = dfl_decode(box_logits + 2 * reg_max , reg_max);
    const float d_b = dfl_decode(box_logits + 3 * reg_max , reg_max);
    pred_cx = cell_cx + (d_r - d_l) * inv_g * 0.5f;
    pred_cy = cell_cy + (d_b - d_t) * inv_g * 0.5f;
    pred_w  = (d_l + d_r) * inv_g;
    pred_h  = (d_t + d_b) * inv_g;
}

static TalResult tal_assign_head(const TensorView& output,
                                  const float* gt_list,
                                  Index batch_size, Index G, Index C,
                                  Index max_gt_boxes,
                                  Index reg_max = 1)
{
    const Index cells  = G * G;
    const float inv_g  = 1.0f / float(G);
    const Index box_ch = 4 * reg_max;
    const Index ch_out = box_ch + C;

    TalResult res;
    res.assign.assign(size_t(batch_size * cells), 0);
    res.iou_map.assign(size_t(batch_size * cells), 0.0f);

    for (Index n = 0; n < batch_size; ++n)
    {
        const float* gt    = gt_list + n * max_gt_boxes * 5;
        const float* out_n = output.as<float>() + n * cells * ch_out;

        vector<float> assign_iou(size_t(cells), -1.0f);

        for (Index gi = 0; gi < max_gt_boxes; ++gi)
        {
            if (gt[gi*5 + 4] < 0.5f) continue;

            const float gt_cx  = gt[gi*5 + 0];
            const float gt_cy  = gt[gi*5 + 1];
            const float gt_w   = gt[gi*5 + 2];
            const float gt_h   = gt[gi*5 + 3];
            const Index gt_cls = Index(gt[gi*5 + 4]) - 1;

            const float gt_x0 = gt_cx - gt_w*0.5f, gt_x1 = gt_cx + gt_w*0.5f;
            const float gt_y0 = gt_cy - gt_h*0.5f, gt_y1 = gt_cy + gt_h*0.5f;

            struct CellScore { float score; Index cell; float iou; };
            vector<CellScore> cands;
            cands.reserve(size_t(cells));

            for (Index row = 0; row < G; ++row)
                for (Index col = 0; col < G; ++col)
                {
                    const float cx_c = (float(col) + 0.5f) * inv_g;
                    const float cy_c = (float(row) + 0.5f) * inv_g;
                    if (cx_c <= gt_x0 || cx_c >= gt_x1 || cy_c <= gt_y0 || cy_c >= gt_y1) continue;

                    const Index base_o = (row * G + col) * ch_out;
                    float pred_cx, pred_cy, pred_w, pred_h;
                    dfl_decode_box(out_n + base_o, reg_max, col, row, G,
                                   pred_cx, pred_cy, pred_w, pred_h);
                    const float cls_p = (gt_cls < C) ? out_n[base_o + box_ch + gt_cls] : 0.0f;
                    const float iou   = iou_cxcywh(pred_cx, pred_cy, pred_w, pred_h,
                                                    gt_cx,   gt_cy,   gt_w,   gt_h);
                    const float score = powf(cls_p + EPSILON, TAL_ALPHA) *
                                        powf(iou   + EPSILON, TAL_BETA);
                    cands.push_back({score, row*G + col, iou});
                }

            ranges::sort(cands, greater<>{}, &CellScore::score);

            const Index k = min<Index>(TAL_TOP_K, ssize(cands));
            for (Index i = 0; i < k; ++i)
            {
                const Index cell = cands[size_t(i)].cell;
                const float iou  = cands[size_t(i)].iou;
                if (iou > assign_iou[size_t(cell)])
                {
                    assign_iou[size_t(cell)] = iou;
                    res.assign [size_t(n * cells + cell)] = gi + 1;
                    res.iou_map[size_t(n * cells + cell)] = iou;
                }
            }
        }
    }
    return res;
}

static float yolo_v8_error_kernel_tal(const TensorView& output,
                                       const float* gt_list,
                                       Index batch_size, Index G, Index C,
                                       Index max_gt_boxes,
                                       const TalResult& tal,
                                       YoloLambdas lam,
                                       Index reg_max = 1)
{
    const float* out   = output.as<float>();
    const float inv_g  = 1.0f / float(G);
    const Index cells  = G * G;
    const Index box_ch = 4 * reg_max;
    const Index ch_out = box_ch + C;

    float coord_loss = 0.0f;
    float dfl_loss   = 0.0f;
    float cls_loss   = 0.0f;

    for (Index n = 0; n < batch_size; ++n)
    {
        const float* gt = gt_list + n * max_gt_boxes * 5;
        for (Index row = 0; row < G; ++row)
            for (Index col = 0; col < G; ++col)
            {
                const Index cell   = row * G + col;
                const Index base_o = (n * cells + cell) * ch_out;
                const Index gt_id1 = tal.assign [size_t(n * cells + cell)];
                const float q      = tal.iou_map[size_t(n * cells + cell)];

                if (gt_id1 > 0)
                {
                    const float* gr    = gt + (gt_id1 - 1) * 5;
                    const Index gt_cls = Index(gr[4]) - 1;

                    float pred_cx, pred_cy, pred_w, pred_h;
                    dfl_decode_box(out + base_o, reg_max, col, row, G,
                                   pred_cx, pred_cy, pred_w, pred_h);
                    const float ob[4] = {pred_cx, pred_cy, pred_w, pred_h};
                    const float tb[4] = {gr[0], gr[1], gr[2], gr[3]};
                    coord_loss += 1.0f - yolo_loss_giou_forward(ob, tb).giou;

                    if (reg_max > 1)
                    {
                        const float gt_cx = gr[0], gt_cy = gr[1];
                        const float gt_w  = gr[2], gt_h  = gr[3];
                        const float cell_cx = (float(col) + 0.5f) * inv_g;
                        const float cell_cy = (float(row) + 0.5f) * inv_g;
                        const float rm1     = float(reg_max - 1);
                        const float d_tgts[4] = {
                            clamp((cell_cx - (gt_cx - gt_w*0.5f)) * float(G), 0.0f, rm1),
                            clamp((cell_cy - (gt_cy - gt_h*0.5f)) * float(G), 0.0f, rm1),
                            clamp(((gt_cx + gt_w*0.5f) - cell_cx) * float(G), 0.0f, rm1),
                            clamp(((gt_cy + gt_h*0.5f) - cell_cy) * float(G), 0.0f, rm1)
                        };
                        for (Index g = 0; g < 4; ++g)
                        {
                            const float* logits = out + base_o + g * reg_max;
                            const float  dt     = d_tgts[g];
                            const Index  df     = min(Index(dt), reg_max - 2);
                            const Index  dc     = df + 1;
                            const float  wl     = float(dc) - dt;
                            const float  wu     = dt - float(df);
                            float max_l = *max_element(logits, logits + reg_max);
                            float sum   = 0.0f;
                            for (Index i = 0; i < reg_max; ++i) sum += expf(logits[i] - max_l);
                            const float pf = expf(logits[df] - max_l) / sum;
                            const float pc = expf(logits[dc] - max_l) / sum;
                            dfl_loss -= wl * logf(pf + EPSILON) + wu * logf(pc + EPSILON);
                        }
                    }

                    for (Index c = 0; c < C; ++c)
                    {
                        const float p = out[base_o + box_ch + c];
                        if (c == gt_cls)
                            cls_loss -= q * logf(p + EPSILON);
                        else
                            cls_loss -= powf(p, lam.focal_gamma) * logf(1.0f - p + EPSILON);
                    }
                }
                else
                {
                    for (Index c = 0; c < C; ++c)
                    {
                        const float p = out[base_o + box_ch + c];
                        cls_loss -= powf(p, lam.focal_gamma) * logf(1.0f - p + EPSILON);
                    }
                }
            }
    }
    return lam.giou * coord_loss + lam.dfl * dfl_loss + lam.cls * cls_loss;
}

static void yolo_v8_gradient_kernel_tal(const TensorView& output,
                                         const float* gt_list,
                                         const TensorView& output_delta,
                                         Index batch_size, Index G, Index C,
                                         Index max_gt_boxes,
                                         const TalResult& tal,
                                         float inv_batch,
                                         YoloLambdas lam,
                                         Index reg_max = 1)
{
    constexpr float grad_clip = 10.0f;

    const float* out   = output.as<float>();
    float*       delta = output_delta.as<float>();
    const float inv_g  = 1.0f / float(G);
    const Index cells  = G * G;
    const Index box_ch = 4 * reg_max;
    const Index ch_out = box_ch + C;

    fill_n(delta, output_delta.size(), 0.0f);

    for (Index n = 0; n < batch_size; ++n)
    {
        const float* gt = gt_list + n * max_gt_boxes * 5;
        for (Index row = 0; row < G; ++row)
            for (Index col = 0; col < G; ++col)
            {
                const Index cell   = row * G + col;
                const Index base_o = (n * cells + cell) * ch_out;
                const Index gt_id1 = tal.assign [size_t(n * cells + cell)];
                const float q      = tal.iou_map[size_t(n * cells + cell)];

                const float cls_s = lam.cls  * inv_batch;
                const float box_s = lam.giou * inv_batch;
                const float gam   = lam.focal_gamma;

                if (gt_id1 > 0)
                {
                    const float* gr    = gt + (gt_id1 - 1) * 5;
                    const Index gt_cls = Index(gr[4]) - 1;

                    float pred_cx, pred_cy, pred_w, pred_h;
                    dfl_decode_box(out + base_o, reg_max, col, row, G,
                                   pred_cx, pred_cy, pred_w, pred_h);
                    const float ob[4] = {pred_cx, pred_cy, pred_w, pred_h};
                    const float tb[4] = {gr[0], gr[1], gr[2], gr[3]};
                    const GIoUResult gr_res = yolo_loss_giou_grad(ob, tb);

                    if (reg_max > 1)
                    {

                        const float gt_cx = gr[0], gt_cy = gr[1];
                        const float gt_w  = gr[2], gt_h  = gr[3];
                        const float cell_cx = (float(col) + 0.5f) * inv_g;
                        const float cell_cy = (float(row) + 0.5f) * inv_g;
                        const float rm1     = float(reg_max - 1);
                        const float d_tgts[4] = {
                            clamp((cell_cx - (gt_cx - gt_w*0.5f)) * float(G), 0.0f, rm1),
                            clamp((cell_cy - (gt_cy - gt_h*0.5f)) * float(G), 0.0f, rm1),
                            clamp(((gt_cx + gt_w*0.5f) - cell_cx) * float(G), 0.0f, rm1),
                            clamp(((gt_cy + gt_h*0.5f) - cell_cy) * float(G), 0.0f, rm1)
                        };

                        const float cx_g = clamp(gr_res.cx_gradient, -grad_clip, grad_clip);
                        const float cy_g = clamp(gr_res.cy_gradient, -grad_clip, grad_clip);
                        const float w_g  = clamp(gr_res.w_gradient,  -grad_clip, grad_clip);
                        const float h_g  = clamp(gr_res.h_gradient,  -grad_clip, grad_clip);
                        const float d_ciou_dd[4] = {
                            cx_g * (-inv_g * 0.5f) + w_g * inv_g,
                            cy_g * (-inv_g * 0.5f) + h_g * inv_g,
                            cx_g * (inv_g * 0.5f)  + w_g * inv_g,
                            cy_g * (inv_g * 0.5f)  + h_g * inv_g,
                        };

                        float d_g[4] = {};
                        vector<float> all_probs(size_t(4 * reg_max));
                        for (Index g = 0; g < 4; ++g)
                        {
                            const float* logits = out + base_o + g * reg_max;
                            float max_l = *max_element(logits, logits + reg_max);
                            float sum   = 0.0f;
                            for (Index i = 0; i < reg_max; ++i) sum += expf(logits[i] - max_l);
                            for (Index i = 0; i < reg_max; ++i)
                            {
                                const float p = expf(logits[i] - max_l) / sum;
                                all_probs[size_t(g * reg_max + i)] = p;
                                d_g[g] += float(i) * p;
                            }
                        }

                        const float dfl_s = lam.dfl * inv_batch;
                        for (Index g = 0; g < 4; ++g)
                        {
                            float*      dlogit = delta + base_o + g * reg_max;
                            const float dt     = d_tgts[g];
                            const Index df     = min(Index(dt), reg_max - 2);
                            const Index dc     = df + 1;
                            const float wl     = float(dc) - dt;
                            const float wu     = dt - float(df);
                            for (Index i = 0; i < reg_max; ++i)
                            {
                                const float p = all_probs[size_t(g * reg_max + i)];
                                float w_tgt = 0.0f;
                                if (i == df) w_tgt += wl;
                                if (i == dc) w_tgt += wu;
                                dlogit[i] = dfl_s * (p - w_tgt)
                                          + box_s * d_ciou_dd[g] * p * (float(i) - d_g[g]);
                            }
                        }
                    }
                    else
                    {
                        delta[base_o+0] = box_s * inv_g * clamp(gr_res.cx_gradient, -grad_clip, grad_clip);
                        delta[base_o+1] = box_s * inv_g * clamp(gr_res.cy_gradient, -grad_clip, grad_clip);
                        delta[base_o+2] = box_s * clamp(gr_res.w_gradient, -grad_clip, grad_clip);
                        delta[base_o+3] = box_s * clamp(gr_res.h_gradient, -grad_clip, grad_clip);
                    }

                    for (Index c = 0; c < C; ++c)
                    {
                        const float p = out[base_o + box_ch + c];
                        float d;
                        if (c == gt_cls)
                        {
                            d = -q / (p + EPSILON);
                        }
                        else
                        {
                            d = (gam > 0.0f ? -gam * powf(p, gam-1.0f) * logf(1.0f-p+EPSILON) : 0.0f)
                                + powf(p, gam) / (1.0f-p+EPSILON);
                        }
                        delta[base_o + box_ch + c] = cls_s * d;
                    }
                }
                else
                {
                    for (Index c = 0; c < C; ++c)
                    {
                        const float p = out[base_o + box_ch + c];
                        const float d = (gam > 0.0f ? -gam * powf(p, gam-1.0f) * logf(1.0f-p+EPSILON) : 0.0f)
                                        + powf(p, gam) / (1.0f-p+EPSILON);
                        delta[base_o + box_ch + c] = cls_s * d;
                    }
                }
            }
    }
}

template<typename HeadFn>
static void for_each_v8_head(const ForwardPropagation& forward_propagation,
                             const TensorView& target_flat,
                             BackPropagation* back_propagation,
                             const NeuralNetwork* nn,
                             const vector<Index>& detection_indices,
                             HeadFn&& fn)
{
    const bool on_device = target_flat.is_cuda();

    const float* tgt = nullptr;
    vector<float> tgt_cpu;
#ifdef OPENNN_HAS_CUDA
    if (on_device)
    {
        tgt_cpu.resize(size_t(target_flat.size()));
        device::copy_async(tgt_cpu.data(), target_flat.as<float>(),
                           Index(target_flat.size()) * Index(sizeof(float)),
                           device::CopyKind::DeviceToHost, device::get_compute_stream());
        device::synchronize(device::get_compute_stream());
        tgt = tgt_cpu.data();
    }
#endif
    if (!on_device) tgt = target_flat.as<float>();

    for (Index detection_idx : detection_indices)
    {
        const TensorView head_view = forward_propagation.slots[size_t(detection_idx)].back();
        const DetectionHeadMetadata metadata =
            get_detection_head_metadata(*nn, detection_idx);
        throw_if(!metadata.is_anchor_free(),
                 "YOLO v8 loss requires anchor-free detection heads.");
        const Index G       = nn->get_layer(detection_idx)->get_output_shape()[0];
        const Index reg_max = metadata.regression_bins;

        TensorView head_output = head_view;
        vector<float> out_cpu;
        vector<float> delta_cpu;
        TensorView staged_delta;
        TensorView* head_delta = nullptr;

#ifdef OPENNN_HAS_CUDA
        if (on_device)
        {
            out_cpu.resize(size_t(head_view.size()));
            device::copy_async(out_cpu.data(), head_view.as<float>(),
                               Index(head_view.size()) * Index(sizeof(float)),
                               device::CopyKind::DeviceToHost, device::get_compute_stream());
            device::synchronize(device::get_compute_stream());
            head_output = TensorView(out_cpu.data(), head_view.get_shape(), Type::FP32);
        }
#endif
        if (back_propagation)
        {
            TensorView& device_delta = back_propagation->output_deltas[size_t(detection_idx)];
            if (on_device)
            {
                delta_cpu.assign(size_t(head_view.size()), 0.0f);
                staged_delta = TensorView(delta_cpu.data(), device_delta.get_shape(), Type::FP32);
                head_delta = &staged_delta;
            }
            else
                head_delta = &device_delta;
        }

        fn(head_output, tgt, head_delta, G, reg_max);

#ifdef OPENNN_HAS_CUDA
        if (back_propagation && on_device)
            device::copy_async(back_propagation->output_deltas[size_t(detection_idx)].as<float>(),
                               delta_cpu.data(),
                               Index(head_view.size()) * Index(sizeof(float)),
                               device::CopyKind::HostToDevice, device::get_compute_stream());
            device::synchronize(device::get_compute_stream());
#endif
    }
}

Index get_max_gt_boxes(const TensorView& target_flat, Index batch_size)
{
    constexpr Index values_per_box = 5;
    throw_if(batch_size <= 0
             || target_flat.size() % (batch_size * values_per_box) != 0,
             "YOLO v8 targets must contain five values per ground-truth box.");
    const Index max_gt_boxes = target_flat.size() / (batch_size * values_per_box);
    throw_if(max_gt_boxes <= 0,
             "YOLO v8 targets must reserve at least one ground-truth box.");
    return max_gt_boxes;
}

Loss::EvaluationResult yolo_v8_error_multi(const ForwardPropagation& forward_propagation,
                                           const TensorView& target_flat,
                                           const NeuralNetwork* nn,
                                           const vector<Index>& detection_indices,
                                           Index classes_number,
                                           YoloLambdas lam)
{
    const Index batch_size = target_flat.get_shape()[0];
    const Index max_gt_boxes = get_max_gt_boxes(target_flat, batch_size);

    float total_error = 0.0f;
    for_each_v8_head(forward_propagation, target_flat, nullptr, nn, detection_indices,
        [&](const TensorView& head_output, const float* tgt, TensorView*, Index G, Index reg_max)
        {
            const TalResult tal = tal_assign_head(head_output, tgt, batch_size, G,
                                                  classes_number, max_gt_boxes, reg_max);
            total_error += yolo_v8_error_kernel_tal(head_output, tgt, batch_size, G,
                                                    classes_number, max_gt_boxes,
                                                    tal, lam, reg_max);
        });
    return {.error = total_error / float(batch_size)};
}

void yolo_v8_gradient_multi(const ForwardPropagation& forward_propagation,
                            const TensorView& target_flat,
                            BackPropagation& back_propagation,
                            const NeuralNetwork* nn,
                            const vector<Index>& detection_indices,
                            Index classes_number,
                            YoloLambdas lam)
{
    const Index batch_size = target_flat.get_shape()[0];
    const float inv_batch = 1.0f / float(batch_size);
    const Index max_gt_boxes = get_max_gt_boxes(target_flat, batch_size);

    for_each_v8_head(forward_propagation, target_flat, &back_propagation, nn, detection_indices,
        [&](const TensorView& head_output, const float* tgt, TensorView* head_delta, Index G, Index reg_max)
        {
            const TalResult tal = tal_assign_head(head_output, tgt, batch_size, G,
                                                  classes_number, max_gt_boxes, reg_max);
            yolo_v8_gradient_kernel_tal(head_output, tgt, *head_delta, batch_size, G,
                                        classes_number, max_gt_boxes,
                                        tal, inv_batch, lam, reg_max);
        });
}

#ifdef OPENNN_HAS_CUDA

template <typename HeadFn>
void for_each_yolo_head_gpu(const ForwardPropagation& forward_propagation,
                            const NeuralNetwork* nn,
                            const vector<Index>& detection_indices,
                            const TensorView& target_flat,
                            Buffer& target_device,
                            HeadFn&& fn)
{
    const Index batch_size = target_flat.get_shape()[0];

    if (target_flat.is_cuda())
    {
        const float* tgt = target_flat.as<float>();
        for_each_yolo_head_layout(nn, detection_indices,
            [&](Index detection_idx, const Shape& head_shape, Index channels,
                Index per_sample_floats, Index head_offset, Index head_floats)
            {
                const float* head_target = tgt;
                if (per_sample_floats != head_floats)
                {
                    target_device.grow_to(batch_size * head_floats * Index(sizeof(float)));
                    yolo_assemble_head_target_cuda(tgt, target_device.as<float>(),
                                                   batch_size, per_sample_floats,
                                                   head_offset, head_floats);
                    head_target = target_device.as<float>();
                }

                fn(detection_idx,
                   forward_propagation.slots[size_t(detection_idx)].back(),
                   TensorView(const_cast<float*>(head_target),
                              Shape({batch_size, head_shape[0], head_shape[1], channels}),
                              Type::FP32, Device::CUDA));
            });
        return;
    }

    for_each_yolo_head(forward_propagation, nn, detection_indices,
                       target_flat.as<float>(), batch_size,
        [&](Index detection_idx, const TensorView& head_output, const TensorView& head_target)
        {
            const Index target_bytes = head_target.size() * Index(sizeof(float));
            target_device.grow_to(target_bytes);
            device::copy_async(target_device.as<float>(), head_target.as<float>(),
                               Index(target_bytes),
                               device::CopyKind::HostToDevice, device::get_compute_stream());

            fn(detection_idx, head_output,
               TensorView(target_device.as<float>(), head_target.get_shape(),
                          Type::FP32, Device::CUDA));
        });
}

void yolo_error_gpu_accumulate(const ForwardPropagation& forward_propagation,
                               const TensorView& target_flat,
                               const NeuralNetwork* nn,
                               const vector<Index>& detection_indices,
                               const DetectionHeadMetadata& head,
                               Buffer& target_device,
                               float* error_accum,
                               YoloLambdas lam)
{
    const Index boxes_per_head = head.boxes_per_cell;
    const Index classes_number = head.classes_number;
    const bool sigmoid_classes = head.uses_sigmoid_classes();
    const Index values_per_box = 5 + classes_number;
    const Index batch_size = target_flat.get_shape()[0];

    device::set_zero_async(error_accum, Index(sizeof(float)), device::get_compute_stream());

    for_each_yolo_head_gpu(forward_propagation, nn, detection_indices, target_flat, target_device,
        [&](Index, const TensorView& head_output, const TensorView& head_target)
        {
            const Index grid_size = head_target.get_shape()[1];
            yolo_error_cuda(head_output.as<float>(), head_target.as<float>(), error_accum,
                            to_int(batch_size), to_int(grid_size), to_int(boxes_per_head),
                            to_int(values_per_box), to_int(classes_number),
                            sigmoid_classes ? 1 : 0, lam.giou, lam.noobj, lam.cls, lam.focal_gamma, lam.obj_focal_gamma);
        });
}

Loss::EvaluationResult yolo_error_gpu_multi(const ForwardPropagation& forward_propagation,
                                            const TensorView& target_flat,
                                            const NeuralNetwork* nn,
                                            const vector<Index>& detection_indices,
                                            const DetectionHeadMetadata& head,
                                            Buffer& target_device,
                                            Buffer& error_device,
                                            YoloLambdas lam)
{
    const Index batch_size = target_flat.get_shape()[0];

    error_device.grow_to(Index(sizeof(float)));
    yolo_error_gpu_accumulate(forward_propagation, target_flat, nn,
                              detection_indices, head, target_device,
                              error_device.as<float>(), lam);

    device::synchronize(device::get_compute_stream());
    float total_error = 0.0f;
    device::copy_async(&total_error, error_device.as<float>(), Index(sizeof(float)),
                       device::CopyKind::DeviceToHost, device::get_compute_stream());
    device::synchronize(device::get_compute_stream());

    return {.error = total_error / float(batch_size)};
}

void yolo_gradient_gpu_multi(const ForwardPropagation& forward_propagation,
                             const TensorView& target_flat,
                             BackPropagation& back_propagation,
                             const NeuralNetwork* nn,
                             const vector<Index>& detection_indices,
                             const DetectionHeadMetadata& head,
                             Buffer& target_device,
                             YoloLambdas lam)
{
    const Index boxes_per_head = head.boxes_per_cell;
    const Index classes_number = head.classes_number;
    const bool sigmoid_classes = head.uses_sigmoid_classes();
    const Index values_per_box = 5 + classes_number;
    const Index batch_size = target_flat.get_shape()[0];
    const float inv_batch = 1.0f / float(batch_size);

    for_each_yolo_head_gpu(forward_propagation, nn, detection_indices, target_flat, target_device,
        [&](Index detection_idx, const TensorView& head_output, const TensorView& head_target)
        {
            TensorView& head_delta = back_propagation.output_deltas[size_t(detection_idx)];
            const Index grid_size = head_target.get_shape()[1];
            yolo_gradient_cuda(head_output.as<float>(), head_target.as<float>(),
                               head_delta.as<float>(),
                               to_int(batch_size), to_int(grid_size), to_int(boxes_per_head),
                               to_int(values_per_box), to_int(classes_number),
                               sigmoid_classes ? 1 : 0, inv_batch,
                               lam.giou, lam.noobj, lam.cls, lam.focal_gamma, lam.obj_focal_gamma);
        });
}

#endif

}
#endif

Loss::Loss(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    set(new_neural_network, new_dataset);
}

void Loss::set(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    neural_network = new_neural_network;
    dataset = new_dataset;

    set_error(Error::MeanSquaredError);

}

vector<Index> Loss::get_output_delta_layer_indices() const
{
    if (!neural_network || neural_network->get_layers_number() == 0)
        return {};

    const auto& layers = neural_network->get_layers();

    if (error == Error::Yolo)
    {
        vector<Index> anchor_free_heads;
        vector<Index> anchor_based_heads;
        for (size_t i = 0; i < layers.size(); ++i)
        {
            const auto* const head = layers[i]
                ? dynamic_cast<const DetectionHeadEndpoint*>(layers[i].get())
                : nullptr;
            if (!head) continue;

            if (head->get_detection_head_metadata().is_anchor_free())
                anchor_free_heads.push_back(Index(i));
            else
                anchor_based_heads.push_back(Index(i));
        }

        if (!anchor_free_heads.empty())
            return anchor_free_heads;
        if (!anchor_based_heads.empty())
            return anchor_based_heads;
    }

    return {neural_network->get_last_trainable_layer_index()};
}

void Loss::set_normalization_coefficient()
{
    normalization_coefficient = 1.0f;
    positives_weight = 1.0f;
    negatives_weight = 1.0f;
    weighted_samples_number = 0;

    if (!dataset || dataset->get_samples_number() == 0)
        return;

    if (error == Error::NormalizedSquaredError)
    {
        const vector<Index> training_indices = dataset->get_sample_indices(SampleRole::Training);
        const Shape target_shape = dataset->get_shape(VariableRole::Target);

        if (training_indices.empty() || target_shape.empty()) return;

        MatrixR targets(training_indices.size(), target_shape.size());
        dataset->fill_targets(training_indices,
                              dataset->get_feature_indices(VariableRole::Target),
                              targets.data(),
                              FillMode::Inference);

        const VectorR targets_mean = mean(targets);
        const float coefficient = (targets.rowwise() - targets_mean.transpose()).squaredNorm();

        normalization_coefficient = (coefficient < EPSILON) ? 1.0f : coefficient;
        return;
    }

    if (error == Error::WeightedSquaredError)
    {
        const Index targets_number = dataset->get_features_number(VariableRole::Target);
        if (targets_number != 1) return;

        const auto [negatives, positives] = dataset->count_binary_targets("Training");

        if (positives == 0 || negatives == 0) return;

        weighted_samples_number = positives + negatives;

        const float total = float(positives + negatives);
        positives_weight = total / (2.0f * float(positives));
        negatives_weight = total / (2.0f * float(negatives));

        const float p = float(positives) / total;
        const float mean_model_error =
            0.5f * (float(positives) * positives_weight * (1.0f - p) * (1.0f - p)
                  + float(negatives) * negatives_weight * p * p);

        normalization_coefficient = (mean_model_error < EPSILON) ? 1.0f : mean_model_error;
    }

    if (error == Error::CrossEntropy && neural_network
        && dataset->get_features_number(VariableRole::Target) > 1)
    {
        const auto& layers = neural_network->get_layers();
        const Index last_trainable = neural_network->get_last_trainable_layer_index();
        throw_if(layers[last_trainable]->get_output_activation() != ActivationFunction::Softmax,
                 "Cross-entropy error with multiple target features requires a softmax output layer.");
    }
}

void Loss::back_propagate(const Batch& batch,
                          ForwardPropagation& forward_propagation,
                          BackPropagation& back_propagation) const
{
    if (batch.is_empty()) return;

    neural_network->link_gradients(back_propagation.gradient);

    {
        PROFILE_SCOPE("loss:calculate_error");
        const EvaluationResult evaluation_result = calculate_error(batch, forward_propagation);
        back_propagation.metrics.error                = evaluation_result.error;
        back_propagation.metrics.accuracy             = evaluation_result.accuracy;
        back_propagation.metrics.active_tokens_count  = evaluation_result.active_tokens_count;
    }

    calculate_layers_error_gradient(batch, forward_propagation, back_propagation);

    back_propagation.metrics.regularization = 0.0f;
    back_propagation.metrics.loss_value = back_propagation.metrics.error;

    add_regularization_gradient(back_propagation);
}

float Loss::get_batch_scale(const Batch& batch) const
{
    const Index total = weighted_samples_number > 0 ? weighted_samples_number
                      : dataset                     ? dataset->get_samples_number("Training")
                                                    : batch.get_batch_size();
    const Index samples = batch.get_batch_size();

    return samples > 0 ? float(total) / float(samples) : 1.0f;
}

#ifndef OPENNN_NO_VISION

Loss::EvaluationResult Loss::calculate_yolo(const ForwardPropagation& forward_propagation,
                                            const TensorView& target,
                                            BackPropagation* back_propagation) const
{
    const bool is_gradient = back_propagation != nullptr;
    const YoloLambdas lam{yolo_lambda_giou, yolo_lambda_dfl, yolo_lambda_noobj, yolo_lambda_class, yolo_focal_gamma, yolo_obj_focal_gamma};
    const vector<Index> detection_indices = get_output_delta_layer_indices();
    const DetectionHeadMetadata head =
        get_loss_head_metadata(*neural_network, detection_indices);

    if (head.is_anchor_free())
    {
        if (!is_gradient)
            return yolo_v8_error_multi(forward_propagation, target, neural_network,
                                       detection_indices, head.classes_number, lam);
        yolo_v8_gradient_multi(forward_propagation, target, *back_propagation,
                               neural_network, detection_indices, head.classes_number, lam);
        return {};
    }

#ifdef OPENNN_HAS_CUDA
    const bool on_gpu = runs_on_gpu();
    if (on_gpu)
    {

        if (!is_gradient)
            return yolo_error_gpu_multi(forward_propagation, target, neural_network,
                                        detection_indices, head,
                                        forward_propagation.loss_target_workspace,
                                        forward_propagation.loss_workspace, lam);
        yolo_gradient_gpu_multi(forward_propagation, target, *back_propagation,
                                neural_network, detection_indices, head,
                                back_propagation->execution_workspace, lam);
        return {};
    }
#endif
    if (!is_gradient)
        return yolo_error_cpu_multi(forward_propagation, target, neural_network,
                                    detection_indices, head, lam);

    yolo_gradient_cpu_multi(forward_propagation, target, *back_propagation,
                            neural_network, detection_indices, head, lam);
    return {};
}

#endif

Index Loss::error_workspace_floats(const TensorView& input) const
{
    return (error == Error::CrossEntropy3d)
        ? 3 * (input.size() / input.get_shape().back())
        : input.size();
}

float* Loss::ensure_error_workspace(Buffer& storage,
                                    const TensorView& input,
                                    Index batch_samples,
                                    Index reduction_floats) const
{
    const Index workspace_floats = error_workspace_floats(input);
    storage.grow_to((workspace_floats + reduction_floats)
                    * Index(sizeof(float)));
    if (memory_debug::enabled())
        memory_debug::record("loss", "ForwardPropagation::loss_workspace",
                             (workspace_floats + reduction_floats)
                                 * Index(sizeof(float)),
                             format("batch={}", batch_samples));
    return storage.as<float>();
}

Loss::EvaluationResult Loss::calculate_error(const Batch& batch,
                                              const ForwardPropagation& forward_propagation) const
{
    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
    const TensorView target = batch.get_targets();

    EvaluationResult result;

    float* workspace_device = nullptr;
    const bool device_on_gpu = runs_on_gpu();
    if (device_on_gpu && error != Error::Yolo)
        workspace_device = ensure_error_workspace(
            forward_propagation.loss_workspace,
            input,
            batch.get_batch_size(),
            error == Error::CrossEntropy3d ? 3 : 0);

    using enum Error;
    switch (error)
    {
    case MeanSquaredError:
        mean_squared_error(input, target, result.error, workspace_device);
        break;
    case MeanAbsoluteError:
        mean_absolute_error(input, target, result.error, workspace_device);
        break;
    case NormalizedSquaredError:
        normalized_squared_error(input, target, normalization_coefficient, result.error, workspace_device);
        result.error *= get_batch_scale(batch);
        break;
    case WeightedSquaredError:
        weighted_squared_error(input, target, positives_weight, negatives_weight, result.error, workspace_device);
        result.error *= get_weighted_coefficient(batch);
        break;
    case CrossEntropy:
        cross_entropy(input, target, result.error, workspace_device);
        break;
    case CrossEntropy3d:
    {
        Index correct_tokens = 0;
        float* const reduction_device = workspace_device
            ? workspace_device + error_workspace_floats(input)
            : nullptr;
        cross_entropy_3d(input, target, result.error,
                         result.active_tokens_count, correct_tokens,
                         workspace_device, reduction_device);
        result.accuracy = result.active_tokens_count > 0
            ? float(correct_tokens) / float(result.active_tokens_count)
            : 0.0f;
        break;
    }
    case MinkowskiError:
        minkowski_error(input, target, minkowski_parameter, result.error, workspace_device);
        break;
    case Yolo:
#ifndef OPENNN_NO_VISION
        result = calculate_yolo(forward_propagation, target, nullptr);
#else
        throw runtime_error("YOLO loss not available: opennn was built with OpenNN_BUILD_VISION=OFF.");
#endif
        break;
    }

    return result;
}

bool Loss::supports_device_epoch_metrics() const
{
    if (error == Error::Yolo) return false;
    return runs_on_gpu() && error != Error::MinkowskiError;
}

#ifdef OPENNN_HAS_CUDA

bool Loss::calculate_error_device_metrics(const Batch& batch,
                                          const ForwardPropagation& forward_propagation,
                                          float* error_sum_device,
                                          float* accuracy_sum_device) const
{
    if (!supports_device_epoch_metrics() || !error_sum_device) return false;

    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
    const TensorView target = batch.get_targets();
    if (input.empty() || target.empty()) return false;

    const Index workspace_floats = error_workspace_floats(input);
    float* const workspace = ensure_error_workspace(
        forward_propagation.loss_workspace,
        input,
        batch.get_batch_size(),
        3);
    float* const results_device = workspace + workspace_floats;
    cublasHandle_t handle = device::get_cublas_handle();

    auto reduce_abs_and_accumulate = [&](Index n, float scale)
    {
        {
            device::CublasPointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_DEVICE);
            CHECK_CUBLAS(cublasSasum(handle, to_int(n), workspace, 1, results_device));
        }
        accumulate_scaled_metric_cuda(results_device, scale, error_sum_device);
    };

    auto reduce_dot_and_accumulate = [&](Index n, float scale)
    {
        {
            device::CublasPointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_DEVICE);
            CHECK_CUBLAS(cublasSdot(handle, to_int(n), workspace, 1, workspace, 1, results_device));
        }
        accumulate_scaled_metric_cuda(results_device, scale, error_sum_device);
    };

    using enum Error;
    switch (error)
    {
    case MeanAbsoluteError:
        input.dispatch([&]<typename TIn>()
        {
            scaled_diff_cuda_typed<TIn, float>(input.size(), input.as<TIn>(), target.as_float(),
                                               1.0f, workspace);
        });
        reduce_abs_and_accumulate(input.size(), 1.0f / static_cast<float>(input.size()));
        return true;

    case MeanSquaredError:
    case NormalizedSquaredError:
        input.dispatch([&]<typename TIn>()
        {
            scaled_diff_cuda_typed<TIn, float>(input.size(), input.as<TIn>(), target.as_float(),
                                               1.0f, workspace);
        });
        reduce_dot_and_accumulate(input.size(),
                                  error == MeanSquaredError
                                      ? 1.0f / static_cast<float>(2 * input.get_shape()[0])
                                      : get_weighted_coefficient(batch));
        return true;

    case WeightedSquaredError:
        input.dispatch([&]<typename T>()
        {
            weighted_squared_error_cuda<T>(input.size(), workspace, target.as<float>(), input.as<T>(),
                                           positives_weight, negatives_weight);
        });
        reduce_abs_and_accumulate(input.size(), 0.5f * get_weighted_coefficient(batch));
        return true;

    case CrossEntropy:
        input.dispatch([&]<typename T>()
        {
            if (input.get_shape().back() == 1)
                binary_cross_entropy_cuda<T>(input.size(), workspace, target.as<float>(), input.as<T>(), EPSILON);
            else
                categorical_cross_entropy_cuda<T>(input.size(), workspace, target.as<float>(), input.as<T>(), EPSILON);
        });
        reduce_abs_and_accumulate(input.size(),
                                  1.0f / static_cast<float>(input.get_shape()[0]));
        return true;

    case CrossEntropy3d:
    {
        const Index vocabulary_size = input.get_shape().back();
        const Index token_count = input.size() / vocabulary_size;

        device::set_zero_async(results_device, 3 * Index(sizeof(float)), device::get_compute_stream());
        input.dispatch([&]<typename T>()
        {
            cross_entropy_3d_metrics_cuda<T>(token_count, to_int(vocabulary_size),
                input.as<T>(), target.as<float>(), EPSILON, results_device);
        });

        accumulate_cross_entropy_3d_metrics_cuda(results_device, error_sum_device, accuracy_sum_device);
        return true;
    }

    case Yolo:
    case MinkowskiError:
        return false;
    }

    return false;
}

bool Loss::back_propagate_device_metrics(const Batch& batch,
                                         ForwardPropagation& forward_propagation,
                                         BackPropagation& back_propagation,
                                         float* error_sum_device,
                                         float* accuracy_sum_device) const
{
    if (!supports_device_epoch_metrics()) return false;

    bool output_delta_ready = false;
    if (error == Error::MeanSquaredError && error_sum_device)
    {
        const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
        const TensorView target = batch.get_targets();
        const TensorView input_delta = back_propagation.get_output_delta();

        if (input.empty() || target.empty() || input_delta.empty()) return false;

        visit_type_pair<Type::FP32, Type::BF16>(
            input.get_type(), input_delta.get_type(),
            [&]<typename TIn, typename TOut>()
            {
                mean_squared_error_metrics_gradient_cuda<TIn, TOut>(
                    input.size(), batch.get_batch_size(),
                    input.as<TIn>(), target.as_float(),
                    input_delta.as<TOut>(), error_sum_device);
            });
        output_delta_ready = true;
    }
    else if (!calculate_error_device_metrics(batch, forward_propagation,
                                              error_sum_device,
                                              accuracy_sum_device))
        return false;

    neural_network->link_gradients(back_propagation.gradient);

    if (error == Error::CrossEntropy3d)
    {
        const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
        const TensorView target = batch.get_targets();

        if (output_delta_overwrites_outputs())
            back_propagation.get_output_delta() = input;

        TensorView& input_delta = back_propagation.get_output_delta();

        const float* const results_device =
            forward_propagation.loss_workspace.as<float>()
            + error_workspace_floats(input);
        cross_entropy_3d_gradient_device_count(input, target, input_delta,
                                               results_device + 1);
    }
    else if (!output_delta_ready)
    {
        calculate_output_deltas(batch, forward_propagation, back_propagation);
    }

    back_propagation.metrics.reset();

    back_propagate_layers(forward_propagation, back_propagation);
    add_regularization_gradient(back_propagation);

    return true;
}

#else

bool Loss::calculate_error_device_metrics(const Batch&,
                                          const ForwardPropagation&,
                                          float*,
                                          float*) const
{
    return false;
}

bool Loss::back_propagate_device_metrics(const Batch&,
                                         ForwardPropagation&,
                                         BackPropagation&,
                                         float*,
                                         float*) const
{
    return false;
}

#endif

bool Loss::output_delta_overwrites_outputs() const
{
    if (error != Error::CrossEntropy3d || !neural_network || !neural_network->is_gpu())
        return false;

    const auto& layers = neural_network->get_layers();
    return layers[neural_network->get_last_trainable_layer_index()]->get_output_activation()
        == ActivationFunction::Softmax;
}

void Loss::calculate_output_deltas(const Batch& batch, const ForwardPropagation& forward_propagation, BackPropagation& back_propagation) const
{
    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
    const TensorView target = batch.get_targets();

    if (output_delta_overwrites_outputs())
        back_propagation.get_output_delta() = input;

    const TensorView input_delta = back_propagation.get_output_delta();

    using enum Error;
    switch (error)
    {
    case MeanSquaredError:
        mean_squared_error_gradient(input, target, input_delta);
        break;
    case MeanAbsoluteError:
        mean_absolute_error_gradient(input, target, input_delta);
        break;
    case NormalizedSquaredError:
        normalized_squared_error_gradient(input, target,
                                          normalization_coefficient / get_batch_scale(batch),
                                          input_delta);
        break;
    case WeightedSquaredError:
        weighted_squared_error_gradient(input, target, positives_weight, negatives_weight, get_weighted_coefficient(batch), input_delta);
        break;
    case CrossEntropy:
        cross_entropy_gradient(input, target, input_delta);
        break;
    case CrossEntropy3d:
        cross_entropy_3d_gradient(input, target, input_delta, back_propagation.metrics.active_tokens_count);
        break;
    case MinkowskiError:
        minkowski_error_gradient(input, target, minkowski_parameter, input_delta,
                                 neural_network && neural_network->is_gpu());
        break;
    case Yolo:
#ifndef OPENNN_NO_VISION
        calculate_yolo(forward_propagation, target, &back_propagation);
#else
        throw runtime_error("YOLO gradient not available: opennn was built with OpenNN_BUILD_VISION=OFF.");
#endif
        break;
    }
}

void Loss::back_propagate_layers(ForwardPropagation& forward_propagation,
                                 BackPropagation& back_propagation) const
{
    check_neural_network();

    const vector<unique_ptr<Layer>>& layers = neural_network->get_layers();
    const size_t layers_number = neural_network->get_layers_number();

    if (layers_number == 0) return;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {
        if (i != last_trainable_layer_index)
        {
            PROFILE_SCOPE("bwd:accumulate_output_deltas");
            back_propagation.accumulate_output_deltas(static_cast<size_t>(i));
        }
        forward_propagation.recompute_for_backward(i);
        PROFILE_SCOPE("bwd:" + layers[i]->get_name());
        layers[i]->back_propagate(forward_propagation, back_propagation, i);
    }
}

float Loss::calculate_regularization(const VectorR& parameters_vec) const
{
    const TensorView parameters(const_cast<float*>(parameters_vec.data()), { ssize(parameters_vec) });
    return calculate_regularization(parameters);
}

float Loss::calculate_regularization(const TensorView& parameters) const
{
    if (!has_regularization()) return 0.0f;

    float penalty = 0.0f;

    if (regularization_method == Regularization::L1)
        l1_regularization(parameters, regularization_weight, penalty);
    else if (regularization_method == Regularization::L2)
        l2_regularization(parameters, regularization_weight, penalty);

    return penalty;
}

void Loss::calculate_layers_error_gradient(const Batch& batch,
                                           ForwardPropagation& forward_propagation,
                                           BackPropagation& back_propagation) const
{
    {
        PROFILE_SCOPE("loss:calculate_output_deltas");
        calculate_output_deltas(batch, forward_propagation, back_propagation);
    }

    back_propagate_layers(forward_propagation, back_propagation);
}

static const EnumMap<Loss::Error> error_map{
    {Loss::Error::MeanSquaredError,       "MeanSquaredError"},
    {Loss::Error::MeanAbsoluteError,      "MeanAbsoluteError"},
    {Loss::Error::NormalizedSquaredError, "NormalizedSquaredError"},
    {Loss::Error::WeightedSquaredError,   "WeightedSquaredError"},
    {Loss::Error::CrossEntropy,           "CrossEntropy"},
    {Loss::Error::CrossEntropy3d,         "CrossEntropyError3d"},
    {Loss::Error::MinkowskiError,         "MinkowskiError"},
    {Loss::Error::Yolo,                   "Yolo"},
    {Loss::Error::Yolo,                   "YoloError"}
};

void Loss::set_error(const Error& new_error)
{
    error = new_error;
    name = error_map.to_string(new_error);
}

void Loss::set_error(const string& new_name)
{
    set_error(error_map.from_string(new_name));
}

void Loss::add_regularization_gradient(const TensorView& gradient) const
{
    if (!has_regularization()) return;

    check_neural_network();

    const Device gradient_device = gradient.get_device();

    if (gradient_device == Device::CUDA && neural_network->get_parameters_device() != Device::CUDA)
        neural_network->copy_parameters_device();
    else if (gradient_device == Device::CPU && neural_network->get_parameters_device() == Device::CUDA)
        neural_network->copy_parameters_host();

    const TensorView parameters(neural_network->get_parameters_data(),
                                { neural_network->get_parameters_buffer_size() },
                                Type::FP32,
                                gradient_device);

    if (regularization_method == Regularization::L1)
        l1_regularization_gradient(parameters, regularization_weight, gradient);
    else if (regularization_method == Regularization::L2)
        l2_regularization_gradient(parameters, regularization_weight, gradient);
}

void Loss::add_regularization_gradient(BackPropagation& back_propagation) const
{
    if (!has_regularization()) return;

    check_neural_network();

    add_regularization_gradient(TensorView(back_propagation.gradient.as<float>(),
                                           { neural_network->get_parameters_buffer_size() },
                                           Type::FP32,
                                           back_propagation.gradient.get_device()));
}

void Loss::regularization_from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "Regularization");

    set_regularization(read_json_string(root_element, "Type"));

    set_regularization_weight(read_json_float(root_element, "RegularizationWeight", regularization_weight));
}

void Loss::regularization_to_JSON(JsonWriter& file_stream) const
{
    file_stream.open_element("Regularization");
    write_json(file_stream, {
        {"Type", regularization_to_string(regularization_method)},
        {"RegularizationWeight", regularization_weight}
    });
    file_stream.close_element();
}

void Loss::to_JSON(JsonWriter& printer) const
{

    printer.open_element(get_name());
    write_json(printer, {
        {"Method", get_name()},
        {"Regularization", regularization_to_string(regularization_method)},
        {"RegularizationWeight", regularization_weight}
    });

    if (error == Error::NormalizedSquaredError)
        add_json_field(printer, "NormalizationCoefficient", normalization_coefficient);

    if (error == Error::WeightedSquaredError)
        write_json(printer, {
            {"PositivesWeight", positives_weight},
            {"NegativesWeight", negatives_weight}
        });

    if (error == Error::MinkowskiError)
        add_json_field(printer, "MinkowskiParameter", minkowski_parameter);

    printer.close_element();
}

void Loss::from_JSON(const JsonDocument& document)
{

    const Json* root = document.first_child(get_name());
    if (!root) root = document.first_child("Loss");
    throw_if(!root, "Loss::from_JSON error: missing Loss element.");

    set_error(read_json_string(root, "Method"));

    set_regularization(read_json_string(root, "Regularization"));
    regularization_weight = read_json_float(root, "RegularizationWeight");

    if (root->find("NormalizationCoefficient"))
        normalization_coefficient = read_json_float(root, "NormalizationCoefficient");

    if (root->find("PositivesWeight")) {
        positives_weight = read_json_float(root, "PositivesWeight");
        negatives_weight = read_json_float(root, "NegativesWeight");
    }

    if (root->find("MinkowskiParameter"))
        minkowski_parameter = read_json_float(root, "MinkowskiParameter");
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
