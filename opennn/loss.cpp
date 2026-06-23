//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_types.h"
#include "batch.h"
#include "dataset.h"
#include "tabular_dataset.h"
#include "loss.h"
#include "memory_debug.h"
#include "yolo_dataset.h"
#include "detection_layer.h"
#include "neural_network.h"
#include "error_functions.h"
#include "profiler.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "statistics.h"
#include <Eigen/LU>

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
    return r;
}

void check_yolo_loss(const Dataset* dataset, const NeuralNetwork* neural_network)
{
    throw_if(neural_network && neural_network->is_gpu(),
             "YOLO loss GPU implementation not available yet.");

    throw_if(!dynamic_cast<const YoloDataset*>(dataset),
             "YOLO loss requires YoloDataset.");
}

bool yolo_uses_sigmoid_classes(const NeuralNetwork* nn)
{
    if (!nn) return false;
    const vector<unique_ptr<Layer>>& layers = nn->get_layers();
    for (const unique_ptr<Layer>& layer : layers)
        if (layer && layer->get_type() == LayerType::Detection)
            return static_cast<const Detection*>(layer.get())->get_class_activation()
                   == Detection::ClassActivation::Sigmoid;
    return false;
}

float yolo_error_kernel(const TensorView& output,
                        const TensorView& target,
                        Index boxes_per_cell,
                        Index classes_number,
                        bool sigmoid_classes)
{
    constexpr float lambda_giou = 5.0f;
    constexpr float lambda_noobject = 0.5f;

    const Index values_per_box = 5 + classes_number;
    const Index batch_size = output.shape[0];
    const Index grid_size = output.shape[1];
    const Index channels = output.shape[3];

    const float* out = output.as<float>();
    const float* tgt = target.as<float>();

    float coordinate_loss = 0.0f;
    float object_loss = 0.0f;
    float noobject_loss = 0.0f;
    float class_loss = 0.0f;

    for (Index n = 0; n < batch_size; ++n)
        for (Index row = 0; row < grid_size; ++row)
            for (Index col = 0; col < grid_size; ++col)
            {
                const Index cell = ((n * grid_size + row) * grid_size + col) * channels;

                for (Index box = 0; box < boxes_per_cell; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    if (tgt[base + 4] == 1.0f)
                    {
                        const float output_box[4] = {out[base + 0], out[base + 1], out[base + 2], out[base + 3]};
                        const float target_box[4] = {tgt[base + 0], tgt[base + 1], tgt[base + 2], tgt[base + 3]};
                        const GIoUResult g = yolo_loss_giou_forward(output_box, target_box);

                        coordinate_loss += 1.0f - g.giou;
                        object_loss += pow(out[base + 4] - g.iou, 2.0f);

                        if (sigmoid_classes)
                        {
                            for (Index c = 0; c < classes_number; ++c)
                            {
                                const float p = out[base + 5 + c];
                                const float t = tgt[base + 5 + c];
                                class_loss -= t * log(p + EPSILON) + (1.0f - t) * log(1.0f - p + EPSILON);
                            }
                        }
                        else
                        {
                            for (Index c = 0; c < classes_number; ++c)
                                if (tgt[base + 5 + c] > 0.0f)
                                    class_loss -= log(out[base + 5 + c] + EPSILON);
                        }
                    }
                    else
                    {
                        noobject_loss += pow(out[base + 4], 2.0f);
                    }
                }
            }

    return lambda_giou * coordinate_loss + object_loss
         + lambda_noobject * noobject_loss + class_loss;
}

Loss::EvaluationResult yolo_error_cpu(const TensorView& output,
                                      const TensorView& target,
                                      const Dataset* dataset,
                                      const NeuralNetwork* neural_network,
                                      bool sigmoid_classes)
{
    check_yolo_loss(dataset, neural_network);
    const auto* yolo_dataset = static_cast<const YoloDataset*>(dataset);
    const Index boxes_per_cell = yolo_dataset->get_boxes_per_cell();
    const Index classes_number = yolo_dataset->get_classes_number();
    const Index batch_size = output.shape[0];

    Loss::EvaluationResult result;
    result.error = yolo_error_kernel(output, target, boxes_per_cell, classes_number, sigmoid_classes) / float(batch_size);
    return result;
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
                          float inv_batch)
{
    constexpr float lambda_giou = 5.0f;
    constexpr float lambda_noobject = 0.5f;
    constexpr float grad_clip = 10.0f;

    const Index values_per_box = 5 + classes_number;
    const Index batch_size = output.shape[0];
    const Index grid_size = output.shape[1];
    const Index channels = output.shape[3];

    const float* out = output.as<float>();
    const float* tgt = target.as<float>();
    float* delta = output_delta.as<float>();

    fill_n(delta, output_delta.size(), 0.0f);

    for (Index n = 0; n < batch_size; ++n)
        for (Index row = 0; row < grid_size; ++row)
            for (Index col = 0; col < grid_size; ++col)
            {
                const Index cell = ((n * grid_size + row) * grid_size + col) * channels;

                for (Index box = 0; box < boxes_per_cell; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    if (tgt[base + 4] == 1.0f)
                    {
                        const float output_box[4] = {out[base + 0], out[base + 1], out[base + 2], out[base + 3]};
                        const float target_box[4] = {tgt[base + 0], tgt[base + 1], tgt[base + 2], tgt[base + 3]};
                        const GIoUResult g = yolo_loss_giou_grad(output_box, target_box);

                        const float scale = lambda_giou * inv_batch;
                        delta[base + 0] = scale * clamp(g.cx_gradient, -grad_clip, grad_clip);
                        delta[base + 1] = scale * clamp(g.cy_gradient, -grad_clip, grad_clip);
                        delta[base + 2] = scale * clamp(g.w_gradient,  -grad_clip, grad_clip);
                        delta[base + 3] = scale * clamp(g.h_gradient,  -grad_clip, grad_clip);
                        delta[base + 4] = 2.0f * (out[base + 4] - g.iou) * inv_batch;

                        if (sigmoid_classes)
                        {
                            for (Index c = 0; c < classes_number; ++c)
                            {
                                const float p = out[base + 5 + c];
                                const float t = tgt[base + 5 + c];
                                delta[base + 5 + c] = (p - t) / (p * (1.0f - p) + EPSILON) * inv_batch;
                            }
                        }
                        else
                        {
                            for (Index c = 0; c < classes_number; ++c)
                                if (tgt[base + 5 + c] > 0.0f)
                                    delta[base + 5 + c] = -tgt[base + 5 + c] / (out[base + 5 + c] + EPSILON) * inv_batch;
                        }
                    }
                    else
                    {
                        delta[base + 4] = 2.0f * lambda_noobject * out[base + 4] * inv_batch;
                    }
                }
            }
}
#ifdef _MSC_VER
#pragma optimize("", on)
#endif

void yolo_gradient_cpu(const TensorView& output,
                       const TensorView& target,
                       const TensorView& output_delta,
                       const Dataset* dataset,
                       const NeuralNetwork* neural_network,
                       bool sigmoid_classes)
{
    check_yolo_loss(dataset, neural_network);
    const auto* yolo_dataset = static_cast<const YoloDataset*>(dataset);
    const Index boxes_per_cell = yolo_dataset->get_boxes_per_cell();
    const Index classes_number = yolo_dataset->get_classes_number();
    const float inv_batch = 1.0f / float(output.shape[0]);
    yolo_gradient_kernel(output, target, output_delta, boxes_per_cell, classes_number, sigmoid_classes, inv_batch);
}

vector<Index> yolo_detection_layer_indices(const NeuralNetwork* nn)
{
    vector<Index> result;
    if (!nn) return result;

    const auto& layers = nn->get_layers();
    result.reserve(layers.size());

    for (size_t i = 0; i < layers.size(); ++i)
        if (layers[i] && layers[i]->get_type() == LayerType::Detection)
            result.push_back(Index(i));

    return result;
}

Loss::EvaluationResult yolo_error_cpu_multi(const ForwardPropagation& fp,
                                            const TensorView& target_flat,
                                            const Dataset* dataset,
                                            const NeuralNetwork* nn,
                                            const vector<Index>& detection_indices,
                                            bool sigmoid_classes)
{
    check_yolo_loss(dataset, nn);
    const auto* yolo_dataset = static_cast<const YoloDataset*>(dataset);
    const Index boxes_per_head = yolo_dataset->get_boxes_per_head();
    const Index classes_number = yolo_dataset->get_classes_number();

    const float* tgt = target_flat.as<float>();
    const Index batch_size = target_flat.shape[0];

    Index per_sample_floats = 0;
    for (Index idx : detection_indices)
    {
        const Shape head_shape = nn->get_layer(idx)->get_output_shape();
        per_sample_floats += head_shape[0] * head_shape[1] * head_shape[2];
    }

    float total_error = 0.0f;
    Index head_offset = 0;
    for (Index detection_idx : detection_indices)
    {
        const TensorView head_output = fp.forward_slots[size_t(detection_idx)].back();
        const Shape head_shape = nn->get_layer(detection_idx)->get_output_shape();
        const Index head_floats = head_shape[0] * head_shape[1] * head_shape[2];

        vector<float> head_target(size_t(batch_size) * size_t(head_floats));
        for (Index n = 0; n < batch_size; ++n)
            copy_n(tgt + n * per_sample_floats + head_offset,
                   head_floats,
                   head_target.data() + n * head_floats);

        Shape head_target_shape = Shape({batch_size}).append(head_shape);
        TensorView head_target_view(head_target.data(), head_target_shape, Type::FP32);

        total_error += yolo_error_kernel(head_output, head_target_view, boxes_per_head, classes_number, sigmoid_classes);
        head_offset += head_floats;
    }

    Loss::EvaluationResult result;
    result.error = total_error / float(batch_size);
    return result;
}

void yolo_gradient_cpu_multi(const ForwardPropagation& fp,
                             const TensorView& target_flat,
                             BackPropagation& bp,
                             const Dataset* dataset,
                             const NeuralNetwork* nn,
                             const vector<Index>& detection_indices,
                             bool sigmoid_classes)
{
    check_yolo_loss(dataset, nn);
    const auto* yolo_dataset = static_cast<const YoloDataset*>(dataset);
    const Index boxes_per_head = yolo_dataset->get_boxes_per_head();
    const Index classes_number = yolo_dataset->get_classes_number();

    const float* tgt = target_flat.as<float>();
    const Index batch_size = target_flat.shape[0];
    const float inv_batch = 1.0f / float(batch_size);

    Index per_sample_floats = 0;
    for (Index idx : detection_indices)
    {
        const Shape head_shape = nn->get_layer(idx)->get_output_shape();
        per_sample_floats += head_shape[0] * head_shape[1] * head_shape[2];
    }

    Index head_offset = 0;
    for (Index detection_idx : detection_indices)
    {
        const TensorView head_output = fp.forward_slots[size_t(detection_idx)].back();
        const Shape head_shape = nn->get_layer(detection_idx)->get_output_shape();
        const Index head_floats = head_shape[0] * head_shape[1] * head_shape[2];

        vector<float> head_target(size_t(batch_size) * size_t(head_floats));
        for (Index n = 0; n < batch_size; ++n)
            copy_n(tgt + n * per_sample_floats + head_offset,
                   head_floats,
                   head_target.data() + n * head_floats);

        Shape head_target_shape = Shape({batch_size}).append(head_shape);
        TensorView head_target_view(head_target.data(), head_target_shape, Type::FP32);

        TensorView& head_delta = bp.layer_output_deltas[size_t(detection_idx)];
        yolo_gradient_kernel(head_output, head_target_view, head_delta, boxes_per_head, classes_number, sigmoid_classes, inv_batch);

        head_offset += head_floats;
    }
}

}
#endif // OPENNN_NO_VISION

Loss::Loss(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    set(new_neural_network, new_dataset);
}

void Loss::set(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    neural_network = new_neural_network;
    dataset = new_dataset;

    regularization_method = Regularization::L2;
    set_error(Error::MeanSquaredError);

}

void Loss::set_normalization_coefficient()
{
    normalization_coefficient = 1.0f;
    positives_weight = 1.0f;
    negatives_weight = 1.0f;

    if (!dataset || dataset->get_samples_number() == 0)
        return;

    if (error == Error::NormalizedSquaredError)
    {
        const vector<Index> training_indices = dataset->get_sample_indices("Training");
        const Shape target_shape = dataset->get_shape("Target");

        if (training_indices.empty() || target_shape.empty()) return;

        MatrixR targets(training_indices.size(), target_shape.size());
        dataset->fill_targets(training_indices,
                              dataset->get_feature_indices("Target"),
                              targets.data(),
                              false);

        const VectorR targets_mean = mean(targets);
        const float coefficient = (targets.rowwise() - targets_mean.transpose()).squaredNorm();

        normalization_coefficient = (coefficient < EPSILON) ? 1.0f : coefficient;
        return;
    }

    if (error == Error::WeightedSquaredError)
    {
        const Index targets_number = dataset->get_features_number("Target");
        if (targets_number != 1) return;

        const VectorI distribution = dataset->calculate_target_distribution();

        if (distribution.size() < 2) return;

        const Index negatives = distribution(0);
        const Index positives = distribution(1);

        if (positives == 0 || negatives == 0) return;

        const float total = float(positives + negatives);
        positives_weight = total / (2.0f * float(positives));
        negatives_weight = total / (2.0f * float(negatives));
    }
}

void Loss::back_propagate(const Batch& batch,
                          ForwardPropagation& forward_propagation,
                          BackPropagation& back_propagation) const
{
    if (batch.is_empty()) return;

    {
        PROFILE_SCOPE("loss:calculate_error");
        const EvaluationResult evaluation_result = calculate_error(batch, forward_propagation);
        back_propagation.error                = evaluation_result.error;
        back_propagation.accuracy             = evaluation_result.accuracy;
        back_propagation.active_tokens_count  = evaluation_result.active_tokens_count;
    }

    calculate_layers_error_gradient(batch, forward_propagation, back_propagation);

    back_propagation.regularization = 0.0f;
    back_propagation.loss = back_propagation.error;


    add_regularization(back_propagation);

    add_regularization_gradient(back_propagation);
}

float Loss::get_weighted_coefficient(const Batch& batch) const
{
    const Index total = dataset ? dataset->get_samples_number() : batch.get_samples_number();
    const Index samples = batch.get_samples_number();
    return float(total) / (float(samples) * (normalization_coefficient + EPSILON));
}

Loss::EvaluationResult Loss::calculate_error(const Batch& batch,
                                              const ForwardPropagation& forward_propagation) const
{
    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
    const TensorView target = batch.get_targets();

    EvaluationResult result;

    float* workspace_device = nullptr;
    const bool on_gpu = device::is_cuda_build() && neural_network && neural_network->is_gpu();
    if (on_gpu)
    {
        const Index workspace_floats = (error == Error::CrossEntropy3d)
            ? 3 * (input.size() / input.shape.back())
            : input.size();
        errors_device.grow_to(workspace_floats * Index(sizeof(float)));
        memory_debug::record("loss", "Loss::errors_device",
                             workspace_floats * Index(sizeof(float)),
                             format("batch={}", batch.get_samples_number()));
        workspace_device = errors_device.as<float>();
    }

    using enum Error;
    switch (error)
    {
    case MeanSquaredError:
        mean_squared_error(input, target, result.error, workspace_device);
        break;
    case NormalizedSquaredError:
        normalized_squared_error(input, target, normalization_coefficient, result.error, workspace_device);
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
        cross_entropy_3d(input, target, result.error, result.active_tokens_count, correct_tokens, workspace_device);
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
    {
        const vector<Index> detection_indices = yolo_detection_layer_indices(neural_network);
        const bool sigmoid = yolo_uses_sigmoid_classes(neural_network);
        result = detection_indices.size() > 1
            ? yolo_error_cpu_multi(forward_propagation, target, dataset, neural_network,
                                   detection_indices, sigmoid)
            : yolo_error_cpu(input, target, dataset, neural_network, sigmoid);
    }
#else
        throw runtime_error("YOLO loss not available: opennn was built with OpenNN_BUILD_VISION=OFF.");
#endif
        break;
    }

    return result;
}

bool Loss::supports_device_epoch_metrics() const
{
    return device::is_cuda_build()
        && neural_network
        && neural_network->is_gpu()
        && error != Error::MinkowskiError
        && error != Error::Yolo;
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

    const Index workspace_floats = (error == Error::CrossEntropy3d)
        ? 3 * (input.size() / input.shape.back())
        : input.size();

    errors_device.grow_to(workspace_floats * Index(sizeof(float)));
    metric_results_device.grow_to(Index(3 * sizeof(float)));
    memory_debug::record("loss", "Loss::errors_device",
                         workspace_floats * Index(sizeof(float)),
                         format("batch={}", batch.get_samples_number()));
    memory_debug::record("loss", "Loss::metric_results_device",
                         Index(3 * sizeof(float)),
                         format("batch={}", batch.get_samples_number()));

    float* const workspace = errors_device.as<float>();
    float* const results_device = metric_results_device.as<float>();
    cublasHandle_t handle = Backend::get_cublas_handle();

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
    case MeanSquaredError:
    case NormalizedSquaredError:
        input.dispatch([&](auto tag)
        {
            using TIn = decltype(tag);
            diff_to_fp32_cuda<TIn>(input.size(), input.as<TIn>(), target.as_float(), workspace);
        });
        reduce_dot_and_accumulate(input.size(),
                                  error == MeanSquaredError
                                      ? 1.0f / static_cast<float>(2 * input.shape[0])
                                      : 1.0f / (2.0f * (normalization_coefficient + EPSILON)));
        return true;

    case WeightedSquaredError:
        input.dispatch([&](auto tag)
        {
            using T = decltype(tag);
            weighted_squared_error_cuda<T>(input.size(), workspace, target.as<float>(), input.as<T>(),
                                           positives_weight, negatives_weight);
        });
        reduce_abs_and_accumulate(input.size(), 0.5f * get_weighted_coefficient(batch));
        return true;

    case CrossEntropy:
        input.dispatch([&](auto tag)
        {
            using T = decltype(tag);
            if (input.shape.back() == 1)
                binary_cross_entropy_cuda<T>(input.size(), workspace, target.as<float>(), input.as<T>(), EPSILON);
            else
                categorical_cross_entropy_cuda<T>(input.size(), workspace, target.as<float>(), input.as<T>(), EPSILON);
        });
        reduce_abs_and_accumulate(input.size(), 1.0f / static_cast<float>(input.shape[0]));
        return true;

    case CrossEntropy3d:
    {
        const Index vocabulary_size = input.shape.back();
        const Index token_count = input.size() / vocabulary_size;

        float* valid_mask_device = workspace + token_count;
        float* correct_mask_device = workspace + 2 * token_count;

        input.dispatch([&](auto tag)
        {
            using T = decltype(tag);
            cross_entropy_3d_multiple_forward_cuda<T>(token_count, to_int(vocabulary_size),
                input.as<T>(), target.as<float>(), workspace, valid_mask_device, correct_mask_device, EPSILON);
        });

        {
            device::CublasPointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_DEVICE);
            CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), workspace,             1, results_device + 0));
            CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), valid_mask_device,     1, results_device + 1));
            CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), correct_mask_device,   1, results_device + 2));
        }

        accumulate_cross_entropy_3d_metrics_cuda(results_device, error_sum_device, accuracy_sum_device);
        return true;
    }

    case MinkowskiError:
    case Yolo:
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

    if (!calculate_error_device_metrics(batch, forward_propagation, error_sum_device, accuracy_sum_device))
        return false;

    if (error == Error::CrossEntropy3d)
    {
        const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
        const TensorView target = batch.get_targets();
        TensorView& input_delta = back_propagation.get_output_delta();

        cross_entropy_3d_gradient_device_count(input, target, input_delta, metric_results_device.as<float>() + 1);
    }
    else
    {
        calculate_output_deltas(batch, forward_propagation, back_propagation);
    }

    back_propagation.error = 0.0f;
    back_propagation.accuracy = 0.0f;
    back_propagation.active_tokens_count = 0;
    back_propagation.regularization = 0.0f;
    back_propagation.loss = 0.0f;

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

void Loss::calculate_output_deltas(const Batch& batch, const ForwardPropagation& forward_propagation, BackPropagation& back_propagation) const
{
    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
    const TensorView target = batch.get_targets();
    const TensorView input_delta = back_propagation.get_output_delta();

    using enum Error;
    switch (error)
    {
    case MeanSquaredError:
        mean_squared_error_gradient(input, target, input_delta);
        break;
    case NormalizedSquaredError:
        normalized_squared_error_gradient(input, target, normalization_coefficient, input_delta);
        break;
    case WeightedSquaredError:
        weighted_squared_error_gradient(input, target, positives_weight, negatives_weight, get_weighted_coefficient(batch), input_delta);
        break;
    case CrossEntropy:
        cross_entropy_gradient(input, target, input_delta);
        break;
    case CrossEntropy3d:
        cross_entropy_3d_gradient(input, target, input_delta, back_propagation.active_tokens_count);
        break;
    case MinkowskiError:
        minkowski_error_gradient(input, target, minkowski_parameter, input_delta,
                                 neural_network && neural_network->is_gpu());
        break;
    case Yolo:
#ifndef OPENNN_NO_VISION
    {
        const vector<Index> detection_indices = yolo_detection_layer_indices(neural_network);
        const bool sigmoid = yolo_uses_sigmoid_classes(neural_network);
        if (detection_indices.size() > 1)
            yolo_gradient_cpu_multi(forward_propagation, target, back_propagation,
                                    dataset, neural_network, detection_indices, sigmoid);
        else
            yolo_gradient_cpu(input, target, input_delta, dataset, neural_network, sigmoid);
    }
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
        PROFILE_SCOPE("bwd:" + layers[i]->get_name());
        layers[i]->back_propagate(forward_propagation, back_propagation, i);
    }
}

void Loss::add_regularization(BackPropagation& back_propagation) const
{
    if (regularization_method == Regularization::NoRegularization) return;

    check_neural_network();

    const TensorView parameters(neural_network->get_parameters_data(),
                                {neural_network->get_parameters_size()},
                                Type::FP32,
                                neural_network->get_parameters_device());

    back_propagation.regularization = calculate_regularization(parameters);
    back_propagation.loss += back_propagation.regularization;
}

float Loss::calculate_regularization(const VectorR& parameters_vec) const
{
    const TensorView parameters(const_cast<float*>(parameters_vec.data()), { ssize(parameters_vec) });
    return calculate_regularization(parameters);
}

float Loss::calculate_regularization(const TensorView& parameters) const
{
    if (regularization_method == Regularization::NoRegularization || regularization_weight == 0.0f) return 0.0f;

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

static const vector<pair<Loss::Error, string>> error_entries = {
    {Loss::Error::MeanSquaredError,       "MeanSquaredError"},
    {Loss::Error::NormalizedSquaredError, "NormalizedSquaredError"},
    {Loss::Error::WeightedSquaredError,   "WeightedSquaredError"},
    {Loss::Error::CrossEntropy,           "CrossEntropy"},
    {Loss::Error::CrossEntropy3d,         "CrossEntropyError3d"},
    {Loss::Error::MinkowskiError,         "MinkowskiError"},
    {Loss::Error::Yolo,                   "Yolo"},
    {Loss::Error::Yolo,                   "YoloError"}
};

static const EnumMap<Loss::Error> error_map{error_entries};

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
    if (regularization_method == Regularization::NoRegularization || regularization_weight == 0.0f) return;

    check_neural_network();

    const Device gradient_device = gradient.device;

    if (gradient_device == Device::CUDA && neural_network->get_parameters_device() != Device::CUDA)
        neural_network->copy_parameters_device();
    else if (gradient_device == Device::CPU && neural_network->get_parameters_device() == Device::CUDA)
        neural_network->copy_parameters_host();

    const TensorView parameters(neural_network->get_parameters_data(),
                                { neural_network->get_parameters_size() },
                                Type::FP32,
                                gradient_device);

    if (regularization_method == Regularization::L1)
        l1_regularization_gradient(parameters, regularization_weight, gradient);
    else if (regularization_method == Regularization::L2)
        l2_regularization_gradient(parameters, regularization_weight, gradient);
}

void Loss::add_regularization_gradient(BackPropagation& back_propagation) const
{
    if (regularization_method == Regularization::NoRegularization || regularization_weight == 0.0f) return;

    check_neural_network();

    add_regularization_gradient(TensorView(back_propagation.gradient.as<float>(),
                                           { neural_network->get_parameters_size() },
                                           Type::FP32,
                                           back_propagation.gradient.device_type));
}

void Loss::regularization_from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "Regularization");

    set_regularization(read_json_string(root_element, "Type"));

    if (root_element->has("RegularizationWeight"))
        set_regularization_weight(float(read_json_float(root_element, "RegularizationWeight")));
}

void Loss::regularization_to_JSON(JsonWriter& file_stream) const
{
    file_stream.open_element("Regularization");
    write_json(file_stream, {
        {"Type", regularization_to_string(regularization_method)},
        {"RegularizationWeight", to_string(regularization_weight)}
    });
    file_stream.close_element();
}

float Loss::calculate_h(const float x)
{
    constexpr float finite_difference_step = 1e-3f;
    return finite_difference_step * (1.0f + abs(x));
}

void Loss::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Loss");
    write_json(printer, {
        {"Method", get_name()},
        {"Regularization", regularization_to_string(regularization_method)},
        {"RegularizationWeight", to_string(regularization_weight)}
    });

    if (error == Error::NormalizedSquaredError)
        add_json_field(printer, "NormalizationCoefficient", to_string(normalization_coefficient));

    if (error == Error::WeightedSquaredError)
        write_json(printer, {
            {"PositivesWeight", to_string(positives_weight)},
            {"NegativesWeight", to_string(negatives_weight)}
        });

    if (error == Error::MinkowskiError)
        add_json_field(printer, "MinkowskiParameter", to_string(minkowski_parameter));

    printer.close_element();
}

void Loss::from_JSON(const JsonDocument& document)
{
    const Json* root = document.first_child("Loss");
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
