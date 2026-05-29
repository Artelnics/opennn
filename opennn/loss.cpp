//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_utilities.h"
#include "batch.h"
#include "dataset.h"
#include "tabular_dataset.h"
#include "loss.h"
#include "yolo_dataset.h"
#include "error_utilities.h"
#include "profiler.h"
#include "forward_propagation.h"
#include "cuda_dispatch.h"
#include "back_propagation.h"
#include "statistics.h"
#include <Eigen/LU>

namespace opennn
{

#ifndef OPENNN_NO_VISION
namespace
{

float yolo_loss_iou(const float* a, const float* b)
{
    const float a_left = a[0] - 0.5f * a[2];
    const float a_top = a[1] - 0.5f * a[3];
    const float a_right = a[0] + 0.5f * a[2];
    const float a_bottom = a[1] + 0.5f * a[3];

    const float b_left = b[0] - 0.5f * b[2];
    const float b_top = b[1] - 0.5f * b[3];
    const float b_right = b[0] + 0.5f * b[2];
    const float b_bottom = b[1] + 0.5f * b[3];

    const float inter_w = max(0.0f, min(a_right, b_right) - max(a_left, b_left));
    const float inter_h = max(0.0f, min(a_bottom, b_bottom) - max(a_top, b_top));
    const float inter = inter_w * inter_h;
    const float area = a[2] * a[3] + b[2] * b[3] - inter;

    return area > 0.0f ? inter / area : 0.0f;
}

void check_yolo_loss(const Dataset* dataset)
{
    if (is_gpu())
        throw runtime_error("YOLO loss GPU implementation not available yet.");

    if (!dynamic_cast<const YoloDataset*>(dataset))
        throw runtime_error("YOLO loss requires YoloDataset.");
}

Loss::EvaluationResult yolo_error_cpu(const TensorView& output,
                                      const TensorView& target,
                                      const Dataset* dataset)
{
    check_yolo_loss(dataset);

    constexpr float lambda_coord = 5.0f;
    constexpr float lambda_noobject = 0.5f;

    const auto* yolo_dataset = static_cast<const YoloDataset*>(dataset);
    const Index B = yolo_dataset->get_boxes_per_cell();
    const Index C = yolo_dataset->get_classes_number();
    const Index values_per_box = 5 + C;
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

                for (Index box = 0; box < B; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    if (tgt[base + 4] == 1.0f)
                    {
                        const float target_box[4] = {tgt[base + 0], tgt[base + 1], tgt[base + 2], tgt[base + 3]};
                        const float output_box[4] = {out[base + 0], out[base + 1], out[base + 2], out[base + 3]};
                        const float iou = yolo_loss_iou(target_box, output_box);

                        coordinate_loss += pow(out[base + 0] - tgt[base + 0], 2.0f)
                                         +  pow(out[base + 1] - tgt[base + 1], 2.0f)
                                         +  pow(sqrt(out[base + 2] + EPSILON) - sqrt(tgt[base + 2] + EPSILON), 2.0f)
                                         +  pow(sqrt(out[base + 3] + EPSILON) - sqrt(tgt[base + 3] + EPSILON), 2.0f);

                        object_loss += pow(out[base + 4] - iou, 2.0f);

                        for (Index c = 0; c < C; ++c)
                            if (tgt[base + 5 + c] > 0.0f)
                                class_loss -= log(out[base + 5 + c] + EPSILON);
                    }
                    else
                    {
                        noobject_loss += pow(out[base + 4], 2.0f);
                    }
                }
            }

    Loss::EvaluationResult result;
    result.error = (lambda_coord * coordinate_loss + object_loss
                  + lambda_noobject * noobject_loss + class_loss) / float(batch_size);
    return result;
}

void yolo_gradient_cpu(const TensorView& output,
                       const TensorView& target,
                       const TensorView& output_delta,
                       const Dataset* dataset)
{
    check_yolo_loss(dataset);

    constexpr float lambda_coord = 5.0f;
    constexpr float lambda_noobject = 0.5f;

    const auto* yolo_dataset = static_cast<const YoloDataset*>(dataset);
    const Index B = yolo_dataset->get_boxes_per_cell();
    const Index C = yolo_dataset->get_classes_number();
    const Index values_per_box = 5 + C;
    const Index batch_size = output.shape[0];
    const Index grid_size = output.shape[1];
    const Index channels = output.shape[3];
    const float inv_batch = 1.0f / float(batch_size);

    const float* out = output.as<float>();
    const float* tgt = target.as<float>();
    float* delta = output_delta.as<float>();

    fill_n(delta, output_delta.size(), 0.0f);

    for (Index n = 0; n < batch_size; ++n)
        for (Index row = 0; row < grid_size; ++row)
            for (Index col = 0; col < grid_size; ++col)
            {
                const Index cell = ((n * grid_size + row) * grid_size + col) * channels;

                for (Index box = 0; box < B; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    if (tgt[base + 4] == 1.0f)
                    {
                        const float target_box[4] = {tgt[base + 0], tgt[base + 1], tgt[base + 2], tgt[base + 3]};
                        const float output_box[4] = {out[base + 0], out[base + 1], out[base + 2], out[base + 3]};
                        const float iou = yolo_loss_iou(target_box, output_box);

                        delta[base + 0] = 2.0f * lambda_coord * (out[base + 0] - tgt[base + 0]) * inv_batch;
                        delta[base + 1] = 2.0f * lambda_coord * (out[base + 1] - tgt[base + 1]) * inv_batch;

                        const float sqrt_w = sqrt(out[base + 2] + EPSILON);
                        const float sqrt_h = sqrt(out[base + 3] + EPSILON);
                        delta[base + 2] = lambda_coord * (sqrt_w - sqrt(tgt[base + 2] + EPSILON)) / sqrt_w * inv_batch;
                        delta[base + 3] = lambda_coord * (sqrt_h - sqrt(tgt[base + 3] + EPSILON)) / sqrt_h * inv_batch;
                        delta[base + 4] = 2.0f * (out[base + 4] - iou) * inv_batch;

                        for (Index c = 0; c < C; ++c)
                            if (tgt[base + 5 + c] > 0.0f)
                                delta[base + 5 + c] = -tgt[base + 5 + c] / (out[base + 5 + c] + EPSILON) * inv_batch;
                    }
                    else
                    {
                        delta[base + 4] = 2.0f * lambda_noobject * out[base + 4] * inv_batch;
                    }
                }
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
        const EvaluationResult eval = calculate_error(batch, forward_propagation);
        back_propagation.error                = eval.error;
        back_propagation.accuracy             = eval.accuracy;
        back_propagation.active_tokens_count  = eval.active_tokens_count;
    }

    calculate_layers_error_gradient(batch, forward_propagation, back_propagation);

    back_propagation.regularization = 0.0f;
    back_propagation.loss = back_propagation.error;

    // Regularization

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
#ifdef OPENNN_HAS_CUDA
    if (is_gpu())
    {
        const Index workspace_floats = (error == Error::CrossEntropy3d)
            ? 3 * (input.size() / input.shape.back())
            : input.size();
        errors_device.grow_to(workspace_floats * Index(sizeof(float)));
        workspace_device = errors_device.as<float>();
    }
#endif

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
        if (input.shape.back() == 1)
            binary_cross_entropy(input, target, result.error, workspace_device);
        else
            categorical_cross_entropy(input, target, result.error, workspace_device);
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
        result = yolo_error_cpu(input, target, dataset);
#else
        throw runtime_error("YOLO loss not available: opennn was built with OpenNN_BUILD_VISION=OFF.");
#endif
        break;
    }

    return result;
}

#ifdef OPENNN_HAS_CUDA

bool Loss::supports_device_epoch_metrics() const
{
    return is_gpu() && error != Error::MinkowskiError && error != Error::Yolo;
}

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

    float* const workspace = errors_device.as<float>();
    float* const results_device = metric_results_device.as<float>();
    cublasHandle_t handle = Backend::get_cublas_handle();

    auto reduce_abs_and_accumulate = [&](Index n, float scale)
    {
        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        CHECK_CUBLAS(cublasSasum(handle, to_int(n), workspace, 1, results_device));
        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        accumulate_scaled_metric_cuda(results_device, scale, error_sum_device);
    };

    auto reduce_dot_and_accumulate = [&](Index n, float scale)
    {
        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        CHECK_CUBLAS(cublasSdot(handle, to_int(n), workspace, 1, workspace, 1, results_device));
        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        accumulate_scaled_metric_cuda(results_device, scale, error_sum_device);
    };

    using enum Error;
    switch (error)
    {
    case MeanSquaredError:
        input.dispatch([&](auto tag)
        {
            using TIn = decltype(tag);
            diff_to_fp32_cuda<TIn>(input.size(), input.as<TIn>(), target.as_float(), workspace);
        });
        reduce_dot_and_accumulate(input.size(), 1.0f / static_cast<float>(2 * input.shape[0]));
        return true;

    case NormalizedSquaredError:
        input.dispatch([&](auto tag)
        {
            using TIn = decltype(tag);
            diff_to_fp32_cuda<TIn>(input.size(), input.as<TIn>(), target.as_float(), workspace);
        });
        reduce_dot_and_accumulate(input.size(), 1.0f / (2.0f * (normalization_coefficient + EPSILON)));
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
                multiple_cross_entropy_cuda<T>(input.size(), workspace, target.as<float>(), input.as<T>(), EPSILON);
        });
        reduce_abs_and_accumulate(input.size(), 1.0f / static_cast<float>(input.shape[0]));
        return true;

    case CrossEntropy3d:
    {
        const Index vocabulary_size = input.shape.back();
        const Index sequence_length = input.shape[input.get_rank() - 2];
        const Index batch_size = input.size() / (sequence_length * vocabulary_size);
        const Index token_count = batch_size * sequence_length;

        float* valid_mask_device = workspace + token_count;
        float* correct_mask_device = workspace + 2 * token_count;

        input.dispatch([&](auto tag)
        {
            using T = decltype(tag);
            cross_entropy_3d_multiple_forward_cuda<T>(token_count, to_int(vocabulary_size),
                input.as<T>(), target.as<float>(), workspace, valid_mask_device, correct_mask_device, EPSILON);
        });

        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), workspace,             1, results_device + 0));
        CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), valid_mask_device,     1, results_device + 1));
        CHECK_CUBLAS(cublasSasum(handle, to_int(token_count), correct_mask_device,   1, results_device + 2));
        CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

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

    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
    const TensorView target = batch.get_targets();
    TensorView& input_delta = back_propagation.get_output_delta();

    if (error == Error::CrossEntropy3d)
    {
        if (!calculate_error_device_metrics(batch, forward_propagation, error_sum_device, accuracy_sum_device))
            return false;

        cross_entropy_3d_gradient_device_count(input, target, input_delta, metric_results_device.as<float>() + 1);
    }
    else
    {
        if (!calculate_error_device_metrics(batch, forward_propagation, error_sum_device, accuracy_sum_device))
            return false;

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
            weighted_squared_error_gradient(input, target, positives_weight, negatives_weight,
                                            get_weighted_coefficient(batch), input_delta);
            break;
        case CrossEntropy:
            cross_entropy_gradient(input, target, input_delta);
            break;
        case CrossEntropy3d:
        case MinkowskiError:
        case Yolo:
            return false;
        }
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
        minkowski_error_gradient(input, target, minkowski_parameter, input_delta);
        break;
    case Yolo:
#ifndef OPENNN_NO_VISION
        yolo_gradient_cpu(input, target, input_delta, dataset);
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
                                {neural_network->get_parameters_size()});

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

static const vector<pair<Loss::Error, string>> error_map = {
    {Loss::Error::MeanSquaredError,       "MeanSquaredError"},
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

    for (const auto& [error_value, error_name] : error_map)
        if (error_value == error) { name = error_name; return; }
}

void Loss::set_error(const string& new_name)
{
    for (const auto& [error_value, error_name] : error_map)
        if (error_name == new_name) { set_error(error_value); return; }

    throw runtime_error(format("Unknown loss method: {}", new_name));
}

void Loss::add_regularization_gradient(BackPropagation& back_propagation) const
{
    if (regularization_method == Regularization::NoRegularization || regularization_weight == 0.0f) return;

    check_neural_network();

    const Index parameters_number = neural_network->get_parameters_size();

    const TensorView parameters(neural_network->get_parameters_data(), { parameters_number });
    TensorView gradient(back_propagation.gradient.as<float>(), { parameters_number });

    if (regularization_method == Regularization::L1)
        l1_regularization_gradient(parameters, regularization_weight, gradient);
    else if (regularization_method == Regularization::L2)
        l2_regularization_gradient(parameters, regularization_weight, gradient);
}

void Loss::regularization_from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "Regularization");

    set_regularization(read_json_string(root_element, "Type"));

    if (root_element->has("RegularizationWeight"))
        set_regularization_weight(float(read_json_type(root_element, "RegularizationWeight")));
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
    regularization_weight = read_json_type(root, "RegularizationWeight");

    if (root->first_child("NormalizationCoefficient"))
        normalization_coefficient = read_json_type(root, "NormalizationCoefficient");

    if (root->first_child("PositivesWeight")) {
        positives_weight = read_json_type(root, "PositivesWeight");
        negatives_weight = read_json_type(root, "NegativesWeight");
    }

    if (root->first_child("MinkowskiParameter"))
        minkowski_parameter = read_json_type(root, "MinkowskiParameter");
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
