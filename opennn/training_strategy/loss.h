//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/device_backend.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/back_propagation.h"

namespace opennn
{

class Dataset;

struct Batch;
struct ForwardPropagation;

class Loss
{

public:

    enum class Error{MeanSquaredError,
                     MeanAbsoluteError,
                     NormalizedSquaredError,
                     WeightedSquaredError,
                     CrossEntropy,
                     CrossEntropy3d,
                     MinkowskiError,
                     Yolo};

    enum class Regularization{L1, L2, NoRegularization};

    static const EnumMap<Regularization>& regularization_map()
    {
        static const EnumMap<Regularization> map{
            {Regularization::NoRegularization, "None"},
            {Regularization::L1,               "L1"},
            {Regularization::L2,               "L2"}
        };
        return map;
    }

    static const string& regularization_to_string(Regularization regularization)
    {
        return regularization_map().to_string(regularization);
    }

    static Regularization string_to_regularization(const string& name)
    {
        if (name == "NoRegularization") return Regularization::NoRegularization;
        return regularization_map().from_string(name);
    }

    explicit Loss(NeuralNetwork* = nullptr, Dataset* = nullptr);

    virtual ~Loss() = default;

    const NeuralNetwork* get_neural_network() const noexcept
    {
        return neural_network;
    }

    NeuralNetwork* get_neural_network()
    {
        return neural_network;
    }

    const Dataset* get_dataset() const noexcept
    {
        return dataset;
    }

    Dataset* get_dataset()
    {
        return dataset;
    }

    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);

    void set_neural_network(NeuralNetwork* new_neural_network) { neural_network = new_neural_network; }

    virtual void set_dataset(Dataset* new_dataset) { dataset = new_dataset; }

    string get_regularization_method() const { return regularization_to_string(regularization_method); }
    void set_regularization(const string& new_regularization_method) { regularization_method = string_to_regularization(new_regularization_method); }
    void set_regularization(Regularization new_regularization) { regularization_method = new_regularization; }
    void set_regularization_weight(const float new_regularization_weight) { regularization_weight = new_regularization_weight; }

    void set_normalization_coefficient();

    struct EvaluationResult
    {
        float error = 0.0f;
        float accuracy = 0.0f;
        Index active_tokens_count = 0;
    };

    EvaluationResult calculate_error(const Batch&,
                                     const ForwardPropagation&) const;

    void set_error(const Error&);
    void set_error(const string&);

    Error get_error() const noexcept { return error; }

    vector<Index> get_output_delta_layer_indices() const;

    bool output_delta_overwrites_outputs() const;

    void back_propagate(const Batch&,
                        ForwardPropagation&,
                        BackPropagation&) const;

    bool supports_device_epoch_metrics() const;

    bool back_propagate_device_metrics(const Batch&,
                                       ForwardPropagation&,
                                       BackPropagation&,
                                       float*,
                                       float*) const;

    bool calculate_error_device_metrics(const Batch&,
                                        const ForwardPropagation&,
                                        float*,
                                        float*) const;

    float calculate_regularization(const VectorR&) const;
    float calculate_regularization(const TensorView&) const;

    void add_regularization_gradient(BackPropagation&) const;
    void add_regularization_gradient(const TensorView&) const;

    void from_JSON(const JsonDocument&);

    void to_JSON(JsonWriter&) const;

    void regularization_from_JSON(const JsonDocument&);
    void regularization_to_JSON(JsonWriter&) const;

    const string& get_name() const noexcept { return name; }

    void set_yolo_lambda_noobj(float v)     { yolo_lambda_noobj     = v; }
    void set_yolo_lambda_class(float v)     { yolo_lambda_class     = v; }
    void set_yolo_lambda_giou(float v)      { yolo_lambda_giou      = v; }
    void set_yolo_lambda_dfl(float v)       { yolo_lambda_dfl       = v; }
    void set_yolo_focal_gamma(float v)      { yolo_focal_gamma      = v; }
    void set_yolo_obj_focal_gamma(float v)  { yolo_obj_focal_gamma  = v; }
protected:

    Error error = Error::MeanSquaredError;

    float normalization_coefficient = 1.0f;
    float positives_weight = 1.0f;
    float negatives_weight = 1.0f;
    Index weighted_samples_number = 0;
    float minkowski_parameter = 1.5f;

    float yolo_lambda_giou     = 5.0f;
    float yolo_lambda_dfl      = 1.5f;
    float yolo_lambda_noobj    = 0.5f;
    float yolo_lambda_class    = 1.0f;
    float yolo_focal_gamma     = 0.0f;
    float yolo_obj_focal_gamma = 0.0f;

    Regularization regularization_method = Regularization::NoRegularization;
    float regularization_weight = 0.001f;

    NeuralNetwork* neural_network = nullptr;
    Dataset* dataset = nullptr;

    string name = "Loss";

private:

    void add_regularization_gradient(const TensorView&,
                                     Index parameter_offset) const;

    void check_neural_network() const
    {
        throw_if(!neural_network, "Loss error: neural network is not set.");
    }

    bool runs_on_gpu() const noexcept
    {
        return device::is_cuda_build() && neural_network && neural_network->is_gpu();
    }

    bool has_regularization() const noexcept
    {
        return regularization_method != Regularization::NoRegularization
            && regularization_weight != 0.0f;
    }

    Index error_workspace_floats(const TensorView&) const;
    float* ensure_error_workspace(Buffer&, const TensorView&,
                                  Index batch_samples,
                                  Index reduction_floats = 0) const;

    float get_weighted_coefficient(const Batch& batch) const { return get_batch_scale(batch) / (normalization_coefficient + EPSILON); }

    float get_batch_scale(const Batch&) const;

    void calculate_layers_error_gradient(const Batch&,
                                         ForwardPropagation&,
                                         BackPropagation&) const;

    void back_propagate_layers(ForwardPropagation&,
                               BackPropagation&) const;

    void calculate_output_deltas(const Batch&,
                                 const ForwardPropagation&,
                                 BackPropagation&) const;

#ifndef OPENNN_NO_VISION
    EvaluationResult calculate_yolo(const ForwardPropagation&,
                                    const TensorView& target,
                                    BackPropagation*) const;
#endif
};

#ifndef OPENNN_NO_VISION

struct YoloLambdas
{
    float giou            = 5.0f;
    float dfl             = 1.5f;
    float noobj           = 0.5f;
    float cls             = 1.0f;
    float focal_gamma     = 0.0f;
    float obj_focal_gamma = 0.0f;
};

float yolo_error_kernel(const TensorView& output,
                        const TensorView& target,
                        Index boxes_per_cell,
                        Index classes_number,
                        bool sigmoid_classes,
                        YoloLambdas lam);

void yolo_gradient_kernel(const TensorView& output,
                          const TensorView& target,
                          const TensorView& output_delta,
                          Index boxes_per_cell,
                          Index classes_number,
                          bool sigmoid_classes,
                          float inv_batch,
                          YoloLambdas lam);

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
