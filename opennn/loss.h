//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "neural_network.h"
#include "back_propagation.h"

namespace opennn
{

class Dataset;

struct Batch;
struct ForwardPropagation;

class Loss
{

public:

    enum class Error{MeanSquaredError,
                     NormalizedSquaredError,
                     WeightedSquaredError,
                     CrossEntropy,
                     CrossEntropy3d,
                     MinkowskiError,
                     Yolo};

    enum class Regularization{L1, L2, NoRegularization};

    static const EnumMap<Regularization>& regularization_map()
    {
        static const vector<pair<Regularization, string>> entries = {
            {Regularization::NoRegularization, "None"},
            {Regularization::L1,               "L1"},
            {Regularization::L2,               "L2"}
        };
        static const EnumMap<Regularization> map{entries};
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

    const string& get_regularization_method() const { return regularization_to_string(regularization_method); }
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
    static float calculate_h(const float);

    void print() const {}

    void set_yolo_lambda_giou(float v)      { yolo_lambda_giou      = v; }
    void set_yolo_lambda_noobj(float v)     { yolo_lambda_noobj     = v; }
    void set_yolo_lambda_class(float v)     { yolo_lambda_class     = v; }
    void set_yolo_focal_gamma(float v)      { yolo_focal_gamma      = v; }
    void set_yolo_obj_focal_gamma(float v)  { yolo_obj_focal_gamma  = v; }

    float get_yolo_lambda_giou()      const noexcept { return yolo_lambda_giou;      }
    float get_yolo_lambda_noobj()     const noexcept { return yolo_lambda_noobj;      }
    float get_yolo_lambda_class()     const noexcept { return yolo_lambda_class;      }
    float get_yolo_focal_gamma()      const noexcept { return yolo_focal_gamma;       }
    float get_yolo_obj_focal_gamma()  const noexcept { return yolo_obj_focal_gamma;   }

private:

    void check_neural_network() const
    {
        throw_if(!neural_network, "Loss error: neural network is not set.");
    }

    void add_regularization(BackPropagation&) const;

    float get_weighted_coefficient(const Batch&) const;

    void calculate_layers_error_gradient(const Batch&,
                                         ForwardPropagation&,
                                         BackPropagation&) const;

    void back_propagate_layers(ForwardPropagation&,
                               BackPropagation&) const;

    void calculate_output_deltas(const Batch&,
                                    const ForwardPropagation&,
                                    BackPropagation&) const;

protected:

    Error error = Error::MeanSquaredError;

    float normalization_coefficient = 1.0f;
    float positives_weight = 1.0f;
    float negatives_weight = 1.0f;
    Index weighted_samples_number = 0;
    float minkowski_parameter = 1.5f;

    float yolo_lambda_giou     = 5.0f;
    float yolo_lambda_noobj    = 0.5f;
    float yolo_lambda_class    = 1.0f;
    float yolo_focal_gamma     = 0.0f;  // 0 = standard BCE; 2.0 = focal on class
    float yolo_obj_focal_gamma = 0.0f;  // 0 = standard BCE; 2.0 = focal on objectness

    mutable Buffer errors_device{Device::CUDA};
    mutable Buffer metric_results_device{Device::CUDA};
    mutable Buffer yolo_target_device{Device::CUDA};

    Regularization regularization_method = Regularization::NoRegularization;
    float regularization_weight = 0.001f;

    NeuralNetwork* neural_network = nullptr;
    Dataset* dataset = nullptr;

    string name = "Loss";
};

// CPU numerical gradient check: returns max relative error between analytical
// and finite-difference gradients. Below 1e-4 means the loss is self-consistent.
float yolo_loss_gradient_check_cpu();

// CPU expected-value check: compares forward loss against hand-computed expected
// values, and verifies gradient directions are correct (not just self-consistent).
// Returns max absolute error; prints per-test results. Below 1e-4 means correct.
float yolo_loss_expected_value_check_cpu();

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
