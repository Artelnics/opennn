//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E T W O R K   D I F F E R E N T I A L   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"
#include "opennn/core/variable.h"
#include "opennn/core/tensor_operations.h"

namespace opennn
{

class NeuralNetwork;

struct NetworkDifferential
{
    enum class Kind { Scale, Dense, Unscale, Clamp, Activate };

    struct LayerSnapshot
    {
        Kind kind = Kind::Dense;
        MatrixR weights;
        VectorR bias;
        ActivationFunction activation = ActivationFunction::Identity;
        vector<ScalerMethod> methods;
        VectorR minimum, maximum, mean, deviation;
        float min_range = -1.0f, max_range = 1.0f;
        bool clamping_active = true;
    };

    vector<LayerSnapshot> layers;

    mutable optional<VectorR> tape_x;
    mutable vector<VectorR> layer_inputs;
    mutable vector<VectorR> layer_outputs;

    static float guarded(const float value)
    {
        constexpr float floor_value = 1e-12f;
        if (value > floor_value)  return value;
        if (value < -floor_value) return value;
        return floor_value;
    }

    static bool is_degenerate(const float span) { return abs(span) < EPSILON; }

    VectorR scale_forward(const LayerSnapshot& layer, const VectorR& in) const
    {
        VectorR out(in.size());
        for (Index j = 0; j < in.size(); ++j)
        {
            const float x = in(j);
            switch (layer.methods[j])
            {
            case ScalerMethod::None:                  out(j) = x; break;
            case ScalerMethod::MinimumMaximum:        out(j) = is_degenerate(layer.maximum(j) - layer.minimum(j)) ? 0.0f : (x - layer.minimum(j)) / (layer.maximum(j) - layer.minimum(j)) * (layer.max_range - layer.min_range) + layer.min_range; break;
            case ScalerMethod::MeanStandardDeviation: out(j) = is_degenerate(layer.deviation(j)) ? 0.0f : (x - layer.mean(j)) / layer.deviation(j); break;
            case ScalerMethod::StandardDeviation:     out(j) = is_degenerate(layer.deviation(j)) ? 0.0f : x / layer.deviation(j); break;
            case ScalerMethod::Logarithm:             out(j) = log(guarded(x)); break;
            case ScalerMethod::ImageMinMax:           out(j) = x / 255.0f; break;
            }
        }
        return out;
    }

    VectorR scale_derivative(const LayerSnapshot& layer, const VectorR& in) const
    {
        VectorR d(in.size());
        for (Index j = 0; j < in.size(); ++j)
            switch (layer.methods[j])
            {
            case ScalerMethod::None:                  d(j) = 1.0f; break;
            case ScalerMethod::MinimumMaximum:        d(j) = is_degenerate(layer.maximum(j) - layer.minimum(j)) ? 0.0f : (layer.max_range - layer.min_range) / (layer.maximum(j) - layer.minimum(j)); break;
            case ScalerMethod::MeanStandardDeviation: d(j) = is_degenerate(layer.deviation(j)) ? 0.0f : 1.0f / layer.deviation(j); break;
            case ScalerMethod::StandardDeviation:     d(j) = is_degenerate(layer.deviation(j)) ? 0.0f : 1.0f / layer.deviation(j); break;
            case ScalerMethod::Logarithm:             d(j) = 1.0f / guarded(in(j)); break;
            case ScalerMethod::ImageMinMax:           d(j) = 1.0f / 255.0f; break;
            }
        return d;
    }

    VectorR unscale_forward(const LayerSnapshot& layer, const VectorR& in) const
    {
        VectorR out(in.size());
        for (Index j = 0; j < in.size(); ++j)
        {
            const float x = in(j);
            switch (layer.methods[j])
            {
            case ScalerMethod::None:                  out(j) = x; break;
            case ScalerMethod::MinimumMaximum:        out(j) = is_degenerate(layer.maximum(j) - layer.minimum(j)) ? layer.minimum(j) : (x - layer.min_range) / guarded(layer.max_range - layer.min_range) * (layer.maximum(j) - layer.minimum(j)) + layer.minimum(j); break;
            case ScalerMethod::MeanStandardDeviation: out(j) = x * layer.deviation(j) + layer.mean(j); break;
            case ScalerMethod::StandardDeviation:     out(j) = x * layer.deviation(j); break;
            case ScalerMethod::Logarithm:             out(j) = exp(x); break;
            case ScalerMethod::ImageMinMax:           out(j) = x * 255.0f; break;
            }
        }
        return out;
    }

    VectorR unscale_derivative(const LayerSnapshot& layer, const VectorR& in) const
    {
        VectorR d(in.size());
        for (Index j = 0; j < in.size(); ++j)
            switch (layer.methods[j])
            {
            case ScalerMethod::None:                  d(j) = 1.0f; break;
            case ScalerMethod::MinimumMaximum:        d(j) = is_degenerate(layer.maximum(j) - layer.minimum(j)) ? 0.0f : (layer.maximum(j) - layer.minimum(j)) / guarded(layer.max_range - layer.min_range); break;
            case ScalerMethod::MeanStandardDeviation: d(j) = layer.deviation(j); break;
            case ScalerMethod::StandardDeviation:     d(j) = layer.deviation(j); break;
            case ScalerMethod::Logarithm:             d(j) = exp(in(j)); break;
            case ScalerMethod::ImageMinMax:           d(j) = 255.0f; break;
            }
        return d;
    }

    VectorR clamp_derivative(const LayerSnapshot& layer, const VectorR& in) const
    {
        if (!layer.clamping_active) return VectorR::Ones(in.size());

        VectorR d(in.size());
        for (Index j = 0; j < in.size(); ++j)
            d(j) = (in(j) > layer.minimum(j) && in(j) < layer.maximum(j)) ? 1.0f : 0.0f;
        return d;
    }

    void build(const NeuralNetwork&);

    VectorR forward(const VectorR& x) const
    {
        const size_t layers_number = layers.size();
        layer_inputs.assign(layers_number, VectorR());
        layer_outputs.assign(layers_number, VectorR());

        VectorR activation = x;
        for (size_t i = 0; i < layers_number; ++i)
        {
            layer_inputs[i] = activation;
            const LayerSnapshot& layer = layers[i];

            if (layer.kind == Kind::Scale)
                activation = scale_forward(layer, activation);
            else if (layer.kind == Kind::Unscale)
                activation = unscale_forward(layer, activation);
            else if (layer.kind == Kind::Clamp)
                activation = layer.clamping_active ? activation.cwiseMax(layer.minimum).cwiseMin(layer.maximum).eval() : activation;
            else if (layer.kind == Kind::Activate)
                activation = activation_forward_values(layer.activation, activation);
            else
                activation = activation_forward_values(layer.activation,
                                                       (layer.weights.transpose() * activation + layer.bias).eval());

            layer_outputs[i] = activation;
        }

        tape_x = x;
        return activation;
    }

    VectorR vjp(const VectorR& x, const VectorR& cotangent) const
    {
        if (!tape_x || tape_x->size() != x.size()
            || !(tape_x->array() == x.array()).all())
            forward(x);

        VectorR carried = cotangent;
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
        {
            const LayerSnapshot& layer = layers[i];

            if (layer.kind == Kind::Scale)
                carried = (carried.array() * scale_derivative(layer, layer_inputs[i]).array()).matrix();
            else if (layer.kind == Kind::Unscale)
                carried = (carried.array() * unscale_derivative(layer, layer_inputs[i]).array()).matrix();
            else if (layer.kind == Kind::Clamp)
                carried = (carried.array() * clamp_derivative(layer, layer_inputs[i]).array()).matrix();
            else if (layer.kind == Kind::Activate)
                carried = (carried.array()
                         * activation_derivative_from_output_values(layer.activation, layer_outputs[i]).array()).matrix();
            else
            {
                carried = layer.weights
                        * (carried.array()
                         * activation_derivative_from_output_values(layer.activation, layer_outputs[i]).array()).matrix();
            }
        }
        return carried;
    }
};

struct NetworkJacobian
{
    unique_ptr<NetworkDifferential> differential;
    bool ready = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
