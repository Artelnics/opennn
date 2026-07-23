//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E T W O R K   D I F F E R E N T I A L   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "variable.h"
#include "tensor_operations.h"

namespace opennn
{

class NeuralNetwork;

struct NetworkDifferential // @todo this should not exist
{
    static inline long long benchmark_vjp_count = 0;

    enum class Kind { Scale, Dense, Unscale, Bound, Activate };

    struct LayerSnapshot
    {
        Kind kind = Kind::Dense;
        MatrixR weights;
        VectorR bias;
        ActivationFunction activation = ActivationFunction::Identity;
        vector<ScalerMethod> methods;
        VectorR minimum, maximum, mean, deviation;
        float min_range = -1.0f, max_range = 1.0f;
        bool bounding_active = true;
    };

    vector<LayerSnapshot> layers;

    mutable bool tape_valid = false;
    mutable VectorR tape_x;
    mutable vector<VectorR> layer_inputs;
    mutable vector<VectorR> layer_outputs;

    static float guarded(const float value)
    {
        constexpr float floor_value = 1e-12f;
        if (value > floor_value)  return value;
        if (value < -floor_value) return value;
        return floor_value;
    }

    VectorR scale_forward(const LayerSnapshot& layer, const VectorR& in) const
    {
        VectorR out(in.size());
        for (Index j = 0; j < in.size(); ++j)
        {
            const float x = in(j);
            switch (layer.methods[j])
            {
            case ScalerMethod::None:                  out(j) = x; break;
            case ScalerMethod::MinimumMaximum:        out(j) = (x - layer.minimum(j)) / guarded(layer.maximum(j) - layer.minimum(j)) * (layer.max_range - layer.min_range) + layer.min_range; break;
            case ScalerMethod::MeanStandardDeviation: out(j) = (x - layer.mean(j)) / guarded(layer.deviation(j)); break;
            case ScalerMethod::StandardDeviation:     out(j) = x / guarded(layer.deviation(j)); break;
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
            case ScalerMethod::MinimumMaximum:        d(j) = (layer.max_range - layer.min_range) / guarded(layer.maximum(j) - layer.minimum(j)); break;
            case ScalerMethod::MeanStandardDeviation: d(j) = 1.0f / guarded(layer.deviation(j)); break;
            case ScalerMethod::StandardDeviation:     d(j) = 1.0f / guarded(layer.deviation(j)); break;
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
            case ScalerMethod::MinimumMaximum:        out(j) = (x - layer.min_range) / guarded(layer.max_range - layer.min_range) * (layer.maximum(j) - layer.minimum(j)) + layer.minimum(j); break;
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
            case ScalerMethod::MinimumMaximum:        d(j) = (layer.maximum(j) - layer.minimum(j)) / guarded(layer.max_range - layer.min_range); break;
            case ScalerMethod::MeanStandardDeviation: d(j) = layer.deviation(j); break;
            case ScalerMethod::StandardDeviation:     d(j) = layer.deviation(j); break;
            case ScalerMethod::Logarithm:             d(j) = exp(in(j)); break;
            case ScalerMethod::ImageMinMax:           d(j) = 255.0f; break;
            }
        return d;
    }

    VectorR bound_derivative(const LayerSnapshot& layer, const VectorR& in) const
    {
        if (!layer.bounding_active) return VectorR::Ones(in.size());

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
            else if (layer.kind == Kind::Bound)
                activation = layer.bounding_active ? activation.cwiseMax(layer.minimum).cwiseMin(layer.maximum).eval() : activation;
            else if (layer.kind == Kind::Activate)
                activation = activation_forward_values(layer.activation, activation);
            else
                activation = activation_forward_values(layer.activation,
                                                       (layer.weights.transpose() * activation + layer.bias).eval());

            layer_outputs[i] = activation;
        }

        tape_x = x;
        tape_valid = true;
        return activation;
    }

    VectorR vjp(const VectorR& x, const VectorR& cotangent) const
    {
        ++benchmark_vjp_count;
        if (!tape_valid || tape_x.size() != x.size() || !(tape_x.array() == x.array()).all())
            forward(x);

        VectorR carried = cotangent;
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
        {
            const LayerSnapshot& layer = layers[i];

            if (layer.kind == Kind::Scale)
                carried = (carried.array() * scale_derivative(layer, layer_inputs[i]).array()).matrix();
            else if (layer.kind == Kind::Unscale)
                carried = (carried.array() * unscale_derivative(layer, layer_inputs[i]).array()).matrix();
            else if (layer.kind == Kind::Bound)
                carried = (carried.array() * bound_derivative(layer, layer_inputs[i]).array()).matrix();
            else if (layer.kind == Kind::Activate)
                carried = (carried.array()
                         * activation_derivative_from_output_values(layer.activation, layer_outputs[i]).array()).matrix();
            else
                carried = layer.weights
                        * (carried.array()
                         * activation_derivative_from_output_values(layer.activation, layer_outputs[i]).array()).matrix();
        }
        return carried;
    }
};

struct NetworkJacobian // @todo this should not exist this is a matrix/tensor
{
    unique_ptr<NetworkDifferential> differential;
    bool ready = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
