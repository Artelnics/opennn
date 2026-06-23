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

struct NetworkDifferential
{
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

    static VectorR activation_forward(const ActivationFunction function, const VectorR& z)
    {
        switch (function)
        {
        case ActivationFunction::Identity: return z;
        case ActivationFunction::Sigmoid:  return (1.0f + (-z.array()).exp()).inverse();
        case ActivationFunction::Tanh:     return z.array().tanh();
        case ActivationFunction::ReLU:     return z.array().max(0.0f);
        case ActivationFunction::LeakyReLU: return (z.array() >= 0.0f).select(z.array(), z.array() * LEAKY_RELU_SLOPE);
        case ActivationFunction::Softmax:
        default: throw runtime_error("NetworkDifferential: unsupported activation");
        }
    }

    static VectorR activation_derivative(const ActivationFunction function, const VectorR& a)
    {
        switch (function)
        {
        case ActivationFunction::Identity: return VectorR::Ones(a.size());
        case ActivationFunction::Sigmoid:  return a.array() * (1.0f - a.array());
        case ActivationFunction::Tanh:     return 1.0f - a.array().square();
        case ActivationFunction::ReLU:     return (a.array() > 0.0f).cast<float>();
        case ActivationFunction::LeakyReLU: return a.unaryExpr([](float y) { return y >= 0.0f ? 1.0f : LEAKY_RELU_SLOPE; });
        case ActivationFunction::Softmax:
        default: throw runtime_error("NetworkDifferential: unsupported activation");
        }
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

    void build(const NeuralNetwork& network);

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
                activation = activation_forward(layer.activation, activation);
            else
                activation = activation_forward(layer.activation, (layer.weights.transpose() * activation + layer.bias).eval());

            layer_outputs[i] = activation;
        }

        tape_x = x;
        tape_valid = true;
        return activation;
    }

    VectorR vjp(const VectorR& x, const VectorR& cotangent) const
    {
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
                carried = (carried.array() * activation_derivative(layer.activation, layer_outputs[i]).array()).matrix();
            else
            {
                const VectorR through_activation = (carried.array() * activation_derivative(layer.activation, layer_outputs[i]).array()).matrix();
                carried = layer.weights * through_activation;
            }
        }
        return carried;
    }
};


// Built-once analytic Jacobian plus the flag tracking whether it has been built/decided for the
// current network; the two always change together. A null differential means the finite-difference
// fallback is in effect.
struct NetworkJacobian
{
    unique_ptr<NetworkDifferential> differential;
    bool ready = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
