//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E T W O R K   D I F F E R E N T I A L   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "network_differential.h"
#include "statistics.h"
#include "neural_network.h"
#include "dense_layer.h"
#include "activation_layer.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"
#include "bounding_layer.h"

namespace opennn
{

void NetworkDifferential::build(const NeuralNetwork& network)
{
    layers.clear();

    for (const unique_ptr<Layer>& layer : network.get_layers())
    {
        LayerSnapshot snapshot;
        const LayerType type = layer->get_type();

        if (type == LayerType::Scaling || type == LayerType::Unscaling)
        {
            snapshot.kind = (type == LayerType::Scaling) ? Kind::Scale : Kind::Unscale;

            const vector<Descriptives>* descriptives = nullptr;
            const vector<ScalerMethod>* scalers = nullptr;

            if (type == LayerType::Scaling)
            {
                const Scaling* scaling = static_cast<const Scaling*>(layer.get());
                descriptives = &scaling->get_descriptives();
                scalers = &scaling->get_scalers();
                snapshot.min_range = scaling->get_min_range();
                snapshot.max_range = scaling->get_max_range();
            }
            else
            {
                const Unscaling* unscaling = static_cast<const Unscaling*>(layer.get());
                descriptives = &unscaling->get_descriptives();
                scalers = &unscaling->get_scalers();
                snapshot.min_range = unscaling->get_min_range();
                snapshot.max_range = unscaling->get_max_range();
            }

            const Index features = static_cast<Index>(descriptives->size());
            snapshot.methods = *scalers;
            snapshot.minimum.resize(features);
            snapshot.maximum.resize(features);
            snapshot.mean.resize(features);
            snapshot.deviation.resize(features);

            for (Index j = 0; j < features; ++j)
            {
                snapshot.minimum(j)   = static_cast<float>((*descriptives)[j].minimum);
                snapshot.maximum(j)   = static_cast<float>((*descriptives)[j].maximum);
                snapshot.mean(j)      = static_cast<float>((*descriptives)[j].mean);
                snapshot.deviation(j) = static_cast<float>((*descriptives)[j].standard_deviation);
            }
        }
        else if (type == LayerType::Dense)
        {
            const Dense* dense = static_cast<const Dense*>(layer.get());

            throw_if(dense->get_batch_normalization(),
                     "NetworkDifferential: batch normalization is not supported");

            const vector<TensorView>& views = dense->get_parameter_views();
            throw_if(views.size() < 2, "NetworkDifferential: unexpected Dense parameter layout");

            snapshot.kind = Kind::Dense;
            snapshot.bias = views[0].as_vector();
            snapshot.weights = views[1].as_matrix();
            snapshot.activation = dense->get_activation_function();

            throw_if(snapshot.activation == ActivationFunction::Softmax,
                     "NetworkDifferential: softmax activation is not supported");
        }
        else if (type == LayerType::Activation)
        {
            const Activation* activation_layer = static_cast<const Activation*>(layer.get());
            snapshot.kind = Kind::Activate;
            snapshot.activation = activation_layer->get_output_activation();

            throw_if(snapshot.activation == ActivationFunction::Softmax,
                     "NetworkDifferential: softmax activation is not supported");
        }
        else if (type == LayerType::Bounding)
        {
            const Bounding* bounding = static_cast<const Bounding*>(layer.get());
            snapshot.kind = Kind::Bound;
            snapshot.minimum = bounding->get_lower_bounds();
            snapshot.maximum = bounding->get_upper_bounds();
            snapshot.bounding_active =
                (bounding->get_bounding_method() == Bounding::BoundingMethod::Bounding);
        }
        else
            throw runtime_error(format("Unsupported layer type '{}' for analytic Jacobian", layer_type_to_string(type)));

        layers.push_back(move(snapshot));
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
