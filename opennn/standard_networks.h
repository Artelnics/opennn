//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   STANDARD   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef STANDARDNETWORKS_H
#define STANDARDNETWORKS_H

#include "scaling_layer_2d.h"
#include "scaling_layer_4d.h"
#include "unscaling_layer.h"
#include "perceptron_layer.h"
#include "bounding_layer.h"
#include "recurrent_layer.h"
#include "embedding_layer.h"
#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "pooling_layer_3d.h"
#include "flatten_layer.h"
#include "flatten_layer_3d.h"
#include "neural_network.h"

namespace opennn
{

class ApproximationNetwork : public NeuralNetwork
{
public:

    ApproximationNetwork(const dimensions& input_dimensions,
                         const dimensions& complexity_dimensions,
                         const dimensions& output_dimensions) : NeuralNetwork()
    {
        const Index complexity_size = complexity_dimensions.size();

        add_layer(make_unique<Scaling2d>(input_dimensions));

        for (Index i = 0; i < complexity_size; i++)
            add_layer(make_unique<Dense2d>(get_output_dimensions(),
                                           dimensions{ complexity_dimensions[i] },
                                           "RectifiedLinear",
                                           false,
                                           "dense2d_layer_" + to_string(i + 1)));

        add_layer(make_unique<Dense2d>(get_output_dimensions(),
                                       output_dimensions,
                                       "Linear",
                                       false,
                                       "approximation_layer"));

        add_layer(make_unique<Unscaling>(output_dimensions));

        add_layer(make_unique<Bounding>(output_dimensions));

        const Index inputs_number = get_inputs_number();
        input_names.resize(inputs_number);

        const Index outputs_number = get_outputs_number();
        output_names.resize(outputs_number);
    }
};


class ClassificationNetwork : public NeuralNetwork
{
public:

    ClassificationNetwork(const dimensions& input_dimensions,
                          const dimensions& complexity_dimensions,
                          const dimensions& output_dimensions) : NeuralNetwork()
    {
        const Index complexity_size = complexity_dimensions.size();

        add_layer(make_unique<Scaling2d>(input_dimensions));

        for (Index i = 0; i < complexity_size; i++)
            add_layer(make_unique<Dense2d>(get_output_dimensions(),
                                           dimensions{complexity_dimensions[i]},
                                           "HyperbolicTangent",
                                           false,
                                           "dense2d_layer_" + to_string(i + 1)));

        add_layer(make_unique<Dense2d>(get_output_dimensions(),
                                       output_dimensions,
                                       "Logistic",
                                       false,
                                       "classification_layer"));
    }
};


class ForecastingNetwork : public NeuralNetwork
{
public:

    ForecastingNetwork(const dimensions& input_dimensions,
                       const dimensions& complexity_dimensions,
                       const dimensions& output_dimensions) : NeuralNetwork()
    {
        add_layer(make_unique<Scaling2d>(input_dimensions));

        add_layer(make_unique<Recurrent>(get_output_dimensions(),
                                         output_dimensions));

        add_layer(make_unique<Unscaling>(output_dimensions));

        add_layer(make_unique<Bounding>(output_dimensions));

    }
};


class AutoAssociationNetwork : public NeuralNetwork
{
public:

    AutoAssociationNetwork(const dimensions& input_dimensions,
                           const dimensions& complexity_dimensions,
                           const dimensions& output_dimensions) : NeuralNetwork()
    {
        add_layer(make_unique<Scaling2d>(input_dimensions));

        const Index mapping_neurons_number = 10;
        const Index bottle_neck_neurons_number = complexity_dimensions[0];

        add_layer(make_unique<Dense2d>(input_dimensions,
                                       dimensions{mapping_neurons_number},
                                       "HyperbolicTangent",
                                       false,
                                       "mapping_layer"));

        add_layer(make_unique<Dense2d>(dimensions{ mapping_neurons_number },
                                       dimensions{ bottle_neck_neurons_number },
                                       "Linear",
                                       false,
                                       "bottleneck_layer"));

        add_layer(make_unique<Dense2d>(dimensions{ bottle_neck_neurons_number },
                                       dimensions{ mapping_neurons_number },
                                       "HyperbolicTangent",
                                       false,
                                       "demapping_layer"));

        add_layer(make_unique<Dense2d>(dimensions{ mapping_neurons_number },
                                       dimensions{ output_dimensions },
                                       "Linear",
                                       false,
                                       "output_layer"));

        add_layer(make_unique<Unscaling>(output_dimensions));
    }
};


class ImageClassificationNetwork : public NeuralNetwork
{
public:

    ImageClassificationNetwork(const dimensions& input_dimensions,
                               const dimensions& complexity_dimensions,
                               const dimensions& output_dimensions) : NeuralNetwork()
    {
        if (input_dimensions.size() != 3)
            throw runtime_error("Input dimensions size is not 3.");

        add_layer(make_unique<Scaling4d>(input_dimensions));
        
        const Index complexity_size = complexity_dimensions.size();

        for (Index i = 0; i < complexity_size; i++)
        {
            const dimensions kernel_dimensions = { 3, 3, get_output_dimensions()[2], complexity_dimensions[i] };
            const dimensions stride_dimensions = { 1, 1 };

            add_layer(make_unique<Convolutional>(get_output_dimensions(),
                                                 kernel_dimensions,
                                                 "RectifiedLinear",
                                                 stride_dimensions,
                                                 Convolutional::Convolution::Same,
                                                 true, // Batch normalization
                                                 "convolutional_layer_" + to_string(i + 1)));
            
            const dimensions pool_dimensions = { 2, 2 };
            const dimensions pooling_stride_dimensions = { 2, 2 };
            const dimensions padding_dimensions = { 0, 0 };

            add_layer(make_unique<Pooling>(get_output_dimensions(),
                                           pool_dimensions,
                                           pooling_stride_dimensions,
                                           padding_dimensions,
                                           Pooling::PoolingMethod::MaxPooling,
                                           "pooling_layer_" + to_string(i + 1)));
        }
        
        add_layer(make_unique<Flatten>(get_output_dimensions()));

        add_layer(make_unique<Dense2d>(get_output_dimensions(),
                                       output_dimensions,
                                       "Softmax",
                                       false,
                                       "dense_2d_layer"));
    }
};


class TextClassificationNetwork : public NeuralNetwork
{
public:

    TextClassificationNetwork(const dimensions& input_dimensions,
                              const dimensions& complexity_dimensions,
                              const dimensions& output_dimensions) : NeuralNetwork()
    {
        layers.clear();

        const Index vocabulary_size = input_dimensions[0];
        const Index sequence_length = input_dimensions[1];
        const Index embedding_dimension = input_dimensions[2];

        add_layer(make_unique<Embedding>(dimensions({vocabulary_size, sequence_length}),
                                         embedding_dimension,
                                         "embedding_layer"
                                         ));

        // add_layer(make_unique<Pooling3d>(
        //     get_output_dimensions()
        //     ));

        add_layer(make_unique<Flatten3d>(
            get_output_dimensions()
            ));

        add_layer(make_unique<Dense2d>(
            get_output_dimensions(),
            output_dimensions,
            "Logistic",
            "classification_layer"));
    }
};

} // namespace opennn

#endif // VGG16_H

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software

// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
