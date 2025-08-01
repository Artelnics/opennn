//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V G G 1 6   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "scaling_layer_4d.h"
#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "perceptron_layer.h"
#include "flatten_layer.h"

#include "vgg16.h"

namespace opennn
{
    VGG16::VGG16(const dimensions& new_input_dimensions, const dimensions& new_target_dimensions)
        : NeuralNetwork()
    {
        set(new_input_dimensions, new_target_dimensions);
    }


    void VGG16::set(const dimensions& new_input_dimensions, const dimensions& new_target_dimensions)
    {
        reference_all_layers();

        // Scaling 4D
        add_layer(make_unique<Scaling4d>(new_input_dimensions));

        // --- Conv 3×3, 64 kernels, ReLU x2 -> Pooling 2×2 stride 2 ---
        {
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, new_input_dimensions[2], 64 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_1"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 64, 64 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_2"));
            add_layer(make_unique<Pooling>(
                get_output_dimensions(),
                dimensions{ 2, 2 },
                dimensions{ 2, 2 },
                dimensions{ 0, 0 },
                Pooling::PoolingMethod::MaxPooling,
                "pool1"));
        }

        // --- Conv 3×3, 128 kernels, ReLU x2 -> Pooling 2×2 stride 2 ---
        {
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 64, 128 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_3"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 128, 128 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_4"));
            add_layer(make_unique<Pooling>(
                get_output_dimensions(),
                dimensions{ 2, 2 },
                dimensions{ 2, 2 },
                dimensions{ 0, 0 },
                Pooling::PoolingMethod::MaxPooling,
                "pool2"));
        }

        // --- Conv 3×3, 256 kernels, ReLU x3 -> Pooling 2×2 stride 2 ---
        {
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 128, 256 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_5"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 256, 256 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_6"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 256, 256 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_7"));
            add_layer(make_unique<Pooling>(
                get_output_dimensions(),
                dimensions{ 2, 2 },
                dimensions{ 2, 2 },
                dimensions{ 0, 0 },
                Pooling::PoolingMethod::MaxPooling, "pool3"));
        }

        // --- Conv 3×3, 512 kernels, ReLU x3 -> Pooling 2×2 stride 2 ---
        {
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 256, 512 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_8"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 512, 512 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_9"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 512, 512 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_10"));
            add_layer(make_unique<Pooling>(
                get_output_dimensions(),
                dimensions{ 2, 2 },
                dimensions{ 2, 2 },
                dimensions{ 0, 0 },
                Pooling::PoolingMethod::MaxPooling, "pool4"));
        }

        // --- Conv 3×3, 512 kernels, ReLU x3 -> Pooling 2×2 stride 2 ---
        {
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 512, 512 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_11"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 512, 512 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_12"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 512, 512 },
                "RectifiedLinear",
                dimensions{ 1, 1 },
                Convolutional::Convolution::Same,
                "conv_13"));
            add_layer(make_unique<Pooling>(
                get_output_dimensions(),
                dimensions{ 2, 2 },
                dimensions{ 2, 2 },
                dimensions{ 0, 0 },
                Pooling::PoolingMethod::MaxPooling, "pool5"));
        }

        const dimensions pre_pool_dims = get_output_dimensions();

        add_layer(make_unique<Pooling>(
            pre_pool_dims,
            dimensions{ pre_pool_dims[0], pre_pool_dims[1] },
            dimensions{ 1, 1 },
            dimensions{ 0, 0 },
            Pooling::PoolingMethod::AveragePooling,
            "global_avg_pool"));

        // Flatten
        add_layer(make_unique<Flatten<2>>(get_output_dimensions()));

        //Classifier
        add_layer(make_unique<Dense2d>(get_output_dimensions(),
            new_target_dimensions,
            "Softmax",
            false,
            "dense_classifier"));
    }


    VGG16::VGG16(const filesystem::path& file_name)
        : NeuralNetwork(file_name)
    {

    }


} // namespace opennn


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
