//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V G G 1 6   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "vgg16.h"

namespace opennn
{
    VGG16::VGG16(const dimensions& input_dimensions, const dimensions& target_dimensions)
        : NeuralNetwork()
    {
        set_model_type(NeuralNetwork::ModelType::ImageClassification);

        // Scaling 4D
        add_layer(make_unique<Scaling4d>(input_dimensions));

        // --- Conv 3×3, 64 kernels, ReLU x2 -> Pooling 2×2 stride 2 ---
        {
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(), 
                dimensions{ 3, 3, input_dimensions[2], 64 },
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 },
                Convolutional::Convolution::Valid,
                "conv_1"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 64, 64 },
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 },
                Convolutional::Convolution::Valid,
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
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 },
                Convolutional::Convolution::Valid,
                "conv_3"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 128, 128 },
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 },
                Convolutional::Convolution::Valid,
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
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 }, 
                Convolutional::Convolution::Valid, 
                "conv_5"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 256, 256 },
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 }, 
                Convolutional::Convolution::Valid, 
                "conv_6"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 256, 256 },
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 }, 
                Convolutional::Convolution::Valid, 
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
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 },
                Convolutional::Convolution::Valid,
                "conv_8"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 512, 512 },
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 }, 
                Convolutional::Convolution::Valid, 
                "conv_9"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 512, 512 },
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 }, 
                Convolutional::Convolution::Valid, 
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
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 },
                Convolutional::Convolution::Valid, 
                "conv_11"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 512, 512 },
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 }, 
                Convolutional::Convolution::Valid, 
                "conv_12"));
            add_layer(make_unique<Convolutional>(
                get_output_dimensions(),
                dimensions{ 3, 3, 512, 512 },
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 }, 
                Convolutional::Convolution::Valid,
                "conv_13"));
            add_layer(make_unique<Pooling>(
                get_output_dimensions(),
                dimensions{ 2, 2 }, 
                dimensions{ 2, 2 },
                dimensions{ 0, 0 },
                Pooling::PoolingMethod::MaxPooling, "pool5"));
        }

        // --- Flatten ---
        add_layer(make_unique<Flatten>(get_output_dimensions()));

        // --- Dense layers (FC) ---
        {
            // FC1: 4096 neurons, ReLU
            add_layer(make_unique<Dense2d>(
                get_output_dimensions(),
                dimensions{ 4096 },
                Dense2d::Activation::RectifiedLinear,
                "fc1"));
            // FC2: 4096 neurons, ReLU
            add_layer(make_unique<Dense2d>(
                get_output_dimensions(),
                dimensions{ 4096 },
                Dense2d::Activation::RectifiedLinear,
                "fc2"));
            // FC3 : Softmax
            add_layer(make_unique<Dense2d>(
                get_output_dimensions(),
                target_dimensions,
                Dense2d::Activation::Softmax,
                "softmax_output"));
        }
    }

    /*
    VGG16::VGG16(const filesystem::path& file_name)
        : NeuralNetwork(file_name)
    {

    }
    */

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