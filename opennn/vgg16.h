//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V G G 1 6   N E U R A L   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef VGG16_H
#define VGG16_H

#include "neural_network.h"
#include "scaling_layer_4d.h"
#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "flatten_layer.h"
#include "perceptron_layer.h"

namespace opennn
{

    class VGG16 : public NeuralNetwork
    {
    public:

        VGG16(const dimensions& input_dimensions, const dimensions& target_dimensions);

        void set(const dimensions& input_dimensions, const dimensions& target_dimensions);

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
