//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "flatten_layer.h"

namespace opennn
{


/// Default constructor.
/// It creates a flatten layer.

FlattenLayer::FlattenLayer() : Layer()
{
    set();
}


/// Destructor.

FlattenLayer::~FlattenLayer()
{
}


/// Returns the number of images by batch.

Index FlattenLayer::get_inputs_batch() const
{
    return input_variables_dimensions[0];
}


/// Returns the number of channels of the image.

Index FlattenLayer::get_inputs_channels_number() const
{
    return input_variables_dimensions[1];
}


/// Returns the number of columns of the image.

Index FlattenLayer::get_inputs_width() const
{
    return input_variables_dimensions[2];
}


/// Returns the number of rows of the image.

Index FlattenLayer::get_inputs_height() const
{
    return input_variables_dimensions[3];
}


/// Returns the number of rows of the output which matches with the batch size.

Index FlattenLayer::get_outputs_rows_number() const
{
    return (get_inputs_batch());
}


/// Returns the number of columns which mathes with the number of pixels of an image.

Index FlattenLayer::get_outputs_columns_number() const
{
    return (get_inputs_channels_number()*get_inputs_width()*get_inputs_height());
}


/// Returns a vector containing the batch and number of pixels of an image in result of resizing the
/// input tensor taken by the flatten layer.

Tensor<Index, 1> FlattenLayer::get_outputs_dimensions() const
{
    Tensor<Index, 1> outputs_dimensions(2);

    outputs_dimensions(0) = get_outputs_rows_number(); // Batch
    outputs_dimensions(1) = get_outputs_columns_number() ; // Number of pixels of the image

    return outputs_dimensions;
}


/// Sets and initializes the layer's parameters in accordance with the dimensions taken as input.
/// @todo change to memcpy approach
/// @param new_inputs_dimensions A vector containing the desired inputs' dimensions (number of images (batch), number of channels, width, height).

void FlattenLayer::set(const Tensor<Index, 1>& new_inputs_dimensions)
{

#ifdef OPENNN_DEBUG

    const Index inputs_dimensions_number = new_inputs_dimensions.size();

    if(inputs_dimensions_number != 4)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: FlattenLayer class.\n"
               << "FlattenLayer(const Tensor<Index, 1>&) constructor.\n"
               << "Number of inputs dimensions (" << inputs_dimensions_number << ") must be 4 (number of images, channels, width, height).\n";

        throw invalid_argument(buffer.str());
    }

#endif

    input_variables_dimensions = new_inputs_dimensions;
}


/// Obtain the connection between the convolutional and the conventional part
/// of a neural network. That is a column vector in which each of its elements
/// links to a neuron in the perceptron layer.
/// @param inputs

Tensor<type, 2> FlattenLayer::calculate_outputs_2d(const Tensor<type, 4>& inputs)
{
    const Index batch = inputs.dimension(0);
    const Index channels = inputs.dimension(1);
    const Index width = inputs.dimension(2);
    const Index heights = inputs.dimension(3);

    inputs(batch, channels, width, heights);
    Eigen::array<Index, 2> new_dims{{batch, channels*width*heights}};
    Tensor<type, 2> result = inputs.reshape(new_dims);

    return result;
}

void FlattenLayer::forward_propagate(const Tensor<type, 2> &inputs, LayerForwardPropagation* forward_propagation)
{

    FlattenLayerForwardPropagation* flatten_layer_forward_propagation
            = static_cast<FlattenLayerForwardPropagation*>(forward_propagation);

}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
