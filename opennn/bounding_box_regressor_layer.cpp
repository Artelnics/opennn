//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   B O X   R E G R E S S O R   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "bounding_box_regressor_layer.h"
#include "opennn_images.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.

BoundingBoxRegressorLayer::BoundingBoxRegressorLayer() : Layer()
{
}


void BoundingBoxRegressorLayer::forward_propagate(type* inputs_data,
                          const Tensor<Index,1>& inputs_dimensions,
                          LayerForwardPropagation* forward_propagation)
{

}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
