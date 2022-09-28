//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   N O R M A L I Z A T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "batch_normalization_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.

BatchNormalizationLayer::BatchNormalizationLayer(const Index& new_inputs_number) : Layer()
{
    set(new_inputs_number);
    layer_type = Type::BatchNormalization;
}


void BatchNormalizationLayer::set(const Index& new_inputs_number)
{
    normalization_weights.resize(2, new_inputs_number);

    set_parameters_random();

    set_default();
}


void BatchNormalizationLayer::set_parameters_random()
{
    const type minimum = type(-0.2);
    const type maximum = type(0.2);

    for(Index i = 0; i < normalization_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        normalization_weights(i) = minimum + (maximum - minimum)*random;
    }
}


void BatchNormalizationLayer::set_default()
{
    layer_name = "batch_normalization_layer";

    display = true;

    layer_type = Type::BatchNormalization;
}


void BatchNormalizationLayer::calculate_combinations(const Tensor<type, 2>& inputs,
                                                     const Tensor<type, 2>& weights,
                                                     Tensor<type, 2>& outputs)
{


}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
