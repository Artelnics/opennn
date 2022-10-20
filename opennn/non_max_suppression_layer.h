//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O N   M A X   S U P R E S S I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef REGIONPROPOSALLAYER_H
#define REGIONPROPOSALLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "opennn_strings.h"

namespace opennn
{
class NonMaxSupressionLayer : public Layer

{

public:
   // Constructors

   explicit NonMaxSupressionLayer();

   // Perceptron layer outputs

   void forward_propagate(type*, const Tensor<Index, 1>&,
                          LayerForwardPropagation*);


protected:

   bool display = true;
};


struct NonMaxSupressionLayerForwardPropagation : LayerForwardPropagation
{
    // Default constructor

    explicit NonMaxSupressionLayerForwardPropagation()
        : LayerForwardPropagation()
    {
    }

    // Constructor

    explicit NonMaxSupressionLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : NonMaxSupressionLayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }

    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        layer_pointer = new_layer_pointer;

    }

    void print() const
    {

    }
};


}
#endif

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
