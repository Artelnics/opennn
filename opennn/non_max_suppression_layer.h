//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O N   M A X   S U P R E S S I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NONMAXSUPRESSIONLAYER_H
#define NONMAXSUPRESSIONLAYER_H

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
#include "layer_forward_propagation.h"

namespace opennn
{
class NonMaxSuppressionLayer : public Layer

{

public:
   // Constructors

   explicit NonMaxSuppressionLayer();

   void calculate_regions(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&);

   void forward_propagate(const pair<type*, dimensions>&, LayerForwardPropagation*, const bool&) final;

protected:

   bool display = true;
};


struct NonMaxSuppressionLayerForwardPropagation : LayerForwardPropagation
{
    // Default constructor

    explicit NonMaxSuppressionLayerForwardPropagation()
        : LayerForwardPropagation()
    {
    }

    // Constructor

    explicit NonMaxSuppressionLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : NonMaxSuppressionLayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }
    
    
    pair<type*, dimensions> get_outputs_pair() const final
    {
        return pair<type*, dimensions>();
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer) final
    {
        layer_pointer = new_layer_pointer;

        outputs.resize(2);

        // Bounding boxes

        //outputs(0).set_data(nullptr);

        // Scores

        //outputs(1).set_data(nullptr);
    }

    void print() const
    {
        cout << "Non max suppression layer forward propagation structure" << endl;

        cout << "Outputs:" << endl;
//        cout << TensorMap<Tensor<type,4>>(outputs_data,
//                                          outputs_dimensions[0],
//                                          outputs_dimensions(1),
//                                          outputs_dimensions(2),
//                                          outputs_dimensions(3)) << endl;

        cout << "Outputs dimensions:" << endl;

//        cout << outputs_dimensions << endl;
    }

    Tensor<Tensor<type, 2>, 1> outputs;
};

}
#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
