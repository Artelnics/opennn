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
//#include "opennn_images.h"

namespace opennn
{
class NonMaxSuppressionLayer : public Layer

{

public:
   // Constructors

   explicit NonMaxSuppressionLayer();

   void calculate_regions(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&);

   void forward_propagate(Tensor<type*, 1>,
                          const Tensor<Tensor<Index, 1>, 1>&,
                          LayerForwardPropagation*,
                          const bool&);


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

    explicit NonMaxSuppressionLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : NonMaxSuppressionLayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }

    void set(const Index& new_batch_samples_number, Layer* new_layer)
    {
    layer = new_layer;

    /*NonMaxSuppressionLayerForwardPropagation* perceptron_layer_3d = static_cast<NonMaxSuppressionLayerForwardPropagation*>(layer);
*/
    batch_samples_number = new_batch_samples_number;

//    const Index neurons_number = perceptron_layer_3d->get_neurons_number();

//    const Index inputs_number = perceptron_layer_3d->get_inputs_number();

   /* outputs.resize(batch_samples_number, grid_height, grid_width, bounding_box_predictions_number);

    outputs_data = outputs.data();

//    activations_derivatives.resize(batch_samples_number, inputs_number, neurons_number);

        layer = new_layer;

        outputs.resize(2);

        // Bounding boxes

        outputs(0).set_data(nullptr);

        // Scores

        outputs(1).set_data(nullptr);*/
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

    //Tensor<pair<type*, dimensions>, 1> outputs;
};

}
#endif

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
