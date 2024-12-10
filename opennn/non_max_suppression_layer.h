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

   explicit NonMaxSuppressionLayer(const dimensions&,
                                   const Index&,
                                   const string = "non_max_suppression_layer");

   void set(const dimensions&,
            const Index&,
            const string = "non_max_suppression_layer");

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>&,
                          const bool&) final;

   void calculate_boxes(const Tensor<type, 4>&, Tensor<type, 3>&, const Index&, Tensor<type, 0>&);

   void print() const;


   dimensions get_input_dimensions() const;
   dimensions get_output_dimensions() const;


protected:

   bool display = true;

   const type overlap_threshold = 0.4;

   dimensions input_dimensions;
   Index boxes_per_cell;
   Index classes_number;                    // For VOC2007 dataset there are 20 classes
   Index grid_size;
   const Index output_box_info = 5;         // x_center, y_center, width, height and object_confidence
   const Index final_box_info = 6;          // x_center, y_center, width, height, object_confidence * class_probability and class

};


struct NonMaxSuppressionLayerForwardPropagation : LayerForwardPropagation
{
    // Default constructor

    // explicit NonMaxSuppressionLayerForwardPropagation()
    //     : LayerForwardPropagation()
    // {
    // }

    // Constructor

    explicit NonMaxSuppressionLayerForwardPropagation(const Index& = 0, Layer* = nullptr);


    void set(const Index& = 0, Layer* = nullptr) final;

    pair<type *, dimensions> get_outputs_pair() const final;

    void print() const
    {
        cout << "Non max suppression layer forward propagation structure" << endl;

        cout << "Outputs:" << endl;

        cout<< outputs <<endl;

        cout << "Outputs dimensions:" << endl;

        cout << outputs.dimensions() << endl;
    }

    Tensor<type, 3> outputs;
    Tensor<type, 0> maximum_box_number;
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
