//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef FLATTENLAYER_H
#define FLATTENLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "layer.h"
#include "opennn_strings.h"
#include "config.h"

namespace opennn
{

struct FlattenLayerForwardPropagation;


/// This class represents a layer of scaling neurons.

///
/// Scaling layers are included in the definition of a neural network. 
/// They are used to normalize variables so they are in an appropriate range for computer processing.  

class FlattenLayer : public Layer
{

public:

    // Constructors

    explicit FlattenLayer();

    explicit FlattenLayer(const Index&);

    // Destructors

    virtual ~FlattenLayer();

    // Get methods
    Index get_inputs_batch() const;
    Index get_inputs_channels_number() const;
    Index get_inputs_width() const;
    Index get_inputs_height() const;


    Tensor<Index, 1> get_outputs_dimensions() const;

    Index get_outputs_rows_number() const;

    Index get_outputs_columns_number() const;

    // Set methods

    void set();
    void set(const Index&);
    void set(const Tensor<Index, 1>&);
    void set(const tinyxml2::XMLDocument&);

    void set_inputs_number(const Index&);

    void set_default();

    // Display messages

    void set_display(const bool&);

    // Check methods

    bool is_empty() const;

    void check_range(const Tensor<type, 1>&) const;

    // Outputs

    Tensor<type, 2> calculate_outputs_2d(const Tensor<type, 4>&);

    void forward_propagate(const Tensor<type, 2>&, LayerForwardPropagation*);


protected:

    Tensor<Index, 1> input_variables_dimensions;

    /// Display warning messages to screen.

    bool display = true;

    struct FlattenLayerForwardPropagation : LayerForwardPropagation
    {
       // Default constructor

       explicit FlattenLayerForwardPropagation()
           : LayerForwardPropagation()
       {
       }

       // Constructor

       explicit FlattenLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
           : LayerForwardPropagation()
       {
           set(new_batch_samples_number, new_layer_pointer);
       }


       void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
       {
           layer_pointer = new_layer_pointer;

           const Index outputs_rows_number = static_cast<FlattenLayer*>(layer_pointer)->get_outputs_rows_number();
           const Index outputs_columns_number = static_cast<FlattenLayer*>(layer_pointer)->get_outputs_columns_number();

           batch_samples_number = new_batch_samples_number;
       }


       void print() const
       {

       }

    };

};

}

#endif


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

