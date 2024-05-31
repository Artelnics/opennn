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
#include "perceptron_layer.h"
#include "config.h"

namespace opennn
{

struct FlattenLayerForwardPropagation;
struct FlattenLayerBackPropagation;

#ifdef OPENNN_CUDA
struct FlattenLayerForwardPropagationCuda;
struct FlattenLayerBackPropagationCuda;
#endif

/// This class represents a flatten layer.

/// Flatten layers are included in the definition of a neural network.
/// They are used to resize the input data to make it usable for the
/// perceptron layer.

class FlattenLayer : public Layer
{

public:
    void allocate_parameters_device() {};
    // Constructors

    explicit FlattenLayer();

    explicit FlattenLayer(const dimensions&);

    // Get methods

    dimensions get_inputs_dimensions() const;
    Index get_outputs_number() const;
    
    dimensions get_outputs_dimensions() const final;

    Index get_inputs_number() const;
    Index get_inputs_channels_number() const;
    Index get_inputs_rows_number() const;
    Index get_inputs_collums_number() const;
    Index get_neurons_number() const;

    // Set methods

    void set();
    void set(const Index&);
    void set(const dimensions&);
    void set(const tinyxml2::XMLDocument&);

    void set_default();

    void set_name(const string&);

    // Display messages

//    void set_display(const bool&);

    // Check methods

//    bool is_empty() const;

    // Forward propagation

    void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&, 
                           LayerForwardPropagation*, 
                           const bool&) final;

    // Back-propagation

    void back_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                        const Tensor<pair<type*, dimensions>, 1>&,
                        LayerForwardPropagation*,
                        LayerBackPropagation*) const final;

    // Serialization methods

    void from_XML(const tinyxml2::XMLDocument&) final;

    void write_XML(tinyxml2::XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/flatten_layer_cuda.h"
    #endif

protected:

    dimensions inputs_dimensions;

    /// Display warning messages to screen.

    bool display = true;
};


struct FlattenLayerForwardPropagation : LayerForwardPropagation
{
   // Default constructor

   explicit FlattenLayerForwardPropagation() : LayerForwardPropagation()
   {
   }

   // Constructor

   explicit FlattenLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
       : LayerForwardPropagation()
   {
       set(new_batch_samples_number, new_layer);
   }
   
   
   pair<type*, dimensions> get_outputs_pair() const final;


   void set(const Index& new_batch_samples_number, Layer* new_layer) final;


   void print() const
   {
       cout << "Outputs:" << endl;

       cout << outputs << endl;
   }


   Tensor<type, 2> outputs;
};


struct FlattenLayerBackPropagation : LayerBackPropagation
{

    // Default constructor

    explicit FlattenLayerBackPropagation() : LayerBackPropagation()
    {
    }


    explicit FlattenLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    virtual ~FlattenLayerBackPropagation()
    {
    }

    void set(const Index& new_batch_samples_number, Layer* new_layer) final;


    void print() const
    {
    }

    Tensor<type, 4> input_derivatives;
};


#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/flatten_layer_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/flatten_layer_back_propagation_cuda.h"
#endif


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
