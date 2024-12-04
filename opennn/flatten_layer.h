//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef FLATTENLAYER_H
#define FLATTENLAYER_H

#include "layer.h"

namespace opennn
{

struct FlattenLayerForwardPropagation;
struct FlattenLayerBackPropagation;

#ifdef OPENNN_CUDA
struct FlattenLayerForwardPropagationCuda;
struct FlattenLayerBackPropagationCuda;
#endif

class FlattenLayer : public Layer
{

public:

    FlattenLayer(const dimensions& = {0,0,0});

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    Index get_input_height() const;
    Index get_input_width() const;
    Index get_input_channels() const;

    void set(const dimensions & = {0,0,0});

    // Forward propagation

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    // Back-propagation

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    // Serialization

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

    void print() const override;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/flatten_layer_cuda.h"
    #endif

private:

    dimensions input_dimensions;
};


struct FlattenLayerForwardPropagation : LayerForwardPropagation
{
   FlattenLayerForwardPropagation(const Index& = 0, Layer* = nullptr);
      
   pair<type*, dimensions> get_outputs_pair() const override;

   void set(const Index& = 0, Layer* = nullptr) override;

   void print() const override;

   Tensor<type, 2> outputs;
};


struct FlattenLayerBackPropagation : LayerBackPropagation
{
    FlattenLayerBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

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
