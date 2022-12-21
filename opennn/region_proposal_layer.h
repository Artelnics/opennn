//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E G I O N   P R O P O S A L   L A Y E R   C L A S S   H E A D E R
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
//#include "opennn_images.h"

namespace opennn
{
class RegionProposalLayer : public Layer

{

public:
    // Constructors

    explicit RegionProposalLayer();

    // Region proposal layer outputs

    const Tensor<type, 4> get_input_regions();

    void calculate_regions(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&);

//    void calculate_outputs(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) final;

    void forward_propagate(type*, const Tensor<Index, 1>&, LayerForwardPropagation*, bool&);

protected:

   bool display = true;

   const Index regions_number = 2000;
   const Index region_rows = 22;
   const Index region_columns = 22;
   const Index channels_number = 3;
};


struct RegionProposalLayerForwardPropagation : LayerForwardPropagation
{
    // Default constructor

    explicit RegionProposalLayerForwardPropagation()
        : LayerForwardPropagation()
    {
    }

    // Constructor

    explicit RegionProposalLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : RegionProposalLayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }

    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        layer_pointer = new_layer_pointer;      

        const Index regions_number = 2000;
        const Index region_rows = 6;
        const Index region_columns = 6;
        const Index channels_number = 3;

        outputs_regions.resize(regions_number,4);

        outputs_dimensions.resize(4);
        outputs_dimensions(0) = region_rows;
        outputs_dimensions(1) = region_columns;
        outputs_dimensions(2) = channels_number;
        outputs_dimensions(3) = regions_number;

        outputs_data = outputs.data();


//        outputs_data = (float*) malloc(outputs_dimensions.prod() * sizeof(float));
    }

    void print() const
    {

    }

    Tensor<type, 2> outputs;

    Tensor<type, 2> outputs_regions;
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
