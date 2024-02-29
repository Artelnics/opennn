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

    explicit RegionProposalLayer(const Tensor<Index, 1>&);

    Tensor<Index, 1> get_inputs_dimensions() const final;
    Tensor<Index, 1> get_outputs_dimensions() const final;

    Index get_regions_number() const;
    Index get_region_rows() const;
    Index get_region_columns() const;
    Index get_channels_number() const;

    void set_regions_number(const Index&);
    void set_region_rows(const Index&);
    void set_region_columns(const Index&);
    void set_channels_number(const Index&);

    // Region proposal layer outputs

    void calculate_regions(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&);

    void forward_propagate(Tensor<type*, 1>, const Tensor<Tensor<Index, 1>, 1>&, LayerForwardPropagation*, const bool&);

protected:

    Tensor<Index, 1> inputs_dimensions;

    Index regions_number = 2000;
    Index region_rows = 22;
    Index region_columns = 22;
    Index channels_number = 3;

   bool display = true;


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

        const RegionProposalLayer* region_proposal_layer_pointer = static_cast<RegionProposalLayer*>(layer_pointer);

        const Index regions_number = region_proposal_layer_pointer->get_regions_number();

        batch_samples_number = new_batch_samples_number*regions_number;

        const Index region_rows = region_proposal_layer_pointer->get_region_rows();
        const Index region_columns =  region_proposal_layer_pointer->get_region_columns();
        const Index channels_number =  region_proposal_layer_pointer->get_channels_number();

        // TODO proposal fix to undelclared variables: outputs_data, outputs_dimensions
        std::vector<type*> outputs_data;
        std::vector<Tensor<long, 1>> outputs_dimensions;

        outputs_data.resize(2);
        outputs_dimensions.resize(2);

        // Image patches
        outputs_data[0] = (type*)malloc(static_cast<size_t>(batch_samples_number*regions_number*region_rows*region_columns*channels_number*sizeof(type)));

        outputs_dimensions[0].resize(4);

        outputs_dimensions[0].setValues({batch_samples_number,
                                         region_rows,
                                         region_columns,
                                         channels_number});

        // Bounding boxes

        outputs_data[0] = (type*)malloc(static_cast<size_t>(1));

        outputs_dimensions[1].resize(1);

        outputs_dimensions[1].setValues({1});

    }

    void print() const
    {
        cout << "Region proposal layer forward propagation structure" << endl;

        const RegionProposalLayer* region_proposal_layer_pointer = static_cast<RegionProposalLayer*>(layer_pointer);

        const Index region_rows = region_proposal_layer_pointer->get_region_rows();
        const Index region_columns =  region_proposal_layer_pointer->get_region_columns();
        const Index channels_number =  region_proposal_layer_pointer->get_channels_number();

        cout << "Image patches:" << endl;

//        cout << TensorMap<Tensor<type,4>>(outputs_data(0),
//                                          outputs_dimensions[0](0),
//                                          outputs_dimensions[0](1),
//                                          outputs_dimensions[0](2),
//                                          outputs_dimensions[0](3)) << endl;

//        outputs_dimensions[0].setValues({batch_samples_number,
//                                         region_rows,
//                                         region_columns,
//                                         channels_number});


        cout << "Batch samples number: " << batch_samples_number << endl;
        cout << "Region rows: " << region_rows << endl;
        cout << "Region columns: " << region_columns << endl;
        cout << "Channels number: " << channels_number << endl;


//        cout << "Outputs dimensions:" << endl;

//        cout << outputs_dimensions[0] << endl;

        cout << "Bounding boxes:" << endl;

    }

    Tensor<Index, 2> regions;

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
