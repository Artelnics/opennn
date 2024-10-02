//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B O U N D I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef BOUNDINGLAYER_H
#define BOUNDINGLAYER_H

// System includes

#include <iostream>
#include <string>

// OpenNN includes

#include "layer.h"

#include "config.h"
#include "layer_forward_propagation.h"

namespace opennn
{

class BoundingLayer : public Layer
{

public:

   // Constructors

   explicit BoundingLayer();

   explicit BoundingLayer(const dimensions&);

   // Enumerations

   enum class BoundingMethod{NoBounding, Bounding};

   // Check

   bool is_empty() const;

   // Get

   Index get_inputs_number() const final;
   Index get_neurons_number() const final;

   dimensions get_output_dimensions() const final;

   const BoundingMethod& get_bounding_method() const;

   string get_bounding_method_string() const;

   const Tensor<type, 1>& get_lower_bounds() const;
   type get_lower_bound(const Index&) const;

   const Tensor<type, 1>& get_upper_bounds() const;
   type get_upper_bound(const Index&) const;

   // Variables bounds

   void set();
   void set(const Index&);
   void set(const tinyxml2::XMLDocument&);
   void set(const BoundingLayer&);

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;

   void set_bounding_method(const BoundingMethod&);
   void set_bounding_method(const string&);

   void set_lower_bounds(const Tensor<type, 1>&);
   void set_lower_bound(const Index&, const type&);

   void set_upper_bounds(const Tensor<type, 1>&);
   void set_upper_bound(const Index&, const type&);

   void set_display(const bool&);

   void set_default();

   // Lower and upper bounds

   void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&, 
                          LayerForwardPropagation*, 
                          const bool&) final;

   // Expression

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

   // Serialization

   void print() const;

   void from_XML(const tinyxml2::XMLDocument&) final;

   void to_XML(tinyxml2::XMLPrinter&) const final;

private:

   // MEMBERS

   BoundingMethod bounding_method = BoundingMethod::Bounding;

   Tensor<type, 1> lower_bounds;

   Tensor<type, 1> upper_bounds;

   bool display = true;
};


struct BoundingLayerForwardPropagation : LayerForwardPropagation
{
    // Constructor

    explicit BoundingLayerForwardPropagation() : LayerForwardPropagation()
    {
    }

    // Constructor

    explicit BoundingLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    virtual ~BoundingLayerForwardPropagation()
    {
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

