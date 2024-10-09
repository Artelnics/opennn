//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   4 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ScalingLayer4D_H
#define ScalingLayer4D_H

#include <iostream>
#include <string>

#include "scaling.h"
#include "layer.h"
#include "layer_forward_propagation.h"

namespace opennn
{

class ScalingLayer4D : public Layer
{

public:

   // Constructors

   explicit ScalingLayer4D();

   explicit ScalingLayer4D(const dimensions&);

   // Get

   dimensions get_output_dimensions() const;

   Index get_inputs_number() const final;
   dimensions get_input_dimensions() const;
   Index get_neurons_number() const final;

   // Display messages

   const bool& get_display() const;

   // Set

   void set();
   void set(const dimensions&);
   void set(const tinyxml2::XMLDocument&);

   void set_default();

   // Descriptives

   void set_min_max_range(const type& min, const type& max);

   // Display messages

   void set_display(const bool&);

   // Check

   bool is_empty() const;

   void forward_propagate(const vector<pair<type*, dimensions>>&, 
                          LayerForwardPropagation*, 
                          const bool&) final;

   // Serialization

   void print() const;

   virtual void from_XML(const tinyxml2::XMLDocument&) final;

   void to_XML(tinyxml2::XMLPrinter&) const final;

protected:

   dimensions input_dimensions;

   type min_range;
   type max_range;

   bool display = true;

};


struct ScalingLayer4DForwardPropagation : LayerForwardPropagation
{
    // Constructor

    explicit ScalingLayer4DForwardPropagation() : LayerForwardPropagation()
    {
    }


    virtual ~ScalingLayer4DForwardPropagation()
    {
    }

    // Constructor

    explicit ScalingLayer4DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }
    
    
    pair<type*, dimensions> get_outputs_pair() const final;


    void set(const Index& new_batch_samples_number, Layer* new_layer) final;


    void print() const
    {
        cout << "Scaling Outputs:" << endl;
        cout << outputs.dimensions() << endl;
    }

    Tensor<type, 4> outputs;
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
