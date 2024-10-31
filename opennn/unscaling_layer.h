//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   C L A S S   H E A D E R               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef UNSCALINGLAYER_H
#define UNSCALINGLAYER_H

#include <iostream>
#include <string>

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "scaling.h"

namespace opennn
{

class UnscalingLayer : public Layer
{

public:

   explicit UnscalingLayer(const dimensions& = {0}, const string& = "unscaling_layer");
   
   Index get_inputs_number() const final;
   Index get_neurons_number() const final;

   dimensions get_output_dimensions() const final;

   Tensor<Descriptives, 1> get_descriptives() const; 

   Tensor<type, 1> get_minimums() const;
   Tensor<type, 1> get_maximums() const;

   Tensor<Scaler, 1> get_unscaling_method() const;

   Tensor<string, 1> write_unscaling_methods() const;
   Tensor<string, 1> write_unscaling_method_text() const;

   const bool& get_display() const;

   void set(const Index& = 0, const string& = "unscaling_layer");
   void set(const Tensor<Descriptives, 1>&, const Tensor<Scaler, 1>&);

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;

   void set_descriptives(const Tensor<Descriptives, 1>&);

   void set_item_descriptives(const Index&, const Descriptives&);

   void set_min_max_range(const type min, const type max);


   void set_scalers(const Tensor<Scaler,1>&);
   void set_scalers(const string&);
   void set_scalers(const Tensor<string, 1>&);
   void set_scalers(const Scaler&);

   void set_scaler(const Index&, const string&);

   void set_display(const bool&);

   bool is_empty() const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>&,
                          const bool&) final;

   Tensor<string, 1> write_scalers_text() const;

   void print() const;

   void from_XML(const tinyxml2::XMLDocument&) final;
   void to_XML(tinyxml2::XMLPrinter&) const final;

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

protected:

   // MEMBERS

   Tensor<Descriptives, 1> descriptives;

   Tensor<Scaler, 1> scalers;

   type min_range;
   type max_range;

   bool display = true;
};


struct UnscalingLayerForwardPropagation : LayerForwardPropagation
{
    explicit UnscalingLayerForwardPropagation(const Index& = 0, Layer* = 0);
    
    pair<type*, dimensions> get_outputs_pair() const final;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

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
