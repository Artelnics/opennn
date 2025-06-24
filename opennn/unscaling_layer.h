//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   C L A S S   H E A D E R               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef UNSCALINGLAYER_H
#define UNSCALINGLAYER_H

#include "layer.h"
#include "scaling.h"

namespace opennn
{

class Unscaling : public Layer
{

public:

   Unscaling(const dimensions& = {0}, const string& = "unscaling_layer");
   
   dimensions get_input_dimensions() const override;
   dimensions get_output_dimensions() const override;

   vector<Descriptives> get_descriptives() const; 

   Tensor<type, 1> get_minimums() const;
   Tensor<type, 1> get_maximums() const;

   vector<Scaler> get_unscaling_method() const;

   vector<string> write_unscaling_methods() const;
   vector<string> write_unscaling_method_text() const;

   void set(const Index& = 0, const string& = "unscaling_layer");

   void set_input_dimensions(const dimensions&) override;
   void set_output_dimensions(const dimensions&) override;

   void set_descriptives(const vector<Descriptives>&);

   void set_min_max_range(const type min, const type max);

   void set_scalers(const vector<Scaler>&);
   void set_scalers(const string&);
   void set_scalers(const vector<string>&);
   void set_scalers(const Scaler&);

   bool is_empty() const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>&,
                          const bool&) override;

   vector<string> write_scalers_text() const;

   void print() const override;

   void from_XML(const XMLDocument&) override;
   void to_XML(XMLPrinter&) const override;

   string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

private:

   vector<Descriptives> descriptives;

   vector<Scaler> scalers;

   type min_range;
   type max_range;
};


struct UnscalingForwardPropagation : LayerForwardPropagation
{
    UnscalingForwardPropagation(const Index& = 0, Layer* = 0);
    
    pair<type*, dimensions> get_output_pair() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 2> outputs;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
