// @todo Test this method
//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   C L A S S   H E A D E R               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef UNSCALINGLAYER_H
#define UNSCALINGLAYER_H

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
#include "statistics.h"
#include "opennn_strings.h"


namespace opennn
{

/// This class represents a layer of unscaling neurons.

///
/// Unscaling layers are included in the definition of a neural network.
/// They are used to unnormalize variables so they are in the original range after computer processing.

class   UnscalingLayer : public Layer
{

public:

   // Constructors

   explicit UnscalingLayer();

   explicit UnscalingLayer(const Index&);

   explicit UnscalingLayer(const Tensor<Descriptives, 1>&);

   // Get methods  

   Index get_inputs_number() const override;
   Index get_neurons_number() const final;

   Tensor<Descriptives, 1> get_descriptives() const;
   

   Tensor<type, 1> get_minimums() const;
   Tensor<type, 1> get_maximums() const;

   Tensor<Scaler, 1> get_unscaling_method() const;

   Tensor<string, 1> write_unscaling_methods() const;
   Tensor<string, 1> write_unscaling_method_text() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&);
   void set(const Tensor<Descriptives, 1>&);
   void set(const Tensor<Descriptives, 1>&, const Tensor<Scaler, 1>&);
   void set(const tinyxml2::XMLDocument&);
   void set(const UnscalingLayer&);

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;

   virtual void set_default();

   // Output variables descriptives

   void set_descriptives(const Tensor<Descriptives, 1>&);

   void set_item_descriptives(const Index&, const Descriptives&);

   void set_minimum(const Index&, const type&);
   void set_maximum(const Index&, const type&);
   void set_mean(const Index&, const type&);
   void set_standard_deviation(const Index&, const type&);

   void set_min_max_range(const type min, const type max);

   // Outputs unscaling method

   void set_scalers(const Tensor<Scaler,1>&);
   void set_scalers(const string&);
   void set_scalers(const Tensor<string, 1>&);
   void set_scalers(const Scaler&);

   // Display messages

   void set_display(const bool&);

   // Check methods

   bool is_empty() const;
  
   void calculate_outputs(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) final;

   void check_range(const Tensor<type, 1>&) const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;

   // Expression methods

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

   string write_expression_c() const final;
   string write_expression_python() const final;


protected:

   // MEMBERS

   /// Descriptives of output variables.

   Tensor<Descriptives, 1> descriptives;

   /// Unscaling method for the output variables.

   Tensor<Scaler, 1> scalers;

   /// min and max range for unscaling

   type min_range;
   type max_range;

   /// Display warning messages to screen. 

   bool display = true;
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
