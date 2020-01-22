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

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// OpenNN includes

#include "layer.h"
#include "functions.h"
#include "tinyxml2.h"
#include "config.h"

namespace OpenNN
{

/// This class represents a layer of bounding neurons. 

/// A bounding layer is used to ensure that variables will never fall below or above given values. 

class BoundingLayer : public Layer
{

public:

   // Constructors

   explicit BoundingLayer();

   explicit BoundingLayer(const int&);

   explicit BoundingLayer(const tinyxml2::XMLDocument&);

   BoundingLayer(const BoundingLayer&);

   // Destructor
   
   virtual ~BoundingLayer();

   // Enumerations

   /// Enumeration of available methods for bounding the output variables.

   enum BoundingMethod{NoBounding, Bounding};

   // Check methods

   bool is_empty() const;

   // Get methods

   Tensor<int, 1> get_input_variables_dimensions() const;
   int get_inputs_number() const;
   int get_neurons_number() const;

   const BoundingMethod& get_bounding_method() const;

   string write_bounding_method() const;

   const Tensor<type, 1>& get_lower_bounds() const;
   double get_lower_bound(const int&) const;

   const Tensor<type, 1>& get_upper_bounds() const;
   double get_upper_bound(const Index&) const;

   vector<Tensor<type, 1>> get_bounds();

   // Variables bounds

   void set();
   void set(const int&);
   void set(const tinyxml2::XMLDocument&);
   void set(const BoundingLayer&);

   void set_inputs_number(const int&);
   void set_neurons_number(const int&);


   void set_bounding_method(const BoundingMethod&);
   void set_bounding_method(const string&);

   void set_lower_bounds(const Tensor<type, 1>&);
   void set_lower_bound(const int&, const double&);

   void set_upper_bounds(const Tensor<type, 1>&);
   void set_upper_bound(const int&, const double&);

   void set_bounds(const vector<Tensor<type, 1>>&);

   void set_display(const bool&);

   void set_default();

   // Pruning and growing

   void prune_neuron(const int&);

   // Lower and upper bounds

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

   // Expression methods

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_expression_php(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   // Serialization methods

   string object_to_string() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML();

protected:

   // MEMBERS

   /// Method used to bound the values.

   BoundingMethod bounding_method;

   /// Lower bounds of output variables

   Tensor<type, 1> lower_bounds;

   /// Upper bounds of output variables

   Tensor<type, 1> upper_bounds;

   /// Display messages to screen. 

   bool display;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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

