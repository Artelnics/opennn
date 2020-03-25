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

#include "vector.h"
#include "matrix.h"
#include "layer.h"
#include "functions.h"
#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents a layer of bounding neurons. 

/// A bounding layer is used to ensure that variables will never fall below or above given values. 

class BoundingLayer : public Layer
{

public:

   // Constructors

   explicit BoundingLayer();

   explicit BoundingLayer(const size_t&);

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

   Vector<size_t> get_input_variables_dimensions() const;
   size_t get_inputs_number() const;
   size_t get_neurons_number() const;

   const BoundingMethod& get_bounding_method() const;

   string write_bounding_method() const;

   const Vector<double>& get_lower_bounds() const;
   double get_lower_bound(const size_t&) const;

   const Vector<double>& get_upper_bounds() const;
   double get_upper_bound(const size_t&) const;

   Vector<Vector<double>> get_bounds();

   // Variables bounds

   void set();
   void set(const size_t&);
   void set(const tinyxml2::XMLDocument&);
   void set(const BoundingLayer&);

   void set_inputs_number(const size_t&);
   void set_neurons_number(const size_t&);


   void set_bounding_method(const BoundingMethod&);
   void set_bounding_method(const string&);

   void set_lower_bounds(const Vector<double>&);
   void set_lower_bound(const size_t&, const double&);

   void set_upper_bounds(const Vector<double>&);
   void set_upper_bound(const size_t&, const double&);

   void set_bounds(const Vector<Vector<double>>&);

   void set_display(const bool&);

   void set_default();

   // Pruning and growing

   void prune_neuron(const size_t&);

   // Lower and upper bounds

   Tensor<double> calculate_outputs(const Tensor<double>&);

   // Expression methods

   string write_expression(const Vector<string>&, const Vector<string>&) const;
   string write_expression_php(const Vector<string>&, const Vector<string>&) const;

   // Serialization methods

   string object_to_string() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML( );

protected:

   // MEMBERS

   /// Method used to bound the values.

   BoundingMethod bounding_method;

   /// Lower bounds of output variables

   Vector<double> lower_bounds;

   /// Upper bounds of output variables

   Vector<double> upper_bounds;

   /// Display messages to screen. 

   bool display;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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

