/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R E N D I N G   L A Y E R   C L A S S   H E A D E R                                                      */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __TRENDINGLAYER_H__
#define __TRENDINGLAYER_H__

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>


// OpenNN includes

#include "vector.h"
#include "matrix.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

/// This class represents a layer of trending neurons.
/// A trending layer is used to ensure that variables will never fall below or above given values.

class TrendingLayer
{

public:

   // DEFAULT CONSTRUCTOR

   explicit TrendingLayer(void);

   // TRENDING NEURONS NUMBER CONSTRUCTOR

   explicit TrendingLayer(const size_t&);

   // XML CONSTRUCTOR

   explicit TrendingLayer(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   TrendingLayer(const TrendingLayer&);

   // DESTRUCTOR

   virtual ~TrendingLayer(void);

   // ASSIGNMENT OPERATOR

   TrendingLayer& operator = (const TrendingLayer&);

   // EQUAL TO OPERATOR

   bool operator == (const TrendingLayer&) const;

   // ENUMERATIONS

   /// Enumeration of available methods for trending the output variables.

   enum TrendingMethod{Exponential, Linear};

   // METHODS

   bool is_empty(void) const;

   size_t get_trending_neurons_number(void) const;

   const TrendingMethod& get_trending_method(void) const;

   std::string write_trending_method(void) const;

   const Vector<double>& get_intercepts(void) const;
   double get_intercept(const size_t&) const;

   const Vector<double>& get_slopes(void) const;
   double get_slope(const size_t&) const;

   //Vector< Vector<double>* > get_bounds(void);

   // Variables

   void set(void);
   void set(const size_t&);
   void set(const tinyxml2::XMLDocument&);
   void set(const TrendingLayer&);

   void set_trending_method(const TrendingMethod&);
   void set_trending_method(const std::string&);

   void set_intercepts(const Vector<double>&);
   void set_intercept(const size_t&, const double&);

   void set_slopes(const Vector<double>&);
   void set_slope(const size_t&, const double&);

   //void set_bounds(const Vector< Vector<double> >&);

   void set_display(const bool&);

   void set_default(void);

   // Pruning and growing

   void prune_trending_neuron(const size_t&);

   // Initialization

   void initialize_random(void);

   Vector<double> calculate_outputs(const Vector<double>&, const double&) const;
   Vector<double> calculate_derivative(const Vector<double>&) const;
   Vector<double> calculate_second_derivative(const Vector<double>&) const;

   Matrix<double> arrange_Jacobian(const Vector<double>&) const;
   Vector< Matrix<double> > arrange_Hessian_form(const Vector<double>&) const;

   // Expression methods

   std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const;

   // Serialization methods

   std::string to_string(void) const;

   tinyxml2::XMLDocument* to_XML(void) const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML(  );

protected:

   // MEMBERS

   TrendingMethod trending_method;

   Vector<double> intercepts;

   Vector<double> slopes;

   bool display;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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

