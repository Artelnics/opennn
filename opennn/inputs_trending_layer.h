/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N P U T S   T R E N D I N G   L A Y E R   C L A S S   H E A D E R                                        */
/*                                                                                                              */
/*   Patricia Garcia                                                                                            */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   patriciagarcia@artelnics.com                                                                               */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __INPUTSTRENDINGLAYER_H__
#define __INPUTSTRENDINGLAYER_H__

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

#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents a layer of inputs trending neurons.

class InputsTrendingLayer
{

public:

   // DEFAULT CONSTRUCTOR

   explicit InputsTrendingLayer();

   // INPUNTS TRENDING NEURONS NUMBER CONSTRUCTOR

   explicit InputsTrendingLayer(const size_t&);

   // XML CONSTRUCTOR

   explicit InputsTrendingLayer(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   InputsTrendingLayer(const InputsTrendingLayer&);

   // DESTRUCTOR

   virtual ~InputsTrendingLayer();

   // ASSIGNMENT OPERATOR

   InputsTrendingLayer& operator = (const InputsTrendingLayer&);

   // EQUAL TO OPERATOR

   bool operator == (const InputsTrendingLayer&) const;

   // ENUMERATIONS

   enum InputsTrendingMethod{NoTrending, Linear};

   // METHODS

   bool is_empty() const;

   size_t get_inputs_trending_neurons_number() const;

   const InputsTrendingMethod& get_inputs_trending_method() const;

   string write_inputs_trending_method() const;

   Vector<double> get_intercepts() const;
   double get_intercept(const size_t&) const;

   Vector<double> get_slopes() const;
   double get_slope(const size_t&) const;

   Vector<double> get_correlations() const;
   double get_correlation(const size_t&) const;

   Vector< LinearRegressionParameters<double> > get_inputs_trends() const;

   // Variables

   void set();
   void set(const size_t&);
   void set(const tinyxml2::XMLDocument&);
   void set(const InputsTrendingLayer&);

   void set_inputs_trending_method(const InputsTrendingMethod&);
   void set_inputs_trending_method(const string&);

   void set_intercepts(const Vector<double>&);
   void set_intercept(const size_t&, const double&);

   void set_slopes(const Vector<double>&);
   void set_slope(const size_t&, const double&);

   void set_correlations(const Vector<double>&);
   void set_correlation(const size_t&, const double&);

   void set_inputs_trends(const Vector< LinearRegressionParameters<double> >&);

   void set_display(const bool&);

   void set_default();

   // Pruning and growing

   void prune_input_trending_neuron(const size_t&);

   // Initialization

   void initialize_random();

   Matrix<double> calculate_outputs(const Matrix<double>&, const double& = 0.0) const;
   Matrix<double> calculate_derivatives(const Matrix<double>&) const;
   Matrix<double> calculate_second_derivatives(const Matrix<double>&) const;

   Matrix<double> calculate_Jacobian(const Vector<double>&) const;
   Vector< Matrix<double> > calculate_Hessian(const Vector<double>&) const;

   // Expression methods

   string write_expression(const Vector<string>&, const Vector<string>&) const;

   // Serialization methods

   string object_to_string() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML(  );

protected:

   // MEMBERS

   InputsTrendingMethod inputs_trending_method;

   Vector< LinearRegressionParameters<double> > inputs_trends;

   bool display;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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

