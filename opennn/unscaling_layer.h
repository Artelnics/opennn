/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   U N S C A L I N G   L A Y E R   C L A S S   H E A D E R                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __UNSCALINGLAYER_H__
#define __UNSCALINGLAYER_H__

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

/// This class represents a layer of unscaling neurons.
/// Unscaling layers are included in the definition of a neural network. 
/// They are used to unnormalize variables so they are in the original range after computer processing.  

class UnscalingLayer
{

public:

   // DEFAULT CONSTRUCTOR

   explicit UnscalingLayer(void);

   // UNSCALING NEURONS NUMBER CONSTRUCTOR

   explicit UnscalingLayer(const size_t&);

   // STATISTICS CONSTRUCTOR

   explicit UnscalingLayer(const Vector< Statistics<double> >&);

   // XML CONSTRUCTOR

   explicit UnscalingLayer(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   UnscalingLayer(const UnscalingLayer&);

   // DESTRUCTOR

   virtual ~UnscalingLayer(void);

   // ASSIGNMENT OPERATOR

   UnscalingLayer& operator = (const UnscalingLayer&);

   // EQUAL TO OPERATOR

   bool operator == (const UnscalingLayer&) const;

   // ENUMERATIONS

   /// Enumeration of available methods for input variables, output variables and independent parameters scaling.  
   
   enum UnscalingMethod{NoUnscaling, MinimumMaximum, MeanStandardDeviation};

   // GET METHODS

   // Outputs number

   size_t get_unscaling_neurons_number(void) const;

   // Output variables statistics

   Vector< Statistics<double> > get_statistics(void) const;

   Matrix<double> arrange_statistics(void) const;
   Vector<double> arrange_minimums(void) const;
   Vector<double> arrange_maximums(void) const;

   // Outputs unscaling method

   const UnscalingMethod& get_unscaling_method(void) const;

   std::string write_unscaling_method(void) const;
   std::string write_unscaling_method_text(void) const;

   // Display messages

   const bool& get_display(void) const;

   // SET METHODS

   void set(void);
   void set(const size_t&);
   void set(const Vector< Statistics<double> >&);
   void set(const tinyxml2::XMLDocument&);
   void set(const UnscalingLayer&);

   virtual void set_default(void);

   // Output variables statistics

   void set_statistics(const Vector< Statistics<double> >&);
   void set_item_statistics(const size_t&, const Statistics<double>&);

   void set_minimum(const size_t&, const double&);
   void set_maximum(const size_t&, const double&);
   void set_mean(const size_t&, const double&);
   void set_standard_deviation(const size_t&, const double&);

   // Outputs unscaling method

   void set_unscaling_method(const UnscalingMethod&);
   void set_unscaling_method(const std::string&);

   // Display messages

   void set_display(const bool&);

   // Pruning and growing

   void prune_unscaling_neuron(const size_t&);

   // Check methods

   bool is_empty(void) const;
  
   // UnscalingLayer and unscaling

   void initialize_random(void);

   Vector<double> calculate_outputs(const Vector<double>&) const;
   Vector<double> calculate_derivatives(const Vector<double>&) const;
   Vector<double> calculate_second_derivatives(const Vector<double>&) const;

   Vector<double> calculate_minimum_maximum_outputs(const Vector<double>&) const;
   Vector<double> calculate_minimum_maximum_derivatives(const Vector<double>&) const;
   Vector<double> calculate_minimum_maximum_second_derivatives(const Vector<double>&) const;

   Vector<double> calculate_mean_standard_deviation_outputs(const Vector<double>&) const;
   Vector<double> calculate_mean_standard_deviation_derivatives(const Vector<double>&) const;
   Vector<double> calculate_mean_standard_deviation_second_derivatives(const Vector<double>&) const;

   Matrix<double> arrange_Jacobian(const Vector<double>&) const;
   Vector< Matrix<double> > arrange_Hessian_form(const Vector<double>&) const;

   void check_range(const Vector<double>&) const;

   // Serialization methods

   std::string to_string(void) const;

   tinyxml2::XMLDocument* to_XML(void) const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML(   );

   // PMML Methods
   void to_PMML(tinyxml2::XMLElement*, const Vector<std::string>& ) const;
   void write_PMML(tinyxml2::XMLPrinter&, const Vector<std::string>&) const;

   void from_PMML(const tinyxml2::XMLElement*, const Vector<std::string>& );

   // Expression methods

   std::string write_none_expression(const Vector<std::string>&, const Vector<std::string>&) const;
   std::string write_minimum_maximum_expression(const Vector<std::string>&, const Vector<std::string>&) const;
   std::string write_mean_stadard_deviation_expression(const Vector<std::string>&, const Vector<std::string>&) const;

   std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const;

protected:

   // MEMBERS

   /// Statistics of output variables.

   Vector< Statistics<double> > statistics;

   /// Unscaling method for the output variables.

   UnscalingMethod unscaling_method;

   /// Display warning messages to screen. 

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

