/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   U N S C A L I N G   L A Y E R   C L A S S   H E A D E R                                                    */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
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

#include "tinyxml2.h"


namespace OpenNN
{

/// This class represents a layer of unscaling neurons.
/// Unscaling layers are included in the definition of a neural network.
/// They are used to unnormalize variables so they are in the original range after computer processing.

class UnscalingLayer
{

public:

   // DEFAULT CONSTRUCTOR

   explicit UnscalingLayer();

   // UNSCALING NEURONS NUMBER CONSTRUCTOR

   explicit UnscalingLayer(const size_t&);

   // STATISTICS CONSTRUCTOR

   explicit UnscalingLayer(const Vector< Statistics<double> >&);

   // XML CONSTRUCTOR

   explicit UnscalingLayer(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   UnscalingLayer(const UnscalingLayer&);

   // DESTRUCTOR

   virtual ~UnscalingLayer();

   // ASSIGNMENT OPERATOR

   UnscalingLayer& operator = (const UnscalingLayer&);

   // EQUAL TO OPERATOR

   bool operator == (const UnscalingLayer&) const;

   // ENUMERATIONS

   /// Enumeration of available methods for input variables, output variables and independent parameters scaling.  
   
   enum UnscalingMethod{NoUnscaling, MinimumMaximum, MeanStandardDeviation, Logarithmic};

   // GET METHODS

   // Outputs number

   size_t get_unscaling_neurons_number() const;

   // Output variables statistics

   Vector< Statistics<double> > get_statistics() const;

   Matrix<double> get_statistics_matrix() const;
   Vector<double> get_minimums() const;
   Vector<double> get_maximums() const;

   // Outputs unscaling method

   const UnscalingMethod& get_unscaling_method() const;

   string write_unscaling_method() const;
   string write_unscaling_method_text() const;

   // Display messages

   const bool& get_display() const;

   // SET METHODS

   void set();
   void set(const size_t&);
   void set(const Vector< Statistics<double> >&);
   void set(const tinyxml2::XMLDocument&);
   void set(const UnscalingLayer&);

   virtual void set_default();

   // Output variables statistics

   void set_statistics(const Vector< Statistics<double> >&);
   void set_statistics_eigen(const Eigen::MatrixXd&);

   void set_item_statistics(const size_t&, const Statistics<double>&);

   void set_minimum(const size_t&, const double&);
   void set_maximum(const size_t&, const double&);
   void set_mean(const size_t&, const double&);
   void set_standard_deviation(const size_t&, const double&);

   // Outputs unscaling method

   void set_unscaling_method(const UnscalingMethod&);
   void set_unscaling_method(const string&);

   // Display messages

   void set_display(const bool&);

   // Pruning and growing

   void prune_unscaling_neuron(const size_t&);

   // Check methods

   bool is_empty() const;
  
   // UnscalingLayer and unscaling

   void initialize_random();

   Matrix<double> calculate_outputs(const Matrix<double>&) const;
   Matrix<double> calculate_derivatives(const Matrix<double>&) const;
   Matrix<double> calculate_second_derivatives(const Matrix<double>&) const;

   Matrix<double> calculate_minimum_maximum_outputs(const Matrix<double>&) const;
   Matrix<double> calculate_minimum_maximum_derivatives(const Matrix<double>&) const;
   Matrix<double> calculate_minimum_maximum_second_derivatives(const Matrix<double>&) const;

   Matrix<double> calculate_mean_standard_deviation_outputs(const Matrix<double>&) const;
   Matrix<double> calculate_mean_standard_deviation_derivatives(const Matrix<double>&) const;
   Matrix<double> calculate_mean_standard_deviation_second_derivatives(const Matrix<double>&) const;

   Matrix<double> calculate_logarithmic_outputs(const Matrix<double>&) const;
   Matrix<double> calculate_logarithmic_derivatives(const Matrix<double>&) const;
   Matrix<double> calculate_logarithmic_second_derivatives(const Matrix<double>&) const;

   Vector< Matrix<double> > calculate_Jacobian(const Matrix<double>&) const;
   Vector< Matrix<double> > calculate_Hessian(const Vector<double>&) const;

   void check_range(const Vector<double>&) const;

   // Serialization methods

   string object_to_string() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML(   );

   // PMML Methods
   void to_PMML(tinyxml2::XMLElement*, const Vector<string>& ) const;
   void write_PMML(tinyxml2::XMLPrinter&, const Vector<string>&) const;

   void from_PMML(const tinyxml2::XMLElement*, const Vector<string>& );

   // Expression methods

   string write_none_expression(const Vector<string>&, const Vector<string>&) const;
   string write_minimum_maximum_expression(const Vector<string>&, const Vector<string>&) const;
   string write_mean_standard_deviation_expression(const Vector<string>&, const Vector<string>&) const;
   string write_logarithmic_expression(const Vector<string>&, const Vector<string>&) const;

   string write_expression(const Vector<string>&, const Vector<string>&) const;

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

