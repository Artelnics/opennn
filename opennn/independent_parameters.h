/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N D E P E N D E N T    P A R A M E T E R S   C L A S S   H E A D E R                                     */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __INDEPENDENTPARAMETERS_H__
#define __INDEPENDENTPARAMETERS_H__

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

/// This class represents the concept of independent parameters. 
/// Independent parameters are set of free parameters which do not belong to any class of neuron. 
/// They can be used for many different purposes, including a turn around for function optimization problems.

class IndependentParameters
{

public:

   // DEFAULT CONSTRUCTOR

   explicit IndependentParameters();

   // INDEPENDENT PARAMETERS NUMBER CONSTRUCTOR

   explicit IndependentParameters(const size_t&);

   // INDEPENDENT PARAMETERS CONSTRUCTOR

   //explicit IndependentParameters(const Vector<double>&);

   // COPY CONSTRUCTOR

   IndependentParameters(const IndependentParameters&);

   // DESTRUCTOR

   virtual ~IndependentParameters();

   // ASSIGNMENT OPERATOR

   IndependentParameters& operator =(const IndependentParameters&);

   // EQUAL TO OPERATOR

   bool operator ==(const IndependentParameters&) const;

   // ENUMERATIONS

   /// Enumeration of available methods for scaling and unscaling the independent parameters.  
   
   enum ScalingMethod{NoScaling, MeanStandardDeviation, MinimumMaximum};

   /// Enumeration of available methods for bounding the independent parameters.

   enum BoundingMethod{NoBounding, Bounding};

   // GET METHODS

   // Independent parameters

   /// Returns the number of parameters independent of the multilayer perceptron
   /// Independent parameters can be used in the context of neural netwotks for many purposes.

   inline size_t get_parameters_number() const
   {
      return(parameters.size());
   }

   const Vector<double>& get_parameters() const;   
   double get_parameter(const size_t&) const;

   // Independent parameters information

   const Vector<string>& get_names() const;
   const string& get_name(const size_t&) const;

   const Vector<string>& get_units() const;
   const string& get_unit(const size_t&) const;

   const Vector<string>& get_descriptions() const;
   const string& get_description(const size_t&) const;

   // Independent parameters statistics

   const Vector<double>& get_minimums() const;
   double get_minimum(const size_t&) const;

   const Vector<double>& get_maximums() const;
   double get_maximum(const size_t&) const;

   const Vector<double>& get_means() const;
   double get_mean(const size_t&) const;

   const Vector<double>& get_standard_deviations() const;
   double get_standard_deviation(const size_t&) const;

   // Independent parameters scaling and unscaling

   const ScalingMethod& get_scaling_method() const;
   string write_scaling_method() const;

   // Independent parameters bounds

   const Vector<double>& get_lower_bounds() const;
   double get_lower_bound(const size_t&) const;

   const Vector<double>& get_upper_bounds() const;
   double get_upper_bound(const size_t&) const;

   Vector< Vector<double>* > get_bounds();

   const BoundingMethod& get_bounding_method() const;
   string write_bounding_method() const;

   // Display messages

   const bool& get_display() const;

   // SET METHODS

   void set();
   void set(const size_t&);
   void set(const Vector<double>&);
   void set(const IndependentParameters&);

   virtual void set_default();

   // Independent parameters

   void set_parameters_number(const size_t&);

   void set_parameters(const Vector<double>&);
   void set_parameter(const size_t&, const double&);

   // Independent parameters information

   void set_names(const Vector<string>&);
   void set_name(const size_t&, const string&);

   void set_units(const Vector<string>&);
   void set_unit(const size_t&, const string&);

   void set_descriptions(const Vector<string>&);
   void set_description(const size_t&, const string&);

   // Independent parameters statistics

   void set_minimums(const Vector<double>&);
   void set_minimum(const size_t&, const double&);

   void set_maximums(const Vector<double>&);
   void set_maximum(const size_t&, const double&);

   void set_means(const Vector<double>&);
   void set_mean(const size_t&, const double&);

   void set_standard_deviations(const Vector<double>&);
   void set_standard_deviation(const size_t&, const double&);
   
   // Independent parameters scaling and unscaling

   void set_scaling_method(const ScalingMethod&);
   void set_scaling_method(const string&);

   // Independent parameters bounds

   void set_lower_bounds();
   void set_lower_bounds(const Vector<double>&);
   void set_lower_bound(const size_t&, const double&);

   void set_upper_bounds();
   void set_upper_bounds(const Vector<double>&);
   void set_upper_bound(const size_t&, const double&);

   void set_bounds();
   void set_bounds(const Vector< Vector<double> >&);

   void set_bounding_method(const BoundingMethod&);
   void set_bounding_method(const string&);

   // Display messages

   void set_display(const bool&);

   // Check methods

   bool is_empty() const;   

   // Independent parameters initialization methods

   void initialize_random();

   void initialize_parameters(const double&);

   void randomize_parameters_uniform();
   void randomize_parameters_uniform(const double&, const double&);
   void randomize_parameters_uniform(const Vector<double>&, const Vector<double>&);
   void randomize_parameters_uniform(const Vector< Vector<double> >&);

   void randomize_parameters_normal();
   void randomize_parameters_normal(const double&, const double&);
   void randomize_parameters_normal(const Vector<double>&, const Vector<double>&);
   void randomize_parameters_normal(const Vector< Vector<double> >&);

   // Parameters norm 

   //double calculate_parameters_norm() const;

   // Independent parameters

   Vector<double> calculate_scaled_parameters() const;
   void unscale_parameters(const Vector<double>&);

   void bound_parameters();
   void bound_parameter(const size_t&);

   // Information 

   Vector< Vector<string> > arrange_information();     

   //void set_information(const Vector< Vector<string> >&);     

   // Statistics 

   Vector< Vector<double> > arrange_statistics();

   Vector< Vector<double> > arrange_minimums_maximums();
   Vector< Vector<double> > arrange_means_standard_deviations();

   void set_statistics(const Vector< Vector<double> >&);
   void set_minimums_maximums(const Vector< Vector<double> >&);
   void set_means_standard_deviations(const Vector< Vector<double> >&);

   // Serialization methods

   string object_to_string() const;

  /// ownership passed - use delete to destroy
   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

protected:

   // MEMBERS

   /// Independent parameters.

   Vector<double> parameters;

   /// Name of independent parameters.

   Vector<string> names;

   /// Units of independent parameters.

   Vector<string> units;

   /// Description of independent parameters.

   Vector<string> descriptions;

   /// Minimum of independent parameters.

   Vector<double> minimums;

   /// Maximum of independent parameters.

   Vector<double> maximums;

   /// Mean of independent parameters.

   Vector<double> means;

   /// Standard deviation of independent parameters.

   Vector<double> standard_deviations;

   /// Lower bound of independent parameters.

   Vector<double> lower_bounds;

   /// Upper bound of independent parameters.

   Vector<double> upper_bounds;

   /// Independent parameters scaling and unscaling method.

   ScalingMethod scaling_method;

   /// Independent parameters bounding method.

   BoundingMethod bounding_method;

   /// Display warnings when the the independent parameters fall outside their minimum-maximum range. 

   bool display_range_warning;

   /// Display messages to screen. 

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

