/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S C A L I N G   L A Y E R   C L A S S   H E A D E R                                                        */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __SCALINGLAYER_H__
#define __SCALINGLAYER_H__

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

/// This class represents a layer of scaling neurons.
/// Scaling layers are included in the definition of a neural network. 
/// They are used to normalize variables so they are in an appropriate range for computer processing.  

class ScalingLayer
{

public:

   // DEFAULT CONSTRUCTOR

   explicit ScalingLayer();

   // INPUTS NUMBER CONSTRUCTOR

   explicit ScalingLayer(const size_t&);

   // STATISTICS CONSTRUCTOR

   explicit ScalingLayer(const Vector< Statistics<double> >&);

   // COPY CONSTRUCTOR

   ScalingLayer(const ScalingLayer&);

   // DESTRUCTOR

   virtual ~ScalingLayer();

   // ASSIGNMENT OPERATOR

   ScalingLayer& operator = (const ScalingLayer&);

   // EQUAL TO OPERATOR

   bool operator == (const ScalingLayer&) const;

   // ENUMERATIONS

   /// Enumeration of available methods for scaling the input variables.  
   
   enum ScalingMethod{NoScaling, MinimumMaximum, MeanStandardDeviation, StandardDeviation};

   // GET METHODS

   size_t get_scaling_neurons_number() const;

   // Inputs statistics

   Vector< Statistics<double> > get_statistics() const;
   Statistics<double> get_statistics(const size_t&) const;

   Matrix<double> get_statistics_matrix() const;

   Vector<double> get_minimums() const;
   Vector<double> get_maximums() const;
   Vector<double> get_means() const;
   Vector<double> get_standard_deviations() const;

   // Variables scaling and unscaling

   const Vector<ScalingMethod> get_scaling_methods() const;

   Vector<string> write_scaling_methods() const;
   Vector<string> write_scaling_methods_text() const;

   // Display messages

   const bool& get_display() const;

   // SET METHODS

   void set();
   void set(const size_t&);
   void set(const Vector< Statistics<double> >&);
   void set(const tinyxml2::XMLDocument&);
   void set(const ScalingLayer&);

   void set(const Vector<bool>&);

   virtual void set_default();

   // Statistics

   void set_statistics(const Vector< Statistics<double> >&);
   void set_statistics_eigen(const Eigen::MatrixXd&);
   void set_item_statistics(const size_t&, const Statistics<double>&);

   void set_minimum(const size_t&, const double&);
   void set_maximum(const size_t&, const double&);
   void set_mean(const size_t&, const double&);
   void set_standard_deviation(const size_t&, const double&);

   // Scaling method

   void set_scaling_methods(const Vector<ScalingMethod>&);
   void set_scaling_methods(const Vector<string>&);
   void set_scaling_methods(const vector<string>&);

   void set_scaling_methods(const ScalingMethod&);
   void set_scaling_methods(const string&);

   // Display messages

   void set_display(const bool&);

   // Pruning and growing

   void grow_scaling_neuron(const Statistics<double>& new_statistics = Statistics<double>());

   void prune_scaling_neuron(const size_t&);

   // Check methods

   bool is_empty() const;

   // Inputs scaling function

   void initialize_random();

   void check_range(const Vector<double>&) const;

   Matrix<double> calculate_outputs(const Matrix<double>&) const;
   Matrix<double> calculate_derivatives(const Matrix<double>&) const;
   Matrix<double> calculate_second_derivatives(const Matrix<double>&) const;

   Matrix<double> calculate_minimum_maximum_outputs(const Matrix<double>&) const;
   Matrix<double> calculate_minimum_maximum_derivatives(const Matrix<double>&) const;
   Matrix<double> calculate_minimum_maximum_second_derivatives(const Matrix<double>&) const;

   Matrix<double> calculate_mean_standard_deviation_outputs(const Matrix<double>&) const;
   Matrix<double> calculate_mean_standard_deviation_derivatives(const Matrix<double>&) const;
   Matrix<double> calculate_mean_standard_deviation_second_derivatives(const Matrix<double>&) const;

   Vector<Matrix<double>> calculate_Jacobian(const Matrix<double>&) const;
   Vector< Matrix<double> > calculate_Hessian(const Vector<double>&) const;

   // Expression methods

   string write_no_scaling_expression(const Vector<string>&, const Vector<string>&) const;

   string write_minimum_maximum_expression(const Vector<string>&, const Vector<string>&) const;

   string write_mean_standard_deviation_expression(const Vector<string>&, const Vector<string>&) const;

   string write_standard_deviation_expression(const Vector<string>&, const Vector<string>&) const;

   string write_expression(const Vector<string>&, const Vector<string>&) const;

   // Serialization methods

   string object_to_string() const;

   tinyxml2::XMLDocument* to_XML() const;
   virtual void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   // PMML Methods

   void to_PMML(tinyxml2::XMLElement*, const Vector<string>&) const;
   void write_PMML(tinyxml2::XMLPrinter&, const Vector<string>&) const;


   void from_PMML(const tinyxml2::XMLElement*, const Vector<string>&);

protected:

   // MEMBERS

   /// Statistics of input variables.

   Vector< Statistics<double> > statistics;

   /// Vector of scaling methods for each variable.

   Vector<ScalingMethod> scaling_methods;

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

