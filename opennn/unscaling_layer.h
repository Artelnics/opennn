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

#include "vector.h"
#include "matrix.h"
#include "layer.h"
#include "statistics.h"



#include "tinyxml2.h"

// Eigen includes

#include "../eigen/Eigen"

namespace OpenNN
{

/// This class represents a layer of unscaling neurons.

///
/// Unscaling layers are included in the definition of a neural network.
/// They are used to unnormalize variables so they are in the original range after computer processing.

class UnscalingLayer : public Layer
{

public:

   // Constructors

   explicit UnscalingLayer();

   explicit UnscalingLayer(const size_t&);

   explicit UnscalingLayer(const Vector<Descriptives>&);

   explicit UnscalingLayer(const tinyxml2::XMLDocument&);

   UnscalingLayer(const UnscalingLayer&);

   // Destructor

   virtual ~UnscalingLayer();

   // Enumerations

   /// Enumeration of available methods for input variables, output variables and independent parameters scaling.  
   
   enum UnscalingMethod{NoUnscaling, MinimumMaximum, MeanStandardDeviation, Logarithmic};

   // Get methods

   Vector<size_t> get_input_variables_dimensions() const;

   size_t get_inputs_number() const;
   size_t get_neurons_number() const;

   Vector<Descriptives> get_descriptives() const;

   Matrix<double> get_descriptives_matrix() const;
   Vector<double> get_minimums() const;
   Vector<double> get_maximums() const;

   const UnscalingMethod& get_unscaling_method() const;

   string write_unscaling_method() const;
   string write_unscaling_method_text() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const size_t&);
   void set(const Vector<Descriptives>&);
   void set(const tinyxml2::XMLDocument&);
   void set(const UnscalingLayer&);

   void set_inputs_number(const size_t&);
   void set_neurons_number(const size_t&);

   virtual void set_default();

   // Output variables descriptives

   void set_descriptives(const Vector<Descriptives>&);
   void set_descriptives_eigen(const Eigen::MatrixXd&);

   void set_item_descriptives(const size_t&, const Descriptives&);

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

   void prune_neuron(const size_t&);

   // Check methods

   bool is_empty() const;
  
   Tensor<double> calculate_outputs(const Tensor<double>&);

   Tensor<double> calculate_minimum_maximum_outputs(const Tensor<double>&) const;

   Tensor<double> calculate_mean_standard_deviation_outputs(const Tensor<double>&) const;

   Tensor<double> calculate_logarithmic_outputs(const Tensor<double>&) const;

   void check_range(const Vector<double>&) const;

   // Serialization methods

   string object_to_string() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   // Expression methods

   string write_none_expression(const Vector<string>&, const Vector<string>&) const;
   string write_minimum_maximum_expression(const Vector<string>&, const Vector<string>&) const;
   string write_mean_standard_deviation_expression(const Vector<string>&, const Vector<string>&) const;
   string write_logarithmic_expression(const Vector<string>&, const Vector<string>&) const;
   string write_none_expression_php(const Vector<string>&, const Vector<string>&) const;
   string write_minimum_maximum_expression_php(const Vector<string>&, const Vector<string>&) const;
   string write_mean_standard_deviation_expression_php(const Vector<string>&, const Vector<string>&) const;
   string write_logarithmic_expression_php(const Vector<string>&, const Vector<string>&) const;

   string write_expression(const Vector<string>&, const Vector<string>&) const;
   string write_expression_php(const Vector<string>&, const Vector<string>&) const;

protected:

   // MEMBERS

   /// Descriptives of output variables.

   Vector<Descriptives> descriptives;

   /// Unscaling method for the output variables.

   UnscalingMethod unscaling_method;

   /// Display warning messages to screen. 

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

