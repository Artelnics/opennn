//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S   H E A D E R                   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SCALINGLAYER_H
#define SCALINGLAYER_H

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

namespace OpenNN
{

/// This class represents a layer of scaling neurons.

///
/// Scaling layers are included in the definition of a neural network. 
/// They are used to normalize variables so they are in an appropriate range for computer processing.  

class ScalingLayer : public Layer
{

public:

   // Constructors

   explicit ScalingLayer();

   explicit ScalingLayer(const size_t&);
   explicit ScalingLayer(const Vector<size_t>&);

   explicit ScalingLayer(const Vector<Descriptives>&);

   ScalingLayer(const ScalingLayer&);

   // Destructors

   virtual ~ScalingLayer();

   /// Enumeration of available methods for scaling the input variables.  
   
   enum ScalingMethod{NoScaling, MinimumMaximum, MeanStandardDeviation, StandardDeviation};

   // Get methods

   Vector<size_t> get_input_variables_dimensions() const;
   Vector<size_t> get_outputs_dimensions() const;

   size_t get_inputs_number() const;
   size_t get_neurons_number() const;

   // Inputs descriptives

   Vector<Descriptives> get_descriptives() const;
   Descriptives get_descriptives(const size_t&) const;

   Matrix<double> get_descriptives_matrix() const;

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

   // Set methods

   void set();
   void set(const size_t&);
   void set(const Vector<size_t>&);
   void set(const Vector<Descriptives>&);
   void set(const tinyxml2::XMLDocument&);
   void set(const ScalingLayer&);

   void set(const Vector<bool>&);

   void set_inputs_number(const size_t&);
   void set_neurons_number(const size_t&);

   void set_default();

   // Descriptives

   void set_descriptives(const Vector<Descriptives>&);
   void set_descriptives_eigen(const Eigen::MatrixXd&);
   void set_item_descriptives(const size_t&, const Descriptives&);

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

   void grow_neuron(const Descriptives& new_descriptives = Descriptives());

   void prune_neuron(const size_t&);

   // Check methods

   bool is_empty() const;

   void check_range(const Vector<double>&) const;

   Tensor<double> calculate_outputs(const Tensor<double>&);

   Tensor<double> calculate_minimum_maximum_outputs(const Tensor<double>&) const;

   Tensor<double> calculate_mean_standard_deviation_outputs(const Tensor<double>&) const;

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

protected:

   Vector<size_t> inputs_dimensions;

   /// Descriptives of input variables.

   Vector<Descriptives> descriptives;

   /// Vector of scaling methods for each variable.

   Vector<ScalingMethod> scaling_methods;

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

