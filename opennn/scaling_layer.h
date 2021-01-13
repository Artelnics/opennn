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

#include "layer.h"
#include "statistics.h"
#include "opennn_strings.h"

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

   explicit ScalingLayer(const Index&);
   explicit ScalingLayer(const Tensor<Index, 1>&);

   explicit ScalingLayer(const Tensor<Descriptives, 1>&);

   // Destructors

   virtual ~ScalingLayer();

   /// Enumeration of available methods for scaling the input variables.  
   
   enum ScalingMethod{NoScaling, MinimumMaximum, MeanStandardDeviation, StandardDeviation};

   // Get methods

   
   Tensor<Index, 1> get_outputs_dimensions() const;

   Index get_inputs_number() const;
   Index get_neurons_number() const;

   // Inputs descriptives

   Tensor<Descriptives, 1> get_descriptives() const;
   Descriptives get_descriptives(const Index&) const;

   Tensor<type, 2> get_descriptives_matrix() const;

   Tensor<type, 1> get_minimums() const;
   Tensor<type, 1> get_maximums() const;
   Tensor<type, 1> get_means() const;
   Tensor<type, 1> get_standard_deviations() const;

   // Variables scaling and unscaling

   const Tensor<ScalingMethod, 1> get_scaling_methods() const;

   Tensor<string, 1> write_scaling_methods() const;
   Tensor<string, 1> write_scaling_methods_text() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&);
   void set(const Tensor<Index, 1>&);
   void set(const Tensor<Descriptives, 1>&);
   void set(const tinyxml2::XMLDocument&);
   void set(const ScalingLayer&);

//   void set(const Tensor<bool, 1>&);

   void set_inputs_number(const Index&);
   void set_neurons_number(const Index&);

   void set_default();

   // Descriptives

   void set_descriptives(const Tensor<Descriptives, 1>&);
   void set_descriptives_eigen(const Tensor<type, 2>&);
   void set_item_descriptives(const Index&, const Descriptives&);

   void set_minimum(const Index&, const type&);
   void set_maximum(const Index&, const type&);
   void set_mean(const Index&, const type&);
   void set_standard_deviation(const Index&, const type&);

   void set_min_max_range(const type min, const type max);

   // Scaling method

   void set_scaling_methods(const Tensor<ScalingMethod, 1>&);
   void set_scaling_methods(const Tensor<string, 1>&);

   void set_scaling_methods(const ScalingMethod&);
   void set_scaling_methods(const string&);

   // Display messages

   void set_display(const bool&);

   // Check methods

   bool is_empty() const;

   void check_range(const Tensor<type, 1>&) const;

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

   // Expression methods

   string write_no_scaling_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_minimum_maximum_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_mean_standard_deviation_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_standard_deviation_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_expression_c() const;

   string write_expression_python() const;

   // Serialization methods
   
   virtual void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

protected:

   Tensor<Index, 1> input_variables_dimensions;

   /// Descriptives of input variables.

   Tensor<Descriptives, 1> descriptives;

   /// Vector of scaling methods for each variable.

   Tensor<ScalingMethod, 1> scaling_methods;

   /// min and max range for minmaxscaling

   type min_range;
   type max_range;

   /// Display warning messages to screen. 

   bool display = true;

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

