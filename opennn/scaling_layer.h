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
#include "scaling.h"
#include "opennn_strings.h"

namespace opennn
{

/// This class represents a layer of scaling neurons.

///
/// Scaling layers are included in the definition of a neural network. 
/// They are used to normalize variables so they are in an appropriate range for computer processing.  

class ScalingLayer : public Layer
{

public:

   enum class ProjectType{Approximation, Classification, Forecasting, ImageClassification, TextClassification};

   // Constructors

   explicit ScalingLayer();

   explicit ScalingLayer(const Index&);
   explicit ScalingLayer(const Tensor<Index, 1>&);

   explicit ScalingLayer(const Tensor<Descriptives, 1>&);

   // Get methods
   
   ProjectType get_project_type() const;
   string get_project_type_string(const ScalingLayer::ProjectType&) const;
   Tensor<Index, 1> get_outputs_dimensions() const;

   Index get_inputs_number() const final;
   Index get_neurons_number() const final;

   // Inputs descriptives

   Tensor<Descriptives, 1> get_descriptives() const;
   Descriptives get_descriptives(const Index&) const;

   Tensor<type, 1> get_minimums() const;
   Tensor<type, 1> get_maximums() const;
   Tensor<type, 1> get_means() const;
   Tensor<type, 1> get_standard_deviations() const;

   // Variables scaling and unscaling

   Tensor<Scaler, 1> get_scaling_methods() const;

   Tensor<string, 1> write_scalers() const;
   Tensor<string, 1> write_scalers_text() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&);
   void set(const Tensor<Index, 1>&);
   void set(const Tensor<Descriptives, 1>&);
   void set(const Tensor<Descriptives, 1>&, const Tensor<Scaler, 1>&);
   void set(const tinyxml2::XMLDocument&);

   void set_project_type(const ProjectType&);
   void set_project_type_string(const string&);

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;

   void set_default();

   // Descriptives

   void set_descriptives(const Tensor<Descriptives, 1>&);
   void set_item_descriptives(const Index&, const Descriptives&);

   void set_minimum(const Index&, const type&);
   void set_maximum(const Index&, const type&);
   void set_mean(const Index&, const type&);
   void set_standard_deviation(const Index&, const type&);

   void set_min_max_range(const type& min, const type& max);

   // Scaling method

   void set_scalers(const Tensor<Scaler, 1>&);
   void set_scalers(const Tensor<string, 1>&);

   void set_scaler(const Index&, const Scaler&);
   void set_scalers(const Scaler&);
   void set_scalers(const string&);

   // Display messages

   void set_display(const bool&);

   // Check methods

   bool is_empty() const;

   void check_range(const Tensor<type, 1>&) const;

   void calculate_outputs(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) final;

   // Expression methods

   string write_no_scaling_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_minimum_maximum_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_mean_standard_deviation_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_standard_deviation_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

   string write_expression_c() const final;

   string write_expression_python() const final;

   // Serialization methods
   
   void print() const;

   virtual void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

   ScalingLayer::ProjectType project_type;

   Tensor<Index, 1> input_variables_dimensions;

   /// Descriptives of input variables.

   Tensor<Descriptives, 1> descriptives;

   /// Vector of scaling methods for each variable.

   Tensor<Scaler, 1> scalers;

   /// min and max range for minmaxscaling

   type min_range;
   type max_range;

   /// Display warning messages to screen. 

   bool display = true;

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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

