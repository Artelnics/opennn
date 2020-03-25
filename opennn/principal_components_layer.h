//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R I N C I P A L   C O M P O N E N T S   L A Y E R   C L A S S   H E A D E R  
//
//   Pablo Martin                                                          
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                             



#ifndef PrincipalComponentsLayer_H
#define PrincipalComponentsLayer_H

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
#include "metrics.h"
#include "layer.h"



#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the layer of principal component analysis.

///
/// This layer is used to reduce the dimension of a dataset.

class PrincipalComponentsLayer : public Layer
{

public:

   // DEFAULT CONSTRUCTOR

   explicit PrincipalComponentsLayer();

   // INPUTS AND PRINCIPAL COMPONENTS NUMBER CONSTRUCTOR

   explicit PrincipalComponentsLayer(const size_t&, const size_t&);

   // COPY CONSTRUCTOR

   PrincipalComponentsLayer(const PrincipalComponentsLayer&);

   

   virtual ~PrincipalComponentsLayer();

   // Enumerations

   /// Enumeration of available methods for apply the principal components layer.

   enum PrincipalComponentsMethod{NoPrincipalComponents, PrincipalComponents};

   // Principal components state methods

   const PrincipalComponentsMethod& get_principal_components_method() const;

   string write_principal_components_method() const;
   string write_principal_components_method_text() const;

   // Get methods

   Matrix<double> get_principal_components() const;
   Vector<double> get_means() const;

   Vector<double> get_explained_variance() const;

   size_t get_inputs_number() const;
   size_t get_principal_components_number() const;
   size_t get_neurons_number() const;


   // Inputs principal components function

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const size_t&, const size_t&);
   void set(const PrincipalComponentsLayer&);

   void set_inputs_number(const size_t&);
   void set_principal_components_number(const size_t&);

   void set_principal_component(const size_t&, const Vector<double>&);
   void set_principal_components(const Matrix<double>&);

   void set_means(const Vector<double>&);
   void set_means(const size_t&, const double&);

   void set_explained_variance(const Vector<double>&);

   virtual void set_default();

   void set_principal_components_method(const PrincipalComponentsMethod&);
   void set_principal_components_method(const string&);

   // Display messages

   void set_display(const bool&);

   // Check methods

   // Inputs principal components function

   Tensor<double> calculate_outputs(const Tensor<double>&);

   // Expression methods

   string write_expression(const Vector<string>&, const Vector<string>&) const;

   string write_no_principal_components_expression(const Vector<string>&, const Vector<string>&) const;
   string write_principal_components_expression(const Vector<string>&, const Vector<string>&) const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;
   virtual void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   

protected:

   // MEMBERS

   /// Inputs number

   size_t inputs_number;

   /// Principal components number

   size_t principal_components_number;

   /// Means of the input variables

   Vector<double> means;

   /// Contains all the principal components getd in rows and sorted
   /// according to their relative explained variance.

   Matrix<double> principal_components;

   /// Explained variances for every of the principal components

   Vector<double> explained_variance;

   /// Principal components layer method

   PrincipalComponentsMethod principal_components_method;

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
