//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R I N C I P A L   C O M P O N E N T S   L A Y E R   C L A S S   H E A D E R  
//
//   Pablo Martin                                                          
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                             

#ifndef PRINCIPALCOMPONENTSLAYER_H
#define PRINCIPALCOMPONENTSLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"

#include "layer.h"

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

   explicit PrincipalComponentsLayer(const Index&, const Index&);

   virtual ~PrincipalComponentsLayer();

   // Enumerations

   /// Enumeration of available methods for apply the principal components layer.

   enum PrincipalComponentsMethod{NoPrincipalComponents, PrincipalComponents};

   // Principal components state methods

   const PrincipalComponentsMethod& get_principal_components_method() const;

   string write_principal_components_method() const;
   string write_principal_components_method_text() const;

   // Get methods

   Tensor<type, 2> get_principal_components() const;
   Tensor<type, 1> get_means() const;

   Tensor<type, 1> get_explained_variance() const;

   Index get_inputs_number() const;
   Index get_principal_components_number() const;
   Index get_neurons_number() const;


   // Inputs principal components function

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&);
   void set(const PrincipalComponentsLayer&);

   void set_inputs_number(const Index&);
   void set_principal_components_number(const Index&);

   void set_principal_component(const Index&, const Tensor<type, 1>&);
   void set_principal_components(const Tensor<type, 2>&);

   void set_means(const Tensor<type, 1>&);
   void set_means(const Index&, const type&);

   void set_explained_variance(const Tensor<type, 1>&);

   virtual void set_default();

   void set_principal_components_method(const PrincipalComponentsMethod&);
   void set_principal_components_method(const string&);

   // Display messages

   void set_display(const bool&);

   // Check methods

   // Inputs principal components function

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

   // Expression methods

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_no_principal_components_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_principal_components_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   // Serialization methods

   
   virtual void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   

protected:

   // MEMBERS

   /// Inputs number

   Index inputs_number;

   /// Principal components number

   Index principal_components_number;

   /// Means of the input variables

   Tensor<type, 1> means;

   /// Contains all the principal components getd in rows and sorted
   /// according to their relative explained variance.

   Tensor<type, 2> principal_components;

   /// Explained variances for every of the principal components

   Tensor<type, 1> explained_variance;

   /// Principal components layer method

   PrincipalComponentsMethod principal_components_method;

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
