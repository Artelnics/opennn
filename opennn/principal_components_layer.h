/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P R I N C I P A L   C O M P O N E N T S   L A Y E R   C L A S S   H E A D E R                              */
/*                                                                                                              */
/*   Pablo Martin                                                                                               */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   pablomartin@artelnics.com                                                                                  */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __PrincipalComponentsLayer_H__
#define __PrincipalComponentsLayer_H__

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

/// This class represents the layer of principal component analysis.
/// This layer is used to reduce the dimension of a dataset.

class PrincipalComponentsLayer
{

public:

   // DEFAULT CONSTRUCTOR

   explicit PrincipalComponentsLayer(void);

   // INPUTS AND PRINCIPAL COMPONENTS NUMBER CONSTRUCTOR

   explicit PrincipalComponentsLayer(const size_t&, const size_t&);

   // COPY CONSTRUCTOR

   PrincipalComponentsLayer(const PrincipalComponentsLayer&);

   // DESTRUCTOR

   virtual ~PrincipalComponentsLayer(void);

   // ENUMERATIONS

   /// Enumeration of available methods for apply the principal components layer.

   enum PrincipalComponentsMethod{NoPrincipalComponents, PrincipalComponents};

   // Principal components state methods

   const PrincipalComponentsMethod& get_principal_components_method(void) const;

   std::string write_principal_components_method(void) const;
   std::string write_principal_components_method_text(void) const;

   // GET METHODS

   Matrix<double> get_principal_components(void) const;
   Vector<double> get_means(void) const;

   Vector<double> get_explained_variance(void) const;

   size_t get_inputs_number(void) const;
   size_t get_principal_components_number(void) const;

   // Inputs principal components function

//    Vector<double> calculate_ouputs(const Vector<double>&) const;

   // Display messages

   const bool& get_display(void) const;

   // SET METHODS

   void set(void);
   void set(const size_t&, const size_t&);
   void set(const PrincipalComponentsLayer&);

   void set_inputs_number(const size_t&);
   void set_principal_components_number(const size_t&);

   void set_principal_component(const size_t&, const Vector<double>&);
   void set_principal_components(const Matrix<double>&);

   void set_means(const Vector<double>&);
   void set_means(const size_t&, const double&);

   void set_explained_variance(const Vector<double>&);

   virtual void set_default(void);

   void set_principal_components_method(const PrincipalComponentsMethod&);
   void set_principal_components_method(const std::string&);

   // Display messages

   void set_display(const bool&);

   // Check methods

   // Inputs principal components function

   Vector<double> calculate_outputs(const Vector<double>&) const;
   Matrix<double> calculate_Jacobian(const Vector<double>&) const;

   // Expression methods

   std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const;

   std::string write_no_principal_components_expression(const Vector<std::string>&, const Vector<std::string>&) const;
   std::string write_principal_components_expression(const Vector<std::string>&, const Vector<std::string>&) const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML(void) const;
   virtual void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

protected:

   // MEMBERS

   /// Inputs number

   size_t inputs_number;

   /// Principal components number

   size_t principal_components_number;

   /// Means of the input variables

   Vector<double> means;

   /// Contains all the principal components arranged in rows and sorted
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
