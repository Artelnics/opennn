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

   // INPUTS NUMBER CONSTRUCTOR

   explicit PrincipalComponentsLayer(const size_t&);

   // COPY CONSTRUCTOR

   PrincipalComponentsLayer(const PrincipalComponentsLayer&);

   // DESTRUCTOR

   virtual ~PrincipalComponentsLayer(void);

   // ENUMERATIONS

   enum PrincipalComponentsMethod{NoPrincipalComponents, ActivatedPrincipalComponents};

   // Principal components state methods

   const PrincipalComponentsMethod& get_principal_components_method(void) const;

   std::string write_principal_components_method(void) const;
   std::string write_principal_components_method_text(void) const;

   // GET METHODS

   Matrix<double> get_eigenvectors(void) const;
   Vector<double> get_means(void) const;

   // Inputs principal components function

    Vector<double> calculate_ouputs(const Vector<double>&) const;

   // Display messages

   const bool& get_display(void) const;

   // SET METHODS

   void set(void);
   void set(const size_t&);
   void set(const PrincipalComponentsLayer&);

   void set_eigenvectors(const Matrix<double>&);

   void set_means(const Vector<double>&);
   void set_means(const size_t&, const double&);

   virtual void set_default(void);

   void set_principal_components_method(const PrincipalComponentsMethod&);
   void set_principal_components_method(const std::string&);

   // GET METHODS

   size_t get_principal_components_neurons_number(void) const;

   // Display messages

   void set_display(const bool&);

   // Check methods

   // Inputs principal components function

   Vector<double> calculate_outputs(const Vector<double>&) const;
   Vector<double> calculate_derivatives(const Vector<double>&) const;

   // Serialization methods

   std::string to_string(void) const;

   tinyxml2::XMLDocument* to_XML(void) const;
   virtual void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

protected:

   // MEMBERS

   /// Means of the input variables

   Vector<double> means;

   /// Eigenvectors of the variables in columns.

   Matrix<double> eigenvectors;

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
