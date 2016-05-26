/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M A T H E M A T I C A L   M O D E L   C L A S S   H E A D E R                                              */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __MATHEMATICALMODEL_H__
#define __MATHEMATICALMODEL_H__

// System includes

#include<fstream>
#include<iostream>
#include<string>
#include<sstream>
#include <time.h>

// OpenNN includes

#include "neural_network.h"

#include "vector.h"
#include "matrix.h"

namespace OpenNN
{

/// 
/// This class represents the concept of mathematical model.
/// A mathematical model is the base for learning in some types of problems, such as optimal control and inverse problems. 
/// 

class MathematicalModel
{

public:

   // DEFAULT CONSTRUCTOR

   explicit MathematicalModel(void);

   // XML CONSTRUCTOR

   explicit MathematicalModel(const tinyxml2::XMLDocument&);

   // FILE CONSTRUCTOR

   explicit MathematicalModel(const std::string&);

   // COPY CONSTRUCTOR

   MathematicalModel(const MathematicalModel&);

   // DESTRUCTOR

   virtual ~MathematicalModel(void);

   // ASSIGNMENT OPERATOR

   virtual MathematicalModel& operator = (const MathematicalModel&);

   // EQUAL TO OPERATOR

   virtual bool operator == (const MathematicalModel&) const;

   // METHODS

   // Get methods

   const size_t& get_independent_variables_number(void) const;
   const size_t& get_dependent_variables_number(void) const;

   size_t count_variables_number(void) const;

   const bool& get_display(void) const;

   // Set methods

   void set(const MathematicalModel&);

   void set_independent_variables_number(const size_t&);
   void set_dependent_variables_number(const size_t&);

   void set_display(const bool&);

   virtual void set_default(void);

   // Mathematical model

   virtual Matrix<double> calculate_solutions(const NeuralNetwork&) const;

   virtual Vector<double> calculate_final_solutions(const NeuralNetwork&) const;

   virtual Matrix<double> calculate_dependent_variables(const NeuralNetwork&, const Matrix<double>&) const;  


   // Serialization methods

   virtual std::string to_string(void) const;

   void print(void) const;

   virtual tinyxml2::XMLDocument* to_XML(void) const;   
   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;
   // virtual void read_XML(   );

   void save(const std::string&) const;
   void load(const std::string&);

   virtual void save_data(const NeuralNetwork&, const std::string&) const;

protected: 

   /// Number of independent variables defining the mathematical model. 

   size_t independent_variables_number;

   /// Number of dependent variables defining the mathematical model. 

   size_t dependent_variables_number;

   /// Flag for displaying warnings. 

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
