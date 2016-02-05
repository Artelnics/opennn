/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   F I N A L   S O L U T I O N S   E R R O R   C L A S S   H E A D E R                                        */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __FINALSOLUTIONSERROR_H__
#define __FINALSOLUTIONSERROR_H__

// System includes

#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <string>
#include <limits>

// OpenNN includes

#include "mathematical_model.h"

#include "performance_term.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

/// This class represents the final solutions error constraints functional. 
/// The final solutions error is measured on a mathematical model. 
/// This constraints functional often appears in optimal control problems.
   
class FinalSolutionsError : public PerformanceTerm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit FinalSolutionsError(void);

   // NEURAL NETWORK CONSTRUCTOR

   explicit FinalSolutionsError(NeuralNetwork*);

   // MATHEMATICAL MODEL CONSTRUCTOR

   explicit FinalSolutionsError(MathematicalModel*);

   // NEURAL NETWORK AND MATHEMATICAL MODEL CONSTRUCTOR

   explicit FinalSolutionsError(NeuralNetwork*, MathematicalModel*);

   // XML CONSTRUCTOR

   explicit FinalSolutionsError(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   FinalSolutionsError(const FinalSolutionsError&);

   // DESTRUCTOR

   virtual ~FinalSolutionsError(void);

   // ASSIGNMENT OPERATOR

   FinalSolutionsError& operator = (const FinalSolutionsError&);

   // EQUAL TO OPERATOR

   bool operator == (const FinalSolutionsError&) const;

   // METHODS

   // Get methods

   const Vector<double>& get_final_solutions_errors_weights(void) const;

   const Vector<double>& get_target_final_solutions(void) const;

   // Set methods

   void set(void);

   void set(NeuralNetwork*);
   void set(MathematicalModel*);
   void set(NeuralNetwork*, MathematicalModel*);

//   void set(const FinalSolutionsError&);

   void set_mathematical_model_pointer(MathematicalModel*);

   void set_final_solutions_errors_weights(const Vector<double>&);
   void set_final_solution_error_weight(const size_t&, const double&);

   void set_target_final_solutions(const Vector<double>&);
   void set_target_final_solution(const size_t&, const double&);

   void set_default(void);

   // Checking methods

   void check(void) const;

   // Performance term methods

   double calculate_performance(void) const;
   double calculate_performance(const Vector<double>&) const;

   std::string write_performance_term_type(void) const;

   std::string write_information(void) const;
   
   // Serialization methods

   tinyxml2::XMLDocument* to_XML(void) const;   
   void from_XML(const tinyxml2::XMLDocument&);

private:

   // MEMBERS

   /// Desired final values for the dependent variables.
   /// The size of this vector must be equal to the number of dependent variablese in the mathematical model. 

   Vector<double> target_final_solutions;

   /// Weight value for each error on the final solution. 
   /// The size of this vector must be equal to the number of dependent variablese in the mathematical model. 

   Vector<double> final_solutions_errors_weights;

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
