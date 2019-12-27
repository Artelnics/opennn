//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL (Artelnics)                     
//   artelnics@artelnics.com

#ifndef NORMALIZEDSQUAREDERRORTEST_H
#define NORMALIZEDSQUAREDERRORTEST_H

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;


class NormalizedSquaredErrorTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   explicit NormalizedSquaredErrorTest(void); 

   virtual ~NormalizedSquaredErrorTest(void);

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void);

   // Get methods

   // Set methods
 
   // Normalization coefficient

   void test_calculate_training_normalization_coefficient(void);   
   void test_calculate_selection_normalization_coefficient(void);   

   // Error methods

   void test_calculate_training_error(void);

   void test_calculate_training_error_gradient(void);

   // Error terms methods

   void test_calculate_training_error_terms(void);

   void test_calculate_training_error_terms_Jacobian(void);

   // Squared errors methods

   void test_calculate_squared_errors(void);

   void test_calculate_maximal_errors(void);

   // Serialization methods

   void test_to_XML(void);
   void test_from_XML(void);

   // Unit testing methods

   void run_test_case(void);
};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques SL.
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
