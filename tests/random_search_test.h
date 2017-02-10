/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   R A N D O M   S E A R C H   T E S T   C L A S S   H E A D E R                                              */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __RANDOMSEARCHTEST_H__
#define __RANDOMSEARCHTEST_H__

// Unit testing includes

#include "unit_testing.h"


using namespace OpenNN;


class RandomSearchTest : public UnitTesting
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // GENERAL CONSTRUCTOR

   explicit RandomSearchTest(void); 


   // DESTRUCTOR

   virtual ~RandomSearchTest(void);


   // METHODS

   // Constructor and destructor methods

   void test_constructor(void);
   void test_destructor(void); 

   // Get methods

   void test_get_training_rate_reduction_factor(void);

   void test_get_reserve_parameters_history(void);
   void test_get_reserve_parameters_norm_history(void);

   void test_get_reserve_loss_history(void);

   // Set methods

   void test_set_training_rate_reduction_factor(void);

   void test_set_reserve_parameters_history(void);
   void test_set_reserve_parameters_norm_history(void);

   void test_set_reserve_loss_history(void);

   // Training methods

   void test_calculate_training_direction(void);

   void test_perform_training(void);

   // Training history methods

   void test_set_reserve_all_training_history(void);

   // Utiltity methods

   void test_to_XML(void);

   void test_from_XML(void);

   // Unit testing methods

   void run_test_case(void);

};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2016 Roberto Lopez.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later versi
