/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   R A N D O M   S E A R C H   T E S T   C L A S S   H E A D E R                                              */
/*                                                                                                              */ 
 
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
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

   explicit RandomSearchTest(); 


   // DESTRUCTOR

   virtual ~RandomSearchTest();


   // METHODS

   // Constructor and destructor methods

   void test_constructor();
   void test_destructor(); 

   // Get methods

   void test_get_training_rate_reduction_factor();

   void test_get_reserve_parameters_history();
   void test_get_reserve_parameters_norm_history();

   void test_get_reserve_loss_history();

   // Set methods

   void test_set_training_rate_reduction_factor();

   void test_set_reserve_parameters_history();
   void test_set_reserve_parameters_norm_history();

   void test_set_reserve_loss_history();

   // Training methods

   void test_calculate_training_direction();

   void test_perform_training();

   // Training history methods

   void test_set_reserve_all_training_history();

   // Utiltity methods

   void test_to_XML();

   void test_from_XML();

   // Unit testing methods

   void run_test_case();

};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later versi
