/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M O C K   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S   H E A D E                              */
/*                                                                                                              */  
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __MOCKTRAININGALGORITHM_H__
#define __MOCKTRAININGALGORITHM_H__

// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;


class MockOptimizationAlgorithm : public OptimizationAlgorithm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit MockOptimizationAlgorithm(); 


   // GENERAL CONSTRUCTOR

   explicit MockOptimizationAlgorithm(LossIndex*); 


   // DESTRUCTOR

   virtual ~MockOptimizationAlgorithm();


   // STRUCTURES 

   struct MockOptimizationAlgorithmResults : public OptimizationAlgorithmResults
   {
   };


   // METHODS

   // Training methods

   MockOptimizationAlgorithmResults* perform_training();


};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the s of the GNU Lesser General Public
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
