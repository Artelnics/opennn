/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M O C K   T R A I N I N G   A L G O R I T H M   C L A S S   H E A D E R                                    */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __MOCKTRAININGALGORITHM_H__
#define __MOCKTRAININGALGORITHM_H__

// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;


class MockTrainingAlgorithm : public TrainingAlgorithm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit MockTrainingAlgorithm(); 


   // GENERAL CONSTRUCTOR

   explicit MockTrainingAlgorithm(LossIndex*); 


   // DESTRUCTOR

   virtual ~MockTrainingAlgorithm();


   // STRUCTURES 

   struct MockTrainingAlgorithmResults : public TrainingAlgorithmResults
   {
   };


   // METHODS

   // Training methods

   MockTrainingAlgorithmResults* perform_training();


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
