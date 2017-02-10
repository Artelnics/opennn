/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   R A N D O M   S E A R C H   C L A S S   H E A D E R                                                        */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __RANDOMSEARCH_H__
#define __RANDOMSEARCH_H__

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "loss_index.h"

#include "training_algorithm.h"
 
// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

///
/// This concrete class represents a random search training algorithm for a loss functional of a neural network.
///

class RandomSearch : public TrainingAlgorithm
{

public:

   // DEFAULT CONSTRUCTOR

   explicit RandomSearch(void); 


   // PERFORMANCE FUNCTIONAL CONSTRUCTOR

   explicit RandomSearch(LossIndex*); 


   // XML CONSTRUCTOR

   explicit RandomSearch(const tinyxml2::XMLDocument&); 


   // DESTRUCTOR

   virtual ~RandomSearch(void);

   // STRUCTURES

   ///
   /// This structure contains the training results for the random search. 
   ///

   struct RandomSearchResults : public TrainingAlgorithm::TrainingAlgorithmResults
   {  
       /// Default constructor.

       RandomSearchResults(void)
       {
           random_search_pointer = NULL;
       }

       /// Random search constructor.

       RandomSearchResults(RandomSearch* new_random_search_pointer)
       {
           random_search_pointer = new_random_search_pointer;
       }

       /// Destructor.

       virtual ~RandomSearchResults(void)
       {
       }

        /// Pointer to the random search object for which the training results are to be stored.

      RandomSearch* random_search_pointer;


      // TRAINING HISTORY

      /// History of the neural network parameters over the training iterations. 

      Vector< Vector<double> > parameters_history;

      /// History of the parameters norm over the training iterations. 

      Vector<double> parameters_norm_history;

      /// History of the loss function loss over the training iterations. 

      Vector<double> loss_history;

      /// History of the selection loss over the training iterations.

      Vector<double> selection_loss_history;

      /// History of the random search training direction over the training iterations. 

      Vector< Vector<double> > training_direction_history;

      /// History of the norm of the training direction over the training iterations.

      Vector<double> training_direction_norm_history;

      /// History of the random search training rate over the training iterations. 

      Vector<double> training_rate_history;

      /// History of the elapsed time over the training iterations. 

      Vector<double> elapsed_time_history;

      // FINAL VALUES

      /// Final neural network parameters vector. 

      Vector<double> final_parameters;

      /// Final neural network parameters norm. 

      double final_parameters_norm;

      /// Final loss function evaluation.

      double final_loss;

      /// Final selection loss.

      double final_selection_loss;

      /// Final random search training direction. 

      Vector<double> final_training_direction;

      /// Final random search training rate. 

      double final_training_rate;

      /// Elapsed time of the training process. 

      double elapsed_time;

      /// Maximum number of training iterations.

      size_t iterations_number;

      // METHODS

      void resize_training_history(const size_t&);

      std::string to_string(void) const;

      Matrix<std::string> write_final_results(const size_t& precision = 3) const;
  };


   // METHODS

   // Get methods

   const double& get_training_rate_reduction_factor(void) const;
   const size_t& get_training_rate_reduction_period(void) const;

   const bool& get_reserve_parameters_history(void) const;
   const bool& get_reserve_parameters_norm_history(void) const;

   const bool& get_reserve_loss_history(void) const;

   // Training parameters

   const double& get_warning_parameters_norm(void) const;
   const double& get_warning_training_rate(void) const;

   const double& get_error_parameters_norm(void) const;
   const double& get_error_training_rate(void) const;

   // Stopping criteria

   const double& get_loss_goal(void) const;
   const size_t& get_maximum_selection_loss_decreases(void) const;

   const size_t& get_maximum_iterations_number(void) const;
   const double& get_maximum_time(void) const;

   // Reserve training history

   const bool& get_reserve_selection_loss_history(void) const;

   const bool& get_reserve_training_direction_history(void) const;
   const bool& get_reserve_training_rate_history(void) const;
   const bool& get_reserve_elapsed_time_history(void) const;

   // Set methods

   void set_default(void);

   void set_first_training_rate(const double&);
   void set_training_rate_reduction_factor(const double&);
   void set_training_rate_reduction_period(const size_t&);

   void set_reserve_parameters_history(const bool&);
   void set_reserve_parameters_norm_history(const bool&);

   void set_reserve_loss_history(const bool&);

   // Training parameters

   void set_warning_parameters_norm(const double&);
   void set_warning_training_rate(const double&);

   void set_error_parameters_norm(const double&);
   void set_error_training_rate(const double&);

   // Stopping criteria

   void set_loss_goal(const double&);
   void set_maximum_selection_loss_decreases(const size_t&);

   void set_maximum_iterations_number(const size_t&);
   void set_maximum_time(const double&);

   // Reserve training history

   void set_reserve_selection_loss_history(const bool&);

   void set_reserve_training_direction_history(const bool&);
   void set_reserve_training_direction_norm_history(const bool&);
   void set_reserve_training_rate_history(const bool&);
   void set_reserve_elapsed_time_history(const bool&);

   /// Makes the training history of all variables to be reseved or not in memory.

   virtual void set_reserve_all_training_history(const bool&);

   // Utilities

   void set_display_period(const size_t&);

   // Training methods

   Vector<double> calculate_training_direction(void) const;

   RandomSearchResults* perform_training(void);

   std::string write_training_algorithm_type(void) const;

   // Serialization methods

   Matrix<std::string> to_string_matrix(void) const;

   tinyxml2::XMLDocument* to_XML(void) const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML(   );


private: 

   // MEMBERS

   // TRAINING PARAMETERS

   /// Initial training rate following a random training direction.

   double first_training_rate;

   /// Factor which reduces the training rate.
   /// It must be greater than zero and less than one.

   double training_rate_reduction_factor;

   /// Iterations interval at which the training rate is reduced.

   size_t training_rate_reduction_period;

   /// Value for the parameters norm at which a warning message is written to the screen. 

   double warning_parameters_norm;

   /// Training rate value at wich a warning message is written to the screen.

   double warning_training_rate;

   /// Value for the parameters norm at which the training process is assumed to fail. 
   
   double error_parameters_norm;

   /// Training rate at wich the line minimization algorithm is assumed to be unable to bracket a minimum.

   double error_training_rate;


   // STOPPING CRITERIA

   /// Goal value for the loss. It is used as a stopping criterion.

   double loss_goal;

   /// Maximum number of iterations at which the selection loss increases.
   /// This is an early stopping method for improving selection.

   size_t maximum_selection_loss_decreases;

   /// Maximum number of iterations to perform_training. It is used as a stopping criterion.

   size_t maximum_iterations_number;

   /// Maximum training time. It is used as a stopping criterion.

   double maximum_time;


   // TRAINING HISTORY

   /// True if the parameters history vector of vectors is to be reserved, false otherwise.

   bool reserve_parameters_history;

   /// True if the parameters norm history vector is to be reserved, false otherwise.

   bool reserve_parameters_norm_history;

   /// True if the loss history vector is to be reserved, false otherwise.

   bool reserve_loss_history; 

   /// True if the selection loss history vector is to be reserved, false otherwise.

   bool reserve_selection_loss_history;

   /// True if the training direction history matrix is to be reserved, false otherwise.
   
   bool reserve_training_direction_history;

   /// True if the training direction norm history vector is to be reserved, false otherwise.

   bool reserve_training_direction_norm_history;

   /// True if the training rate history vector is to be reserved, false otherwise.

   bool reserve_training_rate_history;

   /// True if the elapsed time history vector is to be reserved, false otherwise.

   bool reserve_elapsed_time_history;
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
