/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N S T A N C E S   C L A S S   H E A D E R                                                                */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __INSTANCES_H__
#define __INSTANCES_H__

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <exception>

// OpenNN includes

#include "vector.h"
#include "matrix.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"


namespace OpenNN
{

/// This class is used to store information about the instances of a data set. 
/// Instances in a data set can be used for training, selection and testing.    

class Instances
{

public:  

   // DEFAULT CONSTRUCTOR

   explicit Instances(void);

   // INSTANCES NUMBER CONSTRUCTOR

   explicit Instances(const size_t&);

   // XML CONSTRUCTOR

   explicit Instances(const tinyxml2::XMLDocument&);


   // COPY CONSTRUCTOR

   Instances(const Instances&);


   // DESTRUCTOR

   virtual ~Instances(void);

   // ASSIGNMENT OPERATOR

   Instances& operator = (const Instances&);

   // EQUAL TO OPERATOR

   bool operator == (const Instances&) const;

   // ENUMERATIONS

   /// This enumeration represents the possible uses of an instance
   /// (no use, training, selection or testing).

   enum Use{Training, Selection, Testing, Unused};

   /// This is an enumeration of the available methods for dividing the instances
   /// into training, selection and testing subsets.

   enum SplittingMethod{Sequential, Random};

   // STRUCTURES

   ///
   /// This structure contains the information of a single instance,
   /// which is only its use (training, selection, testing or unused).
   ///

   struct Item
   {
       /// Default constructor.

       Item(void)
       {
           use = Training;
       }

       /// Use constructor.

       Item(const Use& new_use)
       {
           use = new_use;
       }

       /// Destructor.

       virtual ~Item(void)
       {
       }

       /// Use of an instance (training, selection, testing or unused).

       Use use;
   };


   // METHODS

   static SplittingMethod get_splitting_method(const std::string&);

   /// Returns the number of instances in the data set.

   inline size_t get_instances_number(void) const
   {
      return(items.size());
   }

   bool empty(void) const;

   // Instances methods

   Vector<Use> arrange_uses(void) const;
   Vector<std::string> write_uses(void) const;
   Vector<std::string> write_abbreviated_uses(void) const;

   const Use& get_use(const size_t&) const;
   std::string write_use(const size_t&) const;

   bool is_used(const size_t&) const;
   bool is_unused(const size_t&) const;

   size_t count_training_instances_number(void) const;
   size_t count_selection_instances_number(void) const;
   size_t count_testing_instances_number(void) const;
   size_t count_unused_instances_number(void) const;
   size_t count_used_instances_number(void) const;

   Vector<size_t> count_uses(void) const;

   Vector<size_t> arrange_used_indices(void)  const;
   Vector<size_t> arrange_unused_indices(void) const;
   Vector<size_t> arrange_training_indices(void) const;
   Vector<size_t> arrange_selection_indices(void) const;
   Vector<size_t> arrange_testing_indices(void) const;

   Vector<int> arrange_training_indices_int(void) const;
   Vector<int> arrange_selection_indices_int(void) const;

   const bool& get_display(void) const;

   // Set methods

   void set(void);
   void set(const size_t&);
   void set(const tinyxml2::XMLDocument&);

   void set_default(void);

   // Data methods

   void set_instances_number(const size_t&);

   // Instances methods

   void set_uses(const Vector<Use>&);
   void set_uses(const Vector<std::string>&);

   void set_use(const size_t&, const Use&);
   void set_use(const size_t&, const std::string&);

   void set_unused(const Vector<size_t>&);
   void set_training(void);
   void set_selection(void);
   void set_testing(void);

   void set_display(const bool&);

   // Splitting methods

   void split_sequential_indices(const double& training_ratio = 0.6, const double& selection_ratio = 0.2, const double& testing_ratio = 0.2);

   void split_random_indices(const double& training_ratio = 0.6, const double& selection_ratio = 0.2, const double& testing_ratio = 0.2);

   void split_instances(const SplittingMethod& splitting_method = Random, const double& training_ratio = 0.6, const double& selection_ratio = 0.2, const double& testing_ratio = 0.2);

   Vector<double> calculate_uses_percentage(void) const;

   void convert_time_series(const size_t&);

   // Serialization methods

   std::string to_string(void) const;

   void print(void) const;

   tinyxml2::XMLDocument* to_XML(void) const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

private:

   // MEMBERS

   /// Uses of instances (none, training, selection or testing).

   Vector<Item> items;

   /// Display messages to screen.
   
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

