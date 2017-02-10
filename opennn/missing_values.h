/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M I S S I N G   V A L U E S   C L A S S   H E A D E R                                                      */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __MISSINGVALUES_H__
#define __MISSINGVALUES_H__

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

/// Missing values are those elements which are not present in the data set."
/// Therefore, they represent a lack of information about several variables in several instances.;
/// This class is used to store information about the missing values of a data set.

class MissingValues
{

public:  

   // DEFAULT CONSTRUCTOR

   explicit MissingValues(void);

   // MISSING VALUES NUMBER CONSTRUCTOR

   explicit MissingValues(const size_t&, const size_t&);

   // XML CONSTRUCTOR

   explicit MissingValues(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   MissingValues(const MissingValues&);


   // DESTRUCTOR

   virtual ~MissingValues(void);

   // ASSIGNMENT OPERATOR

   MissingValues& operator = (const MissingValues&);

   // EQUAL TO OPERATOR

   bool operator == (const MissingValues&) const;


   // ENUMERATIONS

   /// Enumeration of available activation functions for the perceptron neuron model.

   enum ScrubbingMethod{Unuse, Mean, NoScrubbing};

   // STRUCTURES

   ///
   /// This structure contains the information of a single missing value,
   /// which is defined by its instance and variable indices.
   ///

   struct Item
   {
       /// Default constructor.

       Item(void)
       {
       }

       /// Indices constructor.

       Item(const size_t& new_instance_index, const size_t& new_variable_index)
       {
           //std::cout << instances_number;


           instance_index = new_instance_index;
           variable_index = new_variable_index;
       }

       /// Destructor.

       virtual ~Item(void)
       {
       }

       /// Returns true if this item is equal to another item.
       /// other_item Other item to be compared with.

       bool operator == (const Item& other_item) const
       {
            if(other_item.instance_index == instance_index && other_item.variable_index == variable_index)
            {
                return(true);
            }
            else
            {
                return(false);
            }
       }

       /// Row index for missing value.

       size_t instance_index;

       /// Column index for missing value.

       size_t variable_index;
   };


   // METHODS

   size_t get_instances_number(void) const;

   size_t get_variables_number(void) const;


   /// Returns the number of missing values in the data set.

   inline size_t get_missing_values_number(void) const
   {
      return(items.size());
   }

   Vector<size_t> get_missing_values_numbers(void) const;

   const Vector<Item>& get_items(void) const;
   const Item& get_item(const size_t&) const;

   ScrubbingMethod get_scrubbing_method(void) const;

   std::string write_scrubbing_method(void) const;
   std::string write_scrubbing_method_text(void) const;

   const bool& get_display(void) const;

   // Set methods

   void set(void);
   void set(const size_t&, const size_t&);
   void set(const tinyxml2::XMLDocument&);

   void set_instances_number(const size_t&);
   void set_variables_number(const size_t&);

   void set_default(void);

   void set_missing_values_number(const size_t&);

   void set_scrubbing_method(const ScrubbingMethod&);
   void set_scrubbing_method(const std::string&);

   // Missing values methods

   void set_display(const bool&);

   void set_items(const Vector<Item>&);
   void set_item(const size_t&, const size_t&, const size_t&);

   void append(const size_t&, const size_t&);

   bool has_missing_values(void) const;
   bool has_missing_values(const size_t&) const;
   bool has_missing_values(const size_t&, const Vector<size_t>&) const;

   bool is_missing_value(const size_t&, const size_t&) const;

   Vector<size_t> arrange_missing_instances(void) const;

   size_t count_missing_instances(void) const;

   Vector<size_t> arrange_missing_variables(void) const;

   Vector< Vector<size_t> > arrange_missing_indices(void) const;

   void convert_time_series(const size_t&);
   void convert_association(void);

   // Serialization methods

   std::string to_string(void) const;

   void print(void) const;

   tinyxml2::XMLDocument* to_XML(void) const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

private:

   // MEMBERS

   /// Number of instances.

   size_t instances_number;

   /// Number of variables.

   size_t variables_number;

   /// Method for handling missing values.

   ScrubbingMethod scrubbing_method;

   /// Missing values.

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
