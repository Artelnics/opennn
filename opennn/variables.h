/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   V A R I A B L E S   C L A S S   H E A D E R                                                                */
/*                                                                                                              */ 

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __VARIABLES_H__
#define __VARIABLES_H__

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

#include<limits>
#include<climits>

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

/// This class is used to store information about the variables of a data set. 
/// Variables in a data set can be used as inputs and targets.
/// This class also stores information about the name, unit and description of all the variables.

class Variables
{

public:  

   // DEFAULT CONSTRUCTOR

   explicit Variables();

   // VARIABLES NUMBER CONSTRUCTOR

   explicit Variables(const size_t&);

   // INPUT AND TARGET VARIABLES NUMBER

   explicit Variables(const size_t&, const size_t&);

   // XML CONSTRUCTOR

   explicit Variables(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   Variables(const Variables&);

   // DESTRUCTOR

   virtual ~Variables();

   // ASSIGNMENT OPERATOR

   Variables& operator = (const Variables&);

   // EQUAL TO OPERATOR

   bool operator == (const Variables&) const;

   // ENUMERATIONS

   /// This enumeration represents the possible uses of a variable(input, target, time or unused).

   enum Use{Input, Target, Time, Unused};

   // STRUCTURES

   ///
   /// This structure contains the information of a single variable.
   ///

   struct Item
   {
       /// Name of a variable.

       string name;

       /// Units of a variable.

       string units;

       /// Description of a variable.

       string description;

       /// Use of a variable(none, input, target or time).

       Use use;
   };

   // METHODS

   const Vector<Item>& get_items() const;

   const Item& get_item(const size_t&) const;

   /// Returns the total number of variables in the data set.

   inline size_t get_variables_number() const
   {
      return(items.size());
   }

   bool empty() const;

   size_t count_used_variables_number() const;
   size_t count_unused_variables_number() const;
   size_t get_inputs_number() const;
   size_t get_targets_number() const;

   Vector<size_t> count_uses() const;

   // Variables methods

   Vector<Use> get_uses() const;
   Vector<string> write_uses() const;

   const Use& get_use(const size_t&) const;
   string write_use(const size_t&) const;

   bool is_input(const size_t&) const;
   bool is_target(const size_t&) const;
   bool is_time(const size_t&) const;
   bool is_unused(const size_t&) const;

   bool is_used(const size_t&) const;

   Vector<size_t> get_used_indices() const;
   Vector<size_t> get_inputs_indices() const;
   Vector<size_t> get_targets_indices() const;
   size_t get_time_index() const;
   Vector<size_t> get_unused_indices() const;  

   Vector<int> get_inputs_indices_int() const;
   Vector<int> get_targets_indices_int() const;
   int get_time_variable_index_int() const;

   // Information methods

   Vector<string> get_names() const;
   Vector<string> get_used_names() const;
   Vector<string> get_used_units() const;
   const string& get_name(const size_t&) const;

   bool has_names() const;

   Vector<string> get_units() const;
   const string& get_unit(const size_t&) const;

   Vector<string> get_descriptions() const;
   const string& get_description(const size_t&) const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const size_t&);
   void set(const size_t&, const size_t&);
   void set(const tinyxml2::XMLDocument&);

   void set_default();

   // Data methods

   void set_variables_number(const size_t&);

   // Variables methods

   void set_items(const Vector<Item>&);

   void set_uses(const Vector<Use>&); 
   void set_uses(const Vector<string>&);

   void set_use(const size_t&, const Use&);
   void set_use(const size_t&, const string&);

   void set_use(const string&, const Use&);

   void set_use_substring(const string&, const Use&);

   void set_input();
   void set_target();
   void set_time();
   void set_unuse();

   void unuse_ahead_variables();
   void unuse_substring_variables(const string&);

   void set_input_indices(const Vector<size_t>&);
   void set_target_indices(const Vector<size_t>&);
   void set_time_index(const size_t&);
   void set_unuse_indices(const Vector<size_t>&);

   void set_default_uses();

   // Information methods

   void set_names(const Vector<string>&);
   void set_name(const size_t&, const string&);

   void set_units(const Vector<string>&);
   void set_units(const size_t&, const string&);

   void set_descriptions(const Vector<string>&);
   void set_description(const size_t&, const string&);

   void set_names(const Vector<string>&, const Vector< Vector<string> >&);

   void set_display(const bool&);

   Matrix<string> get_information() const;

   Vector<string> get_inputs_units() const;
   Vector<string> get_targets_units() const;
   string get_time_unit() const;

   Vector<string> get_inputs_name() const;
   vector<string> get_inputs_name_std() const;

   Vector<string> get_targets_name() const;
   vector<string> get_targets_name_std() const;
   string get_time_name() const;

   Vector<string> get_inputs_description() const;
   Vector<string> get_targets_description() const;
   string get_time_description() const;

   Matrix<string> get_inputs_information() const;
   vector< vector<string> > get_inputs_information_vector_of_vector() const;

   Matrix<string> get_targets_information() const;
   vector< vector<string> > get_targets_information_vector_of_vector() const;
   Vector<string> get_time_information() const;

   size_t get_variable_index(const string&) const;

   void remove_variable(const size_t&);
   void remove_variable(const string&);

   bool has_time() const;

   void convert_time_series(const size_t&, const size_t&, const size_t&);
   void convert_association();

   // Serialization methods

   string object_to_string() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML();

private:

   static string unsigned_to_string(const size_t&);
   static string prepend(const string&, const string&);


   // MEMBERS

   /// Vector of variable items.
   /// Each item contains the name, units, description and use of a single variable.

   Vector<Item> items;

   /// Display messages to screen.
   
   bool display;
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
