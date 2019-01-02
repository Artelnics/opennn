/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N P U T S   C L A S S   H E A D E R                                                                      */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __INPUTS_H__
#define __INPUTS_H__

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "vector.h"
#include "matrix.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

/// This class is used to store some information about the input variables of a neural network.
/// That information basically consists of the names, units and descriptions of the input variables.

class Inputs
{

public:

   // DEFAULT CONSTRUCTOR

   explicit Inputs();

   // INPUTS NUMBER CONSTRUCTOR

   explicit Inputs(const size_t&);

   // XML CONSTRUCTOR

   explicit Inputs(const tinyxml2::XMLDocument&);


   // COPY CONSTRUCTOR

   Inputs(const Inputs&);

   // DESTRUCTOR

   virtual ~Inputs();

   // ASSIGNMENT OPERATOR

   Inputs& operator = (const Inputs&);

   // EQUAL TO OPERATOR

   bool operator == (const Inputs&) const;

   ///
   /// This structure contains the information of a single input.
   ///

   struct Item
   {
       /// Name of output variable.
       string name;

       /// Units of output variable.

       string units;

       /// Description of output variable.

       string description;

       /// Default constructor.

       Item() {}
   };

   // METHOD

   bool is_empty() const;

   /// Returns the number of inputs in the multilayer perceptron

   inline size_t get_inputs_number() const
   {
      return(items.size());
   }

   // Inputs information

   Vector<string> get_names() const;
   const string& get_name(const size_t&) const;

   Vector<string> get_units() const;
   const string& get_unit(const size_t&) const;

   Vector<string> get_descriptions() const;
   const string& get_description(const size_t&) const;

   // Variables

   Matrix<string> get_information() const;

   // Display messages

   const bool& get_display() const;

   // SET METHODS

   void set();
   void set(const size_t&);
   void set(const Vector< Vector<string> >&);
   void set(const Inputs&);

   void set(const Vector<bool>&);

   void set_inputs_number(const size_t&);

   virtual void set_default();

   // Input variables information

   void set_names(const Vector<string>&);
   void set_name(const size_t&, const string&);

   void set_units(const Vector<string>&);
   void set_unit(const size_t&, const string&);

   void set_descriptions(const Vector<string>&);
   void set_description(const size_t&, const string&);

   // Variables

   void set_information(const Matrix<string>&);
   void set_information_vector_of_vector(const vector< vector<string> >&);

   void set_display(const bool&);

   // Growing and pruning

   void grow_input();

   void prune_input(const size_t&);

   // Default names

   Vector<string> write_default_names() const;

   // Serialization methods

   string object_to_string() const;

   virtual tinyxml2::XMLDocument* to_XML() const;
   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;
   // virtual void read_XML(   );

   // PMML Methods
   void to_PMML(tinyxml2::XMLElement*, const bool& is_data_scaled = false, const Vector< Statistics<double> >& inputs_statistics = Vector< Statistics<double> >() ) const;
   void write_PMML_data_dictionary(tinyxml2::XMLPrinter&, const Vector< Statistics<double> >& inputs_statistics = Vector< Statistics<double> >() ) const;
   void write_PMML_mining_schema(tinyxml2::XMLPrinter&) const;
   void write_PMML_neural_inputs(tinyxml2::XMLPrinter&, const bool& is_data_scaled = false) const;

protected:

   // MEMBERS

   /// Input variables.

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

