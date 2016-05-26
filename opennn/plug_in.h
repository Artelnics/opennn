/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P L U G - I N   C L A S S   H E A D E R                                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __PLUGIN_H__
#define __PLUGIN_H__

// System includes

#include<fstream>
#include<iostream>
#include<string>
#include<sstream>
#include <time.h>

// OpenNN includes

#include "vector.h"
#include "matrix.h"

#include "mathematical_model.h"
#include "neural_network.h"

namespace OpenNN
{

///
/// This method represents an external mathematical model which communicates with OpenNN by means of input and output files. 
///

class PlugIn : public MathematicalModel
{

public:

   // DEFAULT CONSTRUCTOR

   explicit PlugIn(void);

   // XML CONSTRUCTOR

   explicit PlugIn(const tinyxml2::XMLDocument&);

   // DESTRUCTOR

   virtual ~PlugIn(void);

   // ENUMERATIONS

   /// Enumeration of available methods for introducing neural network data into the input file. 

   enum InputMethod{IndependentParametersInput};

   // ASSIGNMENT OPERATOR

   PlugIn& operator = (const PlugIn&);

   // EQUAL TO OPERATOR

   bool operator == (const PlugIn&) const;

   // METHODS

   // Get methods

   const InputMethod& get_input_method(void) const;
   std::string write_input_method(void) const;
    
   const std::string& get_template_file_name(void) const;
   const std::string& get_input_file_name(void) const;

   const std::string& get_script_file_name(void) const;

   const std::string& get_output_file_name(void) const;

   const Vector<std::string>& get_input_flags(void) const;
   const std::string& get_input_flag(const size_t&) const;

   // Set methods

   void set_default(void);

   void set_input_method(const InputMethod&);
   void set_input_method(const std::string&);

   void set_template_file_name(const std::string&);
   void set_input_file_name(const std::string&);

   void set_script_file_name(const std::string&);

   void set_output_file_name(const std::string&);

   void set_input_flags(const Vector<std::string>&);

   // Plug-In methods

   void write_input_file(const NeuralNetwork&) const;
   void write_input_file_independent_parameters(const NeuralNetwork&) const;

   void run_script(void) const;

   Matrix<double> read_output_file(void) const;

   Matrix<double> read_output_file_header(void) const;

   Matrix<double> calculate_solutions(const NeuralNetwork&) const;

   // Serialization methods

   std::string to_string(void) const;

   tinyxml2::XMLDocument* to_XML(void) const;   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML(   );

   //tinyxml2::XMLElement* get_output_data_XML(const Matrix<double>&) const;


private: 

   /// Type of data to be entered in the mathematical model. 

   InputMethod input_method;

   /// Name of template file. 

   std::string template_file_name;

   /// Name of input file.

   std::string input_file_name;

   /// Name of script file. 

   std::string script_file_name;

   /// Name of output file. 

   std::string output_file_name;

   /// Vector of flags in the input file. 

   Vector<std::string> input_flags;

   /// Number of rows in the output file. 

   size_t output_rows_number;

   /// Number of columns in the output file. 

   size_t output_columns_number;
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
