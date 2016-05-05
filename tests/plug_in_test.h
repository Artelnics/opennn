/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P L U G - I N   T E S T   C L A S S   H E A D E R                                                          */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __PLUGINTEST_H__
#define __PLUGINTEST_H__

// Unit testing includes

#include "unit_testing.h"

using namespace OpenNN;

class PlugInTest : public UnitTesting 
{

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

public:

   // CONSTRUCTOR

   explicit PlugInTest(void);

   // DESTRUCTOR

   virtual ~PlugInTest(void);

   // Get methods
    
   void test_get_template_file_name(void);
   void test_get_input_file_name(void);

   void test_get_script_file_name(void);

   void test_get_output_file_name(void);

   void test_get_input_flags(void);
   void test_get_display(void);

   // Set methods

   void test_set_template_file_name(void);
   void test_set_input_file_name(void);

   void test_set_script_file_name(void);

   void test_set_output_file_name(void);

   void test_set_input_flags(void);
   void test_set_display(void);

   // Plug-In methods

   void test_write_input_file(void);

   void test_run_script(void);

   void test_read_output_file(void);

   void test_read_output_file_header(void);

   void test_calculate_output_data(void);

   // Serialization methods

   void test_to_XML(void);   
   void test_from_XML(void);   

   void test_save(void);
   void test_load(void);

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
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
