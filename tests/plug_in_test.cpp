/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P L U G - I N   T E S T   C L A S S                                                                        */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "plug_in_test.h"


using namespace OpenNN;


PlugInTest::PlugInTest() : UnitTesting() 
{   
}


PlugInTest::~PlugInTest()
{
}


void PlugInTest::test_get_template_file_name()
{
   message += "test_get_template_file_name\n";
}


void PlugInTest::test_get_input_file_name()
{
   message += "test_get_input_file_name\n";
}


void PlugInTest::test_get_script_file_name()
{
   message += "test_get_script_file_name\n";
}


void PlugInTest::test_get_output_file_name()
{
   message += "test_get_output_file_name\n";
}


void PlugInTest::test_get_input_flags()
{
   message += "test_get_input_flags\n";
}


void PlugInTest::test_get_display()
{
   message += "test_get_display\n";
}


void PlugInTest::test_set_template_file_name()
{
   message += "test_set_template_file_name\n";
}


void PlugInTest::test_set_input_file_name()
{
   message += "test_set_input_file_name\n";
}


void PlugInTest::test_set_script_file_name()
{
   message += "test_set_script_file_name\n";
}


void PlugInTest::test_set_output_file_name()
{
   message += "test_set_output_file_name\n";
}


void PlugInTest::test_set_input_flags()
{
   message += "test_set_input_flags\n";
}


void PlugInTest::test_set_display()
{
   message += "test_set_display\n";
}

// @todo

void PlugInTest::test_write_input_file()
{
   message += "test_write_input_file\n";

   //NeuralNetwork nn;

   //PlugIn pi(&nn);

   //Vector<string> input_flags;

   //// Test

   //nn.set(3);

   //nn.get_independent_parameters_pointer()->set_parameter(0, 1.0);
   //nn.get_independent_parameters_pointer()->set_parameter(1, 3.0);
   //nn.get_independent_parameters_pointer()->set_parameter(2, 5.0);

   //pi.set_template_file_name("../data/opennn_tests/template.dat");
   //pi.set_input_file_name("../data/opennn_tests/input.dat");

   //input_flags.set(3);
   //input_flags[0] = "input_flag_1";
   //input_flags[1] = "input_flag_3";
   //input_flags[2] = "input_flag_5";
   //
   //pi.set_input_flags(input_flags);

   //pi.write_input_file();

}


void PlugInTest::test_run_script()
{
   message += "test_run_script\n";

   PlugIn pi;

   // Test

   //pi.set_script_file_name("../data/script");

//   pi.run_script();

}


void PlugInTest::test_read_output_file()
{
   message += "test_read_output_file\n";

   PlugIn pi;

   string output_file_name;

   // Test

   output_file_name = "../data/output.dat";

   pi.set_output_file_name(output_file_name);
}


void PlugInTest::test_read_output_file_header()
{
   message += "test_read_output_file_header\n";
}


// @todo

void PlugInTest::test_calculate_output_data()
{
   
   message += "test_calculate_output_data\n";

   PlugIn pi;

   string template_file_name;
   string input_file_name;
   string script_file_name;
   string output_file_name;

   Vector<string> input_flags;

   Vector<double> input_values;

   Matrix<double> outut_values;

   // Test

   template_file_name = "../data/template.dat";
   input_file_name = "../data/input.dat";
   script_file_name = "../data/batch.dat";
   output_file_name = "../data/output.dat";

   pi.set_template_file_name(template_file_name);
   pi.set_input_file_name(input_file_name);
   pi.set_script_file_name(script_file_name);
   pi.set_output_file_name(output_file_name);

   input_flags.set(2);
   input_flags[0] = "input_flag_1";
   input_flags[1] = "input_flag_2";

   pi.set_input_flags(input_flags);

//   pi.write_input_file();

}


void PlugInTest::test_to_XML()   
{
   message += "test_to_XML\n";
}


void PlugInTest::test_from_XML()   
{
   message += "test_from_XML\n";
}


void PlugInTest::test_save()
{
   message += "test_save\n";

   string file_name = "../data/plug_in.xml";

   PlugIn pi;

   string template_file_name;
   string input_file_name;
   string script_file_name;
   string output_file_name;

   Vector<string> input_flags;

   bool display;

   Vector<double> input_values;

   Matrix<double> outut_values;

   // Test

   template_file_name = "template.dat";
   input_file_name = "input.dat";
   script_file_name = "batch.dat";
   output_file_name = "output.dat";
   

   pi.set_template_file_name(template_file_name);
   pi.set_input_file_name(input_file_name);
   pi.set_script_file_name(script_file_name);
   pi.set_output_file_name(output_file_name);

   input_flags.set(2);
   input_flags[0] = "input_flag_1";
   input_flags[1] = "input_flag_2";

   pi.set_input_flags(input_flags);

   display = false;

   pi.set_display(display);

   pi.save(file_name);

   pi.load(file_name);

   assert_true(pi.get_template_file_name() == template_file_name, LOG);
   assert_true(pi.get_input_file_name() == input_file_name, LOG);
   assert_true(pi.get_script_file_name() == script_file_name, LOG);
   assert_true(pi.get_output_file_name() == output_file_name, LOG);
   assert_true(pi.get_input_flags() == input_flags, LOG);

}


void PlugInTest::test_load()
{
   message += "test_load\n";
}


void PlugInTest::run_test_case()
{
   message += "Running plug-in test case...\n";  

   test_get_template_file_name();
   test_get_input_file_name();

   test_get_script_file_name();

   test_get_output_file_name();

   test_get_input_flags();
   test_get_display();

   // Set methods

   test_set_template_file_name();
   test_set_input_file_name();

   test_set_script_file_name();

   test_set_output_file_name();

   test_set_input_flags();
   test_set_display();

   // Plug-In methods

   test_write_input_file();

   test_run_script();

   test_read_output_file();

   test_read_output_file_header();

   test_calculate_output_data();

   // Serialization methods

   test_to_XML();   
   test_from_XML();   

   test_save();
   test_load();

   message += "End of plug-in test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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
