/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   F I L E   U T I L I T I E S   C L A S S   H E A D E R                                                      */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __FILEUTILITIES_H__
#define __FILEUTILITIES_H__

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

class FileUtilities
{

public:

   // DEFAULT CONSTRUCTOR

    explicit FileUtilities();

    explicit FileUtilities(const string&);

   // DESTRUCTOR

   virtual ~FileUtilities();

   // SET METHODS

   // Display messages

   const bool& get_display() const;

   // SET METHODS

   // Display messages

   void set_display(const bool&);
   void set_file_name(const string&);

   size_t count_lines_number() const;

   Vector<string> get_output_file_names(const size_t&) const;

   string read_header() const;

   Vector<string> split_file(const size_t&) const;

   void merge_files(const Vector<string>&) const;

   void replace(const string&, const string&) const;

   void sample_file(const size_t&) const;


protected:

   // MEMBERS


   string file_name;

   bool header;

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

