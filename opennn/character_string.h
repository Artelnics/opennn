/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S T R I N G   C L A S S   H E A D E R                                                                      */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __STRING_H__
#define __STRING_H__

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// OpenNN includes

#include "vector.h"


namespace OpenNN
{

class String : public string
{

public:

   // DEFAULT CONSTRUCTOR

    explicit String();
    explicit String(const string&);
    explicit String(const char*);

   // DESTRUCTOR

   virtual ~String();

   // OPERATORS

   String& operator =(const String&);
   String& operator =(const string&);
   String& operator =(const char* s);

   // METHODS

   Vector<String> split(const char& delimiter) const;

   void trim();

   String get_trimmed() const;
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

