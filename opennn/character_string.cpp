/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S T R I N G   C L A S S                                                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */ 
/****************************************************************************************************************/

// OpenNN includes

#include "character_string.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a Newton method training algorithm object not associated to any loss functional object.
/// It also initializes the class members to their default values.

String::String(): string()
{

}


String::String(const string& str): string(str)
{

}


String::String(const char* s) : string(s)
{

}


// DESTRUCTOR

/// Destructor.

String::~String()
{
}


String& String::operator =(const String& other_string)
{
   if(this != &other_string)
   {
        *this = other_string;
   }

   return(*this);
}


String& String::operator =(const string& str)
{
    const String other_string(str);

    *this = other_string;

   return(*this);
}


String& String::operator =(const char* s)
{
    const String other_string(s);

    *this = other_string;

   return(*this);
}


Vector<String> String::split(const char& delimiter) const
{
    Vector<String> elements;

    String element;

    istringstream is(*this);

    while(getline(is, element, delimiter)) {
      elements.push_back(element);
    }

    return(elements);
}

void String::trim()
{
    //prefixing spaces

    this->erase(0, this->find_first_not_of(' '));

    //surfixing spaces

    this->erase(this->find_last_not_of(' ') + 1);
}


String String::get_trimmed() const
{
    String trimmed;

    //prefixing spaces

    trimmed.erase(0, trimmed.find_first_not_of(' '));

    //surfixing spaces

    trimmed.erase(trimmed.find_last_not_of(' ') + 1);

    return(trimmed);
}

}


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

