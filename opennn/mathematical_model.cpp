/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M A T H E M A T I C A L   M O D E L   C L A S S                                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */ 
/*   Artificial Intelligence Techniques SL                                                                      */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "mathematical_model.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor.
/// It constructs a default mathematical model object, with zero independent and dependent variables. 

MathematicalModel::MathematicalModel()
{                                            
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor.
/// It creates a mathematical model and loads its members from a TinyXML document.
/// @param mathematical_model_document XML document containing the mathematical model members.

MathematicalModel::MathematicalModel(const tinyxml2::XMLDocument& mathematical_model_document)
{                
   set_default();

   from_XML(mathematical_model_document);   
}


// FILE CONSTRUCTOR

/// File constructor.
/// It creates a mathematical model and loads its members from a XML file. 
/// @param file_name Name of mathematical model XML file.

MathematicalModel::MathematicalModel(const string& file_name)
{                
   set_default();

   load(file_name);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a mathematical model object and copies its members from another object. 
/// @param other_mathematical_model Mathematical model object to be copied. 

MathematicalModel::MathematicalModel(const MathematicalModel& other_mathematical_model)
{                
   set(other_mathematical_model);   
}


// DESTRUCTOR
 
/// Destructor. 
/// It does not delete any object.  

MathematicalModel::~MathematicalModel()
{ 
}


// ASSIGNMENT OPERATOR

/// Assignment operator. 
/// It assigns to this object the members of an existing mathematical model object.
/// @param other_mathematical_model Mathematical model object to be assigned.

MathematicalModel& MathematicalModel::operator =(const MathematicalModel& other_mathematical_model)
{
   if(this != &other_mathematical_model) 
   {
      independent_variables_number = other_mathematical_model.independent_variables_number;
      dependent_variables_number = other_mathematical_model.dependent_variables_number;

      display = other_mathematical_model.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR

/// Equal to operator. 
/// It compares this object with another object of the same class. 
/// It returns true if the members of the two objects have the same values, and false otherwise.
/// @ param other_mathematical_model Mathematical model to be compared with.

bool MathematicalModel::operator ==(const MathematicalModel& other_mathematical_model) const
{
   if(independent_variables_number == other_mathematical_model.independent_variables_number
   && dependent_variables_number == other_mathematical_model.dependent_variables_number
   && display == other_mathematical_model.display)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}


// METHODS

// const size_t& get_independent_variables_number() const method

/// Returns the number of independent variables in the mathematical model. 

const size_t& MathematicalModel::get_independent_variables_number() const
{
   return(independent_variables_number);
}


// size_t get_dependent_variables_number() const method

/// Returns the number of dependent variables in the mathematical model. 

const size_t& MathematicalModel::get_dependent_variables_number() const
{
   return(dependent_variables_number);
}


// size_t count_variables_number() const method

/// Returns the total number variablesin the mathematical model.
/// This is the sum of the numbers of independent and dependent variables. 

size_t MathematicalModel::count_variables_number() const
{
   return(independent_variables_number + dependent_variables_number);
}


// const bool get_display() method

/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& MathematicalModel::get_display() const
{
   return(display);     
}  


// void set(const MathematicalModel&) method

/// Sets the members of this mathematical model object with those from other mathematical model object. 
/// @param other_mathematical_model Mathematical model object to be copied. 

void MathematicalModel::set(const MathematicalModel& other_mathematical_model)
{
   independent_variables_number = other_mathematical_model.independent_variables_number;
   dependent_variables_number = other_mathematical_model.dependent_variables_number;

   display = other_mathematical_model.display;
}


// void set_independent_variables_number(const size_t&) method

/// Sets the number of independent variables in the mathematical model. 
/// @param new_independent_variables_number Number of independent variables.

void MathematicalModel::set_independent_variables_number(const size_t& new_independent_variables_number)
{
   independent_variables_number = new_independent_variables_number;
}


// void set_dependent_variables_number(const size_t&) method

/// Sets the number of dependent variables in the mathematical model. 
/// @param new_dependent_variables_number Number of dependent variables.

void MathematicalModel::set_dependent_variables_number(const size_t& new_dependent_variables_number)
{
   dependent_variables_number = new_dependent_variables_number;
}


// void set_default() method

/// Sets the following default values in the mathematical model:
/// <ul>
/// <li> Number of dependent variables: 0.
/// <li> Number of independent variables: 0.
/// <li> Display: True.
/// </ul>

void MathematicalModel::set_default()
{
   dependent_variables_number = 0;
   independent_variables_number = 0;

   display = true;
}


// void set_display(const bool&) const method

/// Sets a new display value. 
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void MathematicalModel::set_display(const bool& new_display) 
{
   display = new_display;     
}


// Matrix<double> calculate_solutions(const NeuralNetwork&) const method

/// This virtual method returns the solutions to the mathematical model, 
/// which are given by the independent and the dependent variables. 
/// Needs to be derived, otherwise an exception is thrown.

Matrix<double> MathematicalModel::calculate_solutions(const NeuralNetwork&) const
{
   ostringstream buffer;

   buffer << "OpenNN Exception: MathematicalModel class.\n"
          << "Matrix<double> calculate_solutions(const NeuralNetwork&) const method.\n"
          << "This method has not been derived.\n";
 
   throw logic_error(buffer.str());
}


// Vector<double> calculate_final_solutions(const NeuralNetwork&) const method

/// This virtual method returns the final solutions of the mathematical model, 
/// which are given by the final independent and dependent variables. 
/// Needs to be derived, otherwise an exception is thrown.

Vector<double> MathematicalModel::calculate_final_solutions(const NeuralNetwork&) const
{
   ostringstream buffer;

   buffer << "OpenNN Exception: MathematicalModel class.\n"
          << "Vector<double> calculate_final_solutions(const NeuralNetwork&) const method.\n"
          << "This method has not been derived.\n";
 
   throw logic_error(buffer.str());
}


// Matrix<double> calculate_dependent_variables(const NeuralNetwork&, const Matrix<double>&) const method

/// This virtual method returns the dependent variables solutions to the mathematical model, 
/// Needs to be derived, otherwise an exception is thrown.

Matrix<double> MathematicalModel::calculate_dependent_variables(const NeuralNetwork&, const Matrix<double>&) const
{
   ostringstream buffer;

   buffer << "OpenNN Exception: MathematicalModel class.\n"
          << "Matrix<double> calculate_dependent_variables(const NeuralNetwork&, const Matrix<double>&) const method.\n"
          << "This method has not been derived.\n";
 
   throw logic_error(buffer.str());
}


/// Returns a string representation of the current mathematical model object. 

string MathematicalModel::object_to_string() const
{
   ostringstream buffer; 

   buffer << "Mathematical model\n"
          << "Independent variables number: " << independent_variables_number << "\n" 
          << "Dependent variables number: " << dependent_variables_number << "\n"
          << "Display: " << display << "\n";

   return(buffer.str());
}


// void print() const method

/// This method outputs to the console the string representation of the mathematical model. 

void MathematicalModel::print() const
{
   cout << object_to_string();
}



// tinyxml2::XMLDocument* to_XML() const method

/// Serializes the mathematical model object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document. 

tinyxml2::XMLDocument* MathematicalModel::to_XML() const   
{
   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   ostringstream buffer;

   tinyxml2::XMLElement* mathematical_model_element = document->NewElement("MathematicalModel");

   document->InsertFirstChild(mathematical_model_element);

   // Independent variables number 
   {
      tinyxml2::XMLElement* independent_variables_number_element = document->NewElement("IndependentVariablesNumber");
      mathematical_model_element->LinkEndChild(independent_variables_number_element);

      buffer.str("");
      buffer << independent_variables_number;

      tinyxml2::XMLText* independent_variables_number_text = document->NewText(buffer.str().c_str());
      independent_variables_number_element->LinkEndChild(independent_variables_number_text);
   }

   // Dependent variables number 
   {
      tinyxml2::XMLElement* dependent_variables_number_element = document->NewElement("DependentVariablesNumber");
      mathematical_model_element->LinkEndChild(dependent_variables_number_element);

      buffer.str("");
      buffer << dependent_variables_number;

      tinyxml2::XMLText* dependent_variables_number_text = document->NewText(buffer.str().c_str());
      dependent_variables_number_element->LinkEndChild(dependent_variables_number_text);
   }

   // Display 
   {
      tinyxml2::XMLElement* display_element = document->NewElement("Display");
      mathematical_model_element->LinkEndChild(display_element);

      buffer.str("");
      buffer << display;

      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
      display_element->LinkEndChild(display_text);
   }

   return(document);   
}


// virtual void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the mathematical model object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void MathematicalModel::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("MathematicalModel");

    // Independent variables number

    file_stream.OpenElement("IndependentVariablesNumber");

    buffer.str("");
    buffer << independent_variables_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Dependent variables number

    file_stream.OpenElement("DependentVariablesNumber");

    buffer.str("");
    buffer << dependent_variables_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Display

    file_stream.OpenElement("Display");

    buffer.str("");
    buffer << display;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this mathematical model object. 
/// @param document XML document containing the member data.

void MathematicalModel::from_XML(const tinyxml2::XMLDocument& document)
{
   const tinyxml2::XMLElement* root_element = document.FirstChildElement("MathematicalModel");

   ostringstream buffer;

   if(!root_element)
   {
       buffer << "OpenNN Exception: MathematicalModel class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Unkown root element.\n";

       throw logic_error(buffer.str());
   }

  // Independent variables number
  {
     const tinyxml2::XMLElement* element = root_element->FirstChildElement("IndependentVariablesNumber");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_independent_variables_number(atoi(text));
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
     }
  }

  // Dependent variables number
  {
     const tinyxml2::XMLElement* element = root_element->FirstChildElement("DependentVariablesNumber");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_dependent_variables_number(atoi(text));
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
     }
  }

  // Display
  {
     const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              const string string(text);

              set_display(string != "0");
           }
           catch(const logic_error& e)
           {
              cout << e.what() << endl;
           }
        }
     }
  }
}


// void save(const string&) const method

/// Saves to a file the XML representation of the mathematical object. 
/// @param file_name Name of mathematical model XML file.

void MathematicalModel::save(const string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const string&) method

/// Loads the members of the mathematical model from a XML file. 
/// @param file_name Name of mathematical model XML file.

void MathematicalModel::load(const string& file_name)
{
   ostringstream buffer;

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      buffer << "OpenNN Exception: MathematicalModel class.\n"
             << "void load(const string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw logic_error(buffer.str());
   }

   from_XML(document);
}


// void save_data(const NeuralNetwork&, const string&) const method

/// @todo

void MathematicalModel::save_data(const NeuralNetwork&, const string&) const
{
   ostringstream buffer;

   buffer << "OpenNN Exception: MathematicalModel class.\n"
          << "void save_data(const NeuralNetwork&, const string&) const method.\n"
          << "This method has not been derived.\n";
 
   throw logic_error(buffer.str());
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
