/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P L U G - I N   C L A S S                                                                                  */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "plug_in.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor.
/// It constructs a default plug-in object, with zero independent and dependent variables. 

PlugIn::PlugIn(void) : MathematicalModel()
{                                            
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor.
/// It creates a plug-in and loads its members from a TinyXML document.
/// @param plug_in_document XML document containing the plug-in members.

PlugIn::PlugIn(const tinyxml2::XMLDocument& plug_in_document)
: MathematicalModel(plug_in_document)
{                                            
}


// DESTRUCTOR
 
/// Destructor. 
/// It does not delete any object.  

PlugIn::~PlugIn(void)
{ 
}


// ASSIGNMENT OPERATOR

// PlugIn& operator = (const PlugIn&) method

/// Assignment operator. 
/// It assigns to this object the members of an existing plug-in object.
/// @param other_plug_in Plug-in object to be assigned.

PlugIn& PlugIn::operator = (const PlugIn& other_plug_in)
{
   if(this != &other_plug_in) 
   {
      input_method = other_plug_in.input_method;

      template_file_name = other_plug_in.template_file_name;
      input_file_name = other_plug_in.input_file_name;

      script_file_name = other_plug_in.script_file_name;

      output_file_name = other_plug_in.output_file_name;

      input_flags = other_plug_in.input_flags;

      output_rows_number = other_plug_in.output_rows_number;
      output_columns_number = other_plug_in.output_columns_number;
   }

   return(*this);
}


// EQUAL TO OPERATOR

// bool operator == (const PlugIn&) const method

/// Equal to operator. 
/// It compares this object with another object of the same class. 
/// It returns true if the members of the two objects have the same values, and false otherwise.
/// @ param other_plug_in Plug-in to be compared with.

bool PlugIn::operator == (const PlugIn& other_plug_in) const
{
   if(input_method == other_plug_in.input_method
   && template_file_name == other_plug_in.template_file_name
   && input_file_name == other_plug_in.input_file_name
   && script_file_name == other_plug_in.script_file_name
   && output_file_name == other_plug_in.output_file_name
   && input_flags == other_plug_in.input_flags
   && output_rows_number == other_plug_in.output_rows_number
   && output_columns_number == other_plug_in.output_columns_number)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}


// METHODS

// const InputMethod& get_input_method(void) const method

/// Returns the method for including the information into the input file. 

const PlugIn::InputMethod& PlugIn::get_input_method(void) const
{
   return(input_method);
}


// std::string write_input_method(void) const method

/// Returns a string with the name of the method for including the information into the input file. 

std::string PlugIn::write_input_method(void) const
{
   switch(input_method)
   {
      case IndependentParametersInput:
      {
         return("IndependentParameters");
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: PlugIn class.\n"
                << "std::string get_input_method_name(void) const method.\n"
                << "Unknown inputs method.\n";
 
	     throw std::logic_error(buffer.str());
      }
      break;
   }   
}


// const std::string& get_template_file_name(void) method

/// Returns the name of the template file. 

const std::string& PlugIn::get_template_file_name(void) const
{
   return(template_file_name);
}


// const std::string& get_input_file_name(void) method

/// Returns the name of the input file. 

const std::string& PlugIn::get_input_file_name(void) const
{
   return(input_file_name);
}


// const std::string& get_script_file_name(void) method

/// Returns the name of the script file. 

const std::string& PlugIn::get_script_file_name(void) const
{
   return(script_file_name);
}


// const std::string& get_output_file_name(void) method

/// Returns the name of the output file. 

const std::string& PlugIn::get_output_file_name(void) const
{
   return(output_file_name);
}


// const Vector<std::string>& get_input_flags(void) const method

/// Returns the vector of input file flags. 

const Vector<std::string>& PlugIn::get_input_flags(void) const
{
   return(input_flags);
}


// const std::string& get_input_flag(const size_t&) const method

/// Returns a single input file flag. 
/// @param i Index of flag. 

const std::string& PlugIn::get_input_flag(const size_t& i) const
{
   return(input_flags[i]);
}


// void set_default(void) method

/// Sets the following default values in this object:
/// <ul>
/// <li> Input method: Independent parameters input.
/// <li> Input file_name: input.dat.
/// <li> Script file_name: batch.bat.
/// <li> Output file_name: output.dat.
/// <li> Display: true. 
/// </ul>

void PlugIn::set_default(void)
{
   input_method = IndependentParametersInput;

   template_file_name = "template.dat";
   input_file_name = "input.dat";

   script_file_name = "batch.bat";

   output_file_name = "output.dat";

   display = true;
}


// void set_input_method(const InputMethod&) method

/// Sets the method for writting the input file. 
/// @param new_input_method Method for inputing the input file. 

void PlugIn::set_input_method(const InputMethod& new_input_method)
{
   input_method = new_input_method;
}


// void set_input_method(const std::string&) method

/// Sets the method for writting the input file from a string. 
/// @param new_input_method Method for inputing the input file. 

void PlugIn::set_input_method(const std::string& new_input_method)
{
   if(new_input_method == "IndependentParameters")
   {
      set_input_method(IndependentParametersInput);
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PlugIn class.\n"
             << "void set_input_method(const std::string&) method.\n"
			 << "Unknown plug-in method: " << new_input_method << ".\n";

	  throw std::logic_error(buffer.str());
   }
}


// void set_template_file_name(const std::string&) method

/// Sets the name of the template file. 
/// @param new_template_file_name Name of template file.

void PlugIn::set_template_file_name(const std::string& new_template_file_name)
{
   template_file_name = new_template_file_name;
}


// void set_input_file_name(const std::string&) method

/// Sets the name of the input file. 
/// @param new_input_file_name Name of input file.

void PlugIn::set_input_file_name(const std::string& new_input_file_name)
{
   input_file_name = new_input_file_name;
}


// void set_script_file_name(const std::string&) method

/// Sets the name of the script file. 
/// @param new_script_file_name Name of script file.

void PlugIn::set_script_file_name(const std::string& new_script_file_name)
{
   script_file_name = new_script_file_name;
}


// void set_output_file_name(const std::string&) method

/// Sets the name of the output file. 
/// @param new_output_file_name Name of output file.

void PlugIn::set_output_file_name(const std::string& new_output_file_name)
{
   output_file_name = new_output_file_name;
}


// void set_input_flags(const Vector<std::string>&) method

/// Sets the flags in the input file. 
/// @param new_input_flags Flags strings. 

void PlugIn::set_input_flags(const Vector<std::string>& new_input_flags)
{
   input_flags = new_input_flags;
}


// void write_input_file(const NeuralNetwork&) const method

/// Thise method writes the input file with values obtained from the neural network. 
/// @param neural_network Neural network. 

void PlugIn::write_input_file(const NeuralNetwork& neural_network) const
{ 
   switch(input_method)
   {
      case IndependentParametersInput:
      {
         write_input_file_independent_parameters(neural_network);
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: PlugIn class.\n"
                << "void write_input_file(const NeuralNetwork&) const method.\n"
                << "Unknown input method.\n";
 
	     throw std::logic_error(buffer.str());
      }
      break;
   }   
}


// void write_input_file_independent_parameters(const NeuralNetwork&) const method

/// @todo

void PlugIn::write_input_file_independent_parameters(const NeuralNetwork& neural_network) const
{
   const IndependentParameters* independent_parameters_pointer = neural_network.get_independent_parameters_pointer();

   #ifdef __OPENNN_DEBUG__ 

   if(!independent_parameters_pointer)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PlugIn class.\n"
             << "void write_input_file_independent_parameters(void) const method.\n"
             << "Pointer to independent parameters is null.\n";

	  throw std::logic_error(buffer.str());
   }

   #endif

   const Vector<double> independent_parameters = independent_parameters_pointer->get_parameters();

   //size_t input_flags_number = input_flags.size();

   //size_t independent_parameters_number = independent_parameters.size();

   //// Control sentence

   //if(input_flags_number != independent_parameters_number)
   //{
   //   buffer << "OpenNN Exception: PlugIn class.\n"
   //          << "void write_input_file_independent_parameters(void) const method.\n"
   //          << "Number of inputs flags must be equal to number of independent parameters.\n";

   //   throw std::logic_error(buffer.str());         
   //}

   //// Template file 

   //std::ifstream template_file(template_file_name.c_str());

   //if(!template_file.is_open())
   //{
   //   buffer << "OpenNN Exception: PlugIn class.\n"
   //          << "void write_input_file_independent_parameters(void) const method.\n"
   //          << "Cannot open template file.\n";            
   //         
   //   throw std::logic_error(buffer.str());
   //}

   //std::string file_string;
   //std::string line;

   //while(getline(template_file, line))
   //{
   //   file_string += line;
   //   file_string += "\n";
   //}

   //template_file.close();

   //// Convert values to string

   //Vector<std::string> independent_parameters_string = independent_parameters.get_string_vector();

   //// Replace flags by values as many times flags are found in string

   //for(size_t i = 0; i < input_flags_number; i++)
   //{
   //   while(file_string.find(input_flags[i]) != std::string::npos)
   //   {
   //      size_t found = file_string.find(input_flags[i]);

   //      if(found != std::string::npos)
   //      {
   //         file_string.replace(file_string.find(input_flags[i]), input_flags[i].length(), independent_parameters_string[i]);
   //      }
   //   }
   //}

   //// Input file

   //std::ofstream input_file(input_file_name.c_str());
   //
   //if(!input_file.is_open())
   //{
   //   buffer << "OpenNN Exception: PlugIn class.\n"
   //          << "void write_input_file(void) const method.\n"
   //          << "Cannot open inputs file.\n";            
   //         
   //   throw std::logic_error(buffer.str());
   //}

   //input_file << file_string << "\n";

   //input_file.close();
}


// void run_script(void) const

/// This method runs the script needed for executing the mathematical model. 

void PlugIn::run_script(void) const
{
   if(!script_file_name.empty())
   {
      int ok = system(script_file_name.c_str());

      if(ok == 0)
      {
         // Error message
      }
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PlugIn class.\n"
             << "void run_script(void) const.\n"
             << "Batch file_name is empty.\n";

      throw std::logic_error(buffer.str());
   }
}


// Matrix<double> read_output_file(void) const method

/// This method reads the output file from the mathematical model.
/// Here the output file only contains a data matrix. 

Matrix<double> PlugIn::read_output_file(void) const
{
   Matrix<double> data(output_file_name);

   return(data);
}


// Matrix<double> read_output_file_header(void) const method

/// This method reads the output file from the mathematical model.
/// Here the output file contains a header file and a data matrix. 

Matrix<double> PlugIn::read_output_file_header(void) const
{
   // Open outputs file for reading

   std::ifstream output_file(output_file_name.c_str());
   
   if(!output_file.is_open())
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: PlugIn class.\n"
             << "Matrix<double> read_output_file_header(void) const method.\n"
             << "Cannot open outputs file.\n";            
            
      throw std::logic_error(buffer.str());
   }

   std::string header;
   getline(output_file, header);

   Matrix<double> data;

   output_file >> data;

   output_file.close();

   return(data);
}


// Matrix<double> calculate_solutions(const NeuralNetwork&) const method

/// Returns the solutions to the mathematical model. 
/// which are given by the independent and the dependent variables. 
/// The process is as follows:
/// <ul>
/// <li> Write input file.
/// <li> Run script. 
/// <li> Read output file. 
/// </ul>

Matrix<double> PlugIn::calculate_solutions(const NeuralNetwork& neural_network) const
{
   write_input_file(neural_network);

   run_script();

   return(read_output_file());
}


// std::string to_string(void) const method

/// Returns a string representation of the current plug-in object. 

std::string PlugIn::to_string(void) const
{
   std::ostringstream buffer; 

   buffer << "Plug-in\n" 
          << "Independent variables number: " << independent_variables_number << "\n" 
          << "Dependent variables number: " << dependent_variables_number << "\n"
          << "Input method: " << input_method << "\n"
          << "Template file_name: " << template_file_name << "\n"
          << "Input file_name: " <<  input_file_name << "\n"
          << "Script file_name: " << script_file_name << "\n"
          << "Output file_name: " << output_file_name << "\n"
          << "Input flags: " << input_flags << "\n"
          << "Output rows number: " << output_rows_number << "\n"
          << "Output columns number: " <<  output_columns_number << "\n"
          << "Display: " << display << "\n";

   return(buffer.str());
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes the plug-in object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document. 

tinyxml2::XMLDocument* PlugIn::to_XML(void) const   
{
   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   std::ostringstream buffer;

   tinyxml2::XMLElement* plug_in_element = document->NewElement("PlugIn");

   document->InsertFirstChild(plug_in_element);

   // Independent variables number 

   {
      tinyxml2::XMLElement* independent_variables_number_element = document->NewElement("IndependentVariablesNumber");
      plug_in_element->LinkEndChild(independent_variables_number_element);

      buffer.str("");
      buffer << independent_variables_number;

      tinyxml2::XMLText* independent_variables_number_text = document->NewText(buffer.str().c_str());
      independent_variables_number_element->LinkEndChild(independent_variables_number_text);
   }

   // Dependent variables number 

   {
      tinyxml2::XMLElement* dependent_variables_number_element = document->NewElement("DependentVariablesNumber");
      plug_in_element->LinkEndChild(dependent_variables_number_element);

      buffer.str("");
      buffer << dependent_variables_number;

      tinyxml2::XMLText* dependent_variables_number_text = document->NewText(buffer.str().c_str());
      dependent_variables_number_element->LinkEndChild(dependent_variables_number_text);
   }

   // Input method

   {
      tinyxml2::XMLElement* input_method_element = document->NewElement("InputMethod");
      plug_in_element->LinkEndChild(input_method_element);

      std::string input_method_name = write_input_method();

      tinyxml2::XMLText* input_method_text = document->NewText(input_method_name.c_str());
      input_method_element->LinkEndChild(input_method_text);
   }

   // Template file_name

   {
      tinyxml2::XMLElement* template_file_name_element = document->NewElement("TemplateFileName");
      plug_in_element->LinkEndChild(template_file_name_element);

      tinyxml2::XMLText* template_file_name_text = document->NewText(template_file_name.c_str());
      template_file_name_element->LinkEndChild(template_file_name_text);
   }

   // Input file_name 

   {
      tinyxml2::XMLElement* input_file_name_element = document->NewElement("InputFileName");
      plug_in_element->LinkEndChild(input_file_name_element);

      tinyxml2::XMLText* input_file_name_text = document->NewText(input_file_name.c_str());
      input_file_name_element->LinkEndChild(input_file_name_text);
   }

   // Batch file_name 

   {
      tinyxml2::XMLElement* script_file_name_element = document->NewElement("BatchFileName");
      plug_in_element->LinkEndChild(script_file_name_element);

      tinyxml2::XMLText* script_file_name_text = document->NewText(script_file_name.c_str());
      script_file_name_element->LinkEndChild(script_file_name_text);
   }

   // Output file_name 

   {
      tinyxml2::XMLElement* output_file_name_element = document->NewElement("OutputFileName");
      plug_in_element->LinkEndChild(output_file_name_element);

      tinyxml2::XMLText* output_file_name_text = document->NewText(output_file_name.c_str());
      output_file_name_element->LinkEndChild(output_file_name_text);
   }

   // Input flags

   {
      tinyxml2::XMLElement* input_flags_element = document->NewElement("InputFlags");
      plug_in_element->LinkEndChild(input_flags_element);

      buffer.str("");
      buffer << input_flags;

      tinyxml2::XMLText* input_flags_text = document->NewText(buffer.str().c_str());
      input_flags_element->LinkEndChild(input_flags_text);
   }

   // Output rows number

   {
      tinyxml2::XMLElement* output_rows_number_element = document->NewElement("OutputRowsNumber");
      plug_in_element->LinkEndChild(output_rows_number_element);

      buffer.str("");
      buffer << output_rows_number;

      tinyxml2::XMLText* output_rows_number_text = document->NewText(buffer.str().c_str());
      output_rows_number_element->LinkEndChild(output_rows_number_text);
   }

   // Output columns number

   {
      tinyxml2::XMLElement* output_columns_number_element = document->NewElement("OutputColumnsNumber");
      plug_in_element->LinkEndChild(output_columns_number_element);

      buffer.str("");
      buffer << output_columns_number;

      tinyxml2::XMLText* output_columns_number_text = document->NewText(buffer.str().c_str());
      output_columns_number_element->LinkEndChild(output_columns_number_text);
   }

   // Display

   {
      tinyxml2::XMLElement* display_element = document->NewElement("Display");
      plug_in_element->LinkEndChild(display_element);

      buffer.str("");
      buffer << display;

      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
      display_element->LinkEndChild(display_text);
   }

   return(document);   
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the plug in object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void PlugIn::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    file_stream.OpenElement("PlugIn");

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

    // Input method

    file_stream.OpenElement("InputMethod");

    file_stream.PushText(write_input_method().c_str());

    file_stream.CloseElement();

    // Template file_name

    file_stream.OpenElement("TemplateFileName");

    file_stream.PushText(template_file_name.c_str());

    file_stream.CloseElement();

    // Input file_name

    file_stream.OpenElement("InputFileName");

    file_stream.PushText(input_file_name.c_str());

    file_stream.CloseElement();

    // Batch file_name

    file_stream.OpenElement("BatchFileName");

    file_stream.PushText(script_file_name.c_str());

    file_stream.CloseElement();

    // Output file_name

    file_stream.OpenElement("OutputFileName");

    file_stream.PushText(output_file_name.c_str());

    file_stream.CloseElement();

    // Input flags

    file_stream.OpenElement("InputFlags");

    buffer.str("");
    buffer << input_flags;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Output rows number

    file_stream.OpenElement("OutputRowsNumber");

    buffer.str("");
    buffer << output_rows_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Output columns number

    file_stream.OpenElement("OutputColumnsNumber");

    buffer.str("");
    buffer << output_columns_number;

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

/// Deserializes a TinyXML document into this plug-in object.
/// @param document XML document containing the member data.
/// @todo

void PlugIn::from_XML(const tinyxml2::XMLDocument& document)
{
  // Independent variables number
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("IndependentVariablesNumber");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_independent_variables_number(atoi(text));
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Dependent variables number
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("DependentVariablesNumber");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_dependent_variables_number(atoi(text));
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Input method
  {
     const tinyxml2::XMLElement* input_method_element = document.FirstChildElement("InputMethod");

     if(input_method_element)
     {
        const char* input_method_text = input_method_element->GetText();

        if(input_method_text)
        {
           try
           {
              set_input_method(input_method_text);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Template file_name
  {
     const tinyxml2::XMLElement* template_file_name_element = document.FirstChildElement("TemplateFileName");

     if(template_file_name_element)
     {
        const char* template_file_name_text = template_file_name_element->GetText();

        if(template_file_name_text)
        {
           try
           {
              set_template_file_name(template_file_name_text);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Input file_name
  {
     const tinyxml2::XMLElement* input_file_name_element = document.FirstChildElement("InputFileName");

     if(input_file_name_element)
     {
        const char* input_file_name_text = input_file_name_element->GetText();

        if(input_file_name_text)
        {
           try
           {
              set_input_file_name(input_file_name_text);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Batch file_name
  {
     const tinyxml2::XMLElement* script_file_name_element = document.FirstChildElement("BatchFileName");

     if(script_file_name_element)
     {
        const char* script_file_name_text = script_file_name_element->GetText();

        if(script_file_name_text)
        {
           try
           {
              set_script_file_name(script_file_name_text);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Output file_name
  {
     const tinyxml2::XMLElement* output_file_name_element = document.FirstChildElement("OutputFileName");

     if(output_file_name_element)
     {
        const char* output_file_name_text = output_file_name_element->GetText();

        if(output_file_name_text)
        {
           try
           {
              set_output_file_name(output_file_name_text);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }
/*
  // Input flags
  {
     tinyxml2::XMLElement* input_flags_element = document->FirstChildElement("InputFlags");

     if(input_flags_element)
     {
        const char* input_flags_text = input_flags_element->GetText();

        if(input_flags_text)
        {
           Vector<std::string> new_input_flags;
           new_input_flags.parse(input_flags_text);

           try
           {
              set_input_flags(new_input_flags);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }
*/
  // Display
  {
     const tinyxml2::XMLElement* display_element = document.FirstChildElement("Display");

     if(display_element)
     {
        const char* display_text = display_element->GetText();

        if(display_text)
        {
           try
           {
              std::string display_string(display_text);

              set_display(display_string != "0");
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }
}

}

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
