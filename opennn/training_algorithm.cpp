/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   A L G O R I T H M   C L A S S                                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "training_algorithm.h"

namespace OpenNN
{


// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a training algorithm object not associated to any loss functional object.  

TrainingAlgorithm::TrainingAlgorithm(void)
 : loss_index_pointer(NULL)
{ 
   set_default();
}


// GENERAL CONSTRUCTOR

/// General constructor. 
/// It creates a training algorithm object associated to a loss functional object.
/// @param new_loss_index_pointer Pointer to a loss functional object.

TrainingAlgorithm::TrainingAlgorithm(LossIndex* new_loss_index_pointer)
 : loss_index_pointer(new_loss_index_pointer)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a training algorithm object not associated to any loss functional object. 
/// It also loads the other members from a XML document.

TrainingAlgorithm::TrainingAlgorithm(const tinyxml2::XMLDocument& document)
 : loss_index_pointer(NULL)
{ 
   from_XML(document);
}


// DESTRUCTOR 

/// Destructor

TrainingAlgorithm::~TrainingAlgorithm(void)
{ 
}


// ASSIGNMENT OPERATOR

/// Assignment operator.
/// It assigns to this object the members of an existing training algorithm object.
/// @param other_training_algorithm Training algorithm object to be assigned.

TrainingAlgorithm& TrainingAlgorithm::operator = (const TrainingAlgorithm& other_training_algorithm)
{
   if(this != &other_training_algorithm)
   {
      loss_index_pointer = other_training_algorithm.loss_index_pointer;

      display = other_training_algorithm.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR

/// Equal to operator.
/// @param other_training_algorithm Training algorithm object to be compared with.

bool TrainingAlgorithm::operator == (const TrainingAlgorithm& other_training_algorithm) const
{
   if(loss_index_pointer == other_training_algorithm.loss_index_pointer
   && display == other_training_algorithm.display)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}


// METHODS

// LossIndex* get_loss_index_pointer(void) const method

/// Returns a pointer to the loss functional object to which the training algorithm is
/// associated.

LossIndex* TrainingAlgorithm::get_loss_index_pointer(void) const
{
    #ifdef __OPENNN_DEBUG__

    if(!loss_index_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
               << "LossIndex* get_loss_index_pointer(void) const method.\n"
               << "Loss index pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    #endif

   return(loss_index_pointer);
}


// bool has_loss_index(void) const method

/// Returns true if this training algorithm object has an associated loss functional object,
/// and false otherwise.

bool TrainingAlgorithm::has_loss_index(void) const
{
    if(loss_index_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// const bool& get_display(void) const method

/// Returns true if messages from this class can be displayed on the screen, or false if messages from
/// this class can't be displayed on the screen.

const bool& TrainingAlgorithm::get_display(void) const
{
   return(display);
}


// const size_t& get_display_period(void) const method

/// Returns the number of iterations between the training showing progress.

const size_t& TrainingAlgorithm::get_display_period(void) const
{
   return(display_period);
}


// const size_t& get_save_period(void) const method

/// Returns the number of iterations between the training saving progress.

const size_t& TrainingAlgorithm::get_save_period(void) const
{
   return(save_period);
}


// const size_t& get_neural_network_filename(void) const method

/// Returns the file name where the neural network will be saved.

const std::string& TrainingAlgorithm::get_neural_network_file_name(void) const
{
   return(neural_network_file_name);
}


// void set(void) method

/// Sets the loss functional pointer to NULL.
/// It also sets the rest of members to their default values. 

void TrainingAlgorithm::set(void)
{
   loss_index_pointer = NULL;

   set_default();
}


// void set(LossIndex*) method

/// Sets a new loss functional pointer.
/// It also sets the rest of members to their default values. 
/// @param new_loss_index_pointer Pointer to a loss functional object. 

void TrainingAlgorithm::set(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;

   set_default();
}


// void set_loss_index_pointer(LossIndex*) method

/// Sets a pointer to a loss functional object to be associated to the training algorithm.
/// @param new_loss_index_pointer Pointer to a loss functional object.

void TrainingAlgorithm::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;
}


// void set_display(const bool&) method

/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void TrainingAlgorithm::set_display(const bool& new_display)
{
   display = new_display;
}


// void set_display_period(size_t) method

/// Sets a new number of iterations between the training showing progress.
/// @param new_display_period
/// Number of iterations between the training showing progress.

void TrainingAlgorithm::set_display_period(const size_t& new_display_period)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_display_period <= 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
             << "void set_display_period(const size_t&) method.\n"
             << "Display period must be greater than 0.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   display_period = new_display_period;
}


// void set_save_period(const size_t&) method

/// Sets a new number of iterations between the training saving progress.
/// @param new_save_period
/// Number of iterations between the training saving progress.

void TrainingAlgorithm::set_save_period(const size_t& new_save_period)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_save_period <= 0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
             << "void set_save_period(const size_t&) method.\n"
             << "Save period must be greater than 0.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   save_period = new_save_period;
}


// void set_neural_network_file_name(const std::string&) method

/// Sets a new file name where the neural network will be saved.
/// @param new_neural_network_file_name
/// File name for the neural network object.

void TrainingAlgorithm::set_neural_network_file_name(const std::string& new_neural_network_file_name)
{
   neural_network_file_name = new_neural_network_file_name;
}


// void set_default(void) method 

/// Sets the members of the training algorithm object to their default values.

void TrainingAlgorithm::set_default(void)
{
   display = true;

   display_period = 5;

   save_period = UINT_MAX;

   neural_network_file_name = "neural_network.xml";
}


// std::string write_training_algorithm_type(void) const method

/// This method writes a string with the type of training algoritm.

std::string TrainingAlgorithm::write_training_algorithm_type(void) const
{
   return("USER_TRAINING_ALGORITHM");
}


// void check(void) const method

/// Performs a default checking for training algorithms.
/// In particular, it checks that the loss functional pointer associated to the training algorithm is not NULL,
/// and that the neural network associated to that loss functional is neither NULL.
/// If that checkings are not hold, an exception is thrown. 

void TrainingAlgorithm::check(void) const
{
   std::ostringstream buffer;

   if(!loss_index_pointer)
   {
      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void check(void) const method.\n"
             << "Pointer to loss functional is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   if(neural_network_pointer == NULL)
   {
      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void check(void) const method.\n"
             << "Pointer to neural network is NULL.\n";

      throw std::logic_error(buffer.str());
   }
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes a default training algorithm object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document.

tinyxml2::XMLDocument* TrainingAlgorithm::to_XML(void) const
{
    std::ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Nueral network outputs integrals

    tinyxml2::XMLElement* training_algorithm_element = document->NewElement("TrainingAlgorithm");

    document->InsertFirstChild(training_algorithm_element);

    // Display
    {
       tinyxml2::XMLElement* element = document->NewElement("Display");
       training_algorithm_element->LinkEndChild(element);

       buffer.str("");
       buffer << display;

       tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
       element->LinkEndChild(text);
    }

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the training algorithm object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void TrainingAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    file_stream.OpenElement("TrainingAlgorithm");

    // Display

    file_stream.OpenElement("Display");

    buffer.str("");
    buffer << display;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Loads a default training algorithm from a XML document.
/// @param document TinyXML document containing the error term members.

void TrainingAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("TrainingAlgorithm");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Training algorithm element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

  // Display
  {
     const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

     if(display_element)
     {
        const std::string new_display_string = display_element->GetText();

        try
        {
           set_display(new_display_string != "0");
        }
        catch(const std::logic_error& e)
        {
           std::cout << e.what() << std::endl;
        }
     }
  }

}


// std::string to_string(void) const method

/// Returns a default string representation of a training algorithm.

std::string TrainingAlgorithm::to_string(void) const
{
   std::ostringstream buffer;

   buffer << "Training strategy\n" 
          << "Display: " << display << "\n";

   return(buffer.str());
}


// Matrix<std::string> to_string_matrix(void) const method

/// Returns a default (empty) string matrix containing the members
/// of the training algorithm object.

Matrix<std::string> TrainingAlgorithm::to_string_matrix(void) const
{
    Matrix<std::string> string_matrix;

    return(string_matrix);
}


// void print(void) const method

/// Prints to the screen the XML-type representation of the training algorithm object.

void TrainingAlgorithm::print(void) const
{
   std::cout << to_string();
}


// void save(const std::string&) const method

/// Saves to a XML-type file the members of the training algorithm object.
/// @param file_name Name of training algorithm XML-type file. 

void TrainingAlgorithm::save(const std::string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const std::string&) method

/// Loads a gradient descent object from a XML-type file.
/// Please mind about the file format, wich is specified in the User's Guide. 
/// @param file_name Name of training algorithm XML-type file. 

void TrainingAlgorithm::load(const std::string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void load(const std::string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw std::logic_error(buffer.str());
   }

   from_XML(document);
}


// void initialize_random(void) method

/// Default random initialization for a training algorithm object.
/// It just sets a random display value.

void TrainingAlgorithm::initialize_random(void)
{
   display = true;
}

// std::string write_stopping_condition(void) const method

/// Return a string with the stopping condition of the TrainingAlgorithmResults

std::string TrainingAlgorithm::TrainingAlgorithmResults::write_stopping_condition(void) const
{
    switch (stopping_condition)
    {
    case MinimumParametersIncrementNorm:
    {
        return ("Minimum parameters increment norm");
    }
    case MinimumPerformanceIncrease:
    {
        return("Minimum loss increase");
    }
    case PerformanceGoal:
    {
        return("Performance goal");
    }
    case GradientNormGoal:
    {
        return("Gradient norm goal");
    }
    case MaximumSelectionPerformanceDecreases:
    {
        return("Maximum selection failures");
    }
    case MaximumIterationsNumber:
    {
        return("Maximum number of iterations");
    }
    case MaximumTime:
    {
        return("Maximum training time");
    }
    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingAlgorithmResults struct.\n"
               << "std::string write_stopping_condition(void) const method.\n"
               << "Unknown stopping condition type.\n";

        throw std::logic_error(buffer.str());

        break;
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
