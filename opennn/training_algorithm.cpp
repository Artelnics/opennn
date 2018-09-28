/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   A L G O R I T H M   C L A S S                                                            */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
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

TrainingAlgorithm::TrainingAlgorithm()
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

TrainingAlgorithm::~TrainingAlgorithm()
{ 
}


// ASSIGNMENT OPERATOR

/// Assignment operator.
/// It assigns to this object the members of an existing training algorithm object.
/// @param other_training_algorithm Training algorithm object to be assigned.

TrainingAlgorithm& TrainingAlgorithm::operator =(const TrainingAlgorithm& other_training_algorithm)
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

bool TrainingAlgorithm::operator ==(const TrainingAlgorithm& other_training_algorithm) const
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

// LossIndex* get_loss_index_pointer() const method

/// Returns a pointer to the loss functional object to which the training algorithm is
/// associated.

LossIndex* TrainingAlgorithm::get_loss_index_pointer() const
{
    #ifdef __OPENNN_DEBUG__

    if(!loss_index_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
               << "LossIndex* get_loss_index_pointer() const method.\n"
               << "Loss index pointer is NULL.\n";

        throw logic_error(buffer.str());
    }

    #endif

   return(loss_index_pointer);
}


// bool has_loss_index() const method

/// Returns true if this training algorithm object has an associated loss functional object,
/// and false otherwise.

bool TrainingAlgorithm::has_loss_index() const
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


// const bool& get_display() const method

/// Returns true if messages from this class can be displayed on the screen, or false if messages from
/// this class can't be displayed on the screen.

const bool& TrainingAlgorithm::get_display() const
{
   return(display);
}


// const size_t& get_display_period() const method

/// Returns the number of iterations between the training showing progress.

const size_t& TrainingAlgorithm::get_display_period() const
{
   return(display_period);
}


// const size_t& get_save_period() const method

/// Returns the number of iterations between the training saving progress.

const size_t& TrainingAlgorithm::get_save_period() const
{
   return(save_period);
}


// const size_t& get_neural_network_filename() const method

/// Returns the file name where the neural network will be saved.

const string& TrainingAlgorithm::get_neural_network_file_name() const
{
   return(neural_network_file_name);
}


// void set() method

/// Sets the loss functional pointer to NULL.
/// It also sets the rest of members to their default values. 

void TrainingAlgorithm::set()
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
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_display_period <= 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
             << "void set_display_period(const size_t&) method.\n"
             << "Display period must be greater than 0.\n";

      throw logic_error(buffer.str());
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
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_save_period <= 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ConjugateGradient class.\n"
             << "void set_save_period(const size_t&) method.\n"
             << "Save period must be greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   save_period = new_save_period;
}


// void set_neural_network_file_name(const string&) method

/// Sets a new file name where the neural network will be saved.
/// @param new_neural_network_file_name
/// File name for the neural network object.

void TrainingAlgorithm::set_neural_network_file_name(const string& new_neural_network_file_name)
{
   neural_network_file_name = new_neural_network_file_name;
}


// void set_default() method 

/// Sets the members of the training algorithm object to their default values.

void TrainingAlgorithm::set_default()
{
   display = true;

   display_period = 5;

   save_period = UINT_MAX;

   neural_network_file_name = "neural_network.xml";
}


// string write_training_algorithm_type() const method

/// This method writes a string with the type of training algoritm.

string TrainingAlgorithm::write_training_algorithm_type() const
{
   return("USER_TRAINING_ALGORITHM");
}


// void check() const method

/// Performs a default checking for training algorithms.
/// In particular, it checks that the loss functional pointer associated to the training algorithm is not NULL,
/// and that the neural network associated to that loss functional is neither NULL.
/// If that checkings are not hold, an exception is thrown. 

void TrainingAlgorithm::check() const
{
   ostringstream buffer;

   if(!loss_index_pointer)
   {
      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void check() const method.\n"
             << "Pointer to loss functional is NULL.\n";

      throw logic_error(buffer.str());	  
   }

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   if(neural_network_pointer == NULL)
   {
      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void check() const method.\n"
             << "Pointer to neural network is NULL.\n";

      throw logic_error(buffer.str());
   }
}


// tinyxml2::XMLDocument* to_XML() const method

/// Serializes a default training algorithm object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document.

tinyxml2::XMLDocument* TrainingAlgorithm::to_XML() const
{
    ostringstream buffer;

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
    ostringstream buffer;

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
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Training algorithm element is NULL.\n";

        throw logic_error(buffer.str());
    }

  // Display
  {
     const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

     if(display_element)
     {
        const string new_display_string = display_element->GetText();

        try
        {
           set_display(new_display_string != "0");
        }
        catch(const logic_error& e)
        {
           cout << e.what() << endl;
        }
     }
  }

}


// string object_to_string() const method

/// Returns a default string representation of a training algorithm.

string TrainingAlgorithm::object_to_string() const
{
   ostringstream buffer;

   buffer << "Training strategy\n" 
          << "Display: " << display << "\n";

   return(buffer.str());
}


// Matrix<string> to_string_matrix() const method

/// Returns a default(empty) string matrix containing the members
/// of the training algorithm object.

Matrix<string> TrainingAlgorithm::to_string_matrix() const
{
    Matrix<string> string_matrix;

    return(string_matrix);
}


// void print() const method

/// Prints to the screen the XML-type representation of the training algorithm object.

void TrainingAlgorithm::print() const
{
   cout << object_to_string();
}


// void save(const string&) const method

/// Saves to a XML-type file the members of the training algorithm object.
/// @param file_name Name of training algorithm XML-type file. 

void TrainingAlgorithm::save(const string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const string&) method

/// Loads a gradient descent object from a XML-type file.
/// Please mind about the file format, wich is specified in the User's Guide. 
/// @param file_name Name of training algorithm XML-type file. 

void TrainingAlgorithm::load(const string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: TrainingAlgorithm class.\n"
             << "void load(const string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw logic_error(buffer.str());
   }

   from_XML(document);
}


// void initialize_random() method

/// Default random initialization for a training algorithm object.
/// It just sets a random display value.

void TrainingAlgorithm::initialize_random()
{
   display = true;
}


// string write_stopping_condition() const method

/// Return a string with the stopping condition of the TrainingAlgorithmResults

string TrainingAlgorithm::TrainingAlgorithmResults::write_stopping_condition() const
{
    switch(stopping_condition)
    {
    case MinimumParametersIncrementNorm:
    {
        return("Minimum parameters increment norm");
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
        ostringstream buffer;

        buffer << "OpenNN Exception: TrainingAlgorithmResults struct.\n"
               << "string write_stopping_condition() const method.\n"
               << "Unknown stopping condition type.\n";

        throw logic_error(buffer.str());

        break;
    }
    }

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
