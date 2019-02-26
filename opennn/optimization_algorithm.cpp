/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S                                                    */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "optimization_algorithm.h"

#ifdef __OPENNN_CUDA__
#include <cuda_runtime.h>
#endif

namespace OpenNN
{


// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a optimization algorithm object not associated to any loss index object.  

OptimizationAlgorithm::OptimizationAlgorithm()
 : loss_index_pointer(nullptr)
{ 
   set_default();
}


// GENERAL CONSTRUCTOR

/// General constructor. 
/// It creates a optimization algorithm object associated to a loss index object.
/// @param new_loss_index_pointer Pointer to a loss index object.

OptimizationAlgorithm::OptimizationAlgorithm(LossIndex* new_loss_index_pointer)
 : loss_index_pointer(new_loss_index_pointer)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a optimization algorithm object not associated to any loss index object. 
/// It also loads the other members from a XML document.

OptimizationAlgorithm::OptimizationAlgorithm(const tinyxml2::XMLDocument& document)
 : loss_index_pointer(nullptr)
{ 
   from_XML(document);
}


// DESTRUCTOR 

/// Destructor

OptimizationAlgorithm::~OptimizationAlgorithm()
{ 
}


// ASSIGNMENT OPERATOR

/// Assignment operator.
/// It assigns to this object the members of an existing optimization algorithm object.
/// @param other_optimization_algorithm Optimization algorithm object to be assigned.

OptimizationAlgorithm& OptimizationAlgorithm::operator = (const OptimizationAlgorithm& other_optimization_algorithm)
{
   if(this != &other_optimization_algorithm)
   {
      loss_index_pointer = other_optimization_algorithm.loss_index_pointer;

      display = other_optimization_algorithm.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR

/// Equal to operator.
/// @param other_optimization_algorithm Optimization algorithm object to be compared with.

bool OptimizationAlgorithm::operator == (const OptimizationAlgorithm& other_optimization_algorithm) const
{
   if(loss_index_pointer == other_optimization_algorithm.loss_index_pointer
   && display == other_optimization_algorithm.display)
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

/// Returns a pointer to the loss index object to which the optimization algorithm is
/// associated.

LossIndex* OptimizationAlgorithm::get_loss_index_pointer() const
{
    #ifdef __OPENNN_DEBUG__

    if(!loss_index_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OptimizationAlgorithm class.\n"
               << "LossIndex* get_loss_index_pointer() const method.\n"
               << "Loss index pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    #endif

   return(loss_index_pointer);
}


// bool has_loss_index() const method

/// Returns true if this optimization algorithm object has an associated loss index object,
/// and false otherwise.

bool OptimizationAlgorithm::has_loss_index() const
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

const bool& OptimizationAlgorithm::get_display() const
{
   return(display);
}


// const size_t& get_display_period() const method

/// Returns the number of iterations between the training showing progress.

const size_t& OptimizationAlgorithm::get_display_period() const
{
   return(display_period);
}


/// Returns the number of iterations between the training saving progress.

const size_t& OptimizationAlgorithm::get_save_period() const
{
   return(save_period);
}


/// Returns the file name where the neural network will be saved.

const string& OptimizationAlgorithm::get_neural_network_file_name() const
{
   return(neural_network_file_name);
}


/// Sets the loss index pointer to nullptr.
/// It also sets the rest of members to their default values. 

void OptimizationAlgorithm::set()
{
   loss_index_pointer = nullptr;

   set_default();
}


/// Sets a new loss index pointer.
/// It also sets the rest of members to their default values. 
/// @param new_loss_index_pointer Pointer to a loss index object. 

void OptimizationAlgorithm::set(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;

   set_default();
}

void OptimizationAlgorithm::set_training_batch_size(const size_t& new_training_batch_size)
{
    training_batch_size = new_training_batch_size;
}


void OptimizationAlgorithm::set_selection_batch_size(const size_t& new_selection_batch_size)
{
    selection_batch_size = new_selection_batch_size;
}


/// Sets a pointer to a loss index object to be associated to the optimization algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void OptimizationAlgorithm::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void OptimizationAlgorithm::set_display(const bool& new_display)
{
   display = new_display;
}


/// Sets a new number of iterations between the training showing progress.
/// @param new_display_period
/// Number of iterations between the training showing progress.

void OptimizationAlgorithm::set_display_period(const size_t& new_display_period)
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


/// Sets a new number of iterations between the training saving progress.
/// @param new_save_period
/// Number of iterations between the training saving progress.

void OptimizationAlgorithm::set_save_period(const size_t& new_save_period)
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


/// Sets a new file name where the neural network will be saved.
/// @param new_neural_network_file_name
/// File name for the neural network object.

void OptimizationAlgorithm::set_neural_network_file_name(const string& new_neural_network_file_name)
{
   neural_network_file_name = new_neural_network_file_name;
}


/// Sets the members of the optimization algorithm object to their default values.

void OptimizationAlgorithm::set_default()
{
   display = true;

   display_period = 5;

   save_period = UINT_MAX;

   neural_network_file_name = "neural_network.xml";
}


/// Performs a default checking for optimization algorithms.
/// In particular, it checks that the loss index pointer associated to the optimization algorithm is not nullptr,
/// and that the neural network associated to that loss index is neither nullptr.
/// If that checkings are not hold, an exception is thrown. 

void OptimizationAlgorithm::check() const
{
   ostringstream buffer;

   if(!loss_index_pointer)
   {
      buffer << "OpenNN Exception: OptimizationAlgorithm class.\n"
             << "void check() const method.\n"
             << "Pointer to loss index is nullptr.\n";

      throw logic_error(buffer.str());	  
   }

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   if(neural_network_pointer == nullptr)
   {
      buffer << "OpenNN Exception: OptimizationAlgorithm class.\n"
             << "void check() const method.\n"
             << "Pointer to neural network is nullptr.\n";

      throw logic_error(buffer.str());
   }
}


// tinyxml2::XMLDocument* to_XML() const method

/// Serializes a default optimization algorithm object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document.

tinyxml2::XMLDocument* OptimizationAlgorithm::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Nueral network outputs integrals

    tinyxml2::XMLElement* optimization_algorithm_element = document->NewElement("OptimizationAlgorithm");

    document->InsertFirstChild(optimization_algorithm_element);

    // Display
    {
       tinyxml2::XMLElement* element = document->NewElement("Display");
       optimization_algorithm_element->LinkEndChild(element);

       buffer.str("");
       buffer << display;

       tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
       element->LinkEndChild(text);
    }

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the optimization algorithm object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void OptimizationAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("OptimizationAlgorithm");

    // Display

    file_stream.OpenElement("Display");

    buffer.str("");
    buffer << display;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Loads a default optimization algorithm from a XML document.
/// @param document TinyXML document containing the error term members.

void OptimizationAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("OptimizationAlgorithm");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OptimizationAlgorithm class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Optimization algorithm element is nullptr.\n";

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
           cerr << e.what() << endl;
        }
     }
  }

}


// string object_to_string() const method

/// Returns a default string representation of a optimization algorithm.

string OptimizationAlgorithm::object_to_string() const
{
   ostringstream buffer;

   buffer << "Training strategy\n" 
          << "Display: " << display << "\n";

   return(buffer.str());
}


// Matrix<string> to_string_matrix() const method

/// Returns a default(empty) string matrix containing the members
/// of the optimization algorithm object.

Matrix<string> OptimizationAlgorithm::to_string_matrix() const
{
    Matrix<string> string_matrix;

    return(string_matrix);
}


// void print() const method

/// Prints to the screen the XML-type representation of the optimization algorithm object.

void OptimizationAlgorithm::print() const
{
   cout << object_to_string();
}


// void save(const string&) const method

/// Saves to a XML-type file the members of the optimization algorithm object.
/// @param file_name Name of optimization algorithm XML-type file. 

void OptimizationAlgorithm::save(const string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const string&) method

/// Loads a gradient descent object from a XML-type file.
/// Please mind about the file format, wich is specified in the User's Guide. 
/// @param file_name Name of optimization algorithm XML-type file. 

void OptimizationAlgorithm::load(const string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OptimizationAlgorithm class.\n"
             << "void load(const string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw logic_error(buffer.str());
   }

   from_XML(document);
}


// void initialize_random() method

/// Default random initialization for a optimization algorithm object.
/// It just sets a random display value.

void OptimizationAlgorithm::initialize_random()
{
   display = true;
}


// string write_stopping_condition() const method

/// Return a string with the stopping condition of the OptimizationAlgorithmResults

string OptimizationAlgorithm::OptimizationAlgorithmResults::write_stopping_condition() const
{
    switch(stopping_condition)
    {
    case MinimumParametersIncrementNorm:
    {
        return("Minimum parameters increment norm");
    }
    case MinimumLossDecrease:
    {
        return("Minimum loss decrease");
    }
    case LossGoal:
    {
        return("Loss goal");
    }
    case GradientNormGoal:
    {
        return("Gradient norm goal");
    }
    case MaximumSelectionErrorIncreases:
    {
        return("Maximum selection error increases");
    }
    case MaximumIterationsNumber:
    {
        return("Maximum number of iterations");
    }
    case MaximumTime:
    {
        return("Maximum training time");
    }
    }

    return string();
}


bool OptimizationAlgorithm::check_cuda() const
{
#ifdef __OPENNN_CUDA__

    int deviceCount;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;

    const cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);

    if(cudaResultCode != cudaSuccess)
    {
        deviceCount = 0;
    }

    for(int device = 0; device < deviceCount; ++device)
    {
        cudaGetDeviceProperties(&properties, device);

        cout << properties.major << "." << properties.minor << endl;
        if(properties.major != 9999) /* 9999 means emulation only */
        {
            ++gpuDeviceCount;
        }
        else if(properties.major > 3)
        {
            ++gpuDeviceCount;
        }
        else if(properties.major == 3 && properties.minor >= 5)
        {
            ++gpuDeviceCount;
        }
    }

    if(gpuDeviceCount > 0)
    {
        return true;
    }

#endif

    return false;
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
