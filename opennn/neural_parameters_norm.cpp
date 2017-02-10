/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N E U R A L   P A R A M E T E R S   N O R M   C L A S S                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "neural_parameters_norm.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a neural parameters norm functional not associated to any neural network.
/// It also initializes all the rest of class members to their default values.

NeuralParametersNorm::NeuralParametersNorm(void) 
 : RegularizationTerm()
{
   set_default();
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a neural parameters norm functional associated to a neural network.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

NeuralParametersNorm::NeuralParametersNorm(NeuralNetwork* new_neural_network_pointer) 
: RegularizationTerm(new_neural_network_pointer)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a neural parameters norm object not associated to any neural network.
/// The object members are loaded by means of a XML document.
/// Please be careful with the format of that file, which is specified in the OpenNN manual.
/// @param neural_parameters_norm_document TinyXML document with the neural parameters norm elements.

NeuralParametersNorm::NeuralParametersNorm(const tinyxml2::XMLDocument& neural_parameters_norm_document)
 : RegularizationTerm()
{
   set_default();

   from_XML(neural_parameters_norm_document);
}


// DESTRUCTOR

/// Destructor.
/// This destructor does not delete any pointer. 

NeuralParametersNorm::~NeuralParametersNorm(void) 
{
}


// METHODS

// const double& get_neural_parameters_norm_weight(void) const method

/// Returns the weight value for the neural parameters norm in the error term expression. 

const double& NeuralParametersNorm::get_neural_parameters_norm_weight(void) const
{
   return(neural_parameters_norm_weight);
}


// void set_neural_parameters_norm_weight(const double&) method

/// Sets a new weight value for the neural parameters norm in the error term expression. 

void NeuralParametersNorm::set_neural_parameters_norm_weight(const double& new_neural_parameters_norm_weight)
{
   neural_parameters_norm_weight = new_neural_parameters_norm_weight;
}


// void set_default(void) method

/// Sets the default values for the neural parameters norm object:
/// <ul>
/// <li> Neural parameters norm weight: 0.1.
/// <li> Display: true.
/// </ul>

void NeuralParametersNorm::set_default(void)
{
   neural_parameters_norm_weight = 1.0e-3;

   display = true;
}


// void check(void) const method

/// Checks that there is a neural network associated to this error term,
/// and that there is a multilayer perceptron in the neural network. 
/// If some of the above conditions is not hold, the method throws an exception. 

void NeuralParametersNorm::check(void) const
{
   std::ostringstream buffer;

   // Neural network stuff

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: NeuralParametersNorm class.\n"
             << "void check(void) const method.\n"
             << "Pointer to neural network is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   if(!multilayer_perceptron_pointer)
   {
      buffer << "OpenNN Exception: NeuralParametersNorm class.\n"
             << "void check(void) const method.\n"
             << "Pointer to multilayer perceptron is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   if(inputs_number == 0)
   {
      buffer << "OpenNN Exception: NeuralParametersNorm class.\n"
             << "void check(void) const method.\n"
             << "Number of inputs in multilayer perceptron object is zero.\n";

      throw std::logic_error(buffer.str());	  
   }

   if(outputs_number == 0)
   {
      buffer << "OpenNN Exception: NeuralParametersNorm class.\n"
             << "void check(void) const method.\n"
             << "Number of outputs in multilayer perceptron object is zero.\n";

      throw std::logic_error(buffer.str());	  
   }

}


// double calculate_regularization(void) const method

/// Returns the loss of this peformance term. 
/// It is equal to the weighted norm of the parameters from the associated neural network.

double NeuralParametersNorm::calculate_regularization(void) const
{
   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const Vector<double> neural_parameters = multilayer_perceptron_pointer->arrange_parameters();

   const double neural_parameters_norm = neural_parameters.calculate_norm();

   return(neural_parameters_norm_weight*neural_parameters_norm);
}


// Vector<double> calculate_gradient(void) const method

/// Calculates the objective gradient by means of the back-propagation algorithm, 
/// and returns it in a single vector of size the number of neural parameters.

/// @todo Case including independent parameters.

Vector<double> NeuralParametersNorm::calculate_gradient(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const Vector<double> neural_parameters = multilayer_perceptron_pointer->arrange_parameters();

   return(neural_parameters.calculate_norm_gradient()*neural_parameters_norm_weight);
}


// Matrix<double> calculate_Hessian(void) const method

/// Calculates the objective Hessian by means of the back-propagation algorithm, 
/// and returns it in a single symmetric matrix of size the number of multilayer perceptron parameters. 

/// @todo Second derivatives.
/// @todo Case including independent parameters.

Matrix<double> NeuralParametersNorm::calculate_Hessian(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const Vector<double> neural_parameters = multilayer_perceptron_pointer->arrange_parameters();

   return(neural_parameters.calculate_norm_Hessian()*neural_parameters_norm_weight);
}


// double calculate_regularization(const Vector<double>&) method

/// Returns the neural parameters norm value of a neural network for a vector of parameters.
/// It does not set that vector of parameters to the neural network.
/// @param parameters Vector of parameters for the neural network associated to the error term.

double NeuralParametersNorm::calculate_regularization(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   if(neural_network_pointer->has_independent_parameters())
   {
       const MultilayerPerceptron* multilayer_perceptron_pointer =  neural_network_pointer->get_multilayer_perceptron_pointer();

       const size_t neural_parameters_number = multilayer_perceptron_pointer->count_parameters_number();

       Vector<double> neural_parameters(parameters);
       neural_parameters.resize(neural_parameters_number);

       const double neural_parameters_norm = neural_parameters.calculate_norm();

       return(neural_parameters_norm*neural_parameters_norm_weight);
   }
   else
   {
       const double neural_parameters_norm = parameters.calculate_norm();

       return(neural_parameters_norm*neural_parameters_norm_weight);
   }
}


// Vector<double> calculate_gradient(const Vector<double>&) const method

Vector<double> NeuralParametersNorm::calculate_gradient(const Vector<double>& parameters) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    if(neural_network_pointer->has_independent_parameters())
    {
        const MultilayerPerceptron* multilayer_perceptron_pointer =  neural_network_pointer->get_multilayer_perceptron_pointer();

        const size_t neural_parameters_number = multilayer_perceptron_pointer->count_parameters_number();

        Vector<double> neural_parameters(parameters);
        neural_parameters.resize(neural_parameters_number);

        return(neural_parameters.calculate_norm_gradient()*neural_parameters_norm_weight);
    }
    else
    {
         return(parameters.calculate_norm_gradient()*neural_parameters_norm_weight);
    }
}


// Matrix<double> calculate_Hessian(const Vector<double>&) const method

Matrix<double> NeuralParametersNorm::calculate_Hessian(const Vector<double>& parameters) const
{
    // Control sentence (if debug)

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    if(neural_network_pointer->has_independent_parameters())
    {
        const MultilayerPerceptron* multilayer_perceptron_pointer =  neural_network_pointer->get_multilayer_perceptron_pointer();

        const size_t neural_parameters_number = multilayer_perceptron_pointer->count_parameters_number();

        Vector<double> neural_parameters(parameters);
        neural_parameters.resize(neural_parameters_number);

        return(neural_parameters.calculate_norm_Hessian()*neural_parameters_norm_weight);
    }
    else
    {
         return(parameters.calculate_norm_Hessian()*neural_parameters_norm_weight);
    }
}

/*
// double calculate_selection_loss(void) const method

/// Returns the selection loss of this peformance term.
/// It is equal to the weighted norm of the parameters from the associated neural network.

double NeuralParametersNorm::calculate_selection_loss(void) const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const Vector<double> neural_parameters = multilayer_perceptron_pointer->arrange_parameters();

   return(neural_parameters.calculate_norm()*neural_parameters_norm_weight);
}
*/

// std::string write_error_term_type(void) const method

/// Returns a string with the name of the neural parameters norm loss type,
/// "NEURAL_PARAMETERS_NORM".

std::string NeuralParametersNorm::write_error_term_type(void) const
{
   return("NEURAL_PARAMETERS_NORM");
}


// std::string write_information(void) const method

std::string NeuralParametersNorm::write_information(void) const
{
   std::ostringstream buffer;
   
   buffer << "Regularization norm: " << calculate_regularization() << "\n";

   return(buffer.str());
}


// tinyxml2::XMLDocument* to_XML(void) method method 

/// Returns a representation of the sum squared error object, in XML format. 

tinyxml2::XMLDocument* NeuralParametersNorm::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Neural parameters norm

   tinyxml2::XMLElement* neural_network_parameters_norm_element = document->NewElement("NeuralParametersNorm");

   document->InsertFirstChild(neural_network_parameters_norm_element);

   // Neural parameters norm weight
   {
      tinyxml2::XMLElement* weight_element = document->NewElement("NeuralParametersNormWeight");
      neural_network_parameters_norm_element->LinkEndChild(weight_element);

      buffer.str("");
      buffer << neural_parameters_norm_weight;

      tinyxml2::XMLText* weight_text = document->NewText(buffer.str().c_str());
      weight_element->LinkEndChild(weight_text);
   }

   // Display

//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      neural_network_parameters_norm_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the neural parameters norm object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void NeuralParametersNorm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    //file_stream.OpenElement("NeuralParametersNorm");

    // Neural parameters norm weight

    file_stream.OpenElement("NeuralParametersNormWeight");

    buffer.str("");
    buffer << neural_parameters_norm_weight;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Loads a sum squared error object from a XML document.
/// @param document TinyXML document containing the object members.

void NeuralParametersNorm::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("NeuralParametersNorm");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: NeuralParametersNorm class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Neural parameters norm element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

  // Neural parameters norm weight
  {
     const tinyxml2::XMLElement* element = root_element->FirstChildElement("NeuralParametersNormWeight");

     if(element)
     {
        try
        {
           const double new_neural_parameters_norm_weight = atof(element->GetText());

           set_neural_parameters_norm_weight(new_neural_parameters_norm_weight);
        }
        catch(const std::logic_error& e)
        {
           std::cout << e.what() << std::endl;
        }
     }
  }

  // Display
  {
     const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

     if(element)
     {
        try
        {
           const std::string new_display_string = element->GetText();

           set_display(new_display_string != "0");
        }
        catch(const std::logic_error& e)
        {
           std::cout << e.what() << std::endl;
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
