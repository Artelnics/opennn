/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   R O O T   M E A N   S Q U A R E D   E R R O R   C L A S S                                                  */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/


// OpenNN includes

#include "root_mean_squared_error.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a root mean squared error term object not associated to any 
/// neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

RootMeanSquaredError::RootMeanSquaredError() : LossIndex()
{
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a root mean squared error associated to a neural network object but not to a data set object. 
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

RootMeanSquaredError::RootMeanSquaredError(NeuralNetwork* new_neural_network_pointer)
: LossIndex(new_neural_network_pointer)
{
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a root mean squared error associated to a data set object but not to a neural network object. 
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

RootMeanSquaredError::RootMeanSquaredError(DataSet* new_data_set_pointer)
: LossIndex(new_data_set_pointer)
{
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a root mean squared error term object associated to a 
/// neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

RootMeanSquaredError::RootMeanSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
: LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


// XML CONSTRUCTOR

/// XML constructor. 
/// This constructor creates a root mean squared object neither associated to a neural network nor a data set. 
/// It also loads the member data from a XML document.
/// @param root_mean_squared_error_document TinyXML document with the object members. 

RootMeanSquaredError::RootMeanSquaredError(const tinyxml2::XMLDocument& root_mean_squared_error_document)
: LossIndex(root_mean_squared_error_document)
{
    from_XML(root_mean_squared_error_document);
}


// DESTRUCTOR

/// Destructor.

RootMeanSquaredError::~RootMeanSquaredError()
{
}


// METHODS

/// Returns the loss value of a neural network according to the root mean squared error
/// on the training instances of a data set.

double RootMeanSquaredError::calculate_error(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

   // Control sentence

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Neural network stuff
/*
   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   // Data set stuff

   const Variables& variables = data_set_pointer->get_variables();

   const size_t instances_number = instances_indices.size();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   // Root mean squared error

   double sum_squared_error = 0.0;

   #pragma omp parallel for reduction(+:sum_squared_error)

   for(int i = 0; i < static_cast<int>(instances_number); i++)
   {
       const size_t instance_index = instances_indices[i];

      // Input vector

      const Vector<double> inputs = data_set_pointer->get_instance(instance_index, inputs_indices);

      // Output vector

      const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      const Vector<double> targets = data_set_pointer->get_instance(instance_index, targets_indices);

      // Sum squaresd error

      sum_squared_error += outputs.calculate_sum_squared_error(targets);

   }

   return(sqrt(sum_squared_error/static_cast<double>(instances_number)));
   */
   return 0.0;
}


/// Calculates the gradient the root mean squared error funcion by means of the back-propagation algorithm.

Vector<double> RootMeanSquaredError::calculate_output_gradient(const Vector<size_t>& instances_indices, const Vector<double>& output, const Vector<double>& target) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    /*
    const size_t instances_number = instances_indices.size();

    const double error = calculate_error(instances_indices);

    const Vector<double>  output_gradient = (output-target)/(static_cast<double>(instances_number)*error);

    return(output_gradient);
*/
    return Vector<double>();
}


/*
/// Returns the root mean squared error function gradient of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.

Vector<double> RootMeanSquaredError::calculate_error_gradient(const Vector<size_t>& instances_indices) const
{
    // Control sentence

       #ifdef __OPENNN_DEBUG__

       check();

       #endif

       // Neural network stuff

       const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

       const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

       const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

       const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

       // Data set stuff

       Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

       // Data set stuff

       const size_t instances_number = instances_indices.size();

       const Variables& variables = data_set_pointer->get_variables();

       const Vector<size_t> inputs_indices = variables.get_inputs_indices();
       const Vector<size_t> targets_indices = variables.get_targets_indices();

       const MissingValues& missing_values = data_set_pointer->get_missing_values();

       // Loss index stuff

       const double error = calculate_error(instances_indices);

       Vector< Vector<double> > layers_delta;

       Vector<double> output_gradient(outputs_number);

       Vector<double> point_gradient(parameters_number, 0.0);

       // Main loop

       Vector<double> gradient(parameters_number, 0.0);

       #pragma omp parallel for private(first_order_forward_propagation, output_gradient, \
        layers_delta, point_gradient)

       for(int i = 0; i < static_cast<int>(instances_number); i++)
       {
           const size_t instance_index = instances_indices[i];

           if(missing_values.has_missing_values(instance_index))
           {
               continue;
           }

          const Vector<double> inputs = data_set_pointer->get_instance(instance_index, inputs_indices);

          const Vector<double> targets = data_set_pointer->get_instance(instance_index, targets_indices);

          first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

          const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
          const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

             output_gradient = (layers_activation[layers_number-1]-targets)/(instances_number*error);

             layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

          point_gradient = calculate_point_gradient(inputs, layers_activation, layers_delta);

          #pragma omp critical

          gradient += point_gradient;
       }

       return(gradient);
}

/// Returns the root mean squared error function gradient of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.
/// This method is used with the calculation with MPI.
/// @param total_training_instances_number Number of total training instances in all the processors.
/// @param loss Loss of the neural network.

Vector<double> RootMeanSquaredError::calculate_error_gradient(const double& total_training_instances_number, const double& loss) const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

    // Control sentence

       #ifdef __OPENNN_DEBUG__

       check();

       #endif

       // Neural network stuff

       const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

       const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

       const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

       const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

       // Data set stuff

       Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

       // Data set stuff

       const Instances& instances = data_set_pointer->get_instances();

       const Vector<size_t> training_indices = instances.get_training_indices();

       const size_t training_instances_number = training_indices.size();

       const Variables& variables = data_set_pointer->get_variables();

       const Vector<size_t> inputs_indices = variables.get_inputs_indices();
       const Vector<size_t> targets_indices = variables.get_targets_indices();

       const MissingValues& missing_values = data_set_pointer->get_missing_values();

       // Loss index stuff

       Vector< Vector<double> > layers_delta;

       Vector<double> output_gradient(outputs_number);

       Vector<double> point_gradient(parameters_number, 0.0);

       // Main loop

       Vector<double> gradient(parameters_number, 0.0);

       #pragma omp parallel for private(first_order_forward_propagation, output_gradient, layers_delta, point_gradient)

       for(int i = 0; i < static_cast<int>(training_instances_number); i++)
       {
           const size_t training_index = training_indices[i];

           if(missing_values.has_missing_values(training_index))
           {
               continue;
           }

          const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

          const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

          first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

          const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
          const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

             output_gradient = (layers_activation[layers_number-1]-targets)/(total_training_instances_number*loss);

             layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

          point_gradient = calculate_point_gradient(inputs, layers_activation, layers_delta);

          #pragma omp critical

          gradient += point_gradient;
       }

       return(gradient);
}
*/
// string write_error_term_type() const method

/// Returns a string with the name of the root mean squared error loss type, "ROOT_MEAN_SQUARED_ERROR".

string RootMeanSquaredError::write_error_term_type() const
{
   return("ROOT_MEAN_SQUARED_ERROR");
}


// tinyxml2::XMLDocument* to_XML() const method 

/// Serializes the root mean squared error object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element. 

tinyxml2::XMLDocument* RootMeanSquaredError::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Root mean squared error

   tinyxml2::XMLElement* root_mean_squared_error_element = document->NewElement("RootMeanSquaredError");

   document->InsertFirstChild(root_mean_squared_error_element);

   // Display
//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      root_mean_squared_error_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

void RootMeanSquaredError::write_XML(tinyxml2::XMLPrinter&) const
{
    //file_stream.OpenElement("RootMeanSquaredError");

    //file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Loads a root mean squared error object from a XML document.
/// @param document TinyXML document containing the members of the object.

void RootMeanSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("RootMeanSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RootMeanSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Root mean squared error element is nullptr.\n";

        throw logic_error(buffer.str());
    }

  // Display
  {
     const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

     if(element)
     {
        const string new_display_string = element->GetText();

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
