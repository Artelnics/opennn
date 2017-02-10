/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   R O O T   M E A N   S Q U A R E D   E R R O R   C L A S S                                                  */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
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

RootMeanSquaredError::RootMeanSquaredError(void) : ErrorTerm()
{
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a root mean squared error associated to a neural network object but not to a data set object. 
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

RootMeanSquaredError::RootMeanSquaredError(NeuralNetwork* new_neural_network_pointer)
: ErrorTerm(new_neural_network_pointer)
{
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a root mean squared error associated to a data set object but not to a neural network object. 
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

RootMeanSquaredError::RootMeanSquaredError(DataSet* new_data_set_pointer)
: ErrorTerm(new_data_set_pointer)
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
: ErrorTerm(new_neural_network_pointer, new_data_set_pointer)
{
}


// XML CONSTRUCTOR

/// XML constructor. 
/// This constructor creates a root mean squared object neither associated to a neural network nor a data set. 
/// It also loads the member data from a XML document.
/// @param root_mean_squared_error_document TinyXML document with the object members. 

RootMeanSquaredError::RootMeanSquaredError(const tinyxml2::XMLDocument& root_mean_squared_error_document)
: ErrorTerm(root_mean_squared_error_document)
{
}


// DESTRUCTOR

/// Destructor.

RootMeanSquaredError::~RootMeanSquaredError(void)
{
}


// METHODS

// void check(void) const method

/// Checks that there are a neural network and a data set associated to the root mean squared error, 
/// and that the numbers of inputs and outputs in the neural network are equal to the numbers of inputs and targets in the data set. 
/// If some of the above conditions is not hold, the method throws an exception. 

void RootMeanSquaredError::check(void) const
{
   std::ostringstream buffer;

   // Neural network stuff

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: RootMeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to neural network is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   if(!multilayer_perceptron_pointer)
   {
      buffer << "OpenNN Exception: RootMeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to multilayer perceptron is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   if(inputs_number == 0)
   {
      buffer << "OpenNN Exception: RootMeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Number of inputs in multilayer perceptron object is zero.\n";

      throw std::logic_error(buffer.str());	  
   }

   if(outputs_number == 0)
   {
      buffer << "OpenNN Exception: RootMeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Number of outputs in multilayer perceptron object is zero.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Data set stuff

   if(!data_set_pointer)
   {
      buffer << "OpenNN Exception: RootMeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to data set is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   // Sum squared error stuff

   const Variables& variables = data_set_pointer->get_variables();

   const size_t data_set_inputs_number = variables.count_inputs_number();
   const size_t targets_number = variables.count_targets_number();

   if(data_set_inputs_number != inputs_number)
   {
      buffer << "OpenNN Exception: RootMeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Number of inputs in neural network (" << inputs_number << ") must be equal to number of inputs in data set (" << data_set_inputs_number << ").\n";

      throw std::logic_error(buffer.str());	  
   }

   if(outputs_number != targets_number)
   {
      buffer << "OpenNN Exception: RootMeanSquaredError class.\n"
             << "void check(void) const method.\n"
             << "Number of outputs in neural network must be equal to number of targets in data set.\n";

      throw std::logic_error(buffer.str());
   }
}


// double calculate_error(void) const method

/// Returns the loss value of a neural network according to the root mean squared error 
/// on the training instances of a data set.

double RootMeanSquaredError::calculate_error(void) const
{
   // Control sentence

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Data set stuff

   const Matrix<double>& data = data_set_pointer->get_data();

   const Instances& instances = data_set_pointer->get_instances();

   const size_t training_instances_number = instances.count_training_instances_number();

   const Vector<size_t> training_indices = instances.arrange_training_indices();

   size_t training_index;

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   // Root mean squared error

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   double sum_squared_error = 0.0;

   int i = 0;

   #pragma omp parallel for private(i, training_index, inputs, outputs, targets) reduction(+:sum_squared_error)

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

      // Input vector

      inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

//      targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Sum squaresd error

//	  sum_squared_error += outputs.calculate_sum_squared_error(targets);
      sum_squared_error += outputs.calculate_sum_squared_error(data, training_index, targets_indices);

   }

   return(sqrt(sum_squared_error/(double)training_instances_number));
}


// double calculate_error(const Vector<double>&) const method

/// Returns which would be the loss of a multilayer perceptron for an hypothetical
/// vector of parameters. It does not set that vector of parameters to the multilayer perceptron. 
/// @param parameters Vector of potential parameters for the multilayer perceptron associated 
/// to the error term.

double RootMeanSquaredError::calculate_error(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 
   
   check();

   #endif

   #ifdef __OPENNN_DEBUG__ 

   std::ostringstream buffer;

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      buffer << "OpenNN Exception: RootMeanSquaredError class.\n"
             << "double calculate_error(const Vector<double>&) const method.\n"
             << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ").\n";

      throw std::logic_error(buffer.str());	  
   }

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const size_t training_instances_number = instances.count_training_instances_number();

   const Vector<size_t> training_indices = instances.arrange_training_indices();

   size_t training_index;

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   // Root mean squared error

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   double sum_squared_error = 0.0;

   int i = 0;

   #pragma omp parallel for private(i, training_index, inputs, outputs, targets) reduction(+:sum_squared_error)

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

      // Input vector

      inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

      // Target vector

      targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Sum squaresd error

      sum_squared_error += outputs.calculate_sum_squared_error(targets);
   }

   return(sqrt(sum_squared_error/(double)training_instances_number));
}


// Vector<double> calculate_output_gradient(void) const method

/// Calculates the gradient the root mean squared error funcion by means of the back-propagation algorithm.

Vector<double> RootMeanSquaredError::calculate_output_gradient(const Vector<double>& output, const Vector<double>& target) const
{
    const Vector<double>  output_gradient = (output-target);

    return(output_gradient);
}


// double calculate_selection_error(void) const method

/// Returns the root mean squared error of the multilayer perceptron measured on the selection instances of the data set.

double RootMeanSquaredError::calculate_selection_error(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   // Neural network staff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();
   const size_t selection_instances_number = instances.count_selection_instances_number();

   if(selection_instances_number == 0)
   {
      return(0.0);
   }

   const Vector<size_t> selection_indices = instances.arrange_selection_indices();

   size_t selection_index;

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   int i = 0;

   double selection_loss = 0.0;

   #pragma omp parallel for private(i, selection_index, inputs, outputs, targets) reduction(+ : selection_loss)

   for(i = 0; i < (int)selection_instances_number; i++)
   {
       selection_index = selection_indices[i];

      // Input vector

      inputs = data_set_pointer->get_instance(selection_index, inputs_indices);

      // Output vector

      outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      targets = data_set_pointer->get_instance(selection_index, targets_indices);

      // Sum of squares error

      selection_loss += outputs.calculate_sum_squared_error(targets);
   }

   return(sqrt(selection_loss/(double)selection_instances_number));
}


// Matrix<double> calculate_output_Hessian(const Vector<double>&, const Vector<double>&) const method

Matrix<double> RootMeanSquaredError::calculate_output_Hessian(const Vector<double>& output, const Vector<double>& target) const
{
    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const double loss = calculate_error();

    const size_t outputs_number = neural_network_pointer->get_multilayer_perceptron_pointer()->get_outputs_number();

    const Vector<double> one_vector(outputs_number,1.0);

    const Vector<double> diagonal = one_vector-((output-target)*(output-target))/(training_instances_number*training_instances_number*loss*loss);

    Matrix<double> output_Hessian(outputs_number, outputs_number,0.0);
    output_Hessian.set_diagonal(diagonal);

    return(output_Hessian);
}


// Vector<double> calculate_gradient(void) const method

/// Returns the root mean squared error function gradient of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.

Vector<double> RootMeanSquaredError::calculate_gradient(void) const
{
    // Control sentence

       #ifdef __OPENNN_DEBUG__

       check();

       #endif

       // Neural network stuff

       const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

       const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
       const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

       const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

       const size_t parameters_number = multilayer_perceptron_pointer->count_parameters_number();

       // Data set stuff

       Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

       const bool has_conditions_layer = neural_network_pointer->has_conditions_layer();

       const ConditionsLayer* conditions_layer_pointer = has_conditions_layer ? neural_network_pointer->get_conditions_layer_pointer() : NULL;

       Vector<double> particular_solution;
       Vector<double> homogeneous_solution;

       // Data set stuff

       const Instances& instances = data_set_pointer->get_instances();

       const size_t training_instances_number = instances.count_training_instances_number();

       const Vector<size_t> training_indices = instances.arrange_training_indices();

       size_t training_index;

       const Variables& variables = data_set_pointer->get_variables();

       const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
       const Vector<size_t> targets_indices = variables.arrange_targets_indices();

       const MissingValues& missing_values = data_set_pointer->get_missing_values();

       Vector<double> inputs(inputs_number);
       Vector<double> targets(outputs_number);

       // Loss index stuff

       const double loss = calculate_error();

       Vector< Vector<double> > layers_delta;

       Vector<double> output_gradient(outputs_number);

       Vector<double> point_gradient(parameters_number, 0.0);

       // Main loop

       Vector<double> gradient(parameters_number, 0.0);

       int i = 0;

       #pragma omp parallel for private(i, training_index, inputs, targets, first_order_forward_propagation, output_gradient, \
        layers_delta, particular_solution, homogeneous_solution, point_gradient)

       for(i = 0; i < (int)training_instances_number; i++)
       {
           training_index = training_indices[i];

           if(missing_values.has_missing_values(training_index))
           {
               continue;
           }

          inputs = data_set_pointer->get_instance(training_index, inputs_indices);

          targets = data_set_pointer->get_instance(training_index, targets_indices);

          first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

          const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
          const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

          if(!has_conditions_layer)
          {
             output_gradient = (layers_activation[layers_number-1]-targets)/(training_instances_number*loss);

             layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);
          }
          else
          {
             particular_solution = conditions_layer_pointer->calculate_particular_solution(inputs);
             homogeneous_solution = conditions_layer_pointer->calculate_homogeneous_solution(inputs);

             output_gradient = (particular_solution+homogeneous_solution*layers_activation[layers_number-1] - targets)/(training_instances_number*loss);

             layers_delta = calculate_layers_delta(layers_activation_derivative, homogeneous_solution, output_gradient);
          }

          point_gradient = calculate_point_gradient(inputs, layers_activation, layers_delta);

          #pragma omp critical

          gradient += point_gradient;
       }

       return(gradient);
}

// Vector<double> calculate_gradient(const double&, const double&) const method

/// Returns the root mean squared error function gradient of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.
/// This method is used with the calculation with MPI.
/// @param total_training_instances_number Number of total training instances in all the processors.
/// @param loss Loss of the neural network.

Vector<double> RootMeanSquaredError::calculate_gradient(const double& total_training_instances_number, const double& loss) const
{
    // Control sentence

       #ifdef __OPENNN_DEBUG__

       check();

       #endif

       // Neural network stuff

       const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

       const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
       const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

       const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

       const size_t parameters_number = multilayer_perceptron_pointer->count_parameters_number();

       // Data set stuff

       Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

       const bool has_conditions_layer = neural_network_pointer->has_conditions_layer();

       const ConditionsLayer* conditions_layer_pointer = has_conditions_layer ? neural_network_pointer->get_conditions_layer_pointer() : NULL;

       Vector<double> particular_solution;
       Vector<double> homogeneous_solution;

       // Data set stuff

       const Instances& instances = data_set_pointer->get_instances();

       const size_t training_instances_number = instances.count_training_instances_number();

       const Vector<size_t> training_indices = instances.arrange_training_indices();

       size_t training_index;

       const Variables& variables = data_set_pointer->get_variables();

       const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
       const Vector<size_t> targets_indices = variables.arrange_targets_indices();

       const MissingValues& missing_values = data_set_pointer->get_missing_values();

       Vector<double> inputs(inputs_number);
       Vector<double> targets(outputs_number);

       // Loss index stuff

       Vector< Vector<double> > layers_delta;

       Vector<double> output_gradient(outputs_number);

       Vector<double> point_gradient(parameters_number, 0.0);

       // Main loop

       Vector<double> gradient(parameters_number, 0.0);

       int i = 0;

       #pragma omp parallel for private(i, training_index, inputs, targets, first_order_forward_propagation, output_gradient, \
        layers_delta, particular_solution, homogeneous_solution, point_gradient)

       for(i = 0; i < (int)training_instances_number; i++)
       {
           training_index = training_indices[i];

           if(missing_values.has_missing_values(training_index))
           {
               continue;
           }

          inputs = data_set_pointer->get_instance(training_index, inputs_indices);

          targets = data_set_pointer->get_instance(training_index, targets_indices);

          first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

          const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
          const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

          if(!has_conditions_layer)
          {
             output_gradient = (layers_activation[layers_number-1]-targets)/(total_training_instances_number*loss);

             layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);
          }
          else
          {
             particular_solution = conditions_layer_pointer->calculate_particular_solution(inputs);
             homogeneous_solution = conditions_layer_pointer->calculate_homogeneous_solution(inputs);

             output_gradient = (particular_solution+homogeneous_solution*layers_activation[layers_number-1] - targets)/(total_training_instances_number*loss);

             layers_delta = calculate_layers_delta(layers_activation_derivative, homogeneous_solution, output_gradient);
          }

          point_gradient = calculate_point_gradient(inputs, layers_activation, layers_delta);

          #pragma omp critical

          gradient += point_gradient;
       }

       return(gradient);
}

// std::string write_error_term_type(void) const method

/// Returns a string with the name of the root mean squared error loss type, "ROOT_MEAN_SQUARED_ERROR".

std::string RootMeanSquaredError::write_error_term_type(void) const
{
   return("ROOT_MEAN_SQUARED_ERROR");
}


// tinyxml2::XMLDocument* to_XML(void) const method 

/// Serializes the root mean squared error object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element. 

tinyxml2::XMLDocument* RootMeanSquaredError::to_XML(void) const
{
   std::ostringstream buffer;

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
        std::ostringstream buffer;

        buffer << "OpenNN Exception: RootMeanSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Root mean squared error element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

  // Display
  {
     const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

     if(element)
     {
        const std::string new_display_string = element->GetText();

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
