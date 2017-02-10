/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   R O C   A R E A   E R R O R   C L A S S                                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "roc_area_error.h"


namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a sum squared error term not associated to any neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

RocAreaError::RocAreaError(void) : ErrorTerm()
{
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a sum squared error term associated to a neural network but not measured on any data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

RocAreaError::RocAreaError(NeuralNetwork* new_neural_network_pointer)
: ErrorTerm(new_neural_network_pointer)
{
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a sum squared error not associated to any neural network but to be measured on a data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

RocAreaError::RocAreaError(DataSet* new_data_set_pointer)
: ErrorTerm(new_data_set_pointer)
{
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a sum squared error associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

RocAreaError::RocAreaError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
 : ErrorTerm(new_neural_network_pointer, new_data_set_pointer)
{
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a roc arrea error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from a TinyXML document.
/// @param roc_area_error_document XML document with the class members.

RocAreaError::RocAreaError(const tinyxml2::XMLDocument& roc_area_error_document)
 : ErrorTerm(roc_area_error_document)
{
}



// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a sum roc area error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from another sum squared error object.
/// @param new_roc_area_error Object to be copied.

RocAreaError::RocAreaError(const RocAreaError& new_roc_area_error)
 : ErrorTerm(new_roc_area_error)
{

}


// DESTRUCTOR

/// Destructor.

RocAreaError::~RocAreaError(void)
{
}


// METHODS

// void check(void) const method

/// Checks that there are a neural network and a data set associated to the sum squared error, 
/// and that the numbers of inputs and outputs in the neural network are equal to the numbers of inputs and targets in the data set. 
/// If some of the above conditions is not hold, the method throws an exception. 

void RocAreaError::check(void) const
{
   std::ostringstream buffer;

   // Neural network stuff

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: RocAreaError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to neural network is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   if(!multilayer_perceptron_pointer)
   {
      buffer << "OpenNN Exception: RocAreaError class.\n"
             << "void check(void) const method.\n"
             << "Pointer to multilayer perceptron is NULL.\n";

      throw std::logic_error(buffer.str());	  
   }

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Data set stuff

   if(!data_set_pointer)
   {
      buffer << "OpenNN Exception: RocAreaError class.\n"
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
      buffer << "OpenNN Exception: RocAreaError class.\n"
             << "void check(void) const method.\n"
             << "Number of inputs in neural network (" << inputs_number << ") must be equal to number of inputs in data set (" << data_set_inputs_number << ").\n";

      throw std::logic_error(buffer.str());	  
   }

   if(outputs_number != targets_number)
   {
      buffer << "OpenNN Exception: RocAreaError class.\n"
             << "void check(void) const method.\n"
             << "Number of outputs in neural network must be equal to number of targets in data set.\n";

      throw std::logic_error(buffer.str());
   }
}


// double calculate_error(void) const method

/// Returns the loss value of a neural network according to the sum squared error on a data set.

double RocAreaError::calculate_error(void) const
{
   #ifdef __OPENNN_DEBUG__

   check();

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

   // Sum squared error stuff

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   int i = 0;

//   #pragma omp parallel for private(i, training_index, inputs, outputs, targets) reduction(+ : sum_squared_error)

   double decision_threshold = 0.0;

   double true_positives = 0.0;
   double true_negatives = 0.0;
   double false_positives = 0.0;
   double false_negatives = 0.0;

   const size_t points_number = 101;

   Vector<double> target_data(training_instances_number);
   Vector<double> output_data(training_instances_number);

   Vector<double> true_positive_rate(points_number, 0.0);
   Vector<double> false_positive_rate(points_number, 0.0);

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

       // Input vector

       inputs = data_set_pointer->get_instance(training_index, inputs_indices);

       // Target vector

       targets = data_set_pointer->get_instance(training_index, targets_indices);

       target_data[i] = targets[0];

       // Output vector

       outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

       output_data[i] = outputs[0];
   }

   // Calculate binary classification tests

   for(size_t i = 1; i < points_number-1; i++)
   {
       decision_threshold = i*0.01;

       true_positives = 0.0;
       true_negatives = 0.0;
       false_positives = 0.0;
       false_negatives = 0.0;

       for(size_t j = 0; j < training_instances_number; j++)
       {
           if(target_data[j] == 1.0)
           {
               if(output_data[j] >= decision_threshold)
               {
                   true_positives += output_data[j];
               }
               else
               {
                   false_negatives += (1.0 - output_data[j]);
               }
           }
           else if(target_data[j] == 0.0)
           {
               if(output_data[j] >= decision_threshold)
               {
                   false_positives += output_data[j];
               }
               else
               {
                   true_negatives += (1.0 - output_data[j]);
               }
           }
           else
           {
               std::ostringstream buffer;

               buffer << "OpenNN Exception: RocAreaError class.\n"
                      << "double calculate_error(void) const method.\n"
                      << "Target is not binary.\n";

               throw std::logic_error(buffer.str());
           }
       }

       if(true_negatives+false_positives < 1.0e-12)
       {
           std::ostringstream buffer;

           buffer << "OpenNN Exception: RocAreaError class.\n"
                  << "double calculate_error(void) const method.\n"
                  << "Cannot compute false_positive_rate.\n";

           throw std::logic_error(buffer.str());
       }

       false_positive_rate[i] = false_negatives/(true_positives+false_negatives);

       if(false_positive_rate[i] < 0.0
       || false_positive_rate[i] > 1.0)
       {
           std::ostringstream buffer;

           buffer << "OpenNN Exception: RocAreaError class.\n"
                  << "double calculate_error(void) const method.\n"
                  << "false_positve_rate must be between 0 and 1.\n";

           throw std::logic_error(buffer.str());
       }

       if(true_positives+false_negatives < 1.0e-12)
       {
           std::ostringstream buffer;

           buffer << "OpenNN Exception: RocAreaError class.\n"
                  << "double calculate_error(void) const method.\n"
                  << "Cannot compute true_positive_rate.\n";

           throw std::logic_error(buffer.str());
       }

       true_positive_rate[i] = true_positives/(true_negatives+false_positives);

       if(true_positive_rate[i] < 0.0
       || true_positive_rate[i] > 1.0)
       {
           std::ostringstream buffer;

           buffer << "OpenNN Exception: RocAreaError class.\n"
                  << "double calculate_error(void) const method.\n"
                  << "true_positve_rate must be between 0 and 1.\n";

           throw std::logic_error(buffer.str());
       }

   }

   false_positive_rate[points_number-1] = 1.0;
   true_positive_rate[points_number-1] = 1.0;

   const double roc_area = numerical_integration.calculate_trapezoid_integral(false_positive_rate, true_positive_rate);

   const double roc_area_error = (1.0-roc_area)*(1.0-roc_area);

   return(roc_area_error);
}


// double calculate_error(const Vector<double>&) const method

/// Returns which would be the sum squard error loss of a neural network for an hypothetical vector of parameters. 
/// It does not set that vector of parameters to the neural network. 
/// @param parameters Vector of potential parameters for the neural network associated to the error term.

double RocAreaError::calculate_error(const Vector<double>& parameters) const
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Sum squared error stuff

   #ifdef __OPENNN_DEBUG__

   std::ostringstream buffer;

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      buffer << "OpenNN Exception: RocAreaError class." << std::endl
             << "double calculate_error(const Vector<double>&) const method." << std::endl
             << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ")." << std::endl;

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

   // Sum squared error stuff

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   int i = 0;

   double decision_threshold = 0.0;

   double true_positives = 0.0;
   double true_negatives = 0.0;
   double false_positives = 0.0;
   double false_negatives = 0.0;

   const size_t points_number = 101;

   Vector<double> true_positive_rate(points_number, 0.0);
   Vector<double> false_positive_rate(points_number, 0.0);

   Vector<double> target_data(training_instances_number);
   Vector<double> output_data(training_instances_number);

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

       // Input vector

       inputs = data_set_pointer->get_instance(training_index, inputs_indices);

       // Target vector

       targets = data_set_pointer->get_instance(training_index, targets_indices);

       target_data[i] = targets[0];

       // Output vector

       outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

       output_data[i] = outputs[0];
   }

   // Calculate binary classification tests

   for(size_t i = 1; i < points_number-1; i++)
   {
       decision_threshold = i*0.01;

       true_positives = 0.0;
       true_negatives = 0.0;
       false_positives = 0.0;
       false_negatives = 0.0;

       for(size_t j = 0; j < training_instances_number; j++)
       {
           if(target_data[j] == 1.0)
           {
               if(output_data[j] >= decision_threshold)
               {
                   true_positives += output_data[j];
               }
               else
               {
                   false_negatives += (1.0 - output_data[j]);
               }
           }
           else if(target_data[j] == 0.0)
           {
               if(output_data[j] >= decision_threshold)
               {
                   false_positives += output_data[j];
               }
               else
               {
                   true_negatives += (1.0 - output_data[j]);
               }
           }
           else
           {
               std::ostringstream buffer;

               buffer << "OpenNN Exception: RocAreaError class.\n"
                      << "double calculate_error(void) const method.\n"
                      << "Target is not binary.\n";

               throw std::logic_error(buffer.str());
           }
       }

       if(true_negatives+false_positives < 1.0e-12)
       {
           std::ostringstream buffer;

           buffer << "OpenNN Exception: RocAreaError class.\n"
                  << "double calculate_error(void) const method.\n"
                  << "Cannot compute false_positive_rate.\n";

           throw std::logic_error(buffer.str());
       }

       false_positive_rate[i] = false_negatives/(true_positives+false_negatives);

       if(false_positive_rate[i] < 0.0
       || false_positive_rate[i] > 1.0)
       {
           std::ostringstream buffer;

           buffer << "OpenNN Exception: RocAreaError class.\n"
                  << "double calculate_error(void) const method.\n"
                  << "false_positve_rate must be between 0 and 1.\n";

           throw std::logic_error(buffer.str());
       }

       if(true_positives+false_negatives < 1.0e-12)
       {
           std::ostringstream buffer;

           buffer << "OpenNN Exception: RocAreaError class.\n"
                  << "double calculate_error(void) const method.\n"
                  << "Cannot compute true_positive_rate.\n";

           throw std::logic_error(buffer.str());
       }

       true_positive_rate[i] = true_positives/(true_negatives+false_positives);

       if(true_positive_rate[i] < 0.0
       || true_positive_rate[i] > 1.0)
       {
           std::ostringstream buffer;

           buffer << "OpenNN Exception: RocAreaError class.\n"
                  << "double calculate_error(void) const method.\n"
                  << "true_positve_rate must be between 0 and 1.\n";

           throw std::logic_error(buffer.str());
       }

   }

   false_positive_rate[points_number-1] = 1.0;
   true_positive_rate[points_number-1] = 1.0;

   NumericalIntegration numerical_integration;

   const double roc_area = numerical_integration.calculate_trapezoid_integral(false_positive_rate, true_positive_rate);

   const double roc_area_error = (1.0-roc_area)*(1.0-roc_area);

   return(roc_area_error);
}


// Test combination
/*
double RocAreaError::calculate_loss_combination(const size_t& index, const Vector<double>& combinations) const
{
    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    const Vector<double> targets = data_set_pointer->get_instance(0, targets_indices);

    const Vector<double> activations = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(index).calculate_activations(combinations);

    return activations.calculate_sum_squared_error(targets);
}


double RocAreaError::calculate_loss_combinations(const size_t& index_1, const Vector<double>& combinations_1, const size_t& index_2, const Vector<double>& combinations_2) const
{
    std::cout << index_1 << combinations_1 << std::endl;

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const Variables& variables = data_set_pointer->get_variables();

    const size_t layers_number = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layers_number();

    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    const Vector<double> targets = data_set_pointer->get_instance(0, targets_indices);

//    const Vector<double> activations_1 = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(index_1).calculate_activations(combinations_1);
//    const Vector<double> activations_2 = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(index_2).calculate_activations(combinations_2);

    Vector<double> outputs;

    for(size_t i = index_2; i < layers_number; i++)
    {
        outputs = multilayer_perceptron_pointer->get_layer(index_2).calculate_activations(combinations_2);
    }

    return outputs.calculate_sum_squared_error(targets);
}


// double calculate_selection_loss(void) const method

/// Returns the sum squared error of the neural network measured on the selection instances of the data set.

double RocAreaError::calculate_selection_loss(void) const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();
   const size_t selection_instances_number = instances.count_selection_instances_number();

   const Vector<size_t> selection_indices = instances.arrange_selection_indices();

   size_t selection_index;

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   // Sum squared error stuff

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   double selection_loss = 0.0;

   int i = 0;

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

   return(selection_loss);
}
*/

// Vector<double> calculate_output_gradient(const Vector<double>&, const Vector<double>&) const method

Vector<double> RocAreaError::calculate_output_gradient(const Vector<double>& output, const Vector<double>& target) const
{
    const Vector<double> output_gradient = (output-target)*2.0;

    return(output_gradient);
}


// Vector<double> calculate_gradient(void) const method

/// Calculates the error term gradient by means of the back-propagation algorithm, 
/// and returns it in a single vector of size the number of neural network parameters. 

Vector<double> RocAreaError::calculate_gradient(void) const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   // Neural network stuff

   const bool has_conditions_layer = neural_network_pointer->has_conditions_layer();

   const ConditionsLayer* conditions_layer_pointer = has_conditions_layer ? neural_network_pointer->get_conditions_layer_pointer() : NULL;

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   const size_t neural_parameters_number = multilayer_perceptron_pointer->count_parameters_number();

   Vector< Vector< Vector<double> > > first_order_forward_propagation(2); 

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

   Vector<double> inputs(inputs_number);
   Vector<double> targets(outputs_number);

   // Sum squared error stuff

   Vector<double> output_gradient(outputs_number);

   Vector< Matrix<double> > layers_combination_parameters_Jacobian; 

   Vector< Vector<double> > layers_inputs(layers_number); 
   Vector< Vector<double> > layers_delta; 

   Vector<double> point_gradient(neural_parameters_number, 0.0);

   Vector<double> gradient(neural_parameters_number, 0.0);

   int i;

   #pragma omp parallel for private(i, training_index, inputs, targets, first_order_forward_propagation, layers_inputs, layers_combination_parameters_Jacobian,\
    output_gradient, layers_delta, particular_solution, homogeneous_solution, point_gradient)

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

      inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      targets = data_set_pointer->get_instance(training_index, targets_indices);

      first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

      const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
      const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

      layers_inputs = multilayer_perceptron_pointer->arrange_layers_input(inputs, layers_activation);

      layers_combination_parameters_Jacobian = multilayer_perceptron_pointer->calculate_layers_combination_parameters_Jacobian(layers_inputs);

      if(!has_conditions_layer)
      {
          output_gradient = calculate_output_gradient(layers_activation[layers_number-1], targets);

          layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);
      }
      else
      {
         particular_solution = conditions_layer_pointer->calculate_particular_solution(inputs);
         homogeneous_solution = conditions_layer_pointer->calculate_homogeneous_solution(inputs);

         output_gradient = (particular_solution+homogeneous_solution*layers_activation[layers_number-1] - targets)*2.0;

         layers_delta = calculate_layers_delta(layers_activation_derivative, homogeneous_solution, output_gradient);
      }

      point_gradient = calculate_point_gradient(layers_combination_parameters_Jacobian, layers_delta);

      #pragma omp critical
      gradient += point_gradient;
   }

   return(gradient);
}

/*
// Matrix<double> calculate_output_Hessian(const Vector<double>&, const Vector<double>&) const method

Matrix<double> RocAreaError::calculate_output_Hessian(const Vector<double>&, const Vector<double>&) const
{
    const size_t outputs_number = neural_network_pointer->get_multilayer_perceptron_pointer()->get_outputs_number();

    Matrix<double> output_Hessian(outputs_number, outputs_number);
    output_Hessian.initialize_diagonal(2.0);

    return(output_Hessian);
}


// Matrix<double> calculate_Hessian(void) const method

/// Calculates the Hessian by means of the back-propagation algorithm,
/// and returns it in a single symmetric matrix of size the number of neural network parameters. 

Matrix<double> RocAreaError::calculate_Hessian(void) const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const bool has_conditions_layer = neural_network_pointer->has_conditions_layer();

   const ConditionsLayer* conditions_layer_pointer = has_conditions_layer ? neural_network_pointer->get_conditions_layer_pointer() : NULL;

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   const size_t parameters_number = multilayer_perceptron_pointer->count_parameters_number();

   const Vector<size_t> layers_perceptrons_number = multilayer_perceptron_pointer->arrange_layers_perceptrons_numbers();

   Vector< Vector< Vector<double> > > second_order_forward_propagation(3); 

   Vector < Vector< Vector<double> > > perceptrons_combination_parameters_gradient(layers_number);
   Matrix < Matrix<double> > interlayers_combination_combination_Jacobian;

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

   Vector<double> inputs(inputs_number);
   Vector<double> targets(outputs_number);

   // Sum squared error stuff

   Vector< Vector<double> > layers_delta(layers_number);
   Matrix< Matrix<double> > interlayers_Delta(layers_number, layers_number);

   Vector<double> output_gradient(outputs_number);
   Matrix<double> output_Hessian(outputs_number, outputs_number);

   Matrix<double> Hessian(parameters_number, parameters_number, 0.0);

   for(size_t i = 0; i < training_instances_number; i++)
   {
       training_index = training_indices[i];

      inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      targets = data_set_pointer->get_instance(training_index, targets_indices);

      second_order_forward_propagation = multilayer_perceptron_pointer->calculate_second_order_forward_propagation(inputs);
	  
      const Vector< Vector<double> >& layers_activation = second_order_forward_propagation[0];
      const Vector< Vector<double> >& layers_activation_derivative = second_order_forward_propagation[1];
      const Vector< Vector<double> >& layers_activation_second_derivative = second_order_forward_propagation[2];

	  Vector< Vector<double> > layers_inputs(layers_number);

	  layers_inputs[0] = inputs;

      for(size_t j = 1; j < layers_number; j++)
	  {
	     layers_inputs[j] = layers_activation[j-1];
      }

	  perceptrons_combination_parameters_gradient = multilayer_perceptron_pointer->calculate_perceptrons_combination_parameters_gradient(layers_inputs);

      interlayers_combination_combination_Jacobian = multilayer_perceptron_pointer->calculate_interlayers_combination_combination_Jacobian(inputs);

      if(!has_conditions_layer)
      {
         output_gradient = calculate_output_gradient(layers_activation[layers_number-1], targets);

         output_Hessian = calculate_output_Hessian(layers_activation[layers_number-1], targets);

         layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

         interlayers_Delta = calculate_interlayers_Delta(layers_activation_derivative,
                                                         layers_activation_second_derivative,
                                                         interlayers_combination_combination_Jacobian,
                                                         output_gradient,
                                                         output_Hessian,
                                                         layers_delta);
      }
      else
      {
         particular_solution = conditions_layer_pointer->calculate_particular_solution(inputs);
         homogeneous_solution = conditions_layer_pointer->calculate_homogeneous_solution(inputs);

         output_gradient = (particular_solution+homogeneous_solution*layers_activation[layers_number-1] - targets)*2.0;

         layers_delta = calculate_layers_delta(layers_activation_derivative, homogeneous_solution, output_gradient);
      }

      Hessian += calculate_point_Hessian(layers_activation_derivative, perceptrons_combination_parameters_gradient, interlayers_combination_combination_Jacobian, layers_delta, interlayers_Delta);
   }

   return(Hessian);
}

// Matrix<double> calculate_single_hidden_layer_Hessian(void) const

/// Calculates the Hessian matrix for a neural network with one hidden layer and an arbitrary number of
/// inputs, perceptrons in the hidden layer and outputs.

Matrix<double> RocAreaError::calculate_single_hidden_layer_Hessian(void) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();
    const size_t layers_number = 2;

    const size_t parameters_number = multilayer_perceptron_pointer->count_parameters_number();

    Vector< Vector< Vector<double> > > second_order_forward_propagation(3);

    Vector < Vector< Vector<double> > > perceptrons_combination_parameters_gradient(layers_number);
    Matrix < Matrix<double> > interlayers_combination_combination_Jacobian;

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    //const size_t training_instances_number = instances.count_training_instances_number();

    const Vector<size_t> training_indices = instances.arrange_training_indices();

    size_t training_index;

    //const MissingValues& missing_values = data_set_pointer->get_missing_values();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    Vector<double> inputs(inputs_number);
    Vector<double> targets(outputs_number);

    // Sum squared error stuff

    Vector< Vector<double> > layers_delta(layers_number);

    Vector<double> output_gradient(outputs_number);
    Matrix<double> output_Hessian(outputs_number, outputs_number);

    Matrix<double> single_hidden_layer_Hessian(parameters_number, parameters_number, 0.0);

    const size_t i = 0;

    training_index = training_indices[i];

    inputs = data_set_pointer->get_instance(training_index, inputs_indices);

    targets = data_set_pointer->get_instance(training_index, targets_indices);

    second_order_forward_propagation = multilayer_perceptron_pointer->calculate_second_order_forward_propagation(inputs);

    const Vector< Vector<double> >& layers_activation = second_order_forward_propagation[0];
    const Vector< Vector<double> >& layers_activation_derivative = second_order_forward_propagation[1];
    const Vector< Vector<double> >& layers_activation_second_derivative = second_order_forward_propagation[2];

    Vector< Vector<double> > layers_inputs(layers_number);

    layers_inputs[0] = inputs;

    for(size_t j = 1; j < layers_number; j++)
    {
        layers_inputs[j] = layers_activation[j-1];
    }

    perceptrons_combination_parameters_gradient = multilayer_perceptron_pointer->calculate_perceptrons_combination_parameters_gradient(layers_inputs);

    interlayers_combination_combination_Jacobian = multilayer_perceptron_pointer->calculate_interlayers_combination_combination_Jacobian(inputs);

    output_gradient = calculate_output_gradient(layers_activation[layers_number-1], targets);

    output_Hessian = calculate_output_Hessian(layers_activation[layers_number-1], targets);

    layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

    const size_t first_layer_parameters_number = multilayer_perceptron_pointer->get_layer(0).arrange_parameters().size();
    const size_t second_layer_parameters_number = multilayer_perceptron_pointer->get_layer(1).arrange_parameters().size();

    Vector<size_t> parameter_indices(3);

    size_t layer_index_i;
    size_t neuron_index_i;
    size_t parameter_index_i;

    size_t layer_index_j;
    size_t neuron_index_j;
    size_t parameter_index_j;

    const Matrix<double> output_interlayers_Delta =
    (output_Hessian
     * layers_activation_derivative[layers_number-1]
     * layers_activation_derivative[layers_number-1]
     + output_gradient
     * layers_activation_second_derivative[layers_number-1]);

    // Both weights in the second layer

    for(size_t i = first_layer_parameters_number; i < second_layer_parameters_number + first_layer_parameters_number; i++)
    {
        parameter_indices = multilayer_perceptron_pointer->arrange_parameter_indices(i);
        layer_index_i = parameter_indices[0];
        neuron_index_i = parameter_indices[1];
        parameter_index_i = parameter_indices[2];

        for(size_t j = first_layer_parameters_number; j < second_layer_parameters_number + first_layer_parameters_number; j++)
        {
            parameter_indices = multilayer_perceptron_pointer->arrange_parameter_indices(j);
            layer_index_j = parameter_indices[0];
            neuron_index_j = parameter_indices[1];
            parameter_index_j = parameter_indices[2];           

            single_hidden_layer_Hessian(i,j) =
            perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
            *perceptrons_combination_parameters_gradient[layer_index_j][neuron_index_j][parameter_index_j]
            *calculate_Kronecker_delta(neuron_index_i,neuron_index_j)
            *output_interlayers_Delta(neuron_index_j,neuron_index_i);
        }
    }

    // One weight in each layer

    Matrix<double> second_layer_weights = multilayer_perceptron_pointer->get_layer(1).arrange_synaptic_weights();

    for(size_t i = 0; i < first_layer_parameters_number; i++)
    {
        parameter_indices = multilayer_perceptron_pointer->arrange_parameter_indices(i);
        layer_index_i = parameter_indices[0];
        neuron_index_i = parameter_indices[1];
        parameter_index_i = parameter_indices[2];

        for(size_t j = first_layer_parameters_number; j < first_layer_parameters_number + second_layer_parameters_number ; j++)
        {
            parameter_indices = multilayer_perceptron_pointer->arrange_parameter_indices(j);
            layer_index_j = parameter_indices[0];
            neuron_index_j = parameter_indices[1];
            parameter_index_j = parameter_indices[2];

            single_hidden_layer_Hessian(i,j) =
             (perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
             *perceptrons_combination_parameters_gradient[layer_index_j][neuron_index_j][parameter_index_j]
             *layers_activation_derivative[layer_index_i][neuron_index_i]
             *second_layer_weights(neuron_index_j, neuron_index_i)
             *output_interlayers_Delta(neuron_index_j, neuron_index_j)
             +perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
             *layers_activation_derivative[layer_index_i][neuron_index_i]
             *layers_delta[layer_index_j][neuron_index_j]
             *calculate_Kronecker_delta(parameter_index_j,neuron_index_i+1));
        }
    }

    // Both weights in the first layer

    for(size_t i = 0; i < first_layer_parameters_number; i++)
    {
        parameter_indices = multilayer_perceptron_pointer->arrange_parameter_indices(i);
        layer_index_i = parameter_indices[0];
        neuron_index_i = parameter_indices[1];
        parameter_index_i = parameter_indices[2];

        for(size_t j = 0; j < first_layer_parameters_number; j++)
        {
            parameter_indices = multilayer_perceptron_pointer->arrange_parameter_indices(j);
            layer_index_j = parameter_indices[0];
            neuron_index_j = parameter_indices[1];
            parameter_index_j = parameter_indices[2];

            double sum = 0.0;

            for(size_t k = 0; k < outputs_number; k++)
            {
                sum += second_layer_weights(k, neuron_index_i)
                       *second_layer_weights(k, neuron_index_j)
                       *output_interlayers_Delta(k,k);
            }

            single_hidden_layer_Hessian(i, j) =
                    perceptrons_combination_parameters_gradient[layer_index_i][neuron_index_i][parameter_index_i]
                    *perceptrons_combination_parameters_gradient[layer_index_j][neuron_index_j][parameter_index_j]
                    *(layers_activation_derivative[layer_index_i][neuron_index_i]
                    *layers_activation_derivative[layer_index_j][neuron_index_j]
                    *sum
                    +layers_activation_second_derivative[layer_index_j][neuron_index_j]
                    *calculate_Kronecker_delta(neuron_index_j,neuron_index_i)
                    *second_layer_weights.arrange_column(neuron_index_j).dot(layers_delta[1]));
        }
    }

    // Hessian

    for(size_t i = 0; i < parameters_number; i++)
    {
        for(size_t j = 0; j < parameters_number; j++)
        {
            single_hidden_layer_Hessian(j,i) = single_hidden_layer_Hessian(i,j);
        }
    }

    return single_hidden_layer_Hessian;
}


// Vector<double> calculate_terms(void) const method

/// Calculates the squared error terms for each instance, and returns it in a vector of size the number training instances. 

Vector<double> RocAreaError::calculate_terms(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   check();

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

   // Loss index stuff

   Vector<double> error_terms(training_instances_number);

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   int i = 0;

   #pragma omp parallel for private(i, training_index, inputs, outputs, targets)

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

      // Input vector

      inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Error

      error_terms[i] = outputs.calculate_distance(targets);
   }

   return(error_terms);
}


// Vector<double> calculate_terms(const Vector<double>&) const method

/// Returns the error terms vector for a hypotetical vector of parameters. 
/// @param parameters Neural network parameters for which the error terms vector is to be computed. 

Vector<double> RocAreaError::calculate_terms(const Vector<double>& parameters) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__
   
   check();

   #endif


   #ifdef __OPENNN_DEBUG__

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: RocAreaError class." << std::endl
             << "double calculate_terms(const Vector<double>&) const method." << std::endl
             << "Size (" << size << ") must be equal to number of neural network parameters (" << parameters_number << ")." << std::endl;

      throw std::logic_error(buffer.str());	  
   }

   #endif

   NeuralNetwork neural_network_copy(*neural_network_pointer);

   neural_network_copy.set_parameters(parameters);

   RocAreaError sum_squared_error_copy(*this);

   sum_squared_error_copy.set_neural_network_pointer(&neural_network_copy);

   return(sum_squared_error_copy.calculate_terms());
}


// Matrix<double> calculate_terms_Jacobian(void) const method

/// Returns the terms_Jacobian matrix of the sum squared error function, whose elements are given by the 
/// derivatives of the squared errors data set with respect to the multilayer perceptron parameters.
/// The terms_Jacobian matrix here is computed using a back-propagation algorithm.

Matrix<double> RocAreaError::calculate_terms_Jacobian(void) const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif 

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
   const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();
   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   const size_t neural_parameters_number = multilayer_perceptron_pointer->count_parameters_number();

   Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

   Vector< Vector<double> > layers_inputs(layers_number);
   Vector< Matrix<double> > layers_combination_parameters_Jacobian(layers_number);

   Vector<double> particular_solution;
   Vector<double> homogeneous_solution;

   const bool has_conditions_layer = neural_network_pointer->has_conditions_layer();

   const ConditionsLayer* conditions_layer_pointer = has_conditions_layer ? neural_network_pointer->get_conditions_layer_pointer() : NULL;

   // Data set

   const Instances& instances = data_set_pointer->get_instances();

   const size_t training_instances_number = instances.count_training_instances_number();

   const Vector<size_t> training_indices = instances.arrange_training_indices();

   size_t training_index;

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
   const Vector<size_t> targets_indices = variables.arrange_targets_indices();

   const MissingValues missing_values = data_set_pointer->get_missing_values();

   Vector<double> inputs(inputs_number);
   Vector<double> targets(outputs_number);

   // Loss index

   Vector<double> term(outputs_number);
   double term_norm;

   Vector<double> output_gradient(outputs_number);

   Vector< Vector<double> > layers_delta(layers_number);
   Vector<double> point_gradient(neural_parameters_number);

   Matrix<double> terms_Jacobian(training_instances_number, neural_parameters_number);

   // Main loop

   int i = 0;

   #pragma omp parallel for private(i, training_index, inputs, targets, first_order_forward_propagation, layers_inputs, \
    layers_combination_parameters_Jacobian, term, term_norm, output_gradient, layers_delta, particular_solution, homogeneous_solution, point_gradient)

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

      inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      targets = data_set_pointer->get_instance(training_index, targets_indices);

      first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

      layers_inputs = multilayer_perceptron_pointer->arrange_layers_input(inputs, first_order_forward_propagation);

      //      const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

      layers_combination_parameters_Jacobian = multilayer_perceptron_pointer->calculate_layers_combination_parameters_Jacobian(layers_inputs);

      if(!has_conditions_layer)
      {
         //const Vector<double>& outputs = first_order_forward_propagation[0][layers_number-1];

         term = first_order_forward_propagation[0][layers_number-1] - targets;
         term_norm = term.calculate_norm();

         if(term_norm == 0.0)
   	     {
             output_gradient.set(outputs_number, 0.0);
	     }
         else
	     {
            output_gradient = term/term_norm;
	     }

         layers_delta = calculate_layers_delta(first_order_forward_propagation[1], output_gradient);
      }
      else
      {

         particular_solution = conditions_layer_pointer->calculate_particular_solution(inputs);
         homogeneous_solution = conditions_layer_pointer->calculate_homogeneous_solution(inputs);

         //const Vector<double>& output_layer_activation = first_order_forward_propagation[0][layers_number-1];

         term = (particular_solution+homogeneous_solution*first_order_forward_propagation[0][layers_number-1] - targets);
         term_norm = term.calculate_norm();

         if(term_norm == 0.0)
   	     {
             output_gradient.set(outputs_number, 0.0);
         }
	     else
	     {
            output_gradient = term/term_norm;
	     }

         layers_delta = calculate_layers_delta(first_order_forward_propagation[1], homogeneous_solution, output_gradient);
      }

      point_gradient = calculate_point_gradient(layers_combination_parameters_Jacobian, layers_delta);

      terms_Jacobian.set_row(i, point_gradient);
  }

   return(terms_Jacobian);
}


// FirstOrderTerms calculate_first_order_terms(void) const method

/// Returns the first order loss of the terms loss function.
/// This is a structure containing the error terms vector and the error terms Jacobian.

ErrorTerm::FirstOrderTerms RocAreaError::calculate_first_order_terms(void) const
{
   FirstOrderTerms first_order_terms;

   first_order_terms.terms = calculate_terms();
   first_order_terms.Jacobian = calculate_terms_Jacobian();

   return(first_order_terms);
}


// Vector<double> calculate_squared_errors(void) const method

/// Returns the squared errors of the training instances. 

Vector<double> RocAreaError::calculate_squared_errors(void) const
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__

   check();

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

   const MissingValues missing_values = data_set_pointer->get_missing_values();

   // Loss index

   Vector<double> squared_errors(training_instances_number);

   Vector<double> inputs(inputs_number);
   Vector<double> outputs(outputs_number);
   Vector<double> targets(outputs_number);

   int i = 0;

   #pragma omp parallel for private(i, training_index, inputs, outputs, targets)

   for(i = 0; i < (int)training_instances_number; i++)
   {
       training_index = training_indices[i];

      // Input vector

      inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Error

      squared_errors[i] = outputs.calculate_sum_squared_error(targets);
   }

   return(squared_errors);
}


// Vector<double> calculate_gradient(const Vector<double>&) const method

/// @todo

Vector<double> RocAreaError::calculate_gradient(const Vector<double>&) const
{
    Vector<double> gradient;

    return(gradient);
}


// Matrix<double> calculate_Hessian(const Vector<double>&) const metod

/// @todo

Matrix<double> RocAreaError::calculate_Hessian(const Vector<double>&) const
{
    Matrix<double> Hessian;


    return(Hessian);
}


// std::string write_error_term_type(void) const method

/// Returns a string with the name of the sum squared error loss type, "SUM_SQUARED_ERROR".

std::string RocAreaError::write_error_term_type(void) const
{
   return("SUM_SQUARED_ERROR");
}


// tinyxml2::XMLDocument* to_XML(void) method method 

/// Returns a representation of the sum squared error object, in XML format. 

tinyxml2::XMLDocument* RocAreaError::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Sum squared error

   tinyxml2::XMLElement* root_element = document->NewElement("RocAreaError");

   document->InsertFirstChild(root_element);

   // Display

//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      root_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

   return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

void RocAreaError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("RocAreaError");

    file_stream.CloseElement();
}


// void load(const tinyxml2::XMLDocument&) method

/// Loads a sum squared error object from a XML document.
/// @param document TinyXML document containing the members of the object.

void RocAreaError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("RocAreaError");

    if(!root_element)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: RocAreaError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Sum squared error element is NULL.\n";

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
*/


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
