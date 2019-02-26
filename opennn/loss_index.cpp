/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   L O S S   I N D E X   C L A S S                                                                            */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "loss_index.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a default error term object, with all pointers initialized to nullptr.
/// It also initializes all the rest of class members to their default values.

LossIndex::LossIndex()
 : neural_network_pointer(nullptr), 
   data_set_pointer(nullptr)
{
   set_default();
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a error term object associated to a neural network object.
/// The rest of pointers are initialized to nullptr.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

LossIndex::LossIndex(NeuralNetwork* new_neural_network_pointer)
 : neural_network_pointer(new_neural_network_pointer), 
   data_set_pointer(nullptr)
{
   set_default();
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a error term object associated to a given data set object.
/// The rest of pointers are initialized to nullptr.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

LossIndex::LossIndex(DataSet* new_data_set_pointer)
 : neural_network_pointer(nullptr), 
   data_set_pointer(new_data_set_pointer)
{
   set_default();
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a error term object associated to a neural network and to be measured on a data set.
/// The rest of pointers are initialized to nullptr.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

LossIndex::LossIndex(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
 : neural_network_pointer(new_neural_network_pointer), 
   data_set_pointer(new_data_set_pointer)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a default error term object, with all pointers initialized to nullptr.
/// It also loads all the rest of class members from a XML document.
/// @param error_term_document Pointer to a TinyXML document with the object data.

LossIndex::LossIndex(const tinyxml2::XMLDocument& error_term_document)
 : neural_network_pointer(nullptr), 
   data_set_pointer(nullptr)
{
   set_default();

   from_XML(error_term_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing error term object.
/// @param other_error_term Error term object to be copied.

LossIndex::LossIndex(const LossIndex& other_error_term)
 : neural_network_pointer(nullptr), 
   data_set_pointer(nullptr)
{
   neural_network_pointer = other_error_term.neural_network_pointer;

   data_set_pointer = other_error_term.data_set_pointer;

   display = other_error_term.display;
}


// DESTRUCTOR

/// Destructor.

LossIndex::~LossIndex()
{
}


// ASSIGNMENT OPERATOR


/// Assignment operator. 
/// It assigns to this error term object the members from another error term object.
/// @param other_error_term Error term object to be copied.

LossIndex& LossIndex::operator = (const LossIndex& other_error_term)
{
   if(this != &other_error_term)
   {
      neural_network_pointer = other_error_term.neural_network_pointer;

      data_set_pointer = other_error_term.data_set_pointer;

      display = other_error_term.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR


/// Equal to operator. 
/// It compares this object to another object. 
/// The return is true if both objects have the same member data, and false otherwise. 

bool LossIndex::operator == (const LossIndex& other_error_term) const
{
   if(neural_network_pointer != other_error_term.neural_network_pointer
   || data_set_pointer != other_error_term.data_set_pointer)
   {
       return(false);
   }

   else if(display != other_error_term.display)
   {
      return(false);
   }

   return(true);

}


// METHODS

const double& LossIndex::get_regularization_weight() const
{
   return(regularization_weight);
}


/// Returns true if messages from this class can be displayed on the screen, or false if messages
/// from this class can't be displayed on the screen.

const bool& LossIndex::get_display() const
{
   return(display);
}


/// Returns true if this error term has a neural network associated,
/// and false otherwise.

bool LossIndex::has_neural_network() const
{
    if(neural_network_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns true if this error term has a data set associated,
/// and false otherwise.

bool LossIndex::has_data_set() const
{
    if(data_set_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Sets all the member pointers to nullptr(neural network, data set, mathematical model).
/// It also initializes all the rest of class members to their default values.

void LossIndex::set()
{
   neural_network_pointer = nullptr;
   data_set_pointer = nullptr;

   set_default();
}


/// Sets all the member pointers to nullptr, but the neural network, which set to a given pointer.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

void LossIndex::set(NeuralNetwork* new_neural_network_pointer)
{
   neural_network_pointer = new_neural_network_pointer;
   data_set_pointer = nullptr;

   set_default();
}


/// Sets all the member pointers to nullptr, but the data set, which set to a given pointer.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

void LossIndex::set(DataSet* new_data_set_pointer)
{
   neural_network_pointer = nullptr;
   data_set_pointer = new_data_set_pointer;

   set_default();
}


/// Sets new neural network and data set pointers.
/// Finally, it initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

void LossIndex::set(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
{
   neural_network_pointer = new_neural_network_pointer;

   data_set_pointer = new_data_set_pointer;

   set_default();
}


/// Sets to this error term object the members of another error term object.
/// @param other_error_term Error term to be copied.

void LossIndex::set(const LossIndex& other_error_term)
{
   neural_network_pointer = other_error_term.neural_network_pointer;

   data_set_pointer = other_error_term.data_set_pointer;

   regularization_method = other_error_term.regularization_method;

   display = other_error_term.display;
}


/// Sets a pointer to a neural network object which is to be associated to the error term.
/// @param new_neural_network_pointer Pointer to a neural network object to be associated to the error term.

void LossIndex::set_neural_network_pointer(NeuralNetwork* new_neural_network_pointer)
{
   neural_network_pointer = new_neural_network_pointer;
}


/// Sets a new data set on which the error term is to be measured.

void LossIndex::set_data_set_pointer(DataSet* new_data_set_pointer)
{
   data_set_pointer = new_data_set_pointer;
}


/// Sets the members of the error term to their default values:
/// <ul>
/// <li> Display: true.
/// </ul>

void LossIndex::set_default()
{
   regularization_method = L2;
   display = true;
}


void LossIndex::set_regularization_method(const string& new_regularization_method)
{
    if(new_regularization_method == "L1_NORM")
    {
        set_regularization_method(L1);
    }
    else if(new_regularization_method == "L2_NORM")
    {
        set_regularization_method(L2);
    }
    else if(new_regularization_method == "NO_REGULARIZATION")
    {
        set_regularization_method(None);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LossIndex class.\n"
               << "void set_regularization_method(const string&) const method.\n"
               << "Unknown regularization method: " << new_regularization_method << ".";

        throw logic_error(buffer.str());
    }
}


void LossIndex::set_regularization_method(const LossIndex::RegularizationMethod& new_regularization_method)
{
    regularization_method = new_regularization_method;
}

void LossIndex::set_regularization_weight(const double& new_regularization_weight)
{
    regularization_weight = new_regularization_weight;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void LossIndex::set_display(const bool& new_display)
{
   display = new_display;
}


/// Checks that there is a neural network associated to the error term.
/// If some of the above conditions is not hold, the method throws an exception. 

void LossIndex::check() const
{
   ostringstream buffer;

   if(!neural_network_pointer)
   {
      buffer << "OpenNN Exception: LossIndex class.\n"
             << "void check() const.\n"
             << "Pointer to neural network is nullptr.\n";

      throw logic_error(buffer.str());	  
   }

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

  if(!multilayer_perceptron_pointer)
  {
        buffer << "OpenNN Exception: LossIndex class.\n"
            << "Vector< Vector<double> > calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&) const method.\n"
            << "Pointer to multilayer perceptron in neural network is nullptr.\n";

        throw logic_error(buffer.str());
  }

  const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
  const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

  // Data set stuff

  if(!data_set_pointer)
  {
     buffer << "OpenNN Exception: LossIndex class.\n"
            << "void check() const method.\n"
            << "Pointer to data set is nullptr.\n";

     throw logic_error(buffer.str());
  }

  const Variables& variables = data_set_pointer->get_variables();

  const size_t data_set_inputs_number = variables.get_inputs_number();
  const size_t targets_number = variables.get_targets_number();

  if(data_set_inputs_number != inputs_number)
  {
     buffer << "OpenNN Exception: LossIndex class.\n"
            << "void check() const method.\n"
            << "Number of inputs in neural network (" << inputs_number << ") must be equal to number of inputs in data set (" << data_set_inputs_number << ").\n";

     throw logic_error(buffer.str());
  }

  if(outputs_number != targets_number)
  {
     buffer << "OpenNN Exception: LossIndex class.\n"
            << "void check() const method.\n"
            << "Number of outputs in neural network must be equal to number of targets in data set.\n";

     throw logic_error(buffer.str());
  }
}


/// Returns the delta vector for all the layers in the multilayer perceptron.
/// The format of this quantity is a vector of vectors. 
/// @param layers_activation_derivative Forward propagation activation derivative. 
/// @param output_gradient Gradient of the outputs error function.
/*
Vector< Vector<double> > LossIndex::calculate_layers_delta
(const Vector< Vector<double> >& layers_activation_derivative, 
 const Vector<double>& output_gradient) const
{
   // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   #ifdef __OPENNN_DEBUG__
   
   check();

   #endif

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   const Vector<size_t> layers_perceptrons_number = multilayer_perceptron_pointer->get_layers_perceptrons_numbers();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   // Forward propagation activation derivative size

   const size_t layers_activation_derivative_size = layers_activation_derivative.size();

   if(layers_activation_derivative_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Vector<double> > calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
             << "Size of forward propagation activation derivative vector must be equal to number of layers.\n";

      throw logic_error(buffer.str());	  
   }

   if(layers_number > 0)
   {
      const size_t output_gradient_size = output_gradient.size();

      if(output_gradient_size != layers_perceptrons_number[layers_number-1])
      {
          ostringstream buffer;

         buffer << "OpenNN Exception: LossIndex class.\n"
                << "Vector<double> calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
                << "Size of outputs error gradient (" << output_gradient_size << ") must be equal to "
                << "number of outputs (" << layers_perceptrons_number[layers_number-1] << ").\n";

         throw logic_error(buffer.str());	     
      }
   }

   #endif

   // Neural network stuff

   Matrix<double> layer_synaptic_weights;

   // Loss index stuff

   Vector< Vector<double> > layers_delta(layers_number);

   // Output layer

   if(layers_number > 0)
   {
      layers_delta[layers_number-1] = layers_activation_derivative[layers_number-1]*output_gradient;

      // Rest of hidden layers

      for(int i = static_cast<int>(layers_number)-2; i >= 0; i--)
      {
         layer_synaptic_weights = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(static_cast<size_t>(i+1)).get_synaptic_weights();

         layers_delta[static_cast<size_t>(i)] = layers_activation_derivative[static_cast<size_t>(i)]*(layers_delta[static_cast<size_t>(i+1)].dot(layer_synaptic_weights));
      }
   }

   return(layers_delta);
}
*/


double LossIndex::calculate_columns_product_sum(const Matrix<double>& A, const size_t& column_a, const Matrix<double>& B, const size_t& column_b) const
{
    double sum = 0.0;

    for(size_t i = 0; i < A.get_rows_number(); i++)
    {
        sum += A(i, column_a)*B(i, column_b);
    }

    return sum;
}


Vector< Matrix<double> > LossIndex::calculate_layers_delta
(const Vector< Matrix<double> >& layers_activation_derivative,
 const Matrix<double>& output_gradient) const
{
    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const Vector<size_t> layers_perceptrons_number = multilayer_perceptron_pointer->get_layers_perceptrons_numbers();

   // Forward propagation activation derivative size

   const size_t layers_activation_derivative_size = layers_activation_derivative.size();

   if(layers_activation_derivative_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Matrix<double> > calculate_layers_delta(const Vector< Matrix<double> >&, const Matrix<double>&) method.\n"
             << "Size of forward propagation activation derivative vector must be equal to number of layers.\n";

      throw logic_error(buffer.str());
   }

   if(layers_number > 0)
   {
      const size_t output_gradient_columns_number = output_gradient.get_columns_number();

      if(output_gradient_columns_number != layers_perceptrons_number[layers_number-1])
      {
          ostringstream buffer;

         buffer << "OpenNN Exception: LossIndex class.\n"
                << "Vector<double> calculate_layers_delta(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
                << "Size of outputs error gradient (" << output_gradient_columns_number << ") must be equal to "
                << "number of outputs (" << layers_perceptrons_number[layers_number-1] << ").\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   // Neural network stuff

   Matrix<double> layer_synaptic_weights_transpose;

   // Loss index stuff

   Vector< Matrix<double> > layers_delta(layers_number);

   // Output layer

   if(layers_number > 0)
   {
      layers_delta[layers_number-1] = layers_activation_derivative[layers_number-1]*output_gradient;

      // Rest of hidden layers

      for(int i = static_cast<int>(layers_number)-2; i >= 0; i--)
      {
         layer_synaptic_weights_transpose = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(static_cast<size_t>(i+1)).get_synaptic_weights().calculate_transpose();

         layers_delta[static_cast<size_t>(i)] = layers_activation_derivative[static_cast<size_t>(i)]*(layers_delta[static_cast<size_t>(i+1)].dot(layer_synaptic_weights_transpose));
      }
   }

   return layers_delta;
}


/// Returns the gradient of the error term function at some input point.
/// @param layers_combination_parameters_Jacobian
/// @param layers_delta
/// @todo
/*
Vector<double> LossIndex::calculate_point_gradient
(const Vector< Matrix<double> >& layers_combination_parameters_Jacobian, 
 const Vector< Vector<double> >& layers_delta) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__
 
   // Input size

   const size_t layers_combination_parameters_Jacobian_size = layers_combination_parameters_Jacobian.size();

   if(layers_combination_parameters_Jacobian_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Vector<double> > calculate_layers_error_gradient(const Vector< Vector<double> >&, const Vector<double>&, const Vector<double>&) method.\n"
             << "Size of forward propagation activation(" << layers_combination_parameters_Jacobian_size << ") must be equal to number of layers(" << layers_number << ").\n";

      throw logic_error(buffer.str());	  
   }

   // Hidden errors size

   const size_t layers_delta_size = layers_delta.size();
      
   if(layers_delta_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Vector<double> > calculate_layers_error_gradient(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
             << "Size of layers delta("<< layers_delta_size << ") must be equal to number of layers(" << layers_number << ").\n";

      throw logic_error(buffer.str());	  
   }

   #endif

   const Vector<size_t> layers_parameters_number = multilayer_perceptron_pointer->get_layers_parameters_number();

   const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

   Vector<double> point_gradient(parameters_number);

   size_t index = 0;

   for(size_t i = 0; i < layers_number; i++)
   {
      const Vector<double> layer_point_gradient = layers_delta[i].dot(layers_combination_parameters_Jacobian[i]);

      point_gradient.tuck_in(index, layer_point_gradient);

      index += layers_parameters_number[i];
   }

   return(point_gradient);
}
*/


Vector<double> LossIndex::calculate_error_gradient
(const Matrix<double>& inputs,
 const Vector< Matrix<double> >& layers_activations,
 const Vector< Matrix<double> >& layers_delta) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   // Hidden errors size

   const size_t layers_delta_size = layers_delta.size();

   if(layers_delta_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Vector< Vector<double> > calculate_layers_error_gradient(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
             << "Size of layers delta("<< layers_delta_size << ") must be equal to number of layers(" << layers_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

   const Vector<size_t> layers_parameters_number = multilayer_perceptron_pointer->get_layers_parameters_number();

   Vector<double> error_gradient(parameters_number);

   size_t index = 0;

   error_gradient.tuck_in(index, calculate_layer_error_gradient(layers_delta[0], inputs));

   index += layers_parameters_number[0];

   for(size_t i = 1; i < layers_number; i++)
   {
      error_gradient.tuck_in(index, calculate_layer_error_gradient(layers_delta[i], layers_activations[i-1]));

      index += layers_parameters_number[i];
   }

   return error_gradient;
}


Matrix<double> LossIndex::calculate_error_terms_Jacobian
(const Matrix<double>& inputs,
 const Vector< Matrix<double> >& layers_activations,
 const Vector< Matrix<double> >& layers_delta) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   // Hidden errors size

   const size_t layers_delta_size = layers_delta.size();

   if(layers_delta_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Matrix<double> calculate_layers_error_Jacobian(const Vector< Vector<double> >&, const Vector<double>&) method.\n"
             << "Size of layers delta("<< layers_delta_size << ") must be equal to number of layers(" << layers_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();
   const size_t instances_number = inputs.get_rows_number();

   const Vector<size_t> layers_parameters_number = multilayer_perceptron_pointer->get_layers_parameters_number();

   Matrix<double> error_Jacobian(instances_number, parameters_number);

   size_t index = 0;

   error_Jacobian.tuck_in(0, index, calculate_layer_error_terms_Jacobian(layers_delta[0], inputs));

   index += layers_parameters_number[0];

   for(size_t i = 1; i < layers_number; i++)
   {
      error_Jacobian.tuck_in(0, index, calculate_layer_error_terms_Jacobian(layers_delta[i], layers_activations[i-1]));

      index += layers_parameters_number[i];
   }

   return error_Jacobian;
}


Vector<double> LossIndex::calculate_layer_error_gradient(const Matrix<double>& layer_deltas, const Matrix<double>& layer_inputs) const
{
    const size_t inputs_number = layer_inputs.get_columns_number();
    const size_t perceptrons_number = layer_deltas.get_columns_number();

    const size_t synaptic_weights_number = perceptrons_number*inputs_number;

    Vector<double> layer_error_gradient(perceptrons_number*(1+inputs_number), 0.0);

    // Synaptic weights

    layer_error_gradient.tuck_in(0, layer_inputs.calculate_transpose().dot(layer_deltas).to_vector());

    // Biases

    layer_error_gradient.tuck_in(synaptic_weights_number, layer_deltas.calculate_columns_sum());
//    for(size_t perceptron = 0; perceptron < perceptrons_number; perceptron++)
//    {
//        layer_error_gradient[synaptic_weights_number+perceptron] = layer_deltas.calculate_column_sum(perceptron);
//    }

    return layer_error_gradient;

//    size_t parameter = 0;

//    for(size_t perceptron = 0; perceptron < perceptrons_number; perceptron++)
//    {
//        for(size_t input = 0; input < inputs_number; input++)
//        {
//            layer_error_gradient[parameter] = calculate_columns_product_sum(layer_deltas, perceptron, layer_inputs, input);

//            parameter++;
//        }
//     }

//    const size_t instances_number = layer_inputs.get_rows_number();

//    Matrix<double> layer_error_Jacobian(instances_number, perceptrons_number*(1+inputs_number), 0.0);

//    for(size_t instance = 0; instance < instances_number; instance++)
//    {
//        parameter = 0;

//        for(size_t perceptron = 0; perceptron < perceptrons_number; perceptron++)
//        {
//            const double layer_delta = layer_deltas(instance, perceptron);

//            for(size_t input = 0; input < inputs_number; input++)
//            {
//                layer_error_gradient[parameter] += layer_delta*layer_inputs(instance, input);

//                parameter++;
//            }

//            layer_error_gradient[synaptic_weights_number+perceptron] += layer_delta;
//         }
//    }

//    return layer_error_gradient;
}


Matrix<double> LossIndex::calculate_layer_error_terms_Jacobian(const Matrix<double>& layer_deltas, const Matrix<double>& layer_inputs) const
{
    const size_t instances_number = layer_inputs.get_rows_number();
    const size_t inputs_number = layer_inputs.get_columns_number();
    const size_t perceptrons_number = layer_deltas.get_columns_number();

    const size_t synaptic_weights_number = perceptrons_number*inputs_number;

    Matrix<double> layer_error_Jacobian(instances_number, perceptrons_number*(1+inputs_number), 0.0);

    size_t parameter;

    for(size_t instance = 0; instance < instances_number; instance++)
    {
        parameter = 0;

        for(size_t perceptron = 0; perceptron < perceptrons_number; perceptron++)
        {
            const double layer_delta = layer_deltas(instance, perceptron);

            for(size_t input = 0; input < inputs_number; input++)
            {
                layer_error_Jacobian(instance, parameter) = layer_delta*layer_inputs(instance, input);

                parameter++;
            }

            layer_error_Jacobian(instance, synaptic_weights_number+perceptron) = layer_delta;
         }
    }

    return layer_error_Jacobian;
}



double LossIndex::calculate_training_loss() const
{
    if(regularization_method == None)
    {
        return calculate_training_error();
    }
    else
    {
        return calculate_training_error() + regularization_weight*calculate_regularization();
    }
}


double LossIndex::calculate_training_loss(const Vector<double>& parameters) const
{
    if(regularization_method == None)
    {
        return calculate_training_error(parameters);
    }
    else
    {
        return calculate_training_error(parameters) + regularization_weight*calculate_regularization(parameters);
    }
}


double LossIndex::calculate_training_loss(const Vector<double>& direction, const double& rate) const
{    
    const Vector<double> parameters = neural_network_pointer->get_parameters();

    return calculate_training_loss(parameters + direction*rate);
}


Vector<double> LossIndex::calculate_training_loss_gradient() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    if(regularization_method == None)
    {
        return calculate_training_error_gradient();
    }
    else
    {
        return calculate_training_error_gradient() + calculate_regularization_gradient()*regularization_weight;
    }
}

/// Returns a string with the default type of error term, "USER_PERFORMANCE_TERM".

string LossIndex::write_error_term_type() const
{
   return("USER_ERROR_TERM");
}


/// Returns a string with the default information of the error term.
/// It will be used by the training strategy to monitor the training process. 
/// By default this information is empty. 

string LossIndex::write_information() const
{
   return string();
}


/// Returns a string with teh regularization method.

string LossIndex::write_regularization_method() const
{
    switch(regularization_method)
    {
       case L1:
       {
            return "L1_NORM";
       }
       case L2:
       {
            return "L2_NORM";
       }
       case None:
       {
            return "NO_REGULARIZATION";
       }
    }

    return "NO_REGULARIZATION";
}


double LossIndex::calculate_regularization() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    switch(regularization_method)
    {
       case L1:
       {
            return neural_network_pointer->get_parameters().calculate_L1_norm();;
       }
       case L2:
       {
            return neural_network_pointer->get_parameters().calculate_L2_norm();
       }
       case None:
       {
            return 0.0;
       }
    }

    return 0.0;
}


double LossIndex::calculate_regularization(const Vector<double>& parameters) const
{
    switch(regularization_method)
    {
       case L1:
       {
            return parameters.calculate_L1_norm();
       }
       case L2:
       {
            return parameters.calculate_L2_norm();
       }
       case None:
       {
            return 0.0;
       }
    }

    return 0.0;
}


Vector<double> LossIndex::calculate_regularization_gradient() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    switch(regularization_method)
    {
       case L1:
       {
            return neural_network_pointer->get_parameters().calculate_L1_norm_gradient();
       }
       case L2:
       {
            return neural_network_pointer->get_parameters().calculate_L2_norm_gradient();
       }
       case None:
       {
            return Vector<double>(neural_network_pointer->get_parameters_number(), 0.0);
       }
    }

    return Vector<double>();
}


Vector<double> LossIndex::calculate_regularization_gradient(const Vector<double>& parameters) const
{
    switch(regularization_method)
    {
       case L1:
       {
            return parameters.calculate_L1_norm_gradient();
       }
       case L2:
       {
            return parameters.calculate_L2_norm_gradient();
       }
       case None:
       {
            return Vector<double>(parameters.size(), 0.0);
       }
    }

    return Vector<double>();
}


Matrix<double> LossIndex::calculate_regularization_Hessian() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    switch(regularization_method)
    {
       case L1:
       {
            return neural_network_pointer->get_parameters().calculate_L1_norm_Hessian();
       }
       case L2:
       {
            return neural_network_pointer->get_parameters().calculate_L2_norm_Hessian();
       }
       case None:
       {
            const size_t parameters_number = neural_network_pointer->get_parameters_number();

            return Matrix<double>(parameters_number,parameters_number,0.0);
       }
    }

    return Matrix<double>();
}



Matrix<double> LossIndex::calculate_regularization_Hessian(const Vector<double>& parameters) const
{
    switch(regularization_method)
    {
       case L1:
       {
            return parameters.calculate_L1_norm_Hessian();
       }
       case L2:
       {
            return parameters.calculate_L2_norm_Hessian();
       }
       case None:
       {
            const size_t parameters_number = parameters.size();

            return Matrix<double>(parameters_number,parameters_number,0.0);
       }
    }

    return Matrix<double>();
}


/// Returns the default string representation of a error term.

string LossIndex::object_to_string() const
{
   ostringstream buffer;

   buffer << "Error term\n";
          //<< "Display: " << display << "\n";

   return(buffer.str());
}


/// Serializes a default error term object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document.

tinyxml2::XMLDocument* LossIndex::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Error term

   tinyxml2::XMLElement* root_element = document->NewElement("LossIndex");

   document->InsertFirstChild(root_element);

   return(document);
}


/// Serializes a default error term object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void LossIndex::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("LossIndex");

    file_stream.CloseElement();
}


void LossIndex::regularization_from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("Regularization");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LossIndex class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Regularization tag not found.\n";

        throw logic_error(buffer.str());
    }

    const string new_regularization_method = root_element->Attribute("Type");

    set_regularization_method(new_regularization_method);

    const tinyxml2::XMLElement* element = root_element->FirstChildElement("NeuralParametersNormWeight");

    if(element)
    {
       const double new_regularization_weight = atof(element->GetText());

       try
       {
          set_regularization_weight(new_regularization_weight);
       }
       catch(const logic_error& e)
       {
          cerr << e.what() << endl;
       }
    }
}


void LossIndex::write_regularization_XML(tinyxml2::XMLPrinter& file_stream) const
{
     ostringstream buffer;

     file_stream.OpenElement("Regularization");

     // Regularization method

     switch (regularization_method)
     {
        case L1:
        {
            file_stream.PushAttribute("Type", "L1_NORM");
        }
        break;

        case L2:
        {
            file_stream.PushAttribute("Type", "L2_NORM");
        }
        break;

        case None:
        {
            file_stream.PushAttribute("Type", "NO_REGULARIZATION");
        }
        break;
     }

     // Regularization weight

     file_stream.OpenElement("NeuralParametersNormWeight");

     buffer.str("");
     buffer << regularization_weight;

     file_stream.PushText(buffer.str().c_str());

     // Close regularization weight

     file_stream.CloseElement();

     // Close regularization

     file_stream.CloseElement();
}


/// Loads a default error term from a XML document.
/// @param document TinyXML document containing the error term members.

void LossIndex::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("MeanSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MeanSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Mean squared element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Regularization

    tinyxml2::XMLDocument regularization_document;
    tinyxml2::XMLNode* element_clone;

    const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");

    element_clone = regularization_element->DeepClone(&regularization_document);

    regularization_document.InsertFirstChild(element_clone);

    regularization_from_XML(regularization_document);
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
