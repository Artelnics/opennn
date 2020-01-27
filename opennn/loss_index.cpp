//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S                                       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "loss_index.h"

namespace OpenNN
{

/// Default constructor. 
/// It creates a default error term object, with all pointers initialized to nullptr.
/// It also initializes all the rest of class members to their default values.

LossIndex::LossIndex()
 : neural_network_pointer(nullptr), 
   data_set_pointer(nullptr)
{
   set_default();
}


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


/// Destructor.

LossIndex::~LossIndex()
{
}


/// Returns regularization weights.

const type& LossIndex::get_regularization_weight() const
{
   return regularization_weight;
}


/// Returns true if messages from this class can be displayed on the screen, or false if messages
/// from this class can't be displayed on the screen.

const bool& LossIndex::get_display() const
{
   return display;
}


/// Returns true if this error term object has a neural nework class pointer associated,
/// and false otherwise

bool LossIndex::has_neural_network() const
{
    if(neural_network_pointer)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns true if this error term object has a data set pointer associated,
/// and false otherwise.

bool LossIndex::has_data_set() const
{
    if(data_set_pointer)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns the regularization method

LossIndex::RegularizationMethod LossIndex::get_regularization_method() const
{
    return regularization_method;
}


/// Sets all the member pointers to nullptr(neural network, data set).
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

void LossIndex::set_default()
{
   regularization_method = L2;
   display = true;
}


/// Sets the object with the regularization method.
/// @param new_regularization_method String with method.

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
        set_regularization_method(NoRegularization);
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


/// Sets the object with the regularization method.
/// @param new_regularization_method String with method.

void LossIndex::set_regularization_method(const LossIndex::RegularizationMethod& new_regularization_method)
{
    regularization_method = new_regularization_method;
}


/// Sets the object with the regularization weights.
/// @param new_regularization_method New regularization weight.

void LossIndex::set_regularization_weight(const type& new_regularization_weight)
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


/// Returns true if there are selection instances and false otherwise.

bool LossIndex::has_selection() const
{
   if(data_set_pointer->get_selection_instances_number() != 0)
   {
       return true;
   }
   else
   {
       return false;
   }
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

   const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

  if(trainable_layers_number == 0)
  {
        buffer << "OpenNN Exception: LossIndex class.\n"
            << "void check() const method.\n"
            << "Neural network has no layers.\n";

        throw logic_error(buffer.str());
  }

  // Data set stuff

  if(!data_set_pointer)
  {
     buffer << "OpenNN Exception: LossIndex class.\n"
            << "void check() const method.\n"
            << "Pointer to data set is nullptr.\n";

     throw logic_error(buffer.str());
  }

  const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

  const Index inputs_number = trainable_layers_pointers[0]->get_inputs_number();
  const Index outputs_number = trainable_layers_pointers[trainable_layers_number-1]->get_neurons_number();

  const Index data_set_inputs_number = data_set_pointer->get_input_variables_number();
  const Index targets_number = data_set_pointer->get_target_variables_number();

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
            << "Number of outputs in neural network (" << outputs_number << ") must be equal to number of targets in data set (" << targets_number << ").\n";

     throw logic_error(buffer.str());
  }
}


/// Calculate the gradient error from layers.
/// Returns the gradient of the objective, according to the objective type.
/// That gradient is the vector of partial derivatives of the objective with respect to the parameters.
/// The size is thus the number of parameters.
/// @param inputs Tensor with inputs.
/// @param layers_activations vector of tensors with layers activations.
/// @param layers_delta vector of tensors with layers delta.

Tensor<type, 1> LossIndex::calculate_error_gradient(const Tensor<type, 2>& inputs,
                                                   const Tensor<Layer::ForwardPropagation, 1>& forward_propagation,
                                                   const Tensor<Tensor<type, 2>, 1>& layers_delta) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    #ifdef __OPENNN_DEBUG__

    check();

    // Hidden errors size

    const Index layers_delta_size = layers_delta.size();

    if(layers_delta_size != trainable_layers_number)
    {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Tensor<Tensor<type, 1>, 1> calculate_layers_error_gradient(const Tensor<Tensor<type, 1>, 1>&, const Tensor<type, 1>&) method.\n"
             << "Size of layers delta(" << layers_delta_size << ") must be equal to number of layers(" << trainable_layers_number << ").\n";

      throw logic_error(buffer.str());
    }

    #endif

    const Index parameters_number = neural_network_pointer->get_trainable_parameters_number();

    const Tensor<Index, 1> trainable_layers_parameters_number = neural_network_pointer->get_trainable_layers_parameters_numbers();

    const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

    Tensor<type, 1> error_gradient(parameters_number);

    Index index = 0;

//    error_gradient.embed(index, trainable_layers_pointers[0]->calculate_error_gradient(inputs, forward_propagation[0], layers_delta[0]));
    for(Index i = 0; i < trainable_layers_pointers[0]->calculate_error_gradient(inputs, forward_propagation[0], layers_delta[0]).size(); i++)
    {
        error_gradient(i + index) = (trainable_layers_pointers[0]->calculate_error_gradient(inputs, forward_propagation[0], layers_delta[0]))(i);
    }

    index += trainable_layers_parameters_number[0];

    for(Index i = 1; i < trainable_layers_number; i++)
    {

//      error_gradient.embed(index, trainable_layers_pointers[i]->calculate_error_gradient(forward_propagation[i-1].activations, forward_propagation[i-1], layers_delta[i]));
      for(Index i = 0; i < trainable_layers_pointers[i]->calculate_error_gradient(forward_propagation[i-1].activations, forward_propagation[i-1], layers_delta[i]).size(); i++)
      {
          error_gradient(i + index) = (trainable_layers_pointers[i]->calculate_error_gradient(
                                      forward_propagation[i-1].activations, forward_propagation[i-1], layers_delta[i]))(i);
      }

      index += trainable_layers_parameters_number[i];
    }

    return error_gradient;
}


/// Calculates the <i>Jacobian</i> matrix of the error terms from layers.
/// Returns the Jacobian of the error terms function, according to the objective type used in the loss index expression.
/// Note that this function is only defined when the objective can be expressed as a sum of squared terms.
/// The Jacobian elements are the partial derivatives of a single term with respect to a single parameter.
/// The number of rows in the Jacobian matrix are the number of parameters, and the number of columns the number of terms composing the objective.
/// @param inputs Tensor with inputs.
/// @param layers_activations vector of tensors with layers activations.
/// @param layers_delta vector of tensors with layers delta.

Tensor<type, 2> LossIndex::calculate_error_terms_Jacobian(const Tensor<type, 2>& inputs,
                                                         const Tensor<Layer::ForwardPropagation, 1>& forward_propagation,
                                                         const Tensor<Tensor<type, 2>, 1>& layers_delta) const
{  
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   const Index layers_number = neural_network_pointer->get_trainable_layers_number();

   #ifdef __OPENNN_DEBUG__

   // Hidden errors size

   const Index layers_delta_size = layers_delta.size();

   if(layers_delta_size != layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Tensor<type, 2> calculate_layers_error_Jacobian(const Tensor<Tensor<type, 1>, 1>&, const Tensor<type, 1>&) method.\n"
             << "Size of layers delta("<< layers_delta_size << ") must be equal to number of layers(" << layers_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const Index parameters_number = neural_network_pointer->get_parameters_number();
   const Index instances_number = data_set_pointer->get_instances_number();

   const Tensor<Index, 1> layers_parameters_number = neural_network_pointer->get_trainable_layers_parameters_numbers();

   Tensor<type, 2> error_Jacobian(instances_number, parameters_number);

   Index index = 0;

//   error_Jacobian.embed(0, index, calculate_layer_error_terms_Jacobian(layers_delta[0], inputs));
   for(Index i = 0; i < calculate_layer_error_terms_Jacobian(layers_delta[0], inputs).size(); i++)
   {
       error_Jacobian(i + index) = (calculate_layer_error_terms_Jacobian(layers_delta[0], inputs))(i);
   }

   index += layers_parameters_number[0];

   for(Index i = 1; i < layers_number; i++)
   {

//      error_Jacobian.embed(0, index, calculate_layer_error_terms_Jacobian(layers_delta[i], forward_propagation[i-1].activations));
      for(Index i = 0; i < calculate_layer_error_terms_Jacobian(layers_delta[i], forward_propagation[i-1].activations).size(); i++)
      {
          error_Jacobian(i + index) = (calculate_layer_error_terms_Jacobian(layers_delta[i], forward_propagation[i-1].activations))(i);
      }

      index += layers_parameters_number[i];
   }

   return error_Jacobian;
}


/// Calculates the <i>Jacobian</i> matrix of the error terms of the layer.
/// Returns the Jacobian of the error terms function, according to the objective type used in the loss index expression.
/// Note that this function is only defined when the objective can be expressed as a sum of squared terms.
/// The Jacobian elements are the partial derivatives of a single layer term with respect to a single layer parameter.
/// The number of rows in the Jacobian matrix are the number of parameters, and the number of columns the number of terms composing the objective.
/// @param layer_deltas Tensor with layers delta.
/// @param layer_inputs Tensor with layers inputs.

Tensor<type, 2> LossIndex::calculate_layer_error_terms_Jacobian(const Tensor<type, 2>& layer_deltas,
                                                               const Tensor<type, 2>& layer_inputs) const
{
    const Index instances_number = layer_inputs.dimension(0);
    const Index inputs_number = layer_inputs.dimension(1);
    const Index neurons_number = layer_deltas.dimension(1);

    const Index synaptic_weights_number = neurons_number*inputs_number;

    Tensor<type, 2> layer_error_Jacobian(instances_number, neurons_number*(1+inputs_number));

    Index parameter;

    for(Index instance = 0; instance < instances_number; instance++)
    {
        parameter = 0;

        for(Index perceptron = 0; perceptron < neurons_number; perceptron++)
        {
            const type layer_delta = layer_deltas(instance, perceptron);

            for(Index input = 0; input < inputs_number; input++)
            {
                layer_error_Jacobian(instance, parameter) = layer_delta*layer_inputs(instance, input);

                parameter++;
            }

            layer_error_Jacobian(instance, synaptic_weights_number+perceptron) = layer_delta;
         }
    }

    return layer_error_Jacobian;
}


/// It calculates training loss, obtaining the term of error and the regularization if it had it.
/// Note that the error term can be obtained by different methodata_set.
/// Returns the training loss.

type LossIndex::calculate_training_loss() const
{
    if(regularization_method == NoRegularization)
    {
        return calculate_training_error();
    }
    else
    {
        return calculate_training_error() + regularization_weight*calculate_regularization();
    }
}


/// It calculates training loss, obtaining the term of error and the regularization if it had it.
/// Note that the error term can be obtained by different methodata_set.
/// Returns the training loss.
/// @param parameters vector with the parameters to get the training loss term.

type LossIndex::calculate_training_loss(const Tensor<type, 1>& parameters) const
{
    if(regularization_method == NoRegularization)
    {
        return calculate_training_error(parameters);
    }
    else
    {
        return calculate_training_error(parameters) + regularization_weight*calculate_regularization(parameters);
    }
}


/// Returns the value of the loss function at some step along some direction.

type LossIndex::calculate_training_loss(const Tensor<type, 1>& direction, const type& rate) const
{    
    const Tensor<type, 1> parameters = neural_network_pointer->get_parameters();

    return calculate_training_loss(parameters + direction*rate);
}


/// It calculates training loss using the gradient method, obtaining the term of error and the regularization if it had it.
/// Note that the error term can be obtained by different methods.
/// That gradient is the vector of partial derivatives of the loss index with respect to the parameters.
/// Returns the training loss.

Tensor<type, 1> LossIndex::calculate_training_loss_gradient() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    if(regularization_method == NoRegularization)
    {
        return calculate_training_error_gradient();
    }
    else
    {
        return calculate_training_error_gradient() + calculate_regularization_gradient()*regularization_weight;
    }
}


/// Returns a string with the default type of error term, "USER_PERFORMANCE_TERM".

string LossIndex::get_error_type() const
{
   return "USER_ERROR_TERM";
}


/// Returns a string with the default type of error term in text format, "USER_PERFORMANCE_TERM".

string LossIndex::get_error_type_text() const
{
   return "USER_ERROR_TERM";
}


/// Returns a string with the default information of the error term.
/// It will be used by the training strategy to monitor the training process. 
/// By default this information is empty. 

string LossIndex::write_information() const
{
   return string();
}


/// Returns a string with the regularization information of the error term.
/// It will be used by the training strategy to monitor the training process.

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
       case NoRegularization:
       {
            return "NO_REGULARIZATION";
       }
    }

    return string();
}


/// It calculate the regularization term using different methods.
/// Returns the regularization evaluation, according to the respective regularization type used in the loss index expression.

type LossIndex::calculate_regularization() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    switch(regularization_method)
    {
       case L1:
       {
            return l1_norm(neural_network_pointer->get_parameters());
       }
       case L2:
       {
            return l2_norm(neural_network_pointer->get_parameters());
       }
       case NoRegularization:
       {
            return 0.0;
       }
    }

    return 0.0;
}


/// It calculates the regularization term using through the use of parameters.
/// Returns the regularization evaluation, according to the respective regularization type used in the loss index expression.
/// @param parameters vector with the parameters to get the regularization term.

type LossIndex::calculate_regularization(const Tensor<type, 1>& parameters) const
{

    switch(regularization_method)
    {
       case L1:
       {
            return l1_norm(parameters);
       }
       case L2:
       {
            return l2_norm(parameters);
       }
       case NoRegularization:
       {
            return 0.0;
       }
    }

    return 0.0;
}


/// It calculate the regularization term using the gradient method.
/// Returns the gradient of the regularization, according to the regularization type.
/// That gradient is the vector of partial derivatives of the regularization with respect to the parameters.
/// The size is thus the number of parameters.

Tensor<type, 1> LossIndex::calculate_regularization_gradient() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    switch(regularization_method)
    {
       case L1:
       {
            return l1_norm_gradient(neural_network_pointer->get_parameters());
       }
       case L2:
       {
            return l2_norm_gradient(neural_network_pointer->get_parameters());
       }
       case NoRegularization:
       {
            return Tensor<type, 1>(neural_network_pointer->get_parameters_number()).setConstant(0.0);
       }
    }

    return Tensor<type, 1>();
}


/// It calculate the regularization term using the gradient method.
/// Returns the gradient of the regularization, according to the regularization type.
/// That gradient is the vector of partial derivatives of the regularization with respect to the parameters.
/// The size is thus the number of parameters
/// @param parameters vector with the parameters to get the regularization term.

Tensor<type, 1> LossIndex::calculate_regularization_gradient(const Tensor<type, 1>& parameters) const
{

    switch(regularization_method)
    {
       case L1:
       {
            return l1_norm_gradient(parameters);
       }
       case L2:
       {
            return l2_norm_gradient(parameters);
       }
       case NoRegularization:
       {
            return Tensor<type, 1>(parameters.size()).setConstant(0.0);
       }
    }

    return Tensor<type, 1>();
}


/// It calculate the regularization term using the <i>Hessian</i>.
/// Returns the <i>Hessian</i> of the regularization, according to the regularization type.
/// That Hessian is the matrix of second partial derivatives of the regularization with respect to the parameters.
/// That matrix is symmetric, with size the number of parameters.

Tensor<type, 2> LossIndex::calculate_regularization_hessian() const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    switch(regularization_method)
    {
       case L1:
       {
            return l1_norm_hessian(neural_network_pointer->get_parameters());
       }
       case L2:
       {
            return l2_norm_hessian(neural_network_pointer->get_parameters());
       }
       case NoRegularization:
       {
            const Index parameters_number = neural_network_pointer->get_parameters_number();

            return Tensor<type, 2>(parameters_number,parameters_number).setConstant(0.0);
       }
    }

    return Tensor<type, 2>();
}


/// It calculate the regularization term using the <i>Hessian</i>.
/// Returns the Hessian of the regularization, according to the regularization type.
/// That Hessian is the matrix of second partial derivatives of the regularization with respect to the parameters.
/// That matrix is symmetric, with size the number of parameters.
/// @param parameters vector with the parameters to get the regularization term.

Tensor<type, 2> LossIndex::calculate_regularization_hessian(const Tensor<type, 1>& parameters) const
{

    switch(regularization_method)
    {
       case L1:
       {
            return l1_norm_hessian(parameters);
       }
       case L2:
       {
            return l2_norm_hessian(parameters);
       }
       case NoRegularization:
       {
            const Index parameters_number = parameters.size();

            return Tensor<type, 2>(parameters_number,parameters_number).setConstant(0.0);
       }
    }

    return Tensor<type, 2>();
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

   return document;
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
       const type new_regularization_weight = atof(element->GetText());

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

        case NoRegularization:
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


/// Default constructor.
/// Set of loss value and gradient vector of the loss index.
/// A method returning this structure might be implemented more efficiently than the loss and gradient methods separately.

LossIndex::FirstOrderLoss::FirstOrderLoss(const LossIndex* loss_index_pointer)
{
    // Data set

    DataSet* data_set_pointer = loss_index_pointer->get_data_set_pointer();

    const Index batch_instances_number = data_set_pointer->get_batch_instances_number();

    // Neural network

    NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

    const Index parameters_number = neural_network_pointer->get_parameters_number();

    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Index outputs_number = neural_network_pointer->get_outputs_number();

    const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

    // First order loss

    output_gradient = Tensor<type, 2>(batch_instances_number, outputs_number);

    layers_delta.resize(trainable_layers_number);

    layers_error_gradient.resize(trainable_layers_number);

    for(Index i = 0; i < trainable_layers_number; i++)
    {
        const Layer::Type layer_type = trainable_layers_pointers[i]->get_type();

        if(layer_type == Layer::Convolutional)
        {
            ConvolutionalLayer* layer_pointer = dynamic_cast<ConvolutionalLayer*>(trainable_layers_pointers[i]);

            const Index output_channels_number = layer_pointer->get_filters_number();
            const Index output_rows_number = layer_pointer->get_outputs_rows_number();
            const Index output_columns_number = layer_pointer->get_outputs_columns_number();
/*
            layers_delta[i].resize(Tensor<Index, 1>({batch_instances_number, output_channels_number, output_rows_number, output_columns_number}));
*/
        }
        else if(layer_type == Layer::Pooling)
        {
            PoolingLayer* layer_pointer = dynamic_cast<PoolingLayer*>(trainable_layers_pointers[i]);

            const Index output_channels_number = layer_pointer->get_inputs_channels_number();
            const Index output_rows_number = layer_pointer->get_outputs_rows_number();
            const Index output_columns_number = layer_pointer->get_outputs_columns_number();
/*
            layers_delta[i].resize(Tensor<Index, 1>({batch_instances_number, output_channels_number, output_rows_number, output_columns_number}));
*/
        }
        else if(layer_type == Layer::Perceptron)
        {
            PerceptronLayer* layer_pointer = dynamic_cast<PerceptronLayer*>(trainable_layers_pointers[i]);

            const Index neurons_number = layer_pointer->get_neurons_number();

            layers_delta[i] = Tensor<type, 2>(batch_instances_number, neurons_number);
            layers_delta[i].setRandom();
        }
        else if(layer_type == Layer::Recurrent)
        {
            /// @todo add
        }
        else if(layer_type == Layer::LongShortTermMemory)
        {
            /// @todo add
        }
        else if(layer_type == Layer::Probabilistic)
        {
            ProbabilisticLayer* layer_pointer = dynamic_cast<ProbabilisticLayer*>(trainable_layers_pointers[i]);

            const Index output_columns_number = layer_pointer->get_neurons_number();

            layers_delta[i] = Tensor<type, 2>(batch_instances_number, output_columns_number);
        }
        else
        {
            /// @todo add exception
        }

        layers_error_gradient[i] = Tensor<type, 1>(trainable_layers_pointers[i]->get_parameters_number());
        layers_error_gradient[i].setRandom();
    }

    loss = 0.0;

    error_gradient = Tensor<type, 1>(parameters_number);
    regularization_gradient = Tensor<type, 1>(parameters_number);
    gradient = Tensor<type, 1>(parameters_number);

    error_gradient.setRandom();
    regularization_gradient.setRandom();
    gradient.setRandom();
}


/// Destructor.

LossIndex::FirstOrderLoss::~FirstOrderLoss()
{
}


Tensor<Tensor<type, 2>, 1> LossIndex::calculate_layers_delta(const Tensor<Layer::ForwardPropagation, 1>& forward_propagation,
                                                         const Tensor<type, 2>& output_gradient) const
{
    const Index forward_propagation_size = forward_propagation.size();

   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

    check();

   const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

   if(forward_propagation_size != trainable_layers_number)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LossIndex class.\n"
             << "Tensor<Tensor<type, 2>, 1> calculate_layers_delta(const Tensor<Tensor<type, 2>, 1>&, const Tensor<type, 2>&) method.\n"
             << "Size of forward propagation activation derivative vector ("<< forward_propagation_size << ") must be equal to number of layers (" << trainable_layers_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

   Tensor<Tensor<type, 2>, 1> layers_delta(forward_propagation_size);

   if(forward_propagation_size == 0) return layers_delta;

   // Output layer

   layers_delta[forward_propagation_size-1] = trainable_layers_pointers[forward_propagation_size-1]
           ->calculate_output_delta(forward_propagation[forward_propagation_size-1].activations_derivatives, output_gradient);

   // Hidden layers

   for(Index i = static_cast<Index>(forward_propagation_size)-2; i >= 0; i--)
   {
       Layer* previous_layer_pointer = trainable_layers_pointers[static_cast<Index>(i+1)];

       layers_delta[i] = trainable_layers_pointers[i]
               ->calculate_hidden_delta(previous_layer_pointer,
                                        forward_propagation[i].activations,
                                        forward_propagation[i].activations_derivatives,
                                        layers_delta[static_cast<Index>(i+1)]);
   }

   return layers_delta;
}


/// This method separates training instances and calculates batches from the dataset.
/// It also calculates the outputs and the sum squared error from the targets and outputs.
/// Returns a sum squared error of the training instances.

type LossIndex::calculate_training_error() const
{

#ifdef __OPENNN_DEBUG__

check();

#endif

    //Neural network

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    Tensor<Index, 2> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const Index batches_number = training_batches.size();

    type training_error = 0.0;

    #pragma omp parallel for reduction(+ : training_error)

    for(Index i = 0; i < batches_number; i++)
    {
        const type batch_error = calculate_batch_error(training_batches.chip(i,0));

        training_error += batch_error;
    }

    return training_error;
}


type LossIndex::calculate_training_error(const Tensor<type, 1>& parameters) const
{

#ifdef __OPENNN_DEBUG__

check();

#endif

    //Neural network

    bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    Tensor<Index, 2> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const Index batches_number = training_batches.size();

    type training_error = 0.0;

    #pragma omp parallel for reduction(+ : training_error)

    for(Index i = 0; i < batches_number; i++)
    {
        const type batch_error = calculate_batch_error(training_batches.chip(i,0), parameters);

        training_error += batch_error;
    }

    return training_error;
}


/// This method separates selection instances and calculates batches from the dataset.
/// It also calculates the outputs and the sum squared error from the targets and outputs.
/// Returns a sum squared error of the training instances.

type LossIndex::calculate_selection_error() const
{

#ifdef __OPENNN_DEBUG__

check();

#endif

    //Neural network

     bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    Tensor<Index, 2> selection_batches = data_set_pointer->get_selection_batches(!is_forecasting);

    const Index batches_number = selection_batches.size();

    type selection_error = 0.0;

    #pragma omp parallel for reduction(+ : selection_error)

    for(Index i = 0; i < batches_number; i++)
    {
        const type batch_error = calculate_batch_error(selection_batches.chip(i,0));

        selection_error += batch_error;
    }

    return selection_error;
}


/// This method calculates the error term gradient for batch instances.
/// It is used for optimization of parameters during training.
/// Returns the value of the error term gradient.
/// @param batch_indices Indices of the batch instances corresponding to the dataset.

Tensor<type, 1> LossIndex::calculate_batch_error_gradient(const Tensor<Index, 1>& batch_indices) const
{
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const Tensor<type, 2> inputs = data_set_pointer->get_input_data(batch_indices);
    const Tensor<type, 2> targets = data_set_pointer->get_target_data(batch_indices);

    const Tensor<Layer::ForwardPropagation, 1> forward_propagation = neural_network_pointer->calculate_forward_propagation(inputs);

    const Index size = forward_propagation.size();

    const Tensor<type, 2> output_gradient = calculate_output_gradient(forward_propagation[size-1].activations, targets);

    const Tensor<Tensor<type, 2>, 1> layers_delta = calculate_layers_delta(forward_propagation, output_gradient);

    return calculate_error_gradient(inputs, forward_propagation, layers_delta);
}


/// This method calculates the error term gradient for training instances.
/// It is used for optimization of parameters during training.
/// Returns the value of the error term gradient.

Tensor<type, 1> LossIndex::calculate_training_error_gradient() const
{

#ifdef __OPENNN_DEBUG__

check();

#endif

    // Neural network

    const Index parameters_number = neural_network_pointer->get_parameters_number();
    bool is_forecasting = false;

    if(neural_network_pointer->has_long_short_term_memory_layer() || neural_network_pointer->has_recurrent_layer()) is_forecasting = true;

    // Data set

    Tensor<Index, 2> training_batches = data_set_pointer->get_training_batches(!is_forecasting);

    const Index batches_number = training_batches.size();

    // Loss index

    Tensor<type, 1> training_error_gradient(parameters_number);

    #pragma omp parallel for

    for(Index i = 0; i < batches_number; i++)
    {
        const Tensor<type, 1> batch_gradient = calculate_batch_error_gradient(training_batches.chip(i,0));

        #pragma omp critical

        training_error_gradient += batch_gradient;
    }

    return training_error_gradient;

}



Tensor<type, 1> LossIndex::calculate_training_error_gradient_numerical_differentiation() const
{
    NumericalDifferentiation numerical_differentiation;

    numerical_differentiation.set_numerical_differentiation_method(NumericalDifferentiation::CentralDifferences);

    const Tensor<type, 1> parameters = neural_network_pointer->get_parameters();

    return numerical_differentiation.calculate_gradient(*this, &LossIndex::calculate_training_error, parameters);

}


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
