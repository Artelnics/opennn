//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "neural_network_forward_propagation.h"
#include "loss_index.h"
#include "loss_index_back_propagation.h"

namespace opennn
{

/// Default constructor.
/// It creates a default error term object, with all pointers initialized to nullptr.
/// It also initializes all the rest of the class members to their default values.

LossIndex::LossIndex()
{
    set_default();
}


/// Neural network and data set constructor.
/// It creates a error term object associated with a neural network and to be measured on a data set.
/// It initializes the rest of pointers to nullptr.
/// It also initializes all the rest of the class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

LossIndex::LossIndex(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : neural_network_pointer(new_neural_network_pointer),
      data_set_pointer(new_data_set_pointer)
{
    set_default();
}


/// Destructor.

LossIndex::~LossIndex()
{
    delete thread_pool;
    delete thread_pool_device;
}


/// Returns regularization weight.

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
/// It also initializes all the rest of the class members to their default values.

void LossIndex::set()
{
    neural_network_pointer = nullptr;
    data_set_pointer = nullptr;

    set_default();
}


/// Sets all the member pointers to nullptr, but the neural network, which set to a given pointer.
/// It also initializes all the rest of the class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

void LossIndex::set(NeuralNetwork* new_neural_network_pointer)
{
    neural_network_pointer = new_neural_network_pointer;
    data_set_pointer = nullptr;

    set_default();
}


/// Sets all the member pointers to nullptr, but the data set, which set to a given pointer.
/// It also initializes all the rest of the class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

void LossIndex::set(DataSet* new_data_set_pointer)
{
    neural_network_pointer = nullptr;
    data_set_pointer = new_data_set_pointer;

    set_default();
}


/// Sets new neural network and data set pointers.
/// Finally, it initializes all the rest of the class members to their default values.
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


void LossIndex::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr) delete thread_pool;
    if(thread_pool_device != nullptr) delete thread_pool_device;

    thread_pool = new ThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(thread_pool, new_threads_number);
}


/// Sets a pointer to a neural network object which is to be associated with the error term.
/// @param new_neural_network_pointer Pointer to a neural network object to be associated with the error term.

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
    delete thread_pool;
    delete thread_pool_device;

    const int n = omp_get_max_threads();

    thread_pool = new ThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(thread_pool, n);

    regularization_method = RegularizationMethod::L2;
}


/// Sets the object with the regularization method.
/// @param new_regularization_method String with method.

void LossIndex::set_regularization_method(const string& new_regularization_method)
{
    if(new_regularization_method == "L1_NORM")
    {
        set_regularization_method(RegularizationMethod::L1);
    }
    else if(new_regularization_method == "L2_NORM")
    {
        set_regularization_method(RegularizationMethod::L2);
    }
    else if(new_regularization_method == "NO_REGULARIZATION")
    {
        set_regularization_method(RegularizationMethod::NoRegularization);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LossIndex class.\n"
               << "void set_regularization_method(const string&) const method.\n"
               << "Unknown regularization method: " << new_regularization_method << ".";

        throw invalid_argument(buffer.str());
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
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void LossIndex::set_display(const bool& new_display)
{
    display = new_display;
}


/// Returns true if there are selection samples and false otherwise.

bool LossIndex::has_selection() const
{
    if(data_set_pointer->get_selection_samples_number() != 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Checks whether there is a neural network associated with the error term.
/// If some of the above conditions is not hold, the method throws an exception.

void LossIndex::check() const
{
    ostringstream buffer;

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: LossIndex class.\n"
               << "void check() const.\n"
               << "Pointer to neural network is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Data set

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: LossIndex class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw invalid_argument(buffer.str());
    }
}


void LossIndex::calculate_errors(const DataSetBatch& batch,
                                 const ForwardPropagation& forward_propagation,
                                 BackPropagation& back_propagation) const
{
    const Index last_trainable_layer_index = neural_network_pointer->get_last_trainable_layer_index();

    const LayerForwardPropagation* output_layer_forward_propagation = forward_propagation.layers(last_trainable_layer_index);
    
    const pair<type*, dimensions> outputs_pair = output_layer_forward_propagation->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0][0], outputs_pair.second[0][1]);

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0][0], targets_pair.second[0][1]);

    back_propagation.errors.device(*thread_pool_device) = outputs - targets;
}


void LossIndex::calculate_errors(const pair<type*, dimensions>& targets_pair,
                                 const ForwardPropagation& forward_propagation,
                                 BackPropagation& back_propagation) const
{
    const Index last_trainable_layer_index = neural_network_pointer->get_last_trainable_layer_index();

    const LayerForwardPropagation* output_layer_forward_propagation = forward_propagation.layers(last_trainable_layer_index);

    const pair<type*, dimensions> outputs_pair = output_layer_forward_propagation->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0][0], outputs_pair.second[0][1]);

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0][0], targets_pair.second[0][1]);

    back_propagation.errors.device(*thread_pool_device) = outputs - targets;
}


void LossIndex::calculate_errors_lm(const DataSetBatch& batch,
                                    const ForwardPropagation & neural_network_forward_propagation,
                                    LossIndexBackPropagationLM & loss_index_back_propagation) const
{
    const Index last_trainable_layer_index = neural_network_pointer->get_last_trainable_layer_index();
    
    const pair<type*, dimensions> outputs_pair = neural_network_forward_propagation.layers(last_trainable_layer_index)->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0][0], outputs_pair.second[0][1]);

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0][0], targets_pair.second[0][1]);

    loss_index_back_propagation.errors.device(*thread_pool_device) = outputs - targets;
}


void LossIndex::calculate_squared_errors_lm(const DataSetBatch&,
                                            const ForwardPropagation&,
                                            LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
    const Tensor<type, 2>& errors = loss_index_back_propagation_lm.errors;

    Tensor<type, 1>& squared_errors = loss_index_back_propagation_lm.squared_errors;

    squared_errors.device(*thread_pool_device) = errors.square().sum(rows_sum).sqrt();
}


void LossIndex::back_propagate(const DataSetBatch& batch,
                               ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation) const
{
    // Loss index
   
    calculate_error(batch, forward_propagation, back_propagation);

    calculate_layers_error_gradient(batch, forward_propagation, back_propagation);

    // Loss

    back_propagation.loss = back_propagation.error;

    // Regularization

    if(regularization_method != RegularizationMethod::NoRegularization)
    {
        const type regularization = calculate_regularization(back_propagation.parameters);

        back_propagation.regularization = regularization;

        back_propagation.loss += regularization_weight * regularization;

        calculate_regularization_gradient(back_propagation.parameters, back_propagation.regularization_gradient);

        back_propagation.gradient.device(*thread_pool_device) += regularization_weight * back_propagation.regularization_gradient;
    }

    // Assemble gradient

    assemble_layers_error_gradient(back_propagation);

}


/// This method calculates the second-order loss.
/// It is used for optimization of parameters during training.
/// Returns a second-order terms loss structure, which contains the values and the Hessian of the error terms function.

void LossIndex::back_propagate_lm(const DataSetBatch& batch,
                                  ForwardPropagation& forward_propagation,
                                  LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
    calculate_errors_lm(batch, forward_propagation, loss_index_back_propagation_lm);

    calculate_squared_errors_lm(batch, forward_propagation, loss_index_back_propagation_lm);

    calculate_error_lm(batch, forward_propagation, loss_index_back_propagation_lm);

    calculate_layers_delta_lm(batch, forward_propagation, loss_index_back_propagation_lm);

    calculate_squared_errors_jacobian_lm(batch, forward_propagation, loss_index_back_propagation_lm);

    calculate_error_gradient_lm(batch, loss_index_back_propagation_lm);

    calculate_error_hessian_lm(batch, loss_index_back_propagation_lm);

    // Loss

    loss_index_back_propagation_lm.loss = loss_index_back_propagation_lm.error;

    // Regularization

    if(regularization_method != RegularizationMethod::NoRegularization)
    {
        const type regularization = calculate_regularization(loss_index_back_propagation_lm.parameters);

        loss_index_back_propagation_lm.loss += regularization_weight*regularization;

        calculate_regularization_gradient(loss_index_back_propagation_lm.parameters, loss_index_back_propagation_lm.regularization_gradient);

        loss_index_back_propagation_lm.gradient.device(*thread_pool_device) += regularization_weight*loss_index_back_propagation_lm.regularization_gradient;

        calculate_regularization_hessian(loss_index_back_propagation_lm.parameters, loss_index_back_propagation_lm.regularization_hessian);

        loss_index_back_propagation_lm.hessian += regularization_weight*loss_index_back_propagation_lm.regularization_hessian;
    }
}


/// Calculates the <i>Jacobian</i> matrix of the error terms from layers.
/// Returns the Jacobian of the error terms function, according to the objective type used in the loss index expression.
/// Note that this function is only defined when the objective can be expressed as a sum of squared terms.
/// The Jacobian elements are the partial derivatives of a single term with respect to a single parameter.
/// The number of rows in the Jacobian matrix are the number of parameters, and the number of columns
/// the number of terms composing the objective.
/// @param inputs Tensor with inputs.
/// @param layers_activations vector of tensors with layers activations.
/// @param layers_delta vector of tensors with layers delta.

void LossIndex::calculate_squared_errors_jacobian_lm(const DataSetBatch& batch,
                                                  ForwardPropagation& forward_propagation,
                                                  LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Index first_trainable_layer_index = neural_network_pointer->get_first_trainable_layer_index();

    loss_index_back_propagation_lm.squared_errors_jacobian.setZero();

    const Index batch_samples_number = batch.get_batch_samples_number();

    Index memory_index = 0;

    Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

    Tensor<Layer*, 1> layers_pointers = neural_network_pointer->get_layers_pointers();

    const Tensor<Index, 1> trainable_layers_parameters_number = neural_network_pointer->get_trainable_layers_parameters_numbers();

    // Layer 0

    const Layer::Type first_trainable_layer_type = layers_pointers(first_trainable_layer_index)->get_type();

    if(first_trainable_layer_type != Layer::Type::Perceptron
    && first_trainable_layer_type != Layer::Type::Probabilistic)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LossIndex class.\n"
               << "void calculate_squared_errors_jacobian_lm(const DataSetBatch&, NeuralNetworkForwardPropagation&, LossIndexBackPropagationLM&) const method "
               << "Levenberg - Marquardt algorithm can only be used with Perceptron and Probabilistic layers.\n";

        throw invalid_argument(buffer.str());
    }
    else
    {
        const pair<type*, dimensions> inputs_pair = batch.get_inputs_pair();

        const TensorMap<Tensor<type, 2>> inputs(inputs_pair.first, inputs_pair.second[0][0], inputs_pair.second[0][1]);

        layers_pointers(first_trainable_layer_index)
            ->calculate_squared_errors_Jacobian_lm(inputs,
                                                   forward_propagation.layers(first_trainable_layer_index),
                                                   loss_index_back_propagation_lm.neural_network.layers(0));

        layers_pointers(first_trainable_layer_index)
            ->insert_squared_errors_Jacobian_lm(loss_index_back_propagation_lm.neural_network.layers(0),
                                                memory_index,
                                                loss_index_back_propagation_lm.squared_errors_jacobian);

        memory_index += trainable_layers_parameters_number(0)*batch_samples_number;
    }

    // Rest of the layers

    for(Index i = 1; i < trainable_layers_number; i++)
    {
        switch(forward_propagation.layers(first_trainable_layer_index + i - 1)->layer_pointer->get_type())
        {
        case Layer::Type::Perceptron:
        {
            const PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
                    = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.layers(first_trainable_layer_index + i - 1));

            const Tensor<type, 2>& outputs = perceptron_layer_forward_propagation->outputs;

            trainable_layers_pointers(i)->calculate_squared_errors_Jacobian_lm(outputs,
                                                                   forward_propagation.layers(first_trainable_layer_index + i),
                                                                   loss_index_back_propagation_lm.neural_network.layers(i));

            trainable_layers_pointers(i)->insert_squared_errors_Jacobian_lm(loss_index_back_propagation_lm.neural_network.layers(i),
                                                                            memory_index,
                                                                            loss_index_back_propagation_lm.squared_errors_jacobian);

            memory_index += trainable_layers_parameters_number(i)*batch_samples_number;
        }
            break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: LossIndex class.\n"
                   << "void calculate_squared_errors_jacobian_lm(const DataSetBatch&, NeuralNetworkForwardPropagation&, LossIndexBackPropagationLM&) const method "
                   << "Levenberg - Marquardt algorithm can only be used with Perceptron and Probabilistic layers.\n";

            throw invalid_argument(buffer.str());
        }
        }
    }
}


void LossIndex::calculate_error_gradient_lm(const DataSetBatch& batch,
                                      LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
    loss_index_back_propagation_lm.gradient.device(*thread_pool_device)
            = loss_index_back_propagation_lm.squared_errors_jacobian.contract(loss_index_back_propagation_lm.squared_errors, AT_B);
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


/// Returns a string with the regularization information of the error term.
/// It will be used by the training strategy to monitor the training process.

string LossIndex::write_regularization_method() const
{
    switch(regularization_method)
    {
    case RegularizationMethod::NoRegularization:
        return "NO_REGULARIZATION";

    case RegularizationMethod::L1:
        return "L1_NORM";

    case RegularizationMethod::L2:
        return "L2_NORM";

    default: return string();
    }
}


/// It calculates the regularization term using through the use of parameters.
/// Returns the regularization evaluation, according to the respective regularization type used in the
/// loss index expression.
/// @param parameters vector with the parameters to get the regularization term.

type LossIndex::calculate_regularization(const Tensor<type, 1>& parameters) const
{   
    switch(regularization_method)
    {
        case RegularizationMethod::NoRegularization: return type(0);

        case RegularizationMethod::L1: return l1_norm(thread_pool_device, parameters);

        case RegularizationMethod::L2: return l2_norm(thread_pool_device, parameters);

        default: return type(0);
    }

    return type(0);
}


/// Returns the gradient of the regularization, according to the regularization type.
/// That gradient is the vector of partial derivatives of the regularization with respect to the parameters.
/// The size is thus the number of parameters
/// @param parameters vector with the parameters to get the regularization term.

void LossIndex::calculate_regularization_gradient(const Tensor<type, 1>& parameters, Tensor<type, 1>& regularization_gradient) const
{
    switch(regularization_method)
    {
    case RegularizationMethod::NoRegularization:
        regularization_gradient.setZero(); return;

    case RegularizationMethod::L1:
        l1_norm_gradient(thread_pool_device, parameters, regularization_gradient); return;

    case RegularizationMethod::L2:
        l2_norm_gradient(thread_pool_device, parameters, regularization_gradient); return;

    default:
        return;
    }
}


/// It calculate the regularization term using the <i>Hessian</i>.
/// Returns the Hessian of the regularization, according to the regularization type.
/// That Hessian is the matrix of second partial derivatives of the regularization with respect to the parameters.
/// That matrix is symmetric, with size the number of parameters.
/// @param parameters vector with the parameters to get the regularization term.

void LossIndex::calculate_regularization_hessian(Tensor<type, 1>& parameters, Tensor<type, 2>& regularization_hessian) const
{
    switch(regularization_method)
    {
    case RegularizationMethod::L1:
        l1_norm_hessian(thread_pool_device, parameters, regularization_hessian);

        return;

    case RegularizationMethod::L2:
        l2_norm_hessian(thread_pool_device, parameters, regularization_hessian);

        return;

    default:
        
        return;
    }
}


void LossIndex::calculate_layers_error_gradient(const DataSetBatch& batch,
                                                const ForwardPropagation& forward_propagation,
                                                BackPropagation& back_propagation) const
{
    const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

    const Index trainable_layers_number = trainable_layers_pointers.size();

    if(trainable_layers_number == 0) return;

    Layer* layer = nullptr;

    LayerForwardPropagation* layer_forward_propagation = nullptr;
    LayerForwardPropagation* previous_layer_forward_propagation = nullptr;
    LayerBackPropagation* layer_back_propagation = nullptr;
    LayerBackPropagation* next_layer_back_propagation = nullptr;

    pair<type*, dimensions> inputs_pair;

    // Hidden layers

    for(Index i = trainable_layers_number-1; i >= 0; i--)
    {
        layer = trainable_layers_pointers(i);

        layer_forward_propagation = forward_propagation.layers(i);
        layer_back_propagation = back_propagation.neural_network.layers(i);

        if(i == trainable_layers_number - 1)
        {
//            calculate_output_delta(batch, *layer_forward_propagation, back_propagation);
        }
        else
        {
            next_layer_back_propagation = back_propagation.neural_network.layers(i + 1);

//            layer->calculate_hidden_delta(layer_forward_propagation, next_layer_back_propagation, layer_back_propagation);
        }
       
        if(i == 0)
        {
            inputs_pair = batch.get_inputs_pair();
        }
        else
        {
            previous_layer_forward_propagation = forward_propagation.layers(i-1);
            
            inputs_pair = previous_layer_forward_propagation->get_outputs_pair();
        }

        layer->calculate_error_gradient(inputs_pair, layer_forward_propagation, layer_back_propagation);
    }
}


void LossIndex::calculate_layers_delta_lm(const DataSetBatch& batch,
                                          ForwardPropagation& forward_propagation,
                                          LossIndexBackPropagationLM& back_propagation) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    const Index first_trainable_layer_index = neural_network_pointer->get_first_trainable_layer_index();

    if(trainable_layers_number == 0) return;

    const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

    // Output layer

    calculate_output_delta_lm(batch,
                              forward_propagation,
                              back_propagation);

    // Hidden layers

    for(Index i = trainable_layers_number-2; i >= 0; i--)
    {
        trainable_layers_pointers(i)
                ->calculate_hidden_delta_lm(forward_propagation.layers(first_trainable_layer_index+i+1),
                                            back_propagation.neural_network.layers(i+1),
                                            back_propagation.neural_network.layers(i));
    }
}


void LossIndex::assemble_layers_error_gradient(BackPropagation& back_propagation) const
{
    #ifdef OPENNN_DEBUG

    check();

    #endif

    const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

    const Index trainable_layers_number = trainable_layers_pointers.size();

    const Tensor<Index, 1> trainable_layers_parameters_number
            = neural_network_pointer->get_trainable_layers_parameters_numbers();

    Index index = 0;

    trainable_layers_pointers(0)->insert_gradient(back_propagation.neural_network.layers(0),
                                                  index,
                                                  back_propagation.gradient);

    index += trainable_layers_parameters_number(0);

    for(Index i = 1; i < trainable_layers_number; i++)
    {
        trainable_layers_pointers(i)->insert_gradient(back_propagation.neural_network.layers(i),
                                                      index,
                                                      back_propagation.gradient);

        index += trainable_layers_parameters_number(i);
    }
}


/// Serializes a default error term object into an XML document of the TinyXML library without keeping the DOM tree in memory.
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

        throw invalid_argument(buffer.str());
    }

    const string new_regularization_method = root_element->Attribute("Type");

    set_regularization_method(new_regularization_method);

    const tinyxml2::XMLElement* element = root_element->FirstChildElement("RegularizationWeight");

    if(element)
    {
        const type new_regularization_weight = type(atof(element->GetText()));

        try
        {
            set_regularization_weight(new_regularization_weight);
        }
        catch(const invalid_argument& e)
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

    switch(regularization_method)
    {
    case RegularizationMethod::L1:
    {
        file_stream.PushAttribute("Type", "L1_NORM");
    }
    break;

    case RegularizationMethod::L2:
    {
        file_stream.PushAttribute("Type", "L2_NORM");
    }
    break;

    case RegularizationMethod::NoRegularization:
    {
        file_stream.PushAttribute("Type", "NO_REGULARIZATION");
    }
    break;

    default: break;
    }

    // Regularization weight

    file_stream.OpenElement("RegularizationWeight");

    buffer.str("");
    buffer << regularization_weight;

    file_stream.PushText(buffer.str().c_str());

    // Close regularization weight

    file_stream.CloseElement();

    // Close regularization

    file_stream.CloseElement();
}


/// Loads a default error term from an XML document.
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

        throw invalid_argument(buffer.str());
    }

    // Regularization

    tinyxml2::XMLDocument regularization_document;
    tinyxml2::XMLNode* element_clone;

    const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");

    element_clone = regularization_element->DeepClone(&regularization_document);

    regularization_document.InsertFirstChild(element_clone);

    regularization_from_XML(regularization_document);
}


/// Destructor.

BackPropagation::~BackPropagation()
{
}


Tensor<type, 1> LossIndex::calculate_numerical_gradient()
{
    const Index samples_number = data_set_pointer->get_training_samples_number();

    const Tensor<Index, 1> samples_indices = data_set_pointer->get_training_samples_indices();
    const Tensor<Index, 1> input_variables_indices = data_set_pointer->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set_pointer->get_target_numeric_variables_indices();

    DataSetBatch batch(samples_number, data_set_pointer);
    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network_pointer);

    BackPropagation back_propagation(samples_number, this);

    const Tensor<type, 1> parameters = neural_network_pointer->get_parameters();

    const Index parameters_number = parameters.size();

    type h;
    Tensor<type, 1> parameters_forward(parameters);
    Tensor<type, 1> parameters_backward(parameters);

    type error_forward;
    type error_backward;

    Tensor<type, 1> numerical_gradient(parameters_number);
    numerical_gradient.setConstant(type(0));

    for(Index i = 0; i < parameters_number; i++)
    {
       h = calculate_h(parameters(i));

       parameters_forward(i) += h;
       
       neural_network_pointer->forward_propagate(batch.get_inputs_pair(),
                                                 parameters_forward,
                                                 forward_propagation);

       calculate_errors(batch, forward_propagation, back_propagation);

       calculate_error(batch, forward_propagation, back_propagation);

       error_forward = back_propagation.error;

       parameters_forward(i) -= h;

       parameters_backward(i) -= h;
       
       neural_network_pointer->forward_propagate(batch.get_inputs_pair(),
                                                 parameters_backward,
                                                 forward_propagation);

       calculate_errors(batch, forward_propagation, back_propagation);

       calculate_error(batch, forward_propagation, back_propagation);

       error_backward = back_propagation.error;

       parameters_backward(i) += h;

       numerical_gradient(i) = (error_forward - error_backward)/(type(2)*h);
    }

    return numerical_gradient;
}


Tensor<type, 2> LossIndex::calculate_numerical_jacobian()
{
    LossIndexBackPropagationLM back_propagation_lm;

    const Index samples_number = data_set_pointer->get_training_samples_number();

    DataSetBatch batch(samples_number, data_set_pointer);

    const Tensor<Index, 1> samples_indices = data_set_pointer->get_training_samples_indices();

    const Tensor<Index, 1> input_variables_indices = data_set_pointer->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set_pointer->get_target_numeric_variables_indices();

    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network_pointer);

    BackPropagation back_propagation(samples_number, this);

    Tensor<type, 1> parameters = neural_network_pointer->get_parameters();

    const Index parameters_number = parameters.size();

    back_propagation_lm.set(samples_number, this);
    
    neural_network_pointer->forward_propagate(batch.get_inputs_pair(),
                                              parameters,
                                              forward_propagation);

    calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

    calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

    type h;

    Tensor<type, 1> parameters_forward(parameters);
    Tensor<type, 1> parameters_backward(parameters);

    Tensor<type, 1> error_terms_forward(parameters_number);
    Tensor<type, 1> error_terms_backward(parameters_number);

    Tensor<type, 2> jacobian(samples_number,parameters_number);

    for(Index j = 0; j < parameters_number; j++)
    {
        h = calculate_h(parameters(j));

        parameters_backward(j) -= h;
        neural_network_pointer->forward_propagate(batch.get_inputs_pair(),
                                                  parameters_backward,
                                                  forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);
        error_terms_backward = back_propagation_lm.squared_errors;
        parameters_backward(j) += h;

        parameters_forward(j) += h;
        neural_network_pointer->forward_propagate(batch.get_inputs_pair(),
                                                  parameters_forward,
                                                  forward_propagation);
        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);
        error_terms_forward = back_propagation_lm.squared_errors;
        parameters_forward(j) -= h;

        for(Index i = 0; i < samples_number; i++)
        {
            jacobian(i,j) = (error_terms_forward(i) - error_terms_backward(i))/(type(2.0)*h);
        }
    }

    return jacobian;
}

void LossIndex::calculate_errors(const Tensor<type, 2>& outputs, const Tensor<type, 2>& targets, Tensor<type, 2>& errors) const
{
    errors.device(*thread_pool_device) = outputs - targets;
}

void LossIndex::calculate_errors(const Tensor<type, 3>& outputs, const Tensor<type, 3>& targets, Tensor<type, 3>& errors) const
{
    errors.device(*thread_pool_device) = outputs - targets;
}


type LossIndex::calculate_eta() const
{
    const Index precision_digits = 6;

    return pow(type(10.0), type(-1.0*precision_digits));
}


/// Calculates a proper step size for computing the derivatives, as a function of the inputs point value.
/// @param x Input value.

type LossIndex::calculate_h(const type& x) const
{
    const type eta = calculate_eta();

    return sqrt(eta)*(type(1) + abs(x));
}


void LossIndexBackPropagationLM::print() const {
    cout << "Loss index back-propagation LM" << endl;
    
    cout << "Errors:" << endl;
    cout << errors << endl;
    
    cout << "Squared errors:" << endl;
    cout << squared_errors << endl;
    
    cout << "Squared errors Jacobian:" << endl;
    cout << squared_errors_jacobian << endl;
    
    cout << "Error:" << endl;
    cout << error << endl;
    
    cout << "Loss:" << endl;
    cout << loss << endl;
    
    cout << "Gradient:" << endl;
    cout << gradient << endl;
    
    cout << "Hessian:" << endl;
    cout << hessian << endl;
}


void LossIndexBackPropagationLM::set(const Index &new_batch_samples_number,
                                     LossIndex *new_loss_index_pointer) 
{
    loss_index_pointer = new_loss_index_pointer;
    
    batch_samples_number = new_batch_samples_number;
    
    NeuralNetwork *neural_network_pointer =
        loss_index_pointer->get_neural_network_pointer();
    
    const Index parameters_number =
        neural_network_pointer->get_parameters_number();
    
    const Index outputs_number = neural_network_pointer->get_outputs_number();
    
    neural_network.set(batch_samples_number, neural_network_pointer);
    
    parameters = neural_network_pointer->get_parameters();
    
    error = type(0);
    
    loss = type(0);
    
    gradient.resize(parameters_number);
    
    regularization_gradient.resize(parameters_number);
    regularization_gradient.setZero();
    
    squared_errors_jacobian.resize(batch_samples_number, parameters_number);
    
    hessian.resize(parameters_number, parameters_number);
    
    regularization_hessian.resize(parameters_number, parameters_number);
    regularization_hessian.setZero();
    
    errors.resize(batch_samples_number, outputs_number);
    
    squared_errors.resize(batch_samples_number);
}


LossIndexBackPropagationLM::LossIndexBackPropagationLM(const Index &new_batch_samples_number, LossIndex *new_loss_index_pointer) 
{
    set(new_batch_samples_number, new_loss_index_pointer);
}


LossIndexBackPropagationLM::LossIndexBackPropagationLM() 
{
}

}
 // namespace opennn
//  // namespace opennnOpenNN: Open Neural // namespace  // namespace opennnopennn Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
