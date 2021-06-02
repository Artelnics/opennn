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


/// Destructor.

LossIndex::~LossIndex()
{
    delete non_blocking_thread_pool;
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


void LossIndex::set_threads_number(const int& new_threads_number)
{
    if(non_blocking_thread_pool != nullptr) delete this->non_blocking_thread_pool;
    if(thread_pool_device != nullptr) delete this->thread_pool_device;

    non_blocking_thread_pool = new NonBlockingThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, new_threads_number);
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
    delete non_blocking_thread_pool;
    delete thread_pool_device;

    const int n = omp_get_max_threads();

    non_blocking_thread_pool = new NonBlockingThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    regularization_method = L2;
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

    // Data set

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: LossIndex class.\n"
               << "void check() const method.\n"
               << "Pointer to data set is nullptr.\n";

        throw logic_error(buffer.str());
    }
}


void LossIndex::calculate_errors(const DataSetBatch& batch,
                                 const NeuralNetworkForwardPropagation& forward_propagation,
                                 LossIndexBackPropagation& back_propagation) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    switch(forward_propagation.layers(trainable_layers_number-1)->layer_pointer->get_type())
    {
    case Layer::Perceptron:
    {
        back_propagation.errors.device(*thread_pool_device) =
                static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.layers(trainable_layers_number-1))->activations -
                batch.targets_2d;
    }
        break;

    case Layer::Probabilistic:
    {
        back_propagation.errors.device(*thread_pool_device) =
                static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation.layers(trainable_layers_number-1))->activations -
                batch.targets_2d;
    }
        break;

    case Layer::Recurrent:
    {
        back_propagation.errors.device(*thread_pool_device) =
                static_cast<RecurrentLayerForwardPropagation*>(forward_propagation.layers(trainable_layers_number-1))->activations -
                batch.targets_2d;
    }
        break;

    case Layer::LongShortTermMemory:
    {
        back_propagation.errors.device(*thread_pool_device) =
                static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation.layers(trainable_layers_number-1))->activations -
                batch.targets_2d;
    }
        break;

    default: break;
    }
}


void LossIndex::calculate_errors(const DataSetBatch &batch,
                                 const NeuralNetworkForwardPropagation & neural_network_forward_propagation,
                                 LossIndexBackPropagationLM & loss_index_back_propagation) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    switch(neural_network_forward_propagation.layers(trainable_layers_number-1)->layer_pointer->get_type())
    {
    case Layer::Perceptron:
    {
        loss_index_back_propagation.errors.device(*thread_pool_device) =
                static_cast<PerceptronLayerForwardPropagation*>(neural_network_forward_propagation.layers(trainable_layers_number-1))->activations -
                batch.targets_2d;
    }
        break;

    case Layer::Probabilistic:
    {
        loss_index_back_propagation.errors.device(*thread_pool_device) =
                static_cast<ProbabilisticLayerForwardPropagation*>(neural_network_forward_propagation.layers(trainable_layers_number-1))->activations -
                batch.targets_2d;
    }
        break;

    case Layer::Recurrent:
    {
        loss_index_back_propagation.errors.device(*thread_pool_device) =
                static_cast<RecurrentLayerForwardPropagation*>(neural_network_forward_propagation.layers(trainable_layers_number-1))->activations -
                batch.targets_2d;
    }
        break;

    case Layer::LongShortTermMemory:
    {
        loss_index_back_propagation.errors.device(*thread_pool_device) =
                static_cast<LongShortTermMemoryLayerForwardPropagation*>(neural_network_forward_propagation.layers(trainable_layers_number-1))->activations -
                batch.targets_2d;
    }
        break;

    default: break;
    }
}


void LossIndex::calculate_squared_errors(const DataSetBatch& batch,
                                         const NeuralNetworkForwardPropagation& forward_propagation,
                                         LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    LayerForwardPropagation* output_layer_forward_propagation = forward_propagation.layers(trainable_layers_number-1);

    const Layer* output_layer_pointer = output_layer_forward_propagation->layer_pointer;

    const Eigen::array<int, 1> rows_sum = {Eigen::array<int, 1>({1})};

    const Tensor<type, 2>& targets = batch.targets_2d;

    switch(output_layer_pointer->get_type())
    {
    case Layer::Perceptron:
    {
        PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
                = static_cast<PerceptronLayerForwardPropagation*>(output_layer_forward_propagation);

        const Tensor<type, 2>& outputs = perceptron_layer_forward_propagation->activations;

        loss_index_back_propagation_lm.squared_errors.device(*thread_pool_device) = ((outputs - targets).square().sum(rows_sum)).sqrt();
    }
        break;

    case Layer::Probabilistic:
    {
        ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
                = static_cast<ProbabilisticLayerForwardPropagation*>(output_layer_forward_propagation);

        const Tensor<type, 2>& outputs = probabilistic_layer_forward_propagation->activations;

        loss_index_back_propagation_lm.squared_errors.device(*thread_pool_device) = ((outputs - targets).square().sum(rows_sum)).sqrt();
    }
        break;

    case Layer::Recurrent:
    {
        RecurrentLayerForwardPropagation* recurrent_layer_forward_propagation
                = static_cast<RecurrentLayerForwardPropagation*>(output_layer_forward_propagation);

        const Tensor<type, 2>& outputs = recurrent_layer_forward_propagation->activations;

        loss_index_back_propagation_lm.squared_errors.device(*thread_pool_device) = ((outputs - targets).square().sum(rows_sum)).sqrt();
    }
        break;

    case Layer::LongShortTermMemory:
    {
        LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation
                = static_cast<LongShortTermMemoryLayerForwardPropagation*>(output_layer_forward_propagation);

        const Tensor<type, 2>& outputs = long_short_term_memory_layer_forward_propagation->activations;

        loss_index_back_propagation_lm.squared_errors.device(*thread_pool_device) = ((outputs - targets).square().sum(rows_sum)).sqrt();
    }
        break;

    default: break;
    }
}


void LossIndex::back_propagate(const DataSetBatch& batch,
                               NeuralNetworkForwardPropagation& forward_propagation,
                               LossIndexBackPropagation& back_propagation) const
{
    // Loss index

    calculate_errors(batch, forward_propagation, back_propagation);

    calculate_error(batch, forward_propagation, back_propagation);

    calculate_layers_delta(batch, forward_propagation, back_propagation);

    calculate_error_gradient(batch, forward_propagation, back_propagation);

    // Loss

    back_propagation.loss = back_propagation.error;

    // Regularization

    if(regularization_method != RegularizationMethod::NoRegularization)
    {
        const type regularization = calculate_regularization(back_propagation.parameters);

        back_propagation.loss += regularization_weight*regularization;

        calculate_regularization_gradient(back_propagation.parameters, back_propagation.regularization_gradient);

        back_propagation.gradient.device(*thread_pool_device) += regularization_weight*back_propagation.regularization_gradient;
    }
}


/// This method calculates the second order loss.
/// It is used for optimization of parameters during training.
/// Returns a second order terms loss structure, which contains the values and the Hessian of the error terms function.
/// @todo Update method.

void LossIndex::back_propagate(const DataSetBatch& batch,
                               NeuralNetworkForwardPropagation& forward_propagation,
                               LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
    // First Order

    calculate_squared_errors(batch, forward_propagation, loss_index_back_propagation_lm);

    calculate_errors(batch, forward_propagation, loss_index_back_propagation_lm);

    calculate_error(batch, forward_propagation, loss_index_back_propagation_lm);

    calculate_layers_delta(batch, forward_propagation, loss_index_back_propagation_lm);

    // Second Order

    calculate_squared_errors_jacobian(batch, forward_propagation, loss_index_back_propagation_lm);

    calculate_gradient(batch, loss_index_back_propagation_lm);

    calculate_hessian_approximation(batch, loss_index_back_propagation_lm);

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

void LossIndex::calculate_squared_errors_jacobian(const DataSetBatch& batch,
                                                  NeuralNetworkForwardPropagation& forward_propagation,
                                                  LossIndexBackPropagationLM& loss_index_back_propagation_lm) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    loss_index_back_propagation_lm.squared_errors_jacobian.setZero();

    const Index batch_samples_number = batch.get_samples_number();
    Index mem_index = 0;

    Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();
    const Tensor<Index, 1> trainable_layers_parameters_number = neural_network_pointer->get_trainable_layers_parameters_numbers();

    // Layer 0

    if(trainable_layers_pointers(0)->get_type() != Layer::Perceptron && trainable_layers_pointers(0)->get_type() != Layer::Probabilistic)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LossIndex class.\n"
               << "void calculate_squared_errors_jacobian(const DataSetBatch&, NeuralNetworkForwardPropagation&, LossIndexBackPropagationLM&) const method "
               << "Levenberg - Marquardt algorithm can only be used with Perceptron and Probabilistic layers.\n";

        throw logic_error(buffer.str());
    }
    else
    {
        trainable_layers_pointers(0)->calculate_squared_errors_Jacobian(batch.inputs_2d,
                                                                        forward_propagation.layers(0),
                                                                        loss_index_back_propagation_lm.neural_network.layers(0));

        trainable_layers_pointers(0)->insert_squared_errors_Jacobian(loss_index_back_propagation_lm.neural_network.layers(0),
                                                                     mem_index,
                                                                     loss_index_back_propagation_lm.squared_errors_jacobian);

        mem_index += trainable_layers_parameters_number(0)*batch_samples_number;
    }

    // Rest of the layers

    for(Index i = 1; i < trainable_layers_number; i++)
    {
        switch (forward_propagation.layers(i-1)->layer_pointer->get_type())
        {
        case Layer::Perceptron:
        {
            PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
                    = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.layers(i-1));

            trainable_layers_pointers(i)->calculate_squared_errors_Jacobian(perceptron_layer_forward_propagation->activations,
                                                                            forward_propagation.layers(i),
                                                                            loss_index_back_propagation_lm.neural_network.layers(i));

            trainable_layers_pointers(i)->insert_squared_errors_Jacobian(loss_index_back_propagation_lm.neural_network.layers(i),
                                                                         mem_index,
                                                                         loss_index_back_propagation_lm.squared_errors_jacobian);

            mem_index += trainable_layers_parameters_number(i)*batch_samples_number;
        }
            break;

        case Layer::Probabilistic:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: LossIndex class.\n"
                   << "void calculate_squared_errors_jacobian(const DataSetBatch&, NeuralNetworkForwardPropagation&, LossIndexBackPropagationLM&) const method "
                   << "Probabilistic layer can only occupy the last position in the neural network. Please, check network structure.\n";

            throw logic_error(buffer.str());
        }

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: LossIndex class.\n"
                   << "void calculate_squared_errors_jacobian(const DataSetBatch&, NeuralNetworkForwardPropagation&, LossIndexBackPropagationLM&) const method "
                   << "Levenberg - Marquardt algorithm can only be used with Perceptron and Probabilistic layers.\n";

            throw logic_error(buffer.str());
        }
        }
    }
}


void LossIndex::calculate_gradient(const DataSetBatch& batch,
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
    case NoRegularization:
        return "NO_REGULARIZATION";

    case L1:
        return "L1_NORM";

    case L2:
        return "L2_NORM";
    }

    return string();
}


/// It calculates the regularization term using through the use of parameters.
/// Returns the regularization evaluation, according to the respective regularization type used in the
/// loss index expression.
/// @param parameters vector with the parameters to get the regularization term.

type LossIndex::calculate_regularization(const Tensor<type, 1>& parameters) const
{
    switch(regularization_method)
    {
        case NoRegularization: return 0.0;

        case L1: return l1_norm(thread_pool_device, parameters);

        case L2: return l2_norm(thread_pool_device, parameters);
    }

    return 0.0;
}


/// It calculate the regularization term using the gradient method.
/// Returns the gradient of the regularization, according to the regularization type.
/// That gradient is the vector of partial derivatives of the regularization with respect to the parameters.
/// The size is thus the number of parameters
/// @param parameters vector with the parameters to get the regularization term.

void LossIndex::calculate_regularization_gradient(const Tensor<type, 1>& parameters, Tensor<type, 1>& regularization_gradient) const
{
    switch(regularization_method)
    {
    case L1: l1_norm_gradient(thread_pool_device, parameters, regularization_gradient); return;

    case L2: l2_norm_gradient(thread_pool_device, parameters, regularization_gradient); return;

    default: return;
    }
}


/// It calculate the regularization term using the <i>Hessian</i>.
/// Returns the Hessian of the regularization, according to the regularization type.
/// That Hessian is the matrix of second partial derivatives of the regularization with respect to the parameters.
/// That matrix is symmetric, with size the number of parameters.
/// @param parameters vector with the parameters to get the regularization term.

void LossIndex::calculate_regularization_hessian(const Tensor<type, 1>& parameters, Tensor<type, 2>& regularization_hessian) const
{
    switch(regularization_method)
    {
    case L1: l1_norm_hessian(thread_pool_device, parameters, regularization_hessian); return;

    case L2: l2_norm_hessian(thread_pool_device, parameters, regularization_hessian); return;

    default: return;
    }
}


void LossIndex::calculate_layers_delta(const DataSetBatch& batch,
                                       NeuralNetworkForwardPropagation& forward_propagation,
                                       LossIndexBackPropagation& back_propagation) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    if(trainable_layers_number == 0) return;

    const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

    // Output layer

    calculate_output_delta(batch,
                           forward_propagation,
                           back_propagation);

    // Hidden layers

    for(Index i = static_cast<Index>(trainable_layers_number)-2; i >= 0; i--)
    {
        trainable_layers_pointers(i)
                ->calculate_hidden_delta(forward_propagation.layers(i+1),
                                         back_propagation.neural_network.layers(i+1),
                                         back_propagation.neural_network.layers(i));
    }
}


void LossIndex::calculate_layers_delta(const DataSetBatch& batch,
                                       NeuralNetworkForwardPropagation& forward_propagation,
                                       LossIndexBackPropagationLM& back_propagation) const
{
    const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

    if(trainable_layers_number == 0) return;

    const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

    // Output layer

    calculate_output_delta(batch,
                           forward_propagation,
                           back_propagation);

    // Hidden layers

    for(Index i = static_cast<Index>(trainable_layers_number)-2; i >= 0; i--)
    {
        trainable_layers_pointers(i)
                ->calculate_hidden_delta(forward_propagation.layers(i+1),
                                         back_propagation.neural_network.layers(i+1),
                                         back_propagation.neural_network.layers(i));
    }
}


void LossIndex::calculate_error_gradient(const DataSetBatch& batch,
                                         const NeuralNetworkForwardPropagation& forward_propagation,
                                         LossIndexBackPropagation& back_propagation) const
{
    #ifdef OPENNN_DEBUG

    check();

    #endif

    const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

    const Index trainable_layers_number = trainable_layers_pointers.size();

    const Tensor<Index, 1> trainable_layers_parameters_number
            = neural_network_pointer->get_trainable_layers_parameters_numbers();

    if(trainable_layers_pointers(0)->get_type() == Layer::Convolutional)
    {
        trainable_layers_pointers(0)->calculate_error_gradient(batch.inputs_4d,
                                                               forward_propagation.layers(0),
                                                               back_propagation.neural_network.layers(0));
    }
    else
    {
        trainable_layers_pointers(0)->calculate_error_gradient(batch.inputs_2d,
                                                               forward_propagation.layers(0),
                                                               back_propagation.neural_network.layers(0));
    }

    Index index = 0;

    trainable_layers_pointers(0)->insert_gradient(back_propagation.neural_network.layers(0),
                                                  index,
                                                  back_propagation.gradient);

    index += trainable_layers_parameters_number(0);

    for(Index i = 1; i < trainable_layers_number; i++)
    {
        switch(forward_propagation.layers(i-1)->layer_pointer->get_type())
        {
        case Layer::Perceptron:
        {
            PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
                    = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.layers(i-1));

            trainable_layers_pointers(i)->
                    calculate_error_gradient(perceptron_layer_forward_propagation->activations,
                                             forward_propagation.layers(i),
                                             back_propagation.neural_network.layers(i));
        }
            break;

        case Layer::Probabilistic:
        {
            ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
                    = static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation.layers(i-1));

            trainable_layers_pointers(i)->
                    calculate_error_gradient(probabilistic_layer_forward_propagation->activations,
                                             forward_propagation.layers(i),
                                             back_propagation.neural_network.layers(i));
        }
            break;

        case Layer::Recurrent:
        {
            RecurrentLayerForwardPropagation* recurrent_layer_forward_propagation
                    = static_cast<RecurrentLayerForwardPropagation*>(forward_propagation.layers(i-1));

            trainable_layers_pointers(i)->
                    calculate_error_gradient(recurrent_layer_forward_propagation->activations,
                                             forward_propagation.layers(i),
                                             back_propagation.neural_network.layers(i));
        }
            break;

        case Layer::LongShortTermMemory:
        {
            LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation
                    = static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation.layers(i-1));

            trainable_layers_pointers(i)->
                    calculate_error_gradient(long_short_term_memory_layer_forward_propagation->activations,
                                             forward_propagation.layers(i),
                                             back_propagation.neural_network.layers(i));
        }
            break;

        case Layer::Convolutional:
        {
            // @todo

            //trainable_layers_pointers(i)->
            //        calculate_error_gradient(static_cast<ConvolutionalLayer::ConvolutionalLayerForwardPropagation*>(forward_propagation.layers(i-1))->activations,
            //                                 forward_propagation.layers(i),
            //                                 back_propagation.neural_network.layers(i));
        }
            break;

        case Layer::Pooling:
        {
            // @todo
        }
            break;

        default: break;

        }

        trainable_layers_pointers(i)->insert_gradient(back_propagation.neural_network.layers(i),
                                                      index,
                                                      back_propagation.gradient);

        index += trainable_layers_parameters_number(i);
    }
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

    const tinyxml2::XMLElement* element = root_element->FirstChildElement("RegularizationWeight");

    if(element)
    {
        const type new_regularization_weight = static_cast<type>(atof(element->GetText()));

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

    switch(regularization_method)
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

    file_stream.OpenElement("RegularizationWeight");

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


/// Destructor.

LossIndexBackPropagation::~LossIndexBackPropagation()
{
}


Tensor<type, 1> LossIndex::calculate_gradient_numerical_differentiation()
{
    const Index samples_number = data_set_pointer->get_training_samples_number();

    DataSetBatch batch(samples_number, data_set_pointer);

    const Tensor<Index, 1> samples_indices = data_set_pointer->get_training_samples_indices();
    const Tensor<Index, 1> input_indices = data_set_pointer->get_input_variables_indices();
    const Tensor<Index, 1> target_indices = data_set_pointer->get_target_variables_indices();

    batch.fill(samples_indices, input_indices, target_indices);

    NeuralNetworkForwardPropagation forward_propagation(samples_number, neural_network_pointer);

    LossIndexBackPropagation back_propagation(samples_number, this);

    const Tensor<type, 1> parameters = neural_network_pointer->get_parameters();

    const Index parameters_number = parameters.size();

    type h;
    Tensor<type, 1> parameters_forward(parameters);
    Tensor<type, 1> parameters_backward(parameters);

    type error_forward;
    type error_backward;

    Tensor<type, 1> numerical_gradient(parameters_number);

    for(Index i = 0; i < parameters_number; i++)
    {
       h = calculate_h(parameters(i));

       parameters_forward(i) += h;

       neural_network_pointer->forward_propagate(batch, parameters_forward, forward_propagation);

       calculate_errors(batch, forward_propagation, back_propagation);

       calculate_error(batch, forward_propagation, back_propagation);

       error_forward = back_propagation.error;

       parameters_forward(i) -= h;

       parameters_backward(i) -= h;

       neural_network_pointer->forward_propagate(batch, parameters_backward, forward_propagation);
       calculate_errors(batch, forward_propagation, back_propagation);
       calculate_error(batch, forward_propagation, back_propagation);
       error_backward = back_propagation.error;

       parameters_backward(i) += h;

       numerical_gradient(i) = (error_forward - error_backward)/(2*h);
    }

    return numerical_gradient;
}


Tensor<type, 2> LossIndex::calculate_Jacobian_numerical_differentiation()
{
    LossIndexBackPropagationLM back_propagation_lm;

    const Index samples_number = data_set_pointer->get_training_samples_number();

    DataSetBatch batch(samples_number, data_set_pointer);

    const Tensor<Index, 1> samples_indices = data_set_pointer->get_training_samples_indices();

    const Tensor<Index, 1> input_indices = data_set_pointer->get_input_variables_indices();
    const Tensor<Index, 1> target_indices = data_set_pointer->get_target_variables_indices();

    batch.fill(samples_indices, input_indices, target_indices);

    NeuralNetworkForwardPropagation forward_propagation(samples_number, neural_network_pointer);

    LossIndexBackPropagation back_propagation(samples_number, this);

    Tensor<type, 1> parameters = neural_network_pointer->get_parameters();

    const Index parameters_number = parameters.size();

    back_propagation_lm.set(samples_number, this);

    neural_network_pointer->forward_propagate(batch, parameters, forward_propagation);
    calculate_squared_errors(batch, forward_propagation, back_propagation_lm);

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
        neural_network_pointer->forward_propagate(batch, parameters_backward, forward_propagation);
        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
        error_terms_backward = back_propagation_lm.squared_errors;
        parameters_backward(j) += h;

        parameters_forward(j) += h;
        neural_network_pointer->forward_propagate(batch, parameters_forward, forward_propagation);
        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);
        error_terms_forward = back_propagation_lm.squared_errors;
        parameters_forward(j) -= h;

        for(Index i = 0; i < samples_number; i++)
        {
            jacobian(i,j) = (error_terms_forward(i) - error_terms_backward(i))/(static_cast<type>(2.0)*h);
        }
    }

    return jacobian;
}


type LossIndex::calculate_eta() const
{
    const Index precision_digits = 6;

    return pow(static_cast<type>(10.0), static_cast<type>(-1.0*precision_digits));
}


/// Calculates a proper step size for computing the derivatives, as a function of the inputs point value.
/// @param x Input value.

type LossIndex::calculate_h(const type& x) const
{
    const type eta = calculate_eta();

    return sqrt(eta)*(static_cast<type>(1.0) + abs(x));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
