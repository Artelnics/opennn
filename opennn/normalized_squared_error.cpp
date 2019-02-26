/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S                                                */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "normalized_squared_error.h"

#ifdef __OPENNN_CUDA__
#include <cuda_runtime.h>
#include <cublas_v2.h>

void calculateFirstOrderLossCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                                const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                                const double* input_data_h, const size_t input_rows, const size_t input_columns,
                                const double* target_data_h, const size_t target_rows, const size_t target_columns,
                                std::vector<double*> error_gradient_data,
                                double* output_data_h, const size_t output_rows, const size_t output_columns,
                                const std::vector<std::string> layers_activations, const std::string loss_method,
                                const std::vector<double> loss_parameters = vector<double>());

void calculateOutputsCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                          const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                          const double* input_data_h, const size_t input_rows, const size_t input_columns,
                          double* output_data_h, const size_t output_rows, const size_t output_columns,
                          const std::vector<std::string> layers_activations);

#endif

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a normalized squared error term object not associated to any 
/// neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

NormalizedSquaredError::NormalizedSquaredError() : LossIndex()
{
    set_default();
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a normalized squared error term associated to a neural network object but not measured on any data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

NormalizedSquaredError::NormalizedSquaredError(NeuralNetwork* new_neural_network_pointer)
: LossIndex(new_neural_network_pointer)
{
    set_default();
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a normalized squared error term not associated to any 
/// neural network but to be measured on a data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

NormalizedSquaredError::NormalizedSquaredError(DataSet* new_data_set_pointer) 
: LossIndex(new_data_set_pointer)
{
    set_default();
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a normalized squared error term associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

NormalizedSquaredError::NormalizedSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
: LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
    set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a normalized squared error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from a TinyXML document->
/// @param normalized_squared_error_document XML document with the class members. 

NormalizedSquaredError::NormalizedSquaredError(const tinyxml2::XMLDocument& normalized_squared_error_document)
 : LossIndex(normalized_squared_error_document)
{
    set_default();

    from_XML(normalized_squared_error_document);
}


// DESTRUCTOR

/// Destructor.

NormalizedSquaredError::~NormalizedSquaredError()
{
}


// METHODS


// Get methods


double NormalizedSquaredError::get_normalization_coefficient() const
{
    return(normalization_coefficient);
}

// Set methods


void NormalizedSquaredError::set_normalization_coefficient()
{
    // Neural network stuff

//    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

//    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> training_indices = instances.get_training_indices();

    const size_t training_instances_number = training_indices.size();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> targets_indices = variables.get_targets_indices();

    const Vector<double> training_targets_mean = data_set_pointer->calculate_training_targets_mean();

    // Normalized squared error stuff

    double new_normalization_coefficient = 0.0;

    #pragma omp parallel for reduction(+ : new_normalization_coefficient)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[static_cast<size_t>(i)];

       // Target vector

       const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

       // Normalization coefficient

       new_normalization_coefficient += targets.calculate_sum_squared_error(training_targets_mean);
    }

    normalization_coefficient = new_normalization_coefficient;
}


void NormalizedSquaredError::set_normalization_coefficient(const double& new_normalization_coefficient)
{
    normalization_coefficient = new_normalization_coefficient;
}


void NormalizedSquaredError::set_selection_normalization_coefficient()
{
    // Neural network stuff

//    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

//    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> selection_indices = instances.get_selection_indices();

    const size_t selection_instances_number = selection_indices.size();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> targets_indices = variables.get_targets_indices();

    const Vector<double> selection_targets_mean = data_set_pointer->calculate_selection_targets_mean();

    // Normalized squared error stuff

    double new_selection_normalization_coefficient = 0.0;

    #pragma omp parallel for reduction(+ : new_selection_normalization_coefficient)

    for(int i = 0; i < static_cast<int>(selection_instances_number); i++)
    {
        const size_t selection_index = selection_indices[static_cast<size_t>(i)];

       // Target vector

       const Vector<double> targets = data_set_pointer->get_instance(selection_index, targets_indices);

       // Normalization coefficient

       new_selection_normalization_coefficient += targets.calculate_sum_squared_error(selection_targets_mean);
    }

    selection_normalization_coefficient = new_selection_normalization_coefficient;
}


void NormalizedSquaredError::set_selection_normalization_coefficient(const double& new_selection_normalization_coefficient)
{
    selection_normalization_coefficient = new_selection_normalization_coefficient;
}


void NormalizedSquaredError::set_default()
{
    if(has_neural_network() && has_data_set() && data_set_pointer->has_data())
    {
        set_normalization_coefficient();
        set_selection_normalization_coefficient();
    }
    else
    {
        normalization_coefficient = -1;
        selection_normalization_coefficient = -1;
    }

}


/// Returns the normalization coefficient to be used for the loss of the error. 
/// This is measured on the training instances of the data set. 

double NormalizedSquaredError::calculate_normalization_coefficient(const Matrix<double>& targets, const Vector<double>& targets_mean) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

   return(targets.calculate_sum_squared_error(targets_mean));
}


double NormalizedSquaredError::calculate_training_error() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

    const size_t batches_number = training_batches.size();

    double training_error = 0.0;

    #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

        const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        const double batch_error = outputs.calculate_sum_squared_error(targets);

        training_error += batch_error;
    }

    return training_error/normalization_coefficient;
}


double NormalizedSquaredError::calculate_selection_error() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > selection_batches = data_set_pointer->get_instances_pointer()->get_selection_batches(batch_size);

    const size_t batches_number = selection_batches.size();

    double selection_error = 0.0;

    #pragma omp parallel for reduction(+ : selection_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(selection_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(selection_batches[static_cast<unsigned>(i)]);

        const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        const double batch_error = outputs.calculate_sum_squared_error(targets);

        selection_error += batch_error;
    }

    return selection_error/selection_normalization_coefficient;
}


double NormalizedSquaredError::calculate_training_error(const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

    const size_t batches_number = training_batches.size();

    double training_error = 0.0;

    #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

        const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

        const double batch_error = outputs.calculate_sum_squared_error(targets);

        training_error += batch_error;
    }

    return training_error/normalization_coefficient;
}


double NormalizedSquaredError::calculate_batch_error(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices);
    const Matrix<double> targets = data_set_pointer->get_targets(batch_indices);

    const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

    const double batch_error = outputs.calculate_sum_squared_error(targets);

    return batch_error / normalization_coefficient;
}


Vector<double> NormalizedSquaredError::calculate_training_error_gradient() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Neural network

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    // Data set

    const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

    const size_t batches_number = training_batches.size();

    // Loss index

    Vector<double> training_error_gradient(parameters_number, 0.0);

    #pragma omp parallel for

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

        const MultilayerPerceptron::FirstOrderForwardPropagation first_order_forward_propagation
                = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

        const Matrix<double> output_gradient
                = calculate_output_gradient(first_order_forward_propagation.layers_activations[layers_number-1], targets);

        const Vector< Matrix<double> > layers_delta
                = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

        const Vector<double> batch_gradient
                = calculate_error_gradient(inputs, first_order_forward_propagation.layers_activations, layers_delta);

        #pragma omp critical

        training_error_gradient += batch_gradient;
    }

    return training_error_gradient/normalization_coefficient;
}


Vector<double> NormalizedSquaredError::calculate_batch_error_gradient(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Neural network

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices);
    const Matrix<double> targets = data_set_pointer->get_targets(batch_indices);

    const MultilayerPerceptron::FirstOrderForwardPropagation first_order_forward_propagation
            = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

    const Matrix<double> output_gradient
            = calculate_output_gradient(first_order_forward_propagation.layers_activations[layers_number-1], targets);

    const Vector< Matrix<double> > layers_delta
            = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

    const Vector<double> batch_gradient
            = calculate_error_gradient(inputs, first_order_forward_propagation.layers_activations, layers_delta);

    return batch_gradient/normalization_coefficient;
}


double NormalizedSquaredError::calculate_error(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

   // Control sentence

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   if(normalization_coefficient < numeric_limits<double>::min())
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
             << "double calculate_error() const method.\n"
             << "Normalization coefficient is zero.\n"
             << "Unuse constant target variables or choose another error functional. ";

      throw logic_error(buffer.str());
   }

   return outputs.calculate_sum_squared_error(targets)/normalization_coefficient;
}


/// Returns which would be the loss of a multilayer perceptron for an hypothetical vector of parameters.
/// It does not set that vector of parameters to the multilayer perceptron. 
/// @param parameters Vector of potential parameters for the multilayer perceptron associated to the loss index.

double NormalizedSquaredError::calculate_error(const Vector<size_t>& instances_indices, const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif

   #ifdef __OPENNN_DEBUG__ 

   ostringstream buffer;

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   if(size != parameters_number)
   {
      buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
             << "double calculate_error(const Vector<double>&) method.\n"
             << "Size(" << size << ") must be equal to number of parameters(" << parameters_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   if(normalization_coefficient < numeric_limits<double>::min())
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
             << "double calculate_error(const Vector<double>&) const method.\n"
             << "Normalization coefficient is zero.\n"
             << "Unuse constant target variables or choose another error functional. ";

      throw logic_error(buffer.str());
   }

   // Data set stuff

   const Matrix<double> inputs = data_set_pointer->get_inputs(instances_indices);
   const Matrix<double> targets = data_set_pointer->get_targets(instances_indices);

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

   return calculate_error(outputs, targets);
}


Matrix<double> NormalizedSquaredError::calculate_output_gradient(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    return (outputs-targets)*2.0/normalization_coefficient;
}


LossIndex::FirstOrderLoss NormalizedSquaredError::calculate_first_order_loss() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    // Data set

    const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

    const size_t batches_number = training_batches.size();

    FirstOrderLoss first_order_loss(parameters_number);

    #pragma omp parallel for

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

        const MultilayerPerceptron::FirstOrderForwardPropagation first_order_forward_propagation
                = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

        const Vector<double> error_terms
                = calculate_error_terms(first_order_forward_propagation.layers_activations[layers_number-1], targets);

        const Matrix<double> output_gradient = (first_order_forward_propagation.layers_activations[layers_number-1] - targets)/error_terms;

        const Vector< Matrix<double> > layers_delta
                = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

        const Matrix<double> error_terms_Jacobian
                = calculate_error_terms_Jacobian(inputs, first_order_forward_propagation.layers_activations, layers_delta);

        const Matrix<double> error_terms_Jacobian_transpose = error_terms_Jacobian.calculate_transpose();

        const double loss = error_terms.dot(error_terms);

        const Vector<double> gradient = error_terms_Jacobian_transpose.dot(error_terms);

        #pragma omp critical
//        {
            first_order_loss.loss += loss;
            first_order_loss.gradient += gradient;
//         }
    }

    first_order_loss.loss /= normalization_coefficient;
    first_order_loss.gradient *= (2.0/normalization_coefficient);

    // Regularization

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

    return first_order_loss;
}


LossIndex::FirstOrderLoss NormalizedSquaredError::calculate_batch_first_order_loss(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    FirstOrderLoss first_order_loss(parameters_number);

    const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices);
    const Matrix<double> targets = data_set_pointer->get_targets(batch_indices);

    const MultilayerPerceptron::FirstOrderForwardPropagation first_order_forward_propagation
            = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

    const Matrix<double> output_gradient
            = calculate_output_gradient(first_order_forward_propagation.layers_activations[layers_number-1], targets);

    const Vector< Matrix<double> > layers_delta
            = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

    const Vector<double> batch_gradient
            = calculate_error_gradient(inputs, first_order_forward_propagation.layers_activations, layers_delta);

    const double batch_error = first_order_forward_propagation.layers_activations[layers_number-1].calculate_sum_squared_error(targets);

    first_order_loss.loss = batch_error / normalization_coefficient;
    first_order_loss.gradient += batch_gradient/normalization_coefficient;

    // Regularization

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

    return first_order_loss;
}

LossIndex::FirstOrderLoss NormalizedSquaredError::calculate_batch_first_order_loss_cuda(const Vector<size_t>& batch_indices,
                                                                                        const MultilayerPerceptron::Pointers& pointers) const
{
    FirstOrderLoss first_order_loss;

#ifdef __OPENNN_CUDA__

    const size_t layers_number = pointers.architecture.size() - 1;

    Matrix<double> inputs_matrix = data_set_pointer->get_inputs(batch_indices);
    double* input_data = inputs_matrix.data();
    const size_t input_rows = inputs_matrix.get_rows_number();
    const size_t input_columns = inputs_matrix.get_columns_number();

    Matrix<double> targets_matrix = data_set_pointer->get_targets(batch_indices);
    double* target_data = targets_matrix.data();
    const size_t target_rows = targets_matrix.get_rows_number();
    const size_t target_columns = targets_matrix.get_columns_number();

    Matrix<double> outputs(inputs_matrix.get_rows_number(), pointers.architecture[layers_number]);
    double* output_data = outputs.data();
    const size_t output_rows = inputs_matrix.get_rows_number();
    const size_t output_columns = pointers.architecture[layers_number];

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    size_t parameters_number = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = pointers.architecture[i];
        weights_columns_numbers[i] = pointers.architecture[i+1];

        bias_rows_numbers[i] = pointers.architecture[i+1];

        parameters_number += pointers.architecture[i]*pointers.architecture[i+1] + pointers.architecture[i+1];
    }

    first_order_loss.gradient.set(parameters_number);
    vector<double*> error_gradient_data(2*layers_number);

    size_t index = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        error_gradient_data[2*i] = first_order_loss.gradient.data() + index;
        index += weights_rows_numbers[i]*weights_columns_numbers[i];

        error_gradient_data[2*i+1] = first_order_loss.gradient.data() + index;
        index += bias_rows_numbers[i];
    }

    vector<double> loss_parameters(1, normalization_coefficient);

    string loss_method = write_error_term_type();

    calculateFirstOrderLossCUDA(pointers.weights_pointers.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                               pointers.biases_pointers.to_std_vector(), bias_rows_numbers,
                               input_data, input_rows, input_columns,
                               target_data, target_rows, target_columns,
                               error_gradient_data,
                               output_data, output_rows, output_columns,
                               pointers.layer_activations.to_std_vector(), loss_method, loss_parameters);

    first_order_loss.loss = outputs.calculate_sum_squared_error(targets_matrix) / normalization_coefficient;

    // Regularization

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

#endif

    return first_order_loss;
}

LossIndex::FirstOrderLoss NormalizedSquaredError::calculate_batch_first_order_loss_cuda(const Vector<size_t>&,
                                                                                        const MultilayerPerceptron::Pointers&, const Vector<double*>&) const
{
    FirstOrderLoss first_order_loss;

    return first_order_loss;
}

/// Returns loss vector of the error terms function for the normalized squared error.
/// It uses the error back-propagation method.

Vector<double> NormalizedSquaredError::calculate_error_terms(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

   // Control sentence

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   return outputs.calculate_error_rows(targets);
}


Vector<double> NormalizedSquaredError::calculate_error_terms(const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const Matrix<double> inputs = data_set_pointer->get_training_inputs();

    const Matrix<double> targets = data_set_pointer->get_training_targets();

    const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

    return outputs.calculate_error_rows(targets)/normalization_coefficient;
}

/// Returns the squared errors of the training instances. 

Vector<double> NormalizedSquaredError::calculate_squared_errors() const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   check();

   #endif
/*
   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   // Data set stuff

   const Instances& instances = data_set_pointer->get_instances();

   const Vector<size_t> training_indices = instances.get_training_indices();

   const size_t training_instances_number = training_indices.size();

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   // Calculate

   Vector<double> squared_errors(training_instances_number);

   // Main loop

   #pragma omp parallel for

   for(int i = 0; i < static_cast<int>(training_instances_number); i++)
   {
       const size_t training_index = training_indices[static_cast<size_t>(i)];

       // Input vector

      const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Error

      squared_errors[static_cast<size_t>(i)] = outputs.calculate_sum_squared_error(targets);
   }

   return(squared_errors);
*/
    return Vector<double>();
}


/// Returns a vector with the indices of the instances which have the maximum error.
/// @param maximal_errors_number Number of instances required.

Vector<size_t> NormalizedSquaredError::calculate_maximal_errors(const size_t& maximal_errors_number) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    check();

    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.get_training_instances_number();

    if(maximal_errors_number > training_instances_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedquaredError class.\n"
               << "Vector<size_t> calculate_maximal_errors() const method.\n"
               << "Number of maximal errors(" << maximal_errors_number << ") must be equal or less than number of training instances(" << training_instances_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    return(calculate_squared_errors().calculate_maximal_indices(maximal_errors_number));
}



LossIndex::SecondOrderLoss NormalizedSquaredError::calculate_terms_second_order_loss() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    // Data set

    const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

    const size_t batches_number = training_batches.size();

    SecondOrderLoss terms_second_order_loss(parameters_number);

    #pragma omp parallel for

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

        const MultilayerPerceptron::FirstOrderForwardPropagation first_order_forward_propagation
                = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

        const Vector<double> error_terms
                = calculate_error_terms(first_order_forward_propagation.layers_activations[layers_number-1], targets);

        const Matrix<double> output_gradient = (first_order_forward_propagation.layers_activations[layers_number-1] - targets)/error_terms;

        const Vector< Matrix<double> > layers_delta
                = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

        const Matrix<double> error_terms_Jacobian
                = calculate_error_terms_Jacobian(inputs, first_order_forward_propagation.layers_activations, layers_delta);

        const Matrix<double> error_terms_Jacobian_transpose = error_terms_Jacobian.calculate_transpose();

        const double loss = error_terms.dot(error_terms);

        const Vector<double> gradient = error_terms_Jacobian_transpose.dot(error_terms);

        Matrix<double> Hessian_approximation;
        Hessian_approximation.dot(error_terms_Jacobian_transpose, error_terms_Jacobian);

        #pragma omp critical
//        {
            terms_second_order_loss.loss += loss;
            terms_second_order_loss.gradient += gradient;
            terms_second_order_loss.Hessian_approximation += Hessian_approximation;
//         }
    }

//    const Matrix<double> regularization_Hessian = loss_index_pointer->calculate_regularization_Hessian();

    terms_second_order_loss.loss /= normalization_coefficient;
    terms_second_order_loss.gradient *= (2.0/normalization_coefficient);
    terms_second_order_loss.Hessian_approximation *= (2.0/normalization_coefficient);

    if(regularization_method == RegularizationMethod::None)
    {
        terms_second_order_loss.loss += calculate_regularization();
        terms_second_order_loss.gradient += calculate_regularization_gradient();
        terms_second_order_loss.Hessian_approximation += calculate_regularization_Hessian();
    }

    return terms_second_order_loss;
}


/// Returns a string with the name of the normalized squared error loss type, "NORMALIZED_SQUARED_ERROR".

string NormalizedSquaredError::write_error_term_type() const
{
   return("NORMALIZED_SQUARED_ERROR");
}


/// Serializes the normalized squared error object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this element. 

tinyxml2::XMLDocument* NormalizedSquaredError::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Normalized squared error

   tinyxml2::XMLElement* normalized_squared_error_element = document->NewElement("NormalizedSquaredError");

   document->InsertFirstChild(normalized_squared_error_element);

   // Display
//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      normalized_squared_error_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

   return(document);
}


void NormalizedSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("Error");

    file_stream.PushAttribute("Type", "NORMALIZED_SQUARED_ERROR");

    file_stream.CloseElement();

    // Regularization

    write_regularization_XML(file_stream);
}


/// Loads a root mean squared error object from a XML document. 
/// @param document Pointer to a TinyXML document with the object data.

void NormalizedSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
     const tinyxml2::XMLElement* root_element = document.FirstChildElement("NormalizedSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NormalizedSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Normalized squared element is nullptr.\n";

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
