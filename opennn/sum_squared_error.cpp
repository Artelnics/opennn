/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S U M   S Q U A R E D   E R R O R   C L A S S                                                              */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "sum_squared_error.h"

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

/// Default constructor. 
/// It creates a sum squared error term not associated to any neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

SumSquaredError::SumSquaredError() : LossIndex()
{
}


/// Neural network constructor. 
/// It creates a sum squared error term associated to a neural network but not measured on any data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

SumSquaredError::SumSquaredError(NeuralNetwork* new_neural_network_pointer) 
: LossIndex(new_neural_network_pointer)
{
}


/// Data set constructor. 
/// It creates a sum squared error not associated to any neural network but to be measured on a data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

SumSquaredError::SumSquaredError(DataSet* new_data_set_pointer)
: LossIndex(new_data_set_pointer)
{
}


/// Neural network and data set constructor. 
/// It creates a sum squared error associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

SumSquaredError::SumSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
 : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


/// XML constructor. 
/// It creates a sum squared error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from a TinyXML document.
/// @param sum_squared_error_document XML document with the class members.

SumSquaredError::SumSquaredError(const tinyxml2::XMLDocument& sum_squared_error_document)
 : LossIndex(sum_squared_error_document)
{
    from_XML(sum_squared_error_document);
}


/// Copy constructor. 
/// It creates a sum squared error not associated to any neural network and not measured on any data set.
/// It also sets all the rest of class members from another sum squared error object.
/// @param new_sum_squared_error Object to be copied. 

SumSquaredError::SumSquaredError(const SumSquaredError& new_sum_squared_error)
 : LossIndex(new_sum_squared_error)
{

}


// DESTRUCTOR

/// Destructor.

SumSquaredError::~SumSquaredError() 
{
}


// METHODS

double SumSquaredError::calculate_batch_error(const Vector<size_t>& batch_indices) const
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

    const double training_error = outputs.calculate_sum_squared_error(targets);

    return training_error;
}

double SumSquaredError::calculate_training_error() const
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

    return training_error;
}

double SumSquaredError::calculate_selection_error() const
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

    return selection_error;
}


double SumSquaredError::calculate_training_error(const Vector<double>& parameters) const
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

    return training_error;
}


Vector<double> SumSquaredError::calculate_training_error_gradient() const
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

    return training_error_gradient;
}

double SumSquaredError::calculate_batch_error_cuda(const Vector<size_t>& batch_indices, const MultilayerPerceptron::Pointers& pointers) const
{
    double batch_error = 0.0;

#ifdef __OPENNN_CUDA__

    const size_t layers_number = pointers.architecture.size() - 1;

    const Matrix<double> inputs_matrix = data_set_pointer->get_inputs(batch_indices);
    const double* input_data = inputs_matrix.data();
    const size_t input_rows = inputs_matrix.get_rows_number();
    const size_t input_columns = inputs_matrix.get_columns_number();

    Matrix<double> outputs(inputs_matrix.get_rows_number(), pointers.architecture[layers_number]);
    double* output_data = outputs.data();
    const size_t output_rows = inputs_matrix.get_rows_number();
    const size_t output_columns = pointers.architecture[layers_number];

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = pointers.architecture[i];
        weights_columns_numbers[i] = pointers.architecture[i+1];

        bias_rows_numbers[i] = pointers.architecture[i+1];
    }

    calculateOutputsCUDA(pointers.weights_pointers.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                         pointers.biases_pointers.to_std_vector(), bias_rows_numbers,
                         input_data, input_rows, input_columns,
                         output_data, output_rows, output_columns,
                         pointers.layer_activations.to_std_vector());

    const Matrix<double> targets_matrix = data_set_pointer->get_targets(batch_indices);

    batch_error = calculate_error(outputs, targets_matrix);

#endif

    return batch_error;
}

Vector<double> SumSquaredError::calculate_batch_error_gradient(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Neural network

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    // Loss index

    const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices);
    const Matrix<double> targets = data_set_pointer->get_targets(batch_indices);

    const MultilayerPerceptron::FirstOrderForwardPropagation first_order_forward_propagation
            = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

    const Matrix<double> output_gradient = calculate_output_gradient(first_order_forward_propagation.layers_activations[layers_number-1], targets);

    const Vector< Matrix<double> > layers_delta = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

    const Vector<double> batch_error_gradient = calculate_error_gradient(inputs, first_order_forward_propagation.layers_activations, layers_delta);

    return batch_error_gradient;
}


Vector<double> SumSquaredError::calculate_batch_error_gradient_cuda(const Vector<size_t>& batch_indices, const MultilayerPerceptron::Pointers& pointers) const
{
    Vector<double> error_gradient;

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

    error_gradient.set(parameters_number);
    vector<double*> error_gradient_data(2*layers_number);

    size_t index = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        error_gradient_data[2*i] = error_gradient.data() + index;
        index += weights_rows_numbers[i]*weights_columns_numbers[i];

        error_gradient_data[2*i+1] = error_gradient.data() + index;
        index += bias_rows_numbers[i];
    }

    vector<double> loss_parameters;

    string loss_method = write_error_term_type();

    calculateFirstOrderLossCUDA(pointers.weights_pointers.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                               pointers.biases_pointers.to_std_vector(), bias_rows_numbers,
                               input_data, input_rows, input_columns,
                               target_data, target_rows, target_columns,
                               error_gradient_data,
                               output_data, output_rows, output_columns,
                               pointers.layer_activations.to_std_vector(), loss_method, loss_parameters);

#endif

    return error_gradient;
}


LossIndex::FirstOrderLoss SumSquaredError::calculate_first_order_loss() const
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
        {
            first_order_loss.loss += loss;
            first_order_loss.gradient += gradient;
         }
    }

//    const Matrix<double> regularization_Hessian = loss_index_pointer->calculate_regularization_Hessian();

    first_order_loss.gradient *= 2.0;

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

    return first_order_loss;
}


LossIndex::FirstOrderLoss SumSquaredError::calculate_batch_first_order_loss(const Vector<size_t>& batch_indices) const
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

    const Matrix<double> output_gradient = calculate_output_gradient(first_order_forward_propagation.layers_activations[layers_number-1], targets);

    const Vector< Matrix<double> > layers_delta = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

    const Vector<double> batch_error_gradient = calculate_error_gradient(inputs, first_order_forward_propagation.layers_activations, layers_delta);

    const double batch_error = first_order_forward_propagation.layers_activations[layers_number-1].calculate_sum_squared_error(targets);

    first_order_loss.loss = batch_error;
    first_order_loss.gradient += batch_error_gradient;

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

    return first_order_loss;
}

LossIndex::FirstOrderLoss SumSquaredError::calculate_batch_first_order_loss_cuda(const Vector<size_t>& batch_indices,
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

    vector<double> loss_parameters;

    string loss_method = write_error_term_type();

    calculateFirstOrderLossCUDA(pointers.weights_pointers.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                               pointers.biases_pointers.to_std_vector(), bias_rows_numbers,
                               input_data, input_rows, input_columns,
                               target_data, target_rows, target_columns,
                               error_gradient_data,
                               output_data, output_rows, output_columns,
                               pointers.layer_activations.to_std_vector(), loss_method, loss_parameters);

    first_order_loss.loss = calculate_error(outputs, targets_matrix);

    // Regularization

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

#endif

    return first_order_loss;
}

LossIndex::FirstOrderLoss SumSquaredError::calculate_batch_first_order_loss_cuda(const Vector<size_t>&,
                                                                                 const MultilayerPerceptron::Pointers&, const Vector<double*>&) const
{
    FirstOrderLoss first_order_loss;

    return first_order_loss;
}

/// Returns the loss value of a neural network according to the sum squared error on a data set.

double SumSquaredError::calculate_error(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   return outputs.calculate_sum_squared_error(targets);
}


double SumSquaredError::calculate_error(const Vector<size_t>& instances_indices, const Vector<double>& parameters) const
{
   // Neural network stuff

   #ifdef __OPENNN_DEBUG__

   check();

   // Sum squared error stuff

   ostringstream buffer;

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->get_parameters_number();

   if(size != parameters_number)
   {
      buffer << "OpenNN Exception: SumSquaredError class." << endl
             << "double calculate_error(const Vector<double>&) const method." << endl
             << "Size(" << size << ") must be equal to number of parameters(" << parameters_number << ")." << endl;

      throw logic_error(buffer.str());
   }

   #endif

   // Data set stuff

   const Matrix<double> inputs = data_set_pointer->get_inputs(instances_indices);

   const Matrix<double> targets = data_set_pointer->get_targets(instances_indices);

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

   return calculate_error(outputs, targets);
}


// Test combination

Matrix<double> SumSquaredError::calculate_output_gradient(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    return (outputs-targets)*2.0;
}



/// Calculates the squared error terms for each instance, and returns it in a vector of size the number training instances. 

Vector<double> SumSquaredError::calculate_error_terms(const Matrix<double>& outputs, const Matrix<double>& targets) const
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


Vector<double> SumSquaredError::calculate_error_terms(const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const Matrix<double> inputs = data_set_pointer->get_training_inputs();

    const Matrix<double> targets = data_set_pointer->get_training_targets();

    const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

    return outputs.calculate_error_rows(targets);
}


/// Returns the squared errors of the training instances. 

Vector<double> SumSquaredError::calculate_squared_errors() const
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

   const MissingValues missing_values = data_set_pointer->get_missing_values();

   // Loss index

   Vector<double> squared_errors(training_instances_number);

   #pragma omp parallel for

   for(int i = 0; i < static_cast<int>(training_instances_number); i++)
   {
       const size_t training_index = training_indices[i];

      // Input vector

      const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

      // Target vector

      const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Error

      squared_errors[i] = outputs.calculate_sum_squared_error(targets);
   }

   return(squared_errors);
*/
    return Vector<double>();
}


LossIndex::SecondOrderLoss SumSquaredError::calculate_terms_second_order_loss() const
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
        {
            terms_second_order_loss.loss += loss;
            terms_second_order_loss.gradient += gradient;
            terms_second_order_loss.Hessian_approximation += Hessian_approximation;
         }
    }

//    const Matrix<double> regularization_Hessian = loss_index_pointer->calculate_regularization_Hessian();

    terms_second_order_loss.gradient *= 2.0;
    terms_second_order_loss.Hessian_approximation *= 2.0;

    return terms_second_order_loss;
}


// string write_error_term_type() const method

/// Returns a string with the name of the sum squared error loss type, "SUM_SQUARED_ERROR".

string SumSquaredError::write_error_term_type() const
{
   return("SUM_SQUARED_ERROR");
}


// tinyxml2::XMLDocument* to_XML() method method 

/// Returns a representation of the sum squared error object, in XML format. 

tinyxml2::XMLDocument* SumSquaredError::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Sum squared error

   tinyxml2::XMLElement* root_element = document->NewElement("SumSquaredError");

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

void SumSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("Error");

    file_stream.PushAttribute("Type", "SUM_SQUARED_ERROR");

    file_stream.CloseElement();

    // Regularization

    write_regularization_XML(file_stream);
}


// void load(const tinyxml2::XMLDocument&) method

/// Loads a sum squared error object from a XML document.
/// @param document TinyXML document containing the members of the object.

void SumSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("SumSquaredError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SumSquaredError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Sum squared element is nullptr.\n";

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
