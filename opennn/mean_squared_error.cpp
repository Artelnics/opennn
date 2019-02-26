/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M E A N   S Q U A R E D   E R R O R   C L A S S                                                            */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "mean_squared_error.h"

#ifdef __OPENNN_CUDA__
#include <cuda_runtime.h>
#include <cublas_v2.h>

void freeCUDA(double* A_d);

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
/// It creates a mean squared error term not associated to any
/// neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

MeanSquaredError::MeanSquaredError() : LossIndex()
{
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor.
/// It creates a mean squared error term object associated to a
/// neural network object but not measured on any data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

MeanSquaredError::MeanSquaredError(NeuralNetwork* new_neural_network_pointer)
: LossIndex(new_neural_network_pointer)
{
}


// DATA SET CONSTRUCTOR

/// Data set constructor.
/// It creates a mean squared error term not associated to any
/// neural network but to be measured on a given data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

MeanSquaredError::MeanSquaredError(DataSet* new_data_set_pointer)
: LossIndex(new_data_set_pointer)
{
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor.
/// It creates a mean squared error term object associated to a
/// neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

MeanSquaredError::MeanSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
: LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
}


// XML CONSTRUCTOR

/// XML constructor.
/// It creates a mean squared error object with all pointers set to nullptr.
/// The object members are loaded by means of a XML document.
/// Please be careful with the format of that file, which is specified in the OpenNN manual.
/// @param mean_squared_error_document TinyXML document with the mean squared error elements.

MeanSquaredError::MeanSquaredError(const tinyxml2::XMLDocument& mean_squared_error_document)
 : LossIndex(mean_squared_error_document)
{
    from_XML(mean_squared_error_document);
}


// COPY CONSTRUCTOR

/// Copy constructor.
/// It creates a copy of an existing mean squared error object.
/// @param other_mean_squared_error Mean squared error object to be copied.

MeanSquaredError::MeanSquaredError(const MeanSquaredError& other_mean_squared_error)
: LossIndex(other_mean_squared_error)
{
}


// DESTRUCTOR

/// Destructor.

MeanSquaredError::~MeanSquaredError()
{
}


// METHODS

double MeanSquaredError::calculate_training_error() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const size_t training_instances_number = data_set_pointer->get_instances_pointer()->get_training_instances_number();

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

    return training_error/static_cast<double>(training_instances_number);
}


double MeanSquaredError::calculate_selection_error() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const size_t selection_instances_number = data_set_pointer->get_instances_pointer()->get_selection_instances_number();

    const Vector< Vector<size_t> > selection_batches = data_set_pointer->get_instances_pointer()->get_selection_batches(batch_size);

    const size_t batches_number = selection_batches.size();

    double selection_error = 0.0;

    #pragma omp parallel for reduction(+ : selection_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(selection_batches[i]);
        const Matrix<double> targets = data_set_pointer->get_targets(selection_batches[i]);

        const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        const double batch_error = outputs.calculate_sum_squared_error(targets);

        selection_error += batch_error;
    }

    return selection_error/static_cast<double>(selection_instances_number);
}


double MeanSquaredError::calculate_training_error(const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const size_t training_instances_number = data_set_pointer->get_instances_pointer()->get_training_instances_number();

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

    return training_error/static_cast<double>(training_instances_number);
}


double MeanSquaredError::calculate_batch_error(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Data set

    const size_t instances_number = batch_indices.size();

    // Neural network

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Loss index

    const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices);
    const Matrix<double> targets = data_set_pointer->get_targets(batch_indices);

    const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

    const double batch_error = outputs.calculate_sum_squared_error(targets);

    return (batch_error/instances_number);

}

double MeanSquaredError::calculate_batch_error_cuda(const Vector<size_t>& batch_indices, const MultilayerPerceptron::Pointers& pointers) const
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

    const size_t instances_number = batch_indices.size();

    batch_error = outputs.calculate_sum_squared_error(targets_matrix) / static_cast<double>(instances_number);

#endif

    return batch_error;
}

Vector<double> MeanSquaredError::calculate_training_error_gradient() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Neural network

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    // Data set

    const size_t training_instances_number = data_set_pointer->get_instances().get_training_instances_number();

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

    return training_error_gradient/static_cast<double>(training_instances_number);
}

LossIndex::FirstOrderLoss MeanSquaredError::calculate_first_order_loss() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    // Data set

    const size_t training_instances_number = data_set_pointer->get_instances_pointer()->get_training_instances_number();

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

        Matrix<double> output_gradient = (first_order_forward_propagation.layers_activations[layers_number-1] - targets)/*/error_terms*/;
        output_gradient.divide_by_rows(error_terms);

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

//    const Matrix<double> regularization_Hessian = loss_index_pointer->calculate_regularization_Hessian();

    first_order_loss.loss /= static_cast<double>(training_instances_number);
    first_order_loss.gradient *= (2.0/static_cast<double>(training_instances_number));

    return first_order_loss;
}


Vector<double> MeanSquaredError::calculate_batch_error_gradient(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Data set

    const size_t instances_number = batch_indices.size();

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

    return batch_error_gradient/static_cast<double>(instances_number);

}


Vector<double> MeanSquaredError::calculate_batch_error_gradient_cuda(const Vector<size_t>& batch_indices, const MultilayerPerceptron::Pointers& pointers) const
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


LossIndex::FirstOrderLoss MeanSquaredError::calculate_batch_first_order_loss(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Data set

    const size_t instances_number = batch_indices.size();

    // Neural network

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    // Loss index

    FirstOrderLoss first_order_loss(parameters_number);

    const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices);

    const Matrix<double> targets = data_set_pointer->get_targets(batch_indices);

    const MultilayerPerceptron::FirstOrderForwardPropagation first_order_forward_propagation=
           multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

    const Matrix<double> output_gradient = calculate_output_gradient(first_order_forward_propagation.layers_activations[layers_number-1], targets);

    const Vector< Matrix<double> > layers_delta = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

    const Vector<double> batch_error_gradient = calculate_error_gradient(inputs, first_order_forward_propagation.layers_activations, layers_delta);

    const double batch_error = first_order_forward_propagation.layers_activations[layers_number-1].calculate_sum_squared_error(targets);

    first_order_loss.loss = batch_error / static_cast<double>(instances_number);
    first_order_loss.gradient = batch_error_gradient/static_cast<double>(instances_number);

    // Regularization

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

    return first_order_loss;
}

LossIndex::FirstOrderLoss MeanSquaredError::calculate_batch_first_order_loss_cuda(const Vector<size_t>& batch_indices,
                                                                                  const MultilayerPerceptron::Pointers& pointers) const
{
    FirstOrderLoss first_order_loss;

#ifdef __OPENNN_CUDA__

    const size_t instances_number = batch_indices.size();
    const size_t layers_number = pointers.architecture.size() - 1;

    const size_t inputs_number = data_set_pointer->get_variables().get_inputs_number();
    const size_t targets_number = data_set_pointer->get_variables().get_targets_number();

    const Matrix<double> targets_matrix = data_set_pointer->get_targets(batch_indices);

    Matrix<double> outputs(instances_number, pointers.architecture[layers_number]);
    double* output_data = outputs.data();
    const size_t output_rows = instances_number;
    const size_t output_columns = pointers.architecture[layers_number];

    Vector<double*> data_device = data_set_pointer->host_to_device(batch_indices);

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
                                data_device[0], instances_number, inputs_number,
                                data_device[1], instances_number, targets_number,
                                error_gradient_data,
                                output_data, output_rows, output_columns,
                                pointers.layer_activations.to_std_vector(), loss_method, loss_parameters);

    const double batch_error = outputs.calculate_sum_squared_error(targets_matrix);

    first_order_loss.loss = batch_error / static_cast<double>(instances_number);

    // Regularization

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

    freeCUDA(data_device[0]);
    freeCUDA(data_device[1]);

#endif

    return first_order_loss;
}

LossIndex::FirstOrderLoss MeanSquaredError::calculate_batch_first_order_loss_cuda(const Vector<size_t>& batch_indices,
                                                                                  const MultilayerPerceptron::Pointers& pointers, const Vector<double*>& data_device) const
{
    FirstOrderLoss first_order_loss;

#ifdef __OPENNN_CUDA__

    const size_t instances_number = batch_indices.size();
    const size_t layers_number = pointers.architecture.size() - 1;

    const size_t inputs_number = data_set_pointer->get_variables().get_inputs_number();
    const size_t targets_number = data_set_pointer->get_variables().get_targets_number();

    const Matrix<double> targets_matrix = data_set_pointer->get_targets(batch_indices);

    Matrix<double> outputs(instances_number, pointers.architecture[layers_number]);
    double* output_data = outputs.data();
    const size_t output_rows = instances_number;
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
                               data_device[0], instances_number, inputs_number,
                               data_device[1], instances_number, targets_number,
                               error_gradient_data,
                               output_data, output_rows, output_columns,
                               pointers.layer_activations.to_std_vector(), loss_method, loss_parameters);

    const double batch_error = outputs.calculate_sum_squared_error(targets_matrix);

    first_order_loss.loss = batch_error / static_cast<double>(instances_number);

    // Regularization

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

#endif

    return first_order_loss;
}

Matrix<double> MeanSquaredError::calculate_output_gradient(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    return (outputs-targets)*2.0;
}


/// Returns loss vector of the error terms function for the mean squared error.
/// It uses the error back-propagation method.

Vector<double> MeanSquaredError::calculate_error_terms(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
   // Control sentence

   #ifdef __OPENNN_DEBUG__

   check();

   #endif

   return outputs.calculate_error_rows(targets);
}


Vector<double> MeanSquaredError::calculate_error_terms(const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const Matrix<double> inputs = data_set_pointer->get_training_inputs();

    const Matrix<double> targets = data_set_pointer->get_training_targets();

    const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

    const size_t training_instances_number = inputs.get_rows_number();

    return outputs.calculate_error_rows(targets)/static_cast<double>(training_instances_number);
}


LossIndex::SecondOrderLoss MeanSquaredError::calculate_terms_second_order_loss() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    // Data set

    const size_t training_instances_number = data_set_pointer->get_instances_pointer()->get_training_instances_number();

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

        /*const */Matrix<double> output_gradient = (first_order_forward_propagation.layers_activations[layers_number-1] - targets)/*/error_terms*/;
        output_gradient.divide_by_rows(error_terms);

        const Vector< Matrix<double> > layers_delta
                = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

        const Matrix<double> error_terms_Jacobian
                = calculate_error_terms_Jacobian(inputs, first_order_forward_propagation.layers_activations, layers_delta);

        const Matrix<double> error_terms_Jacobian_transpose = error_terms_Jacobian.calculate_transpose();

        const double loss = error_terms.dot(error_terms);

        const Vector<double> gradient = error_terms_Jacobian_transpose.dot(error_terms);

        Matrix<double> Hessian_approximation;// = error_terms_Jacobian.dot(error_terms_Jacobian);
        Hessian_approximation.dot(error_terms_Jacobian_transpose, error_terms_Jacobian);

        #pragma omp critical
        {
            terms_second_order_loss.loss += loss;
            terms_second_order_loss.gradient += gradient;
            terms_second_order_loss.Hessian_approximation += Hessian_approximation;
         }
    }

    terms_second_order_loss.loss /= static_cast<double>(training_instances_number);
    terms_second_order_loss.gradient *= (2.0/static_cast<double>(training_instances_number));
    terms_second_order_loss.Hessian_approximation *= (2.0/static_cast<double>(training_instances_number));

    if(regularization_method != RegularizationMethod::None)
    {
        terms_second_order_loss.loss += calculate_regularization();
        terms_second_order_loss.gradient += calculate_regularization_gradient();
        terms_second_order_loss.Hessian_approximation += calculate_regularization_Hessian();
    }

    return terms_second_order_loss;
}


/// Returns a string with the name of the mean squared error loss type, "MEAN_SQUARED_ERROR".

string MeanSquaredError::write_error_term_type() const
{
   return("MEAN_SQUARED_ERROR");
}


/// Serializes the mean squared error object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document->

tinyxml2::XMLDocument* MeanSquaredError::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Mean squared error

   tinyxml2::XMLElement* mean_squared_error_element = document->NewElement("MeanSquaredError");

   document->InsertFirstChild(mean_squared_error_element);

   // Display
//   {
//      tinyxml2::XMLElement* element = document->NewElement("Display");
//      mean_squared_error_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   return(document);
}


void MeanSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Error type

    file_stream.OpenElement("Error");

    file_stream.PushAttribute("Type", "MEAN_SQUARED_ERROR");

    file_stream.CloseElement();

    // Regularization

    write_regularization_XML(file_stream);
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
