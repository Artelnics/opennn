/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   W E I G T H E D   S Q U A R E D   E R R O R   C L A S S                                                    */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "weighted_squared_error.h"

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
/// It creates a weighted squared error term not associated to any
/// neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

WeightedSquaredError::WeightedSquaredError() : LossIndex()
{
    set_default();
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a weighted squared error term object associated to a
/// neural network object but not measured on any data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

WeightedSquaredError::WeightedSquaredError(NeuralNetwork* new_neural_network_pointer)
    : LossIndex(new_neural_network_pointer)
{
    set_default();
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a weighted squared error term not associated to any
/// neural network but to be measured on a given data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

WeightedSquaredError::WeightedSquaredError(DataSet* new_data_set_pointer)
    : LossIndex(new_data_set_pointer)
{
    set_default();
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a weighted squared error term object associated to a
/// neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

WeightedSquaredError::WeightedSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
    set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a weighted squared error object with all pointers set to nullptr.
/// The object members are loaded by means of a XML document.
/// Please be careful with the format of that file, which is specified in the OpenNN manual.
/// @param weighted_squared_error_document TinyXML document with the weighted squared error elements.

WeightedSquaredError::WeightedSquaredError(const tinyxml2::XMLDocument& weighted_squared_error_document)
    : LossIndex(weighted_squared_error_document)
{
    set_default();

    from_XML(weighted_squared_error_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing weighted squared error object.
/// @param other_weighted_squared_error Weighted squared error object to be copied.

WeightedSquaredError::WeightedSquaredError(const WeightedSquaredError& other_weighted_squared_error)
    : LossIndex(other_weighted_squared_error)
{
    negatives_weight = other_weighted_squared_error.negatives_weight;
    positives_weight = other_weighted_squared_error.positives_weight;
    normalization_coefficient = other_weighted_squared_error.normalization_coefficient;
}


// DESTRUCTOR

/// Destructor.

WeightedSquaredError::~WeightedSquaredError()
{
}


// METHODS

// double get_positives_weight() const method

/// Returns the weight of the positives.

double WeightedSquaredError::get_positives_weight() const
{
    return(positives_weight);
}


// double get_negatives_weight() const method

/// Returns the weight of the negatives.

double WeightedSquaredError::get_negatives_weight() const
{
    return(negatives_weight);
}

// double get_normalization_coefficient() const method

/// Returns the normalization coefficient.

double WeightedSquaredError::get_normalization_coefficient() const
{
    return(normalization_coefficient);
}


/// Set the default values for the object.

void WeightedSquaredError::set_default()
{
    if(has_data_set() && data_set_pointer->has_data())
    {
        set_weights();
        set_normalization_coefficient();
        set_selection_normalization_coefficient();
    }
    else
    {
        negatives_weight = 1.0;
        positives_weight = 1.0;

        normalization_coefficient = 1.0;
        selection_normalization_coefficient = 1.0;
    }
}


// void set_positives_weight(const double&)

/// Set a new weight for the positives values.
/// @param new_positives_weight New weight for the positives.

void WeightedSquaredError::set_positives_weight(const double& new_positives_weight)
{
    positives_weight = new_positives_weight;
}


// void set_negatives_weight(const double&)

/// Set a new weight for the negatives values.
/// @param new_negatives_weight New weight for the negatives.

void WeightedSquaredError::set_negatives_weight(const double& new_negatives_weight)
{
    negatives_weight = new_negatives_weight;
}

// void set_normalization_coefficient(const double&)

/// Set a new normalization coefficient.
/// @param new_normalization_coefficient New normalization coefficient.

void WeightedSquaredError::set_normalization_coefficient(const double& new_normalization_coefficient)
{
    normalization_coefficient = new_normalization_coefficient;
}

// void set_weights(const double&, const double&) method

/// Set new weights for the positives and negatives values.
/// @param new_positives_weight New weight for the positives.
/// @param new_negatives_weight New weight for the negatives.

void WeightedSquaredError::set_weights(const double& new_positives_weight, const double& new_negatives_weight)
{
    positives_weight = new_positives_weight;
    negatives_weight = new_negatives_weight;
}


// void set_weights() method

/// Calculates of the weights for the positives and negatives values with the data of the data set.

void WeightedSquaredError::set_weights()
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    const Vector<size_t> target_distribution = data_set_pointer->calculate_target_distribution();

    const size_t negatives = target_distribution[0];
    const size_t positives = target_distribution[1];

    if(positives == 0 || negatives == 0)
    {
        positives_weight = 1.0;
        negatives_weight = 1.0;

        return;
    }

    negatives_weight = 1.0;
    positives_weight = static_cast<double>(negatives)/static_cast<double>(positives);
}


/// Calculates of the normalization coefficient with the data of the data set.

void WeightedSquaredError::set_normalization_coefficient()
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> targets_indices = variables.get_targets_indices();

    const size_t negatives = data_set_pointer->calculate_training_negatives(targets_indices[0]);

    normalization_coefficient = negatives*negatives_weight*0.5;
}


/// Calculates of the selection normalization coefficient with the data of the data set.

void WeightedSquaredError::set_selection_normalization_coefficient()
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    const Vector<size_t> targets_indices = data_set_pointer->get_variables_pointer()->get_targets_indices();

    const size_t negatives = data_set_pointer->calculate_selection_negatives(targets_indices[0]);

    selection_normalization_coefficient = negatives*negatives_weight*0.5;
}



/// Returns the weighted squared error for the positive instances.

double WeightedSquaredError::calculate_positives_error() const
{
    // Control sentence

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

    // Weighted squared error stuff

    double sum_squared_error = 0.0;

    const double positives = get_positives_weight();

    #pragma omp parallel for firstprivate(positives) reduction(+:sum_squared_error)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[i];

        // Input vector

        const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        // Output vector

        const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        // Target vector

        const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

        // Sum squared error

        if(targets[0] == 1.0)
        {
            sum_squared_error += positives*outputs.calculate_sum_squared_error(targets);
        }
    }

    return(sum_squared_error);
*/
    return 0.0;
}


/// Returns the weighted squared error for the negative instances.

double WeightedSquaredError::calculate_negatives_error() const
{
    // Control sentence

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

    // Weighted squared error stuff

    double sum_squared_error = 0.0;

    #pragma omp parallel for reduction(+:sum_squared_error)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[i];

        // Input vector

        const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        // Output vector

        const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        // Target vector

        const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

        // Sum squared error

        if(targets[0] == 0.0)
        {
            sum_squared_error += negatives_weight*outputs.calculate_sum_squared_error(targets);
        }
    }

    return(sum_squared_error);
*/
    return 0.0;
}


double WeightedSquaredError::calculate_training_error() const
{
    // Control sentence

    #ifdef __OPENNN_DEBUG__

        check();

    #endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > batch_indices = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

    const size_t batches_number = batch_indices.size();

    double training_error = 0.0;

    #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(batch_indices[static_cast<unsigned>(i)]);

        const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        training_error += outputs.calculate_weighted_sum_squared_error(targets, positives_weight, negatives_weight);
    }

    return training_error / normalization_coefficient;
}


double WeightedSquaredError::calculate_selection_error() const
{
    // Control sentence

    #ifdef __OPENNN_DEBUG__

        check();

    #endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > batch_indices = data_set_pointer->get_instances_pointer()->get_selection_batches(batch_size);

    const size_t batches_number = batch_indices.size();

    double selection_error = 0.0;

    for(size_t i = 0; i < batches_number; i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(batch_indices[static_cast<unsigned>(i)]);

        const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        selection_error += outputs.calculate_weighted_sum_squared_error(targets, positives_weight, negatives_weight);
    }

    return selection_error / normalization_coefficient;
}



double WeightedSquaredError::calculate_training_error(const Vector<double>& parameters) const
{
    // Control sentence

    #ifdef __OPENNN_DEBUG__

        check();

    #endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > batch_indices = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

    const size_t batches_number = batch_indices.size();

    double training_error = 0.0;

    for(size_t i = 0; i < batches_number; i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(batch_indices[static_cast<unsigned>(i)]);

        const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

        training_error += outputs.calculate_weighted_sum_squared_error(targets, positives_weight, negatives_weight);
    }

    return training_error / normalization_coefficient;
}


double WeightedSquaredError::calculate_batch_error(const Vector<size_t>& batch_indices) const
{
    // Control sentence

    #ifdef __OPENNN_DEBUG__

        check();

    #endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices);
    const Matrix<double> targets = data_set_pointer->get_targets(batch_indices);

    const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

    const double batch_error = outputs.calculate_weighted_sum_squared_error(targets, positives_weight, negatives_weight);

    return batch_error / normalization_coefficient;
}


Vector<double> WeightedSquaredError::calculate_training_error_gradient() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Neural network

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    // Data set

//    const size_t training_instances_number = data_set_pointer->get_instances().get_training_instances_number();

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

    return training_error_gradient / normalization_coefficient;
}

Vector<double> WeightedSquaredError::calculate_batch_error_gradient(const Vector<size_t>& batch_indices) const
{
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

    return batch_error_gradient/normalization_coefficient;
}


LossIndex::FirstOrderLoss WeightedSquaredError::calculate_first_order_loss() const
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

    first_order_loss.loss /= normalization_coefficient;
    first_order_loss.gradient *= (2.0/normalization_coefficient);

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

    return first_order_loss;
}


LossIndex::FirstOrderLoss WeightedSquaredError::calculate_batch_first_order_loss(const Vector<size_t>& batch_indices) const
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

LossIndex::FirstOrderLoss WeightedSquaredError::calculate_batch_first_order_loss_cuda(const Vector<size_t>& batch_indices,
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

    vector<double> loss_parameters(3);

    loss_parameters[0] = positives_weight;
    loss_parameters[1] = negatives_weight;
    loss_parameters[2] = normalization_coefficient;

    string loss_method = write_error_term_type();

    calculateFirstOrderLossCUDA(pointers.weights_pointers.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                               pointers.biases_pointers.to_std_vector(), bias_rows_numbers,
                               input_data, input_rows, input_columns,
                               target_data, target_rows, target_columns,
                               error_gradient_data,
                               output_data, output_rows, output_columns,
                               pointers.layer_activations.to_std_vector(), loss_method, loss_parameters);

//    first_order_loss.loss = outputs.calculate_weighted_sum_squared_error(targets_matrix, positives_weight, negatives_weight) / normalization_coefficient;

    const Vector<double> error_terms = calculate_error_terms(outputs, targets_matrix);

    first_order_loss.loss = error_terms.dot(error_terms)/ normalization_coefficient;

    // Regularization

    if(regularization_method != RegularizationMethod::None)
    {
        first_order_loss.loss += calculate_regularization();
        first_order_loss.gradient += calculate_regularization_gradient();
    }

#endif

    return first_order_loss;
}

LossIndex::FirstOrderLoss WeightedSquaredError::calculate_batch_first_order_loss_cuda(const Vector<size_t>&,
                                                                                      const MultilayerPerceptron::Pointers&, const Vector<double*>&) const
{
    FirstOrderLoss first_order_loss;

    return first_order_loss;
}

/*
/// Returns the weighted squared error of a neural network on a data set.
/// @param given_normalization_coefficient Normalization coefficient to be used.

double WeightedSquaredError::calculate_error(const double& given_normalization_coefficient) const
{
    // Control sentence

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

    // Weighted squared error stuff

    double sum_squared_error = 0.0;

    #pragma omp parallel for reduction(+:sum_squared_error)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[i];

        // Input vector

        const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        // Output vector

        const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        // Target vector

        const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

        // Sum squared error

        double error;

        if(targets[0] == 1.0)
        {
            error = positives_weight*outputs.calculate_sum_squared_error(targets);
        }
        else if(targets[0] == 0.0)
        {
            error = negatives_weight*outputs.calculate_sum_squared_error(targets);
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: WeightedSquaredError class.\n"
                   << "double calculate_error() const method.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw logic_error(buffer.str());
        }

        sum_squared_error += error;
    }

    return(sum_squared_error/given_normalization_coefficient);

    return 0.0;
}


/// Returns which would be the error term of a neural network for an hypothetical
/// vector of parameters. It does not set that vector of parameters to the neural network.
/// @param parameters Vector of potential parameters for the neural network associated to the error term.
/// @param given_normalization_coefficient Normalization coefficient to be used.

double WeightedSquaredError::calculate_error(const Vector<double>& parameters, const double& given_normalization_coefficient) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    check();

#endif

#ifdef __OPENNN_DEBUG__

    const size_t size = parameters.size();

    const size_t parameters_number = neural_network_pointer->get_parameters_number();

    if(size != parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "double calculate_error(const Vector<double>&) const method.\n"
               << "Size(" << size << ") must be equal to number of parameters(" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

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

    // Weighted squared error stuff

    double sum_squared_error = 0.0;

    #pragma omp parallel for reduction(+:sum_squared_error)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[i];

        // Input vector

        const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        // Output vector

        const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

        // Target vector

        const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

        // Sum squared error

        double error;

        if(targets[0] == 1.0)
        {
            error = positives_weight*outputs.calculate_sum_squared_error(targets);
        }
        else if(targets[0] == 0.0)
        {
            error = negatives_weight*outputs.calculate_sum_squared_error(targets);
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: WeightedSquaredError class.\n"
                   << "double calculate_error(const Vector<double>&) const method.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw logic_error(buffer.str());
        }

        sum_squared_error += error;

    }

    return(sum_squared_error/given_normalization_coefficient);

    return 0.0;
}


/// Returns the weighted squared error of the neural network measured on the selection instances of the
/// data set.
/// @param given_normalization_coefficient Normalization coefficient to be used.

double WeightedSquaredError::calculate_selection_error(const double& given_normalization_coefficient) const
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    check();

#endif

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> selection_indices = instances.get_selection_indices();

    const size_t selection_instances_number = selection_indices.size();

    if(selection_instances_number == 0)
    {
        return(0.0);
    }

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    double selection_error = 0.0;

    #pragma omp parallel for reduction(+:selection_error)

    for(int i = 0; i < static_cast<int>(selection_instances_number); i++)
    {
        const size_t selection_index = selection_indices[i];

        // Input vector

        const Vector<double> inputs = data_set_pointer->get_instance(selection_index, inputs_indices);

        // Output vector

        const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        // Target vector

        const Vector<double> targets = data_set_pointer->get_instance(selection_index, targets_indices);

        // Sum squared error

        double loss;

        if(targets[0] == 1.0)
        {
            loss = positives_weight*outputs.calculate_sum_squared_error(targets);
        }
        else if(targets[0] == 0.0)
        {
            loss = negatives_weight*outputs.calculate_sum_squared_error(targets);
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: WeightedSquaredError class.\n"
                   << "double calculate_error(const Vector<double>&) const method.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw logic_error(buffer.str());
        }

        selection_error += loss;
    }

    return(selection_error/given_normalization_coefficient);

    return 0.0;
}
*/

/// Calculates the loss output gradient by means of the back-propagation algorithm,
/// and returns it in a single vector of size the number of multilayer perceptron parameters.
/// @param output Vector of the outputs of the model.
/// @param target Vector of targets of the data set.

Vector<double> WeightedSquaredError::calculate_output_gradient(const Vector<size_t>&, const Vector<double>& output, const Vector<double>& target) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    Vector<double> output_gradient;

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

//    const size_t training_instances_number = target.size();

//    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

//    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

//    Vector<double> targets(outputs_number);

    //size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

//    for(size_t i = 0; i < training_instances_number; i++)
//    {
        //targets = data_set_pointer->get_instance(training_index, targets_indices);

    if(target[0] == 1.0)
    {
        output_gradient = (output-target)*positives_w*2.0;
    }
    else if(target[0] == 0.0)
    {
        output_gradient = (output-target)*negatives_w*2.0;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "Vector<double> calculate_output_gradient() const method.\n"
               << "Target is neither a positive nor a negative.\n";

        throw logic_error(buffer.str());
    }
//    }

//    const size_t negatives = data_set_pointer->calculate_training_negatives(targets_indices[0]);

//    const double normalization_coefficient = negatives*negatives_weight*0.5;

    return output_gradient/normalization_coefficient;
}


//Matrix<double> WeightedSquaredError::calculate_output_gradient(const Matrix<double>& outputs, const Matrix<double>& targets) const
//{
//    return Matrix<double>();
//}


/// Calculates the loss output gradient by means of the back-propagation algorithm,
/// and returns it in a single vector of size the number of multilayer perceptron parameters.
/// @param output Vector of the outputs of the model.
/// @param target Vector of targets of the data set.
/// @param given_normalization_coefficient Coefficient of the normalization for the gradient.

Vector<double> WeightedSquaredError::calculate_output_gradient(const Vector<double>& output, const Vector<double>& target, const double& given_normalization_coefficient) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    Vector<double> output_gradient;

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

//    const size_t training_instances_number = target.size();

//    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

//    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

//    Vector<double> targets(outputs_number);

    //size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

//    for(size_t i = 0; i < training_instances_number; i++)
//    {
        //targets = data_set_pointer->get_instance(training_index, targets_indices);

    if(target[0] == 1.0)
    {
        output_gradient = (output-target)*positives_w*2.0;
    }
    else if(target[0] == 0.0)
    {
        output_gradient = (output-target)*negatives_w*2.0;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "Vector<double> calculate_output_gradient() const method.\n"
               << "Target is neither a positive nor a negative.\n";

        throw logic_error(buffer.str());
    }
//    }

    return output_gradient/given_normalization_coefficient;
}


Matrix<double> WeightedSquaredError::calculate_output_gradient(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    return (outputs-targets)*(targets*(negatives_weight/positives_weight-negatives_weight) + negatives_weight)*2.0/normalization_coefficient;
}


/*
Vector<double> WeightedSquaredError::calculate_error_terms() const
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> training_indices = instances.get_training_indices();

    const size_t training_instances_number = training_indices.size();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    // Weighted squared error stuff

    Vector<double> error_terms(training_instances_number);

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

        if(targets[0] == 1.0)
        {
            error_terms[i] = positives_weight*outputs.calculate_euclidean_distance(targets);
        }
        else if(targets[0] == 0.0)
        {
            error_terms[i] = negatives_weight*outputs.calculate_euclidean_distance(targets);
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: WeightedSquaredError class.\n"
                   << "Vector<double> WeightedSquaredError::calculate_error_terms() const.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw logic_error(buffer.str());
        }
    }

//    const size_t negatives = data_set_pointer->calculate_training_negatives(targets_indices[0]);

//    const double normalization_coefficient = negatives*negatives_weight*0.5;

    return(error_terms/sqrt(normalization_coefficient));

    return Vector<double>();
}
*/

/// Returns loss vector of the error terms function for the weighted squared error.
/// It uses the error back-propagation method.
/// @param outputs Output data
/// @param targets Target data

Vector<double> WeightedSquaredError::calculate_error_terms(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
    // Control sentence

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    return outputs.calculate_weighted_error_rows(targets, positives_weight, negatives_weight);
}


/// Returns loss vector of the error terms function for the weighted squared error for a given set of parameters.
/// It uses the error back-propagation method.
/// @param parameters Parameters of the neural network

Vector<double> WeightedSquaredError::calculate_error_terms(const Vector<double>& parameters) const
{
    // Control sentence

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const Matrix<double> inputs = data_set_pointer->get_training_inputs();
    const Matrix<double> targets = data_set_pointer->get_training_targets();

    const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

    return outputs.calculate_weighted_error_rows(targets, positives_weight, negatives_weight);
}


/*
/// Returns the Jacobian matrix of the weighted squared error function, whose elements are given by the
/// derivatives of the squared errors data set with respect to the multilayer perceptron parameters.

Matrix<double> WeightedSquaredError::calculate_error_terms_Jacobian() const
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t neural_parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const Vector<size_t> training_indices = instances.get_training_indices();

    const size_t training_instances_number = training_indices.size();

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.get_inputs_indices();
    const Vector<size_t> targets_indices = variables.get_targets_indices();

    // Loss index

    Vector<double> output_gradient(outputs_number);

    Vector< Vector<double> > layers_delta(layers_number);
    Vector<double> point_gradient(neural_parameters_number);

    Matrix<double> terms_Jacobian(training_instances_number, neural_parameters_number);

    // Main loop

    #pragma omp parallel for private(first_order_forward_propagation,  \
    output_gradient, layers_delta, point_gradient)

    for(int i = 0; i < static_cast<int>(training_instances_number); i++)
    {
        const size_t training_index = training_indices[i];

        const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

        first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

        const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
        const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

        Vector<double> term;
        double term_norm;

            const Vector<double>& outputs = first_order_forward_propagation[0][layers_number-1];

            term = (outputs-targets);

            term_norm = term.calculate_L2_norm();

            if(term_norm == 0.0)
            {
                output_gradient.set(outputs_number, 0.0);
            }
            else
            {
                output_gradient = term/term_norm;
            }

            layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

        point_gradient = calculate_point_gradient(inputs, layers_activation, layers_delta);

        terms_Jacobian.set_row(i, point_gradient);
    }

    const double negatives = training_instances_number
                           - data_set_pointer->get_training_targets().get_column(0).calculate_sum();

    const double normalization_coefficient = negatives*negatives_weight*0.5;

    return(terms_Jacobian/sqrt(normalization_coefficient));

    return Matrix<double>();
}
*/

LossIndex::SecondOrderLoss WeightedSquaredError::calculate_terms_second_order_loss() const
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

    terms_second_order_loss.loss /= normalization_coefficient;
    terms_second_order_loss.gradient *= (2.0/normalization_coefficient);
    terms_second_order_loss.Hessian_approximation *= (2.0/normalization_coefficient);

    if(regularization_method != RegularizationMethod::None)
    {
        terms_second_order_loss.loss += calculate_regularization();
        terms_second_order_loss.gradient += calculate_regularization_gradient();
        terms_second_order_loss.Hessian_approximation += calculate_regularization_Hessian();
    }

    return terms_second_order_loss;
}



/// Returns a string with the name of the weighted squared error loss type, "WEIGHTED_SQUARED_ERROR".

string WeightedSquaredError::write_error_term_type() const
{
    return("WEIGHTED_SQUARED_ERROR");
}


/// Serializes the weighted squared error object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document-> 

tinyxml2::XMLDocument* WeightedSquaredError::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Weighted squared error

    tinyxml2::XMLElement* weighted_squared_error_element = document->NewElement("WeightedSquaredError");

    document->InsertFirstChild(weighted_squared_error_element);

    // Positives weight
    {
    tinyxml2::XMLElement* element = document->NewElement("PositivesWeight");
    weighted_squared_error_element->LinkEndChild(element);

    buffer.str("");
    buffer << positives_weight;

    tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);
    }

    // Negatives weight
    {
    tinyxml2::XMLElement* element = document->NewElement("NegativesWeight");
    weighted_squared_error_element->LinkEndChild(element);

    buffer.str("");
    buffer << negatives_weight;

    tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);
    }

    // Display
    //   {
    //      tinyxml2::XMLElement* element = document->NewElement("Display");
    //      weighted_squared_error_element->LinkEndChild(element);

    //      buffer.str("");
    //      buffer << display;

    //      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
    //      element->LinkEndChild(text);
    //   }

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

void WeightedSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Error type

    file_stream.OpenElement("Error");

    file_stream.PushAttribute("Type", "WEIGHTED_SQUARED_ERROR");

    // Positives weight

    file_stream.OpenElement("PositivesWeight");

    buffer.str("");
    buffer << positives_weight;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Negatives weight

    file_stream.OpenElement("NegativesWeight");

    buffer.str("");
    buffer << negatives_weight;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Close error

    file_stream.CloseElement();

    // Regularization

    write_regularization_XML(file_stream);
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Loads a weighted squared error object from a XML document.
/// @param document Pointer to a TinyXML document with the object data.

void WeightedSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
   const tinyxml2::XMLElement* root_element = document.FirstChildElement("WeightedSquaredError");

   if(!root_element)
   {
       ostringstream buffer;

       buffer << "OpenNN Exception: WeightedSquaredError class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Weighted squared element is nullptr.\n";

       throw logic_error(buffer.str());
   }

   // Positives weight

   const tinyxml2::XMLElement* error_element = root_element->FirstChildElement("Error");

   const tinyxml2::XMLElement* positives_weight_element = error_element->FirstChildElement("PositivesWeight");

   if(positives_weight_element)
   {
      const string string = positives_weight_element->GetText();

      try
      {
         set_positives_weight(atof(string.c_str()));
      }
      catch(const logic_error& e)
      {
         cerr << e.what() << endl;
      }
   }

   // Negatives weight

   const tinyxml2::XMLElement* negatives_weight_element = error_element->FirstChildElement("NegativesWeight");

   if(negatives_weight_element)
   {
      const string string = negatives_weight_element->GetText();

      try
      {
         set_negatives_weight(atof(string.c_str()));
      }
      catch(const logic_error& e)
      {
         cerr << e.what() << endl;
      }
   }

   // Regularization

   tinyxml2::XMLDocument regularization_document;
   tinyxml2::XMLNode* element_clone;

   const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");

   element_clone = regularization_element->DeepClone(&regularization_document);

   regularization_document.InsertFirstChild(element_clone);

   regularization_from_XML(regularization_document);
}


// string print_weights() const method

string WeightedSquaredError::object_to_string() const
{
    ostringstream buffer;

    buffer << "Weighted squared error.\n"
           << "Positives weight: " << positives_weight << "\n"
           << "Negatives weight: " << negatives_weight << endl;

    return(buffer.str());
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
