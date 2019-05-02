/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M I N K O W S K I   E R R O R   C L A S S                                                                  */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "minkowski_error.h"

#ifdef __OPENNN_CUDA__
#include <cuda_runtime.h>
#include <cublas_v2.h>

int getHostVector(const double* A_d, double* A_h, int nBytes);
int mallocCUDA(double** A_d, int nBytes);
int memcpyCUDA(double* A_d, const double* A_h, int nBytes);
void freeCUDA(double* A_d);

double calculateFirstOrderLossCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                                const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                                const double* input_data_h, const size_t input_rows, const size_t input_columns,
                                const double* target_data_h, const size_t target_rows, const size_t target_columns,
                                double* gradient_vec_d,
                                double* output_data_h, const size_t output_rows, const size_t output_columns,
                                const std::vector<std::string> layers_activations, const std::string loss_method,
                                const std::vector<double> loss_parameters = vector<double>());

//void calculateOutputsCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
//                          const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
//                          const double* input_data_h, const size_t input_rows, const size_t input_columns,
//                          double* output_data_h, const size_t output_rows, const size_t output_columns,
//                          const std::vector<std::string> layers_activations);

double calculateLossCUDA(const vector<double*> weights_d, const vector<size_t> weights_rows_numbers, const vector<size_t> weights_columns_numbers,
                         const vector<double*> biases_d, const vector<size_t> bias_rows_numbers,
                         const double* input_data_h, const size_t input_rows, const size_t input_columns,
                         const double* target_data_d, const size_t target_rows, const size_t target_columns,
                         const vector<string> layers_activations, const string loss_method,
                         const vector<double> loss_parameters);

void calculateGradientCUDA(const vector<double*> weights_d, const vector<size_t> weights_rows_numbers, const vector<size_t> weights_columns_numbers,
                           const vector<double*> biases_d, const vector<size_t> bias_rows_numbers,
                           const double* input_data_d, const size_t input_rows, const size_t input_columns,
                           const double* target_data_d, const size_t target_rows, const size_t target_columns,
                           double* gradient_vec_d,
                           const vector<string> layers_activations, const string loss_method,
                           const vector<double> loss_parameters);

#endif

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates Minkowski error term not associated to any neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

MinkowskiError::MinkowskiError() : LossIndex()
{
   set_default();
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a Minkowski error term associated to a neural network but not measured on any data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

MinkowskiError::MinkowskiError(NeuralNetwork* new_neural_network_pointer)
: LossIndex(new_neural_network_pointer)
{
   set_default();
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a Minkowski error term not associated to any neural network but to be measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

MinkowskiError::MinkowskiError(DataSet* new_data_set_pointer)
: LossIndex(new_data_set_pointer)
{
   set_default();
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a Minkowski error term object associated to a neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

MinkowskiError::MinkowskiError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
 : LossIndex(new_neural_network_pointer, new_data_set_pointer)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a Minkowski error object neither associated to a neural network nor to a data set. 
/// The object members are loaded by means of a XML document.
/// @param mean_squared_error_document TinyXML document with the Minkowski error elements.

MinkowskiError::MinkowskiError(const tinyxml2::XMLDocument& mean_squared_error_document)
 : LossIndex(mean_squared_error_document)
{
   set_default();

   from_XML(mean_squared_error_document);
}


// DESTRUCTOR

/// Destructor.
/// It does not delete any pointer. 

MinkowskiError::~MinkowskiError() 
{
}


// METHODS


/// Returns the Minkowski exponent value used to calculate the error. 

double MinkowskiError::get_Minkowski_parameter() const
{
   return(Minkowski_parameter);
}


/// Sets the default values to a Minkowski error object:
/// <ul>
/// <li> Minkowski parameter: 1.5.
/// <li> Display: true.
/// </ul>

void MinkowskiError::set_default()
{
   Minkowski_parameter = 1.5;

   display = true;
}


/// Sets a new Minkowski exponent value to be used in order to calculate the error. 
/// The Minkowski R-value must be comprised between 1 and 2. 
/// @param new_Minkowski_parameter Minkowski exponent value. 

void MinkowskiError::set_Minkowski_parameter(const double& new_Minkowski_parameter)
{
   // Control sentence

   if(new_Minkowski_parameter < 1.0 || new_Minkowski_parameter > 2.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Error. MinkowskiError class.\n"
             << "void set_Minkowski_parameter(const double&) method.\n"
             << "The Minkowski parameter must be comprised between 1 and 2\n";
    
      throw logic_error(buffer.str());
   }

   // Set Minkowski parameter
  
   Minkowski_parameter = new_Minkowski_parameter;
}


double MinkowskiError::calculate_training_error() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

     const size_t training_instances_number = data_set_pointer->get_instances_pointer()->get_training_instances_number();

    const size_t batches_number = training_batches.size();

    double training_error = 0.0;

    #pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

        const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        training_error += outputs.calculate_minkowski_error(targets, Minkowski_parameter);
    }

    return  training_error/training_instances_number;
}

double MinkowskiError::calculate_training_error_cuda() const
{
    double batch_error = 0.0;

#ifdef __OPENNN_CUDA__

    const Vector<size_t> training_indices = data_set_pointer->get_instances().get_training_indices();
    const Vector<size_t> architecture = neural_network_pointer->get_multilayer_perceptron_pointer()->get_architecture();
    const Vector<string> layer_activations = neural_network_pointer->get_multilayer_perceptron_pointer()->write_layers_activation_function();

    const size_t layers_number = architecture.size() - 1;

    const Matrix<double> inputs_matrix = data_set_pointer->get_inputs(training_indices);
    const double* input_data = inputs_matrix.data();
    const size_t input_rows = inputs_matrix.get_rows_number();
    const size_t input_columns = inputs_matrix.get_columns_number();

    Matrix<double> targets = data_set_pointer->get_targets(training_indices);
    const double* target_data = targets.data();
    const size_t target_rows = targets.get_rows_number();
    const size_t target_columns = targets.get_columns_number();

    double* target_data_d;

    mallocCUDA(&target_data_d, target_rows*target_columns*sizeof(double));
    memcpyCUDA(target_data_d, target_data, target_rows*target_columns*sizeof(double));

    vector<double*> weights_pointers(layers_number);
    vector<double*> biases_pointers(layers_number);

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = architecture[i];
        weights_columns_numbers[i] = architecture[i+1];

        bias_rows_numbers[i] = architecture[i+1];

        const double* weights_data = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(i).get_synaptic_weights().to_vector().data();
        const double* biases_data = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(i).get_biases().data();

        mallocCUDA(&weights_pointers[i], weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double));
        mallocCUDA(&biases_pointers[i], bias_rows_numbers[i]*sizeof(double));

        memcpyCUDA(weights_pointers[i], weights_data, weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double));
        memcpyCUDA(biases_pointers[i], biases_data, bias_rows_numbers[i]*sizeof(double));
    }

    //    calculateOutputsCUDA(pointers.weights_pointers.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
    //                         pointers.biases_pointers.to_std_vector(), bias_rows_numbers,
    //                         input_data, input_rows, input_columns,
    //                         output_data, output_rows, output_columns,
    //                         pointers.layer_activations.to_std_vector());

    //    const Matrix<double> targets_matrix = data_set_pointer->get_targets(batch_indices);

    //    const size_t instances_number = batch_indices.size();

    //    batch_error = outputs.calculate_sum_squared_error(targets_matrix) / static_cast<double>(instances_number);

    vector<double> loss_parameters(1, Minkowski_parameter);

    string loss_method = get_error_type();

    batch_error = calculateLossCUDA(weights_pointers, weights_rows_numbers, weights_columns_numbers,
                                    biases_pointers, bias_rows_numbers,
                                    input_data, input_rows, input_columns,
                                    target_data_d, target_rows, target_columns,
                                    layer_activations.to_std_vector(), loss_method,
                                    loss_parameters);

    freeCUDA(target_data_d);

    for(size_t i = 0; i < layers_number; i++)
    {
        freeCUDA(weights_pointers[i]);
        freeCUDA(biases_pointers[i]);
    }

#endif

    return batch_error;
}


double MinkowskiError::calculate_selection_error() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > selection_batches = data_set_pointer->get_instances_pointer()->get_selection_batches(batch_size);

    const size_t selection_instances_number = data_set_pointer->get_instances_pointer()->get_selection_instances_number();

    const size_t batches_number = selection_batches.size();

    double selection_error = 0.0;

#pragma omp parallel for reduction(+ : selection_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(selection_batches[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_inputs(selection_batches[static_cast<unsigned>(i)]);

        const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        selection_error += outputs.calculate_minkowski_error(targets, Minkowski_parameter);
    }

    return selection_error/selection_instances_number;
}


double MinkowskiError::calculate_selection_error_cuda() const
{
    double batch_error = 0.0;

#ifdef __OPENNN_CUDA__

    const Vector<size_t> selection_indices = data_set_pointer->get_instances().get_selection_indices();
    const Vector<size_t> architecture = neural_network_pointer->get_multilayer_perceptron_pointer()->get_architecture();
    const Vector<string> layer_activations = neural_network_pointer->get_multilayer_perceptron_pointer()->write_layers_activation_function();

    const size_t layers_number = architecture.size() - 1;

    const Matrix<double> inputs_matrix = data_set_pointer->get_inputs(selection_indices);
    const double* input_data = inputs_matrix.data();
    const size_t input_rows = inputs_matrix.get_rows_number();
    const size_t input_columns = inputs_matrix.get_columns_number();

    Matrix<double> targets = data_set_pointer->get_targets(selection_indices);
    const double* target_data = targets.data();
    const size_t target_rows = targets.get_rows_number();
    const size_t target_columns = targets.get_columns_number();

    double* target_data_d;

    mallocCUDA(&target_data_d, target_rows*target_columns*sizeof(double));
    memcpyCUDA(target_data_d, target_data, target_rows*target_columns*sizeof(double));

    vector<double*> weights_pointers(layers_number);
    vector<double*> biases_pointers(layers_number);

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = architecture[i];
        weights_columns_numbers[i] = architecture[i+1];

        bias_rows_numbers[i] = architecture[i+1];

        const double* weights_data = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(i).get_synaptic_weights().to_vector().data();
        const double* biases_data = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(i).get_biases().data();

        mallocCUDA(&weights_pointers[i], weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double));
        mallocCUDA(&biases_pointers[i], bias_rows_numbers[i]*sizeof(double));

        memcpyCUDA(weights_pointers[i], weights_data, weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double));
        memcpyCUDA(biases_pointers[i], biases_data, bias_rows_numbers[i]*sizeof(double));
    }

    vector<double> loss_parameters(1, Minkowski_parameter);

    string loss_method = get_error_type();

    batch_error = calculateLossCUDA(weights_pointers, weights_rows_numbers, weights_columns_numbers,
                                    biases_pointers, bias_rows_numbers,
                                    input_data, input_rows, input_columns,
                                    target_data_d, target_rows, target_columns,
                                    layer_activations.to_std_vector(), loss_method,
                                    loss_parameters);

    freeCUDA(target_data_d);

    for(size_t i = 0; i < layers_number; i++)
    {
        freeCUDA(weights_pointers[i]);
        freeCUDA(biases_pointers[i]);
    }

#endif

    return batch_error;
}


double MinkowskiError::calculate_training_error(const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer percptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Vector< Vector<size_t> > training_bathces = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

    const size_t training_instances_number = data_set_pointer->get_instances_pointer()->get_training_instances_number();

    const size_t batches_number = training_bathces.size();

    double training_error = 0.0;

#pragma omp parallel for reduction(+ : training_error)

    for(int i = 0; i < static_cast<int>(batches_number); i++)
    {
        const Matrix<double> inputs = data_set_pointer->get_inputs(training_bathces[static_cast<unsigned>(i)]);
        const Matrix<double> targets = data_set_pointer->get_targets(training_bathces[static_cast<unsigned>(i)]);

        const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

        training_error += outputs.calculate_minkowski_error(targets, Minkowski_parameter);
    }

    return training_error/training_instances_number;
}


double MinkowskiError::calculate_training_error_cuda(const Vector<double>& parameters) const
{
    double batch_error = 0.0;

#ifdef __OPENNN_CUDA__

    const Vector<size_t> training_indices = data_set_pointer->get_instances().get_training_indices();
    const Vector<size_t> architecture = neural_network_pointer->get_multilayer_perceptron_pointer()->get_architecture();
    const Vector<string> layer_activations = neural_network_pointer->get_multilayer_perceptron_pointer()->write_layers_activation_function();

    const size_t layers_number = architecture.size() - 1;

    const Matrix<double> inputs_matrix = data_set_pointer->get_inputs(training_indices);
    const double* input_data = inputs_matrix.data();
    const size_t input_rows = inputs_matrix.get_rows_number();
    const size_t input_columns = inputs_matrix.get_columns_number();

    Matrix<double> targets = data_set_pointer->get_targets(training_indices);
    const double* target_data = targets.data();
    const size_t target_rows = targets.get_rows_number();
    const size_t target_columns = targets.get_columns_number();

    double* target_data_d;

    mallocCUDA(&target_data_d, target_rows*target_columns*sizeof(double));
    memcpyCUDA(target_data_d, target_data, target_rows*target_columns*sizeof(double));

    vector<double*> weights_pointers(layers_number);
    vector<double*> biases_pointers(layers_number);

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    size_t index = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = architecture[i];
        weights_columns_numbers[i] = architecture[i+1];

        bias_rows_numbers[i] = architecture[i+1];

        const double* weights_data = parameters.get_subvector(index, index+weights_rows_numbers[i]*weights_columns_numbers[i]-1).data();
        index += weights_rows_numbers[i]*weights_columns_numbers[i];

        const double* biases_data = parameters.get_subvector(index, index+bias_rows_numbers[i]-1).data();
        index += bias_rows_numbers[i];

//        const double* weights_data = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(i).get_synaptic_weights().to_vector().data();
//        const double* biases_data = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(i).get_biases().data();

        mallocCUDA(&weights_pointers[i], weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double));
        mallocCUDA(&biases_pointers[i], bias_rows_numbers[i]*sizeof(double));

        memcpyCUDA(weights_pointers[i], weights_data, weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double));
        memcpyCUDA(biases_pointers[i], biases_data, bias_rows_numbers[i]*sizeof(double));
    }

//    calculateOutputsCUDA(pointers.weights_pointers.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
//                         pointers.biases_pointers.to_std_vector(), bias_rows_numbers,
//                         input_data, input_rows, input_columns,
//                         output_data, output_rows, output_columns,
//                         pointers.layer_activations.to_std_vector());

//    const Matrix<double> targets_matrix = data_set_pointer->get_targets(batch_indices);

//    batch_error = calculate_error(outputs, targets_matrix);


    vector<double> loss_parameters(1, Minkowski_parameter);

    string loss_method = get_error_type();

    batch_error = calculateLossCUDA(weights_pointers, weights_rows_numbers, weights_columns_numbers,
                                    biases_pointers, bias_rows_numbers,
                                    input_data, input_rows, input_columns,
                                    target_data_d, target_rows, target_columns,
                                    layer_activations.to_std_vector(), loss_method,
                                    loss_parameters);

    freeCUDA(target_data_d);

    for(size_t i = 0; i < layers_number; i++)
    {
        freeCUDA(weights_pointers[i]);
        freeCUDA(biases_pointers[i]);
    }

#endif

    return batch_error;
}


double MinkowskiError::calculate_batch_error(const Vector<size_t>& batch_indices) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Multilayer perceptron

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Data set

    const Matrix<double> inputs = data_set_pointer->get_inputs(batch_indices);
    const Matrix<double> targets = data_set_pointer->get_targets(batch_indices);

    const size_t instances_number = inputs.get_rows_number();

    const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

    return outputs.calculate_minkowski_error(targets, Minkowski_parameter)/instances_number;
}


Vector<double> MinkowskiError::calculate_training_error_gradient() const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Neural network

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

    // Data set

    const size_t training_instances_number = data_set_pointer->get_instances_pointer()->get_training_instances_number();

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

Vector<double> MinkowskiError::calculate_training_error_gradient_cuda() const
{
    const size_t parameters_number = neural_network_pointer->get_multilayer_perceptron_pointer()->get_parameters_number();

    Vector<double> training_error_gradient(parameters_number, 0.0);

#ifdef __OPENNN_CUDA__

    const Vector<size_t> training_indices = data_set_pointer->get_instances().get_training_indices();
    const Vector<size_t> architecture = neural_network_pointer->get_multilayer_perceptron_pointer()->get_architecture();
    const Vector<string> layer_activations = neural_network_pointer->get_multilayer_perceptron_pointer()->write_layers_activation_function();

    const size_t layers_number = architecture.size() - 1;

    const Matrix<double> inputs_matrix = data_set_pointer->get_inputs(training_indices);
    const double* input_data = inputs_matrix.data();
    const size_t input_rows = inputs_matrix.get_rows_number();
    const size_t input_columns = inputs_matrix.get_columns_number();

    Matrix<double> targets = data_set_pointer->get_targets(training_indices);
    const double* target_data = targets.data();
    const size_t target_rows = targets.get_rows_number();
    const size_t target_columns = targets.get_columns_number();

    double* input_data_d;
    double* target_data_d;

    mallocCUDA(&input_data_d, input_rows*input_columns*sizeof(double));
    mallocCUDA(&target_data_d, target_rows*target_columns*sizeof(double));

    memcpyCUDA(input_data_d, input_data, input_rows*input_columns*sizeof(double));
    memcpyCUDA(target_data_d, target_data, target_rows*target_columns*sizeof(double));

    vector<double*> weights_d(layers_number);
    vector<double*> biases_d(layers_number);

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = architecture[i];
        weights_columns_numbers[i] = architecture[i+1];

        bias_rows_numbers[i] = architecture[i+1];

        const double* weights_data = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(i).get_synaptic_weights().to_vector().data();
        const double* biases_data = neural_network_pointer->get_multilayer_perceptron_pointer()->get_layer(i).get_biases().data();

        mallocCUDA(&weights_d[i], weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double));
        mallocCUDA(&biases_d[i], bias_rows_numbers[i]*sizeof(double));

        memcpyCUDA(weights_d[i], weights_data, weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double));
        memcpyCUDA(biases_d[i], biases_data, bias_rows_numbers[i]*sizeof(double));
    }

    vector<double> loss_parameters(1, Minkowski_parameter);

    string loss_method = get_error_type();

    double* training_error_gradient_d;

    mallocCUDA(&training_error_gradient_d, parameters_number*sizeof(double));

    calculateGradientCUDA(weights_d, weights_rows_numbers, weights_columns_numbers,
                          biases_d, bias_rows_numbers,
                          input_data_d, input_rows, input_columns,
                          target_data_d, target_rows, target_columns,
                          training_error_gradient_d,
                          layer_activations, loss_method, loss_parameters);

    getHostVector(training_error_gradient_d, training_error_gradient.data(), parameters_number*sizeof(double));

    freeCUDA(input_data_d);
    freeCUDA(target_data_d);

    for(size_t i = 0; i < layers_number; i++)
    {
        freeCUDA(weights_d[i]);
        freeCUDA(biases_d[i]);
    }

#endif

    return training_error_gradient;
}


Matrix<double> MinkowskiError::calculate_output_gradient(const Matrix<double>& outputs, const Matrix<double>& targets) const
{

#ifdef __OPENNN_DEBUG__

check();

#endif

//    const size_t training_instances_number = data_set_pointer->get_instances_pointer()->get_training_instances_number();

    return (outputs-targets).calculate_LP_norm_gradient(Minkowski_parameter);//static_cast<double>(training_instances_number);

}


double MinkowskiError::calculate_error(const Matrix<double>& outputs, const Matrix<double>& targets) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    // Control sentence

    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    return outputs.calculate_minkowski_error(targets, Minkowski_parameter);
}


/// Returns which would be the Minkowski error of for an hypothetical vector of parameters.
/// It does not set that vector of parameters to the neural network.
/// @param parameters Vector of potential parameters for the neural network associated to the Minkowski error.
/// @todo

double MinkowskiError::calculate_error(const Vector<double>& parameters) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

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

      buffer << "OpenNN Exception: MeanSquaredError class.\n"
             << "double calculate_error(const Vector<double>&) const method.\n"
             << "Size(" << size << ") must be equal to number of parameters(" << parameters_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Neural network stuff

   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   // Data set

   const Instances& instances = data_set_pointer->get_instances();

   const Vector<size_t> training_indices = instances.get_training_indices();

   const size_t training_instances_number = training_indices.size();

   const Variables& variables = data_set_pointer->get_variables();

   const Vector<size_t> inputs_indices = variables.get_inputs_indices();
   const Vector<size_t> targets_indices = variables.get_targets_indices();

   double Minkowski_error = 0.0;
/*
   #pragma omp parallel for reduction(+ : Minkowski_error)

   for(int i = 0; i < static_cast<int>(training_instances_number); i++)
   {
       const size_t training_index = training_indices[static_cast<size_t>(i)];

      // Input vector

      const Vector<double> inputs = data_set_pointer->get_instance(training_index, inputs_indices);

      // Output vector

      const Vector<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

      // Target vector

      const Vector<double> targets = data_set_pointer->get_instance(training_index, targets_indices);

      // Minkowski error

      Minkowski_error += (outputs-targets).calculate_Lp_norm(Minkowski_parameter);
   }
*/
   return Minkowski_error/static_cast<double>(training_instances_number);
}


/// Returns the Minkowski error function output gradient of a multilayer perceptron on a data set.
/// It uses the error back-propagation method.

Vector<double> MinkowskiError::calculate_output_gradient(const Vector<size_t>& instances_indices, const Vector<double>& output, const Vector<double>& target) const
{
#ifdef __OPENNN_DEBUG__

check();

#endif

    const size_t instances_number = instances_indices.size();

    return (output-target).calculate_Lp_norm_gradient(Minkowski_parameter)/static_cast<double>(instances_number);
}


/// Returns a string with the name of the Minkowski error loss type, "MINKOWSKI_ERROR".

string MinkowskiError::get_error_type() const
{
   return "MINKOWSKI_ERROR";
}


string MinkowskiError::get_error_type_text() const
{
   return "Minkowski error";
}


/// Serializes the Minkowski error object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document-> 

tinyxml2::XMLDocument* MinkowskiError::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Minkowski error

   tinyxml2::XMLElement* Minkowski_error_element = document->NewElement("MinkowskiError");

   document->InsertFirstChild(Minkowski_error_element);

   // Minkowski parameter
   {
      tinyxml2::XMLElement* Minkowski_parameter_element = document->NewElement("MinkowskiParameter");
      Minkowski_error_element->LinkEndChild(Minkowski_parameter_element);

      buffer.str("");
      buffer << Minkowski_parameter;

      tinyxml2::XMLText* Minkowski_parameter_text = document->NewText(buffer.str().c_str());
      Minkowski_parameter_element->LinkEndChild(Minkowski_parameter_text);
   }

   // Display
//   {
//      tinyxml2::XMLElement* display_element = document->NewElement("Display");
//      Minkowski_error_element->LinkEndChild(display_element);

//      buffer.str("");
//      buffer << display;

//      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
//      display_element->LinkEndChild(display_text);
//   }

   return(document);
}


void MinkowskiError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Error type

    file_stream.OpenElement("Error");

    file_stream.PushAttribute("Type", "MINKOWSKI_ERROR");

    // Minkowski parameter

    file_stream.OpenElement("MinkowskiParameter");

    buffer.str("");
    buffer << Minkowski_parameter;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Close error

    file_stream.CloseElement();

    // Regularization

    write_regularization_XML(file_stream);

}


/// Loads a Minkowski error object from a XML document.
/// @param document TinyXML document containing the members of the object.

void MinkowskiError::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("MinkowskiError");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MinkowskiError class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Minkowski error element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Minkowski parameter

     const tinyxml2::XMLElement* error_element = root_element->FirstChildElement("Error");

     if(error_element)
     {
        const tinyxml2::XMLElement* parameter_element = error_element->FirstChildElement("MinkowskiParameter");

        const double new_Minkowski_parameter = atof(parameter_element->GetText());

        try
        {
           set_Minkowski_parameter(new_Minkowski_parameter);
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

//  // Display
//  {
//     const tinyxml2::XMLElement* display_element = root_element->FirstChildElement("Display");

//     if(display_element)
//     {
//        const string new_display_string = display_element->GetText();

//        try
//        {
//           set_display(new_display_string != "0");
//        }
//        catch(const logic_error& e)
//        {
//           cerr << e.what() << endl;
//        }
//     }
//  }
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
