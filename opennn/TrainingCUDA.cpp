/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */ 
/*   T R A I N I N G   C U D A    C L A S S                                                                     */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "TrainingCUDA.h"


#ifdef __OPENNN_CUDA__
#include <cuda_runtime.h>
#include <cublas_v2.h>

void initCUDA();

int mallocCUDA(double** A_d, int nBytes);
int memcpyCUDA(double* A_d, const double* A_h, int nBytes);
int getHostVector(const double* A_d, double* A_h, int nBytes);
void freeCUDA(double* A_d);

void randomizeVector(double* A_d, int n);

void calculateOutputsCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                          const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                          const double* input_data_h, const size_t input_rows, const size_t input_columns,
                          double* output_data_h, const size_t output_rows, const size_t output_columns,
                          const std::vector<std::string> layers_activations);

void calculateFirstOrderForwardPropagationCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                                               const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                                               const double* input_data_h, const size_t input_rows, const size_t input_columns,
                                               std::vector<double*> layers_activations_data, std::vector<double*> layers_activation_derivatives_data,
                                               const std::vector<size_t> activations_rows_numbers, const std::vector<size_t> activations_columns_numbers,
                                               const std::vector<std::string> layers_activations);

void calculateFirstOrderLossCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                                const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                                const double* input_data_h, const size_t input_rows, const size_t input_columns,
                                const double* target_data_h, const size_t target_rows, const size_t target_columns,
                                std::vector<double*> error_gradient_data,
                                double* output_data_h, const size_t output_rows, const size_t output_columns,
                                const std::vector<std::string> layers_activations, const std::string loss_method,
                                const std::vector<double> loss_parameters = vector<double>());

void updateParametersCUDA(std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                          std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                          const double* gradient_h, const size_t parameters_number);
#endif


namespace OpenNN
{
// DEFAULT CONSTRUCTOR

TrainingCUDA::TrainingCUDA()
{

}

// LOSS INDEX CONSTRUCTOR

TrainingCUDA::TrainingCUDA(DataSet* new_data_set_pointer)
: data_set_pointer(new_data_set_pointer)
{

}

// DESTRUCTOR

TrainingCUDA::~TrainingCUDA()
{
    for(size_t i = 0; i < weights_gpu.size(); i++)
    {
        freeCUDA(weights_gpu[i]);
    }

    for(size_t i = 0; i < biases_gpu.size(); i++)
    {
        freeCUDA(biases_gpu[i]);
    }
}

// Getters

Matrix<double> TrainingCUDA::get_weights_host(const size_t& index) const
{
    Matrix<double> weights(neural_network_architecture[index],neural_network_architecture[index+1]);
    double* weights_data = weights.data();

    const int num_bytes = neural_network_architecture[index]*neural_network_architecture[index+1]*sizeof(double);

    getHostVector(weights_gpu[index], weights_data, num_bytes);

    return weights;
}

Vector<double> TrainingCUDA::get_biases_host(const size_t& index) const
{
    Vector<double> biases(neural_network_architecture[index+1]);
    double* biases_data = biases.data();

    const int num_bytes = neural_network_architecture[index+1]*sizeof(double);

    getHostVector(biases_gpu[index], biases_data, num_bytes);

    return biases;
}

Vector<double> TrainingCUDA::get_parameters_host() const
{
    const size_t layers_number = neural_network_architecture.size() - 1;

    size_t parameters_number = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        parameters_number += neural_network_architecture[i]*neural_network_architecture[i+1] + neural_network_architecture[i+1];
    }

    Vector<double> parameters(parameters_number);
    double* parameters_data = parameters.data();

    size_t index = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        const int weights_num_bytes = neural_network_architecture[i]*neural_network_architecture[i+1]*sizeof(double);

        getHostVector(weights_gpu[i], parameters_data+index, weights_num_bytes);
        index += neural_network_architecture[i]*neural_network_architecture[i+1];

        const int biases_num_bytes = neural_network_architecture[i+1]*sizeof(double);

        getHostVector(biases_gpu[i], parameters_data+index, biases_num_bytes);
        index += neural_network_architecture[i+1];
    }

    return parameters;
}

// Setters

void TrainingCUDA::set_neural_network_architecture(const Vector<size_t>& new_neural_network_architecture)
{
    neural_network_architecture = new_neural_network_architecture;
}

// CUDA initialization methods

void TrainingCUDA::initialize_CUDA(void)
{
    const size_t layers_number = neural_network_architecture.size() - 1;

    int error;

    int deviceCount;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;

    weights_gpu.set(layers_number);
    biases_gpu.set(layers_number);

    layer_activations.set(layers_number, "RectifiedLinear");
    layer_activations[layers_number-1] = "RectifiedLinear";

    loss_method = "SUM_SQUARED_ERROR";

    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);

    if(cudaResultCode != cudaSuccess)
    {
        deviceCount = 0;
    }

    for(int device = 0; device < deviceCount; ++device)
    {
        cudaGetDeviceProperties(&properties, device);

        if(properties.major != 9999) /* 9999 means emulation only */
        {
            ++gpuDeviceCount;
        }
        else if(properties.major > 3)
        {
            ++gpuDeviceCount;
        }
        else if(properties.major == 3 && properties.minor >= 5)
        {
            ++gpuDeviceCount;
        }
    }

    if(gpuDeviceCount > 0)
    {
        for(size_t i = 0; i < layers_number; i++)
        {
            const int weights_num_bytes = neural_network_architecture[i]*neural_network_architecture[i+1]*sizeof(double);

            error = mallocCUDA(&weights_gpu[i], weights_num_bytes);
            if(error != 0)
            {
                gpuDeviceCount = 0;
                break;
            }

            const int biases_num_bytes = neural_network_architecture[i+1]*sizeof(double);

            error = mallocCUDA(&biases_gpu[i], biases_num_bytes);
            if(error != 0)
            {
                gpuDeviceCount = 0;
                break;
            }
        }
    }

    if(gpuDeviceCount > 0)
    {
        CUDA_initialized = true;
    }
}

void TrainingCUDA::randomize_parameters(void)
{
    for(size_t i = 0; i < weights_gpu.size(); i++)
    {
        randomizeVector(weights_gpu[i], neural_network_architecture[i]*neural_network_architecture[i+1]);
    }

    for(size_t i = 0; i < biases_gpu.size(); i++)
    {
        randomizeVector(biases_gpu[i], neural_network_architecture[i+1]);
    }
}

// Operation methods

Matrix<double> TrainingCUDA::calculate_outputs(const Matrix<double>& inputs_matrix)
{
    const size_t layers_number = neural_network_architecture.size() - 1;

    const double* input_data = inputs_matrix.data();
    const size_t input_rows = inputs_matrix.get_rows_number();
    const size_t input_columns = inputs_matrix.get_columns_number();

    Matrix<double> outputs(inputs_matrix.get_rows_number(), neural_network_architecture[layers_number]);
    double* output_data = outputs.data();
    const size_t output_rows = inputs_matrix.get_rows_number();
    const size_t output_columns = neural_network_architecture[layers_number];

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = neural_network_architecture[i];
        weights_columns_numbers[i] = neural_network_architecture[i+1];

        bias_rows_numbers[i] = neural_network_architecture[i+1];
    }

    calculateOutputsCUDA(weights_gpu.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                         biases_gpu.to_std_vector(), bias_rows_numbers,
                         input_data, input_rows, input_columns,
                         output_data, output_rows, output_columns,
                         layer_activations.to_std_vector());

    return outputs;
}

MultilayerPerceptron::FirstOrderForwardPropagation TrainingCUDA::calculate_first_order_forward_propagation(const Matrix<double>& inputs_matrix)
{
    const size_t layers_number = neural_network_architecture.size() - 1;

    const double* input_data = inputs_matrix.data();
    const size_t input_rows = inputs_matrix.get_rows_number();
    const size_t input_columns = inputs_matrix.get_columns_number();

    MultilayerPerceptron::FirstOrderForwardPropagation first_order_propagation(layers_number);

    vector<double*> layers_activations_data(layers_number);
    vector<double*> layers_activation_derivatives_data(layers_number);

    vector<size_t> activations_rows_numbers(layers_number);
    vector<size_t> activations_columns_numbers(layers_number);

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = neural_network_architecture[i];
        weights_columns_numbers[i] = neural_network_architecture[i+1];

        bias_rows_numbers[i] = neural_network_architecture[i+1];

        activations_rows_numbers[i] = input_rows;
        activations_columns_numbers[i] = neural_network_architecture[i+1];

        first_order_propagation.layers_activations[i].set(activations_rows_numbers[i], activations_columns_numbers[i]);
        first_order_propagation.layers_activation_derivatives[i].set(activations_rows_numbers[i], activations_columns_numbers[i]);

        layers_activations_data[i] = first_order_propagation.layers_activations[i].data();
        layers_activation_derivatives_data[i] = first_order_propagation.layers_activation_derivatives[i].data();
    }

    calculateFirstOrderForwardPropagationCUDA(weights_gpu.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                                              biases_gpu.to_std_vector(), bias_rows_numbers,
                                              input_data, input_rows, input_columns,
                                              layers_activations_data, layers_activation_derivatives_data,
                                              activations_rows_numbers, activations_columns_numbers,
                                              layer_activations.to_std_vector());

    return first_order_propagation;
}


Vector<double> TrainingCUDA::calculate_batch_error_gradient(const Vector<size_t>& batch_indices)
{
    const size_t layers_number = neural_network_architecture.size() - 1;

    const Matrix<double> inputs_matrix = data_set_pointer->get_inputs(batch_indices);
    const double* input_data = inputs_matrix.data();
    const size_t input_rows = inputs_matrix.get_rows_number();
    const size_t input_columns = inputs_matrix.get_columns_number();

    const Matrix<double> targets_matrix = data_set_pointer->get_targets(batch_indices);
    const double* target_data = targets_matrix.data();
    const size_t target_rows = targets_matrix.get_rows_number();
    const size_t target_columns = targets_matrix.get_columns_number();

    Matrix<double> outputs(inputs_matrix.get_rows_number(), neural_network_architecture[layers_number]);
    double* output_data = outputs.data();
    const size_t output_rows = inputs_matrix.get_rows_number();
    const size_t output_columns = neural_network_architecture[layers_number];

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    size_t parameters_number = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = neural_network_architecture[i];
        weights_columns_numbers[i] = neural_network_architecture[i+1];

        bias_rows_numbers[i] = neural_network_architecture[i+1];

        parameters_number += neural_network_architecture[i]*neural_network_architecture[i+1] + neural_network_architecture[i+1];
    }

    Vector<double> error_gradient(parameters_number);
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

    if(loss_method == "MINKOWSKI_ERROR")
    {
        loss_parameters.resize(1);

        loss_parameters[0] = 1; // Minkowski parameter
    }
    else if(loss_method == "NORMALIZED_SQUARED_ERROR")
    {
        loss_parameters.resize(1);

        loss_parameters[0] = 100; // Normalization coefficient
    }
    else if(loss_method == "WEIGHTED_SQUARED_ERROR")
    {
        loss_parameters.resize(3);

        loss_parameters[0] = 1; // Positives weight
        loss_parameters[1] = 1; // Negatives weight
        loss_parameters[2] = 1; // Normalization coefficient
    }

    calculateFirstOrderLossCUDA(weights_gpu.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                               biases_gpu.to_std_vector(), bias_rows_numbers,
                               input_data, input_rows, input_columns,
                               target_data, target_rows, target_columns,
                               error_gradient_data,
                               output_data, output_rows, output_columns,
                               layer_activations.to_std_vector(), loss_method, loss_parameters);


    return error_gradient;
}


void TrainingCUDA::update_parameters(const Vector<double>& gradient)
{
    const size_t layers_number = neural_network_architecture.size() - 1;

    const double* gradient_data = gradient.data();

    vector<size_t> weights_rows_numbers(layers_number);
    vector<size_t> weights_columns_numbers(layers_number);

    vector<size_t> bias_rows_numbers(layers_number);

    size_t parameters_number = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        weights_rows_numbers[i] = neural_network_architecture[i];
        weights_columns_numbers[i] = neural_network_architecture[i+1];

        bias_rows_numbers[i] = neural_network_architecture[i+1];

        parameters_number += neural_network_architecture[i]*neural_network_architecture[i+1] + neural_network_architecture[i+1];
    }

    updateParametersCUDA(weights_gpu.to_std_vector(), weights_rows_numbers, weights_columns_numbers,
                         biases_gpu.to_std_vector(), bias_rows_numbers,
                         gradient_data, parameters_number);
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

