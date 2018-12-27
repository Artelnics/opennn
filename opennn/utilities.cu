#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <stdio.h>
#include <vector>
#include <string>

void printHostVector(const double* vector, int n)
{
    for(int i = 0; i < n; i++)
    {
        printf("%g", vector[i]);
        if(i != n-1)
        {
            printf(",");
        }
    }

    printf("\n");
}

void printDeviceVector(const double* vector, int n)
{
    double* vector_h = (double*)malloc(n*sizeof(double));

    cudaMemcpy(vector_h, vector, n*sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0;
    for(int i = 0; i < n; i++)
    {
        sum+=vector_h[i];
        printf("%g", vector_h[i]);
        if(i != n-1)
        {
            printf(",");
        }
    }

    printf("\n");
    printf("%g\n", sum);
}

void createHandle(cublasHandle_t* handle)
{
	cublasCreate(handle);
}

void destroyHandle(cublasHandle_t* handle)
{
    cublasDestroy(*handle);
}

void initCUDA()
{
	cudaMalloc(nullptr, 0);
}

int mallocCUDA(double** A_d, int nBytes)
{
    cudaError_t error;

    error = cudaMalloc(A_d, nBytes);

    if(error != cudaSuccess)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int memcpyCUDA(double* A_d, const double* A_h, int nBytes)
{
    cudaError_t error;

    error = cudaMemcpy(A_d, A_h, nBytes, cudaMemcpyHostToDevice);

    if(error != cudaSuccess)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int getHostVector(const double* A_d, double* A_h, int nBytes)
{
    cudaError_t error;

    error = cudaMemcpy(A_h, A_d, nBytes, cudaMemcpyDeviceToHost);

    if(error != cudaSuccess)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void freeCUDA(double* A_d)
{
	cudaFree(A_d);
}

void randomizeVector(double* A_d, int n)
{
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    curandGenerateUniformDouble(gen, A_d, n);

//    curandGenerateNormalDouble(gen, A_d, n, 0, 1);

    curandDestroyGenerator(gen);
}

void dfpInverseHessian(
        double* old_parameters, double* parameters,
        double* old_gradient, double* gradient,
        double* old_inverse_Hessian, double* auxiliar_matrix,
        double* inverse_Hessian_host, int n)
{
	double alpha;
	double beta;
	
	double parameters_dot_gradient;
	double gradient_dot_Hessian_dot_gradient;
	
    double* vector = (double*)malloc(n*sizeof(double));

	cublasHandle_t handle;
	
	createHandle(&handle);
	
	alpha = -1;

    cublasDaxpy(handle, n, &alpha, old_parameters, 1, parameters, 1);
    cublasDaxpy(handle, n, &alpha, old_gradient, 1, gradient, 1);

    cublasDdot(handle, n, parameters, 1, gradient, 1, &parameters_dot_gradient);

    alpha = 1;
	beta = 0;
	
	cublasDgemv(handle, CUBLAS_OP_N, n, n, &alpha, old_inverse_Hessian, n, gradient, 1, &beta, old_gradient, 1);
	
    cublasDdot(handle, n, gradient, 1, old_gradient, 1, &gradient_dot_Hessian_dot_gradient);
	
	alpha = 0;

    cublasDscal(handle, n*n, &alpha, auxiliar_matrix, 1);
	
	alpha = 1;
	
	cublasDger(handle, n, n, &alpha, parameters, 1, parameters, 1, auxiliar_matrix, n);
	
	alpha = 1/parameters_dot_gradient;
	
    cublasDscal(handle, n*n, &alpha, auxiliar_matrix, 1);
	
	alpha = 1;
	beta = 1;
	
	cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, old_inverse_Hessian, n, 
                &beta, auxiliar_matrix, n, old_inverse_Hessian, n);

	alpha = 0;
	
    cublasDscal(handle, n*n, &alpha, auxiliar_matrix, 1);
	
	alpha = 1;
	
	cublasDger(handle, n, n, &alpha, old_gradient, 1, old_gradient, 1, auxiliar_matrix, n);
	
	alpha = 1;
    beta = -1/gradient_dot_Hessian_dot_gradient;
	
	cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, old_inverse_Hessian, n, 
                &beta, auxiliar_matrix, n, old_inverse_Hessian, n);

    cudaMemcpy(inverse_Hessian_host, old_inverse_Hessian, n*n*sizeof(double), cudaMemcpyDeviceToHost);
}

void bfgsInverseHessian(
        double* old_parameters, double* parameters,
        double* old_gradient, double* gradient,
        double* old_inverse_Hessian, double* auxiliar_matrix,
        double* inverse_Hessian_host, int n)
{
	double alpha;
	double beta;
	
	double parameters_dot_gradient;
	double gradient_dot_Hessian_dot_gradient;
	
	cublasHandle_t handle;
	
	createHandle(&handle);
	
	alpha = -1;
	
    cublasDaxpy(handle, n, &alpha, old_parameters, 1, parameters, 1);
    cublasDaxpy(handle, n, &alpha, old_gradient, 1, gradient, 1);
	
    cublasDdot(handle, n, parameters, 1, gradient, 1, &parameters_dot_gradient);
	
	alpha = 1;
	beta = 0;
	
	cublasDgemv(handle, CUBLAS_OP_N, n, n, &alpha, old_inverse_Hessian, n, gradient, 1, &beta, old_gradient, 1);
	
    cublasDdot(handle, n, gradient, 1, old_gradient, 1, &gradient_dot_Hessian_dot_gradient);
	
	alpha = 0;
	
    cublasDscal(handle, n*n, &alpha, auxiliar_matrix, 1);
	
	alpha = 1;
	
	cublasDger(handle, n, n, &alpha, parameters, 1, parameters, 1, auxiliar_matrix, n);
	
	alpha = 1/parameters_dot_gradient;
	
    cublasDscal(handle, n*n, &alpha, auxiliar_matrix, 1);
	
	alpha = 1;
	beta = 1;
	
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, old_inverse_Hessian, n,
                &beta, auxiliar_matrix, n, old_inverse_Hessian, n);
				
	alpha = 0;
	
    cublasDscal(handle, n*n, &alpha, auxiliar_matrix, 1);
	
	alpha = 1;
	
	cublasDger(handle, n, n, &alpha, old_gradient, 1, old_gradient, 1, auxiliar_matrix, n);
	
	alpha = 1;
    beta = -1/gradient_dot_Hessian_dot_gradient;

    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, old_inverse_Hessian, n,
                &beta, auxiliar_matrix, n, old_inverse_Hessian, n);
				
	alpha = 1/parameters_dot_gradient;
	beta = -1/gradient_dot_Hessian_dot_gradient;
	
    cublasDscal(handle, n, &alpha, parameters, 1);
	
    cublasDaxpy(handle, n, &beta, old_gradient, 1, parameters, 1);
	
	alpha = 0;
	
    cublasDscal(handle, n*n, &alpha, auxiliar_matrix, 1);
	
	cublasDger(handle, n, n, &gradient_dot_Hessian_dot_gradient, parameters, 1, parameters, 1, auxiliar_matrix, n);
	
	alpha = 1;
	beta = 1;
	
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, old_inverse_Hessian, n,
                &beta, auxiliar_matrix, n, old_inverse_Hessian, n);
				
    cudaMemcpy(inverse_Hessian_host, old_inverse_Hessian, n*n*sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void tanh_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = tanh(src[i]);
    }
}

__global__ void sigmoid_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = 1.0 / (1.0 + exp (-src[i]));
    }
}

__global__ void linear_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = src[i];
    }
}


void calculateActivation(double* vector_src, double* vector_dst, const int lenght, const std::string activation)
{
    /* Compute execution configuration */
    dim3 dimBlock(256);
    int threadBlocks = (lenght + (dimBlock.x - 1)) / dimBlock.x;
    if (threadBlocks > 65520) threadBlocks = 65520;
    dim3 dimGrid(threadBlocks);

    if(activation == "tanh")
    {
        tanh_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "sigmoid")
    {
        sigmoid_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "linear")
    {
        linear_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else
    {
        // error
    }

    cudaDeviceSynchronize();
}

__global__ void tanh_derivative_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        const double hyperbolic_tangent = tanh(src[i]);
        dst[i] = 1.0 - hyperbolic_tangent*hyperbolic_tangent;
    }
}

__global__ void sigmoid_derivative_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        const double exponential = exp(-src[i]);
        dst[i] = exponential/((1.0 + exponential)*(1.0 + exponential));
    }
}

__global__ void linear_derivative_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = 1.0;
    }
}

void calculateActivationDerivative(const double* vector_src, double* vector_dst, const int lenght, const std::string activation)
{
    /* Compute execution configuration */
    dim3 dimBlock(256);
    int threadBlocks = (lenght + (dimBlock.x - 1)) / dimBlock.x;
    if (threadBlocks > 65520) threadBlocks = 65520;
    dim3 dimGrid(threadBlocks);

    if(activation == "tanh")
    {
        tanh_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "sigmoid")
    {
        sigmoid_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "linear")
    {
        linear_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else
    {
        // error
    }

    cudaDeviceSynchronize();
}

__global__ void mean_squared_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        output_gradient[i] = (outputs[i] - targets[i])*2.0;
    }
}

void calculateOutputDerivative(const double* outputs, const double* targets, double* output_gradient, const int lenght, const std::string loss_method)
{
    /* Compute execution configuration */
    dim3 dimBlock(256);
    int threadBlocks = (lenght + (dimBlock.x - 1)) / dimBlock.x;
    if (threadBlocks > 65520) threadBlocks = 65520;
    dim3 dimGrid(threadBlocks);

    if(loss_method == "mean_squared_error" ||
       loss_method == "sum_squared_error")
    {
        mean_squared_error_derivative_kernel<<<dimGrid,dimBlock>>>(outputs, targets, output_gradient, lenght);
    }
    else
    {
        // error
    }

    cudaDeviceSynchronize();
}

__global__ void elementwise_multiplication_kernel (const double * vector1, const double* vector2, double* result, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        result[i] = vector1[i] * vector2[i];
    }
}

void elementwiseMultiplication(const double* vector1, const double* vector2, double* result, const int lenght)
{
    /* Compute execution configuration */
    dim3 dimBlock(256);
    int threadBlocks = (lenght + (dimBlock.x - 1)) / dimBlock.x;
    if (threadBlocks > 65520) threadBlocks = 65520;
    dim3 dimGrid(threadBlocks);

    elementwise_multiplication_kernel<<<dimGrid,dimBlock>>>(vector1, vector2, result, lenght);

    cudaDeviceSynchronize();
}

void calculateOutputsCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                          const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                          const double* input_data_h, const size_t input_rows, const size_t input_columns,
                          double* output_data_h, const size_t output_rows, const size_t output_columns,
                          const std::vector<std::string> layers_activations)
{
    const size_t layers_number = weights_d.size();

    double alpha, beta;

    double* layer_outputs;
    double* input_data_d;

    double* ones_vector;

    std::vector<double> ones_vector_h(input_rows, 1);
    const double* ones_vector_h_data = ones_vector_h.data();

    cublasHandle_t handle;
//    cublasXtHandle_t handleXt;

    createHandle(&handle);
//    cublasXtCreate(&handleXt);

    alpha = 1;
    beta = 1;

    // Initialize GPU auxiliar data

    cudaMalloc(&input_data_d, input_rows*input_columns*sizeof(double));
    cudaMemcpy(input_data_d, input_data_h, input_rows*input_columns*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&ones_vector, input_rows*sizeof(double));
    cudaMemcpy(ones_vector, ones_vector_h_data, input_rows*sizeof(double), cudaMemcpyHostToDevice);

    // First layer

    cudaMalloc(&layer_outputs, input_rows*weights_columns_numbers[0]*sizeof(double));
    cudaMemset(layer_outputs, 0, input_rows*weights_columns_numbers[0]*sizeof(double));

    cublasDger(handle, input_rows, weights_columns_numbers[0], &alpha, ones_vector, 1, biases_d[0], 1, layer_outputs, input_rows);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[0], input_columns, &alpha, input_data_d, input_rows,
                weights_d[0], input_columns, &beta, layer_outputs, input_rows);

//    cublasXtDgemm(handleXt, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[0], input_columns, &alpha, input_data_d, input_rows,
//                  weights_d[0], input_columns, &beta, layer_outputs, input_rows);

    calculateActivation(layer_outputs, layer_outputs, input_rows*weights_columns_numbers[0], layers_activations[0]);

    cudaFree(input_data_d);

    for(size_t i = 1; i < layers_number; i++)
    {
        double* input_layer;

        cudaMalloc(&input_layer, input_rows*weights_columns_numbers[i-1]*sizeof(double));
        cudaMemcpy(input_layer, layer_outputs, input_rows*weights_columns_numbers[i-1]*sizeof(double), cudaMemcpyDeviceToDevice);

        cudaFree(layer_outputs);

        cudaMalloc(&layer_outputs, input_rows*weights_columns_numbers[i]*sizeof(double));
        cudaMemset(layer_outputs, 0, input_rows*weights_columns_numbers[i]*sizeof(double));

        cublasDger(handle, input_rows, weights_columns_numbers[i], &alpha, ones_vector, 1, biases_d[i], 1, layer_outputs, input_rows);

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[i], weights_columns_numbers[i-1], &alpha, input_layer, input_rows,
                    weights_d[i], weights_columns_numbers[i-1], &beta, layer_outputs, input_rows);

//        cublasXtDgemm(handleXt, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[i], weights_columns_numbers[i-1], &alpha, input_layer, input_rows,
//                      weights_d[i], weights_columns_numbers[i-1], &beta, layer_outputs, input_rows);

        calculateActivation(layer_outputs, layer_outputs, input_rows*weights_columns_numbers[i], layers_activations[i]);

        cudaFree(input_layer);
    }

    // Copy to host memory

    cudaMemcpy(output_data_h, layer_outputs, output_rows*output_columns*sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory

    cudaFree(ones_vector);
    cudaFree(layer_outputs);
}

void calculateFirstOrderForwardPropagationCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                                               const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                                               const double* input_data_h, const size_t input_rows, const size_t input_columns,
                                               std::vector<double*> layers_activations_data, std::vector<double*> layers_activation_derivatives_data,
                                               const std::vector<size_t> activations_rows_numbers, const std::vector<size_t> activations_columns_numbers,
                                               const std::vector<std::string> layers_activations)
{
    const size_t layers_number = weights_d.size();

    double alpha, beta;

    double* layer_combinations;
    double* input_data_d;

    double* ones_vector;

    std::vector<double*> layers_activations_d(layers_number);
    std::vector<double*> layers_activation_derivatives_d(layers_number);

    std::vector<double> ones_vector_h(input_rows, 1);
    const double* ones_vector_h_data = ones_vector_h.data();

    cublasHandle_t handle;

    createHandle(&handle);

    alpha = 1;
    beta = 1;

    // Initialize GPU auxiliar data

    cudaMalloc(&input_data_d, input_rows*input_columns*sizeof(double));
    cudaMemcpy(input_data_d, input_data_h, input_rows*input_columns*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&ones_vector, input_rows*sizeof(double));
    cudaMemcpy(ones_vector, ones_vector_h_data, input_rows*sizeof(double), cudaMemcpyHostToDevice);

    for(size_t i = 0; i < layers_number; i++)
    {
        cudaMalloc(&layers_activations_d[i], activations_rows_numbers[i]*activations_columns_numbers[i]*sizeof(double));
        cudaMalloc(&layers_activation_derivatives_d[i], activations_rows_numbers[i]*activations_columns_numbers[i]*sizeof(double));
    }

    // First layer

    cudaMalloc(&layer_combinations, activations_rows_numbers[0]*activations_columns_numbers[0]*sizeof(double));
    cudaMemset(layer_combinations, 0, activations_rows_numbers[0]*activations_columns_numbers[0]*sizeof(double));

    cublasDger(handle, activations_rows_numbers[0], activations_columns_numbers[0], &alpha, ones_vector, 1, biases_d[0], 1, layer_combinations, input_rows);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[0], input_columns, &alpha, input_data_d, input_rows,
                weights_d[0], input_columns, &beta, layer_combinations, input_rows);

    calculateActivation(layer_combinations, layers_activations_d[0], activations_rows_numbers[0]*activations_columns_numbers[0], layers_activations[0]);
    calculateActivationDerivative(layer_combinations, layers_activation_derivatives_d[0], activations_rows_numbers[0]*activations_columns_numbers[0], layers_activations[0]);

    cudaFree(input_data_d);

    // Other layers

    for(size_t i = 1; i < layers_number; i++)
    {
        double* input_layer;

        cudaMalloc(&input_layer, activations_rows_numbers[i-1]*activations_columns_numbers[i-1]*sizeof(double));
        cudaMemcpy(input_layer, layers_activations_d[i-1], activations_rows_numbers[i-1]*activations_columns_numbers[i-1]*sizeof(double), cudaMemcpyDeviceToDevice);

        cudaFree(layer_combinations);

        cudaMalloc(&layer_combinations, activations_rows_numbers[i]*activations_columns_numbers[i]*sizeof(double));
        cudaMemset(layer_combinations, 0, activations_rows_numbers[i]*activations_columns_numbers[i]*sizeof(double));

        cublasDger(handle, activations_rows_numbers[i], activations_columns_numbers[i], &alpha, ones_vector, 1, biases_d[i], 1, layer_combinations, input_rows);

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[i], weights_columns_numbers[i-1], &alpha, input_layer, input_rows,
                    weights_d[i], weights_columns_numbers[i-1], &beta, layer_combinations, input_rows);

        calculateActivation(layer_combinations, layers_activations_d[i], activations_rows_numbers[i]*activations_columns_numbers[i], layers_activations[i]);
        calculateActivationDerivative(layer_combinations, layers_activation_derivatives_d[i], activations_rows_numbers[i]*activations_columns_numbers[i], layers_activations[i]);

        cudaFree(input_layer);
    }

    // Copy to host memory

    for(size_t i = 0; i < layers_number; i++)
    {
        cudaMemcpy(layers_activations_data[i], layers_activations_d[i], activations_rows_numbers[i]*activations_columns_numbers[i]*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(layers_activation_derivatives_data[i], layers_activation_derivatives_d[i], activations_rows_numbers[i]*activations_columns_numbers[i]*sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Free GPU memory

    cudaFree(ones_vector);
    cudaFree(layer_combinations);

    for(size_t i = 0; i < layers_number; i++)
    {
        cudaFree(layers_activations_d[i]);
        cudaFree(layers_activation_derivatives_d[i]);
    }
}

void calculateErrorGradientCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                                const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                                const double* input_data_h, const size_t input_rows, const size_t input_columns,
                                const double* target_data_h, const size_t target_rows, const size_t target_columns,
                                std::vector<double*> error_gradient_data,
                                const std::vector<std::string> layers_activations, const std::string loss_method)
{
    const size_t layers_number = weights_d.size();

    double alpha, beta;

    double* input_data_d;
    double* target_data_d;

    double* layer_combinations;
    double* output_gradient;

    double* ones_vector;

    std::vector<double*> layers_activations_d(layers_number);
    std::vector<double*> layers_activation_derivatives_d(layers_number);

    std::vector<double*> layers_delta_d(layers_number);

    std::vector<double*> error_gradient_d(2*layers_number);

    std::vector<double> ones_vector_h(input_rows, 1);
    const double* ones_vector_h_data = ones_vector_h.data();

    cublasHandle_t handle;

    createHandle(&handle);

    alpha = 1;
    beta = 1;

    // Initialize GPU data

    cudaMalloc(&input_data_d, input_rows*input_columns*sizeof(double));
    cudaMemcpy(input_data_d, input_data_h, input_rows*input_columns*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&target_data_d, target_rows*target_columns*sizeof(double));
    cudaMemcpy(target_data_d, target_data_h, target_rows*target_columns*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&output_gradient, target_rows*target_columns*sizeof(double));

    cudaMalloc(&ones_vector, input_rows*sizeof(double));
    cudaMemcpy(ones_vector, ones_vector_h_data, input_rows*sizeof(double), cudaMemcpyHostToDevice);

    for(size_t i = 0; i < layers_number; i++)
    {
        cudaMalloc(&layers_activations_d[i], input_rows*weights_columns_numbers[i]*sizeof(double));
        cudaMalloc(&layers_activation_derivatives_d[i], input_rows*weights_columns_numbers[i]*sizeof(double));

        cudaMalloc(&layers_delta_d[i], input_rows*weights_columns_numbers[i]*sizeof(double));

        cudaMalloc(&error_gradient_d[2*i], weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double));
        cudaMalloc(&error_gradient_d[2*i+1], bias_rows_numbers[i]*sizeof(double));
    }

    // CALCULATE FIRST ORDER FORWARD PROPAGATION

    // First layer

    cudaMalloc(&layer_combinations, input_rows*weights_columns_numbers[0]*sizeof(double));
    cudaMemset(layer_combinations, 0, input_rows*weights_columns_numbers[0]*sizeof(double));

    cublasDger(handle, input_rows, weights_columns_numbers[0], &alpha, ones_vector, 1, biases_d[0], 1, layer_combinations, input_rows);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[0], input_columns, &alpha, input_data_d, input_rows,
                weights_d[0], input_columns, &beta, layer_combinations, input_rows);

    calculateActivation(layer_combinations, layers_activations_d[0], input_rows*weights_columns_numbers[0], layers_activations[0]);
    calculateActivationDerivative(layer_combinations, layers_activation_derivatives_d[0], input_rows*weights_columns_numbers[0], layers_activations[0]);

    // Other layers

    for(size_t i = 1; i < layers_number; i++)
    {
        double* input_layer;

        cudaMalloc(&input_layer, input_rows*weights_columns_numbers[i-1]*sizeof(double));
        cudaMemcpy(input_layer, layers_activations_d[i-1], input_rows*weights_columns_numbers[i-1]*sizeof(double), cudaMemcpyDeviceToDevice);

        cudaFree(layer_combinations);

        cudaMalloc(&layer_combinations, input_rows*weights_columns_numbers[i]*sizeof(double));
        cudaMemset(layer_combinations, 0, input_rows*weights_columns_numbers[i]*sizeof(double));

        cublasDger(handle, input_rows, weights_columns_numbers[i], &alpha, ones_vector, 1, biases_d[i], 1, layer_combinations, input_rows);

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[i], weights_columns_numbers[i-1], &alpha, input_layer, input_rows,
                    weights_d[i], weights_columns_numbers[i-1], &beta, layer_combinations, input_rows);

        calculateActivation(layer_combinations, layers_activations_d[i], input_rows*weights_columns_numbers[i], layers_activations[i]);
        calculateActivationDerivative(layer_combinations, layers_activation_derivatives_d[i], input_rows*weights_columns_numbers[i], layers_activations[i]);

        cudaFree(input_layer);
    }

    // Free GPU memory

    cudaFree(layer_combinations);

    // CALCULATE OUTPUT DERIVATIVE

    calculateOutputDerivative(layers_activations_d[layers_number-1], target_data_d, output_gradient, target_rows*target_columns, loss_method);

    // CALCULATE LAYERS DELTA

    elementwiseMultiplication(layers_activation_derivatives_d[layers_number-1], output_gradient, layers_delta_d[layers_number-1], target_rows*target_columns);

    for(int i = (layers_number-2); i >= 0; i--)
    {
        double* auxiliar_matrix;

        cudaMalloc(&auxiliar_matrix, input_rows*weights_columns_numbers[i]*sizeof(double));
        cudaMemset(auxiliar_matrix, 0, input_rows*weights_columns_numbers[i]*sizeof(double));

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, input_rows, weights_columns_numbers[i], weights_columns_numbers[i+1], &alpha, layers_delta_d[i+1], input_rows,
                    weights_d[i+1], weights_columns_numbers[i], &beta, auxiliar_matrix, input_rows);

        elementwiseMultiplication(layers_activation_derivatives_d[i], auxiliar_matrix, layers_delta_d[i], input_rows*weights_columns_numbers[i]);

        cudaFree(auxiliar_matrix);
    }

    // CALCULATE ERROR GRADIENT

    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, input_columns, weights_columns_numbers[0], input_rows, &alpha, input_data_d, input_rows,
                layers_delta_d[0], input_rows, &beta, error_gradient_d[0], input_columns);

    for(size_t j = 0; j < weights_columns_numbers[0]; j++)
    {
        double sum;
        cublasDdot(handle, input_rows, layers_delta_d[0] + j*input_rows, 1, ones_vector, 1, &sum);// error_gradient_d[1]+j);
        cudaMemcpy(error_gradient_d[1]+j, &sum, sizeof(double), cudaMemcpyHostToDevice);
    }

    for(size_t i = 1; i < layers_number; i++)
    {
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, weights_columns_numbers[i-1], weights_columns_numbers[i], input_rows, &alpha, layers_activations_d[i-1], input_rows,
                    layers_delta_d[i], input_rows, &beta, error_gradient_d[2*i], weights_columns_numbers[i-1]);

        for(size_t j = 0; j < weights_columns_numbers[i]; j++)
        {
            double sum;
            cublasDdot(handle, input_rows, layers_delta_d[i] + j*input_rows, 1, ones_vector, 1, &sum);
            cudaMemcpy(error_gradient_d[2*i+1]+j, &sum, sizeof(double), cudaMemcpyHostToDevice);
        }
    }

    // Copy to host memory

    for(size_t i = 0; i < layers_number; i++)
    {
        cudaMemcpy(error_gradient_data[2*i], error_gradient_d[2*i], weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(error_gradient_data[2*i+1], error_gradient_d[2*i+1], bias_rows_numbers[i]*sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Free GPU memory

    cudaFree(ones_vector);
    cudaFree(input_data_d);
    cudaFree(target_data_d);
    cudaFree(output_gradient);

    for(size_t i = 0; i < layers_number; i++)
    {
        cudaFree(layers_activations_d[i]);
        cudaFree(layers_activation_derivatives_d[i]);

        cudaFree(layers_delta_d[i]);

        cudaFree(error_gradient_d[2*i]);
        cudaFree(error_gradient_d[2*i+1]);
    }
}
