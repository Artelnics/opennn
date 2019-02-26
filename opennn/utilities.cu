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
#include <algorithm>
#include <time.h>

using namespace std;

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
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

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

    cudaMemset(auxiliar_matrix, 0, n*n*sizeof(double));

    alpha = 1;

    cublasDger(handle, n, n, &alpha, parameters, 1, parameters, 1, auxiliar_matrix, n);

    alpha = 1/parameters_dot_gradient;

    cublasDscal(handle, n*n, &alpha, auxiliar_matrix, 1);

    alpha = 1;
    beta = 1;

    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, old_inverse_Hessian, n,
                &beta, auxiliar_matrix, n, old_inverse_Hessian, n);

    cudaMemset(auxiliar_matrix, 0, n*n*sizeof(double));

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

    cudaMemset(auxiliar_matrix, 0, n*n*sizeof(double));

    alpha = 1;

    cublasDger(handle, n, n, &alpha, parameters, 1, parameters, 1, auxiliar_matrix, n);

    alpha = 1/parameters_dot_gradient;

    cublasDscal(handle, n*n, &alpha, auxiliar_matrix, 1);

    alpha = 1;
    beta = 1;

    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, old_inverse_Hessian, n,
                &beta, auxiliar_matrix, n, old_inverse_Hessian, n);

    cudaMemset(auxiliar_matrix, 0, n*n*sizeof(double));

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

    cudaMemset(auxiliar_matrix, 0, n*n*sizeof(double));

    cublasDger(handle, n, n, &gradient_dot_Hessian_dot_gradient, parameters, 1, parameters, 1, auxiliar_matrix, n);

    alpha = 1;
    beta = 1;

    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &alpha, old_inverse_Hessian, n,
                &beta, auxiliar_matrix, n, old_inverse_Hessian, n);

    cudaMemcpy(inverse_Hessian_host, old_inverse_Hessian, n*n*sizeof(double), cudaMemcpyDeviceToHost);
}


__global__ void logistic_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    dst[id] = 1.0 / (1.0 + exp (-src[id]));
}

__global__ void hyperbolic_tangent_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    dst[id] = tanh(src[id]);
}

__global__ void threshold_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(src[id] < 0)
    {
        dst[id] = 0.0;
    }
    else
    {
        dst[id] = 1.0;
    }
}

__global__ void symmetric_threshold_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(src[id] < 0)
    {
        dst[id] = -1.0;
    }
    else
    {
        dst[id] = 1.0;
    }
}

__global__ void linear_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    dst[id] = src[id];
}

__global__ void rectified_linear_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    dst[id] = max(0.0, src[id]);
}

__global__ void scaled_exponential_linear_kernel (const double * src, double* dst, int len)
{
    const double lambda =1.0507;
    const double alpha =1.67326;

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    src[id] < 0.0 ? dst[id] = lambda * alpha * (exp(src[id]) - 1) : dst[id] = lambda * src[id];
}

__global__ void soft_plus_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    dst[id] = log(1 + exp(src[id]));
}

__global__ void soft_sign_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    src[id] < 0.0 ? dst[id] = src[id] / (1 - src[id]) : dst[id] = src[id] / (1 + src[id]);
}

__global__ void hard_logistic_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(src[id] < -2.5)
    {
        dst[id] = 0;
    }
    else if(src[id] > 2.5)
    {
        dst[id] = 1;
    }
    else
    {
        dst[id] = 0.2 * src[id] + 0.5;
    }
}

__global__ void exponential_linear_kernel (const double * src, double* dst, int len)
{
    const double alpha = 1.0;

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    src[id] < 0.0 ? dst[id] = alpha * (exp(src[id])- 1) : dst[id] = src[id];
}

void calculateActivation(double* vector_src, double* vector_dst, const int rows, const int columns, const string activation)
{
    const int length = rows*columns;

    if(activation == "Logistic")
    {
        logistic_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "HyperbolicTangent")
    {
       hyperbolic_tangent_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "Threshold")
    {
        threshold_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "SymmetricThreshold")
    {
        symmetric_threshold_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "Linear")
    {
        linear_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "RectifiedLinear")
    {
        rectified_linear_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "ScaledExponentialLinear")
    {
        scaled_exponential_linear_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "SoftPlus")
    {
        soft_plus_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "SoftSign")
    {
        soft_sign_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "Hardlogistic")
    {
        hard_logistic_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "ExponentialLinear")
    {
        exponential_linear_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }

    cudaDeviceSynchronize();
}

__global__ void logistic_derivative_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    const double exponential = exp(-src[id]);
    dst[id] = exponential/((1.0 + exponential)*(1.0 + exponential));
}

__global__ void hyperbolic_tangent_derivative_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    const double hyperbolic_tangent = tanh(src[id]);
    dst[id] = 1.0 - hyperbolic_tangent*hyperbolic_tangent;
}

__global__ void threshold_derivative_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(src[id] == 0)
    {
        // TODO: error
    }
    else
    {
        dst[id] = 0.0;
    }
}

__global__ void symmetric_threshold_derivative_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(src[id] == 0)
    {
        // TODO: error
    }
    else
    {
        dst[id] = 0.0;
    }
}

__global__ void linear_derivative_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    dst[id] = 1.0;
}

__global__ void rectified_linear_derivative_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    src[id] < 0.0 ? dst[id] = 0.0 : dst[id] = 1.0;
}

__global__ void scaled_exponential_linear_derivative_kernel (const double * src, double* dst, int len)
{
    const double lambda =1.0507;
    const double alpha =1.67326;

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    src[id] < 0.0 ? dst[id] = lambda * alpha * exp(src[id]) : dst[id] = lambda;
}

__global__ void soft_plus_derivative_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    dst[id] = 1/(1 + exp(-src[id]));
}

__global__ void soft_sign_derivative_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    src[id] < 0.0 ? dst[id] = 1 / pow((1 - src[id]), 2) : dst[id] = 1 / pow((1 + src[id]), 2);
}

__global__ void hard_logistic_derivative_kernel (const double * src, double* dst, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    src[id] < -2.5 || src[id] > 2.5 ? dst[id] = 0.0 : dst[id] = 0.2;
}

__global__ void exponential_linear_derivative_kernel (const double * src, double* dst, int len)
{
    const double alpha = 1.0;

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    src[id] < 0.0 ? dst[id] = alpha * exp(src[id]) : dst[id] = 1.0;
}

void calculateActivationDerivative(const double* vector_src, double* vector_dst, const int rows, const int columns, const string activation)
{
    const int length = rows*columns;

    if(activation == "Logistic")
    {
        logistic_derivative_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "HyperbolicTangent")
    {
       hyperbolic_tangent_derivative_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "Threshold")
    {
        threshold_derivative_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "SymmetricThreshold")
    {
        symmetric_threshold_derivative_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "Linear")
    {
        linear_derivative_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "RectifiedLinear")
    {
        rectified_linear_derivative_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "ScaledExponentialLinear")
    {
        scaled_exponential_linear_derivative_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "SoftPlus")
    {
        soft_plus_derivative_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "SoftSign")
    {
        soft_sign_derivative_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "Hardlogistic")
    {
        hard_logistic_derivative_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }
    else if(activation == "ExponentialLinear")
    {
        exponential_linear_derivative_kernel<<<rows,columns>>>(vector_src, vector_dst, length);
    }

    cudaDeviceSynchronize();
}

__global__ void mean_squared_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    output_gradient[id] = (outputs[id] - targets[id])*2.0;
}

__global__ void sum_squared_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    output_gradient[id] = (outputs[id] - targets[id])*2.0;
}

__global__ void cross_entropy_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    output_gradient[id] = (targets[id]/outputs[id])*(-1.0) + (targets[id]*(-1.0) + 1.0)/(outputs[id]*(-1.0) + 1.0);
}

__global__ void pow_p_error_kernel (const double * outputs, const double * targets, double* dst, int len, const double p)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    dst[id] = pow(outputs[id]-targets[id], p);
}

__global__ void minkowski_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len,
                                                   const double p, const double p_norm)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    const double error = outputs[id] - targets[id];
    output_gradient[id] = ((error) * pow(fabs(error), p - 2.0) / pow(p_norm, p - 1.0))/len;
}

__global__ void normalized_squared_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len,
                                                            const double normalization_coefficient)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    output_gradient[id] = (outputs[id] - targets[id])*2.0/normalization_coefficient;
}

__global__ void root_mean_squared_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len,
                                                           const double error)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    output_gradient[id] = (outputs[id] - targets[id]) / (len * error);
}

__global__ void weighted_squared_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len,
                                                          const double positives_weight, const double negatives_weight, const double normalization_coefficient)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    output_gradient[id] = (outputs[id]-targets[id])*(targets[id]*(negatives_weight/positives_weight-negatives_weight) + negatives_weight)*2.0/normalization_coefficient;
}

void calculateOutputDerivative(const double* outputs, const double* targets, double* output_gradient, const int rows, const int columns,
                               const string loss_method, const vector<double> loss_parameters)
{
    const int length = rows*columns;

    if(loss_method == "MEAN_SQUARED_ERROR")
    {
        mean_squared_error_derivative_kernel<<<rows,columns>>>(outputs, targets, output_gradient, length);
    }
    else if(loss_method == "SUM_SQUARED_ERROR")
    {
        sum_squared_error_derivative_kernel<<<rows,columns>>>(outputs, targets, output_gradient, length);
    }
    else if(loss_method == "CROSS_ENTROPY_ERROR")
    {
        cross_entropy_error_derivative_kernel<<<rows,columns>>>(outputs, targets, output_gradient, length);
    }
    else if(loss_method == "MINKOWSKI_ERROR")
    {
        // Calculate p norm
        double p_norm = 0;
        cublasHandle_t handle;

        createHandle(&handle);

        pow_p_error_kernel<<<rows,columns>>>(outputs, targets, output_gradient, length, loss_parameters[0]);
        cublasDasum(handle, length, output_gradient, 1, &p_norm);
        p_norm = pow(p_norm, 1/loss_parameters[0]);
        cublasDestroy(handle);

        minkowski_error_derivative_kernel<<<rows,columns>>>(outputs, targets, output_gradient, length, loss_parameters[0], p_norm);
    }
    else if(loss_method == "NORMALIZED_SQUARED_ERROR")
    {
        normalized_squared_error_derivative_kernel<<<rows,columns>>>(outputs, targets, output_gradient, length, loss_parameters[0]);
    }
    else if(loss_method == "ROOT_MEAN_SQUARED_ERROR")
    {
        // Calculate error
        double error;
        cublasHandle_t handle;

        createHandle(&handle);

        pow_p_error_kernel<<<rows,columns>>>(outputs, targets, output_gradient, length, 2);
        cublasDasum(handle, length, output_gradient, 1, &error);
        error = pow(error/length, 1/2);
        cublasDestroy(handle);

        root_mean_squared_error_derivative_kernel<<<rows,columns>>>(outputs, targets, output_gradient, length, error);
    }
    else if(loss_method == "WEIGHTED_SQUARED_ERROR")
    {
        weighted_squared_error_derivative_kernel<<<rows,columns>>>(outputs, targets, output_gradient, length,
                                                                       loss_parameters[0], loss_parameters[1], loss_parameters[2]);
    }
    else
    {
        // TODO: error
    }

    cudaDeviceSynchronize();
}

__global__ void elementwise_multiplication_kernel (const double * vector1, const double* vector2, double* result)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    result[id] = vector1[id] * vector2[id];
}

void elementwiseMultiplication(const double* vector1, const double* vector2, double* result, const int rows, const int columns)
{
    elementwise_multiplication_kernel<<<rows,columns>>>(vector1, vector2, result);

    cudaDeviceSynchronize();
}

void calculateOutputsCUDA(const vector<double*> weights_d, const vector<size_t> weights_rows_numbers, const vector<size_t> weights_columns_numbers,
                          const vector<double*> biases_d, const vector<size_t> bias_rows_numbers,
                          const double* input_data_h, const size_t input_rows, const size_t input_columns,
                          double* output_data_h, const size_t output_rows, const size_t output_columns,
                          const vector<string> layers_activations)
{
    const size_t layers_number = weights_d.size();

    double alpha, beta;

    double* layer_outputs;
    double* input_data_d;

    double* ones_vector;

    vector<double> ones_vector_h(input_rows, 1);
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
//    cudaDeviceSynchronize();

    calculateActivation(layer_outputs, layer_outputs, input_rows, weights_columns_numbers[0], layers_activations[0]);

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
//        cudaDeviceSynchronize();

        calculateActivation(layer_outputs, layer_outputs, input_rows, weights_columns_numbers[i], layers_activations[i]);

        cudaFree(input_layer);
    }

    // Copy to host memory

    cudaMemcpy(output_data_h, layer_outputs, output_rows*output_columns*sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory

    cudaFree(ones_vector);
    cudaFree(layer_outputs);

    cublasDestroy(handle);

    cudaDeviceSynchronize();
}

void calculateFirstOrderForwardPropagationCUDA(const vector<double*> weights_d, const vector<size_t> weights_rows_numbers, const vector<size_t> weights_columns_numbers,
                                               const vector<double*> biases_d, const vector<size_t> bias_rows_numbers,
                                               const double* input_data_h, const size_t input_rows, const size_t input_columns,
                                               vector<double*> layers_activations_data, vector<double*> layers_activation_derivatives_data,
                                               const vector<size_t> activations_rows_numbers, const vector<size_t> activations_columns_numbers,
                                               const vector<string> layers_activations)
{
    const size_t layers_number = weights_d.size();

    double alpha, beta;

    double* layer_combinations;
    double* input_data_d;

    double* ones_vector;

    vector<double*> layers_activations_d(layers_number);
    vector<double*> layers_activation_derivatives_d(layers_number);

    vector<double> ones_vector_h(input_rows, 1);
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

    calculateActivation(layer_combinations, layers_activations_d[0], activations_rows_numbers[0], activations_columns_numbers[0], layers_activations[0]);
    calculateActivationDerivative(layer_combinations, layers_activation_derivatives_d[0], activations_rows_numbers[0], activations_columns_numbers[0], layers_activations[0]);

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

        calculateActivation(layer_combinations, layers_activations_d[i], activations_rows_numbers[i], activations_columns_numbers[i], layers_activations[i]);
        calculateActivationDerivative(layer_combinations, layers_activation_derivatives_d[i], activations_rows_numbers[i], activations_columns_numbers[i], layers_activations[i]);

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

    cublasDestroy(handle);

    cudaDeviceSynchronize();
}

void calculateFirstOrderLossCUDA(const vector<double*> weights_d, const vector<size_t> weights_rows_numbers, const vector<size_t> weights_columns_numbers,
                                const vector<double*> biases_d, const vector<size_t> bias_rows_numbers,
                                const double* input_data_d, const size_t input_rows, const size_t input_columns,
                                const double* target_data_d, const size_t target_rows, const size_t target_columns,
                                vector<double*> error_gradient_data,
                                double* output_data_h, const size_t output_rows, const size_t output_columns,
                                const vector<string> layers_activations, const string loss_method,
                                const vector<double> loss_parameters)
{
    const size_t layers_number = weights_d.size();

    double alpha, beta;

//    double* input_data_d;
//    double* target_data_d;

    double* layer_combinations;
    double* output_gradient;

    double* ones_vector;

    vector<double*> layers_activations_d(layers_number);
    vector<double*> layers_activation_derivatives_d(layers_number);

    vector<double*> layers_delta_d(layers_number);

    vector<double*> error_gradient_d(2*layers_number);

    vector<double> ones_vector_h(input_rows, 1);
    const double* ones_vector_h_data = ones_vector_h.data();

    cublasHandle_t handle;
//    cublasXtHandle_t handleXt;

    createHandle(&handle);
//    cublasXtCreate(&handleXt);

//    const int nDevices = 1;
//    int deviceId[nDevices] = {0};
//    cublasXtDeviceSelect(handleXt, nDevices, deviceId);

    alpha = 1;
    beta = 1;

    // Initialize GPU data

//    cudaMalloc(&input_data_d, input_rows*input_columns*sizeof(double));
//    cudaMemcpyAsync(input_data_d, input_data_h, input_rows*input_columns*sizeof(double), cudaMemcpyHostToDevice);

//    cudaMalloc(&target_data_d, target_rows*target_columns*sizeof(double));
//    cudaMemcpyAsync(target_data_d, target_data_h, target_rows*target_columns*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&output_gradient, target_rows*target_columns*sizeof(double));

    cudaMalloc(&ones_vector, input_rows*sizeof(double));
    cudaMemcpyAsync(ones_vector, ones_vector_h_data, input_rows*sizeof(double), cudaMemcpyHostToDevice);

    for(size_t i = 0; i < layers_number; i++)
    {
        cudaMalloc(&layers_activations_d[i], input_rows*weights_columns_numbers[i]*sizeof(double));
        cudaMalloc(&layers_activation_derivatives_d[i], input_rows*weights_columns_numbers[i]*sizeof(double));

        cudaMalloc(&layers_delta_d[i], input_rows*weights_columns_numbers[i]*sizeof(double));

        cudaMalloc(&error_gradient_d[2*i], weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double));
        cudaMalloc(&error_gradient_d[2*i+1], weights_columns_numbers[i]*sizeof(double));
    }

    cudaDeviceSynchronize();

    // CALCULATE FIRST ORDER FORWARD PROPAGATION

    // First layer

    cudaMalloc(&layer_combinations, input_rows*weights_columns_numbers[0]*sizeof(double));
    cudaMemset(layer_combinations, 0, input_rows*weights_columns_numbers[0]*sizeof(double));

    cublasDger(handle, input_rows, weights_columns_numbers[0], &alpha, ones_vector, 1, biases_d[0], 1, layer_combinations, input_rows);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[0], input_columns, &alpha, input_data_d, input_rows,
                weights_d[0], input_columns, &beta, layer_combinations, input_rows);

//    cublasXtDgemm(handleXt, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[0], input_columns, &alpha, input_data_d, input_rows,
//                  weights_d[0], input_columns, &beta, layer_combinations, input_rows);

    calculateActivation(layer_combinations, layers_activations_d[0], input_rows, weights_columns_numbers[0], layers_activations[0]);
    calculateActivationDerivative(layer_combinations, layers_activation_derivatives_d[0], input_rows, weights_columns_numbers[0], layers_activations[0]);

    // Other layers

    for(size_t i = 1; i < layers_number; i++)
    {
        double* input_layer;

        cudaMalloc(&input_layer, input_rows*weights_columns_numbers[i-1]*sizeof(double));
        cudaMemcpyAsync(input_layer, layers_activations_d[i-1], input_rows*weights_columns_numbers[i-1]*sizeof(double), cudaMemcpyDeviceToDevice);

        cudaFree(layer_combinations);

        cudaMalloc(&layer_combinations, input_rows*weights_columns_numbers[i]*sizeof(double));
        cudaMemsetAsync(layer_combinations, 0, input_rows*weights_columns_numbers[i]*sizeof(double));

        cudaDeviceSynchronize();

        cublasDger(handle, input_rows, weights_columns_numbers[i], &alpha, ones_vector, 1, biases_d[i], 1, layer_combinations, input_rows);

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[i], weights_columns_numbers[i-1], &alpha, input_layer, input_rows,
                    weights_d[i], weights_columns_numbers[i-1], &beta, layer_combinations, input_rows);

//        cublasXtDgemm(handleXt, CUBLAS_OP_N, CUBLAS_OP_N, input_rows, weights_columns_numbers[i], weights_columns_numbers[i-1], &alpha, input_layer, input_rows,
//                      weights_d[i], weights_columns_numbers[i-1], &beta, layer_combinations, input_rows);

        calculateActivation(layer_combinations, layers_activations_d[i], input_rows, weights_columns_numbers[i], layers_activations[i]);
        calculateActivationDerivative(layer_combinations, layers_activation_derivatives_d[i], input_rows, weights_columns_numbers[i], layers_activations[i]);

        cudaFree(input_layer);
    }

    // Free GPU memory

    cudaFree(layer_combinations);

    // CALCULATE OUTPUT DERIVATIVE

    calculateOutputDerivative(layers_activations_d[layers_number-1], target_data_d, output_gradient, target_rows, target_columns, loss_method, loss_parameters);

    // CALCULATE LAYERS DELTA

    elementwiseMultiplication(layers_activation_derivatives_d[layers_number-1], output_gradient, layers_delta_d[layers_number-1], target_rows, target_columns);

    beta = 0;

    for(int i = (layers_number-2); i >= 0; i--)
    {
        double* auxiliar_matrix;

        cudaMalloc(&auxiliar_matrix, input_rows*weights_columns_numbers[i]*sizeof(double));

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, input_rows, weights_columns_numbers[i], weights_columns_numbers[i+1], &alpha, layers_delta_d[i+1], input_rows,
                    weights_d[i+1], weights_columns_numbers[i], &beta, auxiliar_matrix, input_rows);

        elementwiseMultiplication(layers_activation_derivatives_d[i], auxiliar_matrix, layers_delta_d[i], input_rows, weights_columns_numbers[i]);

        cudaFree(auxiliar_matrix);
    }

    // CALCULATE ERROR GRADIENT

    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, input_columns, weights_columns_numbers[0], input_rows, &alpha, input_data_d, input_rows,
                layers_delta_d[0], input_rows, &beta, error_gradient_d[0], input_columns);

    cublasDgemv(handle, CUBLAS_OP_T, input_rows, weights_columns_numbers[0], &alpha, layers_delta_d[0], input_rows,
                ones_vector, 1, &beta, error_gradient_d[1], 1);

    for(size_t i = 1; i < layers_number; i++)
    {
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, weights_columns_numbers[i-1], weights_columns_numbers[i], input_rows, &alpha, layers_activations_d[i-1], input_rows,
                    layers_delta_d[i], input_rows, &beta, error_gradient_d[2*i], weights_columns_numbers[i-1]);

        cublasDgemv(handle, CUBLAS_OP_T, input_rows, weights_columns_numbers[i], &alpha, layers_delta_d[i], input_rows,
                    ones_vector, 1, &beta, error_gradient_d[2*i+1], 1);
    }

    // LOSS OPERATIONS

    if(loss_method == "MEAN_SQUARED_ERROR")
    {
        alpha = 1/static_cast<double>(input_rows);

        for(size_t i = 0; i < layers_number; i++)
        {
            cublasDscal(handle, weights_rows_numbers[i]*weights_columns_numbers[i], &alpha, error_gradient_d[2*i], 1);
            cublasDscal(handle, weights_columns_numbers[i], &alpha, error_gradient_d[2*i+1], 1);
        }
    }
    else if(loss_method == "SUM_SQUARED_ERROR")
    {
        // Do nothing
    }
    else if(loss_method == "CROSS_ENTROPY_ERROR")
    {
        alpha = 1/static_cast<double>(input_rows);

        for(size_t i = 0; i < layers_number; i++)
        {
            cublasDscal(handle, weights_rows_numbers[i]*weights_columns_numbers[i], &alpha, error_gradient_d[2*i], 1);
            cublasDscal(handle, weights_columns_numbers[i], &alpha, error_gradient_d[2*i+1], 1);
        }
    }
    else if(loss_method == "MINKOWSKI_ERROR")
    {
        alpha = 1/static_cast<double>(input_rows);

        for(size_t i = 0; i < layers_number; i++)
        {
            cublasDscal(handle, weights_rows_numbers[i]*weights_columns_numbers[i], &alpha, error_gradient_d[2*i], 1);
            cublasDscal(handle, weights_columns_numbers[i], &alpha, error_gradient_d[2*i+1], 1);
        }
    }
    else if(loss_method == "NORMALIZED_SQUARED_ERROR")
    {
        alpha = 1/loss_parameters[0];

        for(size_t i = 0; i < layers_number; i++)
        {
            cublasDscal(handle, weights_rows_numbers[i]*weights_columns_numbers[i], &alpha, error_gradient_d[2*i], 1);
            cublasDscal(handle, weights_columns_numbers[i], &alpha, error_gradient_d[2*i+1], 1);
        }
    }
    else if(loss_method == "ROOT_MEAN_SQUARED_ERROR")
    {
        // TODO
    }
    else if(loss_method == "WEIGHTED_SQUARED_ERROR")
    {
        alpha = 1/loss_parameters[2];

        for(size_t i = 0; i < layers_number; i++)
        {
            cublasDscal(handle, weights_rows_numbers[i]*weights_columns_numbers[i], &alpha, error_gradient_d[2*i], 1);
            cublasDscal(handle, weights_columns_numbers[i], &alpha, error_gradient_d[2*i+1], 1);
        }
    }

    // Copy to host memory

    for(size_t i = 0; i < layers_number; i++)
    {
        cudaMemcpyAsync(error_gradient_data[2*i], error_gradient_d[2*i], weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(error_gradient_data[2*i+1], error_gradient_d[2*i+1], weights_columns_numbers[i]*sizeof(double), cudaMemcpyDeviceToHost);
    }

    cudaMemcpyAsync(output_data_h, layers_activations_d[layers_number-1], output_rows*output_columns*sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory

    cudaFree(ones_vector);
//    cudaFree(input_data_d);
//    cudaFree(target_data_d);
    cudaFree(output_gradient);

    for(size_t i = 0; i < layers_number; i++)
    {
        cudaFree(layers_activations_d[i]);
        cudaFree(layers_activation_derivatives_d[i]);

        cudaFree(layers_delta_d[i]);

        cudaFree(error_gradient_d[2*i]);
        cudaFree(error_gradient_d[2*i+1]);
    }

    cublasDestroy(handle);
//    cublasXtDestroy(handleXt);

    cudaDeviceSynchronize();
}

void updateParametersCUDA(vector<double*> weights_d, const vector<size_t> weights_rows_numbers, const vector<size_t> weights_columns_numbers,
                          vector<double*> biases_d, const vector<size_t> bias_rows_numbers,
                          const double* gradient_h, const size_t parameters_number)
{
    const size_t layers_number = weights_d.size();

    double alpha = 1;

    double* gradient_d;

    cublasHandle_t handle;

    createHandle(&handle);

    cudaMalloc(&gradient_d, parameters_number*sizeof(double));
    cudaMemcpy(gradient_d, gradient_h, parameters_number*sizeof(double), cudaMemcpyHostToDevice);

    size_t index = 0;

    for(size_t i = 0; i < layers_number; i++)
    {
        cublasDaxpy(handle, weights_rows_numbers[i]*weights_columns_numbers[i], &alpha,
                    gradient_d + index, 1, weights_d[i], 1);

        index += weights_rows_numbers[i]*weights_columns_numbers[i];

        cublasDaxpy(handle, bias_rows_numbers[i], &alpha,
                    gradient_d + index, 1, biases_d[i], 1);

        index += bias_rows_numbers[i];
    }

    cudaFree(gradient_d);

    cublasDestroy(handle);

    cudaDeviceSynchronize();
}

void updateParametersSgdCUDA(std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                             std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                             const double* gradient_d, const size_t parameters_number,
                             const double& momentum, const bool& nesterov, const double& initial_learning_rate,
                             const double& initial_decay, const size_t& learning_rate_iteration, double*& last_increment)
{
    const size_t layers_number = weights_d.size();

    const double learning_rate =  initial_learning_rate * (1.0 / (1.0 + learning_rate_iteration*initial_decay));

    double* parameters_increment;

    double alpha;

    cublasHandle_t handle;

    createHandle(&handle);

    cudaMalloc(&parameters_increment, parameters_number*sizeof(double));
    cudaMemcpy(parameters_increment, gradient_d, parameters_number*sizeof(double), cudaMemcpyDeviceToDevice);

    alpha = -learning_rate;
    cublasDscal(handle, parameters_number, &alpha, parameters_increment, 1);

    if(momentum > 0.0 && !nesterov)
    {
        cublasDaxpy(handle, parameters_number, &momentum,
                    last_increment, 1, parameters_increment, 1);

        cudaMemcpy(last_increment, parameters_increment, parameters_number*sizeof(double), cudaMemcpyDeviceToDevice);
    }
    else if(momentum > 0.0 && nesterov)
    {
        cublasDaxpy(handle, parameters_number, &momentum,
                    last_increment, 1, parameters_increment, 1);

        cudaMemcpy(last_increment, parameters_increment, parameters_number*sizeof(double), cudaMemcpyDeviceToDevice);

        cublasDscal(handle, parameters_number, &momentum, parameters_increment, 1);

        alpha = -learning_rate;
        cublasDaxpy(handle, parameters_number, &alpha,
                    gradient_d, 1, parameters_increment, 1);
    }

    size_t index = 0;

    alpha = 1;

    for(size_t i = 0; i < layers_number; i++)
    {
        cublasDaxpy(handle, weights_rows_numbers[i]*weights_columns_numbers[i], &alpha,
                    parameters_increment + index, 1, weights_d[i], 1);

        index += weights_rows_numbers[i]*weights_columns_numbers[i];

        cublasDaxpy(handle, bias_rows_numbers[i], &alpha,
                    parameters_increment + index, 1, biases_d[i], 1);

        index += bias_rows_numbers[i];
    }

    cudaFree(parameters_increment);

    cublasDestroy(handle);

    cudaDeviceSynchronize();
}
