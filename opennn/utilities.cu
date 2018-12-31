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


__global__ void logistic_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = 1.0 / (1.0 + exp (-src[i]));
    }
}

__global__ void hyperbolic_tangent_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = tanh(src[i]);
    }
}

__global__ void threshold_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        if(src[i] < 0)
        {
            dst[i] = 0.0;
        }
        else
        {
            dst[i] = 1.0;
        }
    }
}

__global__ void symmetric_threshold_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        if(src[i] < 0)
        {
            dst[i] = -1.0;
        }
        else
        {
            dst[i] = 1.0;
        }
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

__global__ void rectified_linear_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        src[i] < 0.0 ? dst[i] = 0.0 : dst[i] = src[i];
    }
}

__global__ void scaled_exponential_linear_kernel (const double * src, double* dst, int len)
{
    const double lambda =1.0507;
    const double alpha =1.67326;

    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        src[i] < 0.0 ? dst[i] = lambda * alpha * (exp(src[i]) - 1) : dst[i] = lambda * src[i];
    }
}

__global__ void soft_plus_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = log(1 + exp(src[i]));
    }
}

__global__ void soft_sign_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        src[i] < 0.0 ? dst[i] = src[i] / (1 - src[i]) : dst[i] = src[i] / (1 + src[i]);
    }
}

__global__ void hard_logistic_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        if(src[i] < -2.5)
        {
           dst[i] = 0;
        }
        else if(src[i] > 2.5)
        {
            dst[i] = 1;
        }
        else
        {
            dst[i] = 0.2 * src[i] + 0.5;
        }
    }
}

__global__ void exponential_linear_kernel (const double * src, double* dst, int len)
{
    const double alpha = 1.0;

    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        src[i] < 0.0 ? dst[i] = alpha * (exp(src[i])- 1) : dst[i] = src[i];
    }
}

void calculateActivation(double* vector_src, double* vector_dst, const int lenght, const std::string activation)
{
    /* Compute execution configuration */
    dim3 dimBlock(256);
    int threadBlocks = (lenght + (dimBlock.x - 1)) / dimBlock.x;
    if (threadBlocks > 65520) threadBlocks = 65520;
    dim3 dimGrid(threadBlocks);

    if(activation == "Logistic")
    {
        logistic_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "HyperbolicTangent")
    {
       hyperbolic_tangent_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "Threshold")
    {
        threshold_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "SymmetricThreshold")
    {
        symmetric_threshold_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "Linear")
    {
        linear_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "RectifiedLinear")
    {
        rectified_linear_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "ScaledExponentialLinear")
    {
        scaled_exponential_linear_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "SoftPlus")
    {
        soft_plus_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "SoftSign")
    {
        soft_sign_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "Hardlogistic")
    {
        hard_logistic_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "ExponentialLinear")
    {
        exponential_linear_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }

    cudaDeviceSynchronize();
}

__global__ void logistic_derivative_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        const double exponential = exp(-src[i]);
        dst[i] = exponential/((1.0 + exponential)*(1.0 + exponential));
    }
}

__global__ void hyperbolic_tangent_derivative_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        const double hyperbolic_tangent = tanh(src[i]);
        dst[i] = 1.0 - hyperbolic_tangent*hyperbolic_tangent;
    }
}

__global__ void threshold_derivative_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        if(src[i] == 0)
        {
            // TODO: error
        }
        else
        {
            dst[i] = 0.0;
        }
    }
}

__global__ void symmetric_threshold_derivative_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        if(src[i] == 0)
        {
            // TODO: error
        }
        else
        {
            dst[i] = 0.0;
        }
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

__global__ void rectified_linear_derivative_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        src[i] < 0.0 ? dst[i] = 0.0 : dst[i] = 1.0;
    }
}

__global__ void scaled_exponential_linear_derivative_kernel (const double * src, double* dst, int len)
{
    const double lambda =1.0507;
    const double alpha =1.67326;

    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        src[i] < 0.0 ? dst[i] = lambda * alpha * exp(src[i]) : dst[i] = lambda;
    }
}

__global__ void soft_plus_derivative_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = 1/(1 + exp(-src[i]));
    }
}

__global__ void soft_sign_derivative_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        src[i] < 0.0 ? dst[i] = 1 / pow((1 - src[i]), 2) : dst[i] = 1 / pow((1 + src[i]), 2);
    }
}

__global__ void hard_logistic_derivative_kernel (const double * src, double* dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        src[i] < -2.5 || src[i] > 2.5 ? dst[i] = 0.0 : dst[i] = 0.2;
    }
}

__global__ void exponential_linear_derivative_kernel (const double * src, double* dst, int len)
{
    const double alpha = 1.0;

    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        src[i] < 0.0 ? dst[i] = alpha * exp(src[i]) : dst[i] = 1.0;
    }
}

void calculateActivationDerivative(const double* vector_src, double* vector_dst, const int lenght, const std::string activation)
{
    /* Compute execution configuration */
    dim3 dimBlock(256);
    int threadBlocks = (lenght + (dimBlock.x - 1)) / dimBlock.x;
    if (threadBlocks > 65520) threadBlocks = 65520;
    dim3 dimGrid(threadBlocks);

    if(activation == "Logistic")
    {
        logistic_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "HyperbolicTangent")
    {
       hyperbolic_tangent_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "Threshold")
    {
        threshold_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "SymmetricThreshold")
    {
        symmetric_threshold_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "Linear")
    {
        linear_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "RectifiedLinear")
    {
        rectified_linear_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "ScaledExponentialLinear")
    {
        scaled_exponential_linear_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "SoftPlus")
    {
        soft_plus_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "SoftSign")
    {
        soft_sign_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "Hardlogistic")
    {
        hard_logistic_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
    }
    else if(activation == "ExponentialLinear")
    {
        exponential_linear_derivative_kernel<<<dimGrid,dimBlock>>>(vector_src, vector_dst, lenght);
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

__global__ void sum_squared_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        output_gradient[i] = (outputs[i] - targets[i])*2.0;
    }
}

__global__ void cross_entropy_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        output_gradient[i] = (targets[i]/outputs[i])*(-1.0) + (targets[i]*(-1.0) + 1.0)/(outputs[i]*(-1.0) + 1.0);
    }
}

__global__ void minkowski_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len,
                                                   const double minkowski_parameter)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        // TODO
    }
}

__global__ void normalized_squared_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len,
                                                            const double normalization_coefficient)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        output_gradient[i] = (outputs[i] - targets[i])*2.0/normalization_coefficient;
    }
}

__global__ void root_mean_squared_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len,
                                                           const double error)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        // TODO
    }
}

__global__ void weighted_squared_error_derivative_kernel (const double * outputs, const double* targets, double* output_gradient, int len,
                                                          const double positives_weight, const double negatives_weight, const double normalization_coefficient)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        output_gradient[i] = (outputs[i]-targets[i])*(targets[i]*(negatives_weight/positives_weight-negatives_weight) + negatives_weight)*2.0/normalization_coefficient;
    }
}

void calculateOutputDerivative(const double* outputs, const double* targets, double* output_gradient, const int lenght,
                               const std::string loss_method, const std::vector<double> loss_parameters)
{
    /* Compute execution configuration */
    dim3 dimBlock(256);
    int threadBlocks = (lenght + (dimBlock.x - 1)) / dimBlock.x;
    if (threadBlocks > 65520) threadBlocks = 65520;
    dim3 dimGrid(threadBlocks);

    if(loss_method == "MEAN_SQUARED_ERROR")
    {
        mean_squared_error_derivative_kernel<<<dimGrid,dimBlock>>>(outputs, targets, output_gradient, lenght);
    }
    else if(loss_method == "SUM_SQUARED_ERROR")
    {
        sum_squared_error_derivative_kernel<<<dimGrid,dimBlock>>>(outputs, targets, output_gradient, lenght);
    }
    else if(loss_method == "CROSS_ENTROPY_ERROR")
    {
        cross_entropy_error_derivative_kernel<<<dimGrid,dimBlock>>>(outputs, targets, output_gradient, lenght);
    }
    else if(loss_method == "MINKOWSKI_ERROR")
    {
        // TODO
    }
    else if(loss_method == "NORMALIZED_SQUARED_ERROR")
    {
        normalized_squared_error_derivative_kernel<<<dimGrid,dimBlock>>>(outputs, targets, output_gradient, lenght, loss_parameters[0]);
    }
    else if(loss_method == "ROOT_MEAN_SQUARED_ERROR")
    {
        // TODO
    }
    else if(loss_method == "WEIGHTED_SQUARED_ERROR")
    {
        weighted_squared_error_derivative_kernel<<<dimGrid,dimBlock>>>(outputs, targets, output_gradient, lenght,
                                                                       loss_parameters[0], loss_parameters[1], loss_parameters[2]);
    }
    else
    {
        // TODO: error
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
//    cudaDeviceSynchronize();

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
//        cudaDeviceSynchronize();

        calculateActivation(layer_outputs, layer_outputs, input_rows*weights_columns_numbers[i], layers_activations[i]);

        cudaFree(input_layer);
    }

    // Copy to host memory

    cudaMemcpy(output_data_h, layer_outputs, output_rows*output_columns*sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory

    cudaFree(ones_vector);
    cudaFree(layer_outputs);

    cublasDestroy(handle);
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

    cublasDestroy(handle);
}

void calculateErrorGradientCUDA(const std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                                const std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
                                const double* input_data_h, const size_t input_rows, const size_t input_columns,
                                const double* target_data_h, const size_t target_rows, const size_t target_columns,
                                std::vector<double*> error_gradient_data,
                                const std::vector<std::string> layers_activations, const std::string loss_method,
                                const std::vector<double> loss_parameters)
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

        cudaMemset(error_gradient_d[2*i], 0, weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double));
        cudaMemset(error_gradient_d[2*i+1], 0, bias_rows_numbers[i]*sizeof(double));
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

    calculateOutputDerivative(layers_activations_d[layers_number-1], target_data_d, output_gradient, target_rows*target_columns, loss_method, loss_parameters);

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
        cublasDdot(handle, input_rows, layers_delta_d[0] + j*input_rows, 1, ones_vector, 1, error_gradient_data[1]+j);
    }

    for(size_t i = 1; i < layers_number; i++)
    {
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, weights_columns_numbers[i-1], weights_columns_numbers[i], input_rows, &alpha, layers_activations_d[i-1], input_rows,
                    layers_delta_d[i], input_rows, &beta, error_gradient_d[2*i], weights_columns_numbers[i-1]);

        for(size_t j = 0; j < weights_columns_numbers[i]; j++)
        {
            cublasDdot(handle, input_rows, layers_delta_d[i] + j*input_rows, 1, ones_vector, 1, error_gradient_data[2*i+1]+j);
        }
    }

    // LOSS OPERATIONS

    if(loss_method == "MINKOWSKI_ERROR")
    {
        loss_parameters[0]; // Minkowski parameter
    }
    else if(loss_method == "NORMALIZED_SQUARED_ERROR")
    {
        loss_parameters[0]; // Normalization coefficient
    }
    else if(loss_method == "WEIGHTED_SQUARED_ERROR")
    {
        loss_parameters[0]; // Positives weights
        loss_parameters[1]; // Negatives weights
        loss_parameters[2]; // Normalization coefficient
    }

    // Copy to host memory

    for(size_t i = 0; i < layers_number; i++)
    {
        cudaMemcpy(error_gradient_data[2*i], error_gradient_d[2*i], weights_rows_numbers[i]*weights_columns_numbers[i]*sizeof(double), cudaMemcpyDeviceToHost);
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

    cublasDestroy(handle);
}

void updateParametersCUDA(std::vector<double*> weights_d, const std::vector<size_t> weights_rows_numbers, const std::vector<size_t> weights_columns_numbers,
                          std::vector<double*> biases_d, const std::vector<size_t> bias_rows_numbers,
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
}
