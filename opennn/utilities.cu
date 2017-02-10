#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include<stdio.h>


void printVector(const double* vector, int n)
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
	cudaMalloc(NULL, 0);
}

int mallocCUDA(double** A_d, int nBytes)
{
    cudaError_t error;

    error = cudaMalloc(A_d, nBytes);

    if (error != cudaSuccess)
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

    if (error != cudaSuccess)
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
