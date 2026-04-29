//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R S   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "tensor_utilities.h"

#include <Eigen/Dense>

namespace opennn
{

string shape_to_string(const Shape& x, const string& separator)
{
    const Index size = x.size();

    ostringstream buffer;

    if(size == 0)
        throw runtime_error("Error: Dimensions size must be greater than 0.\n");

    for(Index i = 0; i < size; ++i)
        buffer << x[i] << separator;

    return buffer.str();
}

Shape string_to_shape(const string& x, const string& separator)
{
    Shape result;

    if (x.empty())
        throw runtime_error("Error: Input string must not be empty.\n");

    stringstream ss(x);
    string token;

    while (getline(ss, token, separator[0]))
    {
        try
        {
            if(!token.empty())
                result.push_back(stoi(token));
        }
        catch (const invalid_argument&)
        {
            throw runtime_error("Error: Input string contains non-numeric elements.\n");
        }
    }

    return result;
}

Device::Device()
{
    set_threads_number(0);

#ifdef OPENNN_WITH_CUDA
    CHECK_CUDA(cudaStreamCreate(&compute_stream));

    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));
    CHECK_CUBLAS(cublasLtCreate(&cublas_lt_handle));
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, compute_stream));

    // Compute type for OpTensor is the precision of the elementwise op, which
    // must be FP32 when input tensors are FP16/BF16 (cuDNN does not implement
    // BF16/FP16 compute for these ops). Tensor data type is set per-call via
    // the tensor descriptors, so this stays FP32 regardless of activation dtype.
    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&operator_sum_descriptor));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(operator_sum_descriptor, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&operator_multiplication_descriptor));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(operator_multiplication_descriptor, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
#endif
}

Device::~Device()
{
#ifdef OPENNN_WITH_CUDA
    // The cuBLASLt plan cache, workspace and BF16 scratch are TU-local in
    // cuda_gemm.cpp now and have process lifetime; nothing to free here.
    if (operator_sum_descriptor) cudnnDestroyOpTensorDescriptor(operator_sum_descriptor);
    if (operator_multiplication_descriptor) cudnnDestroyOpTensorDescriptor(operator_multiplication_descriptor);
    if (cublas_lt_handle) cublasLtDestroy(cublas_lt_handle);
    if (cublas_handle) cublasDestroy(cublas_handle);
    if (cudnn_handle) cudnnDestroy(cudnn_handle);
    if (compute_stream) cudaStreamDestroy(compute_stream);
#endif
}

// get_lt_gemm_plan implementation moved to cuda_gemm.cpp.

void Device::set_threads_number(int num_threads)
{
    if (num_threads <= 0)
    {
        num_threads = thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = omp_get_max_threads();
        if (num_threads <= 0) num_threads = 1;
    }

    thread_pool = make_unique<ThreadPool>(num_threads);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), num_threads);

    omp_set_num_threads(num_threads);
}

Device& Device::instance()
{
    static Device device;
    return device;
}

// Configuration impls moved to configuration.cpp.

ThreadPoolDevice* Device::get_thread_pool_device()
{
    return thread_pool_device.get();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
