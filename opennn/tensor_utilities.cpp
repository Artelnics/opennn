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

string shape_to_string(const Shape& shape, const string& separator)
{
    const Index size = shape.rank;

    ostringstream buffer;

    if (size == 0)
        throw runtime_error("Dimensions size must be greater than 0.\n");

    for (Index i = 0; i < size; ++i)
        buffer << shape[i] << separator;

    return buffer.str();
}

Shape string_to_shape(const string& text, const string& separator)
{
    Shape result;

    if (text.empty())
        throw runtime_error("Input string must not be empty.\n");

    stringstream stream(text);
    string token;

    while (getline(stream, token, separator[0]))
    {
        try
        {
            if (!token.empty())
                result.push_back(stoi(token));
        }
        catch (const invalid_argument&)
        {
            throw runtime_error("Input string contains non-numeric elements.\n");
        }
    }

    return result;
}

Backend::Backend()
{
    set_threads_number(0);

#ifdef OPENNN_WITH_CUDA
    CHECK_CUDA(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));
    CHECK_CUBLAS(cublasLtCreate(&cublas_lt_handle));
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, compute_stream));

    // OpTensor compute must be FP32 when input tensors are FP16/BF16 (cuDNN does not implement BF16/FP16 compute).
    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&operator_sum_descriptor));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(operator_sum_descriptor, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
#endif
}

Backend::~Backend()
{
#ifdef OPENNN_WITH_CUDA
    if (operator_sum_descriptor) cudnnDestroyOpTensorDescriptor(operator_sum_descriptor);
    if (cublas_lt_handle) cublasLtDestroy(cublas_lt_handle);
    if (cublas_handle) cublasDestroy(cublas_handle);
    if (cudnn_handle) cudnnDestroy(cudnn_handle);
    if (compute_stream) cudaStreamDestroy(compute_stream);
#endif
}

void Backend::set_threads_number(int num_threads)
{
    if (num_threads <= 0)
    {
        num_threads = thread::hardware_concurrency();
        if (num_threads > 1) num_threads /= 2;
        if (num_threads <= 0) num_threads = omp_get_max_threads();
        if (num_threads <= 0) num_threads = 1;
    }

    thread_pool = make_unique<ThreadPool>(num_threads);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), num_threads);

    Eigen::initParallel();
    Eigen::setNbThreads(num_threads);
    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);
}

Backend& Backend::instance()
{
    static Backend device;
    return device;
}

ThreadPoolDevice* Backend::get_thread_pool_device()
{
    return thread_pool_device.get();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
