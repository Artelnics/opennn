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

const EnumMap<ActivationFunction>& activation_function_map()
{
    static const vector<pair<ActivationFunction, string>> entries = {
        {ActivationFunction::Identity, "Identity"},
        {ActivationFunction::Sigmoid,  "Sigmoid"},
        {ActivationFunction::Tanh,     "Tanh"},
        {ActivationFunction::ReLU,     "ReLU"},
        {ActivationFunction::Softmax,  "Softmax"}
    };
    static const EnumMap<ActivationFunction> instance{entries};
    return instance;
}

const string& activation_function_to_string(ActivationFunction function)
{
    return activation_function_map().to_string(function);
}

ActivationFunction activation_function_from_string(const string& name)
{
    return activation_function_map().from_string(name, ActivationFunction::Identity);
}

string shape_to_string(const Shape& shape, const string& separator)
{
    const Index size = shape.rank;

    ostringstream buffer;

    throw_if(size == 0,
             "Dimensions size must be greater than 0.\n");

    for (Index i = 0; i < size; ++i)
        buffer << shape[i] << separator;

    return buffer.str();
}

Shape string_to_shape(const string& text, const string& separator)
{
    Shape result;

    throw_if(text.empty(),
             "Input string must not be empty.\n");

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

void fill_tensor_data(const MatrixR& matrix,
                      const vector<Index>& row_indices,
                      const vector<Index>& column_indices,
                      float* __restrict tensor_data,
                      int contiguous_hint)
{
    const Index rows_number = row_indices.size();
    const Index columns_number = column_indices.size();

    if (rows_number == 0 || columns_number == 0) return;

    const float* matrix_data = matrix.data();

    const Index matrix_cols_number = matrix.cols();

    const bool contiguous = (contiguous_hint >= 0) ? static_cast<bool>(contiguous_hint) : is_contiguous(column_indices);

    if (contiguous)
    {
        #pragma omp parallel for schedule(static)
        for (Index i = 0; i < rows_number; ++i)
            memcpy(tensor_data + i * columns_number, &matrix(row_indices[i], column_indices[0]), static_cast<size_t>(columns_number) * sizeof(float));
    }
    else
    {
        #pragma omp parallel for schedule(static)
        for (Index i = 0; i < rows_number; ++i)
        {
            const Index src_row = row_indices[i];
            const float* src_row_ptr = matrix_data + src_row * matrix_cols_number;
            float* dest_row_ptr = tensor_data + i * columns_number;

            for (Index j = 0; j < columns_number; ++j)
                dest_row_ptr[j] = src_row_ptr[column_indices[j]];
        }
    }
}

Backend::Backend()
{
    set_threads_number(0);

#ifdef OPENNN_HAS_CUDA
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        cerr << "OpenNN: no CUDA device available (" << cudaGetErrorString(status)
             << "); running on CPU.\n";
        return;
    }

    CHECK_CUDA(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));
    CHECK_CUBLAS(cublasLtCreate(&cublas_lt_handle));
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, compute_stream));

    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&operator_sum_descriptor));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(operator_sum_descriptor, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
#endif
}

Backend::~Backend()
{
#ifdef OPENNN_HAS_CUDA
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
        if (num_threads <= 0) num_threads = omp_get_max_threads();
        if (num_threads <= 0) num_threads = 1;
    }

    thread_pool = make_unique<ThreadPool>(num_threads);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), num_threads);

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

#ifdef OPENNN_HAS_CUDA

void copy_device_to_host_float(const void* device_src, Type src_dtype,
                               Index element_count, float* host_dst,
                               cudaStream_t stream)
{
    if (element_count == 0) return;

    if (src_dtype == Type::FP32)
    {
        CHECK_CUDA(cudaMemcpyAsync(host_dst, device_src,
                                   element_count * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        return;
    }

    if (src_dtype == Type::BF16)
    {
        vector<uint16_t> staging(static_cast<size_t>(element_count));
        CHECK_CUDA(cudaMemcpyAsync(staging.data(), device_src,
                                   element_count * sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        for (Index i = 0; i < element_count; ++i)
        {
            const uint32_t bits = static_cast<uint32_t>(staging[size_t(i)]) << 16;
            memcpy(&host_dst[i], &bits, sizeof(float));
        }
        return;
    }

    throw runtime_error("copy_device_to_host_float: unsupported dtype.");
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
