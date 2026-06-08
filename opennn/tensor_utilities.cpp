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

#ifdef OPENNN_HAS_CUDA

cudnnTensorDescriptor_t TensorView::get_descriptor() const
{
    if (!descriptor_handle && !shape.empty())
        set_descriptor(shape);
    return descriptor_handle.get();
}

void TensorView::set_descriptor(const Shape& descriptor_shape) const
{
    // NHWC layout: rank < 4 leading dims default to 1.
    int batch_count = 1, channels = 1, height = 1, width = 1;
    const size_t rank = descriptor_shape.rank;
    if (rank >= 1) channels    = static_cast<int>(descriptor_shape[rank - 1]);
    if (rank >= 2) batch_count = static_cast<int>(descriptor_shape[0]);
    if (rank >= 3) width       = static_cast<int>(descriptor_shape[rank - 2]);
    if (rank >= 4) height      = static_cast<int>(descriptor_shape[rank - 3]);

    if (batch_count <= 0 || channels <= 0 || height <= 0 || width <= 0)
        return;

    if (!descriptor_handle)
    {
        cudnnTensorDescriptor_t raw_desc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&raw_desc));

        descriptor_handle = shared_ptr<cudnnTensorStruct>(raw_desc, [](cudnnTensorDescriptor_t descriptor) {
            cudnnDestroyTensorDescriptor(descriptor);
        });
    }

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor_handle.get(), CUDNN_TENSOR_NHWC, to_cudnn(type),
                                           batch_count, channels, height, width));
}

static bool uses_cuda_fill(const TensorView& view)
{
    cudaPointerAttributes attr{};
    const cudaError_t err = cudaPointerGetAttributes(&attr, view.data);
    const bool gpu_data = (err == cudaSuccess) && (attr.type == cudaMemoryTypeDevice);
    if (err != cudaSuccess) cudaGetLastError();
    return gpu_data;
}

static void fill_cuda(const TensorView& view, float value)
{
    if (value == 0.0f)
    {
        device::set_zero(view.data, view.byte_size(), Device::CUDA);
        return;
    }

    CHECK_CUDNN(cudnnSetTensor(Backend::get_cudnn_handle(),
                               view.get_descriptor(), view.data, &value));
}

static void initialize_cuda_backend(cudaStream_t& compute_stream,
                                    cublasHandle_t& cublas_handle,
                                    cublasLtHandle_t& cublas_lt_handle,
                                    cudnnHandle_t& cudnn_handle,
                                    cudnnOpTensorDescriptor_t& operator_sum_descriptor)
{
    if (!device::has_cuda_device())
    {
        cerr << "OpenNN: no CUDA device available; running on CPU.\n";
        return;
    }

    compute_stream = device::create_stream(cudaStreamNonBlocking);

    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));
    CHECK_CUBLAS(cublasLtCreate(&cublas_lt_handle));
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, compute_stream));

    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&operator_sum_descriptor));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(operator_sum_descriptor,
                                           CUDNN_OP_TENSOR_ADD,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_NOT_PROPAGATE_NAN));
}

static void destroy_cuda_backend(cudaStream_t& compute_stream,
                                 cublasHandle_t& cublas_handle,
                                 cublasLtHandle_t& cublas_lt_handle,
                                 cudnnHandle_t& cudnn_handle,
                                 cudnnOpTensorDescriptor_t& operator_sum_descriptor)
{
    if (operator_sum_descriptor)
    {
        cudnnDestroyOpTensorDescriptor(operator_sum_descriptor);
        operator_sum_descriptor = nullptr;
    }

    if (cublas_lt_handle)
    {
        cublasLtDestroy(cublas_lt_handle);
        cublas_lt_handle = nullptr;
    }

    if (cublas_handle)
    {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }

    if (cudnn_handle)
    {
        cudnnDestroy(cudnn_handle);
        cudnn_handle = nullptr;
    }

    device::destroy_stream(compute_stream);
    compute_stream = nullptr;
}

#else

cudnnTensorDescriptor_t TensorView::get_descriptor() const
{
    throw runtime_error("TensorView::get_descriptor requires CUDA support.");
}

void TensorView::set_descriptor(const Shape&) const
{
    throw runtime_error("TensorView::set_descriptor requires CUDA support.");
}

static bool uses_cuda_fill(const TensorView& view)
{
    return view.device == Device::CUDA;
}

static void fill_cuda(const TensorView&, float)
{
    throw runtime_error("TensorView::fill requires CUDA support for CUDA tensors.");
}

static void initialize_cuda_backend(cudaStream_t&,
                                    cublasHandle_t&,
                                    cublasLtHandle_t&,
                                    cudnnHandle_t&,
                                    cudnnOpTensorDescriptor_t&)
{
}

static void destroy_cuda_backend(cudaStream_t&,
                                 cublasHandle_t&,
                                 cublasLtHandle_t&,
                                 cudnnHandle_t&,
                                 cudnnOpTensorDescriptor_t&)
{
}

#endif

void TensorView::fill(float value)
{
    if (!data) return;

    if (uses_cuda_fill(*this))
    {
        fill_cuda(*this, value);
        return;
    }

    assert(type == Type::FP32);
    float* data_pointer = static_cast<float*>(data);
    std::fill(data_pointer, data_pointer + size(), value);
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
    initialize_cuda_backend(compute_stream,
                            cublas_handle,
                            cublas_lt_handle,
                            cudnn_handle,
                            operator_sum_descriptor);
}

Backend::~Backend()
{
    destroy_cuda_backend(compute_stream,
                         cublas_handle,
                         cublas_lt_handle,
                         cudnn_handle,
                         operator_sum_descriptor);
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

void copy_device_to_host_float(const void* device_src, Type src_dtype,
                               Index element_count, float* host_dst,
                               cudaStream_t stream)
{
    if (element_count == 0) return;

    if (src_dtype == Type::FP32)
    {
        device::copy_async(host_dst, device_src,
                           element_count * Index(sizeof(float)),
                           device::CopyKind::DeviceToHost,
                           stream);
        device::synchronize(stream);
        return;
    }

    if (src_dtype == Type::BF16)
    {
        vector<uint16_t> staging(static_cast<size_t>(element_count));
        device::copy_async(staging.data(), device_src,
                           element_count * Index(sizeof(uint16_t)),
                           device::CopyKind::DeviceToHost,
                           stream);
        device::synchronize(stream);
        for (Index i = 0; i < element_count; ++i)
        {
            const uint32_t bits = static_cast<uint32_t>(staging[size_t(i)]) << 16;
            memcpy(&host_dst[i], &bits, sizeof(float));
        }
        return;
    }

    throw runtime_error("copy_device_to_host_float: unsupported dtype.");
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
