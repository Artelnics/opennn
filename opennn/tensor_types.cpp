//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   T Y P E S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_types.h"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

cudnnTensorDescriptor_t TensorView::get_descriptor() const
{
    if (!descriptor_handle && !shape.empty())
        set_descriptor(shape);
    return descriptor_handle.get();
}

void TensorView::set_descriptor(const Shape& descriptor_shape) const
{
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
    return view.is_cuda();
}

static void fill_cuda(const TensorView&, float)
{
    throw runtime_error("TensorView::fill requires CUDA support for CUDA tensors.");
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

    const bool parallel_fill = rows_number * columns_number >= 65536;

    if (contiguous)
    {
        #pragma omp parallel for schedule(static) if(parallel_fill)
        for (Index i = 0; i < rows_number; ++i)
            memcpy(tensor_data + i * columns_number, &matrix(row_indices[i], column_indices[0]), static_cast<size_t>(columns_number) * sizeof(float));
    }
    else
    {
        #pragma omp parallel for schedule(static) if(parallel_fill)
        for (Index i = 0; i < rows_number; ++i)
        {
            const float* src_row_ptr  = matrix_data + row_indices[i] * matrix_cols_number;
            float*       dest_row_ptr = tensor_data + i * columns_number;
            for (Index j = 0; j < columns_number; ++j)
                dest_row_ptr[j] = src_row_ptr[column_indices[j]];
        }
    }
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
    }
    else if (src_dtype == Type::BF16)
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
    }
    else
        throw runtime_error("copy_device_to_host_float: unsupported dtype.");
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
