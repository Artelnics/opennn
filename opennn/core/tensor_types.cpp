//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   T Y P E S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/tensor_types.h"
#include "opennn/core/string_utilities.h"

#include <algorithm>

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

CudnnDescriptor<cudnnTensorDescriptor_t> TensorView::get_descriptor() const
{
    int batch_count = 1, channels = 1, height = 1, width = 1;
    const size_t rank = shape.get_rank();
    if (rank >= 1) channels    = static_cast<int>(shape[rank - 1]);
    if (rank >= 2) batch_count = static_cast<int>(shape[0]);
    if (rank >= 3) width       = static_cast<int>(shape[rank - 2]);
    if (rank >= 4) height      = static_cast<int>(shape[rank - 3]);

    CudnnDescriptor<cudnnTensorDescriptor_t> descriptor;
    if (batch_count <= 0 || channels <= 0 || height <= 0 || width <= 0)
        return descriptor;

    throw_if(Index(batch_count) * channels * height * width > Index(numeric_limits<int>::max()),
             "TensorView descriptor: {}x{}x{}x{} exceeds the cuDNN 4d descriptor limit of INT32_MAX elements.",
             batch_count, channels, height, width);

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&descriptor.handle));
    descriptor.deleter = &cudnnDestroyTensorDescriptor;
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor, CUDNN_TENSOR_NHWC,
                                           to_cudnn(type),
                                           batch_count, channels, height, width));
    return descriptor;
}

static void fill_cuda(const TensorView& view, float value)
{
    if (value == 0.0f)
        return device::set_zero(view.get_data(), view.byte_size(), Device::CUDA);

    CHECK_CUDNN(cudnnSetTensor(device::get_cudnn_handle(),
                               view.get_descriptor(), view.get_data(), &value));
}

#else

CudnnDescriptor<cudnnTensorDescriptor_t> TensorView::get_descriptor() const OPENNN_CUDA_STUB_BODY(TensorView::get_descriptor)

static void fill_cuda(const TensorView&, float)
{
    throw runtime_error("TensorView::fill requires CUDA support for CUDA tensors.");
}

#endif

void TensorView::fill(float value) const
{
    if (!data) return;

    if (is_cuda())
        return fill_cuda(*this, value);

    if (value == 0.0f)
    {
        memset(data, 0, size_t(byte_size()));
        return;
    }

    throw_if(type != Type::FP32,
             "TensorView::fill: a non-zero fill requires FP32 storage.");

    float* values = static_cast<float*>(data);
    std::fill(values, values + size(), value);
}

string shape_to_string(const Shape& shape, const string& separator)
{
    ostringstream buffer;

    throw_if(shape.empty(),
             "Dimensions size must be greater than 0.\n");

    for (size_t i = 0; i < shape.get_rank(); ++i)
        buffer << shape[i] << separator;

    return buffer.str();
}

Shape string_to_shape(const string& text, const string& separator)
{
    Shape result;

    throw_if(text.empty(),
             "Input string must not be empty.\n");
    throw_if(separator.empty(),
             "Shape separator must not be empty.\n");

    for (const Index value : parse_number_list<Index>(text, "Shape", separator[0]))
        result.push_back(value);

    return result;
}

void fill_tensor_data(const MatrixR& matrix,
                      const vector<Index>& row_indices,
                      const vector<Index>& column_indices,
                      const span<float> tensor_span,
                      int contiguous_hint)
{
    const Index rows_number = row_indices.size();
    const Index columns_number = column_indices.size();

    if (rows_number == 0 || columns_number == 0) return;

    throw_if(ssize(tensor_span) < rows_number * columns_number,
             "fill_tensor_data: output buffer holds {} values but {}x{} = {} are required.",
             ssize(tensor_span), rows_number, columns_number, rows_number * columns_number);

    float* __restrict tensor_data = tensor_span.data();

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
                               cudaStream_t stream,
                               vector<uint16_t>& bf16_staging)
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
        bf16_staging.resize(static_cast<size_t>(element_count));
        device::copy_async(bf16_staging.data(), device_src,
                           element_count * Index(sizeof(uint16_t)),
                           device::CopyKind::DeviceToHost,
                           stream);
        device::synchronize(stream);
        ranges::transform(bf16_staging, host_dst, bfloat16_to_float_host);
    }
    else
        throw runtime_error("copy_device_to_host_float: unsupported dtype.");
}

void copy_device_to_host_float(const void* device_src, Type src_dtype,
                               Index element_count, float* host_dst,
                               cudaStream_t stream)
{
    vector<uint16_t> bf16_staging;
    copy_device_to_host_float(device_src, src_dtype, element_count,
                              host_dst, stream, bf16_staging);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
