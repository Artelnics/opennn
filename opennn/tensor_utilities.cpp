//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R S   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "tensor_utilities.h"
#include "random_utilities.h"

#include <Eigen/Dense>

namespace opennn
{

MatrixR append_rows(const MatrixR& starting_matrix, const MatrixR& block)
{
    if (starting_matrix.size() == 0)
        return block;
    if (block.size() == 0)
        return starting_matrix;

    if (starting_matrix.cols() != block.cols())
        throw runtime_error("append_rows: Column mismatch (" +
                                to_string(starting_matrix.cols()) + " vs " +
                                to_string(block.cols()) + ")");

    MatrixR final_matrix(starting_matrix.rows() + block.rows(), starting_matrix.cols());

    final_matrix.topRows(starting_matrix.rows()) = starting_matrix;
    final_matrix.bottomRows(block.rows()) = block;

    return final_matrix;
}

vector<Index> build_feasible_rows_mask(const MatrixR& outputs, const VectorR& minimums, const VectorR& maximums)
{
    const Index rows_unfiltered =  outputs.rows();
    const Index variables_to_filter = outputs.cols();

    if(minimums.size() != variables_to_filter || maximums.size() != variables_to_filter)
        throw runtime_error("build_feasible_rows_mask: Minimums/maximums size mismatch with outputs columns.\n");

    vector<Index> feasible_rows;
    feasible_rows.reserve(static_cast<size_t>(rows_unfiltered));

    const auto min_bound = minimums.transpose().array();
    const auto max_bound = maximums.transpose().array();

    for (Index i = 0; i < rows_unfiltered; ++i)
    {
        const auto row_arr = outputs.row(i).array();

        if ((row_arr >= min_bound && row_arr <= max_bound).all())
            feasible_rows.push_back(i);
    }

    return feasible_rows;
}

VectorI calculate_rank(const VectorR& vector, bool ascending)
{
    const Index size = vector.size();

    VectorI rank(size);
    iota(rank.data(), rank.data() + rank.size(), 0);

    sort(rank.data(),
         rank.data() + rank.size(),
         [&](Index i, Index j){ return ascending ? vector[i] < vector[j] : vector[i] > vector[j]; });

    return rank;
}

Index count_greater_than(const vector<Index>& data, Index bound)
{
    return count_if(data.begin(), data.end(), [&](const Index value) {
        return value > bound;
    });
}

vector<Index> get_elements_greater_than(const vector<Index>& data, Index bound)
{
    vector<Index> indices;
    copy_if(data.begin(), data.end(), back_inserter(indices),
            [bound](Index value) { return value > bound; });
    return indices;
}

vector<Index> get_elements_greater_than(const vector<vector<Index>>& vectors, Index bound)
{
    size_t total = 0;
    for(const auto& v : vectors)
        total += v.size();

    vector<Index> indices;
    indices.reserve(total);

    for(const auto& v : vectors)
    {
        const vector<Index> filtered = get_elements_greater_than(v, bound);

        indices.insert(indices.end(), filtered.begin(), filtered.end());
    }

    return indices;
}

VectorI get_nearest_points(const MatrixR& matrix,const VectorR& point, int n)
{
    const Index rows = matrix.rows();

    const VectorR distances = (matrix.rowwise() - point.transpose()).rowwise().norm();

    vector<pair<type, Index>> pairs(rows);

    for(Index i = 0; i < rows; ++i)
        pairs[i] = {distances(i), i};

    if (n > rows)
        n = rows;

    partial_sort(pairs.begin(), pairs.begin() + n, pairs.end());

    VectorI result(n);

    for(int i = 0; i < n; ++i)
        result(i) = pairs[i].second;

    return result;
}

VectorR perform_Householder_QR_decomposition(const MatrixR& A, const VectorR& b)
{
    return A.colPivHouseholderQr().solve(b);
}

void fill_tensor_data(const MatrixR& matrix,
                      const vector<Index>& row_indices,
                      const vector<Index>& column_indices,
                      type* __restrict tensor_data,
                      bool parallelize,
                      int contiguous_hint)
{
    const Index rows_number = row_indices.size();
    const Index columns_number = column_indices.size();

    if(rows_number == 0 || columns_number == 0) return;

    const type* matrix_data = matrix.data();

    const Index matrix_cols_number = matrix.cols();

    const bool contiguous = (contiguous_hint >= 0) ? static_cast<bool>(contiguous_hint) : is_contiguous(column_indices);

    if (contiguous)
    {
        #pragma omp parallel for schedule(static) if (parallelize)
        for(Index i = 0; i < rows_number; ++i)
            memcpy(tensor_data + i * columns_number, &matrix(row_indices[i], column_indices[0]), static_cast<size_t>(columns_number) * sizeof(float));
    }
    else
    {
        #pragma omp parallel for schedule(static) if (parallelize)
        for(Index i = 0; i < rows_number; ++i)
        {
            const Index src_row = row_indices[i];
            const type* src_row_ptr = matrix_data + src_row * matrix_cols_number;
            type* dest_row_ptr = tensor_data + i * columns_number;

            for(Index j = 0; j < columns_number; ++j)
                dest_row_ptr[j] = src_row_ptr[column_indices[j]];
        }
    }
}

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

VectorMap vector_map(const MatrixR& tensor, Index index_1)
{
    return VectorMap(const_cast<type*>(tensor.data()) + tensor.rows()*index_1, tensor.rows());
}

MatrixMap tensor_map(const Tensor3& tensor, Index index_2)
{
    return MatrixMap(const_cast<type*>(tensor.data()) +  tensor.dimension(0) * tensor.dimension(1)* index_2,
                                      tensor.dimension(0), tensor.dimension(1));
}

TensorMap3 tensor_map(const Tensor4& tensor, Index index_3)
{
    return TensorMap3(const_cast<type*>(tensor.data()) + tensor.dimension(0) * tensor.dimension(1) * tensor.dimension(2) * index_3,
                                      tensor.dimension(0), tensor.dimension(1), tensor.dimension(2));
}

MatrixMap tensor_map(const Tensor4& tensor, Index index_3, Index index_2)
{
    return MatrixMap(const_cast<type*>(tensor.data()) + tensor.dimension(0) * tensor.dimension(1)*(index_3 * tensor.dimension(2) + index_2),
                                      tensor.dimension(0), tensor.dimension(1));
}

type* link(type *pointer, const vector<TensorView*>& views)
{
    for(TensorView* view : views)
    {
        if(!view || view->size() == 0)
            continue;

        view->data = pointer;

        if (reinterpret_cast<uintptr_t>(pointer) % EIGEN_MAX_ALIGN_BYTES != 0)
            throw runtime_error("Master pointer in link() is not aligned.");

        pointer += (view->size() + ALIGN_ELEMENTS - 1) & ALIGN_MASK;
    }

    return pointer;
}

void link(type *pointer, const vector<vector<TensorView*>>& views)
{
    for(size_t i = 0; i < views.size(); ++i)
        pointer = link(pointer, views[i]);
}

Index get_size(const vector<TensorView*>& views)
{
    Index total_size = 0;

    for(const TensorView* view : views)
    {
        if(!view || view->size() == 0)
            continue;

        total_size += (view->size() + ALIGN_ELEMENTS - 1) & ALIGN_MASK;
    }

    return total_size;
}

Index get_size(const vector<vector<TensorView*>>& views)
{
    Index total_size = 0;

    for(size_t i = 0; i < views.size(); ++i)
        total_size += get_size(views[i]);

    return total_size;
}

void shuffle_rows(MatrixR& matrix)
{
    const Index rows_number = matrix.rows();

    if (rows_number <= 1) return;

    for(Index i = rows_number - 1; i > 0; --i)
    {
        const Index j = random_integer(0, i);

        if (i == j) continue;

        matrix.row(i).swap(matrix.row(j));
    }
}

Device::Device()
{
    set_threads_number(0);  // 0 triggers hardware_concurrency auto-detection

#ifdef OPENNN_WITH_CUDA
    CHECK_CUDA(cudaStreamCreate(&compute_stream));

    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, compute_stream));
    CHECK_CUBLAS(cublasLtCreate(&cublas_lt_handle));
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, compute_stream));

    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&operator_sum_descriptor));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(operator_sum_descriptor, CUDNN_OP_TENSOR_ADD, CUDNN_IO_DTYPE, CUDNN_NOT_PROPAGATE_NAN));

    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&operator_multiplication_descriptor));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(operator_multiplication_descriptor, CUDNN_OP_TENSOR_MUL, CUDNN_IO_DTYPE, CUDNN_NOT_PROPAGATE_NAN));
#endif
}

Device::~Device()
{
#ifdef OPENNN_WITH_CUDA
    if (operator_sum_descriptor) cudnnDestroyOpTensorDescriptor(operator_sum_descriptor);
    if (operator_multiplication_descriptor) cudnnDestroyOpTensorDescriptor(operator_multiplication_descriptor);
    if (cublas_lt_handle) cublasLtDestroy(cublas_lt_handle);
    if (cublas_handle) cublasDestroy(cublas_handle);
    if (cudnn_handle) cudnnDestroy(cudnn_handle);
    if (compute_stream) cudaStreamDestroy(compute_stream);
#endif
}

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

ThreadPoolDevice* Device::get_thread_pool_device()
{
    return thread_pool_device.get();
}

void Device::set(DeviceType type)
{
    if(type == DeviceType::Gpu)
    {
#ifndef OPENNN_WITH_CUDA
        throw runtime_error("Device error: GPU requested but OpenNN was compiled without CUDA support.");
#endif
    }
    device_type = type;
}

VectorR filter_missing_values(const VectorR& x)
{
    vector<Index> valid;
    valid.reserve(x.size());

    for (Index i = 0; i < x.size(); ++i)
        if (isfinite(x(i))) valid.push_back(i);

    return slice_rows(x, valid);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
