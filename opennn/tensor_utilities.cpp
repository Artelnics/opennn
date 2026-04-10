//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R S   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "tensor_utilities.h"
#include "random_utilities.h"

#include "../eigen/Eigen/Dense"

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
    vector<Index> indices;

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

    for(int i = 0; i < n; i++)
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
                      bool parallelize)
{    
    const Index rows_number = row_indices.size();
    const Index columns_number = column_indices.size();

    if(rows_number == 0 || columns_number == 0) return;

    const type* matrix_data = matrix.data();

    const Index matrix_cols_number = matrix.cols();

    const bool contiguous = is_contiguous(column_indices);

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

    for(Index i = 0; i < size; i++)
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
            if(!token.empty() && result.rank < Shape::MaxRank)
                result.shape[result.rank++] = stoi(token);
        }
        catch (const invalid_argument&)
        {
            throw runtime_error("Error: Input string contains non-numeric elements.\n");
        }
    }

    return result;
}

string vector_to_string(const VectorI& x, const string& separator)
{
    ostringstream buffer;

    for(Index i = 0; i < x.size(); i++)
        buffer << x(i) << separator;

    return buffer.str();
}

string vector_to_string(const VectorR& x, const string& separator)
{
    ostringstream buffer;

    for(Index i = 0; i < x.size(); i++)
        buffer << x(i) << separator;

    return buffer.str();
}

void string_to_vector(const string& input, VectorR& x)
{
    istringstream stream(input);
    type value;
    vector<type> buffer;

    while (stream >> value)
        buffer.push_back(value);

    x.resize(static_cast<Index>(buffer.size()));

    for (Index i = 0; i < x.size(); ++i)
        x(i) = buffer[i];
}

bool contains(const vector<string>& data, const string& value)
{
    return find(data.begin(), data.end(), value) != data.end();
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
    for(size_t i = 0; i < views.size(); i++)
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

    for(size_t i = 0; i < views.size(); i++)
        total_size += get_size(views[i]);

    return total_size;
}

pair<VectorR, VectorR> filter_missing_values(const VectorR& x, const VectorR& y)
{
    Index new_size = 0;

    for(Index i = 0; i < x.size(); i++)
        if(!isnan(x(i)) && !isnan(y(i)))
            new_size++;

    if(new_size == x.size())
        return make_pair(x, y);

    VectorR new_x(new_size);
    VectorR new_y(new_size);

    Index index = 0;

    for(Index i = 0; i < x.size(); i++)
    {
        if(!isnan(x(i)) && !isnan(y(i)))
        {
            new_x(index) = x(i);
            new_y(index) = y(i);

            index++;
        }
    }

    return {new_x, new_y};
}

pair<VectorR, MatrixR> filter_missing_values(const VectorR& x, const MatrixR& y)
{
    const Index rows_number = x.size();
    const Index y_columns_number = y.cols();

    vector<Index> valid_indices;
    valid_indices.reserve(rows_number);

    for(Index i = 0; i < rows_number; i++)
        if(!isnan(x(i)) && !isnan(y(i)))
            valid_indices.push_back(i);

    const Index new_rows_number = static_cast<Index>(valid_indices.size());
    VectorR new_x(new_rows_number);
    MatrixR new_y(new_rows_number, y_columns_number);

    for(Index i = 0; i < new_rows_number; i++)
    {
        new_x(i) = x(valid_indices[i]);
        new_y.row(i) = y.row(valid_indices[i]);
    }

    return {new_x, new_y};
}

pair<MatrixR, MatrixR> filter_missing_values(const MatrixR& x, const MatrixR& y)
{
    const Index rows_number = x.rows();
    const Index x_columns_number = x.cols();
    const Index y_columns_number = y.cols();

    if(x.rows() != y.rows())
        throw runtime_error("filter_missing_values: Matrices must have the same number of rows.");

    vector<Index> valid_indices;
    valid_indices.reserve(rows_number);

    for(Index i = 0; i < rows_number; i++)
    {
        const bool x_row_ok = x.row(i).array().isFinite().all();
        const bool y_row_ok = y.row(i).array().isFinite().all();

        if(x_row_ok && y_row_ok)
            valid_indices.push_back(i);
    }

    const Index new_rows_number = static_cast<Index>(valid_indices.size());
    MatrixR new_x(new_rows_number, x_columns_number);
    MatrixR new_y(new_rows_number, y_columns_number);

    for(Index i = 0; i < new_rows_number; i++)
    {
        const Index original_index = valid_indices[i];
        new_x.row(i) = x.row(original_index);
        new_y.row(i) = y.row(original_index);
    }

    return {new_x, new_y};
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

// CUDA link/get_size not needed - TensorView unified


Device::Device()
{
    int max_threads = std::thread::hardware_concurrency();
    if (max_threads <= 0) max_threads = omp_get_max_threads();
    if (max_threads <= 0) max_threads = 1;

    set_threads_number(max_threads);

#ifdef CUDA
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));

    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&operator_sum_descriptor));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(operator_sum_descriptor, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

    CHECK_CUDNN(cudnnCreateOpTensorDescriptor(&operator_multiplication_descriptor));
    CHECK_CUDNN(cudnnSetOpTensorDescriptor(operator_multiplication_descriptor, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
#endif
}

Device::~Device()
{
#ifdef CUDA
    if (reduce_add_descriptor) cudnnDestroyReduceTensorDescriptor(reduce_add_descriptor);
    if (reduction_workspace) cudaFree(reduction_workspace);
    if (operator_sum_descriptor) cudnnDestroyOpTensorDescriptor(operator_sum_descriptor);
    if (operator_multiplication_descriptor) cudnnDestroyOpTensorDescriptor(operator_multiplication_descriptor);
    if (cublas_handle) cublasDestroy(cublas_handle);
    if (cudnn_handle) cudnnDestroy(cudnn_handle);
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

VectorR filter_missing_values(const VectorR &input)
{
    VectorR result(input.size());

    auto end_iterator = copy_if(input.data(),
                                input.data() + input.size(),
                                result.data(),
                                [](type value) {
                                    return !isnan(value);
                                });

    result.conservativeResize(end_iterator - result.data());

    return result;
}

#ifdef CUDA
cudnnReduceTensorDescriptor_t Device::get_reduce_add_descriptor()
{
    auto& d = instance();
    if(!d.reduce_add_descriptor)
    {
        cudnnCreateReduceTensorDescriptor(&d.reduce_add_descriptor);
        cudnnSetReduceTensorDescriptor(d.reduce_add_descriptor,
                                       CUDNN_REDUCE_TENSOR_ADD,
                                       CUDNN_DATA_FLOAT,
                                       CUDNN_NOT_PROPAGATE_NAN,
                                       CUDNN_REDUCE_TENSOR_NO_INDICES,
                                       CUDNN_32BIT_INDICES);
    }
    return d.reduce_add_descriptor;
}

void* Device::get_reduction_workspace()
{
    auto& d = instance();
    if(!d.reduction_workspace)
    {
        d.reduction_workspace_size = 1024 * 1024;
        CHECK_CUDA(cudaMalloc(&d.reduction_workspace, d.reduction_workspace_size));
    }
    return d.reduction_workspace;
}

size_t Device::get_reduction_workspace_size()
{
    auto& d = instance();
    if(!d.reduction_workspace)
        get_reduction_workspace();
    return d.reduction_workspace_size;
}
#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
