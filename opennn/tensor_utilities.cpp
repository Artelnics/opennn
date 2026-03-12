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

void multiply_matrices(Tensor3& tensor, const VectorR& vector)
{
    const Index rows = tensor.dimension(0);
    const Index cols = tensor.dimension(1);
    const Index depth = tensor.dimension(2);

    #pragma omp parallel for
    for(Index i = 0; i < depth; i++)
    {
        type* slice = tensor.data() + (i * rows * cols);
        MatrixMap slice_map(slice, rows, cols);

        slice_map *= vector(i);
    }
}


void multiply_matrices(Tensor3& tensor, const Tensor2& matrix)
{
    const Index rows = tensor.dimension(0);
    const Index cols = tensor.dimension(1);
    const Index depth = tensor.dimension(2);

    const MatrixMap multiplier(const_cast<type*>(matrix.data()), matrix.dimension(0), matrix.dimension(1));

    #pragma omp parallel for
    for(Index i = 0; i < depth; i++)
    {
        type* slice = tensor.data() + (i * rows * cols);
        MatrixMap slice_map(slice, rows, cols);

        slice_map = slice_map * multiplier;
    }
}


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


void sum_matrices(const VectorR& vector, Tensor3& tensor)
{
    const Index rows = tensor.dimension(0);
    const Index cols = tensor.dimension(1);
    const Index depth = tensor.dimension(2);

    if (vector.size() < depth) return;

    #pragma omp parallel for
    for(Index i = 0; i < depth; i++)
    {
        type* slice = tensor.data() + (i * rows * cols);
        MatrixMap slice_map(slice, rows, cols);

        slice_map.array() += vector(i);
    }
}


void save_csv(const Tensor2& data, const filesystem::path& path)
{
    ofstream file(path);

    if(!file.is_open())
        throw runtime_error("Cannot open matrix data file: " + path.string() + "\n");

    file.precision(20);

    const Index data_rows = data.dimension(0);
    const Index data_columns = data.dimension(1);

    char separator_string = ';';

    for(Index i = 0; i < data_rows; i++)
    {
        for(Index j = 0; j < data_columns; j++)
        {
            file << data(i, j);

            if(j != data_columns -1)
                file << separator_string;
        }

        file << endl;
    }

    file.close();
}


VectorI calculate_rank_greater(const VectorR& vector)
{
    const Index size = vector.size();

    VectorI rank(size);
    iota(rank.data(), rank.data() + rank.size(), 0);

    sort(rank.data(),
         rank.data() + rank.size(),
         [&](Index i, Index j){return vector[i] > vector[j];});

    return rank;
}


VectorI calculate_rank_less(const VectorR& vector)
{
    const Index size = vector.size();

    VectorI rank(size);
    iota(rank.data(), rank.data() + rank.size(), 0);

    sort(rank.data(),
         rank.data() + rank.size(),
         [&](Index i, Index j){return vector[i] < vector[j];});

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
    const Index indices_size = count_greater_than(data, bound);

    vector<Index> indices(indices_size);

    Index index = 0;

    for(size_t i  = 0; i < data.size(); i++)
        if(data[i] > bound)
            indices[index++] = data[i];

    return indices;
}


vector<Index> get_elements_greater_than(const vector<vector<Index>>& vectors, Index bound)
{
    const Index vectors_number = vectors.size();

    vector<Index> indices(0);

    for(Index i = 0; i < vectors_number; i++)
    {
        const vector<Index> indices_vector = get_elements_greater_than(vectors[i], bound);

        indices = join_vector_vector(indices, indices_vector);
    }

    return indices;
}


//Index count_between(const VectorR& vector,type minimum, type maximum)
//{
//    return (vector.array() >= minimum && vector.array() <= maximum).count();
//}


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
    const Index n = A.rows();

    VectorR x(n);

    const Map<Matrix<type, Dynamic, Dynamic>> A_eigen((type*)A.data(), n, n);

    const Map<Matrix<type, Dynamic, 1>> b_eigen((type*)b.data(), n, 1);

    Map<Matrix<type, Dynamic, 1>> x_eigen((type*)x.data(), n);

    x_eigen = A_eigen.colPivHouseholderQr().solve(b_eigen);

    return x;
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
        for(Index i = 0; i < rows_number; ++i)
            memcpy(tensor_data + i * columns_number, &matrix(row_indices[i], column_indices[0]), static_cast<size_t>(columns_number) * sizeof(float));
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


vector<Index> join_vector_vector(const vector<Index>& x, const vector<Index>& y)
{
    vector<Index> result = x;
    result.reserve(x.size() + y.size());

    result.insert(result.end(), y.begin(), y.end());

    return result;
}


string shape_to_string(const Shape& x, const string& separator)
{
    const Index size = x.size();

    ostringstream buffer;

    if(x.size() == 0)
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


bool contains(const vector<string>& data, const string& value)
{
    return find(data.begin(), data.end(), value) != data.end();
}


type round_to_precision(type x, const int& precision)
{
    const type factor = type(pow(10, precision));

    return round(factor*x)/factor;
}


VectorMap vector_map(const MatrixR& tensor, Index index_1)
{
    return VectorMap((type*)tensor.data() + tensor.rows()*index_1, tensor.rows());
}


VectorMap tensor_map(const Tensor2& tensor, Index index_1)
{
    return VectorMap((type*)tensor.data() + tensor.dimension(0)*index_1, tensor.dimension(0));
}


MatrixMap tensor_map(const Tensor3& tensor, Index index_2)
{
    return MatrixMap((type*)tensor.data() +  tensor.dimension(0) * tensor.dimension(1)* index_2,
                                      tensor.dimension(0), tensor.dimension(1));
}


TensorMap3 tensor_map(const Tensor4& tensor, Index index_3)
{
    return TensorMap3((type*)tensor.data() + tensor.dimension(0) * tensor.dimension(1) * tensor.dimension(2) * index_3,
                                      tensor.dimension(0), tensor.dimension(1), tensor.dimension(2));
}


TensorMap3 tensor_map_(const TensorMap4 tensor, Index index_3)
{
    return TensorMap3(tensor.data() + tensor.dimension(0) * tensor.dimension(1) * tensor.dimension(2) * index_3,
                                      tensor.dimension(0), tensor.dimension(1), tensor.dimension(2));
}


MatrixMap tensor_map(const Tensor4& tensor, Index index_3, Index index_2)
{
    return MatrixMap((type*)tensor.data() + tensor.dimension(0) * tensor.dimension(1)*(index_3 * tensor.dimension(2) + index_2),
                                      tensor.dimension(0), tensor.dimension(1));
}


Index get_size(const Shape& shape)
{
    return accumulate(shape.begin(), shape.end(), 1, multiplies<Index>());
}


Shape prepend(const Index &x, const Shape&d)
{
    Shape result = {x};
    result.insert(result.end(), d.begin(), d.end());
    return result;
}


type* link(type *pointer, vector<TensorView*> views)
{
    constexpr Index ALIGN_ELEMENTS = EIGEN_MAX_ALIGN_BYTES / sizeof(type);
    constexpr Index MASK = ~(ALIGN_ELEMENTS - 1);

    for(TensorView* view : views)
    {
        if(!view || view->size() == 0)
            continue;

        view->data = pointer;

        if (reinterpret_cast<uintptr_t>(pointer) % EIGEN_MAX_ALIGN_BYTES != 0)
            throw runtime_error("Master pointer in link() is not aligned.");

        pointer += (view->size() + ALIGN_ELEMENTS - 1) & MASK;
    }

    return pointer;
}


void link(type *pointer, vector<vector<TensorView*>> views)
{
    for(size_t i = 0; i < views.size(); i++)
        pointer = link(pointer, views[i]);
}


Index get_size(const vector<TensorView*> views)
{
    constexpr Index ALIGN_ELEMENTS = EIGEN_MAX_ALIGN_BYTES / sizeof(type);
    constexpr Index MASK = ~(ALIGN_ELEMENTS - 1);

    Index total_size = 0;

    for(const TensorView* view : views)
    {
        if(!view || view->size() == 0)
            continue;

        total_size += (view->size() + ALIGN_ELEMENTS - 1) & MASK;
    }

    return total_size;
}


Index get_size(vector<vector<TensorView*>> views)
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

    Index new_rows_number = 0;

    VectorB not_NAN_row(rows_number);

    for(Index i = 0; i < rows_number; i++)
    {
        not_NAN_row(i) = true;

        if(isnan(x(i)) || isnan(y(i)))
            not_NAN_row(i) = false;

        if(not_NAN_row(i))
            new_rows_number++;
    }

    VectorR new_x(new_rows_number);
    MatrixR new_y(new_rows_number, y_columns_number);

    Index index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(!not_NAN_row(i))
            continue;

        for(Index j = 0; j < y_columns_number; j++)
            new_y(index, j) = y(i, j);

        new_x(index++) = x(i);
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

#ifdef OPENNN_CUDA

type* link(type* pointer, const vector<TensorViewCuda*>& views)
{
    constexpr Index ALIGN_ELEMENTS = EIGEN_MAX_ALIGN_BYTES / sizeof(type);
    constexpr Index MASK = ~(ALIGN_ELEMENTS - 1);

    for (TensorViewCuda* view : views)
    {
        if (!view || view->size() == 0)
            continue;

        view->data = pointer;

        pointer += (view->size() + ALIGN_ELEMENTS - 1) & MASK;
    }

    return pointer;
}


void link(type* pointer, const vector<vector<TensorViewCuda*>>& views)
{
    for (size_t i = 0; i < views.size(); i++)
        pointer = link(pointer, views[i]);
}


Index get_size(const vector<TensorViewCuda*>& views)
{
    constexpr Index ALIGN_ELEMENTS = EIGEN_MAX_ALIGN_BYTES / sizeof(type);
    constexpr Index MASK = ~(ALIGN_ELEMENTS - 1);

    Index total_size = 0;

    for (const TensorViewCuda* view : views)
    {
        if (!view || view->size() == 0)
            continue;

        total_size += (view->size() + ALIGN_ELEMENTS - 1) & MASK;
    }

    return total_size;
}


Index get_size(const vector<vector<TensorViewCuda*>>& views)
{
    Index total_size = 0;

    for (size_t i = 0; i < views.size(); i++)
        total_size += get_size(views[i]);

    return total_size;
}


#endif

Device::Device()
{
    int max_threads = std::thread::hardware_concurrency();
    if (max_threads <= 0) max_threads = omp_get_max_threads();
    if (max_threads <= 0) max_threads = 1;

    set_threads_number(max_threads);

#ifdef OPENNN_CUDA
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
#ifdef OPENNN_CUDA
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


#ifdef OPENNN_CUDA
cublasHandle_t Device::get_cublas_handle() { return cublas_handle; }
cudnnHandle_t Device::get_cudnn_handle() { return cudnn_handle; }
cudnnOpTensorDescriptor_t Device::get_operator_sum_descriptor() { return operator_sum_descriptor; }
cudnnOpTensorDescriptor_t Device::get_operator_multiplication_descriptor() { return operator_multiplication_descriptor; }
#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
