//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R S   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "random_utilities.h"

#include "../eigen/Eigen/Dense"

namespace opennn
{

void multiply_matrices(Tensor3& tensor, const VectorR& vector)
{
    const Index depth = tensor.dimension(2);

    for(Index i = 0; i < depth; i++)
    {
        MatrixMap matrix = tensor_map(tensor, i);

        matrix.device(get_device()) = matrix * vector(i);
    }
}


void multiply_matrices(Tensor3& tensor, const Tensor2& matrix)
{
    const Index depth = tensor.dimension(2);

    for(Index i = 0; i < depth; i++)
    {
        MatrixMap slice = tensor_map(tensor, i);

        slice.device(get_device()) = slice * matrix;
    }
}

/*
Tensor2 self_kronecker_product(const VectorR& vector)
{
    const Index columns_number = vector.size();

    Tensor2 matrix(columns_number, columns_number);

    for(Index i = 0; i < columns_number; i++)
    {
        VectorMap column = tensor_map(matrix, i);

        column.device(get_device()) = vector * vector(i);
    }

    return matrix;
}
*/

MatrixR append_rows(const MatrixR& starting_matrix, const MatrixR& block)
{
    if (starting_matrix.size() == 0)
        return block;
    if (block.size() == 0)
        return starting_matrix;

    MatrixR final_matrix(starting_matrix.rows() + block.rows(), starting_matrix.cols());

    final_matrix << starting_matrix, block;

    return final_matrix;
}

vector<Index> build_feasible_rows_mask(const MatrixR& outputs, const VectorR& minimums, const VectorR& maximums)
{    
    const Index rows_unfiltered =  outputs.rows();
    const Index variables_to_filter = outputs.cols();

    if(minimums.size() != variables_to_filter || maximums.size() != variables_to_filter)
        throw runtime_error("Minimums/maximums size mismatch.\n");

    vector<Index> feasible_rows;

    feasible_rows.reserve(static_cast<size_t>(rows_unfiltered));

    const auto min_array = minimums.array().transpose();
    const auto max_array = maximums.array().transpose();

    for (Index i = 0; i < rows_unfiltered; ++i)
        if (((outputs.row(i).array() >= min_array) && (outputs.row(i).array() <= max_array)).all())
            feasible_rows.push_back(i);

    return feasible_rows;
}


void sum_matrices(const VectorR& vector, Tensor3& tensor)
{
    const Index depth = tensor.dimension(2);

    for(Index i = 0; i < depth; i++)
    {
        MatrixMap matrix = tensor_map(tensor, i);

        matrix.device(get_device()) = matrix + vector(i);
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


Index count_between(const VectorR& vector,type minimum, type maximum)
{
    const Index size = vector.size();

    Index count = 0;

#pragma omp parallel for reduction(+: count)
    for(Index i = 0; i < size; i++)
        if(vector(i) >= minimum && vector(i) <= maximum)
            count++;

    return count;
}


void set_row(Tensor2& matrix, const VectorR& new_row, Index row_index)
{
    const Index columns_number = new_row.size();

#pragma omp parallel for

    for(Index i = 0; i < columns_number; i++)
        matrix(row_index, i) = new_row(i);
}


MatrixR filter_column_minimum_maximum(const MatrixR& matrix,
                                      Index column_index,
                                      type minimum,
                                      type maximum)
{
    const VectorR column = matrix.col(column_index);
    const Index new_rows_number = count_between(column, minimum, maximum);

    if(new_rows_number == 0) return MatrixR();

    const Index rows_number = matrix.rows();
    const Index columns_number = matrix.cols();

    bool check_conditions = false;

    MatrixR new_matrix(new_rows_number, columns_number);

    Index row_index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(matrix(i, column_index) >= minimum
        && matrix(i, column_index) <= maximum)
        {
            const VectorR row = matrix.row(i);

            set_row(new_matrix, row, row_index);

            row_index++;

            check_conditions = true;
        }
    }

    if(!check_conditions)
        throw runtime_error("Invalid conditions\n");

    return new_matrix;
}


VectorI get_nearest_points(const MatrixR& matrix,const VectorR& point, int n = 1)
{
    const Index number_points_to_compare = matrix.rows();

    const Index coordinates_number = matrix.cols();

    if(point.size() != coordinates_number)
        throw runtime_error("get_nearest_points : Matrix row dimension and point size must match.\n");

    if(n <= 0)
        throw runtime_error("get_nearest_points : n must be positive.\n");

    if(n > number_points_to_compare)
        n = number_points_to_compare;

    vector<pair<type, Index>> dist_index(number_points_to_compare);

#pragma omp parallel for
    for(Index i = 0; i < number_points_to_compare; ++i)
    {       
        const type distance = (matrix.row(i) - point).norm();

        dist_index[i] = make_pair(distance, i);
    }

    std::partial_sort(
        dist_index.begin(),
        dist_index.begin() + n,
        dist_index.end(),
        [](const auto& a, const auto& b){ return a.first < b.first; });

    VectorI nearest_indices(n);
    for(int i = 0; i < n; ++i)
        nearest_indices(i) = dist_index[i].second;

    return nearest_indices;
}

void set_identity(Tensor2& matrix)
{
    const Index rows_number = matrix.dimension(0);

    matrix.setZero();

#pragma omp parallel for
    for(Index i = 0; i < rows_number; i++)
        matrix(i, i) = type(1);
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
                      type* __restrict tensor_data)
{
    const Index rows_number = row_indices.size();
    const Index columns_number = column_indices.size();

    if(rows_number == 0 || columns_number == 0) return;

    const type* matrix_data = matrix.data();
    const Index matrix_rows_number = matrix.rows();

    vector<const type*> col_ptrs(columns_number);

    for(Index j = 0; j < columns_number; ++j)
        col_ptrs[j] = matrix_data + matrix_rows_number * column_indices[j];

    #pragma omp parallel for schedule(static)
    for(Index j = 0; j < columns_number; ++j)
    {
        const type* src_column = col_ptrs[j];
        type* dest_column = tensor_data + rows_number*j;

        //#pragma omp simd
        for(Index i = 0; i < rows_number; ++i)
            dest_column[i] = src_column[row_indices[i]];
    }

/*
    const Index rows_number = row_indices.size();
    const Index columns_number = column_indices.size();

    if(rows_number == 0 || columns_number == 0) return;

    const type* __restrict matrix_data = matrix.data();
    const Index matrix_rows_number = matrix.dimension(0);
    const Index* __restrict row_indices_data = row_indices.data();   

    #pragma omp parallel for if(columns_number >= 32) schedule(static)
    for(Index j = 0; j < columns_number; ++j)
    {
        const type* __restrict matrix_column = matrix_data + (matrix_rows_number * column_indices[j]);
        type* __restrict tensor_value = tensor_data + (rows_number * j);

        for(Index i = 0; i < rows_number; ++i)
            tensor_value[i] = matrix_column[row_indices_data[i]];
    }
*/
}


void fill_tensor_data_row_major(const MatrixR& matrix,
                                const vector<Index>& row_indices,
                                const vector<Index>& column_indices,
                                type* tensor_data)
{
    const Index rows_number = row_indices.size();
    const Index columns_number = column_indices.size();

    const type* const matrix_data = matrix.data();

#pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        const Index row_index = row_indices[i];

        for(Index j = 0; j < columns_number; j++)
        {
            const Index column_index = column_indices[j];
            const type* const matrix_value = matrix_data + row_index + matrix.rows() * column_index;
            type* tensor_value = tensor_data + i * columns_number + j;
            *tensor_value = *matrix_value;
        }
    }
}

/*
void fill_tensor_sequence(const Tensor2& matrix,
                          const vector<Index>& rows_indices,
                          const vector<Index>& columns_indices,
                          Index past_time_steps,
                          type* tensor_data)
{
    if (rows_indices.empty() || columns_indices.empty())
        return;

    const Index rows_number = rows_indices.size();
    const Index columns_number = columns_indices.size();

    const Index batch_size = rows_indices.size();
    const Index input_size = columns_number;

    TensorMap3 batch(tensor_data, batch_size, past_time_steps, input_size);

    //#pragma omp parallel for collapse(3)

    for(Index i = 0; i < batch_size; i++)
    {
        for(Index j = 0; j < past_time_steps; j++)
        {
            const Index actual_row = i + j * batch_size;

            for(Index k = 0; k < input_size; k++)
                batch(i, j, k) = matrix(actual_row, columns_indices[k]);
        }
    }
}
*/

vector<Index> join_vector_vector(const vector<Index>& x, const vector<Index>& y)
{
    const Index size = x.size() + y.size();

    vector<Index> data(size);

    memcpy(data.data(), x.data(), x.size() * sizeof(Index));

    memcpy(data.data() + x.size(), y.data(), y.size() * sizeof(Index));

    return data;
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
    vector<string> copy = data;

    const string* it = find(copy.data(), copy.data()+copy.size(), value);

    return it != copy.data() + copy.size();
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

/*
VectorMap tensor_map_(const MatrixMap tensor, Index index_1)
{
    return VectorMap((type*)tensor.data() + tensor.dimension(0) * index_1,
                                      tensor.dimension(0));
}
*/

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


Index get_size(const Shape&d)
{
    return accumulate(d.begin(), d.end(), 1, multiplies<Index>());
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

void shuffle_rows(Tensor2& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    if (rows_number <= 1) return;

    type* data = matrix.data();

    for (Index i = rows_number - 1; i > 0; --i)
    {
        const Index j = random_integer(0, i);

        if (i == j) continue;

#pragma omp parallel for schedule(static)
        for (Index c = 0; c < columns_number; ++c)
        {
            const Index offset = c * rows_number;

            const type temp = data[i + offset];
            data[i + offset] = data[j + offset];
            data[j + offset] = temp;
        }
    }
}

#ifdef OPENNN_CUDA

type* link(type* pointer, vector<TensorViewCuda*> views)
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


void link(type* pointer, vector<vector<TensorViewCuda*>> views)
{
    for (size_t i = 0; i < views.size(); i++)
        pointer = link(pointer, views[i]);
}


Index get_size(const vector<TensorViewCuda*> views)
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


Index get_size(vector<vector<TensorViewCuda*>> views)
{
    Index total_size = 0;

    for (size_t i = 0; i < views.size(); i++)
        total_size += get_size(views[i]);

    return total_size;
}


void shuffle_rows(MatrixR& matrix)
{
    const Index rows_number = matrix.rows();
    const Index columns_number = matrix.cols();

    if (rows_number <= 1) return;

    type* data = matrix.data();

    for (Index i = rows_number - 1; i > 0; --i)
    {
        const Index j = random_integer(0, i);

        if (i == j) continue;

        #pragma omp parallel for schedule(static)
        for (Index c = 0; c < columns_number; ++c)
        {
            const Index offset = c * rows_number;

            const type temp = data[i + offset];
            data[i + offset] = data[j + offset];
            data[j + offset] = temp;
        }
    }
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
