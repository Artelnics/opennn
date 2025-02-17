//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R S   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "../eigen/Eigen/Dense"

#include "tensors.h"

namespace opennn
{

type bound(const type& value, const type& minimum, const type& maximum)
{
    return std::min(std::max(value, minimum), maximum);
}


Index get_random_index(const Index& min, const Index& max) 
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(min, max);
    return dis(gen);
}


type get_random_type(const type& minimum, const type& maximum) 
{
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<type> dis(minimum, maximum);

    return dis(gen);
}


bool get_random_bool()
{
    return rand() % 2 == 1;
}


void multiply_matrices(const ThreadPoolDevice* thread_pool_device,
                       Tensor<type, 3>& tensor,
                       const Tensor<type, 1>& vector)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);
    const Index channels = tensor.dimension(2);

    for(Index i = 0; i < channels; i++)
    {
        TensorMap<Tensor<type, 2>> matrix(tensor.data() + i * rows_number * columns_number, 
                                          rows_number, 
                                          columns_number);

        matrix.device(*thread_pool_device) = matrix * vector(i);
    }
}


void multiply_matrices(const ThreadPoolDevice* thread_pool_device, Tensor<type, 3>& tensor, const Tensor<type, 2>& matrix)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);
    const Index channels = tensor.dimension(2);

    for(Index i = 0; i < channels; i++)
    {
        TensorMap<Tensor<type, 2>> slice(tensor.data() + i * rows_number * columns_number, 
                                         rows_number, 
                                         columns_number);

        slice.device(*thread_pool_device) = slice * matrix;
    }
}


void batch_matrix_multiplication(const ThreadPoolDevice* thread_pool_device,
                                 const TensorMap<Tensor<type, 3>>& A,
                                 TensorMap<Tensor<type, 3>>& B,
                                 TensorMap<Tensor<type, 3>>& C,
                                 const Eigen::array<IndexPair<Index>, 1> contraction_axes)
{
    // Assumes A, B & C share dimension 2 and A & B share one of their remaining 2 dimensions (the contraction axes)
    // The other 2 dimensions of C will be the non-equal dimensions of A & B, in that order
    // By default contraction axes are (1, 0)

    const Index A_rows = A.dimension(0);
    const Index A_columns = A.dimension(1);

    const Index B_rows = B.dimension(0);
    const Index B_columns = B.dimension(1);

    const Index C_rows = (contraction_axes[0].first == 0) ? A_columns : A_rows;
    const Index C_columns = (contraction_axes[0].second == 1) ? B_rows : B_columns;

    const Index channels = A.dimension(2);

    type* A_data = (type*)A.data();
    type* B_data = (type*)B.data();
    type* C_data = C.data();

    type* a_matrix_data = nullptr;
    type* b_matrix_data = nullptr;
    type* c_matrix_data = nullptr;

    for(Index i = 0; i < channels; i++)
    {
        a_matrix_data = A_data + A_rows * A_columns * i;
        b_matrix_data = B_data + B_rows * B_columns * i;
        c_matrix_data = C_data + C_rows * C_columns * i;

        const TensorMap<Tensor<type, 2>> A_matrix(a_matrix_data, A_rows, A_columns);
        const TensorMap<Tensor<type, 2>> B_matrix(b_matrix_data, B_rows, B_columns);
        TensorMap<Tensor<type, 2>> C_matrix(c_matrix_data, C_rows, C_columns);

        C_matrix.device(*thread_pool_device) = A_matrix.contract(B_matrix, contraction_axes);
    }
}


void batch_matrix_multiplication(const ThreadPoolDevice* thread_pool_device,
    TensorMap<Tensor<type, 3>>& A,
    const TensorMap<Tensor<type, 3>>& B,
    TensorMap<Tensor<type, 3>>& C,
    const Eigen::array<IndexPair<Index>, 1> contraction_axes)
{
// Assumes A, B & C share dimension 2 and A & B share one of their remaining 2 dimensions (the contraction axes)
// The other 2 dimensions of C will be the non-equal dimensions of A & B, in that order
// By default contraction axes are (1, 0)

    const Index A_rows = A.dimension(0);
    const Index A_columns = A.dimension(1);
    const Index B_rows = B.dimension(0);
    const Index B_columns = B.dimension(1);

    const Index C_rows = (contraction_axes[0].first == 0) ? A_columns : A_rows;
    const Index C_columns = (contraction_axes[0].second == 1) ? B_rows : B_columns;

    const Index channels = A.dimension(2);

    type* A_data = (type*)A.data();
    type* B_data = (type*)B.data();
    type* C_data = C.data();

    type* a_matrix_data = nullptr;
    type* b_matrix_data = nullptr;
    type* c_matrix_data = nullptr;

    for(Index i = 0; i < channels; i++)
    {
        a_matrix_data = A_data + A_rows * A_columns * i;
        b_matrix_data = B_data + B_rows * B_columns * i;
        c_matrix_data = C_data + C_rows * C_columns * i;

        const TensorMap<Tensor<type, 2>> A_matrix(a_matrix_data, A_rows, A_columns);
        const TensorMap<Tensor<type, 2>> B_matrix(b_matrix_data, B_rows, B_columns);
        TensorMap<Tensor<type, 2>> C_matrix(c_matrix_data, C_rows, C_columns);

        C_matrix.device(*thread_pool_device) = A_matrix.contract(B_matrix, contraction_axes);
    }
}


void batch_matrix_multiplication(const ThreadPoolDevice* thread_pool_device,
                                 const Tensor<type, 4>& A,
                                 const Tensor<type, 4>& B,
                                 Tensor<type, 4>& C,
                                 const Eigen::array<IndexPair<Index>, 1> contraction_axes)
{
    // Assumes A, B & C share dimensions 2 & 3 and A & B share one of their remaining 2 dimensions (the contraction axes)
    // The other 2 dimensions of C will be the non-equal dimensions of A & B, in that order
    // By default contraction axes are (1, 0).

    const Index A_rows = A.dimension(0);
    const Index A_columns = A.dimension(1);
    const Index B_rows = B.dimension(0);
    const Index B_columns = B.dimension(1);

    const Index C_rows = (contraction_axes[0].first == 0) ? A_columns : A_rows;
    const Index C_columns = (contraction_axes[0].second == 1) ? B_rows : B_columns;

    const Index channels = A.dimension(2);
    const Index blocks_number = A.dimension(3);

    type* A_data = (type*)A.data();
    type* B_data = (type*)B.data();
    type* C_data = C.data();

    type* a_matrix_data = nullptr;
    type* b_matrix_data = nullptr;
    type* c_matrix_data = nullptr;

    for(Index i = 0; i < blocks_number; i++)
    {
        for(Index j = 0; j < channels; j++)
        {
            a_matrix_data = A_data + A_rows * A_columns * (i * channels + j);
            b_matrix_data = B_data + B_rows * B_columns * (i * channels + j);
            c_matrix_data = C_data + C_rows * C_columns * (i * channels + j);

            const TensorMap<Tensor<type, 2>> A_matrix(a_matrix_data, A_rows, A_columns);
            const TensorMap<Tensor<type, 2>> B_matrix(b_matrix_data, B_rows, B_columns);
            TensorMap<Tensor<type, 2>> C_matrix(c_matrix_data, C_rows, C_columns);

            C_matrix.device(*thread_pool_device) = A_matrix.contract(B_matrix, contraction_axes);
        }
    }
}


Tensor<type, 2> self_kronecker_product(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector)
{
    const Index columns_number = vector.size();

    Tensor<type, 2> matrix(columns_number, columns_number);

    for(Index i = 0; i < columns_number; i++)
    {
        TensorMap<Tensor<type, 1>> column = tensor_map(matrix, i);

        column.device(*thread_pool_device) = vector * vector(i);
    }

    return matrix;
}


void divide_columns(const ThreadPoolDevice* thread_pool_device, TensorMap<Tensor<type, 2>>& matrix, const Tensor<type, 1>& vector)
{
    // @ Changes to test (the case in which you can divide by 0)
    const Index columns_number = matrix.dimension(1);
    Tensor<type, 1> corrected_vector = vector;

    for(Index i = 0; i < columns_number; i++)
    {
        //TensorMap<Tensor<type, 1>> column = tensor_map(matrix, i);
        auto column = matrix.chip(i, 1);  // chip slices along dimension 1

        for(Index j = 0; j < vector.size(); j++)
            if(vector(j) == 0)
                corrected_vector(j) = 1;

        column.device(*thread_pool_device) = column / corrected_vector;
    }
}


void sum_columns(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 2>& matrix)
{
    const Index columns_number = matrix.dimension(1);

    for(Index i = 0; i < columns_number; i++)
    {
        TensorMap<Tensor<type, 1>> column = tensor_map(matrix, i);

        column.device(*thread_pool_device) = column + vector(i);
    }
}


void sum_columns(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, TensorMap<Tensor<type, 2>>& matrix)
{
    const Index columns_number = matrix.dimension(1);

    for(Index i = 0; i < columns_number; i++)
    {
        TensorMap<Tensor<type, 1>> column = tensor_map(matrix, i); //Maybe this one gives error because the matrix is a TensorMap<Tensor<type, 2>> instead of just Tensor<type, 2>

        column.device(*thread_pool_device) = column + vector(i);
    }
}


void sum_matrices(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 3>& tensor)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);
    const Index channels = tensor.dimension(2);

    type* tensor_data = tensor.data();

    const Index slice_size = rows_number * columns_number;

    for(Index i = 0; i < channels; i++)
    {
        TensorMap<Tensor<type,2>> matrix(tensor_data + i*slice_size, rows_number, columns_number);

        matrix.device(*thread_pool_device) = matrix + vector(i);
    }
}


void substract_matrices(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 2>& matrix, Tensor<type, 3>& tensor)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);
    const Index channels = tensor.dimension(2);

    for(Index i = 0; i < channels; i++)
    {
        TensorMap<Tensor<type, 2>> slice(tensor.data() + i * rows_number * columns_number, 
                                         rows_number, 
                                         columns_number);

        slice.device(*thread_pool_device) = slice - matrix;
    }
}


Tensor<bool, 2> elements_are_equal(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    Tensor<bool, 2> result(x.dimension(0), x.dimension(1));

    #pragma omp parallel for
    for(int i = 0; i < x.size(); i++) 
        result(i) = (x(i) == y(i)); 

    return result;
}


void save_csv(const Tensor<type,2>& data, const filesystem::path& path)
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


Tensor<Index, 1> calculate_rank_greater(const Tensor<type, 1>& vector)
{
    const Index size = vector.size();

    Tensor<Index, 1> rank(size);
    iota(rank.data(), rank.data() + rank.size(), 0);

    sort(rank.data(),
         rank.data() + rank.size(),
         [&](Index i, Index j){return vector[i] > vector[j];});

    return rank;
}


Tensor<Index, 1> calculate_rank_less(const Tensor<type, 1>& vector)
{
    const Index size = vector.size();

    Tensor<Index, 1> rank(size);
    iota(rank.data(), rank.data() + rank.size(), 0);

    sort(rank.data(),
         rank.data() + rank.size(),
         [&](Index i, Index j){return vector[i] < vector[j];});

    return rank;
}


Index count_greater_than(const vector<Index>& data, const Index& bound)
{
    Index count = 0;

    #pragma omp parallel for reduction(+: count)
    for(Index i = 0; i < Index(data.size()); i++)
        if(data[i] > bound)
            count++;

    return count;
}


vector<Index> get_elements_greater_than(const vector<Index>& data, const Index& bound)
{
    const Index indices_size = count_greater_than(data, bound);

    vector<Index> indices(indices_size);

    Index index = 0;

    for(size_t i  = 0; i < data.size(); i++)
         if(data[i] > bound)
             indices[index++] = data[i];

    return indices;
}


vector<Index> get_elements_greater_than(const vector<vector<Index>>& vectors, const Index& bound)
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


Index count_between(const Tensor<type, 1>& vector,const type& minimum, const type& maximum)
{
    const Index size = vector.size();

    Index count = 0;

    #pragma omp parallel for reduction(+: count)
    for(Index i = 0; i < size; i++)
        if(vector(i) >= minimum && vector(i) <= maximum) 
            count++;

    return count;
}


void set_row(Tensor<type,2>& matrix, const Tensor<type, 1>& new_row, const Index& row_index)
{
    const Index columns_number = new_row.size();

    #pragma omp parallel for    

    for(Index i = 0; i < columns_number; i++)
        matrix(row_index, i) = new_row(i);
}


void set_row(Tensor<type, 2, RowMajor>& matrix, const Tensor<type, 1>& vector, const Index& row_index)
{
    const Index columns_number = vector.size();

    memcpy(matrix.data() + row_index * columns_number, vector.data(), columns_number*sizeof(type));
}


Tensor<type,2> filter_column_minimum_maximum(Tensor<type,2>& matrix, 
                                             const Index& column_index, 
                                             const type& minimum, 
                                             const type& maximum)
{
    const Tensor<type, 1> column = matrix.chip(column_index,1);
    const Index new_rows_number = count_between(column, minimum, maximum);

    if(new_rows_number == 0) return Tensor<type,2>();

    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    bool check_conditions = false;

    Tensor<type,2> new_matrix(new_rows_number, columns_number);

    Index row_index = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(matrix(i, column_index) >= minimum 
        && matrix(i, column_index) <= maximum)
        {
            const Tensor<type, 1> row = matrix.chip(i, 0);

            set_row(new_matrix, row, row_index);

            row_index++;

            check_conditions = true;
        }
    }

    if(!check_conditions)
        throw runtime_error("Invalid conditions\n");

    return new_matrix;
}


type l1_norm(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector)
{
    Tensor<type, 0> norm;

    norm.device(*thread_pool_device) = vector.abs().sum();

    return norm(0);
}


void l1_norm_gradient(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 1>& gradient)
{
    gradient.device(*thread_pool_device) = vector.sign();
}


void l1_norm_hessian(const ThreadPoolDevice*, const Tensor<type, 1>&, Tensor<type, 2>& hessian)
{
    hessian.setZero();
}


type l2_norm(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector)
{
    Tensor<type, 0> norm;

    norm.device(*thread_pool_device) = vector.square().sum().sqrt();

    if(isnan(norm(0)))
        throw runtime_error("OpenNN Exception: l2 norm of vector is not a number.\n");

    return norm(0);
}


void l2_norm_gradient(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 1>& gradient)
{
    const type norm = l2_norm(thread_pool_device, vector);

    if(norm < NUMERIC_LIMITS_MIN)
    {
        gradient.setZero();

        return;
    }

    gradient.device(*thread_pool_device) = vector/norm;
}


void l2_norm_hessian(const ThreadPoolDevice* thread_pool_device, Tensor<type, 1>& vector, Tensor<type, 2>& hessian)
{
    const type norm = l2_norm(thread_pool_device, vector);

    if(norm < NUMERIC_LIMITS_MIN)
    {
        hessian.setZero();

        return;
    }

    hessian = self_kronecker_product(thread_pool_device, vector)/(norm*norm*norm);
}


type l2_distance(const Tensor<type, 1>&x, const Tensor<type, 1>&y)
{
    // @todo add thread pool 

    if(x.size() != y.size())
        throw runtime_error("x and y vector must  have the same dimensions.\n");

    Tensor<type, 0> distance;

    distance = (x-y).square().sum().sqrt();

    return distance(0);
}


type l2_distance(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    // @todo add thread pool 

    Tensor<type, 0> distance;

    distance = (x-y).square().sum().sqrt();

    return distance(0);
}


type l2_distance(const type& x, const type& y)
{
    return type(fabs(x - y));
}


Tensor<type, 1> l2_distance(const Tensor<type, 2>& x, const Tensor<type, 2>& y, const Index& size)
{
    // @todo optimize using thread pool and

    Tensor<type, 1> distance(size);

    const Tensor<type, 2> difference = x - y;

    for(Index i = 0; i < difference.dimension(1); i++)
        distance(i) = abs(difference(i));

    return distance;
}


void set_identity(Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);

    matrix.setZero();

    #pragma omp parallel for
    for(Index i = 0; i < rows_number; i++)
        matrix(i, i) = type(1);
}


void sum_diagonal(Tensor<type, 2>& matrix, const type& value)
{
    const Index rows_number = matrix.dimension(0);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
        matrix(i,i) += value;
}


Tensor<type, 1> perform_Householder_QR_decomposition(const Tensor<type, 2>& A, const Tensor<type, 1>& b)
{
    const Index n = A.dimension(0);

    Tensor<type, 1> x(n);

    const Map<Matrix<type, Dynamic, Dynamic>> A_eigen((type*)A.data(), n, n);

    const Map<Matrix<type, Dynamic, 1>> b_eigen((type*)b.data(), n, 1);
    
    Map<Matrix<type, Dynamic, 1>> x_eigen((type*)x.data(), n);

    x_eigen = A_eigen.colPivHouseholderQr().solve(b_eigen);

    return x;
}


void fill_tensor_data(const Tensor<type, 2>& matrix,
                      const vector<Index>& row_indices,
                      const vector<Index>& column_indices,
                      type* tensor_data)
{
    if(row_indices.empty() || column_indices.empty())
        return;

    const Index rows_number = row_indices.size();
    const Index columns_number = column_indices.size();

    const type* matrix_data = matrix.data();

    #pragma omp parallel for
    for (Index j = 0; j < columns_number; j++)
    {
        const type* matrix_column = matrix_data + matrix.dimension(0) * column_indices[j];

        type* tensor_value = tensor_data + rows_number * j;

        const type* matrix_value = nullptr;

        const Index* rows_indices_data = row_indices.data();

        for (Index i = 0; i < rows_number; i++)
        {
            matrix_value = matrix_column + *rows_indices_data;
            rows_indices_data++;
            *tensor_value = *matrix_value;
            tensor_value++;
        }
    }
}


vector<Index> join_vector_vector(const vector<Index>& x, const vector<Index>& y)
{
    const Index size = x.size() + y.size();

    vector<Index> data(size);

    memcpy(data.data(), x.data(), x.size() * sizeof(Index));

    memcpy(data.data() + x.size(), y.data(), y.size() * sizeof(Index));

    return data;
}


Tensor<type, 2> assemble_vector_vector(const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    const Index rows_number = x.size();
    const Index columns_number = 2;

    Tensor<type, 2> data(rows_number, columns_number);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        data(i, 0) = x(i);
        data(i, 1) = y(i);
    }

    return data;
}


Tensor<type, 2> assemble_vector_matrix(const Tensor<type, 1>& x, const Tensor<type, 2>& y)
{
    const Index rows_number = x.size();
    const Index columns_number = 1 + y.dimension(1);

    Tensor<type, 2> data(rows_number, columns_number);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        data(i, 0) = x(i);

        for(Index j = 0; j < y.dimension(1); j++)
            data(i, 1+j) = y(i, j);
    }

    return data;
}


Tensor<type, 2> assemble_matrix_matrix(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    const Index rows_number = x.dimension(0);
    const Index columns_number = x.dimension(1) + y.dimension(1);

    Tensor<type, 2> data(rows_number, columns_number);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < x.dimension(1); j++)
            data(i, j) = x(i, j);

        for(Index j = 0; j < y.dimension(1); j++)
            data(i, x.dimension(1) + j) = y(i, j);
    }

    return data;
}


string dimensions_to_string(const dimensions& x, const string& separator)
{
    const Index size = x.size();

    ostringstream buffer;

    if(x.size() == 0)
        throw runtime_error("Error: Dimensions size must be greater than 0.\n");

    for(Index i = 0; i < size; i++)
        buffer << x[i] << separator;

    return buffer.str();
}


dimensions string_to_dimensions(const string& x, const string& separator) {

    dimensions result;

    if (x.empty()) {
        throw std::runtime_error("Error: Input string must not be empty.\n");
    }

    size_t start = 0;
    size_t end = x.find(separator);

    while (end != string::npos) {
        string token = x.substr(start, end - start);

        try {
            result.push_back(stoi(token));
        }
        catch (const invalid_argument&) {
            throw runtime_error("Error: Input string contains non-numeric elements.\n");
        }

        start = end + separator.length();
        end = x.find(separator, start);
    }

    if (start < x.size()) {
        string token = x.substr(start);
        try {
            result.push_back(stoi(token));
        }
        catch (const invalid_argument&) {
            throw runtime_error("Error: Input string contains non-numeric elements.\n");
        }
    }

    return result;
}


Tensor<type, 1> string_to_tensor(const string& x, const string& separator) {
    if (x.empty()) {
        throw runtime_error("Error: Input string must not be empty.\n");
    }

    dimensions temp_dimensions;
    size_t start = 0;
    size_t end = x.find(separator);

    while (end != string::npos) {
        string token = x.substr(start, end - start);

        try {
            temp_dimensions.push_back(std::stoi(token));
        }
        catch (const invalid_argument&) {
            throw runtime_error("Error: Input string contains non-numeric elements.\n");
        }

        start = end + separator.length();
        end = x.find(separator, start);
    }

    if (start < x.size()) {
        string token = x.substr(start);
        try {
            temp_dimensions.push_back(stoi(token));
        }
        catch (const invalid_argument&) {
            throw runtime_error("Error: Input string contains non-numeric elements.\n");
        }
    }

    Tensor<type, 1> tensor(temp_dimensions.size());
    for (size_t i = 0; i < temp_dimensions.size(); ++i) {
        tensor(i) = temp_dimensions[i];
    }

    return tensor;
}


string tensor_to_string(const Tensor<type, 1>& x, const string& separator)
{
    const Index size = x.size();

    ostringstream buffer;

    if(x.size() == 0)
        throw runtime_error("Error: Dimensions size must be greater than 0.\n");

    for(Index i = 0; i < size; i++)
        buffer << x[i] << separator;

    return buffer.str();
}


Tensor<type, 2> delete_row(const Tensor<type, 2>& tensor, const Index& row_index)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);

    Tensor<type, 2> new_matrix(rows_number-1, columns_number);

    #pragma omp parallel for
    for(Index i = 0; i < row_index; i++)
        for(Index j = 0; j < columns_number; j++)
            new_matrix(i, j) = tensor(i, j);

    #pragma omp parallel for
    for(Index i = row_index + 1; i < rows_number; i++)
        for(Index j = 0; j < columns_number; j++)
            new_matrix(i-1,j) = tensor(i, j);

    return new_matrix;
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


TensorMap<Tensor<type, 1>> tensor_map(const Tensor<type, 2>& matrix, const Index& column_index)
{
    const TensorMap<Tensor<type, 1>> column((type*) matrix.data() + column_index * matrix.dimension(0),
                                            matrix.dimension(0));

    return column;
}


TensorMap<Tensor<type, 1>> tensor_map_1(const pair<type*, dimensions>& x_pair)
{
    if(x_pair.second.size() != 1)
        throw runtime_error("Dimensions must be 1");

    return TensorMap<Tensor<type, 1>>(x_pair.first,
                                      x_pair.second[0]);
}


TensorMap<Tensor<type, 2>> tensor_map_2(const pair<type*, dimensions>& x_pair)
{
    if(x_pair.second.size() != 2)
        throw runtime_error("Dimensions must be 2");

    return TensorMap<Tensor<type, 2>>(x_pair.first,
                                      x_pair.second[0],
                                      x_pair.second[1]);
}


TensorMap<Tensor<type, 3>> tensor_map_3(const pair<type*, dimensions>& x_pair)
{
    if(x_pair.second.size() != 3){
        cout<<x_pair.second.size()<<endl;
        throw runtime_error("Dimensions must be 3");}

    return TensorMap<Tensor<type, 3>>(x_pair.first,
                                      x_pair.second[0],
                                      x_pair.second[1],
                                      x_pair.second[2]);
}


TensorMap<Tensor<type, 4>> tensor_map_4(const pair<type*, dimensions>& x_pair)
{
    return TensorMap<Tensor<type, 4>>(x_pair.first,
                                      x_pair.second[0],
                                      x_pair.second[1],
                                      x_pair.second[2],
                                      x_pair.second[3]);
}


void print_pairs(const vector<pair<string, Index>>& pairs)
{
    for (size_t i = 0; i < pairs.size(); i++)
        cout << pairs[i].first << ": " << pairs[i].second << endl;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
