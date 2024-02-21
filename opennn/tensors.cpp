//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   U T I L I T I E S   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "tensors.h"

#define GET_VARIABLE_NAME(Variable) (#Variable)

namespace opennn
{
/**
<<<<<<< HEAD:opennn/tensor_utilities.cpp
void update_progress_bar(type progress) {
    const int barWidth = 50;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%\r";
    std::cout.flush();
}
*/

type calculate_random_uniform(const type& minimum, const type& maximum)
{
    const type random = type(rand()/(RAND_MAX+1.0));

    const type random_uniform = minimum + (maximum - minimum) * random;

    return random_uniform;
}


///Generate a random boolean

bool calculate_random_bool()
{
    if(rand() % 2 == 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}


void initialize_sequential(Tensor<type, 1>& vector)
{
    #pragma omp parallel for

    for(Index i = 0; i < vector.size(); i++) 
        vector(i) = type(i);
}


void initialize_sequential(Tensor<Index, 1>& vector)
{
    #pragma omp parallel for

    for(Index i = 0; i < vector.size(); i++) 
        vector(i) = i;
}


void initialize_sequential(Tensor<Index, 1>& new_tensor,
                           const Index& start, const Index& step, const Index& end)
{
    const Index new_size = (end-start)/step+1;

    new_tensor.resize(new_size);
    new_tensor(0) = start;

    for(Index i = 1; i < new_size-1; i++)
    {
        new_tensor(i) = new_tensor(i-1) + step;
    }

    new_tensor(new_size-1) = end;
}


void multiply_rows(Tensor<type, 2>& matrix, const Tensor<type, 1>& vector)
{
    const Index rows_number = matrix.dimension(0);
    const Index raw_variables_number = matrix.dimension(1);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < raw_variables_number; j++)
        {
           matrix(i,j) *= vector(j);
        }
    }
}


void batch_matrix_multiplication(ThreadPoolDevice* thread_pool_device, const Tensor<type, 4>& A, const Tensor<type, 4>& B, Tensor<type, 4>& C)
{
    // Assumes A, B & C share dimensions 2 & 3 and A.dimension(1) == B.dimension(0) (the contraction axes)
    // The other 2 dimensions of C will be C.dimension(0) == A.dimension(0) & B.dimension(1) == C.dimension(1)

    Index channels_number = A.dimension(2);
    Index blocks_number = A.dimension(3);

    Index A_rows_number = A.dimension(0);
    Index B_raw_variables_number = B.dimension(1);

    Index product_dimension = A.dimension(1);

    for (Index i = 0; i < blocks_number; i++)
    {
        for (Index j = 0; j < channels_number; j++)
        {
            const TensorMap<Tensor<type, 2>> A_matrix((type*)A.data() + j * A_rows_number*product_dimension + i * A_rows_number*product_dimension*channels_number,
                A_rows_number, product_dimension);

            const TensorMap<Tensor<type, 2>> B_matrix((type*)B.data() + j * product_dimension*B_raw_variables_number + i * product_dimension*B_raw_variables_number*channels_number,
                product_dimension, B_raw_variables_number);

            TensorMap<Tensor<type, 2>> C_matrix(C.data(), A_rows_number, B_raw_variables_number);

            C_matrix.device(*thread_pool_device) = A_matrix.contract(B_matrix, A_B);
        }
    }
}


void batch_matrix_multiplication(ThreadPoolDevice* thread_pool_device, const Tensor<type, 4>& A, const Tensor<type, 3>& B, Tensor<type, 4>& C)
{
    // Assumes A & C share dimensions 2 & 3, B.dimension(2) == A.dimension(3) and A.dimension(1) == B.dimension(0) (the contraction axes)
    // The other 2 dimensions of C will be C.dimension(0) == A.dimension(0) & B.dimension(1) == C.dimension(1)

    Index channels_number = A.dimension(2);
    Index blocks_number = A.dimension(3);

    Index A_rows_number = A.dimension(0);
    Index B_raw_variables_number = B.dimension(1);

    Index product_dimension = A.dimension(1);

    for (Index i = 0; i < blocks_number; i++)
    {
        const TensorMap<Tensor<type, 2>> B_matrix((type*)B.data() + i * product_dimension * B_raw_variables_number,
            product_dimension, B_raw_variables_number);

        const TensorMap<Tensor<type, 3>> A_block((type*)A.data() + i * A_rows_number*product_dimension*channels_number,
            A_rows_number, product_dimension, channels_number);

        TensorMap<Tensor<type, 3>> C_block(C.data() + i * A_rows_number*B_raw_variables_number*channels_number,
            A_rows_number, B_raw_variables_number, channels_number);

        C_block.device(*thread_pool_device) = A_block.contract(B_matrix, A_B);
    }
}


void batch_matrix_multiplication(ThreadPoolDevice* thread_pool_device, const Tensor<type, 4>& A, const Tensor<type, 3>& B, Tensor<type, 3>& C)
{
    // Assumes A, B & C share dimensions their last 2 dimensions and A.dimension(1) == B.dimension(0) (the contraction axes)
    // The other dimension of C will be C.dimension(0) == A.dimension(0)

    Index channels_number = A.dimension(2);
    Index blocks_number = A.dimension(3);

    Index A_rows_number = A.dimension(0);

    Index product_dimension = A.dimension(1);

    for (Index i = 0; i < blocks_number; i++)
    {
        for (Index j = 0; j < channels_number; j++)
        {
            const TensorMap<Tensor<type, 2>> A_matrix((type*)A.data() + j * A_rows_number*product_dimension + i * A_rows_number*product_dimension*channels_number,
                A_rows_number, product_dimension);

            const TensorMap<Tensor<type, 1>> B_vector((type*)B.data() + j * product_dimension + i * product_dimension*channels_number,
                product_dimension);

            TensorMap<Tensor<type, 1>> C_vector(C.data() + j * A_rows_number + i * A_rows_number*channels_number,
                A_rows_number);

            C_vector.device(*thread_pool_device) = A_matrix.contract(B_vector, A_B);
        }
    }
}


void self_kronecker_product(ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, TensorMap<Tensor<type, 2>>& matrix)
{
    Index columns_number = vector.size();

#pragma omp parallel for

    for (Index i = 0; i < columns_number; i++)
    {
        TensorMap<Tensor<type, 1>>  column = tensor_map(matrix, i);

        column.device(*thread_pool_device) = vector * vector(i);
    }
}


void divide_columns(ThreadPoolDevice* thread_pool_device, Tensor<type, 2>& matrix, const Tensor<type, 1>& vector)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

#pragma omp parallel for

    for(Index j = 0; j < columns_number; j++)
    {
        TensorMap<Tensor<type,1>> column(matrix.data() + j*rows_number, rows_number);

        column.device(*thread_pool_device) = column / vector(j);
    }
}


void divide_matrices(ThreadPoolDevice* thread_pool_device, Tensor<type, 3>& tensor, const Tensor<type, 2>& matrix)
{
    const Index rows_number = tensor.dimension(0);
    const Index columns_number = tensor.dimension(1);
    const Index channels_number = tensor.dimension(2);

#pragma omp parallel for

    for (Index j = 0; j < channels_number; j++)
    {
        TensorMap<Tensor<type, 2>> channel(tensor.data() + j * rows_number*columns_number, rows_number, columns_number);

        channel.device(*thread_pool_device) = channel / matrix;
    }
}


void sum_columns(ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    for(Index i = 0; i < columns_number; i++)
    {
        TensorMap<Tensor<type,1>> column(matrix.data() + i*rows_number, rows_number);

        column.device(*thread_pool_device) = column + vector(i);
    }
}


void sum_matrices(ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 3>& tensor)
{
    const Index rows_number = tensor.dimension(0);
    const Index raw_variables_number = tensor.dimension(1);
    const Index channels_number = tensor.dimension(2);

    for(Index i = 0; i < channels_number; i++)
    {
        TensorMap<Tensor<type,2>> matrix(tensor.data() + i*rows_number*raw_variables_number, rows_number, raw_variables_number);

        matrix.device(*thread_pool_device) = matrix + vector(i);
    }
}


bool is_zero(const Tensor<type, 1>& tensor)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
    {
        if(abs(tensor[i]) > type(NUMERIC_LIMITS_MIN)) return false;
    }

    return true;
}


bool is_zero(const Tensor<type,1>& tensor,const type& limit)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
    {
        if(abs(tensor[i]) > type(limit)) return false;
    }

    return true;
}


bool is_false(const Tensor<bool, 1>& tensor)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
    {
        if(tensor(i)) return false;
    }

    return true;
}


Index count_true(const Tensor<bool, 1>& tensor)
{
    Index trueCount = 0;

#pragma omp parallel for

    for(int i = 0; i < tensor.size(); ++i)
    {
        if(tensor(i))
        {
            trueCount++;
        }
    }

    return trueCount;
}


bool is_binary(const Tensor<type, 2>& matrix)
{
    const Index size = matrix.size();

    for(Index i = 0; i < size; i++)
    {
        if(matrix(i) != type(0) && matrix(i) != type(1) && !isnan(matrix(i)) ) return false;
    }

    return true;
}


bool is_constant(const Tensor<type, 1>& vector)
{
    const Index size = vector.size();

    type first_not_nan_element = type(0);

    for(Index i = 0; i < size; i++)
    {
        if(isnan(vector(i)))
        {
            continue;
        }
        else
        {
            first_not_nan_element = vector(i);
            break;
        }
    }

    for(Index i = 0; i < size; i++)
    {
        if(isnan(vector(i))) continue;

        if(abs(first_not_nan_element - vector(i)) > numeric_limits<float>::min()) return false;
    }

    return true;
}


bool is_equal(const Tensor<type, 2>& matrix, const type& value, const type& tolerance)
{
    const Index size = matrix.size();

    for(Index i = 0; i < size; i++)
    {
        if(abs(matrix(i) - value) > tolerance) return false;
    }

    return true;
}


bool are_equal(const Tensor<type, 1>& vector_1, const Tensor<type, 1>& vector_2, const type& tolerance)
{
    const Index size = vector_1.size();

    for(Index i = 0; i < size; i++)
    {
        if(abs(vector_1(i) - vector_2(i)) > tolerance) return false;
    }

    return true;
}


bool are_equal(const Tensor<bool, 1>& vector_1, const Tensor<bool, 1>& vector_2)
{
    const Index size = vector_1.size();

    for(Index i = 0; i < size; i++)
    {
        if( vector_1(i) != vector_2(i)) return false;
    }

    return true;
}


bool are_equal(const Tensor<type, 2>& matrix_1, const Tensor<type, 2>& matrix_2, const type& tolerance)
{
    const Index size = matrix_1.size();

    for(Index i = 0; i < size; i++)
    {
        if(abs(matrix_1(i) - matrix_2(i)) > tolerance) return false;
    }

    return true;
}

bool are_equal(const Tensor<bool, 2>& matrix_1, const Tensor<bool, 2>& matrix_2)
{
    const Index size = matrix_1.size();

    for(Index i = 0; i < size; i++)
    {
        if( matrix_1(i) != matrix_2(i)) return false;
    }

    return true;
}


Tensor<bool, 2> elements_are_equal(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    if(x.size() != y.size() || x.dimension(0) != y.dimension(0) || x.dimension(1) != y.dimension(1))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Tensor utilities class.\n"
            << "Tensor<bool, 2> elements_are_equal(const Tensor<type, 2>& x, const Tensor<type, 2>& y) method.\n"
            << "Input vectors must have equal sizes.\n";

        throw runtime_error(buffer.str());
    }

    Tensor<bool, 2> result(x.dimension(0), x.dimension(1));

    #pragma omp parallel for

    for(int i = 0; i < x.size(); i++) 
    { 
        result(i) = (x(i) == y(i)); 
    }

    return result;
}


void save_csv(const Tensor<type,2>& data, const string& filename)
{
    ofstream file(filename);

    if(!file.is_open())
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template." << endl
             << "void save_csv(const Tensor<type,2>&, const string&) method." << endl
             << "Cannot open matrix data file: " << filename << endl;

      throw runtime_error(buffer.str());
    }

    file.precision(20);

    const Index data_rows = data.dimension(0);
    const Index data_raw_variables = data.dimension(1);

    char separator_char = ';';

    for(Index i = 0; i < data_rows; i++)
    {
       for(Index j = 0; j < data_raw_variables; j++)
       {
           file << data(i,j);

           if(j != data_raw_variables-1)
           {
               file << separator_char;
           }
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

/**
<<<<<<< HEAD:opennn/tensor_utilities.cpp
Tensor<type, 2> box_plots_to_tensor(const Tensor<BoxPlot, 1>& box_plots)
{
   const Index columns_number = box_plots.dimension(0);

   Tensor<type, 2> summary(5, columns_number);

   for(Index i = 0; i < columns_number; i++)
   {
       const BoxPlot& box_plot = box_plots(i);
       summary(0, i) = box_plot.minimum;
       summary(1, i) = box_plot.first_quartile;
       summary(2, i) = box_plot.median;
       summary(3, i) = box_plot.third_quartile;
       summary(4, i) = box_plot.maximum;
   }

   Eigen::array<Index, 2> new_shape = {1, 5 * columns_number};
   Tensor<type, 2> reshaped_summary = summary.reshape(new_shape);

   return reshaped_summary;
}
*/

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


void scrub_missing_values(Tensor<type, 2>& matrix, const type& value)
{
    replace_if(matrix.data(), matrix.data()+matrix.size(), [](type x){return isnan(x);}, value);
}


Tensor<string, 1> sort_by_rank(const Tensor<string,1>&tokens, const Tensor<Index,1>&rank)
{
    const Index tokens_size = tokens.size();

    if(tokens_size != rank.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Strings Class.\n"
               << "Tensor<string, 1> sort_by_rank(const Tensor<string,1>&tokens, const Tensor<Index,1>&rank) method.\n"
               << "Tokens and rank size must be the same.\n";

        throw runtime_error(buffer.str());
    }

    Tensor<string,1> sorted_tokens(tokens_size);

    #pragma omp parallel for

    for(Index i = 0; i < tokens_size; i++)
    {
        sorted_tokens(i) = tokens(rank(i));
    }

    return sorted_tokens;
};



Tensor<Index, 1> sort_by_rank(const Tensor<Index,1>&tokens, const Tensor<Index,1>&rank)
{
    const Index tokens_size = tokens.size();

    if(tokens_size != rank.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Strings Class.\n"
               << "Tensor<string, 1> sort_by_rank(const Tensor<string,1>&tokens, const Tensor<Index,1>&rank) method.\n"
               << "Tokens and rank size must be the same.\n";

        throw runtime_error(buffer.str());
    }

    Tensor<Index,1> sorted_tokens(tokens_size);

    #pragma omp parallel for

    for(Index i = 0; i < tokens_size; i++)
    {
        sorted_tokens(i) = tokens(rank(i));
    }

    return sorted_tokens;
};



Index count_less_than(const Tensor<Index,1>& vector, const Index& bound)
{
    Index count = 0;

    for(Index i = 0; i < vector.size(); i++)
    {
        if(vector(i) < bound)
            count++;
    }

    return count;
};



Tensor<Index, 1> get_indices_less_than(const Tensor<Index,1>& vector, const Index& bound)
{
    const Index indices_size = count_less_than(vector, bound);

    Tensor<Index, 1> indices(indices_size);

    Index index = 0;

    for(Index i  = type(0); i < vector.size(); i++)
    {
         if(vector(i) < bound)
         {
             indices(index) = i;
             index++;
         }
    }

    return indices;
};



Index count_less_than(const Tensor<double,1>& vector, const double& bound)
{
    Index count = 0;

    for(Index i = 0; i < vector.size(); i++)
    {
        if(vector(i) < bound)
            count++;
    }

    return count;
};



Tensor<Index, 1> get_indices_less_than(const Tensor<double,1>& vector, const double& bound)
{
    const Index indices_size = count_less_than(vector, bound);

    Tensor<Index, 1> indices(indices_size);

    Index index = 0;

    for(Index i  = type(0); i < vector.size(); i++)
    {
         if(vector(i) < bound)
         {
             indices(index) = i;
             index++;
         }
    }

    return indices;
};


Index count_greater_than(const Tensor<Index,1>& vector, const Index& bound)
{
    Index count = 0;

    for(Index i = 0; i < vector.size(); i++)
    {
        if(vector(i) > bound)
            count++;
    }

    return count;
};


Tensor<Index, 1> get_elements_greater_than(const Tensor<Index,1>& vector, const Index& bound)
{
    const Index indices_size = count_greater_than(vector, bound);

    Tensor<Index, 1> indices(indices_size);

    Index index = 0;

    for(Index i  = type(0); i < vector.size(); i++)
    {
         if(vector(i) > bound)
         {
             indices(index) = vector(i);
             index++;
         }
    }

    return indices;
}


Tensor<Index, 1> get_elements_greater_than(const Tensor<Tensor<Index, 1>,1>& vectors, const Index& bound)
{
    const Index vectors_number = vectors.size();

    Tensor<Index, 1> indices(0);

    for(Index i = 0; i < vectors_number; i++)
    {
        Tensor<Index, 1> indices_vector = get_elements_greater_than(vectors(i), bound);

        indices = join_vector_vector(indices, indices_vector);
    }

    return indices;
}


void delete_indices(Tensor<string,1>& vector, const Tensor<Index,1>& indices)
{
    const Index original_size = vector.size();

    const Index new_size = vector.size() - indices.size();

    Tensor<string,1> vector_copy(vector);

    vector.resize(new_size);

    Index index = 0;

    for(Index i = 0; i < original_size; i++)
    {
        if( !contains(indices, i) )
        {
            vector(index) = vector_copy(i);
            index++;
        }
    }
}



void delete_indices(Tensor<Index,1>& vector, const Tensor<Index,1>& indices)
{
    const Index original_size = vector.size();

    const Index new_size = vector.size() - indices.size();

    Tensor<Index,1> vector_copy(vector);

    vector.resize(new_size);

    Index index = 0;

    for(Index i = 0; i < original_size; i++)
    {
        if( !contains(indices, i) )
        {
            vector(index) = vector_copy(i);
            index++;
        }
    }
}


void delete_indices(Tensor<double,1>& vector, const Tensor<Index,1>& indices)
{
    const Index original_size = vector.size();

    const Index new_size = vector.size() - indices.size();

    Tensor<double,1> vector_copy(vector);

    vector.resize(new_size);

    Index index = 0;

    for(Index i = 0; i < original_size; i++)
    {
        if( !contains(indices, i) )
        {
            vector(index) = vector_copy(i);
            index++;
        }
    }
};



Tensor<string, 1> get_first(const Tensor<string,1>& vector, const Index& index)
{
    Tensor<string, 1> new_vector(index);

//    copy(new_vector.data(), new_vector.data() + index, vector.data());

    return new_vector;
};



Tensor<Index, 1> get_first(const Tensor<Index,1>& vector, const Index& index)
{
    Tensor<Index, 1> new_vector(index);

//    copy(new_vector.data(), new_vector.data() + index, vector.data());

    return new_vector;
};



/// Returns the number of elements which are equal or greater than a minimum given value
/// and equal or less than a maximum given value.
/// @param minimum Minimum value.
/// @param maximum Maximum value.

Index count_between(const Tensor<type,1>& vector,const type& minimum, const type& maximum)
{
    const Index size = vector.size();

    Index count = 0;

    for(Index i = 0; i < size; i++)
    {
        if(vector(i) >= minimum && vector(i) <= maximum) 
            count++;
    }

    return count;
}


void get_row(Tensor<type, 1>&, const Tensor<type, 2, RowMajor>&, const Index&)
{
}


void set_row(Tensor<type,2>& matrix, Tensor<type,1>& new_row, const Index& row_index)
{
    const Index raw_variables_number = new_row.size();

    #pragma omp parallel for    

    for(Index i = 0; i < raw_variables_number; i++)
    {
        matrix(row_index,i) = new_row(i);
    }
}


void set_row(Tensor<type, 2, RowMajor>& matrix, const Tensor<type, 1>& vector, const Index& row_index)
{
/*
    copy(execution::par,
        matrix.data(),
        vector.data() + vector.size(),
        matrix.data() + row_index);
*/
}

Tensor<type,2> filter_raw_variable_minimum_maximum(Tensor<type,2>& matrix,const Index& raw_variable_index, const type& minimum, const type& maximum)
{
    const Tensor<type,1> column = matrix.chip(raw_variable_index,1);
    const Index new_rows_number = count_between(column, minimum, maximum);

    if(new_rows_number == 0)
    {
        return Tensor<type,2>();
    }

    const Index rows_number = matrix.dimension(0);
    const Index raw_variables_number = matrix.dimension(1);

    bool check_conditions = false;

    Tensor<type,2> new_matrix(new_rows_number, raw_variables_number);

    Index row_index = 0;
    Tensor<type,1> row(raw_variables_number);

    for(Index i = 0; i < rows_number; i++)
    {
        if(matrix(i,raw_variable_index) >= minimum && matrix(i, raw_variable_index) <= maximum)
        {
            row = matrix.chip(i, 0);

            set_row(new_matrix, row, row_index);

            row_index++;

            check_conditions = true;
        }
    }

    if(!check_conditions)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TensorUtilities class.\n"
               << "Tensor<type,2> filter_raw_variable_minimum_maximum(Tensor<type,2>&,const Index&,const type&,const type&) method.\n"
               << "Invalid conditions\n";

        throw runtime_error(buffer.str());
    }

    return new_matrix;
};


Tensor<type, 2> kronecker_product(const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    type* x_data = (type*)x.data();
    type* y_data = (type*)y.data();

    // Transform Tensors into Dense matrix

    const auto ml = Map<Matrix<type, Dynamic, Dynamic, RowMajor>>(x_data, x.dimension(0), 1);

    const auto mr = Map<Matrix<type, Dynamic, Dynamic, RowMajor>>(y_data, y.dimension(0), 1);

    // Kronecker Product

    auto product = kroneckerProduct(ml, mr).eval();

    // Matrix into a Tensor

    const TensorMap<Tensor<type, 2>> direct_matrix(product.data(), x.size(), y.size());

    return direct_matrix;
}


void kronecker_product(const Tensor<type, 1>& x, Tensor<type, 2>& y)
{
    const Index n = x.dimension(0);
    
    type* x_data = (type*)x.data();
    type* y_data = (type*)y.data();

    const auto x_matrix = Map<Matrix<type, Dynamic, Dynamic>>(x_data, n, 1);

    auto product = Map<Matrix<type, Dynamic, Dynamic>>(y_data, n*n, 1);

    product = kroneckerProduct(x_matrix, x_matrix).eval();
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


/// Returns the l2 norm of a vector.

type l2_norm(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector)
{
    Tensor<type, 0> norm;

    norm.device(*thread_pool_device) = vector.square().sum().sqrt();

    if(isnan(norm(0)))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: l2 norm of vector is not a number."
               << endl;

        throw runtime_error(buffer.str());
    }

    return norm(0);
}


void l2_norm_gradient(const ThreadPoolDevice* thread_pool_device, const Tensor<type, 1>& vector, Tensor<type, 1>& gradient)
{
    const type norm = l2_norm(thread_pool_device, vector);

    if(norm < type(NUMERIC_LIMITS_MIN))
    {
        gradient.setZero();

        return;
    }

    gradient.device(*thread_pool_device) = vector/norm;
}


void l2_norm_hessian(const ThreadPoolDevice* thread_pool_device, Tensor<type, 1>& vector, Tensor<type, 2>& hessian)
{
    const type norm = l2_norm(thread_pool_device, vector);

    if(norm < type(NUMERIC_LIMITS_MIN))
    {
        hessian.setZero();

        return;
    }

    hessian.device(*thread_pool_device) = kronecker_product(vector, vector)/(norm*norm*norm);
}


type l2_distance(const Tensor<type, 1>&x, const Tensor<type, 1>&y)
{
    if(x.size() != y.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Tensor utilites.\n"
               << "type l2_distance(const Tensor<type, 1>&, const Tensor<type, 1>&)\n"
               << "x and y vector must  have the same dimensions.\n";

        throw runtime_error(buffer.str());
    }

    Tensor<type, 0> distance;

    distance = (x-y).square().sum().sqrt();

    return distance(0);
}


type l2_distance(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    Tensor<type, 0> distance;

    distance = (x-y).square().sum().sqrt();

    return distance(0);
}


type l2_distance(const type& x, const type& y)
{
    const type distance = type(fabs(x - y));

    return distance;
}


Tensor<type, 1> l2_distance(const Tensor<type, 2>&x, const Tensor<type, 2>&y, const Index& size)
{
    Tensor<type, 1> distance(size);

    const Tensor<type, 2> difference = x - y;

    for(Index i = 0; i < difference.dimension(1); i++)
    {
        distance(i) = abs(difference(i));
    }

    return distance;
}


void sum_diagonal(Tensor<type, 2>& matrix, const type& value)
{
    const Index rows_number = matrix.dimension(0);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
        matrix(i,i) += value;
}


void sum_diagonal(Tensor<type, 2>& matrix, const Tensor<type, 1>& values)
{
    const Index rows_number = matrix.dimension(0);

    #pragma omp parallel for

    for (Index i = 0; i < rows_number; i++)
        matrix(i, i) += values(i);
}


void sum_diagonal(TensorMap<Tensor<type, 2>>& matrix, const Tensor<type, 1>& values)
{
    const Index rows_number = matrix.dimension(0);

#pragma omp parallel for

    for (Index i = 0; i < rows_number; i++)
        matrix(i, i) += values(i);
}


/// Uses Eigen to solve the system of equations by means of the Householder QR decomposition.

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


void fill_submatrix(const Tensor<type, 2>& matrix,
                    const Tensor<Index, 1>& rows_indices,
                    const Tensor<Index, 1>& raw_variables_indices,
                    type* submatrix)
{
    const Index rows_number = rows_indices.size();
    const Index raw_variables_number = raw_variables_indices.size();

    const type* matrix_data = matrix.data();

    #pragma omp parallel for

    for(Index j = 0; j < raw_variables_number; j++)
    {
        const type* matrix_raw_variable = matrix_data + matrix.dimension(0)*raw_variables_indices[j];
        type* submatrix_raw_variable = submatrix + rows_number*j;

        const type* value = nullptr;

        const Index* rows_indices_data = rows_indices.data();

        for(Index i = 0; i < rows_number; i++)
        {
            value = matrix_raw_variable + *rows_indices_data;
            rows_indices_data++;
            *submatrix_raw_variable = *value;
            submatrix_raw_variable++;
        }
    }
}


Index count_NAN(const Tensor<type, 1>& x)
{
    Index NAN_number = 0;

    #pragma omp parallel for

    for(Index i = 0; i < x.size(); i++)
    {
        if(isnan(x(i))) NAN_number++;
    }

    return NAN_number;
}


Index count_NAN(const Tensor<type, 2>& x)
{
    const Index rows_number = x.dimension(0);
    const Index raw_variables_number = x.dimension(1);

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index row_index = 0; row_index < rows_number; row_index++)
    {
        for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
        {
            if(isnan(x(row_index, raw_variable_index))) count++;
        }
    }

    return count;
}


bool has_NAN(const Tensor<type, 1>& x)
{
    for(Index i = 0; i < x.size(); i++)
    {
        if(isnan(x(i))) return true;
    }

    return false;
}


bool has_NAN(Tensor<type, 2>& x)
{
    for(Index i = 0; i < x.size(); i++)
    {
        if(isnan(x(i))) return true;
    }

    return false;
}


Index count_empty(const Tensor<string, 1>& vector)
{
    const Index words_number = vector.size();

    Index count = 0;

    for( Index i = 0; i < words_number; i++)
    {
        string str = vector(i);

        trim(str);
        
        if(str.empty()) count++;
    }

    return count;
}


void check_size(const Tensor<type, 1>& vector, const Index& size, const string& log)
{
    if(vector.size() != size)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: " << log <<  endl
               << "Size of vector is " << vector.size() << ", but must be " << size << "." << endl;

        throw runtime_error(buffer.str());
    }
}


void check_dimensions(const Tensor<type, 2>& matrix, const Index& rows_number, const Index& raw_variables_number, const string& log)
{
    if(matrix.dimension(0) != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: " << log <<  endl
               << "Number of rows in matrix is " << matrix.dimension(0) << ", but must be " << rows_number << "." <<  endl;

        throw runtime_error(buffer.str());
    }

    if(matrix.dimension(1) != raw_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: " << log <<  endl
               << "Number of raw_variables in matrix is " << matrix.dimension(0) << ", but must be " << raw_variables_number << "." <<  endl;

        throw runtime_error(buffer.str());
    }
}


void check_raw_variables_number(const Tensor<type, 2>& matrix, const Index& raw_variables_number, const string& log)
{
    if(matrix.dimension(1) != raw_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: " << log <<  endl
               << "Number of raw_variables in matrix is " << matrix.dimension(0) << ", but must be " << raw_variables_number << "." <<  endl;

        throw runtime_error(buffer.str());
    }
}


void check_rows_number(const Tensor<type, 2>& matrix, const Index& rows_number, const string& log)
{
    if(matrix.dimension(1) != rows_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: " << log <<  endl
               << "Number of raw_variables in matrix is " << matrix.dimension(0) << ", but must be " << rows_number << "." <<  endl;

        throw runtime_error(buffer.str());
    }
}


Tensor<Index, 1> join_vector_vector(const Tensor<Index, 1>& x, const Tensor<Index, 1>& y)
{
    const Index size = x.size() + y.size();

    Tensor<Index, 1> data(size);

    copy(execution::par, 
         x.data(), x.data() + x.size(), data.data());

    copy(execution::par, 
         y.data(), y.data() + y.size(), data.data() + x.size());

    return data;
}


Tensor<type, 2> assemble_vector_vector(const Tensor<type, 1>& x, const Tensor<type, 1>& y)
{
    const Index rows_number = x.size();
    const Index raw_variables_number = 2;

    Tensor<type, 2> data(rows_number, raw_variables_number);

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
    const Index raw_variables_number = 1 + y.dimension(1);

    Tensor<type, 2> data(rows_number, raw_variables_number);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        data(i, 0) = x(i);

        for(Index j = 0; j < y.dimension(1); j++)
        {
            data(i, 1+j) = y(i,j);
        }
    }

    return data;
}


Tensor<type, 2> assemble_matrix_vector(const Tensor<type, 2>& x, const Tensor<type, 1>& y)
{
    const Index rows_number = y.size();
    const Index raw_variables_number = x.dimension(1) + 1;

    Tensor<type, 2> data(rows_number, raw_variables_number);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < x.dimension(1); j++)
        {
            data(i, j) = x(i,j);
        }

        data(i, raw_variables_number-1) = y(i);
    }

    return data;
}


Tensor<type, 2> assemble_matrix_matrix(const Tensor<type, 2>& x, const Tensor<type, 2>& y)
{
    const Index rows_number = x.dimension(0);
    const Index raw_variables_number = x.dimension(1) + y.dimension(1);

    Tensor<type, 2> data(rows_number, raw_variables_number);

    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < x.dimension(1); j++)
        {
            data(i,j) = x(i,j);
        }

        for(Index j = 0; j < y.dimension(1); j++)
        {
            data(i, x.dimension(1) + j) = y(i,j);
        }
    }

    return data;
}


Tensor<string, 1> assemble_text_vector_vector(const Tensor<string, 1>& x, const Tensor<string, 1>& y)
{
    const Index x_size = x.size();
    const Index y_size = y.size();

    Tensor<string,1> data(x_size + y_size);

    #pragma omp parallel for

    for(Index i = 0; i < x_size; i++)
    {
        data(i) = x(i);
    }

    #pragma omp parallel for

    for(Index i = 0; i < y_size; i++)
    {
        data(i + x_size) = y(i);
    }

    return data;
}


string string_tensor_to_string(const Tensor<string,1>&x, const string& separator)
{
    const Index size = x.size();

    if(x.size() == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Tensor utilites.\n"
              << "Mstring to_string(Tensor<string,1>&x,string& separator).\n"
              << "Input vector must have dimension greater than 0.\n";

       throw runtime_error(buffer.str());
    }

    string line = x(0);

    for(Index i = 1; i < size; i++)
    {
        line = line + separator+ x(i);
    }

    return line;

}


Tensor<type, 2> delete_row(const Tensor<type, 2>& tensor, const Index& row_index)
{
    const Index rows_number = tensor.dimension(0);
    const Index raw_variables_number = tensor.dimension(1);
   #ifdef OPENNN_DEBUG

   if(row_index > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> delete_row(const size_t&) const.\n"
             << "Index of row must be less than number of rows.\n"
             << "row index: " << row_index << "rows_number" << rows_number << "\n";

      throw runtime_error(buffer.str());
   }
   else if(rows_number < 2)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix Template.\n"
             << "Matrix<T> delete_row(const size_t&) const.\n"
             << "Number of rows must be equal or greater than two.\n";

      throw runtime_error(buffer.str());
   }

   #endif

   Tensor<type, 2> new_matrix(rows_number-1, raw_variables_number);

    #pragma omp parallel for

   for(Index i = 0; i < row_index; i++)
   {
      for(Index j = 0; j < raw_variables_number; j++)
      {
        new_matrix(i,j) = tensor(i,j);
      }
   }

    #pragma omp parallel for

   for(Index i = row_index+1; i < rows_number; i++)
   {
      for(Index j = 0; j < raw_variables_number; j++)
      {
         new_matrix(i-1,j) = tensor(i,j);
      }
   }

   return new_matrix;
}


/// Returns true if any value is less or equal than a given value, and false otherwise.

bool is_less_than(const Tensor<type, 1>& column, const type& value)
{
    const Tensor<bool, 1> if_sentence = (column <= column.constant(value));

    Tensor<bool, 1> sentence(column.size());
    sentence.setConstant(true);

    Tensor<bool, 1> else_sentence(column.size());
    else_sentence.setConstant(false);

    const Tensor<bool, 0> is_less = (if_sentence.select(sentence, else_sentence)).any();

    return is_less(0);
}


bool contains(const Tensor<size_t,1>& vector, const size_t& value)
{
    Tensor<size_t, 1> copy(vector);

    const size_t* it = find(copy.data(), copy.data()+copy.size(), value);

    return it != (copy.data()+copy.size());
}


bool contains(const Tensor<type,1>& vector, const type& value)
{
    Tensor<type, 1> copy(vector);

    const type* it = find(copy.data(), copy.data()+copy.size(), value);

    return it != (copy.data()+copy.size());
}


bool contains(const Tensor<Index,1>& vector, const Index& value)
{
    Tensor<Index, 1> copy(vector);

    const Index* it = find(copy.data(), copy.data()+copy.size(), value);

    return it != (copy.data()+copy.size());
}


bool contains(const Tensor<string,1>& vector, const string& value)
{
    Tensor<string, 1> copy(vector);

    const string* it = find(copy.data(), copy.data()+copy.size(), value);

    return it != (copy.data()+copy.size());
};


void push_back_index(Tensor<Index, 1>& old_vector, const Index& new_element)
{
    const Index old_size = old_vector.size();

    const Index new_size = old_size+1;

    Tensor<Index, 1> new_vector(new_size);

    #pragma omp parallel for

    for(Index i = 0; i < old_size; i++) 
        new_vector(i) = old_vector(i);

    new_vector(new_size-1) = new_element;

    old_vector = new_vector;
}


void push_back_string(Tensor<string, 1>& old_vector, const string& new_string)
{
    const Index old_size = old_vector.size();

    const Index new_size = old_size+1;

    Tensor<string, 1> new_vector(new_size);

    #pragma omp parallel for

    for(Index i = 0; i < old_size; i++) 
        new_vector(i) = old_vector(i);

    new_vector(new_size-1) = new_string;

    old_vector = new_vector;
}


void push_back_type(Tensor<type, 1>& vector, const type& new_value)
{
    const Index old_size = vector.size();

    const Index new_size = old_size+1;

    Tensor<type, 1> new_vector(new_size);

    #pragma omp parallel for

    for(Index i = 0; i < old_size; i++) 
        new_vector(i) = vector(i);

    new_vector(new_size-1) = new_value;

    vector = new_vector;
}


Tensor<string, 1> to_string_tensor(const Tensor<type,1>& x)
{
    Tensor<string, 1> vector(x.size());

    #pragma omp parallel for

    for(Index i = 0; i < x.size(); i++)
    {
        vector(i) = std::to_string(x(i));
    }

    return vector;
};


void print_tensor(const float* vector, const int dimensions[])
{
    cout<<"Tensor"<<endl;

    const int rows_number = dimensions[0];
    const int raw_variables_number = dimensions[1];
    const int channels = dimensions[2];
    const int batch = dimensions[3];

    for(int l = 0; l<batch; l++)
    {
        for(int k = 0; k<channels; k++)
        {
            for(int i = 0; i<rows_number; i++)
            {
                for(int j = 0; j<raw_variables_number; j++)
                {
                    if(i + rows_number*j + k*rows_number*raw_variables_number + l*channels*rows_number*raw_variables_number % rows_number*raw_variables_number*channels == 0)

                        cout<< "<--Batch-->"<<endl;

                    if(i + rows_number*j + k*rows_number*raw_variables_number % rows_number*raw_variables_number == 0)

                        cout<< "*--Channel--*"<<endl;

                   cout<<*(vector + i + j*rows_number + k*rows_number*raw_variables_number+ l*channels*rows_number*raw_variables_number)<< " ";
                }

               cout<<" "<<endl;
            }
        }
    }
}


void swap_rows(Tensor<type, 2>& data_matrix, Index row1, Index row2)
{
    const Tensor<type, 1> temp = data_matrix.chip(row1, 0);

    data_matrix.chip(row1, 0) = data_matrix.chip(row2, 0);

    data_matrix.chip(row2, 0) = temp;
}


Tensor<type, 1> calculate_delta(const Tensor<type, 1>& data)
{
    if(data.size() <= 2)
    {
        return Tensor<type, 1>();
    }

    Tensor<type, 1> difference_data(data.size());
    difference_data(0) = type(0);

    if(data.size() <= 1) return Tensor<type, 1>();

    #pragma omp parallel for

    for(int i = 1; i < data.size(); i++) 
        difference_data(i) = data(i) - data(i - 1);

    return difference_data;
}


Tensor<type, 1> mode(Tensor<type, 1>& data)
{
    Tensor<type, 1> mode_and_frequency(2);

    std::map<type, type> frequency_map;

    for(int i = 0; i < data.size(); i++)
    {
        type value = data(i);
        frequency_map[value]++;
    }

    cout << "frequency_map: " << endl;
    
    for(auto it = frequency_map.cbegin(); it != frequency_map.cend(); ++it)
    {
        cout << "Key: " << it->first << ", Value: " << it->second << "\n";
    }

    type mode = type(-1);
    type max_frequency = type(0);

    for(const auto& entry : frequency_map)
    {
        if(entry.second > max_frequency)
        {
            max_frequency = entry.second;
            mode = entry.first;
        }
    }

    if(mode == -1) Tensor<type, 1>();

    mode_and_frequency(0) = mode;
    mode_and_frequency(1) = max_frequency;

    return mode_and_frequency;
}


Tensor<type, 1> fill_gaps_by_value(Tensor<type, 1>& data, Tensor<type, 1>& difference_data, const type& value)
{
    std::vector<type> result;

    for(Index i = 1; i < difference_data.size(); i++)
    {
        if(difference_data(i) != value)
        {
            type previous_time = data(i-1) + value;

            do
            {
                result.push_back(previous_time);

                previous_time += value;
            }while (previous_time < data(i));
        }
    }

    TensorMap<Tensor<type, 1>> filled_data(result.data(), result.size());

    return filled_data;
}


Index partition(Tensor<type, 2>& data_matrix, Index start_index, Index end_index, Index target_column)
{
    Tensor<type, 1> pivot_row = data_matrix.chip(start_index, 0);
    type pivot_value = pivot_row(target_column);
    Index smaller_elements_count = 0;

    for(Index current_index = start_index + 1; current_index <= end_index; current_index++)
    {
        if(data_matrix(current_index, target_column) <= pivot_value)
        {
            smaller_elements_count++;
        }
    }

    Index pivot_position = start_index + smaller_elements_count;
    swap_rows(data_matrix, pivot_position, start_index);

    Index left_index = start_index, right_index = end_index;

    while (left_index < pivot_position && right_index > pivot_position)
    {
        while (data_matrix(left_index, target_column) <= pivot_value)
        {
            left_index++;
        }

        while (data_matrix(right_index, target_column) > pivot_value)
        {
            right_index--;
        }

        if(left_index < pivot_position && right_index > pivot_position)
        {
            swap_rows(data_matrix, left_index++, right_index--);
        }
    }

    return pivot_position;
}


Tensor<Index, 1> intersection(const Tensor<Index, 1>& tensor_1, const Tensor<Index, 1>& tensor_2)
{
    Index intersection_index_number = 0;

    for(Index i = 0; i < tensor_1.size(); i++)
    {
        for(Index j = 0; j < tensor_2.size(); j++)
        {
            if(tensor_1(i) == tensor_2(j)) {
                intersection_index_number++;
            }
        }
    }

    if(intersection_index_number == 0)
    {
        return Tensor<Index, 1>(0);
    }

    Tensor<Index, 1> intersection(intersection_index_number);
    Index count = 0;

    for(Index i = 0; i < tensor_1.size(); i++)
    {
        for(Index j = 0; j < tensor_2.size(); j++)
        {
            if(tensor_1(i) == tensor_2(j))
            {
                intersection(count) = tensor_2(j);
                count++;
            }
        }
    }

    return intersection;
}


TensorMap<Tensor<type, 1>> tensor_map(const Tensor<type, 2>& matrix, const Index& column_index)
{
    TensorMap<Tensor<type, 1>> column((type*) matrix.data() + column_index * matrix.dimension(0), matrix.dimension(0));

    return column;// ((type*)matrix.data(), column_index * matrix.dimension(0), matrix.dimension(0));
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
