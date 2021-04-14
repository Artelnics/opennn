//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   U T I L I T I E S   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "tensor_utilities.h"

namespace OpenNN
{

void initialize_sequential(Tensor<type, 1>& vector)
{
    for(Index i = 0; i < vector.size(); i++) vector(i) = i;
}


void multiply_rows(Tensor<type, 2>& matrix, const Tensor<type, 1>& vector)
{
    const Index columns_number = matrix.dimension(1);
    const Index rows_number = matrix.dimension(0);

//    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
           matrix(i,j) *= vector(j);
        }
    }
}


bool is_zero(const Tensor<type, 1>& tensor)
{
    const Index size = tensor.size();

    for(Index i = 0; i < size; i++)
    {
        if(abs(tensor[i]) > numeric_limits<type>::min()) return false;
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


bool are_equal(const Tensor<type, 2>& matrix_1, const Tensor<type, 2>& matrix_2, const type& tolerance)
{
    const Index size = matrix_1.size();

    for(Index i = 0; i < size; i++)
    {
        if(abs(matrix_1(i) - matrix_2(i)) > tolerance) return false;
    }

    return true;
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

      throw logic_error(buffer.str());
    }

    file.precision(20);

    const Index data_rows = data.dimension(0);
    const Index data_columns = data.dimension(1);

    char separator_char = ';';

    for(Index i = 0; i < data_rows; i++)
    {
       for(Index j = 0; j < data_columns; j++)
       {
           file << data(i,j);

           if(j != data_columns-1)
           {
               file << separator_char;
           }
       }
       file << endl;
    }
    file.close();
}


/// @todo It does not work well.
/*
Tensor<Index, 1> sort_indexes(const Tensor<type, 1> & vector)
{
  Tensor<Index, 1> indexes(vector.size());
  iota(indexes.data(), indexes.data() + indexes.size(), 0);

  stable_sort(indexes.data(), indexes.data()+indexes.size(),
       [&vector](Index i1, Index i2) {return vector(i1) < vector(i2);});

  return indexes;
}
*/

Tensor<Index, 1> rank_sort(const Tensor<type, 1>& v_temp)
{
    vector<pair<float, size_t> > v_sort(v_temp.size());

    for (size_t i = 0U; i < v_sort.size(); ++i) {
        v_sort[i] = make_pair(v_temp[i], i);
    }

    sort(v_sort.begin(), v_sort.end());

    pair<double, size_t> rank;

    Tensor<Index, 1> result(v_temp.size());

    for (size_t i = 0U; i < v_sort.size(); ++i) {
        if (v_sort[i].first != rank.first) {
            rank = make_pair(v_sort[i].first, i);
        }
        result(v_sort[i].second) = rank.second;
    }
    return result;
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
