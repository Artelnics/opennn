//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R A N D O M   U T I L I T I E S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "random_utilities.h"

namespace opennn
{

static long long global_seed = -1;

thread_local mt19937 generator;
thread_local bool is_initialized = false;

void initialize_generator()
{
    if (global_seed == -1)
    {
        random_device rd;
        generator.seed(rd());
    }
    else
    {
        const int thread_id = omp_get_thread_num();
        generator.seed(static_cast<unsigned int>(global_seed + thread_id * 5489u));
    }

    is_initialized = true;
}


void set_seed(Index seed)
{
    global_seed = seed;

    #pragma omp parallel
    {
        is_initialized = false;
    }
}


inline mt19937& get_generator()
{
    if(!is_initialized) initialize_generator();
    return generator;
}


type random_uniform(type min, type max)
{
    uniform_real_distribution<type> distribution(min, max);
    return distribution(get_generator());
}


type random_normal(type mean, type std_dev)
{
    normal_distribution<type> distribution(mean, std_dev);
    return distribution(get_generator());
}


Index random_integer(Index min, Index max)
{
    uniform_int_distribution<Index> distribution(min, max);
    return distribution(get_generator());
}


bool random_bool(type probability)
{
    bernoulli_distribution distribution(probability);
    return distribution(get_generator());
}


void set_random_uniform(VectorR& tensor, type min, type max)
{
    uniform_real_distribution<type> distribution(min, max);

    #pragma omp parallel for
    for(Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(get_generator());
}


void set_random_uniform(MatrixR& tensor, type min, type max)
{
    uniform_real_distribution<type> distribution(min, max);

    #pragma omp parallel for
    for(Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(get_generator());
}


void set_random_uniform(VectorMap tensor, type min, type max)
{
    uniform_real_distribution<type> distribution(min, max);

    #pragma omp parallel for
    for(Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(get_generator());
}


void set_random_uniform(MatrixMap tensor, type min, type max)
{
    uniform_real_distribution<type> distribution(min, max);

    #pragma omp parallel for
    for(Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(get_generator());
}


template<typename T>
void shuffle_vector(vector<T>& vec)
{
    shuffle(vec.begin(), vec.end(), get_generator());
}

template void shuffle_vector<Index>(vector<Index>&);
template void shuffle_vector<size_t>(vector<size_t>&);


void shuffle_vector_blocks(vector<Index>& vec, size_t blocks_number)
{
    const size_t n = vec.size();
    if (n < 2) return;

    const size_t block_size = n / max(size_t(1), blocks_number);

    if (block_size < 10)
    {
        shuffle_vector(vec);
        return;
    }

    for (size_t i = 0; i < n; i += block_size)
    {
        const auto start = vec.begin() + i;
        const auto end = (i + block_size > n) ? vec.end() : start + block_size;

        shuffle(start, end, get_generator());
    }
}


void shuffle(VectorB& v)
{
    shuffle(v.data(), v.data() + v.size(), get_generator());
}


Index get_random_element(const vector<Index>&values)
{
    if (values.empty())
        throw runtime_error("get_random_element: Input vector is empty.");

    uniform_int_distribution<size_t> distribution(0, values.size() - 1);

    const size_t random_index = distribution(get_generator());

    return values[random_index];
}


void set_random_integer(Tensor2& tensor, Index min, Index max)
{
    uniform_int_distribution<Index> distribution(min, max);

#pragma omp parallel for
    for(Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(get_generator());
}


void set_random_integer(MatrixR &tensor, Index min, Index max)
{
    uniform_int_distribution<Index> distribution(min, max);

#pragma omp parallel for
    for(Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(get_generator());
}

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
