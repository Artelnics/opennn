//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R A N D O M   U T I L I T I E S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include <mutex>
#include "random_utilities.h"

namespace opennn
{

static mutex rng_mutex;
static mt19937 generator;
static long long current_seed = -1;

static void reseed_unlocked()
{
    if (current_seed < 0)
    {
        random_device device;
        generator.seed(device());
    }
    else
        generator.seed(uint32_t(current_seed));
}

void set_seed(unsigned seed)
{
    {
        lock_guard<mutex> lock(rng_mutex);
        current_seed = static_cast<long long>(seed);
        reseed_unlocked();
    }
    srand(seed); // covers Eigen helpers that fall back to libc rand().
}

long long get_seed()
{
    lock_guard<mutex> lock(rng_mutex);
    return current_seed;
}

float random_uniform(float min, float max)
{
    lock_guard<mutex> lock(rng_mutex);
    uniform_real_distribution<float> distribution(min, max);
    return distribution(generator);
}

Index random_integer(Index min, Index max)
{
    lock_guard<mutex> lock(rng_mutex);
    uniform_int_distribution<Index> distribution(min, max);
    return distribution(generator);
}

bool random_bool(float probability)
{
    lock_guard<mutex> lock(rng_mutex);
    bernoulli_distribution distribution(probability);
    return distribution(generator);
}

void set_random_uniform(MatrixR& tensor, float min, float max)
{
    lock_guard<mutex> lock(rng_mutex);
    uniform_real_distribution<float> distribution(min, max);
    for (Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(generator);
}

void set_random_uniform(VectorMap tensor, float min, float max)
{
    lock_guard<mutex> lock(rng_mutex);
    uniform_real_distribution<float> distribution(min, max);
    for (Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(generator);
}

void set_random_normal(MatrixMap tensor, float mean, float std_dev)
{
    lock_guard<mutex> lock(rng_mutex);
    normal_distribution<float> distribution(mean, std_dev);
    for (Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(generator);
}

template<typename T>
void shuffle_vector(vector<T>& vec)
{
    lock_guard<mutex> lock(rng_mutex);
    ranges::shuffle(vec, generator);
}

template void shuffle_vector<Index>(vector<Index>&);
template void shuffle_vector<size_t>(vector<size_t>&);

void shuffle_vector_blocks(vector<Index>& vec, size_t blocks_number)
{
    const size_t size = vec.size();
    if (size < 2) return;

    const size_t block_size = size / max(size_t(1), blocks_number);

    if (block_size < 10)
    {
        shuffle_vector(vec);
        return;
    }

    lock_guard<mutex> lock(rng_mutex);
    for (size_t i = 0; i < size; i += block_size)
    {
        const auto start = vec.begin() + i;
        const auto end = (i + block_size > size) ? vec.end() : start + block_size;

        shuffle(start, end, generator);
    }
}

void shuffle(VectorB& vector_to_shuffle)
{
    lock_guard<mutex> lock(rng_mutex);
    shuffle(vector_to_shuffle.data(), vector_to_shuffle.data() + vector_to_shuffle.size(), generator);
}

Index get_random_element(const vector<Index>& values)
{
    if (values.empty())
        throw runtime_error("get_random_element: Input vector is empty.");

    lock_guard<mutex> lock(rng_mutex);
    uniform_int_distribution<size_t> distribution(0, values.size() - 1);

    return values[distribution(generator)];
}

void set_random_integer(MatrixR &tensor, Index min, Index max)
{
    lock_guard<mutex> lock(rng_mutex);
    uniform_int_distribution<Index> distribution(min, max);
    for (Index i = 0; i < tensor.size(); ++i)
        tensor(i) = distribution(generator);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
