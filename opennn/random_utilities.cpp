//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R A N D O M   U T I L I T I E S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include <atomic>
#include <thread>
#include "random_utilities.h"

namespace opennn
{

static std::atomic<long long>    global_seed{-1};
static std::atomic<unsigned int> seed_generation{0};

thread_local mt19937    generator;
thread_local unsigned int local_generation = 0;

static void initialize_generator()
{
    const long long seed = global_seed.load(std::memory_order_relaxed);

    if (seed < 0)
    {
        random_device device;
        generator.seed(device());
    }
    else
    {
        // Hash the std::thread::id so OMP threads and std::thread workers
        // each get a unique seed deterministic-per-thread. omp_get_thread_num()
        // returns 0 outside an OMP region — wrong for std::thread workers.
        const uint32_t tid_hash =
            uint32_t(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::seed_seq seq{ uint32_t(seed), tid_hash };
        generator.seed(seq);
    }

    local_generation = seed_generation.load(std::memory_order_acquire);
}

inline mt19937& get_generator()
{
    if (local_generation != seed_generation.load(std::memory_order_acquire))
        initialize_generator();
    return generator;
}

void set_seed(unsigned seed)
{
    global_seed.store(static_cast<long long>(seed), std::memory_order_relaxed);
    seed_generation.fetch_add(1, std::memory_order_release);
    initialize_generator();   // reseed the calling (main) thread immediately.
    srand(seed);              // covers Eigen helpers that fall back to libc rand().
}

long long get_seed()
{
    return global_seed.load(std::memory_order_relaxed);
}

float random_uniform(float min, float max)
{
    uniform_real_distribution<float> distribution(min, max);
    return distribution(get_generator());
}

Index random_integer(Index min, Index max)
{
    uniform_int_distribution<Index> distribution(min, max);
    return distribution(get_generator());
}

bool random_bool(float probability)
{
    bernoulli_distribution distribution(probability);
    return distribution(get_generator());
}

void set_random_uniform(MatrixR& tensor, float min, float max)
{
    #pragma omp parallel
    {
        uniform_real_distribution<float> distribution(min, max);
        #pragma omp for
        for (Index i = 0; i < tensor.size(); ++i)
            tensor(i) = distribution(get_generator());
    }
}

void set_random_uniform(VectorMap tensor, float min, float max)
{
    #pragma omp parallel
    {
        uniform_real_distribution<float> distribution(min, max);
        #pragma omp for
        for (Index i = 0; i < tensor.size(); ++i)
            tensor(i) = distribution(get_generator());
    }
}

void set_random_normal(MatrixMap tensor, float mean, float std_dev)
{
    #pragma omp parallel
    {
        normal_distribution<float> distribution(mean, std_dev);
        #pragma omp for
        for (Index i = 0; i < tensor.size(); ++i)
            tensor(i) = distribution(get_generator());
    }
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
    const size_t size = vec.size();
    if (size < 2) return;

    const size_t block_size = size / max(size_t(1), blocks_number);

    if (block_size < 10)
    {
        shuffle_vector(vec);
        return;
    }

    for (size_t i = 0; i < size; i += block_size)
    {
        const auto start = vec.begin() + i;
        const auto end = (i + block_size > size) ? vec.end() : start + block_size;

        shuffle(start, end, get_generator());
    }
}

void shuffle(VectorB& vector_to_shuffle)
{
    shuffle(vector_to_shuffle.data(), vector_to_shuffle.data() + vector_to_shuffle.size(), get_generator());
}

Index get_random_element(const vector<Index>& values)
{
    if (values.empty())
        throw runtime_error("get_random_element: Input vector is empty.");

    uniform_int_distribution<size_t> distribution(0, values.size() - 1);

    return values[distribution(get_generator())];
}

void set_random_integer(MatrixR &tensor, Index min, Index max)
{
    #pragma omp parallel
    {
        uniform_int_distribution<Index> distribution(min, max);
        #pragma omp for
        for (Index i = 0; i < tensor.size(); ++i)
            tensor(i) = distribution(get_generator());
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
