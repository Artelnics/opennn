//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R A N D O M   U T I L I T I E S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include <atomic>
#include <omp.h>
#include "random_utilities.h"

namespace opennn
{

// ---------------------------------------------------------------------------
// PRNG state
//
// `global_seed` is the single source of truth for the user-visible seed:
//   -1   → no deterministic seed; each thread initializes from random_device
//   >= 0 → deterministic mode; thread N seeds with `global_seed + N*5489u`,
//          giving each OpenMP worker its own non-correlated stream while
//          keeping the whole run reproducible.
//
// `seed_generation` is a monotonic counter bumped on every set_seed() call.
// Each thread caches the last generation it observed in `local_generation`;
// when they differ, get_generator() reseeds the thread's mt19937 lazily.
// This means a set_seed() from the main thread propagates to OMP workers the
// next time they hit get_generator(), without any explicit synchronization.
// ---------------------------------------------------------------------------

static std::atomic<long long>    global_seed{-1};
static std::atomic<unsigned int> seed_generation{0};

thread_local mt19937    generator;
thread_local unsigned int local_generation = 0;

static void initialize_generator()
{
    const long long seed = global_seed.load(std::memory_order_relaxed);

    if (seed < 0)
    {
        random_device rd;
        generator.seed(rd());
    }
    else
    {
        const int thread_id = omp_get_thread_num();
        generator.seed(static_cast<unsigned int>(seed + thread_id * 5489u));
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

// The set_random_* helpers below run under `#pragma omp parallel`. The PRNG
// machinery above guarantees that each OpenMP worker reseeds itself the first
// time it touches get_generator() after a set_seed(), so parallel init is
// reproducible bit-for-bit (given the same OMP_NUM_THREADS and a `static`
// schedule). Without the seed-generation tracking these were silently
// non-deterministic — that was the airfoil_self_noise CPU divergence bug.

void set_random_uniform(MatrixR& tensor, float min, float max)
{
    #pragma omp parallel
    {
        uniform_real_distribution<float> distribution(min, max);
        #pragma omp for
        for(Index i = 0; i < tensor.size(); ++i)
            tensor(i) = distribution(get_generator());
    }
}

void set_random_uniform(VectorMap tensor, float min, float max)
{
    #pragma omp parallel
    {
        uniform_real_distribution<float> distribution(min, max);
        #pragma omp for
        for(Index i = 0; i < tensor.size(); ++i)
            tensor(i) = distribution(get_generator());
    }
}

void set_random_normal(MatrixMap tensor, float mean, float std_dev)
{
    #pragma omp parallel
    {
        normal_distribution<float> distribution(mean, std_dev);
        #pragma omp for
        for(Index i = 0; i < tensor.size(); ++i)
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

void set_random_integer(MatrixR &tensor, Index min, Index max)
{
    #pragma omp parallel
    {
        uniform_int_distribution<Index> distribution(min, max);
        #pragma omp for
        for(Index i = 0; i < tensor.size(); ++i)
            tensor(i) = distribution(get_generator());
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
