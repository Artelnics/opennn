//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R A N D O M   U T I L I T I E S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{
    // Pins a deterministic seed for the program. After set_seed(N), every
    // thread (main + OpenMP workers) reseeds its thread_local mt19937 with
    // `N + thread_id * 5489u` lazily on the next call to get_generator(),
    // so OMP-parallel init helpers (set_random_uniform, etc.) are reproducible.
    // get_seed() returns -1 when no seed has been set (default: random_device).
    void set_seed(unsigned seed);
    long long get_seed();

    float random_uniform(float = -1, float = 1);
    Index random_integer(Index, Index);
    bool random_bool(float = 0.5);

    void set_random_uniform(MatrixR&, float = -0.1, float = 0.1);
    void set_random_uniform(VectorMap, float = -0.1, float = 0.1);

    void set_random_normal(MatrixMap, float = 0, float = 1);

    void set_random_integer(MatrixR&, Index, Index);

    void shuffle(VectorB& vector_to_shuffle);

    template<typename T>
    void shuffle_vector(vector<T>&);

    void shuffle_vector_blocks(vector<Index>&, size_t = 20);

    Index get_random_element(const vector<Index>&);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
