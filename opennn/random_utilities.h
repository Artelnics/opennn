//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R A N D O M   U T I L I T I E S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{
    void set_seed(unsigned seed);
    long long get_seed();

    float random_uniform(float = -1, float = 1);
    Index random_integer(Index, Index);
    bool random_bool(float = 0.5);

    void set_random_uniform(MatrixR&, float = -0.1, float = 0.1);
    void set_random_uniform(VectorMap, float = -0.1, float = 0.1);

    void set_random_normal(MatrixMap, float = 0, float = 1);

    void set_random_integer(MatrixR&, Index, Index);

    void shuffle(VectorB&);

    template<typename T>
    void shuffle_vector(vector<T>&);

    // Partial Fisher-Yates: randomly permute the first `count` elements of `v` (each becomes a
    // uniform pick from the not-yet-placed elements). Pass count >= v.size() for a full shuffle.
    template<typename T>
    void partial_shuffle(vector<T>& v, Index count)
    {
        const Index n = ssize(v);
        const Index limit = min(count, n - 1);
        for (Index i = 0; i < limit; ++i)
            swap(v[i], v[i + random_integer(0, n - 1 - i)]);
    }

    // Box-respecting K-hot draw into `out` (sized to `count`): `force_on` indices are set to 1,
    // `force_off` to 0, and the remaining k - |force_on| ones are chosen uniformly at random from
    // the free indices. Returns false (with `out` cleared) when exactly-k is infeasible: an index
    // forced both on and off, more than k forced on, or too few free indices to reach k.
    bool draw_k_hot(Index, Index, const vector<char>&,
                    const vector<char>&, vector<float>&);

    void shuffle_vector_blocks(vector<Index>&, size_t = 20);

    Index get_random_element(const vector<Index>&);

    inline float glorot_limit(Index fan_in, Index fan_out)
    {
        return sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
