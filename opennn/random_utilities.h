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
    /// @brief Seeds the library-wide pseudo-random number generator.
    void set_seed(unsigned seed);

    /// @brief Returns the seed currently used by the library RNG.
    [[nodiscard]] long long get_seed();

    /// @brief Draws a float uniformly in [min, max].
    [[nodiscard]] float random_uniform(float = -1, float = 1);

    /// @brief Draws an integer uniformly in [min, max].
    [[nodiscard]] Index random_integer(Index, Index);

    /// @brief Draws a boolean that is true with the given probability.
    [[nodiscard]] bool random_bool(float = 0.5);

    /// @brief Fills the matrix with uniform random values in [min, max].
    void set_random_uniform(MatrixR&, float = -0.1, float = 0.1);

    /// @brief Fills the vector map with uniform random values in [min, max].
    void set_random_uniform(VectorMap, float = -0.1, float = 0.1);

    /// @brief Fills the matrix with normal random values of the given mean and standard deviation.
    void set_random_normal(MatrixMap, float = 0, float = 1);

    /// @brief Fills the matrix with uniform random integers in [min, max].
    void set_random_integer(MatrixR&, Index, Index);

    /// @brief Randomly permutes the entries of a boolean vector in place.
    void shuffle(VectorB& vector_to_shuffle);

    /// @brief Randomly permutes the entries of the given vector in place.
    template<typename T>
    void shuffle_vector(vector<T>&);

    /// @brief Shuffles contiguous blocks of the given length while preserving in-block order.
    void shuffle_vector_blocks(vector<Index>&, size_t = 20);

    /// @brief Picks one element uniformly at random from the given vector.
    [[nodiscard]] Index get_random_element(const vector<Index>&);

    /// @brief Returns the Glorot/Xavier uniform initialization limit sqrt(6 / (fan_in + fan_out)).
    [[nodiscard]] inline float glorot_limit(Index fan_in, Index fan_out)
    {
        return sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
