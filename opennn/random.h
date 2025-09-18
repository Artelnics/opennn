//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R A N D O M   N U M B E R   G E N E R A T O R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com
#include "pch.h"

#ifndef RANDOM_H
#define RANDOM_H

namespace opennn
{


/// @brief A base generator that satisfies UniformRandomBitGenerator C++11 named requirement.
/// @todo Use uniform_random_bit_generator concept for C++20 and newer
class DefaultRandomGenerator
{
protected:
    std::mt19937 generator;
public:
    typedef uint32_t result_type;

    static const result_type default_seed = std::mt19937::default_seed;

    DefaultRandomGenerator() : generator(default_seed) {}
    DefaultRandomGenerator(result_type seed) : generator(seed) {}

    /// @brief Yields the smallest value that operator() may return. The result is strictly less than max().
    /// @return result_type
    static constexpr result_type min()
    {
        return decltype(generator)::min();
    }

    /// @brief Yields the largest value that operator() may return. The result is strictly greater than min().
    /// @return result_type
    static constexpr result_type max()
    {
        return decltype(generator)::max();
    }

    /// @brief Set a new seed for use in the generator.
    /// @param new_seed 
    void seed(result_type new_seed = default_seed)
    {
        generator.seed(new_seed);
    }

    /// @brief Returns the value in the closed interval [min(), max()] in amortized constant time.
    result_type operator()()
    {
        return generator();
    }
};

/// @brief Get global RNG object.
/// @return A reference to the global RNG
DefaultRandomGenerator& getGlobalRandomGenerator();

/// @brief Set a new seed for use in the global RNG.
void setGlobalSeed(DefaultRandomGenerator::result_type new_seed = DefaultRandomGenerator::default_seed)
{
    getGlobalRandomGenerator().seed(new_seed);
}

}


#endif // RANDOM_H