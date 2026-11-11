// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/core/opennn_types.h"

namespace opennn
{
    void set_seed(unsigned seed);
    unsigned get_seed() noexcept;

    uint64_t derive_random_seed(uint64_t seed, uint64_t key) noexcept;

    class RandomStream
    {
    public:
        explicit RandomStream(uint64_t seed) : state(seed) {}
        uint64_t next() noexcept;
        float uniform(float minimum = 0.0f, float maximum = 1.0f) noexcept;

    private:
        uint64_t state;
    };

    // Element `position` of the sequence `first, draw(random), draw(random), ...`
    // that skips repeated values while fewer than `distinct` new values were kept.
    // A long run of repeats is accepted instead of looping on a degenerate draw.
    template<typename Value, typename Draw>
    Value draw_distinct(RandomStream& random, Value first, Index position, double distinct, Draw&& draw)
    {
        constexpr Index maximum_repeats = 1024;

        vector<Value> values{std::move(first)};

        for (Index repeats = 0; ssize(values) <= position;)
        {
            Value value = draw(random);

            if (double(ssize(values)) > distinct || repeats == maximum_repeats
                || ranges::find(values, value) == values.end())
            {
                values.push_back(std::move(value));
                repeats = 0;
            }
            else
                ++repeats;
        }

        return values.back();
    }

    float random_uniform(float = -1, float = 1);
    float random_normal(float = 0, float = 1);
    Index random_integer(Index, Index);
    bool random_bool(float = 0.5);

    void set_random_uniform(MatrixR&, float = -0.1, float = 0.1);
    void set_random_uniform(VectorMap, float = -0.1, float = 0.1);
    void set_random_bernoulli(span<uint8_t>, float);

    void set_random_normal(MatrixMap, float = 0, float = 1);

    void set_random_orthogonal(MatrixMap);

    void set_random_integer(MatrixR&, Index, Index);

    void shuffle(VectorB&);

    template<typename T>
    void shuffle_vector(vector<T>&);

    template<typename T>
    void shuffle_vector_seeded(vector<T>&, unsigned seed);

    template<typename T>
    void partial_shuffle(vector<T>& v, Index count)
    {
        const Index n = ssize(v);
        const Index limit = min(count, n - 1);
        for (Index i = 0; i < limit; ++i)
            swap(v[i], v[i + random_integer(0, n - 1 - i)]);
    }

    bool draw_k_hot(Index, Index, const vector<char>&,
                    const vector<char>&, vector<float>&);

    Index get_random_element(const vector<Index>&);

    inline float glorot_limit(Index fan_in, Index fan_out)
    {
        return sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    }
}
