// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

#include "tests/common/pch.h"
#include "opennn/model_selection/weighted_selection.h"

#include <array>
#include <random>

using namespace opennn;

TEST(WeightedSelection, RetainsElitesSkipsZerosAndNeverRepeatsACandidate)
{
    VectorR weights(5);
    weights << 1, 2, 0, 4, 10;
    VectorB selected = VectorB::Constant(5, false);
    selected(4) = true;
    const std::array<double, 3> draws{0.8, 0.1, 0.6};
    Index calls = 0;
    detail::fill_weighted_selection(weights, 3, selected, [&] { return draws.at(size_t(calls++)); });
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(selected.count(), 3);
    EXPECT_TRUE(selected(1));
    EXPECT_TRUE(selected(3));
    EXPECT_TRUE(selected(4));
    EXPECT_FALSE(selected(2));
    detail::fill_weighted_selection(weights, 2, selected, []() -> double
    {
        ADD_FAILURE() << "A satisfied quota must not consume random numbers.";
        return 0;
    });
    EXPECT_EQ(selected.count(), 3);
}

TEST(WeightedSelection, InvalidInputsLeaveTheMaskUnchanged)
{
    VectorR weights = VectorR::Ones(3);
    VectorB selected = VectorB::Constant(3, false);
    selected(1) = true;
    const VectorB original = selected;
    for (float invalid : {-1.0f, numeric_limits<float>::infinity(), numeric_limits<float>::quiet_NaN()})
    {
        weights(2) = invalid;
        EXPECT_THROW(detail::fill_weighted_selection(weights, 2, selected, [] { return 0.5; }), invalid_argument);
        EXPECT_TRUE((selected.array() == original.array()).all());
    }
    weights.setZero();
    EXPECT_THROW(detail::fill_weighted_selection(weights, 2, selected, [] { return 0.5; }), invalid_argument);
    weights.setOnes();
    Index draw = 0;
    EXPECT_THROW(detail::fill_weighted_selection(weights, 2, selected,
        [&] { return ++draw == 1 ? 0.5 : numeric_limits<double>::quiet_NaN(); }), invalid_argument);
    EXPECT_TRUE((selected.array() == original.array()).all());
    EXPECT_THROW(detail::fill_weighted_selection(weights, 4, selected, [] { return 0.5; }), invalid_argument);
}

TEST(WeightedSelection, BoundaryDrawsAndExtremeWeightsRemainFiniteAndDeterministic)
{
    VectorR weights(3);
    weights << numeric_limits<float>::denorm_min(), 1, numeric_limits<float>::max();
    VectorB selected = VectorB::Constant(3, false);
    detail::fill_weighted_selection(weights, 2, selected, [] { return 1.0; });
    EXPECT_FALSE(selected(0));
    EXPECT_TRUE(selected(1));
    EXPECT_TRUE(selected(2));
    selected.setConstant(false);
    detail::fill_weighted_selection(weights, 2, selected, [] { return 0.0; });
    EXPECT_TRUE(selected(0));
    EXPECT_TRUE(selected(1));
    EXPECT_FALSE(selected(2));
}

TEST(WeightedSelection, PairFrequenciesFollowSuccessiveWeightedSampling)
{
    VectorR weights(3);
    weights << 1, 2, 4;
    std::array<Index, 3> excluded{};
    mt19937 generator(217);
    uniform_real_distribution<double> uniform(0, 1);
    constexpr Index trials = 30000;
    for (Index trial = 0; trial < trials; ++trial)
    {
        VectorB selected = VectorB::Constant(3, false);
        detail::fill_weighted_selection(weights, 2, selected, [&] { return uniform(generator); });
        ASSERT_EQ(selected.count(), 2);
        for (Index i = 0; i < 3; ++i)
            if (!selected(i)) ++excluded[size_t(i)];
    }
    // P({a,b}) = wa/W * wb/(W-wa) + wb/W * wa/(W-wb).
    const std::array<double, 3> probabilities{(2.0/7)*(4.0/5)+(4.0/7)*(2.0/3),
                                        (1.0/7)*(4.0/6)+(4.0/7)*(1.0/3),
                                        (1.0/7)*(2.0/6)+(2.0/7)*(1.0/5)};
    for (size_t i = 0; i < probabilities.size(); ++i)
        EXPECT_NEAR(double(excluded[i]) / trials, probabilities[i], 0.012);
}
