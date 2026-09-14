#include "tests/pch.h"

#include "opennn/core/random_utilities.h"

using namespace opennn;

TEST(RandomUtilitiesTest, SeedReplaysSequence)
{
    set_seed(42);
    const float first_uniform = random_uniform();
    const Index first_integer = random_integer(1, 100);
    const bool first_boolean = random_bool();

    set_seed(42);
    EXPECT_EQ(random_uniform(), first_uniform);
    EXPECT_EQ(random_integer(1, 100), first_integer);
    EXPECT_EQ(random_bool(), first_boolean);
}

TEST(RandomUtilitiesTest, BulkDistributionsPreserveSeededDrawOrder)
{
    constexpr unsigned seed = 913;
    mt19937 expected_generator(seed);
    uniform_int_distribution<Index> integers(-4, 8);
    bernoulli_distribution booleans(0.3f);
    normal_distribution<float> normals(2.0f, 0.5f);
    uniform_real_distribution<float> uniform(-1.0f, 1.0f);

    set_seed(seed);
    MatrixR integer_values(3, 5);
    set_random_integer(integer_values, -4, 8);
    for (Index i = 0; i < integer_values.size(); ++i)
        EXPECT_EQ(integer_values.data()[i], float(integers(expected_generator)));

    vector<uint8_t> boolean_values(11);
    set_random_bernoulli(boolean_values, 0.3f);
    for (uint8_t value : boolean_values)
        EXPECT_EQ(value, uint8_t(booleans(expected_generator)));

    MatrixR normal_values(3, 5);
    set_random_normal(MatrixMap(normal_values.data(), 3, 5), 2.0f, 0.5f);
    for (Index i = 0; i < normal_values.size(); ++i)
        EXPECT_EQ(normal_values.data()[i], normals(expected_generator));

    // An odd normal fill leaves a cached distribution value, but no engine draw
    // may be consumed after the fill ends or reused by a later distribution.
    EXPECT_EQ(random_uniform(), uniform(expected_generator));
}

TEST(RandomUtilitiesTest, OrthogonalInitializationPreservesDrawCountAndColumnNorms)
{
    constexpr unsigned seed = 271;
    mt19937 expected_generator(seed);
    normal_distribution<float> gaussian(0.0f, 1.0f);
    uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (Index i = 0; i < 15; ++i) gaussian(expected_generator);

    MatrixR values(5, 3);
    set_seed(seed);
    set_random_orthogonal(MatrixMap(values.data(), 5, 3));
    EXPECT_EQ(random_uniform(), uniform(expected_generator));

    const MatrixR gram = values.transpose() * values;
    EXPECT_LT((gram - MatrixR::Identity(3, 3)).norm(), 1.0e-5f);

    MatrixR empty(0, 3);
    set_seed(seed);
    set_random_orthogonal(MatrixMap(empty.data(), 0, 3));
    MatrixR invalid(2, 3);
    EXPECT_THROW(set_random_orthogonal(MatrixMap(invalid.data(), 2, 3)), runtime_error);
    expected_generator.seed(seed);
    EXPECT_EQ(random_uniform(), uniform(expected_generator));
}
